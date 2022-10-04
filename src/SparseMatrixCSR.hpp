#pragma once

#define CLASS SparseMatrixCSR
#define BASE  SparseCSR<I>

namespace Tensors
{

    template<typename T, typename I>
    class CLASS : public BASE
    {
    protected:
        
        using BASE::m;
        using BASE::n;
        using BASE::outer;
        using BASE::inner;
        using BASE::thread_count;
        using BASE::job_ptr;
        using BASE::upper_triangular_job_ptr;
        using BASE::lower_triangular_job_ptr;

        Tensor1<T,I> values;
        
    public:
        
        using BASE::RowCount;
        using BASE::ColCount;
        using BASE::NonzeroCount;
        using BASE::ThreadCount;
        using BASE::SetThreadCount;
        using BASE::Outer;
        using BASE::Inner;
        using BASE::JobPtr;
        using BASE::RequireJobPtr;
        using BASE::RequireUpperTriangularJobPtr;
        using BASE::RequireLowerTriangularJobPtr;
        using BASE::UpperTriangularJobPtr;
        using BASE::LowerTriangularJobPtr;
        using BASE::CreateTransposeCounters;
        using BASE::WellFormed;
        using BASE::Dot_;
        
        CLASS() : BASE() {}
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long thread_count_
        )
        :   BASE(m_,n_,thread_count) {}
        
        CLASS(
            const long long m_,
            const long long n_,
            const long long nnz_,
            const long long thread_count_
        )
        :   BASE    ( m_, n_, nnz_, thread_count_ )
        ,   values  ( Tensor1<T,I>(static_cast<I>(nnz_)) )
        {}
        
        template<typename S, typename J0, typename J1>
        CLASS(
            const J0 * const outer_,
            const J1 * const inner_,
            const S  * const values_,
            const long long m_,
            const long long n_,
                  long long thread_count_
        )
        :   BASE    (outer_, inner_, m_, n_, thread_count_)
        ,   values  ( ToTensor1<T,I>(values_,outer_[static_cast<I>(m_)]) )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE( other )
        ,   values  ( other.values  )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( static_cast<BASE&>(A), static_cast<BASE&>(B) );
            swap( A.values,              B.values              );
        }
        
        // Copy assignment operator
        CLASS & operator=(CLASS other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept : CLASS()
        {
            swap(*this, other);
        }
        
        
        CLASS(
          const I * const * const idx,
          const I * const * const jdx,
          const T * const * const val,
          const I * entry_counts,
          const I list_count,
          const I m_,
          const I n_,
          const I final_thread_count,
          const bool compress   = true,
          const int  symmetrize = 0
        )
        :   BASE ( m_, n_, list_count )
        {
            FromTriples( idx, jdx, val, entry_counts,
                    list_count, final_thread_count, compress, symmetrize );
        }

        CLASS(
            const std::vector<std::vector<I>> & idx,
            const std::vector<std::vector<I>> & jdx,
            const std::vector<std::vector<T>> & val,
            const I m_,
            const I n_,
            const I final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE ( m_, n_, static_cast<I>(idx.size()) )
        {
            I list_count = static_cast<I>(idx.size());
            Tensor1<const I*, I> i (list_count);
            Tensor1<const I*, I> j (list_count);
            Tensor1<const T*, I> a (list_count);
            Tensor1<I ,I> entry_counts (list_count);
            
            for( I thread = 0; thread < list_count; ++thread )
            {
                i[thread] = idx[thread].data();
                j[thread] = jdx[thread].data();
                a[thread] = val[thread].data();
                entry_counts[thread] = static_cast<I>(idx[thread].size());
            }
            
            FromTriples( i.data(), j.data(), a.data(), entry_counts.data(),
                    list_count, final_thread_count, compress, symmetrize );
        }
        
        virtual ~CLASS() override = default;

    protected:
        
        void FromTriples(
            const I * const * const idx,    // list of lists of i-indices
            const I * const * const jdx,    // list of lists of j-indices
            const T * const * const val,    // list of lists of nonzero values
            const I * entry_counts,         // list of lengths of the lists above
            const I list_count,             // number of lists
            const I final_thread_count,     // number of threads that the matrix shall use
            const bool compress   = true,   // whether to do additive assembly or not
            const int  symmetrize = 0       // whether to symmetrize the matrix
        )
        {
            // Parallel sparse matrix assembly using counting sort.
            // Counting sort employs list_count threads (one per list).
            // Sorting of column indices and compression step employ final_thread_count threads.
            
            // k-th i-list goes from idx[k] to &idx[k][entry_counts[k]] (last one excluded)
            // k-th j-list goes from jdx[k] to &jdx[k][entry_counts[k]] (last one excluded)
            // and k goes from 0 to list_count (last one excluded)
            
            ptic(ClassName()+"::FromTriples");
            
            if( symmetrize )
            {
                logprint(ClassName()+"::FromTriples symmetrize");
            }
            else
            {
                logprint(ClassName()+"::FromTriples no symmetrize");
            }
            
            if( compress )
            {
                logprint(ClassName()+"::FromTriples compress");
            }
            else
            {
                logprint(ClassName()+"::FromTriples no compress");
            }
            
            Tensor2<I,I> counters = AssemblyCounters(
                idx, jdx, entry_counts, list_count, m, symmetrize
            );
            
            const I nnz = counters(list_count-1,m-1);
            
            if( nnz > 0 )
            {
                inner  = Tensor1<I,I>( nnz );
                values = Tensor1<T,I>( nnz );
            
                I * restrict const outer__ = outer.data();
                I * restrict const inner__ = inner.data();
                T * restrict const value__ = values.data();

                copy_buffer( counters.data(list_count-1), &outer__[1], m );

                // The counters array tells each thread where to write.
                // Since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                
                // TODO: The threads write quite chaotically to inner_ and value_. This might cause a lot of false sharing. Nontheless, it seems to scale quite well -- at least among 4 threads!
                
                // TODO: False sharing can be prevented by not distributing whole sublists of idx, jdx, val to the threads but by distributing the rows of the final matrix, instead. It's just a bit fiddly, though.
                
                ptic(ClassName()+"::FromTriples -- writing reordered data");
                
                #pragma omp parallel for num_threads( list_count )
                for( I thread = 0; thread < list_count; ++thread )
                {
                    const I entry_count = entry_counts[thread];
                    
                    const I * restrict const thread_idx = idx[thread];
                    const I * restrict const thread_jdx = jdx[thread];
                    const T * restrict const thread_val = val[thread];
                    
                          I * restrict const c = counters.data(thread);
                    
                    for( I k = entry_count - 1; k > -1; --k )
                    {
                        const I i = thread_idx[k];
                        const I j = thread_jdx[k];
                        const T a = thread_val[k];
                        
                        {
                            const I pos  = --c[i];
                            inner__[pos] = j;
                            value__[pos] = a;
                        }
                        
                        // Write the transposed matrix (diagonal excluded) in the same go in order to symmetrize the matrix. (Typical use case: Only the upper triangular part of a symmetric matrix is stored in idx, jdx, and val, but we need the full, symmetrized matrix.)
                        if( (symmetrize != 0) && (i != j) )
                        {
                            const I pos  = --c[j];
                            inner__[pos] = i;
                            value__[pos] = a;
                        }
                    }
                }
                
                ptoc(ClassName()+"::FromTriples -- writing reordered data");
                
                // Now all j-indices and nonzero values lie in the correct row (as indexed by outer).
                
                // From here on, we may use as many threads as we want.
                SetThreadCount( final_thread_count );

                // We have to sort b_inner to be compatible with the CSR format.
                SortInner();
                
                // Deal with duplicated {i,j}-pairs (additive assembly).
                if( compress )
                {
                    Compress();
                }
            }
            else
            {
                SetThreadCount( final_thread_count );
            }
            
            ptoc(ClassName()+"::FromTriples");
        }
        
    public:
        
        Tensor1<T,I> & Values()
        {
            return values;
        }
        
        const Tensor1<T,I> & Values() const
        {
            return values;
        }
        
        Tensor1<T,I> & Value()
        {
            return values;
        }
        
        const Tensor1<T,I> & Value() const
        {
            return values;
        }
        
        
        T operator()( const I i, const I j ) const
        {
            const I index = this->FindNonzeroPosition(i,j);
            
            return (index>=static_cast<I>(0)) ? values[index] : static_cast<T>(0) ;
        }
        
        
        CLASS Transpose() const
        {
            ptic(ClassName()+"::Transpose");
            
            if( WellFormed() )
            {
                RequireJobPtr();

                Tensor2<I,I> counters = CreateTransposeCounters();

                CLASS<T,I> B ( n, m, outer[m], thread_count );

                copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);

                          I * restrict const B_inner  = B.Inner().data();
                          T * restrict const B_values = B.Value().data();

                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
                    const T * restrict const A_values = Value().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I k_begin = A_outer[i  ];
                        const I k_end   = A_outer[i+1];
                        
                        for( I k = k_end-1; k > k_begin-1; --k )
                        {
                            const I j = A_inner[k];
                            const I pos = --c[ j ];
                            B_inner [pos] = i;
                            B_values[pos] = A_values[k];
                        }
                    }
                }

                // Finished counting sort.

                // We only have to care about the correct ordering of inner indices and values.
                B.SortInner();

                ptoc(ClassName()+"::Transpose");
                
                return B;
            }
            else
            {
                CLASS<T,I> B ( n, m, 0, thread_count );
                return B;
            }
        }
        
        
        void SortInner() override
        {
            ptic(ClassName()+"::SortInner");
            
            if( WellFormed() )
            {
                RequireJobPtr();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    TwoArrayQuickSort<I,T,I> quick_sort;

                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                    const I * restrict const outer__  = outer.data();
                          I * restrict const inner__  = inner.data();
                          T * restrict const values__ = values.data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I begin = outer__[i  ];
                        const I end   = outer__[i+1];
                        
                        quick_sort( inner__ + begin, values__ + begin, end - begin );
                    }
                }
            }
            
            ptoc(ClassName()+"::SortInner");
        }
        
        
        void Compress() override
        {
            // Removes duplicate {i,j}-pairs by adding their corresponding nonzero values.
            
            ptic(ClassName()+"::Compress");
            
            if( WellFormed() )
            {
                RequireJobPtr();

                Tensor1<I,I> new_outer (outer.Size(),0);
                
                const I * restrict const outer__     = outer.data();
                      I * restrict const inner__     = inner.data();
                      T * restrict const values__    = values.data();
                      I * restrict const new_outer__ = new_outer.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
    //                // Starting position of thread in inner list.
    //                thread_info(thread,0)= outer[i_begin];
    //                // End position of thread in inner list (not important).
    //                thread_info(thread,1)= outer[i_end  ];
    //                // Number of nonzeroes in thread after compression.
    //                thread_info(thread,2)= static_cast<I>(0);
                    
                    // To where we write.
                    I jj_new        = outer__[i_begin];
                    I next_jj_begin = outer__[i_begin];
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = next_jj_begin;
                        const I jj_end   = outer__[i+1];
                        
                        // Memoize the next entry in outer because outer will be overwritten
                        next_jj_begin = jj_end;
                        
                        I row_nonzero_counter = static_cast<I>(0);
                        
                        // From where we read.
                        I jj = jj_begin;
                        
                        while( jj< jj_end )
                        {
                            I j = inner__ [jj];
                            T a = values__[jj];
                            
                            if( jj > jj_new )
                            {
                                inner__ [jj] = static_cast<I>(0);
                                values__[jj] = static_cast<T>(0);
                            }
                            
                            ++jj;
            
                            while( (jj < jj_end) && (j == inner__[jj]) )
                            {
                                a+= values__[jj];
                                if( jj > jj_new )
                                {
                                    inner__ [jj] = static_cast<I>(0);
                                    values__[jj] = static_cast<T>(0);
                                }
                                ++jj;
                            }
                            
                            inner__ [jj_new] = j;
                            values__[jj_new] = a;
                            
                            jj_new++;
                            row_nonzero_counter++;
                        }
                        
                        new_outer__[i+1] = row_nonzero_counter;
                        
    //                    thread_info(thread,2) += row_nonzero_counter;
                    }
                }
                
                // This is the new array of outer indices.
                new_outer.Accumulate( thread_count );
                
                const I nnz = new_outer[m];
                
                Tensor1<I,I> new_inner  (nnz,0);
                Tensor1<T,I> new_values (nnz,0);
                
                I * restrict const new_inner__  = new_inner.data();
                T * restrict const new_values__ = new_values.data();
                  
                //TODO: Parallelization might be a bad idea here.
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];

                    const I new_pos = new_outer__[i_begin];
                    const I     pos =     outer__[i_begin];

                    const I thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];

                    // Starting position of thread in inner list.
                    
                    copy_buffer( &inner__[pos],  &new_inner__[new_pos],  thread_nonzeroes );
                    
                    copy_buffer( &values__[pos], &new_values__[new_pos], thread_nonzeroes );
                }
                
                swap( new_outer,  outer  );
                swap( new_inner,  inner  );
                swap( new_values, values );
                
                job_ptr = JobPointers<I>();
            }
            
            ptoc(ClassName()+"::Compress");
        }
        
        
        CLASS Dot( const CLASS<T,I> & B ) const
        {
            ptic(ClassName()+"::Dot");
                        
            if(WellFormed() )
            {
                RequireJobPtr();
                
                Tensor2<I,I> counters ( thread_count, m, static_cast<I>(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);
                    
                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
                    
                    const I * restrict const B_outer  = B.Outer().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = A_outer[i  ];
                        const I jj_end   = A_outer[i+1];
                        
                        for( I jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const I j = A_inner[jj];
                            
                            c[i] += (B_outer[j+1] - B_outer[j]);
                        }
                    }
                }
                
                AccumulateAssemblyCounters_Parallel( counters );
                
                const I nnz = counters.data(thread_count-1)[m-1];
                
                CLASS<T,I> C ( m, B.ColCount(), nnz, thread_count );
                
                copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                
                #pragma omp parallel for num_threads( thread_count )
                for( I thread = 0; thread < thread_count; ++thread )
                {
                    const I i_begin = job_ptr[thread  ];
                    const I i_end   = job_ptr[thread+1];
                    
                          I * restrict const c = counters.data(thread);

                    const I * restrict const A_outer  = Outer().data();
                    const I * restrict const A_inner  = Inner().data();
                    const T * restrict const A_values = Value().data();
                    
                    const I * restrict const B_outer  = B.Outer().data();
                    const I * restrict const B_inner  = B.Inner().data();
                    const T * restrict const B_values = B.Value().data();
                    
                          I * restrict const C_inner  = C.Inner().data();
                          T * restrict const C_values = C.Value().data();
                    
                    for( I i = i_begin; i < i_end; ++i )
                    {
                        const I jj_begin = A_outer[i  ];
                        const I jj_end   = A_outer[i+1];
                        
                        for( I jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const I j = A_inner[jj];
                            
                            const I kk_begin = B_outer[j  ];
                            const I kk_end   = B_outer[j+1];
                            
                            for( I kk = kk_end-1; kk > kk_begin-1; --kk )
                            {
                                const I k = B_inner[kk];
                                const I pos = --c[ i ];
                                
                                C_inner [pos] = k;
                                C_values[pos] = A_values[jj] * B_values[kk];
                            }
                        }
                    }
                }
                // Finished expansion phase (counting sort).
                
                // Now we have to care about the correct ordering of inner indices and values.
                C.SortInner();
                
                // Finally we compress duplicates in inner and values.
                C.Compress();
                
                ptoc(ClassName()+"::Dot");
                
                return C;
            }
            else
            {
                return CLASS<T,I> ();
            }
        }
        
        SparseBinaryMatrixCSR<I> DotBinary( const CLASS<T,I> & B ) const
        {
            SparseBinaryMatrixCSR<I> result;
            
            BASE C = this->DotBinary_(B);
            
            swap(result,C);
            
            return result;
        }
        
        
    public:
        
//##############################################################################################
//####          Matrix Multiplication
//##############################################################################################
                

        // Use own nonzero values.
        template<typename T_in, typename T_out>
        void Dot(
            const T alpha,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const I cols = static_cast<I>(1)
        ) const
        {
            Dot_( values.data(), alpha, X, beta, Y, cols );
        }

        // Use own nonzero values.
        template<typename T_in, typename T_out>
        void Dot(
            const T alpha,
            const Tensor1<T_in, I> & X,
            const T_out beta,
                  Tensor1<T_out,I> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( values.data(), alpha, X.data(), beta, Y.data(), static_cast<I>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        // Use own nonzero values.
        template<typename T_in, typename T_out>
        void Dot(
            const T alpha,
            const Tensor2<T_in, I> & X,
            const T_out beta,
                  Tensor2<T_out,I> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
            {
                Dot_( values.data(), alpha, X.data(), beta, Y.data(), X.Dimension(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        
        
        // Use external list of values.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const T_ext * ext_values,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const I cols = static_cast<I>(1)
        ) const
        {
            Dot_( ext_values, alpha, X, beta, Y, cols );
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor1<T_ext,    I> & ext_values,
            const Tensor1<T_in, I> & X,
            const T_out beta,
                  Tensor1<T_out,I> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( ext_values.data(), alpha, X.data(), beta, Y.data(), static_cast<I>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor1<T_ext,I> & ext_values,
            const Tensor2<T_in, I> & X,
            const T_out beta,
                  Tensor2<T_out,I> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
            {
                Dot_( ext_values.data(), alpha, X.data(), beta, Y.data(), X.Dimension(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        void FillLowerTriangleFromUpperTriangle()
        {
            FillLowerTriangleFromUpperTriangle( values.data() );
        }
        
        void FillUpperTriangleFromLowerTriangle()
        {
            FillUpperTriangleFromLowerTriangle( values.data() );
        }
        
        
    public:
        
        std::string Stats() const
        {
            std::stringstream s;
            
            s
            << "\n==== "+ClassName()+" Stats ====" << "\n\n"
            << " RowCount()      = " << RowCount() << "\n"
            << " ColCount()      = " << ColCount() << "\n"
            << " NonzeroCount()  = " << NonzeroCount() << "\n"
            << " ThreadCount()   = " << ThreadCount() << "\n"
            << " Outer().Size()  = " << Outer().Size() << "\n"
            << " Inner().Size()  = " << Inner().Size() << "\n"
            << " Value().Size()  = " << Value().Size() << "\n"
            << "\n==== "+ClassName()+" Stats ====\n" << std::endl;
            
            return s.str();
        }
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
    }; // CLASS
    
    
} // namespace Tensors

#undef BASE
#undef CLASS

