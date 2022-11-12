#pragma once

#define CLASS SparseMatrixCSR
#define BASE  SparsityPatternCSR<Int_,LInt_>

namespace Tensors
{

    template<typename Scalar_, typename Int_, typename LInt_>
    class CLASS : public BASE
    {
        
    public:
        
        using Scalar = Scalar_;
        using Int    = Int_;
        using LInt   = LInt_;
        
    protected:
        
        using BASE::m;
        using BASE::n;
        using BASE::outer;
        using BASE::inner;
        using BASE::thread_count;
        using BASE::job_ptr;
        using BASE::upper_triangular_job_ptr;
        using BASE::lower_triangular_job_ptr;

        Tensor1<Scalar,LInt> values;
        
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
        
        CLASS()
        :   BASE()
        {}
        
        template<typename I_0, typename I_1, typename I_3>
        CLASS(
            const I_0 m_,
            const I_1 n_,
            const I_3 thread_count_
        )
        :   BASE( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        template<typename I_0, typename I_1, typename I_2, typename I_3>
        CLASS(
            const I_0 m_,
            const I_1 n_,
            const I_2 nnz_,
            const I_3 thread_count_
        )
        :   BASE   ( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<LInt>(nnz_), static_cast<Int>(thread_count_) )
        ,   values ( static_cast<LInt>(nnz_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_2);
            ASSERT_INT(I_3);
        }
        
        template<typename S, typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
        CLASS(
            const J_0 * const outer_,
            const J_1 * const inner_,
            const S   * const values_,
            const I_0 m_,
            const I_1 n_,
            const I_3 thread_count_
        )
        :   BASE    ( outer_,  inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        ,   values  ( values_, outer_[static_cast<Int>(m_)] )
        {
            ASSERT_ARITHMETIC(S);
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        template<typename I_0, typename I_1, typename I_3>
        CLASS(
            Tensor1<LInt  , Int> && outer_,
            Tensor1< Int  ,LInt> && inner_,
            Tensor1<Scalar, Int> && values_,
            const I_0 m_,
            const I_1 n_,
            const I_3 thread_count_
        )
        :   BASE   ( std::move(outer_), std::move(inner_), static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        ,   values ( std::move(values_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   BASE    ( other        )
        ,   values  ( other.values )
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
          const Int * const * const idx,
          const Int * const * const jdx,
          const Scalar * const * const val,
          const Int * entry_counts,
          const Int list_count,
          const Int m_,
          const Int n_,
          const Int final_thread_count,
          const bool compress   = true,
          const int  symmetrize = 0
        )
        :   BASE ( m_, n_, list_count )
        {
            FromTriples( idx, jdx, val, entry_counts, list_count, final_thread_count, compress, symmetrize );
        }

        CLASS(
            const std::vector<std::vector<Int>>    & idx,
            const std::vector<std::vector<Int>>    & jdx,
            const std::vector<std::vector<Scalar>> & val,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE ( m_, n_, static_cast<Int>(idx.size()) )
        {
            Int list_count = static_cast<Int>(idx.size());
            Tensor1<const Int*,Int> i      (list_count);
            Tensor1<const Int*,Int> j      (list_count);
            Tensor1<const Scalar*,Int> a   (list_count);
            Tensor1<Int ,Int> entry_counts (list_count);
            
            for( Int thread = 0; thread < list_count; ++thread )
            {
                i[thread] = idx[thread].data();
                j[thread] = jdx[thread].data();
                a[thread] = val[thread].data();
                entry_counts[thread] = static_cast<Int>(idx[thread].size());
            }
            
            FromTriples( i.data(), j.data(), a.data(), entry_counts.data(),
                    list_count, final_thread_count, compress, symmetrize );
        }
        
        CLASS(
            const std::vector<TripleAggregator<Int,Int,Scalar,LInt>> & triples,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE ( m_, n_, static_cast<Int>(triples.size()) )
        {
            Int list_count = static_cast<Int>(triples.size());
            Tensor1<const Int*,Int> i      (list_count);
            Tensor1<const Int*,Int> j      (list_count);
            Tensor1<const Scalar*,Int> a   (list_count);
            Tensor1<Int ,Int> entry_counts (list_count);
            
            for( Int thread = 0; thread < list_count; ++thread )
            {
                i[thread] = triples[thread].Get_0().data();
                j[thread] = triples[thread].Get_1().data();
                a[thread] = triples[thread].Get_2().data();
                entry_counts[thread] = static_cast<Int>(triples[thread].Size());
            }
            
            FromTriples( i.data(), j.data(), a.data(), entry_counts.data(),
                    list_count, final_thread_count, compress, symmetrize );
        }
        
        virtual ~CLASS() override = default;

    protected:
        
        void FromTriples(
            const Int * const * const idx,    // list of lists of i-indices
            const Int * const * const jdx,    // list of lists of j-indices
            const Scalar * const * const val,    // list of lists of nonzero values
            const Int * entry_counts,         // list of lengths of the lists above
            const Int list_count,             // number of lists
            const Int final_thread_count,     // number of threads that the matrix shall use
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
            
            Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int>(
                idx, jdx, entry_counts, list_count, m, symmetrize
            );
            
            const Int nnz = counters(list_count-1,m-1);
            
            if( nnz > 0 )
            {
                inner  = Tensor1   <Int,LInt>( nnz );
                values = Tensor1<Scalar,LInt>( nnz );
            
                  LInt * restrict const outer__ = outer.data();
                   Int * restrict const inner__ = inner.data();
                Scalar * restrict const value__ = values.data();

                copy_buffer( counters.data(list_count-1), &outer__[1], m );

                // The counters array tells each thread where to write.
                // Since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                
                // TODO: The threads write quite chaotically to inner_ and value_. This might cause a lot of false sharing. Nontheless, it seems to scale quite well -- at least among 4 threads!
                
                // TODO: False sharing can be prevented by not distributing whole sublists of idx, jdx, val to the threads but by distributing the rows of the final matrix, instead. It's just a bit fiddly, though.
                
                ptic(ClassName()+"::FromTriples -- writing reordered data");
                
                #pragma omp parallel for num_threads( list_count )
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    const Int entry_count = entry_counts[thread];
                    
                    const    Int * restrict const thread_idx = idx[thread];
                    const    Int * restrict const thread_jdx = jdx[thread];
                    const Scalar * restrict const thread_val = val[thread];
                    
                            LInt * restrict const c = counters.data(thread);
                    
                    for( Int k = entry_count - 1; k > -1; --k )
                    {
                        const Int i = thread_idx[k];
                        const Int j = thread_jdx[k];
                        const Scalar a = thread_val[k];
                        
                        {
                            const LInt pos  = --c[i];
                            inner__[pos] = j;
                            value__[pos] = a;
                        }
                        
                        // Write the transposed matrix (diagonal excluded) in the same go in order to symmetrize the matrix. (Typical use case: Only the upper triangular part of a symmetric matrix is stored in idx, jdx, and val, but we need the full, symmetrized matrix.)
                        if( (symmetrize != 0) && (i != j) )
                        {
                            const LInt pos  = --c[j];
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
        
        Tensor1<Scalar,LInt> & Values()
        {
            return values;
        }
        
        const Tensor1<Scalar,LInt> & Values() const
        {
            return values;
        }
        
        Tensor1<Scalar,LInt> & Value()
        {
            return values;
        }
        
        const Tensor1<Scalar,LInt> & Value() const
        {
            return values;
        }
        
        
        Scalar operator()( const Int i, const Int j ) const
        {
            const LInt index = this->FindNonzeroPosition(i,j);
            
            return (index>=static_cast<LInt>(0)) ? values[index] : static_cast<Scalar>(0) ;
        }
        
        
        CLASS Transpose() const
        {
            ptic(ClassName()+"::Transpose");
            
            if( WellFormed() )
            {
                RequireJobPtr();

                Tensor2<Int,Int> counters = CreateTransposeCounters();

                CLASS B ( n, m, outer[m], thread_count );

                copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          Int * restrict const c = counters.data(thread);

                          Int * restrict const B_inner  = B.Inner().data();
                          Scalar * restrict const B_values = B.Value().data();

                    const Int * restrict const A_outer  = Outer().data();
                    const Int * restrict const A_inner  = Inner().data();
                    const Scalar * restrict const A_values = Value().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int k_begin = A_outer[i  ];
                        const Int k_end   = A_outer[i+1];
                        
                        for( Int k = k_end-1; k > k_begin-1; --k )
                        {
                            const Int j = A_inner[k];
                            const Int pos = --c[ j ];
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
                CLASS B ( n, m, 0, thread_count );
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
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    TwoArrayQuickSort<Int,Scalar,LInt> quick_sort;

                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    const   LInt * restrict const outer__  = outer.data();
                             Int * restrict const inner__  = inner.data();
                          Scalar * restrict const values__ = values.data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt begin = outer__[i  ];
                        const LInt end   = outer__[i+1];
                        
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

                Tensor1<LInt,Int> new_outer (outer.Size(),0);
                
                const   LInt * restrict const outer__     = outer.data();
                         Int * restrict const inner__     = inner.data();
                      Scalar * restrict const values__    = values.data();
                        LInt * restrict const new_outer__ = new_outer.data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
    //                // Starting position of thread in inner list.
    //                thread_info(thread,0)= outer[i_begin];
    //                // End position of thread in inner list (not important).
    //                thread_info(thread,1)= outer[i_end  ];
    //                // Number of nonzeroes in thread after compression.
    //                thread_info(thread,2)= static_cast<Int>(0);
                    
                    // To where we write.
                    LInt jj_new        = outer__[i_begin];
                    LInt next_jj_begin = outer__[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt jj_begin = next_jj_begin;
                        const LInt jj_end   = outer__[i+1];
                        
                        // Memoize the next entry in outer because outer will be overwritten
                        next_jj_begin = jj_end;
                        
                        LInt row_nonzero_counter = static_cast<Int>(0);
                        
                        // From where we read.
                        LInt jj = jj_begin;
                        
                        while( jj< jj_end )
                        {
                               Int j = inner__ [jj];
                            Scalar a = values__[jj];
                            
                            if( jj > jj_new )
                            {
                                inner__ [jj] = static_cast<Int>(0);
                                values__[jj] = static_cast<Scalar>(0);
                            }
                            
                            ++jj;
            
                            while( (jj < jj_end) && (j == inner__[jj]) )
                            {
                                a+= values__[jj];
                                if( jj > jj_new )
                                {
                                    inner__ [jj] = static_cast<Int>(0);
                                    values__[jj] = static_cast<Scalar>(0);
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
                
                const LInt nnz = new_outer[m];
                
                Tensor1<   Int,LInt> new_inner  (nnz,0);
                Tensor1<Scalar,LInt> new_values (nnz,0);
                
                   Int * restrict const new_inner__  = new_inner.data();
                Scalar * restrict const new_values__ = new_values.data();
                  
                //TODO: Parallelization might be a bad idea here.
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const  Int i_begin = job_ptr[thread  ];
                    const  Int i_end   = job_ptr[thread+1];

                    const LInt new_pos = new_outer__[i_begin];
                    const LInt     pos =     outer__[i_begin];

                    const LInt thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];

                    // Starting position of thread in inner list.
                    
                    copy_buffer( &inner__[pos],  &new_inner__[new_pos],  thread_nonzeroes );
                    
                    copy_buffer( &values__[pos], &new_values__[new_pos], thread_nonzeroes );
                }
                
                swap( new_outer,  outer  );
                swap( new_inner,  inner  );
                swap( new_values, values );
                
                job_ptr = JobPointers<Int>();
            }
            
            ptoc(ClassName()+"::Compress");
        }
                
        CLASS Permute( const Tensor1<Int,Int> & p, const Tensor1<Int,Int> & q, bool sort = true ) const
        {
            if( p.Dimension(0) != m )
            {
                eprint(ClassName()+"::Permute: Length of first argument does not coincide with RowCount().");
                return CLASS();
            }
            
            if( q.Dimension(0) != n )
            {
                eprint(ClassName()+"::Permute: Length of second argument does not coincide with ColCount().");
                return CLASS();
            }
            
            Permute( p.data(), q.data(), sort );
        }
        
        CLASS Permute( const Int * restrict const p, const Int * restrict const q, bool sort = true ) const
        {
            if( p == nullptr )
            {
                if( q == nullptr )
                {
                    // Just make a copy.
                    return CLASS(*this);
                }
                else
                {
                    return PermuteCols(q,sort);
                }
            }
            else
                if( q == nullptr )
                {
                    return PermuteRows(p);
                }
                else
                {
                    return PermuteRowsCols(p,q,sort);
                }
        }
        
    protected:
        
        CLASS PermuteRows( const Int * restrict const p ) const
        {
            CLASS B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
            
            {
                const LInt * restrict const A_outer = outer.data();
                      LInt * restrict const B_outer = B.Outer().data();
                
                B_outer[0] = 0;
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < m; ++i )
                {
                    const Int p_i = p[i];
                    
                    B_outer[i+1] = A_outer[p_i+1] - A_outer[p_i];
                }
            }

            B.Outer().Accumulate();
            
            {
                auto & B_job_ptr = B.JobPtr();

                const Int thread_count = B_job_ptr.ThreadCount();
                
                const   LInt * restrict const A_outer  = outer.data();
                const    Int * restrict const A_inner  = inner.data();
                const Scalar * restrict const A_values = values.data();

                const   LInt * restrict const B_outer  = B.Outer().data();
                         Int * restrict const B_inner  = B.Inner().data();
                      Scalar * restrict const B_values = B.Values().data();

                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = B_job_ptr[thread  ];
                    const Int i_end   = B_job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int p_i = p[i];
                        const LInt A_begin = A_outer[p_i  ];
//                        const LInt A_end   = A_outer[p_i+1];

                        const LInt B_begin = B_outer[i  ];
                        const LInt B_end   = B_outer[i+1];
                        
                        copy_buffer( &B_inner [B_begin], &A_inner [A_begin], B_end - B_begin);
                        copy_buffer( &B_values[B_begin], &A_values[A_begin], B_end - B_begin);
                    }
                }
            }

            return B;
        }
        
        CLASS PermuteCols( const Int * restrict const q, bool sort = true ) const
        {
            CLASS B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
            
            Tensor1<Int,Int> q_inv_buffer ( ColCount() );
            Int * restrict const q_inv = q_inv_buffer.data();

            {
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int j = 0; j < n; ++j )
                {
                    q_inv[q[j]] = j;
                }
            }
             
            copy_buffer( outer.data(), B.Outer().data() );
           
            {
                auto & B_job_ptr = B.JobPtr();

                const Int thread_count = B_job_ptr.ThreadCount();
                
                const   LInt * restrict const A_outer  = outer.data();
                const    Int * restrict const A_inner  = inner.data();
                const Scalar * restrict const A_values = values.data();

                const   LInt * restrict const B_outer  = B.Outer().data();
                         Int * restrict const B_inner  = B.Inner().data();
                      Scalar * restrict const B_values = B.Values().data();

                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    TwoArrayQuickSort<Int,Scalar,LInt> quick_sort;
                    
                    const Int i_begin = B_job_ptr[thread  ];
                    const Int i_end   = B_job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt B_begin = B_outer[i  ];
                        const LInt B_end   = B_outer[i+1];

                        const LInt k_max = B_end - B_begin;

                        for( LInt k = 0; k < k_max; ++k )
                        {
                            B_inner[B_begin+k] = q_inv[A_inner[B_begin+k]];
                        }

                        copy_buffer( &A_values[A_begin], &B_values[B_begin], k_max );

                        if( sort )
                        {
                            quick_sort( &B_inner[B_begin], &B_values[B_begin], k_max );
                        }
                    }
                }
            }

            return B;
        }
        
        CLASS PermuteRowCols( const Int * restrict const p, const Int * restrict const q, bool sort = true ) const
        {
            CLASS B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
            
            Tensor1<Int,Int> q_inv_buffer ( ColCount() );
            Int * restrict const q_inv = q_inv_buffer.data();

            {
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int j = 0; j < n; ++j )
                {
                    q_inv[q[j]] = j;
                }
            }
            
            {
                const LInt * restrict const A_outer = outer.data();
                      LInt * restrict const B_outer = B.Outer().data();
                
                B_outer[0] = 0;
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < m; ++i )
                {
                    const Int p_i = p[i];
                    
                    B_outer[i+1] = A_outer[p_i+1] - A_outer[p_i];
                }
            }

            B.Outer().Accumulate();
            
            {
                auto & B_job_ptr = B.JobPtr();

                const Int thread_count = B_job_ptr.ThreadCount();
                
                const   LInt * restrict const A_outer  = outer.data();
                const    Int * restrict const A_inner  = inner.data();
                const Scalar * restrict const A_values = values.data();

                const   LInt * restrict const B_outer  = B.Outer().data();
                         Int * restrict const B_inner  = B.Inner().data();
                      Scalar * restrict const B_values = B.Values().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    TwoArrayQuickSort<Int,Scalar,LInt> quick_sort;
                    
                    const Int i_begin = B_job_ptr[thread  ];
                    const Int i_end   = B_job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int p_i = p[i];
                        const LInt A_begin = A_outer[p_i  ];
//                        const LInt A_end   = A_outer[p_i+1];

                        const LInt B_begin = B_outer[i  ];
                        const LInt B_end   = B_outer[i+1];

                        const LInt k_max = B_end - B_begin;

                        for( LInt k = 0; k < k_max; ++k )
                        {
                            B_inner [B_begin+k] = q_inv[A_inner[A_begin+k]];
                        }
                        
                        copy_buffer( &A_values[A_begin], &B_values[B_begin], k_max );

                        if( sort )
                        {
                            quick_sort( &B_inner[B_begin], &B_values[B_begin], k_max );
                        }
                    }
                }
            }

            return B;
        }
        
    public:
        
        CLASS Dot( const CLASS & B ) const
        {
            ptic(ClassName()+"::Dot");
                        
            if(WellFormed() )
            {
                RequireJobPtr();
                
                Tensor2<LInt,Int> counters ( thread_count, m, static_cast<Int>(0) );
                
                // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                // https://en.wikipedia.org/wiki/Counting_sort
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          LInt * restrict const c = counters.data(thread);
                    
                    const LInt * restrict const A_outer  = Outer().data();
                    const  Int * restrict const A_inner  = Inner().data();
                    
                    const LInt * restrict const B_outer  = B.Outer().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt jj_begin = A_outer[i  ];
                        const LInt jj_end   = A_outer[i+1];
                        
                        for( LInt jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const Int j = A_inner[jj];
                            
                            c[i] += (B_outer[j+1] - B_outer[j]);
                        }
                    }
                }
                
                AccumulateAssemblyCounters_Parallel( counters );
                
                const LInt nnz = counters.data(thread_count-1)[m-1];
                
                CLASS C ( m, B.ColCount(), nnz, thread_count );
                
                copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          LInt * restrict const c = counters.data(thread);

                    const   LInt * restrict const A_outer  = Outer().data();
                    const    Int * restrict const A_inner  = Inner().data();
                    const Scalar * restrict const A_values = Value().data();
                    
                    const   LInt * restrict const B_outer  = B.Outer().data();
                    const    Int * restrict const B_inner  = B.Inner().data();
                    const Scalar * restrict const B_values = B.Value().data();
                    
                            LInt * restrict const C_inner  = C.Inner().data();
                          Scalar * restrict const C_values = C.Value().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt jj_begin = A_outer[i  ];
                        const LInt jj_end   = A_outer[i+1];
                        
                        for( LInt jj = jj_begin; jj < jj_end; ++jj )
                        {
                            const Int j = A_inner[jj];
                            
                            const LInt kk_begin = B_outer[j  ];
                            const LInt kk_end   = B_outer[j+1];
                            
//                            for( LInt kk = kk_end-1; kk > kk_begin-1; --kk )
                            for( LInt kk = kk_end; kk --> kk_begin; )
                            {
                                const Int k = B_inner[kk];
                                const LInt pos = --c[ i ];
                                
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
                return CLASS ();
            }
        }
        
        SparseBinaryMatrixCSR<Int,LInt> DotBinary( const CLASS & B ) const
        {
            SparseBinaryMatrixCSR<Int,LInt> result;
            
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
            const Scalar alpha,     const T_in  * X,
            const T_out beta,        T_out * Y,
            const Int cols = static_cast<Int>(1)
        ) const
        {
            Dot_( values.data(), alpha, X, beta, Y, cols );
        }

        // Use own nonzero values.
        template<typename T_in, typename T_out>
        void Dot(
            const Scalar alpha,     const Tensor1<T_in, Int> & X,
            const T_out beta,        Tensor1<T_out,Int> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( values.data(), alpha, X.data(), beta, Y.data(), static_cast<Int>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        // Use own nonzero values.
        template<typename T_in, typename T_out>
        void Dot(
            const Scalar alpha,
            const Tensor2<T_in, Int> & X,
            const T_out beta,
                  Tensor2<T_out,Int> & Y
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
            const Int cols = static_cast<Int>(1)
        ) const
        {
            Dot_( ext_values, alpha, X, beta, Y, cols );
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor1<T_ext,    Int> & ext_values,
            const Tensor1<T_in, Int> & X,
            const T_out beta,
                  Tensor1<T_out,Int> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( ext_values.data(), alpha, X.data(), beta, Y.data(), static_cast<Int>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor1<T_ext,Int> & ext_values,
            const Tensor2<T_in, Int> & X,
            const T_out beta,
                  Tensor2<T_out,Int> & Y
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
            return TO_STD_STRING(CLASS)+"<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
        }
        
    }; // CLASS
    
    
} // namespace Tensors

#undef BASE
#undef CLASS

