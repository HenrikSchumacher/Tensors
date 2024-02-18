#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Scal_, typename Int_, typename LInt_>
        class MatrixCSR : public Sparse::PatternCSR<Int_,LInt_>
        {
        private:
            
            using Base_T = Sparse::PatternCSR<Int_,LInt_>;
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            using Base_T::m;
            using Base_T::n;
            using Base_T::outer;
            using Base_T::inner;
            using Base_T::thread_count;
            using Base_T::job_ptr;
            using Base_T::job_ptr_initialized;
            using Base_T::diag_ptr;
            using Base_T::inner_sorted;
            using Base_T::duplicate_free;
            using Base_T::upper_triangular_job_ptr;
            using Base_T::lower_triangular_job_ptr;
            
            // I have to make this mutable so that SortInner and Compress can have the const attribute.
            mutable Tensor1<Scal,LInt> values;
            
        public:
            
            using Base_T::RowCount;
            using Base_T::ColCount;
            using Base_T::NonzeroCount;
            using Base_T::ThreadCount;
            using Base_T::SetThreadCount;
            using Base_T::Outer;
            using Base_T::Inner;
            using Base_T::RequireDiag;
            using Base_T::JobPtr;
            using Base_T::RequireJobPtr;
            using Base_T::RequireUpperTriangularJobPtr;
            using Base_T::RequireLowerTriangularJobPtr;
            using Base_T::UpperTriangularJobPtr;
            using Base_T::LowerTriangularJobPtr;
            using Base_T::CreateTransposeCounters;
            using Base_T::WellFormed;
            using Base_T::Dot_;
            
            MatrixCSR()
            :   Base_T()
            {}
            
            template<typename I_0, typename I_1, typename I_3>
            MatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            template<typename I_0, typename I_1, typename I_2, typename I_3>
            MatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T   ( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<LInt>(nnz_), static_cast<Int>(thread_count_) )
            ,   values ( static_cast<LInt>(nnz_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_2);
                ASSERT_INT(I_3);
            }
            
            template<typename S, typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            MatrixCSR(
                const J_0 * const outer_,
                const J_1 * const inner_,
                const S   * const values_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T  ( outer_,  inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
            ,   values  ( values_, outer_[static_cast<Int>(m_)] )
            {
                ASSERT_ARITHMETIC(S);
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            template<typename I_0, typename I_1, typename I_3>
            MatrixCSR(
                Tensor1<LInt, Int> && outer_,
                Tensor1< Int,LInt> && inner_,
                Tensor1<Scal,LInt> && values_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T   ( std::move(outer_), std::move(inner_), static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
            ,   values ( std::move(values_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            // Copy constructor
            MatrixCSR( const MatrixCSR & other ) noexcept
            :   Base_T  ( other        )
            ,   values  ( other.values )
            {
                logprint("Copy of "+ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nn z = "+ToString(NonzeroCount()));
            }
            
            friend void swap (MatrixCSR & A, MatrixCSR & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( static_cast<Base_T&>(A), static_cast<Base_T&>(B) );
                swap( A.values,                B.values                );
            }
            
            // (Copy-)assignment operator
            MatrixCSR & operator=( MatrixCSR other ) noexcept // Pass by value is okay, because we use copy-swap idiom and copy elision.
            {
                // see https://stackoverflow.com/a/3279550/8248900 for details
                
                swap(*this, other);
                
                return *this;
            }
            
            // Move constructor
            MatrixCSR( MatrixCSR && other ) noexcept
            :   MatrixCSR()
            {
                swap(*this, other);
            }
            
            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            MatrixCSR(
                  const Int    * const * const idx,
                  const Int    * const * const jdx,
                  const Scal   * const * const val,
                  const LInt   *         const entry_counts,
                  const Int list_count,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( m_, n_, list_count )
            {
                FromTriples( idx, jdx, val, entry_counts, list_count, final_thread_count, compressQ, symmetrize );
            }
            
            MatrixCSR(
                const LInt nnz_,
                const Int  * const i,
                const Int  * const j,
                const Scal * const a,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   Base_T ( m_, n_, thread_count )
            {
                Tensor1<const Int  *,Int> idx    (thread_count);
                Tensor1<const Int  *,Int> jdx    (thread_count);
                Tensor1<const Scal *,Int> val    (thread_count);
                Tensor1<      LInt  ,Int> counts (thread_count);
                
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const LInt begin = JobPointer<LInt>(nnz_, thread_count, thread    );
                    const LInt end   = JobPointer<LInt>(nnz_, thread_count, thread + 1);
                    
                    idx[thread] = &i[begin];
                    jdx[thread] = &j[begin];
                    val[thread] = &a[begin];
                    counts[thread] = end-begin;
                }
                
                FromTriples( idx.data(), jdx.data(), val.data(), counts.data(), thread_count, thread_count, compressQ, symmetrize );
            }
            
            MatrixCSR(
                  cref<std::vector<std::vector<Int>>>  idx,
                  cref<std::vector<std::vector<Int>>>  jdx,
                  cref<std::vector<std::vector<Scal>>> val,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( m_, n_, static_cast<Int>(idx.size()) )
            {
                Int list_count = static_cast<Int>(idx.size());
                Tensor1<const Int *,Int> i      (list_count);
                Tensor1<const Int *,Int> j      (list_count);
                Tensor1<const Scal*,Int> a      (list_count);
                Tensor1<LInt,Int> entry_counts  (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    a[thread] = val[thread].data();
                    entry_counts[thread] = static_cast<LInt>(idx[thread].size());
                }
                
                FromTriples( i.data(), j.data(), a.data(), entry_counts.data(),
                            list_count, final_thread_count, compressQ, symmetrize );
            }
            
            MatrixCSR(
                  const std::vector<TripleAggregator<Int,Int,Scal,LInt>> & triples,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( m_, n_, static_cast<Int>(triples.size()) )
            {
                Int list_count = static_cast<Int>(triples.size());
                
                Tensor1<const Int *,Int> i     (list_count);
                Tensor1<const Int *,Int> j     (list_count);
                Tensor1<const Scal*,Int> a     (list_count);
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = triples[thread].Get_0().data();
                    j[thread] = triples[thread].Get_1().data();
                    a[thread] = triples[thread].Get_2().data();
                    entry_counts[thread] = static_cast<LInt>(triples[thread].Size());
                }
                
                FromTriples( i.data(), j.data(), a.data(), entry_counts.data(),
                            list_count, final_thread_count, compressQ, symmetrize );
            }
            
            virtual ~MatrixCSR() override = default;
            
        protected:
            
            void FromTriples(
                const Int  * const * const idx,               // list of lists of i-indices
                const Int  * const * const jdx,               // list of lists of j-indices
                const Scal * const * const val,               // list of lists of nonzero values
                const LInt         * const entry_counts,      // list of lengths of the lists above
                const Int list_count,                         // number of lists
                const Int final_thread_count,                 // number of threads that the matrix shall use
                const bool compressQ   = true,                 // whether to do additive assembly or not
                const int  symmetrize = 0                     // whether to symmetrize the matrix
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
                
                if( compressQ )
                {
                    logprint(ClassName()+"::FromTriples compressQ");
                }
                else
                {
                    logprint(ClassName()+"::FromTriples no compressQ");
                }
                
                Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int>(
                    idx, jdx, entry_counts, list_count, m, symmetrize
                );
                
                const LInt nnz = counters[list_count-1][m-1];
                
                if( nnz > 0 )
                {
                    inner  = Tensor1<Int ,LInt>( nnz );
                    values = Tensor1<Scal,LInt>( nnz );
                    
                    mptr<LInt> outer__ = outer.data();
                    mptr<Int>  inner__ = inner.data();
                    mptr<Scal> value__ = values.data();
                    
                    copy_buffer<VarSize,Sequential>( counters.data(list_count-1), &outer__[1], m );
                    
                    // The counters array tells each thread where to write.
                    // Since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                    
                    // TODO: The threads write quite chaotically to inner_ and value_. This might cause a lot of false sharing. Nonetheless, it seems to scale quite well -- at least among 4 threads!
                    
                    // TODO: False sharing can be prevented by not distributing whole sublists of idx, jdx, val to the threads but by distributing the rows of the final matrix, instead. It's just a bit fiddly, though.
                    
                    ptic(ClassName()+"::FromTriples -- writing reordered data");
                    
                    ParallelDo(
                        [=,&counters]( const Int thread )
                        {
                            const LInt entry_count = entry_counts[thread];
                            
                            cptr<Int>  thread_idx = idx[thread];
                            cptr<Int>  thread_jdx = jdx[thread];
                            cptr<Scal> thread_val = val[thread];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            for( LInt k = entry_count; k --> 0; )
                            {
                                const Int  i = thread_idx[k];
                                const Int  j = thread_jdx[k];
                                const Scal a = thread_val[k];
                                
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
                        },
                        list_count
                    );
                    
                    ptoc(ClassName()+"::FromTriples -- writing reordered data");
                    
                    // Now all j-indices and nonzero values lie in the correct row (as indexed by outer).

                    // From here on, we may use as many threads as we want.
                    SetThreadCount( final_thread_count );
                    
                    // We have to sort b_inner to be compatible with the CSR format.
                    SortInner();
                    
                    // Deal with duplicated {i,j}-pairs (additive assembly).
                    if( compressQ )
                    {
                        Compress();
                    }
                    else
                    {
                        duplicate_free = true; // We have to rely on the user here...
                    }
                }
                else
                {
                    SetThreadCount( final_thread_count );
                }
                
                ptoc(ClassName()+"::FromTriples");
            }
            
        public:
            
            mref<Tensor1<Scal,LInt>> Values()
            {
                return values;
            }
            
            cref<Tensor1<Scal,LInt>> Values() const
            {
                return values;
            }
            
            mref<Tensor1<Scal,LInt>> Value()
            {
                return values;
            }
            
            cref<Tensor1<Scal,LInt>> Value() const
            {
                return values;
            }
            
            mref<Scal> Value( const LInt k )
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= values.Size() )
                {
                    eprint(ClassName()+"::Value(): Out of bound access.");
                }
#endif
                return values[k];
            }
            
            cref<Scal> Value( const LInt k ) const
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= values.Size() )
                {
                    eprint(ClassName()+"::Value(): Out of bound access.");
                }
#endif
                return values[k];
            }
            
            
            Scal operator()( const Int i, const Int j ) const
            {
                const Sparse::Position<LInt> pos = this->FindNonzeroPosition(i,j);
                
                return ( pos.found ) ? values[pos.index] : static_cast<Scal>(0);
            }
            
        protected:
            
            // Supply an external list of values.
            template<typename T_ext>
            Tensor2<T_ext,Int> ToTensor2(  ) const
            {
                return this->ToTensor2_( values.data() );
            }
            
            template< bool conjugate>
            MatrixCSR transpose() const
            {
                if( WellFormed() )
                {
                    if constexpr ( conjugate )
                    {
                        ptic(ClassName()+"::ConjugateTranspose");
                    }
                    else
                    {
                        ptic(ClassName()+"::Transpose");
                    }
                    
                    RequireJobPtr();
                    
                    Tensor2<LInt,Int> counters = CreateTransposeCounters();
                    
                    MatrixCSR B ( n, m, outer[m], thread_count );

                    copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );

                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            mptr<Int > B_inner  = B.Inner().data();
                            mptr<Scal> B_values = B.Value().data();
                            cptr<LInt> A_outer  = Outer().data();
                            cptr<Int > A_inner  = Inner().data();
                            cptr<Scal> A_values = Value().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = A_outer[i  ];
                                const LInt k_end   = A_outer[i+1];
                                
                                for( LInt k = k_end; k --> k_begin; )
                                {
                                    const Int j = A_inner[k];
                                    const LInt pos = --c[j];
                                    B_inner [pos] = i;
                                    
                                    if constexpr ( conjugate )
                                    {
                                        B_values[pos] = Conj(A_values[k]);
                                    }
                                    else
                                    {
                                        B_values[pos] = A_values[k];
                                    }
                                }
                            }
                        },
                        thread_count
                    );

                    // Finished counting sort.
                    
                    // We only have to care about the correct ordering of inner indices and values.
                    B.SortInner();

                    if constexpr ( conjugate )
                    {
                        ptoc(ClassName()+"::ConjugateTranspose");
                    }
                    else
                    {
                        ptoc(ClassName()+"::Transpose");
                    }
                    
                    return B;
                }
                else
                {
                    MatrixCSR B ( n, m, 0, thread_count );
                    return B;
                }
            }
            
        public:
            
            MatrixCSR Transpose() const
            {
                return transpose<false>();
            }
            
            MatrixCSR ConjugateTranspose() const
            {
                return transpose<true>();
            }
            
            
            void SortInner() const override
            {   
                if( !inner_sorted )
                {
                    ptic(ClassName()+"::SortInner");

                    if( WellFormed() )
                    {
                        RequireJobPtr();
                        
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                TwoArraySort<Int,Scal,LInt> S;
                                
                                const Int i_begin = job_ptr[thread  ];
                                const Int i_end   = job_ptr[thread+1];
                                
                                for( Int i = i_begin; i < i_end; ++i )
                                {
                                    const LInt begin = outer[i  ];
                                    const LInt end   = outer[i+1];
                                    S( inner.data(begin), values.data(begin), end - begin );
                                }
                            },
                            thread_count
                        );
                        
                        inner_sorted = true;
                    }
                    
                    ptoc(ClassName()+"::SortInner");
                }
            }
            
            
            void Compress() const override
            {
                // Removes duplicate {i,j}-pairs by adding their corresponding nonzero values.
                
                if( !duplicate_free )
                {
                    ptic(ClassName()+"::Compress");
                    
                    if( WellFormed() )
                    {
                        RequireJobPtr();
                        SortInner();
                        
                        Tensor1<LInt,Int> new_outer (outer.Size(),0);
                        
                        cptr<LInt> outer__     = outer.data();
                        mptr<Int > inner__     = inner.data();
                        mptr<Scal> values__    = values.data();
                        mptr<LInt> new_outer__ = new_outer.data();
                        
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                const Int i_begin = job_ptr[thread  ];
                                const Int i_end   = job_ptr[thread+1];
      
                                // To where we write.
                                LInt jj_new        = outer__[i_begin];
                                LInt next_jj_begin = outer__[i_begin];
                                
                                for( Int i = i_begin; i < i_end; ++i )
                                {
                                    const LInt jj_begin = next_jj_begin;
                                    const LInt jj_end   = outer__[i+1];
                                    
                                    // Memorize the next entry in outer because outer will be overwritten
                                    next_jj_begin = jj_end;
                                    
                                    LInt row_nonzero_counter = 0;
                                    
                                    // From where we read.
                                    LInt jj = jj_begin;
                                    
                                    while( jj< jj_end )
                                    {
                                        Int j = inner__ [jj];
                                        Scal a = values__[jj];
                                        
                                        if( jj > jj_new )
                                        {
                                            inner__ [jj] = 0;
                                            values__[jj] = 0;
                                        }
                                        
                                        ++jj;
                                        
                                        while( (jj < jj_end) && (j == inner__[jj]) )
                                        {
                                            a+= values__[jj];
                                            
                                            if( jj > jj_new )
                                            {
                                                inner__ [jj] = 0;
                                                values__[jj] = 0;
                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        inner__ [jj_new] = j;
                                        values__[jj_new] = a;
                                        
                                        ++jj_new;
                                        ++row_nonzero_counter;
                                    }
                                    
                                    new_outer__[i+1] = row_nonzero_counter;
                                }
                            },
                            thread_count
                        );
                        
                        // This is the new array of outer indices.
                        new_outer.Accumulate( thread_count );
                        
                        const LInt nnz = new_outer[m];
                        
//                        Tensor1< Int,LInt> new_inner  (nnz,0);
//                        Tensor1<Scal,LInt> new_values (nnz,0);
                        
                        Tensor1< Int,LInt> new_inner  (nnz);
                        Tensor1<Scal,LInt> new_values (nnz);
                        
                        mptr<Int > new_inner__  = new_inner.data();
                        mptr<Scal> new_values__ = new_values.data();
                        
                        //TODO: Parallelization might be a bad idea here.
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                const  Int i_begin = job_ptr[thread  ];
                                const  Int i_end   = job_ptr[thread+1];
                                
                                const LInt new_pos = new_outer__[i_begin];
                                const LInt     pos =     outer__[i_begin];
                                
                                const LInt thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];
                                
                                // Starting position of thread in inner list.
                                
                                copy_buffer( &inner__[pos],  &new_inner__[new_pos],  thread_nonzeroes );
                                
                                copy_buffer( &values__[pos], &new_values__[new_pos], thread_nonzeroes );
                            },
                            thread_count
                        );
                        
                        swap( new_outer,  outer  );
                        swap( new_inner,  inner  );
                        swap( new_values, values );
                        
                        job_ptr = JobPointers<Int>();
                        job_ptr_initialized = false;
                        duplicate_free = true;
                    }
                    
                    ptoc(ClassName()+"::Compress");
                }
            }
            
            
    //###########################################################################################
    //####          Permute
    //###########################################################################################
            
        public:
            
            MatrixCSR Permute(
                cref<Tensor1<Int,Int>> p,
                cref<Tensor1<Int,Int>> q,
                bool sortQ = true
            )
            {
                if( p.Dimension(0) != m )
                {
                    eprint(ClassName()+"::Permute: Length of first argument does not coincide with RowCount().");
                    return MatrixCSR();
                }
                
                if( q.Dimension(0) != n )
                {
                    eprint(ClassName()+"::Permute: Length of second argument does not coincide with ColCount().");
                    return MatrixCSR();
                }
                
                Permute( p.data(), q.data(), sortQ );
            }
            
            MatrixCSR Permute( cptr<Int> p, cptr<Int> q, bool sortQ = true )
            {
                if( p == nullptr )
                {
                    if( q == nullptr )
                    {
                        // Just make a copy.
                        return MatrixCSR(*this);
                    }
                    else
                    {
                        return PermuteCols(q,sortQ);
                    }
                }
                else
                    if( q == nullptr )
                    {
                        return PermuteRows(p);
                    }
                    else
                    {
                        return PermuteRowsCols(p,q,sortQ);
                    }
            }
            
        protected:
            
            MatrixCSR PermuteRows( cptr<Int> p )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                {
                    cptr<LInt> A_outer = outer.data();
                    mptr<LInt> B_outer = B.Outer().data();
                    
                    B_outer[0] = 0;
                    
                    ParallelDo(
                        [=]( const Int i )
                        {
                            const Int p_i = p[i];
                            
                            B_outer[i+1] = A_outer[p_i+1] - A_outer[p_i];
                        },
                        m,
                        ThreadCount()
                    );
                }
                
                B.Outer().Accumulate();
                
                {
                    auto & B_job_ptr = B.JobPtr();
                    
                    const Int thread_count = B_job_ptr.ThreadCount();
                    
                    cptr<LInt> A_outer  = outer.data();
                    cptr<Int > A_inner  = inner.data();
                    mptr<Scal> A_values = values.data();
                    
                    cptr<LInt> B_outer  = B.Outer().data();
                    mptr<Int > B_inner  = B.Inner().data();
                    mptr<Scal> B_values = B.Values().data();
                    
                    ParallelDo(
                        [=,this,&B]( const Int i )
                        {
                            const Int p_i = p[i];
                            const LInt A_begin = A_outer[p_i  ];
                            const LInt A_end   = A_outer[p_i+1];
                            
                            const LInt B_begin = B_outer[i  ];
                            //                        const LInt B_end   = B_outer[i+1];
                            //                        const LInt B_end   = B_outer[i+1];
                            
                            copy_buffer( &A_inner [A_begin], &A_inner [A_end], &B_inner [B_begin] );
                            copy_buffer( &A_values[A_begin], &A_values[A_end], &B_values[B_begin] );
                            B.inner_sorted = true;
                        },
                        B_job_ptr
                    );
                }
                
                return B;
            }
            
            MatrixCSR PermuteCols( cptr<Int> q, bool sortQ = true )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                Tensor1<Int,Int> q_inv_buffer ( ColCount() );
                
                mptr<Int> q_inv = q_inv_buffer.data();
                
                ParallelDo(
                    [=,this]( const Int j )
                    {
                        q_inv[q[j]] = j;
                    },
                    n,
                    ThreadCount()
                );
                
                copy_buffer( outer.data(), B.Outer().data(), m+1 );
                
                ParallelDo(
                    [=,this,&B]( const Int thread )
                    {
                        TwoArraySort<Int,Scal,LInt> S;
                        
//                        cptr<LInt> A_outer  = outer.data();
                        cptr<Int > A_inner  = inner.data();
                        cptr<Scal> A_values = values.data();
                        
                        cptr<LInt> B_outer  = B.Outer().data();
                        mptr<Int > B_inner  = B.Inner().data();
                        mptr<Scal> B_values = B.Values().data();
                        
                        const Int i_begin = B.JobPtr()[thread  ];
                        const Int i_end   = B.JobPtr()[thread+1];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt B_begin = B_outer[i  ];
                            const LInt B_end   = B_outer[i+1];;
                            
                            for( LInt k = B_begin; k < B_begin; ++k )
                            {
                                B_inner[B_begin] = q_inv[A_inner[B_begin]];
                            }
                            
                            const LInt k_max = B_end - B_begin;
                            
                            copy_buffer( &A_values[B_begin], &B_values[B_begin], k_max );
                            
                            if( sortQ )
                            {
                                S( &B_inner[B_begin], &B_values[B_begin], k_max );
                                B.inner_sorted = true;
                            }
                        }
                    },
                    B.JobPtr().ThreadCount()
                );
                
                return B;
            }
            
            MatrixCSR PermuteRowsCols( cptr<Int> p, cptr<Int> q, bool sortQ = true )
            {
                MatrixCSR B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
                
                Tensor1<Int,Int> q_inv_buffer ( ColCount() );
                mptr<Int> q_inv = q_inv_buffer.data();
                
                ParallelDo(
                    [=,this]( const Int j )
                    {
                        q_inv[q[j]] = j;
                    },
                    n,
                    ThreadCount()
                );
                
                {
                    cptr<LInt> A_outer = outer.data();
                    mptr<LInt> B_outer = B.Outer().data();
                    
                    B_outer[0] = 0;
                    
                    ParallelDo(
                        [=]( const Int i )
                        {
                            const Int p_i = p[i];
                            
                            B_outer[i+1] = A_outer[p_i+1] - A_outer[p_i];
                        },
                        m,
                        ThreadCount()
                    );
                }
                
                B.Outer().Accumulate();
                
                ParallelDo(
                    [=,this,&B]( const Int thread )
                    {
                        cptr<LInt> A_outer  = outer.data();
                        cptr<Int > A_inner  = inner.data();
                        cptr<Scal> A_values = values.data();
                        
                        cptr<LInt> B_outer  = B.Outer().data();
                        mptr<Int > B_inner  = B.Inner().data();
                        mptr<Scal> B_values = B.Values().data();
                        
                        TwoArraySort<Int,Scal,LInt> S;
                        
                        const Int i_begin = B.JobPtr()[thread  ];
                        const Int i_end   = B.JobPtr()[thread+1];
                        
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
                                B_inner[B_begin+k] = q_inv[A_inner[A_begin+k]];
                            }
                            
                            copy_buffer( &A_values[A_begin], &B_values[B_begin], k_max );
                            
                            if( sortQ )
                            {
                                S( &B_inner[B_begin], &B_values[B_begin], k_max );
                                B.inner_sorted = true;
                            }
                        }
                    },
                    B.JobPtr().ThreadCount()
                );
                
                return B;
            }
            
        public:
            
            MatrixCSR Dot( const MatrixCSR & B ) const
            {
                ptic(ClassName()+"::Dot");
                
                if(WellFormed() )
                {
                    RequireJobPtr();
                    
                    Tensor2<LInt,Int> counters ( thread_count, m, LInt(0) );
                    
                    // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                    // https://en.wikipedia.org/wiki/Counting_sort
                    
                    ParallelDo(
                        [=,this,&B,&counters]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            cptr<LInt> A_outer  = Outer().data();
                            cptr<Int>  A_inner  = Inner().data();
                            
                            cptr<LInt> B_outer  = B.Outer().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                Int c_i = 0;
                                
                                const LInt jj_begin = A_outer[i  ];
                                const LInt jj_end   = A_outer[i+1];
                                
                                for( LInt jj = jj_begin; jj < jj_end; ++jj )
                                {
                                    const Int j = A_inner[jj];
                                    
                                    c_i += (B_outer[j+1] - B_outer[j]);
                                }
                                
                                c[i] = c_i;
                            }
                        },
                        thread_count
                    );
                    
                    AccumulateAssemblyCounters_Parallel( counters );
                    
                    const LInt nnz = counters.data(thread_count-1)[m-1];
                    
                    MatrixCSR C ( m, B.ColCount(), nnz, thread_count );
                    
                    copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c        = counters.data(thread);
                            
                            cptr<LInt> A_outer  = Outer().data();
                            cptr< Int> A_inner  = Inner().data();
                            cptr<Scal> A_values = Value().data();
                            
                            cptr<LInt> B_outer  = B.Outer().data();
                            cptr< Int> B_inner  = B.Inner().data();
                            cptr<Scal> B_values = B.Value().data();
                            
                            mptr< Int> C_inner  = C.Inner().data();
                            mptr<Scal> C_values = C.Value().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt jj_begin = A_outer[i  ];
                                const LInt jj_end   = A_outer[i+1];
                                
                                for( LInt jj = jj_begin; jj < jj_end; ++jj )
                                {
                                    const Int j = A_inner[jj];
                                    
                                    const LInt kk_begin = B_outer[j  ];
                                    const LInt kk_end   = B_outer[j+1];
                                    
                                    for( LInt kk = kk_end; kk --> kk_begin; )
                                    {
                                        const Int k = B_inner[kk];
                                        const LInt pos = --c[i];
                                        
                                        C_inner [pos] = k;
                                        C_values[pos] = A_values[jj] * B_values[kk];
                                    }
                                }
                            }
                        },
                        thread_count
                    );
                    
                    // Finished expansion phase (counting sort).
                    
                    // Finally we row-sort inner and compressQ duplicates in inner and values.
                    C.Compress();
                    
                    ptoc(ClassName()+"::Dot");
                    
                    return C;
                }
                else
                {
                    return MatrixCSR ();
                }
            }
            
            Sparse::BinaryMatrixCSR<Int,LInt> DotBinary( const MatrixCSR & B ) const
            {
                Sparse::BinaryMatrixCSR<Int,LInt> result;
                
                Base_T C = this->DotBinary_(B);
                
                swap(result,C);
                
                return result;
            }
            
            
        public:
            
//###########################################################################################
//####          Matrix Multiplication
//###########################################################################################
            
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cptr<T_in>  X, const Int ldX,
                cref<S_out> beta,  mptr<T_out> Y, const Int ldY,
                const Int   nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cptr<T_in>  X,
                cref<S_out> beta,  mptr<T_out> Y,
                const Int   nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            // Use own nonzero values.
            template<typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cref<Tensor1<T_in, Int>> X,
                cref<S_out> beta,  mref<Tensor1<T_out,Int>> Y
            ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m )
                {
                    const Int one = static_cast<Int>(1);
                    
                    this->template Dot_<1>( values.data(), alpha, X.data(), one, beta, Y.data(), one, one );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                 cref<R_out> alpha, cref<Tensor2<T_in, Int>> X,
                 cref<S_out> beta,  mref<Tensor2<T_out,Int>> Y
             ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
                {
                    const Int nrhs = X.Dimension(1);
                    
                    this->template Dot_<NRHS>( values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cptr<T_ext>  ext_values,
                cref<R_out> alpha, cptr<T_ext> X, const Int ldX,
                cref<S_out> beta,  mptr<T_ext> Y, const Int ldY,
                const Int   nrhs = static_cast<Int>(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cptr<T_ext> ext_values,
                cref<R_out> alpha, cptr<T_ext> X,
                cref<S_out> beta,  mptr<T_ext> Y,
                const Int   nrhs = static_cast<Int>(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<Tensor1<T_ext,Int>> ext_values,
                cref<R_out> alpha, cref<Tensor1<T_in, Int>> X,
                cref<S_out> beta,  mref<Tensor1<T_out,Int>> Y
            ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m )
                {
                    const Int one = static_cast<Int>(1);
                    
                    this->template Dot_<1>( ext_values.data(), alpha, X.data(), one, beta, Y.data(), one, one );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = VarSize, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                 cref<Tensor1<T_ext,Int>> ext_values,
                 cref<R_out> alpha, cref<Tensor2<T_in, Int>> X,
                 cref<S_out> beta,  mref<Tensor2<T_out,Int>> Y
         ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
                {
                    const Int nrhs = X.Dimension(1);
                    
                    this->template Dot_<NRHS>( ext_values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
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
                return std::string("Sparse::MatrixCSR<")+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // MatrixCSR
    
        
        template<typename Scal, typename Int, typename LInt>
        MatrixCSR<Scal,Int,LInt> MatrixCSR_FromFile( std::string filename, const Int thread_count )
        {
            std::ifstream s ( filename );
            
            Int m;
            Int n;
            Int nnz;
            
            s >> m;
            s >> n;
            s >> nnz;
            
            MatrixCSR<Scal,Int,LInt> A( m, n, nnz, thread_count );
            
            mptr<LInt> rp = A.Outer().data();
            for( Int i = 0; i < n+1; ++i )
            {
                s >> rp[i];
            }
            
            mptr<Int> ci = A.Inner().data();
            for( Int i = 0; i < nnz; ++i )
            {
                s >> ci[i];
            }
            
            mptr<Scal> a = A.Values().data();
            for( Int i = 0; i < nnz; ++i )
            {
                s >> a[i];
            }
            
            return A;
        }
        
    } // namespace Sparse
    
} // namespace Tensors
