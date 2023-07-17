#pragma once

namespace Tensors
{
    namespace Sparse
    {
        
        template<typename LInt>
        struct Position
        {
            const LInt index;
            const bool found;
        };
        
        template<typename Int, typename LInt> class BinaryMatrixCSR;
        
        template<typename Int_, typename LInt_>
        class PatternCSR
        {
            ASSERT_INT (Int_);
            ASSERT_INT (LInt_);
            
            // Int  - an integer type capable of storing both the number of rows and columns
            // LInt - a potentially longer integer type capable of storing the absolute number of nonzeros.
            
        public:
            
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            // I have to make these mutable so that methods that depend on SortInner can be called also from const instances of the class.
            mutable Tensor1<LInt, Int> outer;
            mutable Tensor1< Int,LInt> inner;
            
            // I cannot make these const since swap would not work otherwise.
            Int m = 0;
            Int n = 0;
            
            Int thread_count = 1;
            
            mutable bool inner_sorted         = false;
            mutable bool duplicate_free       = false;
            
            mutable bool diag_ptr_initialized = false;
            mutable bool job_ptr_initialized  = false;
            mutable bool upper_triangular_job_ptr_initialized  = false;
            mutable bool lower_triangular_job_ptr_initialized  = false;
            
            bool symmetric                    = false;
            
            
            // diag_ptr[i] is the first nonzero element in row i such that inner[diag_ptr[i]] >= i
            mutable Tensor1<LInt,Int> diag_ptr;
            mutable JobPointers<Int> job_ptr;
            mutable JobPointers<Int> upper_triangular_job_ptr;
            mutable JobPointers<Int> lower_triangular_job_ptr;
            
        public:
            
            friend class BinaryMatrixCSR<Int,LInt>;
            
            PatternCSR() {}
            
            template<typename I_0, typename I_1, typename I_3>
            PatternCSR(
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer       ( static_cast<Int>(m_+1),static_cast<LInt>(0) )
            ,   m           ( static_cast<Int>(m_)                        )
            ,   n           ( static_cast<Int>(n_)                        )
            ,   thread_count( static_cast<Int>(thread_count_)             )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
                Init();
            }
            
            template<typename I_0, typename I_1, typename I_2, typename I_3>
            PatternCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   outer       ( static_cast<Int>(m_+1), static_cast<LInt>(0) )
            ,   inner       ( static_cast<LInt>(nnz_)                      )
            ,   m           ( static_cast<Int>(m_)                         )
            ,   n           ( static_cast<Int>(n_)                         )
            ,   thread_count( static_cast<Int>(thread_count_)              )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_2);
                ASSERT_INT(I_3);
                Init();
            }
            
            template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            PatternCSR(
                ptr<J_0>  outer_,
                ptr<J_1>  inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer       ( m_+1                            )
            ,   inner       ( static_cast<LInt>(outer_[m_])   )
            ,   m           ( static_cast<Int>(m_)            )
            ,   n           ( static_cast<Int>(n_)            )
            ,   thread_count( static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(J_0);
                ASSERT_INT(J_1);
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
                
                outer.Read(outer_);
                inner.Read(inner_);
            }
            
            template<typename I_0, typename I_1, typename I_3>
            PatternCSR(
                const Tensor1<LInt, Int> & outer_,
                const Tensor1< Int,LInt> & inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer       ( outer_                          )
            ,   inner       ( inner_                          )
            ,   m           ( static_cast<Int>(m_)            )
            ,   n           ( static_cast<Int>(n_)            )
            ,   thread_count( static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            template<typename I_0, typename I_1, typename I_3>
            PatternCSR(
                Tensor1<LInt, Int> && outer_,
                Tensor1< Int,LInt> && inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer       ( std::move(outer_)               )
            ,   inner       ( std::move(inner_)               )
            ,   m           ( static_cast<Int>(m_)            )
            ,   n           ( static_cast<Int>(n_)            )
            ,   thread_count( static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            // Copy constructor
            PatternCSR( const PatternCSR & other )
            :   outer           ( other.outer           )
            ,   inner           ( other.inner           )
            ,   m               ( other.m               )
            ,   n               ( other.n               )
            ,   thread_count    ( other.thread_count    )
            ,   inner_sorted    ( other.inner_sorted    )
            ,   duplicate_free  ( other.duplicate_free  )
            ,   symmetric       ( other.symmetric       )
            {
                logprint("Copy of "+ClassName()+" of size {"+ToString(other.m)+", "+ToString(other.n)+"}, nn z = "+ToString(other.NonzeroCount()));
            }
            
            friend void swap (PatternCSR &A, PatternCSR &B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( A.outer,          B.outer          );
                swap( A.inner,          B.inner          );
                swap( A.m,              B.m              );
                swap( A.n,              B.n              );
                swap( A.thread_count,   B.thread_count   );
                swap( A.inner_sorted,   B.inner_sorted   );
                swap( A.duplicate_free, B.duplicate_free );
                swap( A.symmetric,      B.symmetric      );
            }
            
            //(Copy-)assignment operator
            PatternCSR & operator=( PatternCSR other ) // Passing by value is okay, because of copy elision.
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                
                swap(*this, other);
                
                return *this;
            }
            
            // Move constructor
            PatternCSR( PatternCSR && other ) noexcept : PatternCSR()
            {
                swap(*this, other);
            }
            
            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            PatternCSR(
                const Int    * const * const idx,
                const Int    * const * const jdx,
                const LInt   *         const entry_counts,
                const Int list_count,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, list_count )
            {
                FromPairs( idx, jdx, entry_counts, list_count, final_thread_count, compressQ, symmetrize );
            }
            
            PatternCSR(
                const std::vector<Int> & idx,
                const std::vector<Int> & jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(1) )
            {
                const Int * i = idx.data();
                const Int * j = jdx.data();
                
                Tensor1<LInt,Int> entry_counts (1, static_cast<LInt>(idx.size()));
                
                FromPairs( &i, &j, entry_counts.data(), 1, final_thread_count, compressQ, symmetrize );
            }
            
            PatternCSR(
                const std::vector<std::vector<Int>> & idx,
                const std::vector<std::vector<Int>> & jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(idx.size()) )
            {
                Int list_count = static_cast<Int>(idx.size());
                Tensor1<const Int*,Int> i (list_count);
                Tensor1<const Int*,Int> j (list_count);
                
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    entry_counts[thread] = static_cast<LInt>(idx[thread].size());
                }
                
                FromPairs( i.data(), j.data(), entry_counts.data(), list_count, final_thread_count, compressQ, symmetrize );
            }
            
            PatternCSR(
                const std::vector<PairAggregator<Int,Int,LInt>> & idx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(idx.size()) )
            {
                Int list_count = static_cast<Int>(idx.size());
                Tensor1<const Int*,Int> i (list_count);
                Tensor1<const Int*,Int> j (list_count);
                
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].Get_0().data();
                    j[thread] = idx[thread].Get_1().data();
                    
                    entry_counts[thread] = static_cast<LInt>(idx[thread].Size());
                }
                
                FromPairs( i.data(), j.data(), entry_counts.data(), list_count, final_thread_count, compressQ, symmetrize );
            }
            
            virtual ~PatternCSR() = default;
            
            
        public:
            
            Int ThreadCount() const
            {
                return thread_count;
            }
            
            void SetThreadCount( const Int thread_count_ )
            {
                thread_count = std::max( static_cast<Int>(1), thread_count_);
                
                job_ptr_initialized = false;
            }
            
        protected:
            
            void Init()
            {
                outer[0] = static_cast<LInt>(0);
            }
            
            void FromPairs(
                const  Int * const * const idx,
                const  Int * const * const jdx,
                const LInt * entry_counts,
                const  Int list_count,
                const  Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            {
                ptic(ClassName()+"::FromPairs");
                
                Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int>( idx, jdx, entry_counts, list_count, m, symmetrize );
                
                const LInt nnz = counters(list_count-1,m-1);
                
                if( nnz > 0 )
                {
                    inner = Tensor1<Int,LInt>( nnz );
                    
                    mut<LInt> outer__ = outer.data();
                    mut<Int>  inner__ = inner.data();
                    
                    copy_buffer<VarSize,Sequential>( counters.data(list_count-1), &outer__[1], m );
                    
                    // writing the j-indices into sep_column_indices
                    // the counters array tells each thread where to write
                    // since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                    
                    if( symmetrize != 0 )
                    {
                        ParallelDo(
                            [&]( const Int thread )
                            {
                                const LInt entry_count = entry_counts[thread];
                                
                                ptr<Int> thread_idx = idx[thread];
                                ptr<Int> thread_jdx = jdx[thread];
                                
                                mut<LInt> c = counters.data(thread);
                                
                                for( LInt k = entry_count; k --> 0; )
                                {
                                    const Int i = thread_idx[k];
                                    const Int j = thread_jdx[k];
                                    {
                                        const LInt pos = --c[i];
                                        inner__[pos] = j;
                                    }
                                    
                                    c[j] -= static_cast<LInt>(i != j);
                                    
                                    const LInt pos  = c[j];
                                    
                                    inner__[pos] = i;
                                }
                            },
                            list_count
                        );
                    }
                    else
                    {
                        ParallelDo(
                            [&]( const Int thread )
                            {
                                const LInt entry_count = entry_counts[thread];
                                
                                ptr<Int> thread_idx = idx[thread];
                                ptr<Int> thread_jdx = jdx[thread];
                                
                                mut<LInt> c = counters.data(thread);
                                
                                for( LInt k = entry_count; k --> 0; )
                                {
                                    const Int i = thread_idx[k];
                                    const Int j = thread_jdx[k];
                                    {
                                        const LInt pos = --c[i];
                                        inner__[pos] = j;
                                    }
                                }
                            },
                            list_count
                        );
                    }
                    
                    // From here on, we may use as many threads as we want.
                    SetThreadCount( final_thread_count );
                    
                    
                    // We have to sort b_inner to be compatible with the CSR format.
                    SortInner();
                    
                    if( compressQ )
                    {
                        Compress();
                    }
                    else
                    {
                        duplicate_free = true; // We have to rely on the caller here...
                    }
                    
                }
                else
                {
                    SetThreadCount( final_thread_count );
                }
                ptoc(ClassName()+"::FromPairs");
            }
            
            void RequireJobPtr() const
            {
                if( !job_ptr_initialized )
                {
                    ptic(ClassName()+"::RequireJobPtr");
                    
//                // TODO: Find better cost model.
//                Tensor1<Int,Int> costs ( outer.data(), m+1 );
//
//                for( Int i = 0; i < m ; ++i )
//                {
//                    costs[i+1] += i;
//                }
//
                    
                    job_ptr = JobPointers<Int>( m, outer.data(), thread_count, false );
                    
                    job_ptr_initialized = true;
                    
                    ptoc(ClassName()+"::RequireJobPtr");
                }
            }
            
            void CheckOrdering() const
            {
                if( !inner_sorted )
                {
                    eprint(ClassName()+"::RequireDiag: Column indices might not be sorted appropriately. Better call SortInner() first.");
                }
            }
            
            void RequireDiag() const
            {
                if( !diag_ptr_initialized )
                {
                    ptic(ClassName()+"::RequireDiag");
                    
                    SortInner();
                    
                    if( outer.Last() <= 0 )
                    {
                        diag_ptr = Tensor1<LInt,Int>( outer.data()+1, m );
                    }
                    else
                    {
                        RequireJobPtr();
                        
                        diag_ptr = Tensor1<LInt,Int>( m );
                        
                        mut<LInt> diag_ptr__ = diag_ptr.data();
                        ptr<LInt> outer__    = outer.data();
                        ptr<Int>  inner__    = inner.data();
                        
                        ParallelDo(
                            [=]( const Int i )
                            {
                                const LInt k_begin = outer__[i  ];
                                const LInt k_end   = outer__[i+1];
                                
                                LInt k = k_begin;
                                
                                while( (k < k_end) && (inner__[k] < i)  )
                                {
                                    ++k;
                                }
                                diag_ptr__[i] = k;
                            },
                            job_ptr
                        );
                    }
                    
                    diag_ptr_initialized = true;
                    
                    ptoc(ClassName()+"::RequireDiag");
                }
            }
            
            void RequireUpperTriangularJobPtr() const
            {
                if( (m > 0) && !upper_triangular_job_ptr_initialized )
                {
                    ptic(ClassName()+"::RequireUpperTriangularJobPtr");
                    
                    RequireDiag();
                    
                    Tensor1<LInt,Int> costs (m + 1);
                    costs[0]=0;
                    
                    ptr<LInt> diag_ptr__ = diag_ptr.data();
                    ptr<LInt> outer__    = outer.data();
                    mut<LInt> costs__    = costs.data();
                    
                    ParallelDo(
                        [=]( const Int i )
                        {
                            costs__[i+1] = outer__[i+1] - diag_ptr__[i];
                        },
                        job_ptr
                    );
                    
                    costs.Accumulate( thread_count );
                    
                    upper_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                    
                    upper_triangular_job_ptr_initialized = true;
                    
                    ptoc(ClassName()+"::RequireUpperTriangularJobPtr");
                }
            }
            
            void RequireLowerTriangularJobPtr() const
            {
                if( (m > 0) && !lower_triangular_job_ptr_initialized )
                {
                    ptic(ClassName()+"::RequireLowerTriangularJobPtr");
                    
                    RequireDiag();
                    
                    Tensor1<LInt,Int> costs (m + 1);
                    costs[0]=0;
                    
                    ptr<LInt> diag_ptr__ = diag_ptr.data();
                    ptr<LInt> outer__    = outer.data();
                    mut<LInt> costs__    = costs.data();
                    
                    ParallelDo(
                        [=]( const Int i )
                        {
                            costs__[i+1] = diag_ptr__[i] - outer__[i];
                        },
                        job_ptr
                    );
                    
                    costs.Accumulate( thread_count );
                    
                    lower_triangular_job_ptr = JobPointers( m, costs.data(), thread_count, false );
                    
                    lower_triangular_job_ptr_initialized = true;
                    
                    ptoc(ClassName()+"::RequireLowerTriangularJobPtr");
                }
            }
            
        public:
            
            Int RowCount() const
            {
                return m;
            }
            
            Int ColCount() const
            {
                return n;
            }
            
            LInt NonzeroCount() const
            {
                return inner.Size();
            }
            
            LInt NonzeroCount( const Int i ) const
            {
                return outer(i+1) - outer(i);
            }
            
            Tensor1<LInt,Int> & Outer()
            {
                return outer;
            }
            
            const Tensor1<LInt,Int> & Outer() const
            {
                return outer;
            }
            

            LInt & Outer( const Int i )
            {
#ifndef TOOLS_DEBUG
                if( i < 0 || i >= outer.Size() )
                {
                    eprint(ClassName()+"::Outer(): Out of bound access.");
                }
#endif
                return outer[i];
            }
            
            const LInt & Outer( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                if( i < 0 || i >= outer.Size() )
                {
                    eprint(ClassName()+"::Outer(): Out of bound access.");
                }
#endif
                return outer[i];
            }
            
            
            Tensor1<Int,LInt> & Inner()
            {
                return inner;
            }
            
            const Tensor1<Int,LInt> & Inner() const
            {
                return inner;
            }

            Int & Inner( const LInt k )
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= inner.Size() )
                {
                    eprint(ClassName()+"::Inner(): Out of bound access.");
                }
#endif
                return inner[k];
            }
            
            const Int & Inner( const LInt k ) const
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= inner.Size() )
                {
                    eprint(ClassName()+"::Inner(): Out of bound access.");
                }
#endif
                return inner[k];
            }
            
            const Tensor1<LInt,Int> & Diag() const
            {
                RequireDiag();
                
                return diag_ptr;
            }
            
            
            const LInt & Diag( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                if( i < 0 || i >= diag_ptr.Size() )
                {
                    eprint(ClassName()+"::Diag(): Out of bound access.");
                }
#endif
                return diag_ptr[i];
            }
            
            const JobPointers<Int> & JobPtr() const
            {
                RequireJobPtr();
                
                return job_ptr;
            }
            
            
            
            const JobPointers<Int> & UpperTriangularJobPtr() const
            {
                RequireUpperTriangularJobPtr();
                
                return upper_triangular_job_ptr;
            }
            
            
            const JobPointers<Int> & LowerTriangularJobPtr() const
            {
                RequireLowerTriangularJobPtr();
                
                return lower_triangular_job_ptr;
            }
            
            
        protected:
            
            Tensor2<LInt,Int> CreateTransposeCounters() const
            {
                ptic(ClassName()+"::CreateTransposeCounters");
                
                RequireJobPtr();
                
                Tensor2<LInt,Int> counters ( thread_count, n, static_cast<LInt>(0) );
                
                if( WellFormed() )
                {
                    //            ptic("Counting sort");
                    // Use counting sort to sort outer indices of output matrix.
                    // https://en.wikipedia.org/wiki/Counting_sort
                    //            ptic("Counting");
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mut<LInt> c = counters.data(thread);
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt jj_begin = outer[i  ];
                                const LInt jj_end   = outer[i+1];
                                
                                for( LInt jj = jj_begin; jj < jj_end; ++jj )
                                {
                                    const Int j = inner[jj];
                                    ++c[j];
                                }
                            }
                        },
                        thread_count
                    );
                    
                    AccumulateAssemblyCounters_Parallel<LInt,Int>( counters );
                }
                
                ptoc(ClassName()+"::CreateTransposeCounters");
                
                return counters;
            }
            
        public:
            
            virtual void SortInner() const
            {
                // Sorts the column indices of each matrix row.
                
                if( !inner_sorted )
                {
                    ptic(ClassName()+"::SortInner");
                    
                    if( WellFormed() )
                    {
                        ParallelDo(
                            [this]( const Int i )
                            {
                                Sort( &inner[outer[i]], &inner[outer[i+1]], std::less<LInt>() );
                            },
                            JobPtr()
                        );
                        
                        inner_sorted = true;
                    }
                    
                    ptoc(ClassName()+"::SortInner");
                    
                }
            }
            
        public:
            
            
            virtual void Compress() const
            {
                if( !duplicate_free )
                {
                    ptic(ClassName()+"::Compress");
                    
                    if( WellFormed() )
                    {
                        RequireJobPtr();
                        SortInner();
                        
                        Tensor1<LInt,Int> new_outer ( outer.Size(), 0 );
                        
                        mut<LInt> new_outer__ = new_outer.data();
                        mut<LInt>     outer__ = outer.data();
                        mut<Int>      inner__ = inner.data();
                        
                        ParallelDo(
                            [=]( const Int thread )
                            {
                                const Int i_begin = job_ptr[thread  ];
                                const Int i_end   = job_ptr[thread+1];
                                
                                // To where we write.
                                LInt jj_new        = outer__[i_begin];
                                
                                // Memoize the next entry in outer because outer will be overwritten
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
                                        Int j = inner__[jj];
                                        
                                        {
                                            if( jj > jj_new )
                                            {
                                                inner__[jj] = static_cast<Int>(0);
                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        while( (jj < jj_end) && (j == inner__[jj]) )
                                        {
                                            if( jj > jj_new )
                                            {
                                                inner__[jj] = static_cast<Int>(0);
                                            }
                                            ++jj;
                                        }
                                        
                                        inner__[jj_new] = j;
                                        
                                        ++jj_new;
                                        ++row_nonzero_counter;
                                    }
                                    new_outer__[i+1] = row_nonzero_counter;
                                }
                            },
                            thread_count
                        );
                        
                        // This is the new array of outer indices.
                        new_outer.Accumulate( thread_count  );
                        
                        const LInt nnz = new_outer[m];
                        
                        Tensor1<Int,LInt> new_inner (nnz, 0);
                        
                        //TODO: Parallelization might be a bad idea here.
                        
                        ParallelDo(
                            [&,this]( const Int thread )
                            {
                                const  Int i_begin = job_ptr[thread  ];
                                const  Int i_end   = job_ptr[thread+1];
                                
                                const LInt new_pos = new_outer__[i_begin];
                                const LInt     pos =     outer__[i_begin];
                                
                                const LInt thread_nonzeroes = new_outer__[i_end] - new_outer__[i_begin];
                                
                                copy_buffer<VarSize,Sequential>( &inner.data()[pos], &new_inner.data()[new_pos], thread_nonzeroes );
                            },
                            thread_count
                        );
                        
                        swap( new_outer, outer  );
                        swap( new_inner, inner  );
                        
                        job_ptr = JobPointers<Int>();
                        job_ptr_initialized = false;
                        duplicate_free = true;
                    }
                    
                    ptoc(ClassName()+"::Compress");
                }
            }
            
            //##############################################################################################
            //####          Matrix Multiplication
            //##############################################################################################
            
        protected:
            
            PatternCSR DotBinary_( const PatternCSR & B ) const
            {
                ptic(ClassName()+"::DotBinary_");
                
                if( WellFormed() && B.WellFormed() )
                {
                    RequireJobPtr();
                    
                    ptic("Create counters for counting sort");
                    
                    Tensor2<LInt,Int> counters ( thread_count, m, static_cast<Int>(0) );
                    
                    // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                    // https://en.wikipedia.org/wiki/Counting_sort
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mut<LInt> c       = counters.data(thread);
                            
                            ptr<LInt> A_outer = Outer().data();
                            ptr<Int>  A_inner = Inner().data();
                            
                            ptr<LInt> B_outer = B.Outer().data();
                            
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
                        },
                        thread_count
                    );
                    
                    ptoc("Create counters for counting sort");
                    
                    AccumulateAssemblyCounters_Parallel<LInt,Int>(counters);
                    
                    const LInt nnz = counters.data(thread_count-1)[m-1];
                    
                    PatternCSR C ( m, B.ColCount(), nnz, thread_count );
                    
                    copy_buffer<VarSize,Sequential>( counters.data(thread_count-1), &C.Outer().data()[1], m );
                    
                    ptic("Counting sort");
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mut<LInt> c        = counters.data(thread);
                            
                            ptr<LInt> A_outer  = Outer().data();
                            ptr< Int> A_inner  = Inner().data();
                            
                            ptr<LInt> B_outer  = B.Outer().data();
                            ptr< Int> B_inner  = B.Inner().data();
                            
                            mut< Int> C_inner  = C.Inner().data();
                            
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
//                                        C_values[pos] = A_values[jj] * B_values[kk];
                                    }
                                }
                            }
                        },
                        thread_count
                    );
                    
                    // Finished expansion phase (counting sort).
                    ptoc("Counting sort");
                    
                    // Finally we rwo-sort inner and remove duplicates in inner and values.
                    C.Compress();
                    
                    ptoc(ClassName()+"::DotBinary_");
                    
                    return C;
                }
                else
                {
                    return PatternCSR ();
                }
            }
            
        protected:
            
            // Assume all nonzeros are equal to 1.
            template<Int NRHS = VarSize, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot_(
                const R_out alpha, ptr<T_in>  X, const Int ldX,
                const S_out beta,  mut<T_out> Y, const Int ldY,
                const Int nrhs = static_cast<Int>(1)
            ) const
            {
                if( WellFormed() )
                {
                    SparseBLAS<T_out,Int,LInt> sblas ( thread_count );
                    
                    sblas.template Multiply_DenseMatrix<NRHS>(
                        outer.data(),inner.data(),nullptr,m,n,alpha,X,ldX,beta,Y,ldY,nrhs,JobPtr()
                    );
                }
                else
                {
                    wprint(ClassName()+"::Dot_: Not WellFormed(). Doing nothing.");
                }
            }
            
            // Supply an external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot_(
                      ptr<T_ext>  values,
                      const R_out alpha, ptr<T_in>  X, const Int ldX,
                      const S_out beta,  mut<T_out> Y, const Int ldY,
                      const Int   nrhs = static_cast<Int>(1)
            ) const
            {
                if( WellFormed() )
                {
                    auto sblas = SparseBLAS<T_ext,Int,LInt>( thread_count );
                    
                    sblas.template Multiply_DenseMatrix<NRHS>(
                        outer.data(),inner.data(),values,m,n,alpha,X,ldX,beta,Y,ldY,nrhs,JobPtr()
                    );
                }
                else
                {
                    wprint(ClassName()+"::Dot_: Not WellFormed(). Doing nothing.");
                }
            }
            
//##########################################################################################
//####          Lookup Operations
//##########################################################################################
            
            
        private:
            
            void BoundCheck( const Int i, const Int j ) const
            {
                if( (i < 0) || (i > m) )
                {
                    eprint(ClassName()+": Row index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(m) +" [.");
                }
                if( (j < 0) || (j > n) )
                {
                    eprint(ClassName()+": Column index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(n) +" [.");
                }
            }
            
        public:
            
            bool InnerSorted() const
            {
                return inner_sorted;
            }
            
            void AssumeInnerSorted()
            {
                inner_sorted = true;
            }
            
            void AssumeInnerUnsorted()
            {
                inner_sorted = false;
            }
            
            Sparse::Position<LInt> FindNonzeroPosition( const Int i, const Int j ) const
            {
                // Looks up the entry {i,j}. If existent, its index within the list of nonzeroes is returned. Otherwise, a negative number is returned (-1 if simply not found and -2 if i is out of bounds).
                
#ifdef TOOLS_DEBUG
                BoundCheck(i,j);
#endif
                
                if( (0 <= i) && (i < m) )
                {
                    ptr<Int> inner__ = inner.data();
                    
                    LInt L = outer[i  ];
                    LInt R = outer[i+1]-1;
                    
                    if( inner_sorted )
                    {
                        while( L < R )
                        {
                            const LInt k = R - (R-L)/static_cast<Int>(2);
                            const Int col = inner__[k];
                            
                            if( col > j )
                            {
                                R = k-1;
                            }
                            else
                            {
                                L = k;
                            }
                        }
                    }
                    else
                    {
                        while( L < R && inner__[L] < j)
                        {
                            ++L;
                        }
                        
                    }
                    return (inner__[L]==j) ? Sparse::Position<LInt> {L, true} : Sparse::Position<LInt>{0, false};
                }
                else
                {
                    wprint(ClassName()+"::FindNonzeroPosition: Row index i = "+ToString(i)+" is out of bounds {0,"+ToString(m)+"}.");
                    return Sparse::Position<LInt>{0, false};
                }
            }
            
            
            template<typename S, typename T, typename J>
            void FillLowerTriangleFromUpperTriangle( std::map<S,Tensor1<T,J>> & values )
            {
                ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
                
                if( WellFormed() )
                {
                    if( !inner_sorted )
                    {
                        SortInner();
                    }
                    
                    ptr<LInt> diag__  = Diag().data();
                    ptr<LInt> outer__ = Outer().data();
                    ptr<Int>  inner__ = Inner().data();
                    
                    ParallelDo(
                        [=]( const Int i )
                        {
                            const LInt k_begin = outer__[i];
                            const LInt k_end   =  diag__[i];
                            
                            for( LInt k = k_begin; k < k_end; ++k )
                            {
                                const Int j = inner__[k];
                                
                                LInt L =  diag__[j];
                                LInt R = outer__[j+1]-1;
                                
                                while( L < R )
                                {
                                    const LInt M   = R - (R-L)/static_cast<Int>(2);
                                    const  Int col = inner__[M];
                                    
                                    if( col > i )
                                    {
                                        R = M-1;
                                    }
                                    else
                                    {
                                        L = M;
                                    }
                                }
                                
                                for( auto & f : values )
                                {
                                    f.second[k] = f.second[L];
                                }
                                
                            } // for( Int k = k_begin; k < k_end; ++k )
                        },
                        LowerTriangularJobPtr()
                    );
                    
                }
                
                ptoc(ClassName()+"::FillLowerTriangleFromUpperTriangle");
            }
            
            bool WellFormed() const
            {
                return ( ( outer.Size() > 1 ) && ( outer.Last() > 0 ) );
            }
            
        public:
            
            virtual Int Dimension( const bool dim )
            {
                return dim ? n : m;
            }
            
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
                << "\n==== "+ClassName()+" Stats ====\n" << std::endl;
                
                return s.str();
            }
            
            static std::string ClassName()
            {
                return std::string("PatternCSR<")+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // PatternCSR
        
    } // namespace Sparse
    
} // namespace Tensors
