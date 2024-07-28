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
            static_assert(IntQ<Int_>,"");
            static_assert(IntQ<LInt_>,"");
            
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
            
            PatternCSR() = default;
            
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
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
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
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_2>,"");
                static_assert(IntQ<I_3>,"");
                Init();
            }
            
            template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            PatternCSR(
                cptr<J_0> outer_,
                cptr<J_1> inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   outer       ( m_+1                         )
            ,   inner       ( int_cast<LInt>(outer_[m_])   )
            ,   m           ( int_cast<Int>(m_)            )
            ,   n           ( int_cast<Int>(n_)            )
            ,   thread_count( int_cast<Int>(thread_count_) )
            {
                static_assert(IntQ<J_0>,"");
                static_assert(IntQ<J_1>,"");
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
                
                outer.Read(outer_);
                inner.Read(inner_);
            }
            
            template<typename I_0, typename I_1, typename I_3>
            PatternCSR(
                cref<Tensor1<LInt, Int>> outer_,
                cref<Tensor1< Int,LInt>> inner_,
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
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
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
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
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
            
            template<typename ExtInt>
            PatternCSR(
                const ExtInt * const * const idx,
                const ExtInt * const * const jdx,
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
            
            template<typename ExtInt>
            PatternCSR(
                const LInt nnz_,
                const ExtInt  * const i,
                const ExtInt  * const j,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, thread_count )
            {
                Tensor1<const ExtInt *,Int> idx    (thread_count);
                Tensor1<const ExtInt *,Int> jdx    (thread_count);
                Tensor1<      LInt     ,Int> counts (thread_count);
                
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const LInt begin = JobPointer<LInt>(nnz_, thread_count, thread    );
                    const LInt end   = JobPointer<LInt>(nnz_, thread_count, thread + 1);
                    
                    idx[thread] = &i[begin];
                    jdx[thread] = &j[begin];
                    counts[thread] = end-begin;
                }
                
                FromPairs( idx, jdx, counts.data(), thread_count, thread_count, compressQ, symmetrize );
            }
            
            template<typename ExtInt>
            PatternCSR(
                cref<std::vector<ExtInt>> idx,
                cref<std::vector<ExtInt>> jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(1) )
            {
                const ExtInt * i = idx.data();
                const ExtInt * j = jdx.data();
                
                Tensor1<LInt,Int> entry_counts (1, static_cast<LInt>(idx.size()));
                
                FromPairs( &i, &j, entry_counts.data(), 1, final_thread_count, compressQ, symmetrize );
            }
            
            template<typename ExtInt>
            PatternCSR(
                cref<std::vector<std::vector<ExtInt>>> idx,
                cref<std::vector<std::vector<ExtInt>>> jdx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(idx.size()) )
            {
                Int list_count = static_cast<Int>(idx.size());
                Tensor1<const ExtInt *,Int> i (list_count);
                Tensor1<const ExtInt *,Int> j (list_count);
                
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    entry_counts[thread] = static_cast<LInt>(idx[thread].size());
                }
                
                FromPairs( i.data(), j.data(), entry_counts.data(), list_count, final_thread_count, compressQ, symmetrize );
            }
            
            template<typename ExtInt>
            PatternCSR(
                cref<std::vector<PairAggregator<ExtInt,ExtInt,LInt>>> idx,
                const Int m_,
                const Int n_,
                const Int final_thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   PatternCSR ( m_, n_, static_cast<Int>(idx.size()) )
            {
                Int list_count = static_cast<Int>(idx.size());
                Tensor1<const ExtInt *,Int> i (list_count);
                Tensor1<const ExtInt *,Int> j (list_count);
                
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
                thread_count = Ramp_1(thread_count_);
                
                job_ptr_initialized = false;
            }
            
        protected:
            
            void Init()
            {
                outer[0] = static_cast<LInt>(0);
            }
            
            template<typename ExtInt>
            void FromPairs(
                const  ExtInt * const * const idx,
                const  ExtInt * const * const jdx,
                const LInt             *       entry_counts,
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
                    
                    mptr<LInt> A_outer = outer.data();
                    mptr<Int>  A_inner = inner.data();
                    
                    copy_buffer( counters.data(list_count-1), &A_outer[1], m );
                    
                    // writing the j-indices into sep_column_indices
                    // the counters array tells each thread where to write
                    // since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                    
                    if( symmetrize != 0 )
                    {
                        ParallelDo(
                            [&]( const Int thread )
                            {
                                const LInt entry_count = entry_counts[thread];
                                
                                cptr<ExtInt> thread_idx = idx[thread];
                                cptr<ExtInt> thread_jdx = jdx[thread];
                                
                                mptr<LInt> c = counters.data(thread);
                                
                                for( LInt k = entry_count; k --> 0; )
                                {
                                    const Int i = static_cast<Int>(thread_idx[k]);
                                    const Int j = static_cast<Int>(thread_jdx[k]);
                                    {
                                        const LInt pos = --c[i];
                                        A_inner[pos] = j;
                                    }
                                    
                                    c[j] -= static_cast<LInt>(i != j);
                                    
                                    const LInt pos  = c[j];
                                    
                                    A_inner[pos] = i;
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
                                
                                cptr<Int> thread_idx = idx[thread];
                                cptr<Int> thread_jdx = jdx[thread];
                                
                                mptr<LInt> c = counters.data(thread);
                                
                                for( LInt k = entry_count; k --> 0; )
                                {
                                    const Int i = thread_idx[k];
                                    const Int j = thread_jdx[k];
                                    {
                                        const LInt pos = --c[i];
                                        A_inner[pos] = j;
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
                        
                        ParallelDo(
                            [=,this]( const Int i )
                            {
                                const LInt k_begin = outer[i  ];
                                const LInt k_end   = outer[i+1];
                                
                                LInt k = k_begin;
                                
                                while( (k < k_end) && (inner[k] < i)  )
                                {
                                    ++k;
                                }
                                diag_ptr[i] = k;
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
                    
                    ParallelDo(
                        [this,&costs]( const Int i )
                        {
                            costs[i+1] = outer[i+1] - diag_ptr[i];
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
                    
                    ParallelDo(
                        [this,&costs]( const Int i )
                        {
                            costs[i+1] = diag_ptr[i] - outer[i];
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
            
            mref<Tensor1<LInt,Int>> Outer()
            {
                return outer;
            }
            
            cref<Tensor1<LInt,Int>> Outer() const
            {
                return outer;
            }
            

            mref<LInt> Outer( const Int i )
            {
#ifndef TOOLS_DEBUG
                if( i < 0 || i >= outer.Size() )
                {
                    eprint(this->ClassName()+"::Outer(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return outer[i];
            }
            
            cref<LInt> Outer( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                if( i < 0 || i >= outer.Size() )
                {
                    eprint(this->ClassName()+"::Outer(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return outer[i];
            }
            
            
            mref<Tensor1<Int,LInt>> Inner()
            {
                return inner;
            }
            
            cref<Tensor1<Int,LInt>> Inner() const
            {
                return inner;
            }

            mref<Int> Inner( const LInt k )
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= inner.Size() )
                {
                    eprint(this->ClassName()+"::Inner(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return inner[k];
            }
            
            cref<Int> Inner( const LInt k ) const
            {
#ifdef TOOLS_DEBUG
                if( k < 0 || k >= inner.Size() )
                {
                    eprint(this->ClassName()+"::Inner(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return inner[k];
            }
            
            cref<Tensor1<LInt,Int>> Diag() const
            {
                RequireDiag();
                
                return diag_ptr;
            }
            
            
            cref<LInt> Diag( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                if( i < 0 || i >= diag_ptr.Size() )
                {
                    eprint(this->ClassName()+"::Diag(" + ToString(i) + "): Access out of bounds.");
                }
#endif
                return diag_ptr[i];
            }
            
            cref<JobPointers<Int>> JobPtr() const
            {
                RequireJobPtr();
                
                return job_ptr;
            }
            
            
            
            cref<JobPointers<Int>> UpperTriangularJobPtr() const
            {
                RequireUpperTriangularJobPtr();
                
                return upper_triangular_job_ptr;
            }
            
            
            cref<JobPointers<Int>> LowerTriangularJobPtr() const
            {
                RequireLowerTriangularJobPtr();
                
                return lower_triangular_job_ptr;
            }
            
            
        protected:
            
            [[nodiscard]] Tensor2<LInt,Int> CreateTransposeCounters() const
            {
                ptic(ClassName()+"::CreateTransposeCounters");
                
                RequireJobPtr();
                
                Tensor2<LInt,Int> counters ( thread_count, n, static_cast<LInt>(0) );
                
                if( WellFormedQ() )
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
                            
                            mptr<LInt> c = counters.data(thread);
                            
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
                    
                    if( WellFormedQ() )
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
                    
                    if( WellFormedQ() )
                    {
                        RequireJobPtr();
                        SortInner();
                        
                        Tensor1<LInt,Int> new_outer ( outer.Size(), 0 );
                        
                        mptr<LInt> new_A_outer = new_outer.data();
                        mptr<LInt>     A_outer = outer.data();
                        mptr<Int>      A_inner = inner.data();
                        
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                const Int i_begin = job_ptr[thread  ];
                                const Int i_end   = job_ptr[thread+1];
                                
                                // To where we write.
                                LInt jj_new       = A_outer[i_begin];
                                
                                // Memoize the next entry in outer because outer will be overwritten
                                LInt next_jj_begin = A_outer[i_begin];
                                
                                for( Int i = i_begin; i < i_end; ++i )
                                {
                                    const LInt jj_begin = next_jj_begin;
                                    const LInt jj_end   = A_outer[i+1];
                                    
                                    // Memoize the next entry in outer because outer will be overwritten
                                    next_jj_begin = jj_end;
                                    
                                    LInt row_nonzero_counter = 0;
                                    
                                    // From where we read.
                                    LInt jj = jj_begin;
                                    
                                    while( jj < jj_end )
                                    {
                                        Int j = A_inner[jj];
                                        
                                        {
//                                            // TODO: Can we remove the overwrite?
//                                            if( jj > jj_new )
//                                            {
//                                                A_inner[jj] = 0;
//                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        while( (jj < jj_end) && (j == A_inner[jj]) )
                                        {
//                                            // TODO: Can we remove the overwrite?
//                                            if( jj > jj_new )
//                                            {
//                                                A_inner[jj] = 0;
//                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        A_inner[jj_new] = j;
                                        
                                        ++jj_new;
                                        ++row_nonzero_counter;
                                    }
                                    new_A_outer[i+1] = row_nonzero_counter;
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
                                
                                const LInt new_pos = new_A_outer[i_begin];
                                const LInt     pos =     A_outer[i_begin];
                                
                                const LInt thread_nonzeroes = new_A_outer[i_end] - new_A_outer[i_begin];
                                
                                copy_buffer(
                                    &inner.data()[pos], 
                                    &new_inner.data()[new_pos],
                                    thread_nonzeroes
                                );
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
            
//#########################################################################################
//####          Matrix Multiplication
//#########################################################################################
            
        protected:
            
            PatternCSR DotBinary_( cref<PatternCSR> B ) const
            {
                ptic(ClassName()+"::DotBinary_");
                
                if( WellFormedQ() && B.WellFormedQ() )
                {
                    RequireJobPtr();
                    
                    ptic("Create counters for counting sort");
                    
                    Tensor2<LInt,Int> counters ( thread_count, m, LInt(0) );
                    
                    // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                    // https://en.wikipedia.org/wiki/Counting_sort
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c       = counters.data(thread);
                            
                            cptr<LInt> A_outer = Outer().data();
                            cptr<Int>  A_inner = Inner().data();
                            
                            cptr<LInt> B_outer = B.Outer().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                LInt c_i = 0;
                                
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
                    
                    ptoc("Create counters for counting sort");
                    
                    AccumulateAssemblyCounters_Parallel<LInt,Int>(counters);
                    
                    const LInt nnz = counters[thread_count-1][m-1];
                    
                    PatternCSR C ( m, B.ColCount(), nnz, thread_count );
                    
                    copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                    
                    ptic("Counting sort");
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c        = counters.data(thread);
                            
                            cptr<LInt> A_outer  = Outer().data();
                            cptr< Int> A_inner  = Inner().data();
                            
                            cptr<LInt> B_outer  = B.Outer().data();
                            cptr< Int> B_inner  = B.Inner().data();
                            
                            mptr< Int> C_inner  = C.Inner().data();
                            
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
                const R_out alpha, cptr<T_in>  X, const Int ldX,
                const S_out beta,  mptr<T_out> Y, const Int ldY,
                const Int nrhs = static_cast<Int>(1)
            ) const
            {
                if( WellFormedQ() )
                {
                    SparseBLAS<T_out,Int,LInt> sblas;
                    
                    sblas.template Multiply_DenseMatrix<NRHS>(
                        outer.data(), inner.data(), nullptr,
                        m, n, alpha, X, ldX, beta, Y, ldY, nrhs, JobPtr()
                    );
                }
                else
                {
                    wprint(ClassName()+"::Dot_: Not WellFormedQ(). Doing nothing.");
                }
            }
            
            // Supply an external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot_(
                      cptr<T_ext> values,
                      const R_out alpha, cptr<T_in>  X, const Int ldX,
                      const S_out beta,  mptr<T_out> Y, const Int ldY,
                      const Int   nrhs = static_cast<Int>(1)
            ) const
            {
                if( WellFormedQ() )
                {
                    SparseBLAS<T_ext,Int,LInt> sblas;
                    
                    sblas.template Multiply_DenseMatrix<NRHS>(
                        outer.data(), inner.data(), values,
                        m, n, alpha, X, ldX, beta, Y, ldY, nrhs, JobPtr()
                    );
                }
                else
                {
                    wprint(ClassName()+"::Dot_: Not WellFormedQ(). Doing nothing.");
                }
            }
            
//##########################################################################################
//####          Conversion Operations
//##########################################################################################
            
        public:
            
            // Supply an external list of values.
            template<typename A_T, typename values_T>
            void WriteDense_( cptr<values_T> values, mptr<A_T> A, const Int ldA ) const
            {
                ParallelDo(
                    [this,values,A,ldA]( const Int i )
                    {
                        const LInt k_begin = outer[i    ];
                        const LInt k_end   = outer[i + 1];
                        
                        zerofy_buffer( &A[i * ldA], n );
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            A[i * ldA + inner[k]] = values[k];
                        } // for( Int k = k_begin; k < k_end; ++k )
                    },
                    JobPtr()
                );
            }
            
            // Supply an external list of values.
            template<typename A_T, typename values_T>
            Tensor2<A_T,LInt> ToTensor2_( cptr<values_T> values ) const
            {
                
                Tensor2<A_T,LInt> A ( m, n );
                
                WriteDense_( values, A.data(), n );
                
                return A;
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
            
            bool InnerSortedQ() const
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
            
            bool NonzeroPositionQ( const Int i, const Int j ) const
            {
                return FindNonzeroPosition( i, j ).found;
            }
            
            Sparse::Position<LInt> FindNonzeroPosition( const Int i, const Int j ) const
            {
                // Looks up the entry {i,j}. If existent, its index within the list of nonzeroes is returned. Otherwise, a negative number is returned (-1 if simply not found and -2 if i is out of bounds).
                
#ifdef TOOLS_DEBUG
                BoundCheck(i,j);
#endif
                
                constexpr LInt threshold = 6;
                
                if( (0 <= i) && (i < m) )
                {
                    cptr<Int> A_inner = inner.data();
                    
                    LInt L = outer[i  ];
                    LInt R = outer[i+1]-1;
                    
                    if( inner_sorted && ( L + threshold > R ) )
                    {
                        if( j < A_inner[L] )
                        {
                            return Sparse::Position<LInt>{0, false};
                        }
                        
                        if( j > A_inner[R] )
                        {
                            return Sparse::Position<LInt>{0, false};
                        }
                        
                        while( L < R )
                        {
                            const LInt k = R - (R-L)/static_cast<Int>(2);
                            const Int col = A_inner[k];
                            
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
                        while( (L < R) && (A_inner[L] < j) )
                        {
                            ++L;
                        }
                        
                    }
                    
                    return (A_inner[L]==j) ? Sparse::Position<LInt> {L, true} : Sparse::Position<LInt>{0, false};
                }
                else
                {
                    wprint(ClassName()+"::FindNonzeroPosition: Row index i = "+ToString(i)+" is out of bounds {0,"+ToString(m)+"}.");
                    return Sparse::Position<LInt>{0, false};
                }
            }
            
            
            template<typename S, typename T, typename J>
            void FillLowerTriangleFromUpperTriangle( mref<std::map<S,Tensor1<T,J>>> values )
            {
                ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
                
                if( WellFormedQ() )
                {
                    if( !inner_sorted )
                    {
                        SortInner();
                    }
                    
                    cptr<LInt> A_diag  = Diag().data();
                    cptr<LInt> A_outer = Outer().data();
                    cptr<Int>  A_inner = Inner().data();
                    
                    ParallelDo(
                        [=,this]( const Int i )
                        {
                            const LInt k_begin = A_outer[i];
                            const LInt k_end   = A_diag [i];
                            
                            for( LInt k = k_begin; k < k_end; ++k )
                            {
                                const Int j = A_inner[k];
                                
                                LInt L = A_diag [j];
                                LInt R = A_outer[j+1]-1;
                                
                                while( L < R )
                                {
                                    const LInt M   = R - (R-L)/static_cast<Int>(2);
                                    const  Int col = A_inner[M];
                                    
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
            
            bool WellFormedQ() const
            {
                bool wellformed = ( ( outer.Size() > 1 ) && ( outer.Last() > 0 ) );
                
                if( !wellformed )
                {
                    wprint(ClassName()+"::WellFormedQ: Matrix is not well formed.");
                    
                    dump(m);
                    dump(n);
                    dump(outer.Size());
                    dump(inner.Size());
                    
                    if( outer.Size() > 0 )
                    {
                        dump(outer.First());
                        dump(outer.Last());
                    }
                }
                
                return wellformed;
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
