#pragma once

#include <charconv>

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
            using Base_T::WellFormedQ;
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
            :   Base_T( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            template<typename I_0, typename I_1, typename I_2, typename I_3>
            MatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T   ( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<LInt>(nnz_), int_cast<Int>(thread_count_) )
            ,   values ( int_cast<LInt>(nnz_) )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_2>,"");
                static_assert(IntQ<I_3>,"");
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
            :   Base_T  ( outer_,  inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values  ( values_, outer_[int_cast<Int>(m_)] )
            {
                static_assert(ArithmeticQ<S>,"");
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            // CAUTION: This reserves memory for the nonzero values, but it does not initialize the nonzero values!
            template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            MatrixCSR(
                const J_0 * const outer_,
                const J_1 * const inner_,
                const I_0 m_,
                const I_1 n_,
                const I_3 thread_count_
            )
            :   Base_T  ( outer_,  inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values  ( outer_[int_cast<Int>(m_)] )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
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
            :   Base_T ( std::move(outer_), std::move(inner_), int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            ,   values ( std::move(values_) )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            // Copy constructor
            MatrixCSR( const MatrixCSR & other ) noexcept
            :   Base_T  ( other        )
            ,   values  ( other.values )
            {
                logprint("Copy of "+ ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nn z = "+ToString(NonzeroCount()));
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
            
            template<typename ExtInt, typename ExtScal>
            MatrixCSR(
                  const ExtInt  * const * const idx,
                  const ExtInt  * const * const jdx,
                  const ExtScal * const * const val,
                  const LInt   * const entry_counts,
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
            
            
            template<typename ExtInt, typename ExtScal>
            MatrixCSR(
                const LInt nnz_,
                const ExtInt  * const i,
                const ExtInt  * const j,
                const ExtScal * const a,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   Base_T ( m_, n_, thread_count )
            {
                Tensor1<const ExtInt  *,Int> idx    (thread_count);
                Tensor1<const ExtInt  *,Int> jdx    (thread_count);
                Tensor1<const ExtScal *,Int> val    (thread_count);
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
            
            template<typename ExtInt, typename ExtScal>
            MatrixCSR(
                  cref<std::vector<std::vector<ExtInt>>>  idx,
                  cref<std::vector<std::vector<ExtInt>>>  jdx,
                  cref<std::vector<std::vector<ExtScal>>> val,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( m_, n_, int_cast<Int>(idx.size()) )
            {
                Int list_count = int_cast<Int>(idx.size());
                Tensor1<const ExtInt  *,Int> i      (list_count);
                Tensor1<const ExtInt  *,Int> j      (list_count);
                Tensor1<const ExtScal *,Int> a      (list_count);
                Tensor1<LInt,Int> entry_counts  (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    i[thread] = idx[thread].data();
                    j[thread] = jdx[thread].data();
                    a[thread] = val[thread].data();
                    entry_counts[thread] = int_cast<LInt>(idx[thread].size());
                }
                
                FromTriples(
                    i.data(), j.data(), a.data(),
                    entry_counts.data(), list_count,
                    final_thread_count, compressQ, symmetrize
                );
            }
            
            template<typename ExtInt, typename ExtScal>
            MatrixCSR(
                  cref<std::vector<TripleAggregator<ExtInt,ExtInt,ExtScal,LInt>>> triples,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( m_, n_, int_cast<Int>(triples.size()) )
            {
                Int list_count = int_cast<Int>(triples.size());
                
                Tensor1<const ExtInt  *,Int> i (list_count);
                Tensor1<const ExtInt  *,Int> j (list_count);
                Tensor1<const ExtScal *,Int> a (list_count);
                Tensor1<LInt,Int> entry_counts (list_count);
                
                for( Int thread = 0; thread < list_count; ++thread )
                {
                    const Size_T t = static_cast<Size_T>(thread);
                    i[thread] = triples[t].Get_0().data();
                    j[thread] = triples[t].Get_1().data();
                    a[thread] = triples[t].Get_2().data();
                    entry_counts[thread] = int_cast<LInt>(triples[t].Size());
                }
                
                FromTriples(
                    i.data(), j.data(), a.data(), entry_counts.data(),
                    list_count, final_thread_count, compressQ, symmetrize
                );
            }
            
            template<typename ExtInt, typename ExtScal>
            MatrixCSR(
                  cref<TripleAggregator<ExtInt,ExtInt,ExtScal,LInt>> triples,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ  = true,
                  const int  symmetrize = 0
            )
            :   Base_T ( m_, n_, Int(1) )
            {
                LInt entry_counts = int_cast<LInt>(triples.Size());
                
                const ExtInt  * const i = triples.Get_0().data();
                const ExtInt  * const j = triples.Get_1().data();
                const ExtScal * const a = triples.Get_2().data();

                FromTriples(
                    &i,&j,&a,&entry_counts,
                    ExtInt(1),final_thread_count,compressQ,symmetrize
                );
            }
            
            virtual ~MatrixCSR() override = default;
            
        protected:
            
            template<typename ExtInt, typename ExtScal>
            void FromTriples(
                const ExtInt  * const * const idx,               // list of lists of i-indices
                const ExtInt  * const * const jdx,               // list of lists of j-indices
                const ExtScal * const * const val,               // list of lists of nonzero values
                const LInt            * const entry_counts,      // list of lengths of the lists above
                const Int list_count,                         // number of lists
                const Int final_thread_count,                 // number of threads that the matrix shall use
                const bool compressQ   = true,                // whether to do additive assembly or not
                const int  symmetrize = 0                     // whether to symmetrize the matrix
            )
            {
                // Parallel sparse matrix assembly using counting sort.
                // Counting sort employs list_count threads (one per list).
                // Sorting of column indices and compression step employ final_thread_count threads.
                
                // k-th i-list goes from idx[k] to &idx[k][entry_counts[k]] (last one excluded)
                // k-th j-list goes from jdx[k] to &jdx[k][entry_counts[k]] (last one excluded)
                // and k goes from 0 to list_count (last one excluded)
                
                TOOLS_PTIC(ClassName()+"::FromTriples");
                
                if( symmetrize )
                {
                    pprint(ClassName()+"::FromTriples symmetrize");
                }
                else
                {
                    pprint(ClassName()+"::FromTriples no symmetrize");
                }
                
                if( compressQ )
                {
                    pprint(ClassName()+"::FromTriples compressQ");
                }
                else
                {
                    pprint(ClassName()+"::FromTriples no compressQ");
                }
                
                Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int>(
                    idx, jdx, entry_counts, list_count, m, symmetrize
                );
                
                if( list_count <= 0 )
                {
                    eprint(ClassName()+"::FromTriples: list_count <= 0");
                }
                
                const LInt nnz = counters[list_count-1][m-1];
                
                if( nnz > LInt(0) )
                {
                    inner  = Tensor1<Int ,LInt>( nnz );
                    values = Tensor1<Scal,LInt>( nnz );
                    
                    mptr<LInt> A_outer = outer.data();
                    mptr<Int>  A_inner = inner.data();
                    mptr<Scal> A_value = values.data();
                    
                    copy_buffer( counters.data(list_count-1), &A_outer[1], m );
                    
                    // The counters array tells each thread where to write.
                    // Since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.
                    
                    // TODO: The threads write quite chaotically to inner_ and value_. This might cause a lot of false sharing. Nonetheless, it seems to scale quite well -- at least among 4 threads!
                    
                    // TODO: False sharing can be prevented by not distributing whole sublists of idx, jdx, val to the threads but by distributing the rows of the final matrix, instead. It's just a bit fiddly, though.
                    
                    // Writing reordered data.
                    ParallelDo(
                        [=,&counters]( const Int thread )
                        {
                            const LInt entry_count = entry_counts[thread];
                            
                            cptr<ExtInt>  thread_idx = idx[thread];
                            cptr<ExtInt>  thread_jdx = jdx[thread];
                            cptr<ExtScal> thread_val = val[thread];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            for( LInt k = entry_count; k --> LInt(0); )
                            {
                                const Int  i = static_cast<Int>(thread_idx[k]);
                                const Int  j = static_cast<Int>(thread_jdx[k]);
                                const Scal a = static_cast<Scal>(thread_val[k]);
                                
                                {
                                    const LInt pos  = --c[i];
                                    A_inner[pos] = j;
                                    A_value[pos] = a;
                                }
                                
                                // Write the transposed matrix (diagonal excluded) in the same go in order to symmetrize the matrix. (Typical use case: Only the upper triangular part of a symmetric matrix is stored in idx, jdx, and val, but we need the full, symmetrized matrix.)
                                if( (symmetrize != 0) && (i != j) )
                                {
                                    const LInt pos  = --c[j];
                                    A_inner[pos] = i;
                                    A_value[pos] = a;
                                }
                            }
                        },
                        list_count
                    );
                    
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
                
                TOOLS_PTOC(ClassName()+"::FromTriples");
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
                if( k < LInt(0) || k >= values.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return values[k];
            }
            
            cref<Scal> Value( const LInt k ) const
            {
#ifdef TOOLS_DEBUG
                if( k < LInt(0) || k >= values.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return values[k];
            }
            
            
            Scal operator()( const Int i, const Int j ) const
            {
                const Sparse::Position<LInt> pos = this->FindNonzeroPosition(i,j);
                
                return ( pos.found ) ? values[pos.index] : static_cast<Scal>(0);
            }
            
        public:
            
            template<Tools::Op op,
                typename Return_T =
                std::conditional_t<
                    (op == Tools::Op::Re) || (op == Tools::Op::ReTrans)
                    ||
                    (op == Tools::Op::Im) || (op == Tools::Op::ImTrans)
                    ,
                    Scalar::Real<Scal>,
                    Scal
                >
            >
            MatrixCSR<Return_T,Int,LInt> Op() const
            {
                std::string tag = ClassName() + "::Op"
                    + "<" + ToString(op)
                    + "," + TypeName<Return_T>
                    + ">";
                
                TOOLS_PTIC(tag);
                
                if( !this->WellFormedQ() )
                {
                    TOOLS_PTOC(tag);
                    return MatrixCSR<Return_T,Int,LInt> ( m, n, 0, 1 );
                }
                
                if constexpr (
                    (op == Op::Id)
                    ||
                    (Scalar::RealQ<Scal> && ( (op == Op::Conj) || (op == Op::Re) ) )
                )
                {
                    TOOLS_PTOC(tag);
                    return MatrixCSR<Return_T,Int,LInt>( *this );
                }
                else if constexpr ( Scalar::RealQ<Scal> && (op == Op::Im) )
                {
                    if constexpr ( NotTransposedQ(op) )
                    {
                        TOOLS_PTOC(tag);
                        return MatrixCSR<Return_T,Int,LInt> ( m, n, 0, 1 );
                    }
                    else
                    {
                        TOOLS_PTOC(tag);
                        return MatrixCSR<Return_T,Int,LInt> ( n, m, 0, 1 );
                    }
                }
                else if constexpr ( NotTransposedQ(op) )
                {
                    MatrixCSR<Return_T,Int,LInt> B ( m, n, outer[m], thread_count );
                    
                    B.Outer().Read( outer.data(), thread_count );
                    B.Inner().Read( inner.data(), thread_count );
                    
                    mptr<Return_T> B_val = B.Values();
                    
                    ParallelDo(
                        [B_val,this]( LInt k )
                        {
                            B_val[k] = Scalar::Op<op>( values[k] );
                        },
                        outer[m], thread_count
                    );
                    
                    TOOLS_PTOC(tag);
                    return B;
                }
                else // if constexpr ( TransposedQ(op) )
                {
                    RequireJobPtr();
                    
                    Tensor2<LInt,Int> counters = CreateTransposeCounters();
                    
                    MatrixCSR<Return_T,Int,LInt> B ( n, m, outer[m], thread_count );

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
                            cptr<Scal> A_value  = Value().data();
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = A_outer[i  ];
                                const LInt k_end   = A_outer[i+1];
                                
                                for( LInt k = k_end; k --> k_begin; )
                                {
                                    const Int j = A_inner[k];
                                    const LInt pos = --c[j];
                                    B_inner [pos] = i;
                                    
                                    B_values[pos] = Scalar::Op<op>(A_value[k]);
                                }
                            }
                        },
                        thread_count
                    );

                    // Finished counting sort.
                    
                    // We only have to care about the correct ordering of inner indices and values.
                    B.SortInner();

                    TOOLS_PTOC(tag);
                    return B;
                }
                    
            }
            
            MatrixCSR Transpose() const
            {
                return Op<Op::Trans>();
            }
            
            MatrixCSR ConjugateTranspose() const
            {
                return Op<Op::ConjTrans>();
            }
            
        public:
            
            // Supply an external list of values.
            template<typename A_T = Scal>
            void WriteDense( mptr<A_T> A, const Int ldA ) const
            {
                return this->WriteDense_( values.data(), A, ldA );
            }
            
            // Supply an external list of values.
            template<typename A_T = Scal>
            Tensor2<A_T,LInt> ToTensor2() const
            {
                return this->template ToTensor2_<A_T>( values.data() );
            }
            
            
            void SortInner() const override
            {   
                if( !inner_sorted )
                {
                    TOOLS_PTIC(ClassName()+"::SortInner");

                    if( this->WellFormedQ() )
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
                    
                    TOOLS_PTOC(ClassName()+"::SortInner");
                }
            }
            
            
            void Compress() const override
            {
                // Removes duplicate {i,j}-pairs by adding their corresponding nonzero values.
                
                if( !duplicate_free )
                {
                    TOOLS_PTIC(ClassName()+"::Compress");
                    
                    if( this->WellFormedQ() )
                    {
                        RequireJobPtr();
                        SortInner();
                        
                        Tensor1<LInt,Int> new_outer (outer.Size(),0);
                        
                        cptr<LInt> A_outer     = outer.data();
                        mptr<Int > A_inner     = inner.data();
                        mptr<Scal> A_value     = values.data();
                        mptr<LInt> new_A_outer = new_outer.data();
                        
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                const Int i_begin = job_ptr[thread  ];
                                const Int i_end   = job_ptr[thread+1];
      
                                // To where we write.
                                LInt jj_new        = A_outer[i_begin];
                                LInt next_jj_begin = A_outer[i_begin];
                                
                                for( Int i = i_begin; i < i_end; ++i )
                                {
                                    const LInt jj_begin = next_jj_begin;
                                    const LInt jj_end   = A_outer[i+1];
                                    
                                    // Memorize the next entry in outer because outer will be overwritten
                                    next_jj_begin = jj_end;
                                    
                                    LInt row_nonzero_counter = 0;
                                    
                                    // From where we read.
                                    LInt jj = jj_begin;
                                    
                                    while( jj< jj_end )
                                    {
                                        Int j = A_inner [jj];
                                        Scal a = A_value[jj];
                                        
                                        {
//                                            if( jj > jj_new )
//                                            {
//                                                A_inner[jj] = 0;
//                                                A_value[jj] = 0;
//                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        while( (jj < jj_end) && (j == A_inner[jj]) )
                                        {
                                            a+= A_value[jj];
                                            
//                                            if( jj > jj_new )
//                                            {
//                                                A_inner[jj] = 0;
//                                                A_value[jj] = 0;
//                                            }
                                            
                                            ++jj;
                                        }
                                        
                                        A_inner[jj_new] = j;
                                        A_value[jj_new] = a;
                                        
                                        ++jj_new;
                                        ++row_nonzero_counter;
                                    }
                                    
                                    new_A_outer[i+1] = row_nonzero_counter;
                                }
                            },
                            thread_count
                        );
                        
                        // This is the new array of outer indices.
                        new_outer.Accumulate( thread_count );
                        
                        const LInt nnz = new_outer[m];
                        
                        // Now we create a new arrays for new_inner and new_values.
                        // Then we copy inner and values to it, eliminating the gaps in between.
                        
                        Tensor1< Int,LInt> new_inner  (nnz);
                        Tensor1<Scal,LInt> new_values (nnz);
                        
                        mptr<Int > new_A_inner = new_inner.data();
                        mptr<Scal> new_A_value = new_values.data();
                        
                        //TODO: Parallelization might be a bad idea here.
                        ParallelDo(
                            [=,this]( const Int thread )
                            {
                                const  Int i_begin = job_ptr[thread  ];
                                const  Int i_end   = job_ptr[thread+1];
                                
                                const LInt new_pos = new_A_outer[i_begin];
                                const LInt     pos =     A_outer[i_begin];
                                
                                const LInt thread_nonzeroes = new_A_outer[i_end] - new_A_outer[i_begin];
                                
                                // Starting position of thread in inner list.
                                
                                copy_buffer( &A_inner[pos], &new_A_inner[new_pos],  thread_nonzeroes );
                                
                                copy_buffer( &A_value[pos], &new_A_value[new_pos], thread_nonzeroes );
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
                    
                    TOOLS_PTOC(ClassName()+"::Compress");
                }
            }
            
        
//#########################################################################################
//####          Permute
//#########################################################################################
            
        public:
            
            MatrixCSR Permute(
                cref<Tensor1<Int,Int>> p,
                cref<Tensor1<Int,Int>> q,
                bool sortQ = true
            )
            {
                if( p.Dim(0) != m )
                {
                    eprint(ClassName()+"::Permute: Length of first argument does not coincide with RowCount().");
                    return MatrixCSR();
                }
                
                if( q.Dim(0) != n )
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
                        [=,this]( const Int i )
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
                    
                    cptr<LInt> A_outer  = outer.data();
                    cptr<Int > A_inner  = inner.data();
                    mptr<Scal> A_value = values.data();
                    
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
                            copy_buffer( &A_value[A_begin], &A_value[A_end], &B_values[B_begin] );
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
                        cptr<Scal> A_value = values.data();
                        
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
                            
                            copy_buffer( &A_value[B_begin], &B_values[B_begin], k_max );
                            
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
                        [=,this]( const Int i )
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
                        cptr<Scal> A_value = values.data();
                        
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
                            
                            copy_buffer( &A_value[A_begin], &B_values[B_begin], k_max );
                            
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
                TOOLS_PTIC(ClassName()+"::Dot");
                
                if( this->WellFormedQ() )
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
                    
                    AccumulateAssemblyCounters_Parallel( counters );
                    
                    const LInt nnz = counters.data(thread_count-1)[m-1];
                    
                    MatrixCSR C ( m, B.ColCount(), nnz, thread_count );
                    
                    copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );
                    
                    ParallelDo(
                        [&,this]( const Int thread )
                        {
                            const Int i_begin   = job_ptr[thread  ];
                            const Int i_end     = job_ptr[thread+1];
                            
                            mptr<LInt> c        = counters.data(thread);
                            
                            cptr<LInt> A_outer  = Outer().data();
                            cptr< Int> A_inner  = Inner().data();
                            cptr<Scal> A_value  = Value().data();
                            
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
                                        C_values[pos] = A_value[jj] * B_values[kk];
                                    }
                                }
                            }
                        },
                        thread_count
                    );
                    
                    // Finished expansion phase (counting sort).
                    
                    // Finally we row-sort inner and compressQ duplicates in inner and values.
                    C.Compress();
                    
                    TOOLS_PTOC(ClassName()+"::Dot");
                    
                    return C;
                }
                else
                {
                    eprint(ClassName()+"::Dot: Matrix is not well-formed.");
                    
                    TOOLS_PTOC(ClassName()+"::Dot");
                    
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
            
//###################################################################################
//####          Matrix Multiplication
//###################################################################################        
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X,
                const b_T beta,  mptr<Y_T> Y,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( values.data(), alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            // Use own nonzero values.
            template<typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    const Int nrhs = 1;
                    
                    this->template Dot_<1>( values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            // Use own nonzero values.
            template<Int NRHS = VarSize, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                 const a_T alpha, cref<Tensor2<X_T,Int>> X,
                 const b_T beta,  mref<Tensor2<Y_T,Int>> Y
             ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m && (X.Dim(1) == Y.Dim(1)) )
                {
                    const Int nrhs = X.Dim(1);
                    
                    this->template Dot_<NRHS>( values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<T_ext> X, const Int ldX,
                const b_T beta,  mptr<T_ext> Y, const Int ldY,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Use external list of values.
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<T_ext> X,
                const b_T beta,  mptr<T_ext> Y,
                const Int nrhs = Int(1)
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                cref<Tensor1<T_ext,Int>> ext_values,
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    const Int nrhs = 1;
                    
                    this->template Dot_<1>( ext_values.data(), alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = VarSize, typename T_ext, typename a_T, typename X_T, typename b_T, typename Y_T>
            void Dot(
                 cref<Tensor1<T_ext,Int>> ext_values,
                 const a_T alpha, cref<Tensor2<X_T,Int>> X,
                 const b_T beta,  mref<Tensor2<Y_T,Int>> Y
         ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m && (X.Dim(1) == Y.Dim(1)) )
                {
                    const Int nrhs = X.Dim(1);
                    
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
            
            
            void LoadFromFile(
                const std::filesystem::path & file, const Int thread_count_
            )
            {
                std::string tag = ClassName() + "::LoadFromMatrixMarket";
                
                TOOLS_PTIC(tag);
                
                std::ifstream  s ( file );
        
                if( !s.good() )
                {
                    eprint(tag + ": File " + file.string() + " not found. Aborting.");
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                Int m;
                Int n;
                Int nnz;
                
                s >> m;
                s >> n;
                s >> nnz;
                
                MatrixCSR<Scal,Int,LInt> A( m, n, nnz, thread_count_ );
                
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
                
                swap( A, *this);
            }

            void LoadFromMatrixMarket( cref<std::filesystem::path> file, Int thread_count_ )
            {
                std::string tag = ClassName() + "::LoadFromMatrixMarket";
                
                TOOLS_PTIC(tag);
                
                std::ifstream  s ( file );
        
                if( !s.good() )
                {
                    eprint(tag + ": File " + file.string() + " not found. Aborting.");
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                logprint("Loading from file " + file.string() );
                
                std::string token;
                
                s >> token;
                
                std::transform(token.begin(), token.end(), token.begin(), ::tolower);
                if( token != "%%matrixmarket")
                {
                    eprint( tag + ": Not a MatrixMarket file. Doing nothing.");
                    TOOLS_DUMP( token );
                    TOOLS_LOGDUMP( token );
                    TOOLS_PTOC(tag);
                    return;
                }
                
                s >> token;
                std::transform(token.begin(), token.end(), token.begin(), ::tolower);
                
                if( token != "matrix")
                {
                    eprint( tag + ": Second word in file is not \"matrix\". Doing nothing.");
                    TOOLS_DUMP( token );
                    TOOLS_PTOC(tag);
                    return;
                }
                
                s >> token;
                std::transform(token.begin(), token.end(), token.begin(), ::tolower);
                if( token != "coordinate")
                {
                    eprint( tag + ": Third word in file is not \"coordinate\". Stored matrix is a dense matrix and shall better not be loaded. Doing nothing.");
                    TOOLS_DUMP( token );
                    TOOLS_PTOC(tag);
                    return;
                }
                
                std::string scalar_type;
                s >> scalar_type;
                std::transform(scalar_type.begin(), scalar_type.end(), scalar_type.begin(), ::tolower);
                if constexpr ( Scalar::RealQ<Scal> )
                {
                    if( scalar_type == "complex")
                    {
                        eprint( tag + ": Scalar type requested is " + TypeName<Scal> + ", but type in file is \"complex\". Doing nothing.");
                        TOOLS_PTOC(tag);
                        return;
                    }
                }
                
                if constexpr ( IntQ<Scal> )
                {
                    if( (scalar_type != "integer") && (scalar_type != "pattern") )
                    {
                        eprint( tag + ": Scalar type requested is " + TypeName<Scal> + ", but type in file is \"" + scalar_type + "\". Doing nothin.");
                        TOOLS_PTOC(tag);
                        return;
                    }
                }
                
                std::string symmetry;
                s >> symmetry;
                std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);
                
                bool symmetrizeQ = false;
                
                if( symmetry == "skew-symmetric")
                {
                    eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". The current implementation cannot handle this. Doing nothing.");
                    TOOLS_PTOC(tag);
                    return;
                }
                else if ( symmetry == "hermitian")
                {
                    eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". The current implementation cannot handle this. Doing nothing.");
                    TOOLS_PTOC(tag);
                    return;
                }
                else if ( symmetry == "symmetric")
                {
                    symmetrizeQ = true;
                }
                else if ( symmetry == "general")
                {
                    symmetrizeQ = false;
                }
                else
                {
                    eprint( tag + ": Matrix symmetry is \"" + symmetry + "\". This is invalid for the MatrixMarket format. Doing nothing.");
                    TOOLS_DUMP( token );
                    TOOLS_PTOC(tag);
                    return;
                }
                    

                s.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                
                while( std::getline( s, token ) && (token[0] == '%') )
                {}
                
                std::stringstream line (token);
                
                Int row_count = 0;
                line >> row_count;
                TOOLS_LOGDUMP(row_count);
                
                Int col_count = 0;
                line >> col_count;
                TOOLS_LOGDUMP(col_count);
                
                LInt nonzero_count = 0;
                line >> nonzero_count;
                TOOLS_LOGDUMP(nonzero_count);

                if(row_count < Int(0))
                {
                    eprint( tag + ": Invalid row_count." );
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                if(col_count < Int(0))
                {
                    eprint( tag + ": Invalid col_count." );
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                if(nonzero_count < LInt(0))
                {
                    eprint( tag + ": Invalid nonzero_count." );
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                Tensor1<Int, LInt> i_list ( nonzero_count );
                Tensor1<Int, LInt> j_list ( nonzero_count );
                Tensor1<Scal,LInt> a_list ( nonzero_count );
                
                // TODO: We could also use std::from_chars -- once that is broadly available with floating point support an all compilers.
                
                if ( scalar_type == "pattern" )
                {
                    for( LInt k = 0; k < nonzero_count; ++k )
                    {
                        Int i;
                        Int j;
                        s >> i;
                        i_list[k] = i - Int(1);
                        s >> j;
                        j_list[k] = j - Int(1);
                        
                        a_list[k] = Scal(1);
                    }
                }
                else if ( scalar_type != "complex" )
                {
                    for( LInt k = 0; k < nonzero_count; ++k )
                    {
                        Int i;
                        Int j;
                        
                        s >> i;
                        i_list[k] = i - Int(1);
                        s >> j;
                        j_list[k] = j - Int(1);
                        
                        s >> a_list[k];
                    }
                }
                else if constexpr (Scalar::ComplexQ<Scal>)
                {
                    for( LInt k = 0; k < nonzero_count; ++k )
                    {
                        Int i;
                        Int j;
                        s >> i;
                        i_list[k] = i - Int(1);
                        s >> j;
                        j_list[k] = j - Int(1);
                        
                        Scalar::Real<Scal> re;
                        Scalar::Real<Scal> im;
                        s >> re;
                        s >> im;
                        
                        a_list[k] = Scal(re,im);
                    }
                }
                
                MatrixCSR<Scal,Int,LInt> A (
                    nonzero_count,
                    i_list.data(), j_list.data(), a_list.data(),
                    row_count, col_count, thread_count_, true, symmetrizeQ
                );
                
                swap( *this, A );
                
                TOOLS_PTOC(tag);
            }
            
            
            void WriteToMatrixMarket( cref<std::filesystem::path> file )
            {
                std::string tag = ClassName() + "::WriteToMatrixMarket";
                
                TOOLS_PTIC(tag);
                
                if( !WellFormedQ() )
                {
                    eprint( tag + ": Matrix is not well-formed. Doing nothing." );
                    
                    TOOLS_PTOC(tag);
                    
                    return;
                }
                
                std::ofstream  s ( file );
                
                s << "%%MatrixMarket" << " " << "matrix" << " " << "coordinate" << " ";
                
                if constexpr ( Scalar::ComplexQ<Scal> )
                {
//                    s << std::scientific << std::uppercase << std::setprecision( std::numeric_limits<Scalar::Real<Scal>>::digits10 + 1 );
                    s << "complex";
                }
                else if constexpr ( IntQ<Scal> )
                {
                    s << "integer";
                }
                else /*if constexpr ( Scalar::RealQ<Scal> )*/
                {
//                    s << std::scientific << std::uppercase << std::setprecision( std::numeric_limits<Scal>::digits10 + 1 );
                    s << "real";
                }
                
                s << " " << "general" << "\n";
                
                
                s << RowCount() << " " << ColCount() << " " << NonzeroCount() << "\n";
                
                const Int s_thread_count = 4;
                
                std::vector<std::string> thread_strings ( s_thread_count );
                
                auto s_job_ptr = JobPointers<Int>( m, outer.data(), s_thread_count, false );
                
                ParallelDo(
                    [&,this]( const Int thread )
                    {
                        const Int i_begin = s_job_ptr[thread+0];
                        const Int i_end   = s_job_ptr[thread+1];
   
                        char line[128];
                        
                        std::string s_loc;
                        
                        // TODO: Use std::to_chars .
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt k_begin = outer[i    ];
                            const LInt k_end   = outer[i + 1];
        
                            for( LInt k = k_begin; k < k_end; ++k )
                            {
                                const Int  j = inner[k];
                                const Scal a = values[k];
                                
                                if constexpr ( IntQ<Scal> )
                                {
                                    std::snprintf(line, 128, "%d %d %d\n", i+1,j+1,a);
                                }
                                else if constexpr ( std::is_same_v<Scal,Complex64> )
                                {
                                    std::snprintf(line, 128, "%d %d %.17E %.17E\n", i+1,j+1,Re(a),Im(a));
                                }
                                else if constexpr ( std::is_same_v<Scal,Complex32> )
                                {
                                    std::snprintf(line, 128, "%d %d %.7E %.7E\n", i+1,j+1,Re(a),Im(a));
                                }
                                else if constexpr ( std::is_same_v<Scal,Real64> )
                                {
                                    std::snprintf(line, 128, "%d %d %.17E\n", i+1,j+1,a);
                                }
                                else if constexpr ( std::is_same_v<Scal,Real32> )
                                {
                                    std::snprintf(line, 128, "%d %d %.7E\n", i+1,j+1,a);
                                }
                                
                                s_loc += line;
                            }
                        }

                        thread_strings[thread] = std::move(s_loc);
                    },
                    s_thread_count
                );
                
                ParallelDo(
                    [&,this]( const Int thread )
                    {
                        thread_strings[2 * thread] += thread_strings[2 * thread + 2];
                    },
                    2
                );
                
                s << thread_strings[0];
                s << thread_strings[2];
                
//                for( Int i = 0; i < m; ++i )
//                {
//                    const LInt k_begin = outer[i    ];
//                    const LInt k_end   = outer[i + 1];
//
//                    for( LInt k = k_begin; k < k_end; ++k )
//                    {
//                        const Int j = inner[k];
//
//                        if constexpr ( Scalar::ComplexQ<Scal> )
//                        {
//                            const Scal a = values[k];
//
//                            s << (i+1) << " " << (j+1) << " " << Re(a) << " " << Im(a) << "\n";
//                        }
//                        else
//                        {
//                            s << (i+1) << " " << (j+1) << " " << values[k] << "\n";
//                        }
//                    }
//                }
   
                TOOLS_PTOC(tag);
            }
            
            
            bool SymmetricQ() const
            {
                // This routine is not optimized.
                // Only meant for debugging.
                
                bool symmetricQ = true;
                
                LInt unmatched_count = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    const LInt k_begin = outer[i    ];
                    const LInt k_end   = outer[i + 1];
                    
                    for( LInt k = k_begin; k < k_end; ++k )
                    {
                        const Int j = inner[k];
                        
                        if( i < j )
                        {
                            const Scal A_ij = values[k];
                            const Scal A_ji = this->operator()(j,i);

                            const bool found = (A_ij == A_ji);
                            
                            symmetricQ = symmetricQ && found;
                            
                            unmatched_count += (!found);
                        }
                    }
                }
                
                if( !symmetricQ )
                {
                    TOOLS_DUMP(unmatched_count);
                }
                
                return symmetricQ;
            }
            
            std::string Stats() const
            {
                return std::string()
                + "\n==== "+ClassName()+" Stats ====" + "\n\n"
                + " RowCount()      = " + ToString(RowCount()) + "\n"
                + " ColCount()      = " + ToString(ColCount()) + "\n"
                + " NonzeroCount()  = " + ToString(NonzeroCount()) + "\n"
                + " ThreadCount()   = " + ToString(ThreadCount()) + "\n"
                + " Outer().Size()  = " + ToString(Outer().Size()) + "\n"
                + " Inner().Size()  = " + ToString(Inner().Size()) + "\n"
                + " Value().Size()  = " + ToString(Value().Size()) + "\n"
                + "\n==== "+ClassName()+" Stats ====\n\n";
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::MatrixCSR<")+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // MatrixCSR
    
    
        
    } // namespace Sparse
    
} // namespace Tensors
