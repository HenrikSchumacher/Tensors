#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Int_, typename LInt_>
        class BinaryMatrixCSR : public Sparse::PatternCSR<Int_,LInt_>
        {
        private:
            
            using Base_T = Sparse::PatternCSR<Int_,LInt_>;
            
        public:
            
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            using Base_T::m;
            using Base_T::n;
            using Base_T::outer;
            using Base_T::inner;
            using Base_T::thread_count;
            using Base_T::job_ptr;
            using Base_T::upper_triangular_job_ptr;
            using Base_T::lower_triangular_job_ptr;
            using Base_T::Dot_;
            
        public:
            
            using Base_T::RowCount;
            using Base_T::ColCount;
            using Base_T::NonzeroCount;
            using Base_T::ThreadCount;
            using Base_T::SetThreadCount;
            using Base_T::Outer;
            using Base_T::Inner;
            using Base_T::JobPtr;
            using Base_T::Diag;
            using Base_T::SortInner;
            using Base_T::RequireDiag;
            using Base_T::RequireJobPtr;
            using Base_T::RequireUpperTriangularJobPtr;
            using Base_T::RequireLowerTriangularJobPtr;
            using Base_T::UpperTriangularJobPtr;
            using Base_T::LowerTriangularJobPtr;
            using Base_T::CreateTransposeCounters;
            using Base_T::WellFormedQ;
            
            BinaryMatrixCSR()
            :   Base_T()
            {}
            
            template<typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
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
            BinaryMatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T( int_cast<Int>(m_), int_cast<Int>(n_), int_cast<LInt>(nnz_), int_cast<Int>(thread_count_) )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_2>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            
            template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
                  const J_0 * const outer_,
                  const J_1 * const inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( outer_, inner_, int_cast<Int>(m_), int_cast<Int>(n_), int_cast<Int>(thread_count_) )
            {
                static_assert(IntQ<J_0>,"");
                static_assert(IntQ<J_1>,"");
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            template<typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
                  const Tensor1<LInt, Int> & outer_,
                  const Tensor1< Int,LInt> & inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( 
                    outer_,
                    inner_,
                    int_cast<Int>(m_),
                    int_cast<Int>(n_),
                    int_cast<Int>(thread_count_)
                )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            template<typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
                  Tensor1<LInt, Int> && outer_,
                  Tensor1< Int,LInt> && inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
            )
            :   Base_T( 
                    std::move(outer_), 
                    std::move(inner_),
                    int_cast<Int>(m_),
                    int_cast<Int>(n_),
                    int_cast<Int>(thread_count_)
                )
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_3>,"");
            }
            
            // Copy constructor
            BinaryMatrixCSR( const BinaryMatrixCSR & other )
            :   Base_T( other )
            {
                logprint("Copy of "+ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nnz = "+ToString(NonzeroCount()));
            }
            
            friend void swap( BinaryMatrixCSR & A, BinaryMatrixCSR & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( static_cast<Base_T&>(A), static_cast<Base_T&>(B) );
            }
            
            // (Copy-)assignment operator
            BinaryMatrixCSR & operator=( BinaryMatrixCSR other ) // Pass by value is okay, because we use copy-swap idiom and copy elision.
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                
                swap(*this, other);
                
                return *this;
            }
            
            // Move constructor
            BinaryMatrixCSR( BinaryMatrixCSR && other ) noexcept : BinaryMatrixCSR()
            {
                swap(*this, other);
            }
            
            // We do not need a move-assignment operator, because we use the copy-swap idiom!
            
            
            BinaryMatrixCSR(
                  const Int    * const * const idx,
                  const Int    * const * const jdx,
                  const LInt   *         const entry_counts,
                  const Int list_count,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T ( idx, jdx, entry_counts, list_count, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                const LInt nnz_,
                const Int  * const i,
                const Int  * const j,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ = true,
                const int  symmetrizeQ = 0
            )
            :   Base_T ( nnz_, i, j, m_, n_, thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  std::vector<Int> & idx,
                  std::vector<Int> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  const std::vector<std::vector<Int>> & idx,
                  const std::vector<std::vector<Int>> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  cref<std::vector<PairAggregator<Int, Int, LInt>>> idx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            BinaryMatrixCSR(
                  cref<PairAggregator<Int, Int, LInt>> idx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ = true,
                  const int  symmetrizeQ = 0
                  )
            :   Base_T( idx, m_, n_, final_thread_count, compressQ, symmetrizeQ )
            {}
            
            virtual ~BinaryMatrixCSR() override = default;
            
            
            BinaryMatrixCSR Transpose() const
            {
                TOOLS_PTIC(ClassName()+"::Transpose");
                
                Tensor2<LInt,Int> counters = CreateTransposeCounters();
                
                BinaryMatrixCSR<Int,LInt> B ( n, m, outer[m], thread_count );
                
                copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );
                
                if( this->WellFormedQ() )
                {
                    ParallelDo(
                        [&]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = outer[i  ];
                                const LInt k_end   = outer[i+1];
                                
                                for( LInt k = k_end; k --> k_begin; )
                                {
                                    const Int j = inner[k];
                                    const LInt pos = --c[j];
                                    B.Inner(pos) = i;
                                }
                            }
                        },
                        thread_count
                    );
                }
                
                // Finished counting sort.
                
                // We only have to care about the correct ordering of inner indices and values.
                B.SortInner();
                
                TOOLS_PTOC(ClassName()+"::Transpose");
                
                return B;
            }
            

//#########################################################################
//####          Permute
//#########################################################################
            
        public:
            
            Tensor1<LInt,LInt> Permute(
                const Permutation<Int> & p,     // row    permutation
                const Permutation<Int> & q      // column permutation
            )
            {
                TOOLS_PTIC(ClassName()+"::Permute");
                // Modifies inner and outer  accordingly; returns the permutation to be applied to the nonzero values.
                
                
                this->inner_sorted = true;
                this->diag_ptr_initialized = false;
                this->job_ptr_initialized  = false;
                this->upper_triangular_job_ptr_initialized = false;
                this->lower_triangular_job_ptr_initialized = false;
                
                Tensor1<LInt,LInt> perm = PermutePatternCSR( outer, inner, p, q, inner.Size(), true );
                
                TOOLS_PTOC(ClassName()+"::Permute");
                
                return perm;
            }
            
//#########################################################################
//####          Matrix Multiplication
//#########################################################################
            
        public:
            
            [[nodiscard]] BinaryMatrixCSR Dot( const BinaryMatrixCSR & B ) const
            {
                BinaryMatrixCSR<Int,LInt> result;
                
                Base_T C = this->DotBinary_(B);
                
                swap(result,C);
                
                return result;
            }
            
            [[nodiscard]] BinaryMatrixCSR DotBinary( const BinaryMatrixCSR & B ) const
            {
                BinaryMatrixCSR result;
                
                Base_T C = this->DotBinary_(B);
                
                swap(result,C);
                
                return result;
            }
            
//#########################################################################
//####          Matrix Multiplication
//#########################################################################
            

            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cptr<X_T> X,
                const b_T beta,  mptr<Y_T> Y,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    this->template Dot_<1>(
                        alpha, X.data(), Int(1),
                        beta,  Y.data(), Int(1),
                        Int(1)
                    );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = 0, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                const a_T alpha, cref<Tensor2<X_T,Int>> X,
                const b_T beta,  mref<Tensor2<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m && (X.Dim(1) == Y.Dim(1)) )
                {
                    const Int nrhs = X.Dim(1);
                    
                    this->template Dot_<NRHS>( alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            
            
            // Supply an external list of values.
            template<Int NRHS = 0, typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                cptr<T_ext> ext_values,
                const a_T alpha, cptr<X_T> X, const Int ldX,
                const b_T beta,  mptr<Y_T> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            template<typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                cref<Tensor1<T_ext,LInt>> ext_values,
                const a_T alpha, cref<Tensor1<X_T,Int>> X,
                const b_T beta,  mref<Tensor1<Y_T,Int>> Y
            ) const
            {
                if( X.Dim(0) == n && Y.Dim(0) == m )
                {
                    this->template Dot_<1>( ext_values.data(),
                        alpha, X.data(), Int(1),
                        beta,  Y.data(), Int(1),
                        Int(1)
                    );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = 0, typename T_ext, typename a_T, typename b_T, typename X_T, typename Y_T>
            void Dot(
                     cref<Tensor1<T_ext,LInt>> ext_values,
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
            
        public:
            
            static BinaryMatrixCSR IdentityMatrix(
                const Int n, const Int thread_count = 1
            )
            {
                Sparse::BinaryMatrixCSR<Int,LInt> A ( n, n, n, thread_count );
                A.Outer().iota();
                A.Inner().iota();
                
                return A;
            }
            
            
        public:
            
            static std::string ClassName()
            {
                return std::string("Sparse::BinaryMatrixCSR<")+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // BinaryMatrixCSR
        
    } // namespace Sparse
        
} // namespace Tensors
