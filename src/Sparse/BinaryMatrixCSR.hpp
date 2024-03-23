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
            using Base_T::WellFormed;
            
            BinaryMatrixCSR()
            :   Base_T()
            {}
            
            template<typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
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
            BinaryMatrixCSR(
                const I_0 m_,
                const I_1 n_,
                const I_2 nnz_,
                const I_3 thread_count_
            )
            :   Base_T( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<LInt>(nnz_), static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_2);
                ASSERT_INT(I_3);
            }
            
            
            template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
            BinaryMatrixCSR(
                  const J_0 * const outer_,
                  const J_1 * const inner_,
                  const I_0 m_,
                  const I_1 n_,
                  const I_3 thread_count_
                  )
            :   Base_T( outer_, inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
            {
                ASSERT_INT(J_0);
                ASSERT_INT(J_1);
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
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
                    static_cast<Int>(m_),
                    static_cast<Int>(n_),
                    static_cast<Int>(thread_count_)
                )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
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
                    static_cast<Int>(m_),
                    static_cast<Int>(n_), 
                    static_cast<Int>(thread_count_)
                )
            {
                ASSERT_INT(I_0);
                ASSERT_INT(I_1);
                ASSERT_INT(I_3);
            }
            
            // Copy constructor
            BinaryMatrixCSR( const BinaryMatrixCSR & other )
            :   Base_T( other )
            {
                logprint("Copy of "+ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nn z = "+ToString(NonzeroCount()));
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
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T ( idx, jdx, entry_counts, list_count, m_, n_, final_thread_count, compressQ, symmetrize )
            {}
            
            BinaryMatrixCSR(
                const LInt nnz_,
                const Int  * const i,
                const Int  * const j,
                const Int m_,
                const Int n_,
                const Int thread_count,
                const bool compressQ   = true,
                const int  symmetrize = 0
            )
            :   Base_T ( i, j, m_, n_, thread_count, compressQ, symmetrize )
            {}
            
            BinaryMatrixCSR(
                  std::vector<Int> & idx,
                  std::vector<Int> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrize )
            {}
            
            BinaryMatrixCSR(
                  const std::vector<std::vector<Int>> & idx,
                  const std::vector<std::vector<Int>> & jdx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T( idx, jdx, m_, n_, final_thread_count, compressQ, symmetrize )
            {}
            
            BinaryMatrixCSR(
                  const std::vector<PairAggregator<Int, Int, LInt>> & idx,
                  const Int m_,
                  const Int n_,
                  const Int final_thread_count,
                  const bool compressQ   = true,
                  const int  symmetrize = 0
                  )
            :   Base_T( idx, m_, n_, final_thread_count, compressQ, symmetrize )
            {}
            
            virtual ~BinaryMatrixCSR() = default;
            
            
            BinaryMatrixCSR Transpose() const
            {
                ptic(ClassName()+"::Transpose");
                
                Tensor2<LInt,Int> counters = CreateTransposeCounters();
                
                BinaryMatrixCSR<Int,LInt> B ( n, m, outer[m], thread_count );
                
                copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );
                
                if( WellFormed() )
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
                
                ptoc(ClassName()+"::Transpose");
                
                return B;
            }
            

//#########################################################################################
//####          Permute
//#########################################################################################
            
        public:
            
            Tensor1<LInt,LInt> Permute(
                const Permutation<Int> & p,     // row    permutation
                const Permutation<Int> & q      // column permutation
            )
            {
                ptic(ClassName()+"::Permute");
                // Modifies inner and outer  accordingly; returns the permutation to be applied to the nonzero values.
                
                
                this->inner_sorted = true;
                this->diag_ptr_initialized = false;
                this->job_ptr_initialized  = false;
                this->upper_triangular_job_ptr_initialized = false;
                this->lower_triangular_job_ptr_initialized = false;
                
                Tensor1<LInt,LInt> perm = PermutePatternCSR( outer, inner, p, q, inner.Size(), true );
                
                ptoc(ClassName()+"::Permute");
                
                return perm;
            }
            
//#########################################################################################
//####          Matrix Multiplication
//#########################################################################################
            
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
            
//###########################################################################################
//####          Matrix Multiplication
//###########################################################################################
            

            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cptr<T_in>  X, const Int ldX,
                cref<S_out> beta,  mptr<T_out> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            // Assume all nonzeros are equal to 1.
            template<Int NRHS = 0, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cptr<T_in>  X,
                cref<S_out> beta,  mptr<T_out> Y,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( alpha, X, nrhs, beta, Y, nrhs, nrhs );
            }
            
            template<typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cref<Tensor1<T_in, Int>> X,
                cref<S_out> beta,  mref<Tensor1<T_out,Int>> Y
            ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m )
                {
                    this->template Dot_<1>(
                        alpha, X.data(), static_cast<Int>(1),
                        beta,  Y.data(), static_cast<Int>(1),
                        static_cast<Int>(1)
                    );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = 0, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<R_out> alpha, cref<Tensor2<T_in, Int>> X,
                cref<S_out> beta,  mref<Tensor2<T_out,Int>> Y
            ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
                {
                    const Int nrhs = X.Dimension(1);
                    
                    this->template Dot_<NRHS>( alpha, X.data(), nrhs, beta, Y.data(), nrhs, nrhs );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            
            
            // Supply an external list of values.
            template<Int NRHS = 0, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cptr<T_ext> ext_values,
                cref<R_out> alpha, cptr<T_in>  X, const Int ldX,
                cref<S_out> beta,  mptr<T_out> Y, const Int ldY,
                const Int nrhs = 1
            ) const
            {
                this->template Dot_<NRHS>( ext_values, alpha, X, ldX, beta, Y, ldY, nrhs );
            }
            
            template<typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                cref<Tensor1<T_ext,LInt>> ext_values,
                cref<R_out> alpha, cref<Tensor1<T_in, Int>> X,
                cref<S_out> beta,  mref<Tensor1<T_out,Int>> Y
            ) const
            {
                if( X.Dimension(0) == n && Y.Dimension(0) == m )
                {
                    this->template Dot_<1>( ext_values.data(),
                        alpha, X.data(), static_cast<Int>(1),
                        beta,  Y.data(), static_cast<Int>(1),
                        static_cast<Int>(1)
                    );
                }
                else
                {
                    eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
                }
            }
            
            template<Int NRHS = 0, typename T_ext, typename R_out, typename S_out, typename T_in, typename T_out>
            void Dot(
                     cref<Tensor1<T_ext,LInt>> ext_values,
                     cref<R_out> alpha, cref<Tensor2< T_in,Int>> X,
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
            
            
        public:
            
            static std::string ClassName()
            {
                return std::string("Sparse::BinaryMatrixCSR<")+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // BinaryMatrixCSR
        
    } // namespace Sparse
        
} // namespace Tensors
