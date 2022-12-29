#pragma once

namespace Tensors
{
    
    namespace MyBLAS
    {
        
        template<
            CBLAS_SIDE side,
            CBLAS_UPLO uplo,
            CBLAS_TRANSPOSE opA,
            CBLAS_DIAG diag,
            int N, int NRHS,
            ScalarFlag alpha_flag, ScalarFlag beta_flag,
            typename Scalar
        >
        class TRSM
        {
        public:
            
            static constexpr int MaxN    = 16;
            static constexpr int MaxNRHS = 16;
            
            TRSM() = default;
            
            ~TRSM() = default;
            
            void operator()(
                const int n, const int nrhs,
                const Scalar & alpha, const Scalar * restrict A, const int ldA,
                                      const Scalar * restrict B, const int ldB
            )
            {
                if constexpr( (1 <= N) && (N<=MaxN) && (1 <= MaxNRHS) && (NRHS<=MaxNRHS) )
                {
                    Tiny::trsm<opB,M,N,K,alpha_flag,beta_flag>(
                        side,uplo,opA,diag,
                        alpha,A,ldA,B,ldB
                    );
                }
//                else if constexpr( (1 <= N) && (N<=MaxN) )
//                {
//                    gemm_N_<N>(m,k,alpha,A,ldA,B,ldB,beta,C,ldC);
//                }
                else
                {
//                    if constexpr ( NRHS == 1 )
//                    {
//                        Tensors::BLAS_Wrappers::trsv( CblasRowMajor,
//                            CblasRowMajor,side,uplo,opA,diag,
//                            n,A,ldA,B,ldB
//                        );
//                        const CBLAS_ORDER layout,
//                        const CBLAS_UPLO uplo,
//                        const CBLAS_TRANSPOSE transa,
//                        const CBLAS_DIAG diag,
//                        const int n,
//                        const Scalar * A, const int ldA, Scalar * const x, const int incx
//                    }
//                    else
                    {
                        Tensors::BLAS_Wrappers::trsm( CblasRowMajor,
                            side,uplo,transa,diag,
                            n,nrhs,alpha,A,ldA,B,ldB
                        );
                    }
                }
            }
            
            
        protected:
           
        }; // class TRSM
        
    } // namespace MyBLAS
    
} // namespace Tensors

