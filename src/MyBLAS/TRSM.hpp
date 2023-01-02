#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        
        template<
            Side side, UpLo uplo, Op op, Diag diag,
            int N, int NRHS,
            ScalarFlag alpha_flag,
            typename Scalar
        >
        class TRSM
        {
            static_assert( side == Side::Left, "TRSM is not defined for Side::Right." );
            
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
                    Tiny::trsm<side,uplo,op,diag,N,NRHS,alpha_flag>(alpha,A,ldA,B,ldB);
                }
                else
                {
//                    if constexpr ( NRHS == 1 )
//                    {
//                        Tensors::BLAS_Wrappers::trsv<
//                            Layout::RowMajor, uplo, op, diag
//                          >( n,A,ldA,B,ldB );
//                        const CBLAS_ORDER layout,
//                        const CBLAS_UPLO uplo,
//                        const CBLAS_TRANSPOSE transa,
//                        const CBLAS_DIAG diag,
//                        const int n,
//                        const Scalar * A, const int ldA, Scalar * const x, const int incx
//                    }
//                    else
                    {
                        Tensors::BLAS_Wrappers::trsm<
                            Layout::RowMajor, side, uplo, op, diag
                        >( n,nrhs,alpha,A,ldA,B,ldB );
                    }
                }
            }
            
            
        protected:
           
        }; // class TRSM
        
    } // namespace MyBLAS
    
} // namespace Tensors

