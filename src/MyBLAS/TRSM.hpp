#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        
        template<
            Side side, UpLo uplo, Op op, Diag diag,
            int N, int NRHS,
            Scalar::Flag alpha_flag,
            typename Scal
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
                const Scal & alpha, ptr<Scal> A, const int ldA,
                                    ptr<Scal> B, const int ldB
            )
            {
                if constexpr( (1 <= N) && (N<=MaxN) && (1 <= MaxNRHS) && (NRHS<=MaxNRHS) )
                {
                    Tiny::trsm<side,uplo,op,diag,N,NRHS,alpha_flag>(alpha,A,ldA,B,ldB);
                }
                else
                {
                    Tensors::BLAS::trsm<
                        Layout::RowMajor, side, uplo, op, diag
                    >( n,nrhs,alpha,A,ldA,B,ldB );
                }
            }
            
            
        protected:
           
        }; // class TRSM
        
    } // namespace MyBLAS
    
} // namespace Tensors

