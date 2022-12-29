#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<typename Scalar>
        force_inline void gemm(
            const CBLAS_ORDER layout,
            const CBLAS_TRANSPOSE transA,
            const CBLAS_TRANSPOSE transB,
            const int m, const int n, const int k,
            const Scalar & alpha, const Scalar * A, const int ldA,
                                  const Scalar * B, const int ldB,
            const Scalar & beta,        Scalar * C, const int ldC
        )
        {
//            if( n == 1 )
//            {
//                gemv( layout, transA, m, k, alpha, A, ldA, B, ldB, beta, C, ldC );
//                return;
//            }
            
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dgemm( layout, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_sgemm( layout, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_zgemm( layout, transA, transB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_cgemm( layout, transA, transB, m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC );
            }
            else
            {
                eprint("gemm not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors
