#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<Layout layout, Op opA, Op opB, typename Scalar>
        force_inline void gemm(
            const int m, const int n, const int k,
            const Scalar & alpha, const Scalar * A, const int ldA,
                                  const Scalar * B, const int ldB,
            const Scalar & beta,        Scalar * C, const int ldC
        )
        {
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_sgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_zgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_cgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC
                );
            }
            else
            {
                eprint("gemm not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors
