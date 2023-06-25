#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opA, Op opB, typename Scal,
            typename I0, typename I1, typename I2, typename I3, typename I4, typename I5
        >
        force_inline void gemm(
            const I0 m_, const I1 n_, const I2 k_,
            const Scal & alpha, const Scal * A, const I3 ldA_,
                                const Scal * B, const I4 ldB_,
            const Scal & beta,        Scal * C, const I5 ldC_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            ASSERT_INT(I4);
            ASSERT_INT(I5);
            
            int m    = int_cast<int>(m_);
            int n    = int_cast<int>(n_);
            int k    = int_cast<int>(k_);
            int ldA  = int_cast<int>(ldA_);
            int ldB  = int_cast<int>(ldB_);
            int ldC  = int_cast<int>(ldC_);
            
            assert_positive(m);
            assert_positive(n);
            assert_positive(k);
            assert_positive(ldA);
            assert_positive(ldB);
            assert_positive(ldC);

            if constexpr ( std::is_same_v<Scal,double> )
            {
                return cblas_dgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                return cblas_sgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                return cblas_zgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC
                );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                return cblas_cgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, &alpha, A, ldA, B, ldB, &beta, C, ldC
                );
            }
            else
            {
                eprint("gemm not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors
