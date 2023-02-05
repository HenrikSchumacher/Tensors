#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<
            Layout layout, Op opA, typename Scalar,
            typename I0, typename I1, typename I2, typename I3, typename I4
        >
        force_inline void gemv(
            const I0 m_, const I1 n_,
            const Scalar & alpha, const Scalar * A, const I2 ldA_,
                                  const Scalar * x, const I3 incx_,
            const Scalar & beta,        Scalar * y, const I4 incy_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            ASSERT_INT(I4);
            
            int m    = int_cast<int>(m_);
            int n    = int_cast<int>(n_);
            int ldA  = int_cast<int>(ldA_);
            int incx = int_cast<int>(incx_);
            int incy = int_cast<int>(incy_);
            
            assert_positive(m);
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(incx);
            assert_positive(incy);
            
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, alpha, A, ldA, x, incx, beta, y, incy );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_sgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, alpha, A, ldA, x, incx, beta, y, incy );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_zgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, &alpha, A, ldA, x, incx, &beta, y, incy );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_cgemv(
                    to_BLAS(layout), to_BLAS(opA), m, n, &alpha, A, ldA, x, incx, &beta, y, incy );
            }
            else
            {
                eprint("gemv not defined for scalar type " + TypeName<Scalar> );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors

