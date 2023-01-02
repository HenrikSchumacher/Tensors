#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<Layout layout, Op opA, typename Scalar>
        force_inline void gemv(
            const int m, const int n,
            const Scalar & alpha, const Scalar * A, const int ldA,
                                  const Scalar * x, const int incx,
            const Scalar & beta,        Scalar * y, const int incy
        )
        {
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
                eprint("gemv not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors

