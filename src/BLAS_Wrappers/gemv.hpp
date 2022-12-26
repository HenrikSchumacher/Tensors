#pragma once

namespace Tensors
{
    template<typename Scalar>
    force_inline void gemv(
        const CBLAS_ORDER     layout,
        const CBLAS_TRANSPOSE transA,
        const int m, const int n,
        const Scalar & alpha, const Scalar * A, const int ldA,
                              const Scalar * x, const int incx,
        const Scalar & beta,        Scalar * y, const int incy
    )
    {
        if constexpr ( std::is_same_v<Scalar,double> )
        {
            return cblas_dgemv( layout, transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
        }
        else if constexpr ( std::is_same_v<Scalar,float> )
        {
            return cblas_sgemv( layout, transA, m, n, alpha, A, ldA, x, incx, beta, y, incy );
        }
        else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
        {
            return cblas_zgemv( layout, transA, m, n, &alpha, A, ldA, x, incx, &beta, y, incy );
        }
        else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
        {
            return cblas_cgemv( layout, transA, m, n, &alpha, A, ldA, x, incx, &beta, y, incy );
        }
        else
        {
            eprint("gemv not defined for scalar type " + TypeName<Scalar>::Get() );
        }
        
    }

} // namespace Tensors

