#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<Layout layout, UpLo uplo, Op opA, Diag diag, typename Scalar>
        force_inline void trsv(
            const int n,
            const Scalar * A, const int ldA, Scalar * const x, const int incx
        )
        {
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dtrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_strsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_ztrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_ctrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else
            {
                eprint("trsv not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
        
} // namespace Tensors


