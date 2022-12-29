#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<typename Scalar>
        force_inline void trsv(
            const CBLAS_ORDER layout,
            const CBLAS_UPLO uplo,
            const CBLAS_TRANSPOSE transa,
            const CBLAS_DIAG diag,
            const int n,
            const Scalar * A, const int ldA, Scalar * const x, const int incx
        )
        {
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dtrsv( layout, uplo, transa, diag, n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_strsv( layout, uplo, transa, diag, n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_ztrsv( layout, uplo, transa, diag, n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_ctrsv( layout, uplo, transa, diag, n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else
            {
                eprint("trsv not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
        
} // namespace Tensors


