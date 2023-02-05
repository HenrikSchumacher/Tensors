#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        template<
            Layout layout, UpLo uplo, Op opA, Diag diag, typename Scal,
            typename I0, typename I1, typename I2
        >
        force_inline void trsv(
            const I0 n_,
            const Scalar * A, const I1 ldA_, Scalar * const x, const I2 incx_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            
            int n    = int_cast<int>(n_);
            int ldA  = int_cast<int>(ldA_);
            int incx = int_cast<int>(incx_);
            
            assert_positive(n);
            assert_positive(ldA);
            assert_positive(incx);
            
            if constexpr ( std::is_same_v<Scal,double> )
            {
                return cblas_dtrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                return cblas_strsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                return cblas_ztrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                return cblas_ctrsv( to_BLAS(layout), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, const_cast<Scalar*>(A), ldA, x, incx );
            }
            else
            {
                eprint("trsv not defined for scalar type " + TypeName<Scalar> );
            }
            
        }
        
    } // namespace BLAS_Wrappers
        
} // namespace Tensors


