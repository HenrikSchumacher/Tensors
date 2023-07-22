#pragma once

namespace Tensors
{
    namespace BLAS
    {
        
        template<
            Layout layout, UpLo uplo, typename Scal,
            typename I0, typename I2, typename I3
        >
        
        force_inline void her(
            const I0 n_,
            cref<Scalar::Real<Scal>> alpha, cptr<Scal> x, const I2 incx_,
                                            mptr<Scal> A, const I3 ldA_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            
            int n    = int_cast<int>(n_);
            int incx = int_cast<int>(incx_);
            int ldA  = int_cast<int>(ldA_);
            
            assert_positive(n);
            assert_positive(incx);
            assert_positive(ldA);
            
            
            
            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dsyr( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_ssyr( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_zher( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cher( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else
            {
                eprint("her not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


