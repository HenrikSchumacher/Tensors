#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        
        template<
            Layout layout, UpLo uplo, typename Scalar,
            typename I0, typename I2, typename I3
        >
        
        force_inline void her(
            const I0 n_,
            const typename ScalarTraits<Scalar>::Real & alpha, const Scalar * x, const I2 incx_,
                                                                     Scalar * A, const I3 ldA_
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
            
            
            
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dsyr( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_ssyr( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_zher( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_cher( to_BLAS(layout), to_BLAS(uplo), n, alpha, x, incx, A, ldA );
            }
            else
            {
                eprint("her not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors


