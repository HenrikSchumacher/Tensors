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
            cref<Scalar::Real<Scal>> alpha, cptr<Scal> x_, const I2 incx_,
                                            mptr<Scal> A_, const I3 ldA_
        )
        {
            static_assert(IntQ<I0>,"");
            static_assert(IntQ<I2>,"");
            static_assert(IntQ<I3>,"");
            
            Int n    = int_cast<Int>(n_);
            Int incx = int_cast<Int>(incx_);
            Int ldA  = int_cast<Int>(ldA_);
            
            auto * x = to_BLAS(x_);
            auto * A = to_BLAS(A_);
            
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
                static_assert(Tools::DependentFalse<Scal>,"");
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


