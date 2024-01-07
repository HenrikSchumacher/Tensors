#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<typename Scal, typename I0, typename I1, typename I2>
        [[nodiscard]] force_inline Scal dot(
            const I0 n_, cptr<Scal> x, const I1 inc_x_,
                         cptr<Scal> y, const I2 inc_y_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            
            int n      = int_cast<int>(n_);
            int inc_x  = int_cast<int>(inc_x_);
            int inc_y  = int_cast<int>(inc_y_);
            
            assert_positive(n);
            assert_positive(inc_x);
            assert_positive(inc_y);

            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_ddot( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sdot( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
#if defined(cblas_zdot)
                return cblas_zdot( n, x, inc_x, y, inc_y );
#else
                Scal result {0};
                
                cblas_zdotu_sub(
                    __LAPACK_int(n), x, __LAPACK_int(inc_x), y, __LAPACK_int(inc_y),
                    &result
                );
                
                return result;
#endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cdot( n, x, inc_x, y, inc_y );
                
#if defined(cblas_cdot)
                return cblas_cdot( n, x, inc_x, y, inc_y );
#else
                Scal result {0};
                
                cblas_cdotu_sub(
                    __LAPACK_int(n), x, __LAPACK_int(inc_x), y, __LAPACK_int(inc_y),
                    &result
                );
                
                return result;
#endif
            }
            else
            {
                eprint("dot not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


