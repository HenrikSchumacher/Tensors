#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<typename Scal, typename I0, typename I1, typename I2>
        force_inline void copy(
            const I0 n_, cptr<Scal> x, const I1 inc_x_,
                         mptr<Scal> y, const I2 inc_y_
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

            if constexpr ( std::is_same_v<Scal,double> )
            {
                return cblas_dcopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                return cblas_scopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                return cblas_ccopy( n, x, inc_x, y, inc_y );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                return cblas_zcopy( n, x, inc_x, y, inc_y );
            }
            else
            {
                eprint("copy not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors



