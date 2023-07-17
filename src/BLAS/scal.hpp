#pragma once

namespace Tensors
{

    namespace BLAS
    {
        template<typename Scal, typename I0, typename I1>
        force_inline void scal(
            const I0 n_, cref<Scal> alpha, mptr<Scal> x, const I1 inc_x_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            
            int n      = int_cast<int>(n_);
            int inc_x  = int_cast<int>(inc_x_);
            
            assert_positive(n);
            assert_positive(inc_x);

            if constexpr ( std::is_same_v<Scal,double> )
            {
                return cblas_dscal( n, alpha, x, inc_x );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                return cblas_dscal( n, alpha, x, inc_x );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                return cblas_cscal( n, &alpha, x, inc_x );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                return cblas_zscal( n, &alpha, x, inc_x );
            }
            else
            {
                eprint("scal not defined for scalar type " + TypeName<Scal> );
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors


