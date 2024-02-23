#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template< typename Scal, typename I0, typename I1, typename I2 >
        force_inline int getrf(
            const I0 m_, const I1 n_, Scal * A, const I2 ldA_, int * perm
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            
//            fill_range_buffer ( perm, 1, n_ );
            
            int m    = int_cast<int>(m_);
            int n    = int_cast<int>(n_);
            int ldA  = int_cast<int>(ldA_);
            int info = 0;
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dgetrf
                    LAPACK_dgetrf( &m, &n, A, &ldA, perm, &info );
                #else
                    dgetrf_      ( &m, &n, A, &ldA, perm, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_sgetrf
                    LAPACK_sgetrf( &m, &n, A, &ldA, perm, &info );
                #else
                    sgetrf_      ( &m, &n, A, &ldA, perm, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zgetrf
                    LAPACK_zgetrf( &m, &n, A, &ldA, perm, &info );
                #else
                    zgetrf_      ( &m, &n, A, &ldA, perm, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_cgetrf
                    LAPACK_cgetrf( &m, &n, A, &ldA, perm, &info );
                #else
                    cgetrf_      ( &m, &n, A, &ldA, perm, &info );
                #endif
            }
            else
            {
                eprint("cpotrf not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors

