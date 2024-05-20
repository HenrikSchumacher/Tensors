#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template< typename Scal, typename I0, typename I1, typename I2 >
        force_inline Int getrf(
            const I0 m_, const I1 n_, Scal * A_, const I2 ldA_, Int * perm
        )
        {
            static_assert(IntQ<I0>,"");
            static_assert(IntQ<I1>,"");
            static_assert(IntQ<I2>,"");
            
//            fill_range_buffer ( perm, 1, n_ );
            
            Int m    = int_cast<Int>(m_);
            Int n    = int_cast<Int>(n_);
            Int ldA  = int_cast<Int>(ldA_);
            Int info = 0;
            
            auto * A = to_LAPACK(A_);
            
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

