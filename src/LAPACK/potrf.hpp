#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<Layout layout, UpLo uplo, typename Scal, typename I0, typename I1>
        force_inline Int potrf( const I0 n_, Scal * A_, const I1 ldA_ )
        {
            static_assert(IntQ<I0>,"");
            static_assert(IntQ<I1>,"");
            
            Int n    = int_cast<Int>(n_);
            Int ldA  = int_cast<Int>(ldA_);
            Int info = 0;
            
            auto * A = to_LAPACK(A_);
            
            constexpr char flag = to_LAPACK(
                layout == Layout::ColMajor
                ? uplo
                : ( uplo == UpLo::Upper ) ? UpLo::Lower : UpLo::Upper
            );
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dpotrf
                    LAPACK_dpotrf( &flag, &n, A, &ldA, &info );
                #else
                    dpotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_spotrf
                    LAPACK_spotrf( &flag, &n, A, &ldA, &info );
                #else
                    spotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zpotrf
                    LAPACK_zpotrf( &flag, &n, A, &ldA, &info );
                #else
                    zpotrf_( &flag, &n, A, &ldA, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_cpotrf
                    LAPACK_cpotrf( &flag, &n, A, &ldA, &info );
                #else
                    cpotrf_( &flag, &n, A, &ldA, &info );
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
