#pragma once

#ifndef LAPACK_dpotrf
#define LAPACK_dpotrf dpotrf_
#endif

namespace Tensors
{

    namespace LAPACK
    {
        template<Layout layout, UpLo uplo, typename Scal, typename I0, typename I1>
        force_inline int potrf( const I0 n_, Scal * A, const I1 ldA_ )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            
            int n    = int_cast<int>(n_);
            int ldA  = int_cast<int>(ldA_);
            int info = 0;
            
            char flag = to_LAPACK(
                layout == Layout::ColMajor
                ? uplo
                : ( uplo == UpLo::Upper ) ? UpLo::Lower : UpLo::Upper
            );
            
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( std::is_same_v<Scal,double> )
            {
                LAPACK_dpotrf( &flag, &n, A, &ldA, &info );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                LAPACK_spotrf( &flag, &n, A, &ldA, &info );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                LAPACK_zpotrf( &flag, &n, A, &ldA, &info );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                LAPACK_cpotrf( &flag, &n, A, &ldA, &info );
            }
            else
            {
                eprint("cpotrf not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors



//#pragma once
//
//namespace Tensors
//{
//
//    namespace LAPACK
//    {
//        template<Layout layout, UpLo uplo, typename Scal, typename I0, typename I1>
//        force_inline int potrf( const I0 n_, Scal * A, const I1 ldA_ )
//        {
//            ASSERT_INT(I0);
//            ASSERT_INT(I1);
//
//            int n    = int_cast<int>(n_);
//            int ldA  = int_cast<int>(ldA_);
//
//            assert_positive(n);
//            assert_positive(ldA);
//
//            if constexpr ( std::is_same_v<Scal,double> )
//            {
//                return LAPACKE_dpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
//            }
//            else if constexpr ( std::is_same_v<Scal,float> )
//            {
//                return LAPACKE_spotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
//            }
//            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
//            {
//                return LAPACKE_zpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
//            }
//            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
//            {
//                return LAPACKE_cpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
//            }
//            else
//            {
//                eprint("hetrf not defined for scalar type " + TypeName<Scal> );
//            }
//
//        }
//
//    } // namespace LAPACK
//
//} // namespace Tensors
//
