#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        template<
            Layout layout, UpLo uplo, 
            typename Scal, typename I1, typename I2, typename I3
        >
        force_inline Int hetrf(
            const I1 n_, Scal * A_, const I2 ldA_,
            Scal * work_, const I3 lwork_,
            Int * perm
        )
        {
            static_assert(IntQ<I1>,"");
            static_assert(IntQ<I2>,"");
            static_assert(IntQ<I3>,"");
            
            Int n       = int_cast<Int>(n_);
            Int ldA     = int_cast<Int>(ldA_);
            Int lwork   = int_cast<Int>(lwork_);
            Int info    = 0;
            
            auto * A    = to_LAPACK(A_);
            auto * work = to_LAPACK(work_);
            
            constexpr char flag = to_LAPACK(
                layout == Layout::ColMajor ? uplo : Transpose(uplo)
            );
            
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( SameQ<Scal,double> )
            {
                #if defined LAPACK_dsytrf
                    LAPACK_dsytrf( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    dsytrf_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                #if defined LAPACK_ssytrf
                    LAPACK_ssytrf( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    ssytrf_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                #if defined LAPACK_zhetrf
                    LAPACK_zhetrf( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    zhetrf_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                #if defined LAPACK_chetrf
                    LAPACK_chetrf( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #else
                    chetrf_      ( &flag, &n, A, &ldA, perm, work, &lwork, &info );
                #endif
            }
            else
            {
                eprint("hetrf not defined for scalar type " + TypeName<Scal> );
            }
            
            return info;
        }
        
    } // namespace LAPACK
    
} // namespace Tensors


