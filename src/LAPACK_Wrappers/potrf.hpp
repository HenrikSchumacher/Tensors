#pragma once

namespace Tensors
{

    namespace LAPACK_Wrappers
    {
        template<Layout layout, UpLo uplo, typename Scalar, typename I0, typename I1>
        force_inline int potrf( const I0 n_, Scalar * A, const I1 ldA_ )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            
            int n    = int_cast<int>(n_);
            int ldA  = int_cast<int>(ldA_);
            
            assert_positive(n);
            assert_positive(ldA);
            
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return LAPACKE_dpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return LAPACKE_spotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return LAPACKE_zpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return LAPACKE_cpotrf( to_LAPACK(layout), to_LAPACK(uplo), n, A, ldA );
            }
            else
            {
                eprint("hetrf not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace LAPACK_Wrappers
    
} // namespace Tensors

