#pragma once

namespace Tensors
{

    namespace LAPACK_Wrappers
    {
        template<Layout layout, Triangular uplo, typename Scalar>
        force_inline int potrf(
            const int n,
            Scalar * A, const int ldA
        )
        {
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

