#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        
        template<typename Scalar>
        force_inline void herk(
            const CBLAS_ORDER     layout,
            const CBLAS_UPLO      uplo,
            const CBLAS_TRANSPOSE trans,
            const int n, const int k,
            const typename ScalarTraits<Scalar>::Real & alpha, const Scalar * A, const int ldA,
            const typename ScalarTraits<Scalar>::Real & beta ,       Scalar * C, const int ldC
        )
        {
            // If trans == Cblas_NoTrans, then
            // this computes the upper or lower triangle part (depending on uplo) of
            //
            //      C := alpha* A * A^H + beta * C,
            //
            // where A is a matrix of size n x k and where C is a self-adjoint matrix of size n x n
            // (of which we assume only that the uplo part is defined).

            // If trans == CblasConjTrans, then
            // this computes the upper or lower triangle part of
            //
            //      C := alpha* A^H * A + beta * C,
            //
            // where A is now a matrix of size k x n and C is again a self-adjoint matrix of size n x n.
            
            // We also overload this routine to work with reals scalars (float and double)
            // and redirect the work to ssyrk and dsyrk, respectively.
            
            // Moreover, we graciously interpret CblasTrans as CblasConjTrans.
            
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                if( trans == CblasNoTrans )
                {
                    return cblas_dsyrk( layout, uplo, CblasNoTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
                else
                {
                    return cblas_dsyrk( layout, uplo, CblasTrans  , n, k, alpha, A, ldA, beta, C, ldC );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                if( trans == CblasNoTrans )
                {
                    return cblas_ssyrk( layout, uplo, CblasNoTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
                else
                {
                    return cblas_ssyrk( layout, uplo, CblasTrans  , n, k, alpha, A, ldA, beta, C, ldC );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                if( trans == CblasNoTrans )
                {
                    return cblas_zherk( layout, uplo, CblasNoTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
                else
                {
                    return cblas_zherk( layout, uplo, CblasConjTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                if( trans == CblasNoTrans )
                {
                    return cblas_cherk( layout, uplo, CblasNoTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
                else
                {
                    return cblas_cherk( layout, uplo, CblasConjTrans, n, k, alpha, A, ldA, beta, C, ldC );
                }
            }
            else
            {
                eprint("herk not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors

