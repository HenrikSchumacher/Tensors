#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        
        template<Layout layout, UpLo uplo, Op op, typename Scalar, typename I0, typename I1, typename I2, typename I3>
        force_inline void herk(
            const I0 n_, const I1 k_,
            const typename ScalarTraits<Scalar>::Real & alpha, const Scalar * A, const I2 ldA_,
            const typename ScalarTraits<Scalar>::Real & beta ,       Scalar * C, const I3 ldC_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            
            int n    = int_cast<int>(n_);
            int k    = int_cast<int>(k_);
            int ldA  = int_cast<int>(ldA_);
            int ldC  = int_cast<int>(ldC_);
                        
            assert_positive(n);
            assert_positive(k);
            assert_positive(ldA);
            assert_positive(ldC);
            
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
                if constexpr ( op == Op::Id )
                {
                    return cblas_dsyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_dsyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Trans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                if( op == Op::Id )
                {
                    return cblas_ssyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_ssyrk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Trans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                if( op == Op::Id )
                {
                    return cblas_zherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_zherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::ConjTrans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                if( op == Op::Id )
                {
                    return cblas_cherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::Id),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
                else
                {
                    return cblas_cherk(
                        to_BLAS(layout), to_BLAS(uplo), to_BLAS(Op::ConjTrans),
                        n, k, alpha, A, ldA, beta, C, ldC
                    );
                }
            }
            else
            {
                eprint("herk not defined for scalar type " + TypeName<Scalar>::Get() );
            }
            
        }
        
    } // namespace BLAS_Wrappers
    
} // namespace Tensors

