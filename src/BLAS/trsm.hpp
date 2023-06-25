#pragma once

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Side side, UpLo uplo, Op opA, Diag diag, typename Scal,
            typename I0, typename I1, typename I2, typename I3
        >
        force_inline void trsm(
            const I0 n_, const I1 nrhs_,
            const Scal alpha, const Scal * A, const I2 ldA_,
                                    Scal * B, const I3 ldB_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);

            int n    = int_cast<int>(n_);
            int nrhs = int_cast<int>(nrhs_);
            int ldA  = int_cast<int>(ldA_);
            int ldB  = int_cast<int>(ldB_);
            
            assert_positive(n);
            assert_positive(nrhs);
            assert_positive(ldA);
            assert_positive(ldB);
            
            
            if constexpr ( std::is_same_v<Scal,double> )
            {
                return cblas_dtrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, alpha, const_cast<Scal*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scal,float> )
            {
                return cblas_strsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, alpha, const_cast<Scal*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<double>> )
            {
                return cblas_ztrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, &alpha, const_cast<Scal*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scal,std::complex<float>> )
            {
                return cblas_ctrsm( to_BLAS(layout), to_BLAS(side), to_BLAS(uplo), to_BLAS(opA), to_BLAS(diag), n, nrhs, &alpha, const_cast<Scal*>(A), ldA, B, ldB );
            }
            else
            {
                eprint("trsm not defined for scalar type " + TypeName<Scal> );
            }
        }
        
    } // namespace BLAS

} // namespace Tensors

