#pragma once

namespace Tensors
{

    inline void trsm(
        char side,
        char uplo,
        char transa,
        char diag,
        lapack_int m, lapack_int n
        double alpha, const double * A, lapack_int ldA,
                      const double * B, lapack_int ldB
    )
    {
        (void)LAPACKE_dtrsm(
            side, uplo, transa, diag
            m, n,
            alpha, a, ldA,
                   B, ldB
        );
    }
    
    inline void trsm(
        char side,
        char uplo,
        char transa,
        char diag,
        lapack_int m, lapack_int n
        float alpha, const float * A, lapack_int ldA,
                     const float * B, lapack_int ldB
    )
    {
        (void)LAPACKE_strsm(
            side, uplo, transa, diag
            m, n,
            alpha, a, ldA,
                   B, ldB
        );
    }
    
    inline void trsm(
        char side,
        char uplo,
        char transa,
        char diag,
        lapack_int m, lapack_int n
        const std::complex<double> & alpha, const std::complex<double> * A, lapack_int ldA,
                                            const std::complex<double> * B, lapack_int ldB
    )
    {
        (void)LAPACKE_ztrsm(
            side, uplo, transa, diag
            m, n,
            &alpha, A, ldA,
                    B, ldB
        );
    }
    
    inline void trsm(
        char side,
        char uplo,
        char transa,
        char diag,
        lapack_int m, lapack_int n
        const std::complex<float> & alpha, const std::complex<float> * A, lapack_int ldA,
                                           const std::complex<float> * B, lapack_int ldB
    )
    {
        (void)LAPACKE_ctrsm(
            side, uplo, transa, diag
            m, n,
            &alpha, A, ldA,
                    B, ldB
        );
    }
    

} // namespace Tensors

