#pragma once

namespace Tensors
{

    inline void gemm(
        CBLAS_LAYOUT    layout,
        CBLAS_TRANSPOSE transA,
        CBLAS_TRANSPOSE transB,
        int m, int n, int k,
        double alpha, const double * A, int ldA,
                      const double * B, int ldB,
        double beta,        double * C, int ldC
     )
    {
        cblas_dgemm(
            layout, transA, transB,
            m, n, k,
            alpha, a, ldA,
                   B, ldB,
            beta,  C, ldC
        );
    }
    
    inline void gemm(
        CBLAS_LAYOUT    layout,
        CBLAS_TRANSPOSE transA,
        CBLAS_TRANSPOSE transB,
        int m, int n, int k,
        float alpha, const float * A, int ldA,
                     const float * B, int ldB,
        float beta,        float * C, int ldC
     )
    {
        cblas_sgemm(
            layout, transA, transB,
            m, n, k,
            alpha, A, ldA,
                   B, ldB,
            beta,  C, ldC
        );
    }

    inline void gemm(
        CBLAS_LAYOUT    layout,
        CBLAS_TRANSPOSE transA,
        CBLAS_TRANSPOSE transB,
        int m, int n, int k,
        const std::complex<double> & alpha, const std::complex<double> * A, int ldA,
                                            const std::complex<double> * B, int ldB,
        const std::complex<double> & beta,        std::complex<double> * C, int ldC
     )
    {
        cblas_zgemm(
            layout, transA, transB,
            m, n, k,
            &alpha, A, ldA,
                    B, ldB,
            &beta,  C, ldC
        );
    }
    
    inline void gemm(
        CBLAS_LAYOUT    layout,
        CBLAS_TRANSPOSE transA,
        CBLAS_TRANSPOSE transB,
        int m, int n, int k,
        const std::complex<float> & alpha, const std::complex<float> * A, int ldA,
                                           const std::complex<float> * B, int ldB,
        const std::complex<float> & beta,        std::complex<float> * C, int ldC
     )
    {
        cblas_cgemm(
            layout, transA, transB,
            m, n, k,
            &alpha, A, ldA,
                    B, ldB,
            &beta,  C, ldC
        );
    }

} // namespace Tensors
