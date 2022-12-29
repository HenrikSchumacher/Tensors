#pragma once

namespace Tensors
{
    namespace BLAS_Wrappers
    {
        
        template<typename Scalar>
        force_inline void trsm(
            const CBLAS_ORDER layout,
            const CBLAS_SIDE side,
            const CBLAS_UPLO uplo,
            const CBLAS_TRANSPOSE transa,
            const CBLAS_DIAG diag,
            const int n, const int nrhs,
            const Scalar alpha, const Scalar * A, const int ldA, Scalar * B, const int ldB
        )
        {
            if constexpr ( std::is_same_v<Scalar,double> )
            {
                return cblas_dtrsm( layout, side, uplo, transa, diag, n, nrhs, alpha, const_cast<Scalar*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scalar,float> )
            {
                return cblas_strsm( layout, side, uplo, transa, diag, n, nrhs, alpha, const_cast<Scalar*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<double>> )
            {
                return cblas_ztrsm( layout, side, uplo, transa, diag, n, nrhs, &alpha, const_cast<Scalar*>(A), ldA, B, ldB );
            }
            else if constexpr ( std::is_same_v<Scalar,std::complex<float>> )
            {
                return cblas_ctrsm( layout, side, uplo, transa, diag, n, nrhs, &alpha, const_cast<Scalar*>(A), ldA, B, ldB );
            }
            else
            {
                eprint("trsm not defined for scalar type " + TypeName<Scalar>::Get() );
            }
        }
        
    } // namespace BLAS_Wrappers
    
    
    template<int n, int nrhs,
        CBLAS_UPLO uplo,
        CBLAS_DIAG diag,
        typename Scalar
    >
    force_inline void TriangularSolve(
        const Scalar * restrict const A,
              Scalar * restrict const B
    )
    {
        if constexpr ( uplo == CblasUpper )
        {
            // Upper triangular back substitution

            for( int i = n; i --> 0; )
            {
                Scalar * restrict const B_i = &B[nrhs*i];

                for( int j = i+1; j < n; ++j )
                {
                    Scalar * restrict const B_j = &B[nrhs*j];

                    const Scalar A_ij = A[n*i+j];

                    for( int k = 0; k < nrhs; ++k )
                    {
                        B_i[k] -= A_ij * B_j[k];
                    }
                }

                if constexpr (diag == CblasNonUnit )
                {
                    scale_buffer<nrhs>( static_cast<Scalar>(1) / A[(n+1)*i], B_i );
                }
            }
        }
        else if constexpr ( uplo == CblasLower )
        {
            // Lower triangular back substitution
            for( int i = 0; i < n; ++i )
            {
                Scalar * restrict const B_i = &B[nrhs*i];

                for( int j = 0; j < i; ++j )
                {
                    Scalar * restrict const B_j = &B[nrhs*j];

                    const Scalar A_ij = A[n*i+j];

                    for( int k = 0; k < nrhs; ++k )
                    {
                        B_i[k] -= A_ij * B_j[k];
                    }
                }

                if constexpr (diag == CblasNonUnit )
                {
                    scale_buffer<nrhs>( static_cast<Scalar>(1) / A[(n+1)*i], B_i );
                }
            }
        }
    }
    
    namespace TriangularSolveDetails
    {
        constexpr int MaxN    = 16;
        constexpr int MaxNRHS = 16;
        
        
        template<int n_lo, int n_hi, int nrhs, CBLAS_UPLO uplo, CBLAS_DIAG diag, typename Scalar>
        force_inline void Search(
            const int n,
            const Scalar * restrict const A,
                  Scalar * restrict const B
        )
        {
            constexpr int n_mid = n_lo + (n_hi - n_lo)/2;
            
            if( n == n_mid )
            {
                TriangularSolve<n_mid,nrhs,uplo,diag>(A,B);
            }
            else if( n < n_mid )
            {
                Search<n_lo,n_mid-1,nrhs,uplo,diag>(n,A,B);
            }
            else
            {
                Search<n_mid+1,n_hi,nrhs,uplo,diag>(n,A,B);
            }
        }
        
        template<int nrhs_lo, int nrhs_hi, CBLAS_UPLO uplo, CBLAS_DIAG diag, typename Scalar>
        force_inline void Search(
            const int n,
            const int nrhs,
            const Scalar * restrict const A,
            Scalar * restrict const B
        )
        {
            constexpr int nrhs_mid = nrhs_lo + (nrhs_hi - nrhs_lo)/2;
            
            if( nrhs == nrhs_mid )
            {
                if( n == 1 )
                {
                    TriangularSolve<1,nrhs_lo,uplo,diag>(A,B);
                }
                else
                {
                    Search<2,TriangularSolveDetails::MaxN,nrhs_mid,uplo,diag>(n,A,B);
                }
            }
            else if( nrhs < nrhs_mid )
            {
                Search<nrhs_lo,nrhs_mid-1,uplo,diag>(n,nrhs,A,B);
            }
            else
            {
                Search<nrhs_mid+1,nrhs_hi,uplo,diag>(n,nrhs,A,B);
            }
        }
        
    } // namespace TriangularSolveDetails
    
    template<
        int nrhs,
        CBLAS_UPLO uplo,
        CBLAS_DIAG diag,
        typename Scalar
    >
    force_inline void TriangularSolve(
        const int n,
        const Scalar * restrict const A,
              Scalar * restrict const B
    )
    {
        using namespace TriangularSolveDetails;
        
        // Solves A * X == B for X, where A is a triangular matrix.
        // Stores the result in B.
        
        if( n == 1 )
        {
            TriangularSolve<1,nrhs,uplo,diag>(A,B);
        }
        else if( (1 < n) && (n <= MaxN) )
        {
            Search<2,MaxN,nrhs,uplo,diag>(n,A,B);
        }
        else
        {
            if constexpr ( uplo == CblasUpper )
            {
                // Upper triangular back substitution
                for( int i = n; i --> 0; )
                {
                    Scalar * restrict const B_i = &B[nrhs*i];

                    for( int j = i+1; j < n; ++j )
                    {
                        Scalar * restrict const B_j = &B[nrhs*j];

                        const Scalar A_ij = A[n*i+j];

                        for( int k = 0; k < nrhs; ++k )
                        {
                            B_i[k] -= A_ij * B_j[k];
                        }
                    }

                    if constexpr (diag == CblasNonUnit )
                    {
                        scale_buffer<nrhs>( static_cast<Scalar>(1) / A[(n+1)*i], B_i );
                    }
                }
            }
            else if constexpr ( uplo == CblasLower )
            {
                // Lower triangular back substitution
                for( int i = 0; i < n; ++i )
                {
                    Scalar * restrict const B_i = &B[nrhs*i];
                    
                    for( int j = 0; j < i; ++j )
                    {
                        Scalar * restrict const B_j = &B[nrhs*j];
                        
                        const Scalar A_ij = A[n*i+j];
                        
                        for( int k = 0; k < nrhs; ++k )
                        {
                            B_i[k] -= A_ij * B_j[k];
                        }
                    }
                    
                    if constexpr (diag == CblasNonUnit )
                    {
                        scale_buffer<nrhs>( static_cast<Scalar>(1) / A[(n+1)*i], B_i );
                    }
                }
            }
        }
    }
    
    template<
        CBLAS_UPLO uplo,
        CBLAS_DIAG diag,
        typename Scalar
    >
    force_inline void TriangularSolve(
        const int n,
        const int nrhs,
        const Scalar * restrict const A,
              Scalar * restrict const B
    )
    {
        using namespace TriangularSolveDetails;
        
        // Solves A * X == B for X, where A is a triangular matrix.
        // Stores the result in B.
        
        if( nrhs == 1 )
        {
            if( n == 1 )
            {
                TriangularSolve<1,1,uplo,diag>(A,B);
            }
            else if ( (1 < n) && (n <= MaxN) )
            {
                Search<2,MaxNRHS,1,uplo,diag>(n,A,B);
            }
            else
            {
                TriangularSolve<1,uplo,diag>(n,A,B);
                
//                BLAS_Wrappers::trsv(
//                    CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
//                    n, A, n, B, nrhs
//                );
            }
            
        }
        else if(
            (1 <= n) && (n <= MaxN)
            &&
            (1 < nrhs) && (nrhs <= MaxNRHS)
        )
        {
            Search<2,MaxNRHS,uplo,diag>(n, nrhs,A,B);
        }
        else
        {
            BLAS_Wrappers::trsm(
                CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                n, nrhs, static_cast<Scalar>(1), A, n, B, nrhs
            );
        }
    }

} // namespace Tensors

