#pragma once

namespace Tensors
{
    
    template<typename Scal, typename Int>
    void PositiveDefiniteTridiagonalBlockMatrix_LUFactorize(
        cptr<Scal> L, mptr<Scal> D, mptr<Scal> U, const Int n, const Int m
    )
    {
        // L lower diagonal
        // D diagonal
        // U upper diagonal
        
        // n -- number of block in diagonal
        // m -- each block has size m x m
        
        
        static constexpr auto layout = Layout::RowMajor;
        static constexpr auto side   = Side::Left;
        static constexpr auto diag   = Diag::NonUnit;
        static constexpr auto id     = Op::Id;
        static constexpr auto trans  = Op::Trans;
        static constexpr auto uplo   = UpLo::Lower;
        
        static constexpr Scal one = 1;
        
        const Int mm = m * m;
        
        {
            const Int j    = begin;
            
            mptr<Scal> U_curr = &U[mm * j];
            mptr<Scal> D_curr = &D[mm * j];
            
//            dpotrf_( &cuplo, &m, D + mm * j, &m, &info );
            LAPACK::potrf<layout,uplo>( m, D_curr, m );
            
            // U[j] = D[j] \ U[j];
            BLAS::trsm<layout,side,uplo,id   ,diag>( m, m, one, D_curr, m, U_curr, m );
            
            // U[j] = D[j]^T \ U[j];
            BLAS::trsm<layout,side,uplo,trans,diag>( m, m, one, D_curr, m, U_curr, m );
        }
        
        for( Int j = begin+1; j < end - 1; ++j )
        {
            cptr<Scal> L_prev = &L[mm * (j-1)];
            cptr<Scal> U_prev = &U[mm * (j-1)];
            
            mptr<Scal> U_curr = &U[mm * j];
            mptr<Scal> D_curr = &D[mm * j];
            
            
            // D[j] = D[j] - L[j-1] * U[j-1];
            BLAS::gemm<layout,id,id>( m, m, m, -one, L_prev, m, U_prev, m, one, D_curr, m );
            
//            dpotrf_( &cuplo, &m, D + mm * j, &m, &info );
            LAPACK::potrf<layout,uplo>( m, D_curr, m );
            
            // U[j] = D[j] \ U[j];
            BLAS::trsm<layout,side,uplo,id   ,diag>( m, m, one, D_curr, m, U_curr, m );
            
            // U[j] = D[j]^T \ U[j];
            BLAS::trsm<layout,side,uplo,trans,diag>( m, m, one, D_curr, m, U_curr, m );
        }
        
        {
            const Int j    = end - 1;
            
            cptr<Scal> L_prev = &U[mm * (j-1)];
            cptr<Scal> U_prev = &U[mm * (j-1)];
            mptr<Scal> D_curr = &D[mm * j];
            
            // D[j] = D[j] - L[j-1] * U[j-1];
            BLAS::gemm<layout,id,id>( m, m, m, -one, L_prev, m, U_prev, m, one, D_curr, m );
            
//            dpotrf_( &cuplo, &m, D + mm * j, &m, &info );
            LAPACK::potrf<layout,uplo>( m, D_curr, m );
        }
    }
    
    
    template<typename Scal, typename Int>
    void PositiveDefiniteTridiagonalBlockMatrix_LUSolve(
        cptr<Scal> L, cptr<Scal> D, cptr<Scal> U, const Int n, const Int m, mptr<Scal> X, bool closed = false
    )
    {
        // L, D, U contain block-tridiagonal LU factorization.
        
        // X -- on entry:  right hand side (size = n x m x d).
        // X -- on return: solution.
        
        // L lower diagonal
        // D diagonal
        // U upper diagonal
        
        // n -- number of block in diagonal
        // m -- each block has size m x m
        // d -- number of right hand sides
        
        
        static constexpr auto layout = Layout::RowMajor;
        static constexpr auto side   = Side::Left;
        static constexpr auto diag   = Diag::NonUnit;
        static constexpr auto id     = Op::Id;
        static constexpr auto trans  = Op::Trans;
        static constexpr auto uplo   = UpLo::Lower;
        
        static constexpr Scal one = 1;
        
        const Int begin = bnd;
        const Int end   = n - bnd;
        
        const Int mm = m * m;
        
        const Int md = m * d;
        
        
        if( bnd )
        {
            // x[1] = x[1] - L[0].x[0];
            BLAS::gemm<layout,id,id>(
                m, d, m, -one, &L[mm * 0], m, &[X + md * 0], d, one, &X[md * 1], d
            );

            // x[n-2] = x[n-2] - U[n-2].x[n-1];
            BLAS::gemm<layout,id,id>(
                m, d, m, -one, &U[mm * (n-2)], m, &[X + md * (n-1)], d, one, &X[md * (n-2)], d
            );
        }

        {
            const Int j = begin;
            
            cptr<Scal> D_curr = &D[mm * j];
            mptr<Scal> X_curr = &X[md * j];
            
            // x[j] = D[j] \ x[j];
            BLAS::trsm<layout,side,uplo,id   ,diag>( m, d, one, D_curr, m, X_curr, d );
        
            // x[j] = D[j]^T \ x[j];
            BLAS::trsm<layout,side,uplo,trans,diag>( m, d, one, D_curr, m, X_curr, d );
        }

        for( Int j = begin + 1; j < end; ++j )
        {
            cptr<Scal> L_prev = &L[mm * (j-1)];
            cptr<Scal> D_curr = &D[mm * j];
            
            cptr<Scal> X_prev = &X[md * (j-1)];
            mptr<Scal> X_curr = &X[md * j];
            
            // x[j] = x[j] - L[j-1].x[j-1];
            BLAS::gemm<layout,id,id>( m, d, m, -one, L_prev, m, X_prev, d, one, X_curr, d );

            // x[j] = D[j] \ x[j];
            BLAS::trsm<layout,side,uplo,id   ,diag>( m, d, one, D_curr, m, X_curr, d );

            // x[j] = D[j]^T \ x[j];
            BLAS::trsm<layout,side,uplo,trans,diag>( m, d, one, D_curr, m, X_curr, d );
        }

        // x[end-1] done already.

        for( Int j = end-1; j --> begin; )
        {
            cptr<Scal> X_next = &X[md * (j+1)];
            mptr<Scal> X_curr = &X[md * j];
            
            // x[j] = x[j] - U[j].x[j+1];
            BLAS::gemm<layout,id,id>( m, d, m, -one, &U[mm * j], m, X_next, d, one, X_curr, d );
        }
    }
    
} // namespace Tensors

