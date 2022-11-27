#pragma once

namespace Tensors
{
    template<int n, typename Scalar, typename Int>
    class LUDecomposition
    {
    protected:

        static constexpr Int max_n = n;
        
        Scalar LU [max_n][max_n] = {{}};
        Int    p  [max_n]      = {};

        Scalar buffer [max_n]  = {};
        
        
    public:
        
        LUDecomposition() = default;
        
        template<typename T>
        LUDecomposition( const T * restrict const A_, const Int ldA )
        {
            Factorize( A_, ldA );
        }
        
        ~LUDecomposition() = default;
        
        template<typename T>
        void Solve(
            const T * restrict B, const Int ldB,
                  T * restrict X, const Int ldX,
            const Int nrhs
        )
        {
            const Int max_rhs = 4 * ( (nrhs + static_cast<Int>(3))/ static_cast<Int>(4) );

//            dump(nrhs);
//            dump(max_rhs);

            switch ( max_rhs )
            {
                case 0:
                {
                    return;
                }
                case 4:
                {
                    solve<4>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 8:
                {
                    solve<8>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 12:
                {
                    solve<12>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 16:
                {
                    solve<16>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 20:
                {
                    solve<20>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 24:
                {
                    solve<24>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 28:
                {
                    solve<28>( B, ldB, X, ldX, nrhs );
                    return;
                }
                case 32:
                {
                    solve<32>( B, ldB, X, ldX, nrhs );
                    return;
                }
                default:
                {
                    solve_gen( B, ldB, X, ldX, nrhs );
                    return;
                }
            }
        }
        
//        template<typename T>
//        void Solve(
//            const T * restrict B, const Int ldB,
//                  T * restrict X, const Int ldX,
//            const Int nrhs
//        )
//        {
//            switch ( nrhs )
//            {
//                case 1:
//                {
//                    solve<1>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 2:
//                {
//                    solve<2>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 3:
//                {
//                    solve<3>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 4:
//                {
//                    solve<4>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 5:
//                {
//                    solve<5>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 6:
//                {
//                    solve<6>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 7:
//                {
//                    solve<7>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 8:
//                {
//                    solve<8>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 9:
//                {
//                    solve<12>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 10:
//                {
//                    solve<12>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 11:
//                {
//                    solve<12>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 12:
//                {
//                    solve<12>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 13:
//                {
//                    solve<16>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 14:
//                {
//                    solve<16>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 15:
//                {
//                    solve<16>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 16:
//                {
//                    solve<16>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 17:
//                {
//                    solve<20>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 18:
//                {
//                    solve<20>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 19:
//                {
//                    solve<20>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 20:
//                {
//                    solve<20>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 21:
//                {
//                    solve<24>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 22:
//                {
//                    solve<24>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 23:
//                {
//                    solve<24>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 24:
//                {
//                    solve<24>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 25:
//                {
//                    solve<28>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 26:
//                {
//                    solve<28>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 27:
//                {
//                    solve<28>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 28:
//                {
//                    solve<28>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 29:
//                {
//                    solve<32>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 30:
//                {
//                    solve<32>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 31:
//                {
//                    solve<32>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                case 32:
//                {
//                    solve<32>( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//                default:
//                {
//                    solve_gen( B, ldB, X, ldX, nrhs );
//                    return;
//                }
//            }
//        }
        
        template<Int max_rhs, typename T>
        void solve(
            const T * restrict B, const Int ldB,
                  T * restrict X, const Int ldX,
            const Int nrhs
        )
        {
            //Goal is to solve (L U) X = B

            Scalar Y[n][max_rhs] = {{}};
            
            // Step 1: Permute B and store it in Y.
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &B[p[i]*ldB], &Y[i][0], nrhs );
            }
            
            // Step 2: Inplace solve L Y = B
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                LOOP_UNROLL_FULL
                for( Int j = 0; j < i; ++j )
                {
                    LOOP_UNROLL_FULL
                    for( Int k = 0; k < max_rhs; ++k )
                    {
                        Y[i][k] -= LU[i][j] * Y[j][k];
                    }
                }
            }
            
            // Step 3: Inplace solve U X = Y.
            LOOP_UNROLL_FULL
            for( Int i = n; i --> 0; )
            {
                LOOP_UNROLL_FULL
                for( Int j = i+1; j < n; ++j )
                {
                    LOOP_UNROLL_FULL
                    for( Int k = 0; k < max_rhs; ++k )
                    {
                        Y[i][k] -= LU[i][j] * Y[j][k];
                    }
                }

                const Scalar U_ii_inv = static_cast<Scalar>(1) / LU[i][i];

                LOOP_UNROLL_FULL
                for( Int k = 0; k < max_rhs; ++k )
                {
                    Y[i][k] *= U_ii_inv;
                }
            }
            
            // Step 4: Write result.
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &Y[i][0],  &X[i*ldX], nrhs);
            }
        }
        
        
        template<typename T>
        void solve_gen(
            const T * restrict B, const Int ldB,
                  T * restrict X, const Int ldX,
            const Int nrhs
        )
        {
            //Goal is to solve (L U) X = B

            Small::VectorList<n,Scalar,Int> Y (nrhs);

            // Step 1: Permute B and store it in Y.
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &B[p[i]*ldB], &Y[i][0], nrhs );
            }
            
            // Step 2: Inplace solve L Y = B
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                LOOP_UNROLL_FULL
                for( Int j = 0; j < i; ++j )
                {
                    for( Int k = 0; k < nrhs; ++k )
                    {
                        Y[i][k] -= LU[i][j] * Y[j][k];
                    }
                }
            }
            
            // Step 3: Inplace solve U X = Y.
            LOOP_UNROLL_FULL
            for( Int i = n; i --> 0; )
            {
                for( Int j = i+1; j < n; ++j )
                {
                    for( Int k = 0; k < nrhs; ++k )
                    {
                        Y[i][k] -= LU[i][j] * Y[j][k];
                    }
                }

                const Scalar U_ii_inv = static_cast<Scalar>(1) / LU[i][i];

                for( Int k = 0; k < nrhs; ++k )
                {
                    Y[i][k] *= U_ii_inv;
                }
            }
            
            // Step 4: Write result.
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &Y[i][0], &X[i*ldX], nrhs );
            }
            
        } // Solve_gen
        
        
        template<typename T>
        void Factorize( const T * restrict const A_, const Int ldA )
        {
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &A_[ldA*i], &LU[i][0], n );
            }
            
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                p[i] = i;
            }
            
            LOOP_UNROLL_FULL
            for( Int k = 0; k < n-1; ++k )
            {
                // Find pivot index r = argmax |A[i][k]|, i = k,...,n.
                Int r = k;
                
                Scalar a = std::abs(LU[k][k]);
                LOOP_UNROLL_FULL
                for( Int i = k+1; i < n; ++i )
                {
                    const Scalar b = std::abs(LU[i][k]);
                    
                    if( b > a )
                    {
                        a = b;
                        r = i;
                    }
                }
                
                // Swap rows;
                if( r != k )
                {
                    std::swap( p[r], p[k] );
                    std::swap_ranges( &LU[r][0], &LU[r+1][0], &LU[k][0] );
                }
                
                const Scalar LU_kk_inv = static_cast<Scalar>(1) / LU[k][k];
                
                LOOP_UNROLL_FULL
                for( Int i = k+1; i < n; ++i )
                {
                    LU[i][k] *= LU_kk_inv;
                    
                    LOOP_UNROLL_FULL
                    for( Int j = k+1; j < n; ++j )
                    {
                        LU[i][j] -= LU[i][k] * LU[k][j];
                    }
                }
            }
        }
        
        template<typename T>
        void WriteFactors( T * restrict const A_, const Int ldA ) const
        {
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &LU[i][0], &A_[ldA*i], n );
            }
        }
        
        template<typename I>
        void WritePermutation( I * restrict const p_ ) const
        {
            copy_cast_buffer( &p[0], p_, n );
        }

        template<typename T>
        void ReadFactors( const T * restrict const A_, const Int ldA )
        {
            LOOP_UNROLL_FULL
            for( Int i = 0; i < n; ++i )
            {
                copy_cast_buffer( &A_[ldA*i], &LU[i][0], n );
            }
        }
        
        template<typename I>
        void ReadPermutation( const I * restrict const p_ )
        {
            copy_cast_buffer( p_, &p[0], n );
        }
        
    }; // class LUDecomposition
    
    
} // namespace Tensors
