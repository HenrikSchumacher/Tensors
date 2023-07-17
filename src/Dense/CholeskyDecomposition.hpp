#pragma once

namespace Tensors
{
    namespace Dense
    {
        template<int n, typename Scal, typename Int>
        class CholeskyDecomposition
        {
        protected:
            
            static constexpr Int max_n = n;
            
            Scal U [max_n][max_n] = {{}};
            
            Scal buffer [max_n]  = {};
            
            
        public:
            
            CholeskyDecomposition() = default;
            
            template<typename T>
            CholeskyDecomposition( cptr<T> A_, const Int ldA )
            {
                Factorize( A_, ldA );
            }
            
            ~CholeskyDecomposition() = default;
            
            template<typename T>
            void Solve( cptr<T> B, const Int ldB, mptr<T> X, const Int ldX, const Int nrhs )
            {
                const Int max_rhs = 4 * ( (nrhs + static_cast<Int>(3))/ static_cast<Int>(4) );
                
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
            //            cptr<T> B, const Int ldB,
            //            mptr<T> X, const Int ldX,
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
            void solve( cptr<T> B, const Int ldB, mptr<T> X, const Int ldX, const Int nrhs )
            {
                //Goal is to solve (U^T U) X = B
                
                Scal Y[n][max_rhs] = {{}};
                
                // Step 1: Store B  in Y.
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &B[i*ldB], &Y[i][0], nrhs );
                }
                
                // Step 2: Inplace solve U^T Y = B
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < i; ++j )
                    {
                        LOOP_UNROLL_FULL
                        for( Int k = 0; k < max_rhs; ++k )
                        {
                            Y[i][k] -= U[j][i] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Scalar::Inv<Scal>( U[i][i] );
                    
                    LOOP_UNROLL_FULL
                    for( Int k = 0; k < max_rhs; ++k )
                    {
                        Y[i][k] *= U_ii_inv;
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
                            Y[i][k] -= U[i][j] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Scalar::Inv<Scal>( U[i][i] );
                    
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
                    copy_buffer( &Y[i][0], &X[i*ldX], nrhs);
                }
            }
            
            
            template<typename T>
            void solve_gen( cptr<T> B, const Int ldB, mptr<T> X, const Int ldX, const Int nrhs )
            {
                //Goal is to solve (U^T U) X = B
                
                Tiny::VectorList<n,Scal,Int> Y (nrhs);
                
                // Step 1: Store B in Y.
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &B[i*ldB], &Y[i][0], nrhs );
                }
                
                // Step 2: Inplace solve U^T Y = B
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < i; ++j )
                    {
                        for( Int k = 0; k < nrhs; ++k )
                        {
                            Y[i][k] -= U[j][i] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Scalar::Inv<Scal>( U[i][i] );
                    
                    for( Int k = 0; k < nrhs; ++k )
                    {
                        Y[i][k] *= U_ii_inv;
                    }
                }
                
                // Step 3: Inplace solve U X = Y.
                LOOP_UNROLL_FULL
                for( Int i = n; i --> 0; )
                {
                    LOOP_UNROLL_FULL
                    for( Int j = i+1; j < n; ++j )
                    {
                        for( Int k = 0; k < nrhs; ++k )
                        {
                            Y[i][k] -= U[i][j] * Y[j][k];
                        }
                    }
                    
                    const Scal U_ii_inv = Scalar::Inv<Scal>( U[i][i] );
                    
                    for( Int k = 0; k < nrhs; ++k )
                    {
                        Y[i][k] *= U_ii_inv;
                    }
                }
                
                // Step 4: Write result.
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &Y[i][0], &X[i*ldX], nrhs );
                }
                
            } // Solve_gen
            
            
            template<typename T>
            void Factorize( cptr<T> A_, const Int ldA )
            {
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &A_[ldA*i], &U[i][0], n );
                }
                
                LOOP_UNROLL_FULL
                for( Int k = 0; k < n; ++k )
                {
                    const Scal a = Scalar::Inv<Scal>( std::sqrt(U[k][k]) );
                    
                    LOOP_UNROLL_FULL
                    for( Int j = k; j < n; ++j )
                    {
                        U[k][j] *= a;
                    }
                    
                    LOOP_UNROLL_FULL
                    for( Int i = k+1; i < n; ++i )
                    {
                        LOOP_UNROLL_FULL
                        for( Int j = i; j < n; ++j )
                        {
                            U[i][j] -= U[k][i] * U[k][j];
                        }
                    }
                }
            }
            
            template<typename T>
            void WriteFactors( mptr<T> A_, const Int ldA ) const
            {
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &U[i][0], &A_[ldA*i], n );
                }
            }
            
            template<typename T>
            void ReadFactors( cptr<T> A_, const Int ldA )
            {
                LOOP_UNROLL_FULL
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &A_[ldA*i], &U[i][0], n );
                }
            }
            
        }; // class CholeskyDecomposition
        
    } // namespace Dense
    
} // namespace Tensors
