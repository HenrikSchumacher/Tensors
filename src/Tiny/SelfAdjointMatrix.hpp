#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
#define CLASS SelfAdjointMatrix
        
        template< int n_, typename Scalar_, typename Int_>
        class CLASS
        {
            // Uses only upper triangle.
            
        public:
            
#include "Tiny_Details.hpp"

            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
        protected:
            
            std::array<std::array<Scalar,n>,n> A;

#include "Tiny_Details_Matrix.hpp"
#include "Tiny_Details_UpperTriangular.hpp"
            

            
//######################################################
//##                  Arithmetic                      ##
//######################################################

        public:
            
            force_inline friend void Dot( const CLASS & M, const Vector_T & x, Vector_T & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    Scalar y_i (0);
                    
                    for( Int j = 0; j < i; ++j )
                    {
                        y_i += conj(M.A[j][i]) * x.v[j];
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        y_i += M.A[i][j] * x.v[j];
                    }
                    
                    y.v[i] = y_i;
                }
            }
            
            force_inline friend Scalar InnerProduct( const CLASS & G, const Vector_T & x, const Vector_T & y )
            {
                Scalar result (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    Scalar z_i (0);
                    
                    for( Int j = 0; j < i; ++j )
                    {
                        z_i += G.A[j][i] * x.v[j];
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        z_i += conj(G.A[i][j]) * x.v[j];
                    }

                    result += conj(y.v[i]) * z_i;
                }

                return result;
            }
            
            void Cholesky()
            {
                for( Int k = 0; k < n; ++k )
                {
                    const Real a ( std::sqrt( std::abs(A[k][k]) ) );
                    
                    A[k][k] = a;
                    
                    const Real ainv ( one/a );
                    
                    for( Int j = k+1; j < n; ++j )
                    {
                        A[k][j] *= ainv;
                    }
                    
                    for( Int i = k+1; i < n; ++i )
                    {
                        for( Int j = i; j < n; ++j )
                        {
                            A[i][j] -= A[k][i] * A[k][j];
                        }
                    }
                }
            }
            
            
            void CholeskySolve( const Vector_T & b, Vector_T & x ) const
            {
                x = b;
                CholeskySolve(x);
            }
            
            void CholeskySolve( Vector_T & x ) const
            {
                //In-place solve.
                
                // Lower triangular back substitution
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        x[i] -= conj(A[j][i]) * x[j];
                    }
                    x[i] /= A[i][i];
                }
                
                // Upper triangular back substitution
                for( Int i = n-1; i > -1; --i )
                {
                    for( Int j = i+1; j < n; ++j )
                    {
                        x[i] -= A[i][j] * x[j];
                    }
                    x[i] /= A[i][i];
                }
            }
            
            Real SmallestEigenvalue() const
            {
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    Real lambda_min = half * (
                        A[0][0] + A[1][1]
                        - std::sqrt(
                            std::abs(
                                (A[0][0]-A[1][1])*(A[0][0]-A[1][1]) + four * conj(A[0][1])*A[0][1]
                            )
                        )
                    );
                    
                    return lambda_min;
                }
                
                if constexpr ( n == 3 )
                {
                    Real lambda_min;
                    
                    const Scalar p1 ( conj(A[0][1]*A[0][1]) + conj(A[0][2])*A[0][2] + conj(A[1][2])*A[1][2] );
                    
                    if( std::sqrt(p1) < eps * std::sqrt( std::abs( A[0][0]*A[0][0] + A[1][1]*A[1][1] + A[2][2]*A[2][2])) )
                    {
                        // A is diagonal
                        lambda_min = std::min( A[0][0], std::min(A[1][1],A[2][2]) );
                    }
                    else
                    {
                        const Scalar q         ( ( A[0][0] + A[1][1] + A[2][2] ) / three );
                        const Scalar delta [3] { A[0][0]-q, A[1][1]-q, A[2][2]-q } ;
                        const Scalar p2        ( delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2] + two*p1 );
                        const Scalar p    ( std::sqrt( p2 / static_cast<Scalar>(6) ) );
                        const Scalar pinv ( one/p );
                        const Scalar b11  ( delta[0] * pinv );
                        const Scalar b22  ( delta[1] * pinv );
                        const Scalar b33  ( delta[2] * pinv );
                        const Scalar b12  (  A[0][1] * pinv );
                        const Scalar b13  (  A[0][2] * pinv );
                        const Scalar b23  (  A[1][2] * pinv );
                        
                        const Scalar r (
                                half * (two * b12 * b23 * b13 - b11 * b23 * b23 - b12 *b12 * b33 + b11 * b22 * b33 - b13 *b13 * b22)
                        );
                        
                        
                        const Scalar phi (
                                ( r <= -one )
                                ? ( static_cast<Scalar>(M_PI) / three )
                                : ( ( r >= one ) ? zero : acos(r) / three )
                        );
                        
                        // The eigenvalues are ordered this way: eig2 <= eig1 <= eig0.
                        
                        //                    Scalar eig0 ( q + two * p * cos( phi ) );
                        //                    Scalar eig2 ( q + two * p * cos( phi + two * M_PI/ three ) );
                        //                    Scalar eig1 ( three * q - eig0 - eig2 );
                        
                        lambda_min = q + two * p * cos( phi + two * M_PI/ three );
                    }
                    
                    return lambda_min;
                }
                
                if constexpr ( n > 3 )
                {
                    Vector<n,Real,Int> v;
    
                    Eigenvalues(v);
                    
                    return v.Min();
                }
                
                return 0;
            }
            
            void HessenbergDecomposition(
                SquareMatrix                <n,Scalar,Int> & U,
                SelfAdjointTridiagonalMatrix<n,Real,Int>   & T
            ) const
            {
                // Computes a unitary matrix U and and a self-adjoint tridiagonal matrix T such that U . T . U^H = A.

                if constexpr ( n == 1 )
                {
                    T.Diag(0) = real(A[0][0]);
                    U[0][0]   = one;
                }
                
                if constexpr ( n == 2 )
                {
                    T.Diag(0)  = real(A[0][0]);
                    T.Diag(1)  = real(A[1][1]);
                    T.Upper(0) = std::abs(A[0][1]);
                    U[0][0] = 1;
                    U[0][1] = 0;
                    U[1][0] = 0;
                    U[1][1] = (T.Upper(0) == zero) ? one : conj(A[0][1]) / T.Upper(0);
                }
                
                if constexpr ( n > 2 )
                {
                    SquareMatrix<n, Scalar, Int> B ;
                    Write( &B[0][0] );
                    
//                    Scalar u [n-2][n]; // vectors of the Householder reflections.
//                    Scalar v [n];      // some scratch space
                    
                    Vector_T u [n-2]; // vectors of the Householder reflections.
                    Vector_T v      ; // some scratch space
                    
                    for( Int k = 0; k < n-2; ++k )
                    {
                        // Compute k-th Householder reflection vector.
                        
//                        u[k][k] = 0; // We know that u[k][0] = ... = u[k][k] = 0; we just use this implicitly!
//                        for( Int i = 0; i < k+1; ++i )
//                        {
//                            u[k][i] = 0
//                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] = conj(B[k][i]);
                        }
                        
                        Real uu = 0;
                        for( Int i = k+1; i < n; ++i )
                        {
                            uu += abs_squared(u[k][i]);
                        }
                        
                        Real u_norm = std::sqrt( uu );
                        
                        Scalar u_pivot (u[k][k+1]);
                        
                        Real abs_squared_u_pivot = abs_squared(u_pivot);
                        
                        const Scalar rho (
                            COND(
                                ScalarTraits<Scalar>::IsComplex
                                 ,
                                ( abs_squared_u_pivot <= eps_squared * uu ) ? one : -u_pivot / std::sqrt(abs_squared_u_pivot)
                                ,
                                ( u_pivot > zero ) ? -one : one
                            )
                        );
                        
                        uu -= abs_squared_u_pivot;
                        
                        u[k][k+1] -= rho * u_norm;
                        
                        uu += abs_squared(u[k][k+1]);
                        
                        Real u_norm_inv ( one / std::sqrt( uu ) );
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] *= u_norm_inv;
                        }
                        
                        
                        // Compute v_k = B . u_k and  u_k . B . u_k
                        
                        Scalar ubarBu_accumulator (0);
                        
                        {
                            const Int i = k;
                            
                            Scalar Bu_i (0);
                            
                            // We can skip this.
//                            for( Int j = k+1; j < i; ++j )
//                            {
//                                Bu_i += conj(B[j][i]) * u[k][j];
//                            }
                            
                            for( Int j = k+1; j < n; ++j ) // we implicitly use u[k][k] == 0
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            // We implicitly use u[k][k] = 0;
//                            ubarBu += conj(u[k][i]) * Bu_i;
                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            Scalar Bu_i (0);
                            
                            for( Int j = k+1; j < i; ++j )
                            {
                                Bu_i += conj(B[j][i]) * u[k][j];
                            }
                            
                            for( Int j = i; j < n; ++j )
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            ubarBu_accumulator += conj(u[k][i]) * Bu_i;
                        }
                        
                        Real ubarBu = real(ubarBu_accumulator);

                        {
                            const Int i = k;
                            const Scalar a ( (- two) * v[k] );

                            for( Int j = i+1; j < n; ++j )  // Exploit that u[k][i] = u[k][k] == 0
                            {
                                B[k][j] += a * conj(u[k][j]);
                            }
                        }
                        
                        // Apply Householder reflection to both sides of B.
                        for( Int i = k+1; i < n; ++i )
                        {
                            const Scalar a ( (four * ubarBu) * u[k][i] - two * v[i]);
                            const Scalar b ( two * u[k][i] );
                            
                            for( Int j = i; j < n; ++j )
                            {
                                B[i][j] += a * conj(u[k][j]) - b * conj(v[j]);
                            }
                        }
                    }
                                       
                    // We want a purely real tridiagonal matrix...
                    for( Int i = 0; i < n-1; ++i )
                    {
                        T.Diag(i)  = real(B[i][i]);
                        T.Upper(i) = COND( ScalarTraits<Scalar>::IsComplex, std::abs(B[i][i+1]), B[i][i+1] );
                    }
                    T.Diag(n-1)  = real(B[n-1][n-1]);
                    
                    // ... hence we put appropriate unimodular numbers on the diagonal of U.
                    if constexpr ( ScalarTraits<Scalar>::IsComplex )
                    {
                        U.SetZero();
                        U[0][0] = 1;
                        for( Int k = 1; k < n; ++k )
                        {
                            Real absb = T.Upper(k-1);
                            U[k][k]   = (absb == zero) ? one : U[k-1][k-1] * conj(B[k-1][k]) / absb;
                        }
                    }
                    else
                    {
                        U.SetIdentity();
                    }
                        
                    // Apply Householder transformations from the left (reverse order to safe some flops).
                    for( Int k = n-2; k --> 0 ; )
                    {
                        // Compute v = conj(u[k]) * U;
                        for( Int j = k+1; j < n; ++j )
                        {
                            Scalar ubarU_j = 0;
                            for( Int i = k+1; i < n; ++i )
                            {
                                ubarU_j += conj(u[k][i]) * U[i][j];
                            }
                            v[j] = ubarU_j;
                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            const Scalar a = two * u[k][i];
                            for( Int j = k+1; j < n; ++j )
                            {
                                U[i][j] -= a * v[j];
                            }
                        }
                    }
                }
            }
            
            void HessenbergDecomposition(
                SelfAdjointTridiagonalMatrix<n,Real,Int> & T
            ) const
            {
                // Computes a unitary matrix U and and a self-adjoint tridiagonal matrix T such that U . T . U^H = A.

                if constexpr ( n == 1 )
                {
                    T.Diag(0) = real(A[0][0]);
                }
                
                if constexpr ( n == 2 )
                {
                    T.Diag(0)  = real(A[0][0]);
                    T.Diag(1)  = real(A[1][1]);
                    T.Upper(0) = std::abs(A[0][1]);
                }
                
                if constexpr ( n > 2 )
                {
                    SquareMatrix<n, Scalar, Int> B ;
                    Write( &B[0][0] );
                    
//                    Scalar u [n-2][n]; // vectors of the Householder reflections.
//                    Scalar v [n];      // some scratch space
                    
                    Vector_T u [n-2]; // vectors of the Householder reflections.
                    Vector_T v      ; // some scratch space
                    
                    for( Int k = 0; k < n-2; ++k )
                    {
                        // Compute k-th Householder reflection vector.
                        
//                        u[k][k] = 0; // We know that u[k][0] = ... = u[k][k] = 0; we just use this implicitly!
//                        for( Int i = 0; i < k+1; ++i )
//                        {
//                            u[k][i] = 0
//                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] = conj(B[k][i]);
                        }
                        
                        Real uu = 0;
                        for( Int i = k+1; i < n; ++i )
                        {
                            uu += abs_squared(u[k][i]);
                        }
                        
                        Real u_norm = std::sqrt( uu );
                        
                        Scalar u_pivot (u[k][k+1]);
                        
                        Real abs_squared_u_pivot = abs_squared(u_pivot);
                        
                        const Scalar rho (
                            COND(
                                ScalarTraits<Scalar>::IsComplex
                                 ,
                                ( abs_squared_u_pivot <= eps_squared * uu ) ? one : -u_pivot / std::sqrt(abs_squared_u_pivot)
                                ,
                                ( u_pivot > zero ) ? -one : one
                            )
                        );
                        
                        uu -= abs_squared_u_pivot;
                        
                        u[k][k+1] -= rho * u_norm;
                        
                        uu += abs_squared(u[k][k+1]);
                        
                        Real u_norm_inv ( one / std::sqrt( uu ) );
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            u[k][i] *= u_norm_inv;
                        }
                        
                        
                        // Compute v_k = B . u_k and  u_k . B . u_k
                        
                        Scalar ubarBu_accumulator (0);
                        
                        {
                            const Int i = k;
                            
                            Scalar Bu_i (0);
                            
                            // We can skip this.
//                            for( Int j = k+1; j < i; ++j )
//                            {
//                                Bu_i += conj(B[j][i]) * u[k][j];
//                            }
                            
                            for( Int j = k+1; j < n; ++j ) // we implicitly use u[k][k] == 0
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            // We implicitly use u[k][k] = 0;
//                            ubarBu += conj(u[k][i]) * Bu_i;
                        }
                        
                        for( Int i = k+1; i < n; ++i )
                        {
                            Scalar Bu_i (0);
                            
                            for( Int j = k+1; j < i; ++j )
                            {
                                Bu_i += conj(B[j][i]) * u[k][j];
                            }
                            
                            for( Int j = i; j < n; ++j )
                            {
                                Bu_i += B[i][j] * u[k][j];
                            }
                            
                            v[i] = Bu_i;
                            
                            ubarBu_accumulator += conj(u[k][i]) * Bu_i;
                        }
                        
                        Real ubarBu = real(ubarBu_accumulator);

                        {
                            const Int i = k;
                            const Scalar a ( (- two) * v[k] );

                            for( Int j = i+1; j < n; ++j )  // Exploit that u[k][i] = u[k][k] == 0
                            {
                                B[k][j] += a * conj(u[k][j]);
                            }
                        }
                        
                        // Apply Householder reflection to both sides of B.
                        for( Int i = k+1; i < n; ++i )
                        {
                            const Scalar a ( (four * ubarBu) * u[k][i] - two * v[i]);
                            const Scalar b ( two * u[k][i] );
                            
                            for( Int j = i; j < n; ++j )
                            {
                                B[i][j] += a * conj(u[k][j]) - b * conj(v[j]);
                            }
                        }
                    }
                                       
                    // We want a purely real tridiagonal matrix...
                    for( Int i = 0; i < n-1; ++i )
                    {
                        T.Diag(i)  = real(B[i][i]);
                        T.Upper(i) = COND( ScalarTraits<Scalar>::IsComplex, std::abs(B[i][i+1]), B[i][i+1] );
                    }
                    T.Diag(n-1)  = real(B[n-1][n-1]);
                }
            }
            
            void Eigenvalues( Vector<n,Real,Int> & eigs, const Real tol = eps_sqrt, const Int max_iter = 8 ) const
            {
                SelfAdjointTridiagonalMatrix<n, Real, Int> T;
                
                HessenbergDecomposition(T);
                
                T.QRAlgorithm( eigs, tol, max_iter );
            }
            
            
            void Eigensystem(
                SquareMatrix<n,Scalar,Int> & U,
                Vector      <n,Real,  Int> & eigs,
                const Real tol      = eps_sqrt,
                const Int  max_iter = 8
            ) const
            {
                SelfAdjointTridiagonalMatrix<n, Real, Int> T;
                SquareMatrix<n,Scalar,Int> V;
                SquareMatrix<n,Real,  Int> Q;
                
                HessenbergDecomposition(V,T);

                T.QRAlgorithm( Q, eigs, tol, max_iter );

                //TODO: We might exploit here that V has zeroes in first row and column.
                Dot(V,Q,U);
            }
            
            std::string ToString( const int p = 16) const
            {
                std::stringstream sout;

                sout << "{\n";
                sout << "\t{ ";
                
                sout << Tools::ToString(A[0][0],p);
                for( Int j = 1; j < n; ++j )
                {
                    sout << ", " << Tools::ToString(A[0][j],p);
                }
                
                for( Int i = 1; i < n; ++i )
                {
                    sout << " },\n\t{ ";
                    
                    sout << Tools::ToString(A[i][0],p);
                    
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << Tools::ToString(A[i][j],p);
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            template<typename T = Scalar>
            void ToMatrix( SquareMatrix<n,T,Int> & B ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        B[i][j] = static_cast<T>(conj(A[j][i]));
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        B[i][j] = static_cast<T>(A[i][j]);
                    }
                }
            }
            
            SquareMatrix<n,Scalar,Int> ToMatrix() const
            {
                SquareMatrix<n,Scalar,Int> B;
                
                ToMatrix(B);
                
                return B;
            }
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
#undef CLASS
    } // namespace Tiny
    
} // namespace Tensors
