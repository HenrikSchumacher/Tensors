#pragma once

namespace Tensors
{
    namespace Small
    {
        template< int n_, typename Scalar_, typename Int_>
        struct SelfAdjointMatrix
        {
            // Uses only upper triangle.
        public:
            
            using Scalar   = Scalar_;
            using Real     = typename ScalarTraits<Scalar_>::RealType;
            using Int      = Int_;

            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
            static constexpr Scalar zero              = 0;
            static constexpr Scalar half              = 0.5;
            static constexpr Scalar one               = 1;
            static constexpr Scalar two               = 2;
            static constexpr Scalar three             = 3;
            static constexpr Scalar four              = 4;
            static constexpr Scalar eps               = std::numeric_limits<Scalar>::min();
            static constexpr Scalar infty             = std::numeric_limits<Scalar>::max();
            
            
            
            Scalar A [n][n];
            
            SelfAdjointMatrix() = default;
            
            ~SelfAdjointMatrix() = default;
            
            explicit SelfAdjointMatrix( const Scalar init )
            :   A {{ init }}
            {}
            
            // Copy constructor
            SelfAdjointMatrix( const SelfAdjointMatrix & other )
            {
                Read( &other.A[0][0] );
            }
            
            force_inline Scalar * data()
            {
                return &A[0][0];
            }
            
            force_inline const Scalar * data() const
            {
                return &A[0][0];
            }
            
            force_inline void SetZero()
            {
                zerofy_buffer( &A[0][0], n * n );
            }
            
            force_inline void Fill( const Scalar init )
            {
                fill_buffer( &A[0][0], init, n * n );
            }
            
            force_inline Scalar & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            force_inline const Scalar & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            force_inline Scalar * operator[]( const Int i  )
            {
                return A[i];
            }
            
            friend SelfAdjointMatrix operator+( const SelfAdjointMatrix & x, const SelfAdjointMatrix & y )
            {
                SelfAdjointMatrix z;
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            void operator+=( const SelfAdjointMatrix & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
            }
            
            void operator*=( const SelfAdjointMatrix & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
            }
            
            SelfAdjointMatrix & operator=( const SelfAdjointMatrix & B )
            {
                Read(&B.A[0][0]);
                
                return *this;
            }
            
            friend void Dot( const SelfAdjointMatrix & M, const Vector_T & x, Vector_T & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    Scalar y_i = 0;
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
            
            friend Scalar InnerProduct( const SelfAdjointMatrix & G, const Vector_T & x, const Vector_T & y )
            {
                Scalar result = 0;
                for( Int i = 0; i < n; ++i )
                {
                    Scalar z_i = 0;
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
                    const Scalar a = A[k][k] = std::sqrt( std::abs(A[k][k]) );
                    const Scalar ainv = one/a;
                    
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
                    
                    const Scalar p1 = conj(A[0][1]*A[0][1]) + conj(A[0][2])*A[0][2] + conj(A[1][2])*A[1][2];
                    
                    if( std::sqrt(p1) < eps * std::sqrt( std::abs( A[0][0]*A[0][0] + A[1][1]*A[1][1] + A[2][2]*A[2][2])) )
                    {
                        // A is diagonal
                        lambda_min = std::min( A[0][0], std::min(A[1][1],A[2][2]) );
                    }
                    else
                    {
                        const Scalar q         = ( A[0][0] + A[1][1] + A[2][2] ) / three;
                        const Scalar delta [3] = { A[0][0]-q, A[1][1]-q, A[2][2]-q } ;
                        const Scalar p2   = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2] + two*p1;
                        const Scalar p    = std::sqrt( p2 / static_cast<Scalar>(6) );
                        const Scalar pinv = one/p;
                        const Scalar b11  = delta[0] * pinv;
                        const Scalar b22  = delta[1] * pinv;
                        const Scalar b33  = delta[2] * pinv;
                        const Scalar b12  = A[0][1] * pinv;
                        const Scalar b13  = A[0][2] * pinv;
                        const Scalar b23  = A[1][2] * pinv;
                        
                        const Scalar r = half * (two * b12 * b23 * b13 - b11 * b23 * b23 - b12 *b12 * b33 + b11 * b22 * b33 - b13 *b13 * b22);
                        
                        
                        const Scalar phi = ( r <= -one )
                        ? ( static_cast<Scalar>(M_PI) / three )
                        : ( ( r >= one ) ? zero : acos(r) / three );
                        
                        // The eigenvalues are ordered this way: eig2 <= eig1 <= eig0.
                        
                        //                    Scalar eig0 = q + two * p * cos( phi );
                        //                    Scalar eig2 = q + two * p * cos( phi + two * M_PI/ three );
                        //                    Scalar eig1 = three * q - eig0 - eig2;
                        
                        lambda_min = q + two * p * cos( phi + two * M_PI/ three );
                    }
                    
                    return lambda_min;
                }
                
                if constexpr ( n > 3 )
                {
                    using Matrix_T = Eigen::Matrix<Scalar,n,n>;
                    
                    Matrix_T Sigma (&A[0][0]);
                    
                    Eigen::SelfAdjointEigenSolver<Matrix_T> eigs;
                    
                    eigs.compute(Sigma);
                    
                    return eigs.eigenvalues()[0];
                }
                
                return 0;
            }
            
            
            void HessenbergDecomposition(
                SquareMatrix                <n,Scalar,Int> & U,
                SelfAdjointTridiagonalMatrix<n,Scalar,Int> & T
            ) const
            {
                // Computes a unitary matrix U and and a self-adjoint tridiagonal matrix  T such that U^H * T * U = A.
                SelfAdjointMatrix B ( *this );
                
                Small::Matrix <n-2,n,Scalar,Int> u (0); // vector of the Householder reflection.
                Small::Vector <n,Scalar,Int>     v (0);
                
//                Scalar u [n-2][n] = { {} }; // vectors of the Householder reflection.
//                Scalar v [n]      = {};     // some scratch space

                constexpr Real eps = std::numeric_limits<Real>::min();
                
                for( Int k = 0; k < n-2; ++k )
                {
                    for( Int i = k+1; i < n; ++i )
                    {
                        u[k][i] = conj(B[k][i]);
                    }

                    const Scalar rho = ( std::abs(u[k][k+1]) <= eps ) ? one : - u[k][k+1] / std::abs(u[k][k+1]) ;
                    
                    Real uu = 0;
                    for( Int i = k+1; i < n; ++i )
                    {
                        uu += real(conj(u[k][i]) * u[k][i]);
                    }
                    
                    Scalar u_norm = std::sqrt( uu );
                    
                    uu -= real( conj(u[k][k+1]) * u[k][k+1] );
                    u[k][k+1] -= rho * u_norm;
                    
                    uu += real( conj(u[k][k+1]) * u[k][k+1] );
                    
                    Scalar u_norm_inv = one / std::sqrt( uu );
                    
                    for( Int i = k+1; i < n; ++i )
                    {
                        u[k][i] *= u_norm_inv;
                    }

                    Scalar ubarBu = 0;

                    for( Int i = k; i < n; ++i )
                    {
                        Scalar Bu_i = 0;
                        
                        for( Int j = k+1; j < i; ++j )
                        {
                            Bu_i += conj(B[j][i]) * u[k][j];
                        }
                        
                        for( Int j = i; j < n; ++j )
                        {
                            Bu_i += B[i][j] * u[k][j];
                        }
                        
                        v[i] = Bu_i;
                        
                        ubarBu += conj(u[k][i]) * Bu_i;
                    }
                    
                    {
                        const Scalar a = four * ubarBu * u[k][k];
                        const Scalar b = two * u[k][k];
                        const Scalar c = two * v[k];
                        
                        B[k][k  ] += a * conj(u[k][k  ]) - b * conj(v[k  ]) - c * conj(u[k][k  ]);
                        B[k][k+1] += a * conj(u[k][k+1]) - b * conj(v[k+1]) - c * conj(u[k][k+1]);
                    }
                    
                    // Apply Householder reflection to both sides of B.
                    for( Int i = k+1; i < n; ++i )
                    {
                        const Scalar a = four * ubarBu * u[k][i];
                        const Scalar b = two * u[k][i];
                        const Scalar c = two * v[i];
                        
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] += a * conj(u[k][j]) - b * conj(v[j]) - c * conj(u[k][j]);
                        }
                    }
                }
            
                // Reconstruct U from the Householder vectors (reverse order to safe some flops).
                U.SetIdentity();
                {
                    const Int k = n - 3;
                    
                    for( Int i = k; i < n; ++i )
                    {
                        for( Int j = k; j < n; ++j )
                        {
                            U[i][j] -= two * u[k][i] * conj(u[k][j]);
                        }
                    }
                }
                // Apply Householder transformations from the left.
                for( Int k = n-3; k -->0 ; )
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
                
                for( Int i = 0; i < n-1; ++i )
                {
                    T.Diag(i)  = real(B[i][i]);
                    T.Upper(i) = B[i][i+1];
                }
                T.Diag(n-1)  = real(B[n-1][n-1]);
            }
            
            
            force_inline void Write( Scalar * target ) const
            {
                copy_buffer( &A[0][0], target, n * n );
            }
            
            force_inline void Read( Scalar const * const source )
            {
                copy_buffer(source, &A[0][0], n * n);
            }
            
            std::string ToString( const Int p = 16) const
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
            
            //            Scalar Det() const
            //            {
            //                if( n == 2 )
            //                {
            //                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
            //                }
            //
            //                if( n == 3 )
            //                {
            //                    return (
            //                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
            //                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
            //                    );
            //                }
            //
            //                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
            //
            //                SquareMatrix<n,Scalar,Int> M;
            //
            //                M.Read(&A[0][0]);
            //
            //                Scalar sign = one;
            //
            //                for(Int k = 0; k < n - 1; ++k )
            //                {
            //                    //Pivot - row swap needed
            //                    if( M(k,k) == zero )
            //                    {
            //                        Int m = 0;
            //                        for( m = k + 1; m < n; ++m )
            //                        {
            //                            if( M(m,k) != zero )
            //                            {
            //                                std::swap_ranges( &M(m,0), &M(m,n), &M(k,0) );
            //                                sign = -sign;
            //                                break;
            //                            }
            //                        }
            //
            //                        //No entries != 0 found in column k -> det = 0
            //                        if(m == n) {
            //                            return zero;
            //                        }
            //                    }
            //
            //                    //Apply formula
            //                    for( Int i = k + 1; i < n; ++i )
            //                    {
            //                        for( Int j = k + 1; j < n; ++j )
            //                        {
            //                            M(i,j) = M(k,k) * M(i,j) - M(i,k) * M(k,j);
            //                            if(k != 0)
            //                            {
            //                                M(i,j) /= M(k-1,k-1);
            //                            }
            //                        }
            //                    }
            //                }
            //
            //                return sign * M(n-1,n-1);
            //            }
            
            inline friend std::ostream & operator<<( std::ostream & s, const SelfAdjointMatrix & A )
            {
                s << A.ToString();
                return s;
            }
            
            void ToMatrix( SquareMatrix<n,Scalar,Int> & B ) const
            {
                B.SetZero();
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < i; ++j )
                    {
                        B[i][j] = conj(A[j][i]);
                    }
                    for( Int j = i; j < n; ++j )
                    {
                        B[i][j] = A[i][j];
                    }
                }
            }
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "SelfAdjointMatrix<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
    } // namespace Small
    
} // namespace Tensors
