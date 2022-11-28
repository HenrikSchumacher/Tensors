#pragma once

namespace Tensors
{
    namespace Small
    {
        template< int n_, typename Scalar_, typename Int_>
        struct SquareMatrix
        {
        public:
            
            using Scalar = Scalar_;
            using Real   = typename ScalarTraits<Scalar_>::RealType;
            using Int    = Int_;
            
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
            
            // Uses only upper triangle.
            
            Scalar A [n][n];
            
            SquareMatrix() = default;
            
            ~SquareMatrix() = default;
            
            explicit SquareMatrix( const Scalar init )
            :   A {{init}}
            {}
            
            SquareMatrix( const SquareMatrix & other )
            {
                *this = other;
            }
            
            Scalar * restrict data()
            {
                return &A[0][0];
            }
            
            const Scalar * restrict data() const
            {
                return &A[0][0];
            }
            
            void SetZero()
            {
                zerofy_buffer( &A[0][0], n * n );
            }
            
            void SetIdentity()
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? one : zero;
                    }
                }
            }
            
            void Fill( const Scalar init )
            {
                fill_buffer( &A[0][0], n * n, init );
            }
            
            Scalar & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            const Scalar & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            friend SquareMatrix operator+( const SquareMatrix & x, const SquareMatrix & y )
            {
                SquareMatrix z;
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            void operator+=( const SquareMatrix & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
            }
            
            void operator*=( const SquareMatrix & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
            }
            
            SquareMatrix & operator=( const SquareMatrix & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = B.A[i][j];
                    }
                }
                return *this;
            }
            
            void Dot( const Vector_T & x, Vector_T & y ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    Scalar y_i = 0;
                    for( Int j = 0; j < i; ++j )
                    {
                        y_i += A[j][i] * x[j];
                    }
                    for( Int j = 0; j < n; ++j )
                    {
                        y_i += A[i][j] * x[j];
                    }
                    
                    y[i] = y_i;
                }
            }
            
            
            void Write( Scalar * target ) const
            {
                copy_buffer( &A[0][0], target, n * n );
            }
            
            void Read( Scalar const * const source )
            {
                copy_buffer( source, &A[0][0], n * n );
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
            
            
            Scalar Det() const
            {
                if( n == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if( n == 3 )
                {
                    return (
                            A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
                            - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
                            );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                SquareMatrix<n,Scalar,Int> M;
                
                M.Read(&A[0][0]);
                
                Scalar sign = one;
                
                for(Int k = 0; k < n - 1; ++k )
                {
                    //Pivot - row swap needed
                    if( M(k,k) == zero )
                    {
                        Int m = 0;
                        for( m = k + 1; m < n; ++m )
                        {
                            if( M(m,k) != zero )
                            {
                                std::swap_ranges( &M(m,0), &M(m,n), &M(k,0) );
                                sign = -sign;
                                break;
                            }
                        }
                        
                        //No entries != 0 found in column k -> det = 0
                        if(m == n) {
                            return zero;
                        }
                    }
                    
                    //Apply formula
                    for( Int i = k + 1; i < n; ++i )
                    {
                        for( Int j = k + 1; j < n; ++j )
                        {
                            M(i,j) = M(k,k) * M(i,j) - M(i,k) * M(k,j);
                            if(k != 0)
                            {
                                M(i,j) /= M(k-1,k-1);
                            }
                        }
                    }
                }
                
                return sign * M(n-1,n-1);
            }
            
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "SquareMatrix<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
    } // namespace Small
        
} // namespace Tensors
