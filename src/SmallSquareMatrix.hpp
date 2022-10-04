#pragma once

namespace Tensors {
        
        template< int AmbDim, typename Real, typename Int>
        struct SmallSquareMatrix
        {
        public:
            
            using Vector_T = SmallVector<AmbDim,Real,Int>;
            
            static constexpr Real zero              = 0;
            static constexpr Real half              = 0.5;
            static constexpr Real one               = 1;
            static constexpr Real two               = 2;
            static constexpr Real three             = 3;
            static constexpr Real four              = 4;
            static constexpr Real eps               = std::numeric_limits<Real>::min();
            static constexpr Real infty             = std::numeric_limits<Real>::max();
            
            // Uses only upper triangle.
            
            Real A [AmbDim][AmbDim] = {};
            
             SmallSquareMatrix() = default;
           
            ~SmallSquareMatrix() = default;
            
            explicit SmallSquareMatrix( const Real init )
            {
                Fill(init);
            }
            
            SmallSquareMatrix( const SmallSquareMatrix & other )
            {
                *this = other;
            }
            
            Real * restrict data()
            {
                return &A[0][0];
            }
            
            const Real * restrict data() const
            {
                return &A[0][0];
            }
            
            void SetZero()
            {
                zerofy_buffer( &A[0][0], AmbDim * AmbDim );
            }
            
            void Fill( const Real init )
            {
                fill_buffer( &A[0][0], AmbDim * AmbDim, init );
            }
            
            Real & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            const Real & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            friend SmallSquareMatrix operator+( const SmallSquareMatrix & x, const SmallSquareMatrix & y )
            {
                SmallSquareMatrix z;
                for( Int i = 0; i < AmbDim; ++i )
                {
                    for( Int j = 0; j < AmbDim; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            void operator+=( const SmallSquareMatrix & B )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    for( Int j = 0; j < AmbDim; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
            }
            
            void operator*=( const SmallSquareMatrix & B )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    for( Int j = 0; j < AmbDim; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
            }
            
            SmallSquareMatrix & operator=( const SmallSquareMatrix & B )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    for( Int j = 0; j < AmbDim; ++j )
                    {
                        A[i][j] = B.A[i][j];
                    }
                }
                return *this;
            }
            
            void Dot( const Vector_T & x, Vector_T & y ) const
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    Real y_i = 0;
                    for( Int j = 0; j < i; ++j )
                    {
                        y_i += A[j][i] * x[j];
                    }
                    for( Int j = 0; j < AmbDim; ++j )
                    {
                        y_i += A[i][j] * x[j];
                    }
                    
                    y[i] = y_i;
                }
            }
        
            
            void Write( Real * target ) const
            {
                copy_buffer( &A[0][0], target, AmbDim * AmbDim );
            }
            
            void Read( Real const * const source )
            {
                copy_buffer( source, &A[0][0], AmbDim * AmbDim );
            }
            
            std::string ToString( const Int n = 16) const
            {
                std::stringstream sout;
                sout.precision(n);
                sout << "{\n";
                sout << "\t{ ";
                
                sout << A[0][0];
                for( Int j = 1; j < AmbDim; ++j )
                {
                    sout << ", " << A[0][j];
                }
                
                for( Int i = 1; i < AmbDim; ++i )
                {
                    sout << " },\n\t{ ";
                    
                    sout << A[i][0];
                    
                    for( Int j = 1; j < AmbDim; ++j )
                    {
                        sout << ", " << A[i][j];
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            

            Real Det() const
            {
                if( AmbDim == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if( AmbDim == 3 )
                {
                    return (
                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
                    );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                SmallSquareMatrix<AmbDim,Real,Int> M;
                
                M.Read(&A[0][0]);
                
                Real sign = one;

                for(Int k = 0; k < AmbDim - 1; ++k )
                {
                    //Pivot - row swap needed
                    if( M(k,k) == zero )
                    {
                        Int m = 0;
                        for( m = k + 1; m < AmbDim; ++m )
                        {
                            if( M(m,k) != zero )
                            {
                                std::swap_ranges( &M(m,0), &M(m,AmbDim), &M(k,0) );
                                sign = -sign;
                                break;
                            }
                        }

                        //No entries != 0 found in column k -> det = 0
                        if(m == AmbDim) {
                            return zero;
                        }
                    }

                    //Apply formula
                    for( Int i = k + 1; i < AmbDim; ++i )
                    {
                        for( Int j = k + 1; j < AmbDim; ++j )
                        {
                            M(i,j) = M(k,k) * M(i,j) - M(i,k) * M(k,j);
                            if(k != 0)
                            {
                                M(i,j) /= M(k-1,k-1);
                            }
                        }
                    }
                }

                return sign * M(AmbDim-1,AmbDim-1);
            }
            
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return AmbDim;
            }
            
            static std::string ClassName()
            {
                return "SmallSquareMatrix<"+std::to_string(AmbDim)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
} // namespace CyclicSampler
