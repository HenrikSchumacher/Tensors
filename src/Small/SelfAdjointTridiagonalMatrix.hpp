#pragma once

namespace Tensors
{
    namespace Small
    {
        template< int n_, typename Scalar_, typename Int_ >
        struct SelfAdjointTridiagonalMatrix
        {
            // Uses only upper triangle.
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
            
            
            
            Scalar A [2][n];
            
            SelfAdjointTridiagonalMatrix() = default;
            
            ~SelfAdjointTridiagonalMatrix() = default;
            
            explicit SelfAdjointTridiagonalMatrix( const Scalar init )
            :   A {{ init }}
            {}
            
            // Copy constructor
            SelfAdjointTridiagonalMatrix( const SelfAdjointTridiagonalMatrix & other )
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
                zerofy_buffer( &A[0][0], static_cast<Int>(2) * n );
            }
            
            force_inline void Fill( const Scalar init )
            {
                fill_buffer( &A[0][0], init, static_cast<Int>(2) * n );
            }
            
            force_inline Scalar & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            force_inline const Scalar & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            force_inline Scalar * operator[]( const Int i )
            {
                return A[i];
            }
            
            friend SelfAdjointTridiagonalMatrix operator+(
                const SelfAdjointTridiagonalMatrix & x,
                const SelfAdjointTridiagonalMatrix & y
            )
            {
                SelfAdjointTridiagonalMatrix z;
                for( Int i = 0; i < static_cast<Int>(2); ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            void operator+=( const SelfAdjointTridiagonalMatrix & B )
            {
                for( Int i = 0; i < static_cast<Int>(2); ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
            }
            
            void operator*=( const SelfAdjointTridiagonalMatrix & B )
            {
                for( Int i = 0; i < static_cast<Int>(2); ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
            }
            
            SelfAdjointTridiagonalMatrix & operator=( const SelfAdjointTridiagonalMatrix & B )
            {
                Read(&B.A[0][0]);
                
                return *this;
            }
            
            void Dot( const Vector_T & x, Vector_T & y ) const
            {
                if( n > 0 )
                {
                    y[0] = A[0][0] * x[0] + A[1][0] * x[1];
                }
                for( Int i = 1; i < n-1; ++ i )
                {
                    y[i] = A[1][i-1] * x[i-1] + A[0][i] * x[i] + A[1][i] * x[i+1];
                }
                if( n > 0 )
                {
                    y[n-1] = A[1][n-2] * x[n-1] + A[0][n-1] * x[n-1];
                }
            }
            
            force_inline void Write( Scalar * target ) const
            {
                copy_buffer( &A[0][0], target, static_cast<Int>(2) * n );
            }
            
            force_inline void Read( Scalar const * const source )
            {
                copy_buffer( source, &A[0][0], static_cast<Int>(2) * n );
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
                
                sout << " },\n\t{ ";
                
                sout << Tools::ToString(A[1][0],p);
                
                for( Int j = 1; j < n-1; ++j )
                {
                    sout << ", " << Tools::ToString(A[1][j],p);
                }
                sout << " }\n}";
                return sout.str();
            }
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "SelfAdjointTridiagonalMatrix<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
    } // namespace Small
    
} // namespace Tensors
