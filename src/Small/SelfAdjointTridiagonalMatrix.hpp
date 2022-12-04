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
            
            static constexpr Scalar zero            = 0;
            static constexpr Scalar half            = 0.5;
            static constexpr Scalar one             = 1;
            static constexpr Scalar two             = 2;
            static constexpr Scalar three           = 3;
            static constexpr Scalar four            = 4;
            static constexpr Real eps               = std::numeric_limits<Real>::min();
            static constexpr Real infty             = std::numeric_limits<Real>::max();
            
        protected:
            
            
            Real   diag  [n];   //the main diagonal (should actually only have real values on it.
            Scalar upper [n-1]; //upper diagonal
            

        
        public:
            
            SelfAdjointTridiagonalMatrix() = default;
            
            ~SelfAdjointTridiagonalMatrix() = default;
            
            explicit SelfAdjointTridiagonalMatrix( const Scalar init )
            :   diag  { init }
            ,   upper { init }
            {}
            
            // Copy constructor
            SelfAdjointTridiagonalMatrix( const SelfAdjointTridiagonalMatrix & other )
            {
                Read( &other.A[0][0] );
            }
            
            force_inline Real & Diag( const Int i )
            {
                return diag[i];
            }
            
            force_inline const Real & Diag( const Int i ) const
            {
                return diag[i];
            }
            
            force_inline Scalar & Upper( const Int i )
            {
                return upper[i];
            }
            
            force_inline const Scalar & Upper( const Int i ) const
            {
                return upper[i];
            }
            
            force_inline Scalar Lower( const Int i )
            {
                return conj(upper[i]);
            }
            
            force_inline void SetZero()
            {
                zerofy_buffer( &diag[0] , n   );
                zerofy_buffer( &upper[0], n-1 );
            }
            
            friend SelfAdjointTridiagonalMatrix operator+(
                const SelfAdjointTridiagonalMatrix & x,
                const SelfAdjointTridiagonalMatrix & y
            )
            {
                SelfAdjointTridiagonalMatrix z;
                for( Int i = 0; i < n; ++i )
                {
                    z.diag[i] = x.diag[i] + y.diag[i];
                }
                for( Int i = 0; i < n-1; ++i )
                {
                    z.upper[i] = x.upper[i] + y.upper[i];
                }
                
                return z;
            }
            
            void operator+=( const SelfAdjointTridiagonalMatrix & B )
            {
                add_to_buffer<n>  ( B.diag,  diag       );
                add_to_buffer<n-1>( B.upper, diag.upper );
            }

            
            SelfAdjointTridiagonalMatrix & operator=( const SelfAdjointTridiagonalMatrix & B )
            {
                copy_buffer( diag, B.diag, n);
                copy_buffer( upper, B.upper, n-1);
                
                return *this;
            }
            
            void Dot( const Vector_T & x, Vector_T & y ) const
            {
                if constexpr ( n >= 1 )
                {
                    y[0] = diag[0] * x[0];
                }
                else if constexpr ( n > 1 )
                {
                    y[0] = diag[0] * x[0] + upper[0] * x[1];
                }
                
                for( Int i = 1; i < n-2; ++ i )
                {
                    y[i] = upper[i-1] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1];
                }
                
                if constexpr ( n >= 2 )
                {
                    y[n-1] = diag[n-1] * x[n-1];
                }
                else if constexpr  ( n > 2 )
                {
                    y[n-1] = upper[n-2] * x[n-2] + diag[n-1] * x[n-1];
                }
            }
            
            std::string ToString( const Int p = 16) const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\tdiag  = { ";
                
                sout << Tools::ToString(diag[0],p);
                for( Int j = 1; j < n; ++j )
                {
                    sout << ", " << Tools::ToString(diag[j],p);
                }
                
                sout << " },\n\tupper = { ";
                
                if( n > 1 )
                {
                    sout << Tools::ToString(upper[0],p);
                    
                    for( Int j = 1; j < n-1; ++j )
                    {
                        sout << ", " << Tools::ToString(upper[j],p);
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            inline friend std::ostream & operator<<( std::ostream & s, const SelfAdjointTridiagonalMatrix & A )
            {
                s << A.ToString();
                return s;
            }
            
            template<typename T = Scalar>
            void ToMatrix( SquareMatrix<n,T,Int> & B ) const
            {
                B.SetZero();
                
                for( Int i = 0; i < n-1; ++i )
                {
                    B[i][i]     = static_cast<T>(diag[i]);
                    B[i  ][i+1] = static_cast<T>(upper[i]);
                    B[i+1][i  ] = static_cast<T>(conj(upper[i]));
                }
                B[n-1][n-1] = static_cast<T>(diag[n-1]);
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
