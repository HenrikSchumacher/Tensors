#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
#define CLASS SelfAdjointTridiagonalMatrix
        
        template< int n_, typename Scalar_, typename Int_ >
        class CLASS
        {
            // Uses only upper triangle.
            
        public:
            
#include "Tiny_Details.hpp"
            
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
        protected:
            
            std::array<Real,n>   diag;  //the main diagonal (should actually only have real values on it.
            std::array<Scalar,n> upper; //upper diagonal
            
        
        public:

//######################################################
//##                     Memory                       ##
//######################################################

            explicit CLASS( const Scalar init )
            :   diag  { init }
            ,   upper { init }
            {}
            
            
            force_inline void SetZero()
            {
                zerofy_buffer<n  >( &diag[0]  );
                zerofy_buffer<n-1>( &upper[0] );
            }
            
//######################################################
//##                     Access                       ##
//######################################################

            
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

            
            force_inline friend CLASS operator+( const CLASS & x, const CLASS & y )
            {
                CLASS z;
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
            
            force_inline void operator+=( const CLASS & B )
            {
                add_to_buffer<n>  ( &B.diag[0],  &diag[0]       );
                add_to_buffer<n-1>( &B.upper[0], &diag.upper[0] );
            }
            
            force_inline void Dot( const Vector_T & x, Vector_T & y ) const
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
            
            std::string ToString( const int p = 16) const
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
            
            template<typename T = Scalar>
            force_inline void ToMatrix( SquareMatrix<n,T,Int> & B ) const
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
                return TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
#undef CLASS
        
    } // namespace Tiny
    
} // namespace Tensors
