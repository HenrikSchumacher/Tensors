#pragma once

namespace Tensors
{
    namespace Tiny
    {

#define CLASS UpperTriangularMatrix
        
        template< int n_, typename Scalar_, typename Int_>
        class CLASS
        {
            // Allocates a square array, but accesses only upper triangle.
            
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
                    Scalar y_i ( static_cast<Scalar>(0) );
                    
                    for( Int j = i; j < n; ++j )
                    {
                        y_i += M.A[i][j] * x.v[j];
                    }
                    
                    y.v[i] = y_i;
                }
            }
            
            
            force_inline Scalar Det() const
            {
                Scalar det = A[0][0];
                
                for( Int i = 1; i < n; ++i )
                {
                    det *= A[i][i];
                }
                
                return det;
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
                    zerofy_buffer(&B[i][i], n-i);
                    
                    for( Int j = i; j < n; ++j )
                    {
                        B[i][j] = static_cast<T>(A[i][j]);
                    }
                }
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
