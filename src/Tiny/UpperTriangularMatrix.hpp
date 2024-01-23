#pragma once

namespace Tensors
{
    namespace Tiny
    {

#define CLASS UpperTriangularMatrix
        
        template< int n_, typename Scal_, typename Int_>
        class UpperTriangularMatrix
        {
            // Allocates a square array, but accesses only upper triangle.
            
        public:
            
            using Class_T = UpperTriangularMatrix;
            
#include "Tiny_Details.hpp"
            
        public:
            
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scal,Int>;
                        
        protected:
            
            std::array<std::array<Scal,n>,n> A;

#include "Tiny_UpperTriangular_Common.hpp"

///######################################################
///##                  Arithmetic                      ##
///######################################################
            
        public:
            
            force_inline Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = i; j < n; ++j )
                    {
                        AA += AbsSquared(A[i][j]);
                    }
                }
                return Sqrt(AA);
            }
            
            force_inline friend void Dot( const CLASS & M, const Vector_T & x, Vector_T & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    Scal y_i ( static_cast<Scal>(0) );
                    
                    for( Int j = i; j < n; ++j )
                    {
                        y_i += M.A[i][j] * x.v[j];
                    }
                    
                    y.v[i] = y_i;
                }
            }
            
            
            [[nodiscard]] force_inline Scal Det() const
            {
                Scal det = A[0][0];
                
                for( Int i = 1; i < n; ++i )
                {
                    det *= A[i][i];
                }
                
                return det;
            }

            template<Op op = Op::Id, Diag diag = Diag::NonUnit>
            void Solve( Vector<n,Scal,Int> & b )
            {
                // Solves op(A) x = b and overwrites b with the solution.
                
                if constexpr ( op == Op::Id )
                {
                    // Upper triangular back substitution
                    for( int i = n; i --> 0; )
                    {
                        for( int j = i+1; j < n; ++j )
                        {
                            b[i] -= A[i][j] * b[j];
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            b[i] /= A[i][i];
                        }
                    }
                }
                else if constexpr ( op == Op::Trans )
                {
                    // Lower triangular back substitution from the left
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            b[i] -= A[j][i] * b[j];
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            b[i] /= A[i][i];
                        }
                    }
                }
                else if constexpr ( op == Op::ConjTrans )
                {
                    // Lower triangular back substitution from the left
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            b[i] -= Conj(A[j][i]) * b[j];
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            b[i] /= Conj(A[i][i]);
                        }
                    }
                }
            }
            
            template<int nrhs, Op op = Op::Id, Diag diag = Diag::NonUnit>
            void Solve( Matrix<n,nrhs,Scal,Int> & B )
            {
                // Solves op(A) * X == B and overwrites B the solution with X.
                
                if constexpr ( op == Op::Id )
                {
                    // Upper triangular back substitution from the left
                    for( int i = n; i --> 0; )
                    {
                        for( int j = i+1; j < n; ++j )
                        {
                            for( int k = 0; k < nrhs; ++k )
                            {
                                B[i][k] -= A[i][j] * B[j][k];
                            }
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            scale_buffer<nrhs>( static_cast<Scal>(1) / A[i][i], &B[i][0] );
                        }
                    }
                }
                else if constexpr ( op == Op::Trans )
                {
                    // Lower triangular back substitution from the left
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            for( int k = 0; k < nrhs; ++k )
                            {
                                B[i][k] -= A[j][i] * B[j][k];
                            }
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            scale_buffer<nrhs>( static_cast<Scal>(1) / A[i][i], &B[i][0] );
                        }
                    }
                }
                else if constexpr ( op == Op::ConjTrans )
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            for( int k = 0; k < nrhs; ++k )
                            {
                                B[i][k] -= Conj(A[j][i]) * B[j][k];
                            }
                        }
                        
                        if constexpr (diag == Diag::NonUnit )
                        {
                            scale_buffer<nrhs>( static_cast<Scal>(1) / Conj(A[i][i]), &B[i][0] );
                        }
                    }
                }
            }

            
            [[nodiscard]] friend std::string ToString( cref<CLASS> M, const int p = 16)
            {
                std::stringstream sout;

                sout << "{\n";
                sout << "\t{ ";
                
                sout << ToString(M.A[0][0],p);
                for( Int j = 1; j < n; ++j )
                {
                    sout << ", " << ToString(M.A[0][j],p);
                }
                
                for( Int i = 1; i < n; ++i )
                {
                    sout << " },\n\t{ ";
                    
                    sout << ToString(M.A[i][0],p);
                    
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << ToString(M.A[i][j],p);
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            template<typename T = Scal>
            void ToMatrix( Matrix<n,n,T,Int> & B ) const
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
            
            template<typename T = Scal>
            Matrix<n,n,T,Int> ToMatrix() const
            {
                Matrix<n,n,T,Int> B;
                
                ToMatrix(B);
                
                return B;
            }
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return std::string("Tiny::") + TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
            
        };
        
#undef CLASS
        
    } // namespace Tiny
    
} // namespace Tensors
