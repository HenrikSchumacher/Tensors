#pragma once

namespace Tensors
{
    namespace Tiny
    {
#define CLASS Matrix
        
        template< int m_, int n_, typename Scalar_, typename Int_>
        class CLASS
        {
        public:
            
#include "Tiny_Details.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            using Vector_Out_T = Vector<m,Scalar,Int>;
            using Vector_Int_T = Vector<n,Scalar,Int>;
            
        protected:
            
            std::array<std::array<Scalar,n>,m> A;
            
//######################################################
//##                     Access                       ##
//######################################################

#include "Tiny_Details_Matrix.hpp"
#include "Tiny_Details_RectangularMatrix.hpp"
            
//######################################################
//##                  Arithmetic                      ##
//######################################################
         
        public:
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator+=( const CLASS<m,n,T,Int> & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator-=( const CLASS<m,n,T,Int> & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator*=( const CLASS<m,n,T,Int> & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator/=( const CLASS<m,n,T,Int> & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] /= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
        public:
            
            force_inline Real MaxNorm() const
            {
                Real max = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        max = std::max( max, std::abs(A[i][j]));
                    }
                }
                return max;
            }
            
            force_inline Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        AA += abs_squared(A[i][j]);
                    }
                }
                return std::sqrt(AA);
            }

            
            std::string ToString( const int p = 16) const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\t{ ";
                if( (m > 0) && (n > 0) )
                {
                    sout << Tools::ToString(A[0][0],p);
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << Tools::ToString(A[0][j],p);
                    }
                    for( Int i = 1; i < m; ++i )
                    {
                        sout << " },\n\t{ ";
                        
                        sout << Tools::ToString(A[i][0],p);
                        
                        for( Int j = 1; j < n; ++j )
                        {
                            sout << ", " << Tools::ToString(A[i][j],p);
                        }
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
        public:
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                void
            >
            GivensLeft( const T c, const T s, const Int i, const Int j )
            {
                // Assumes that squared_abs(c) + squared_abs(s) == one.
                // Multiplies matrix with the rotation
                //
                //    /              \
                //    |  c  -conj(s) |
                //    |  s   c       |
                //    \              /
                //
                // in the i-j-plane from the left.
                
                for( Int k = 0; k < n; ++k )
                {
                    const Scalar x = A[i][k];
                    const Scalar y = A[j][k];
                    
                    A[i][k] = c * x - conj(s) * y;
                    A[j][k] = s * x +      c  * y;
                }
            }

            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                void
            >
            GivensRight( const T c, const T s, const Int i, const Int j )
            {
                // Assumes that squared_abs(c) + squared_abs(s) == one.
                // Multiplies matrix with rotation
                //
                //    /              \
                //    |  c  -conj(s) |
                //    |  s      c    |
                //    \              /
                //
                // in the i-j-plane from the right.
                
                for( Int k = 0; k < m; ++k )
                {
                    const Scalar x = A[k][i];
                    const Scalar y = A[k][j];
                    
                    A[k][i] =     c    * x + s * y;
                    A[k][j] = -conj(s) * x + c * y;
                }
            }
            
        public:

            static constexpr Int RowCount()
            {
                return m;
            }
            
            static constexpr Int ColCount()
            {
                return n;
            }
            
            static constexpr Int Dimension( const Int i )
            {
                if( i == 0 )
                {
                    return m;
                }
                if( i == 1 )
                {
                    return n;
                }
                return static_cast<Int>(0);
            }
            
            static std::string ClassName()
            {
                return TO_STD_STRING(CLASS)+"<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
        
    #undef CLASS
        
    } // namespace Tiny
        
    template< int M, int N, typename Scalar, typename Int >
    void Dot(
        const Tiny::Matrix<M,N,Scalar,Int> & A,
        const Tiny::Vector<N,Scalar,Int> & x,
              Tiny::Vector<M,Scalar,Int> & y
    )
    {
        for( Int i = 0; i < M; ++i )
        {
            Scalar y_i (0);
            
            for( Int j = 0; j < N; ++j )
            {
                y_i += A[i][j] * x[j];
            }
            
            y[i] = y_i;
        }
    }
    
    template< int M, int K, int N, typename Scalar, typename Int >
    void Dot(
        const Tiny::Matrix<M,K,Scalar,Int> & A,
        const Tiny::Matrix<K,N,Scalar,Int> & B,
              Tiny::Matrix<M,N,Scalar,Int> & C
    )
    {
        for( Int i = 0; i < M; ++i )
        {
            Tiny::Vector<N,Scalar,Int> C_i = {};
            for( Int k = 0; k < K; ++k )
            {
                for( Int j = 0; j < N; ++j )
                {
                    C_i[j] += A[i][k] * B[k][j];
                }
            }
            C_i.Write( C[i] );
        }
    }
    
} // namespace Tensors

