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
//##                     Memory                       ##
//######################################################
            
        public:
            
            void SetZero()
            {
                zerofy_buffer<m*n>( &A[0][0] );
            }
            
            void Fill( const Scalar init )
            {
                fill_buffer<m*n>( &A[0][0], init );
            }
            
            template<typename T>
            void Write( T * const target ) const
            {
                copy_buffer<m*n>( &A[0][0], target );
            }
            
            template<typename T>
            void Read( T const * const source )
            {
                copy_buffer<m*n>( source, &A[0][0] );
            }
            
//######################################################
//##                     Access                       ##
//######################################################

#include "Tiny_Details_Matrix.hpp"
            
            
//######################################################
//##                  Arithmetic                      ##
//######################################################
         
        public:
            
            friend CLASS operator+( const CLASS & x, const CLASS & y )
            {
                CLASS z;
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            
            template<class T>
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

            
            template<class T>
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator+=( const T lambda )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += lambda;
                    }
                }
                
                return *this;
            }
            
    
            template<class T>
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator-=( const T lambda )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= lambda;
                    }
                }
                
                return *this;
            }
            
            template<class T>
            std::enable_if_t<
                std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
                CLASS &
            >
            operator*=( const T lambda )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= lambda;
                    }
                }
                
                return *this;
            }
            
            
            void Transpose( CLASS & B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B.A[j][i] = A[i][j];
                    }
                }
            }
            
            void ConjugateTranspose( CLASS & B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B.A[j][i] = conj(A[i][j]);
                    }
                }
            }
            
            void Conjugate( CLASS & B ) const
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        B.A[i][j] = conj(A[i][j]);
                    }
                }
            }
            
            Real MaxNorm() const
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
            
            Real FrobeniusNorm() const
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
            
            for( Int j = 0; j < i; ++j )
            {
                y_i += A[j][i] * x[j];
            }
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
        C.SetZero();
        
        for( Int i = 0; i < M; ++i )
        {
            for( Int k = 0; k < K; ++k )
            {
                for( Int j = 0; j < N; ++j )
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    
} // namespace Tensors

