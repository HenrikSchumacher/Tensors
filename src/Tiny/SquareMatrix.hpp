#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
#define CLASS SquareMatrix
        
        template< int n_, typename Scalar_, typename Int_>
        struct CLASS : public Tensors::Tiny::Matrix<n_,n_,Scalar_,Int_>
        {
        private:
            
            using Base_T = Tensors::Tiny::Matrix<n_,n_,Scalar_,Int_>;
            
        public:
            
#include "Tiny_Details.hpp"

            using Base_T::m;
            using Base_T::n;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
            using Base_T::Write;
            using Base_T::Read;
            using Base_T::RowCount;
            using Base_T::ColCount;
            using Base_T::FrobeniusNorm;
            using Base_T::GivensRight;
            using Base_T::GivensLeft;
            using Base_T::operator+=;
            using Base_T::operator-=;
            using Base_T::operator*=;
            using Base_T::operator/=;
            
            explicit CLASS( const Scalar init )
            :   Base_T(init)
            {}
            
            
        protected:
            
            using Base_T::A;
            
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
            operator+=( const CLASS<n,T,Int> & B )
            {
                for( Int i = 0; i < n; ++i )
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
            operator-=( const CLASS<n,T,Int> & B )
            {
                for( Int i = 0; i < n; ++i )
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
            operator*=( const CLASS<n,T,Int> & B )
            {
                for( Int i = 0; i < n; ++i )
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
            operator/=( const CLASS<n,T,Int> & B )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] /= B.A[i][j];
                    }
                }
                return *this;
            }
            
            
        public:
           
            force_inline void SetIdentity()
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
                    }
                }
            }
            
            force_inline void MakeDiagonal( const Tensors::Tiny::Vector<n,Scalar,Int> & v )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : static_cast<Scalar>(0);
                    }
                }
            }
            
            force_inline void SetDiagonal( const Tensors::Tiny::Vector<n,Scalar,Int> & v )
            {
                for( Int i = 0; i < n; ++i )
                {
                    A[i][i] = v[i];
                }
            }
            
            Scalar Det() const
            {
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if constexpr ( n == 3 )
                {
                    return (
                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
                    );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                if constexpr ( n > 3 )
                {
                    Scalar M [n][n];
                    
                    Write( &M[0][0] );
                    
                    Scalar sign (one);
                    
                    for(Int k = 0; k < n - 1; ++k )
                    {
                        //Pivot - row swap needed
                        if( M[k][k] == zero )
                        {
                            Int m = 0;
                            for( m = k + 1; m < n; ++m )
                            {
                                if( M[m][k] != zero )
                                {
                                    std::swap_ranges( &M[m][0], &M[m][n], &M[k][0] );
                                    sign = -sign;
                                    break;
                                }
                            }
                            
                            //No entries != 0 found in column k -> det = 0
                            if(m == n)
                            {
                                return zero;
                            }
                        }
                        
                        //Apply formula
                        for( Int i = k + 1; i < n; ++i )
                        {
                            for( Int j = k + 1; j < n; ++j )
                            {
                                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j];
                                if(k != 0)
                                {
                                    M[i][j] /= M[k-1][k-1];
                                }
                            }
                        }
                    }
                    
                    return sign * M[n-1][n-1];
                }
                
                return static_cast<Scalar>(0);
            }
            
            
        public:
            
            force_inline void SetHouseHolderReflector( const Vector_T & u, const Int begin, const Int end )
            {
                // Write the HouseHolder reflection of u into the matrix; assumes that u is zero outside [begin,...,end[.
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                
                SetIdentity();

                for( Int i = begin; i < end; ++i )
                {
                    for( Int j = begin; j < end; ++j )
                    {
                        A[i][j] -= two * u[i] * conj(u[j]);
                    }
                }
            }
            
            
            force_inline void SetGivensRotation( const Scalar c, const Scalar s, const Int i, const Int j )
            {
                // Mostly meant for debugging purposes, thus not extremely optimized.
                // Assumes that squared_abs(c) + squared_abs(s) == one.
                // Write Givens rotion
                //
                //    /              \
                //    |     c     s  |
                //    | -conj(s)  c  |
                //    \              /
                //
                // in the i-j-plane into the matrix.
                
                SetIdentity();
                
                A[i][i] = c;
                A[i][j] = s;
                A[j][i] = -conj(s);
                A[j][j] = c;
            }
            
            void Diagonal( Vector<n,Scalar,Int> & v ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][i];
                }
            }
            
            Vector<n,Scalar,Int>  Diagonal() const
            {
                Vector<n,Scalar,Int> v;
                Diagonal(v);
                return v;
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
        
        template< int N, typename R, typename S, typename T, typename Int >
        force_inline
        std::enable_if_t<
            (
                std::is_same_v<R,T>
                ||
                (ScalarTraits<T>::IsComplex && std::is_same_v<R,typename ScalarTraits<T>::Real>)
            )
            &&
            (
                std::is_same_v<S,T>
                ||
                (ScalarTraits<T>::IsComplex && std::is_same_v<S,typename ScalarTraits<T>::Real>)
            )
            ,
            void
        >
        Dot(
            const Tiny::SquareMatrix<N,R,Int> & A,
            const Tiny::Vector<N,S,Int> & x,
                  Tiny::Vector<N,T,Int> & y
        )
        {
            for( Int i = 0; i < N; ++i )
            {
                T y_i ( static_cast<T>(0) );
                for( Int j = 0; j < N; ++j )
                {
                    y_i += A[i][j] * x[j];
                }
                y[i] = y_i;
            }
        }
        
        template< int N, typename R, typename S, typename T, typename Int >
        force_inline
        std::enable_if_t<
            (
                std::is_same_v<R,T>
                ||
                (ScalarTraits<T>::IsComplex && std::is_same_v<R,typename ScalarTraits<T>::Real>)
            )
            &&
            (
                std::is_same_v<S,T>
                ||
                (ScalarTraits<T>::IsComplex && std::is_same_v<S,typename ScalarTraits<T>::Real>)
            )
            ,
            void
        >
        Dot(
            const Tiny::SquareMatrix<N,R,Int> & A,
            const Tiny::SquareMatrix<N,S,Int> & B,
                  Tiny::SquareMatrix<N,T,Int> & C
        )
        {
            for( Int i = 0; i < N; ++i )
            {
                Tiny::Vector<N,T,Int> C_i ( static_cast<T>(0) );
                for( Int k = 0; k < N; ++k )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        C_i[j] += A[i][k] * B[k][j];
                    }
                }
                C_i.Write( C[i] );
            }
        }
        
        
        template< int N, typename Scalar, typename Int >
        force_inline
        Tiny::SquareMatrix<N,Scalar,Int> DiagonalMatrix( const Tiny::Vector<N,Scalar,Int> & v )
        {
            Tiny::SquareMatrix<N,Scalar,Int> A;
            
            A.MakeDiagonal(v);
            
            return A;
        }
        
    } // namespace Tiny
        
} // namespace Tensors
