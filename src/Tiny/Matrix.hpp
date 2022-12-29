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
            
            using Vector_T = Vector<n,Scalar,Int>;
            
        protected:
            
            std::array<std::array<Scalar,n>,m> A;
            
            
        public:
            
            explicit CLASS( const Scalar init )
            :   A {{{init}}}
            {}
            
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
            
            
            template<
                bool add_to,
                int K,
                typename R, typename S
            >
            force_inline
            friend
            std::enable_if_t<
                (
                    std::is_same_v<R,Scalar>
                    ||
                    (IsComplex && std::is_same_v<R,Real>)
                )
                &&
                (
                    std::is_same_v<S,Scalar>
                    ||
                    (IsComplex && std::is_same_v<S,Real>)
                )
                ,
                void
            >
            Dot(
                const Tiny::Matrix<m,K,R,Int> & A,
                const Tiny::Matrix<K,n,S,Int> & B,
                      CLASS & C
            )
            {
                // First pass to overwrite (if desired).
//                LOOP_UNROLL_FULL
                for( Int i = 0; i < m; ++i )
                {
//                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < n; ++j )
                    {
                        if constexpr ( add_to )
                        {
                            C[i][j] += A[i][0] * B[0][j];
                        }
                        else
                        {
                            C[i][j] = A[i][0] * B[0][j];
                        }
                    }
                }
                
                // Now add-in the rest.
//                LOOP_UNROLL_FULL
                for( Int k = 1; k < K; ++k )
                {
//                    LOOP_UNROLL_FULL
                    for( Int i = 0; i < m; ++i )
                    {
//                        LOOP_UNROLL_FULL
                        for( Int j = 0; j < n; ++j )
                        {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
                
            }
            
            template<
                bool add_to,
                typename Scalar, typename S, typename T
            >
            friend
            force_inline
            std::enable_if_t<
                (
                    std::is_same_v<Scalar,T>
                    ||
                    (ScalarTraits<T>::IsComplex && std::is_same_v<Scalar,typename ScalarTraits<T>::Real>)
                )
                &&
                (
                    std::is_same_v<S,T>
                    ||
                 (ScalarTraits<T>::IsComplex && std::is_same_v<Scalar,typename ScalarTraits<T>::Real>)
                )
                ,
                void
            >
            Dot(
                const CLASS & A,
                const Tiny::Vector<n,S,Int> & x,
                      Tiny::Vector<m,T,Int> & y
            )
            {
//                LOOP_UNROLL_FULL
                for( Int i = 0; i < m; ++i )
                {
                    T y_i (0);
                    
//                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < n; ++j )
                    {
                        y_i += A[i][j] * x[j];
                    }
                    
                    if constexpr ( add_to )
                    {
                        y[i] += y_i;
                    }
                    else
                    {
                        y[i] = y_i;
                    }
                }
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
            
            void Threshold( const Real threshold )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        if( std::abs(A[i][j]) <= threshold )
                        {
                            A[i][j] = static_cast<Scalar>(0);
                        }
                    }
                }
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
                if constexpr ( n >= 2 )
                {
                    // Assumes that squared_abs(c) + squared_abs(s) == one.
                    // Assumes that i != j.
                    // Multiplies matrix with the rotation
                    //
                    //    /               \
                    //    |      c     s  |
                    //    |  -conj(s)  c  |
                    //    \               /
                    //
                    // in the i-j-plane from the left.
                    
                    for( Int k = 0; k < n; ++k )
                    {
                        const Scalar x = A[i][k];
                        const Scalar y = A[j][k];
                        
                        A[i][k] =      c    * x + s * y;
                        A[j][k] = - conj(s) * x + c * y;
                    }
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
                if constexpr ( n >= 2 )
                {
                    // Assumes that squared_abs(c) + squared_abs(s) == one.
                    // Assumes that i != j.
                    // Multiplies matrix with rotation
                    //
                    //    /               \
                    //    |      c     s  |
                    //    |  -conj(s)  c  |
                    //    \               /
                    //
                    // in the i-j-plane from the right.
                    
                    for( Int k = 0; k < m; ++k )
                    {
                        const Scalar x = A[k][i];
                        const Scalar y = A[k][j];
                        
                        A[k][i] = c * x - conj(s) * y;
                        A[k][j] = s * x +    c    * y;
                    }
                }
            }
            
            
            
        public:
           
            force_inline void SetIdentity()
            {
                static_assert(m==n, "SetIdentity is only defined for square matrices.");
                
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
                static_assert(m==n, "MakeDiagonal is only defined for square matrices.");
                
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
                static_assert(m==n, "SetDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    A[i][i] = v[i];
                }
            }
            
            Scalar Det() const
            {
                if constexpr ( m != n )
                {
                    return 0;
                }
                
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
                static_assert(m==n, "SetHouseHolderReflector is only defined for square matrices.");
                
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
                static_assert(m==n, "SetGivensRotation is only defined for square matrices.");
                
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
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][i];
                }
            }
            
            Vector<n,Scalar,Int>  Diagonal() const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                Vector<n,Scalar,Int> v;
                Diagonal(v);
                return v;
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
    
    
    
} // namespace Tensors

