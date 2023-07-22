#pragma once

namespace Tensors
{
    namespace Tiny
    {
#define CLASS Matrix
        
        template< int m_, int n_, typename Scal_, typename Int_>
        class CLASS
        {
        public:
            
#include "Tiny_Details.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            using Vector_Out_T = Vector<m,Scal,Int>;
            using Vector_Int_T = Vector<n,Scal,Int>;
            
            using Vector_T     = Vector<n,Scal,Int>;
            
        protected:
            
            alignas(Tools::Alignment) std::array<std::array<Scal,n>,m> A;
            
            
        public:
            
            explicit CLASS( cref<Scal> init )
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
            typename std::enable_if_t<
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator+=( cref<Tiny::Matrix<m,n,T,Int>> B )
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
            typename std::enable_if_t<
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator-=( cref<Tiny::Matrix<m,n,T,Int>> B )
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
            typename std::enable_if_t<
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator*=( cref<Tiny::Matrix<m,n,T,Int>> B )
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
            typename std::enable_if_t<
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator/=( cref<Tiny::Matrix<m,n,T,Int>> B )
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
                AddTo_T addto,
                int K,
                typename R, typename S
            >
            force_inline
            friend
            typename std::enable_if_t<
                (
                    SameQ<R,Scal>
                    ||
                    (ComplexQ && SameQ<R,Real>)
                )
                &&
                (
                    SameQ<S,Scal>
                    ||
                    (ComplexQ && SameQ<S,Real>)
                )
                ,
                void
            >
            Dot(
                cref<Tiny::Matrix<m,K,R,   Int>> X,
                cref<Tiny::Matrix<K,n,S,   Int>> Y,
                mref<Tiny::Matrix<m,n,Scal,Int>> Z
            )
            {
                // First pass to overwrite (if desired).
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        if constexpr ( addto == Tensors::AddTo )
                        {
                            Z[i][j] += X[i][0] * Y[0][j];
                        }
                        else
                        {
                            Z[i][j] = X[i][0] * Y[0][j];
                        }
                    }
                }
                
                // Now add-in the rest.
                for( Int k = 1; k < K; ++k )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        for( Int j = 0; j < n; ++j )
                        {
                            Z[i][j] += X[i][k] * Y[k][j];
                        }
                    }
                }
                
            }
            
            template<
                AddTo_T addto,
                typename S, typename T
            >
            friend
            force_inline
            typename std::enable_if_t<
                (
                    SameQ<Scal,T>
                    ||
                    (Scalar::ComplexQ<T> && SameQ<Scal,typename Scalar::Real<T>>)
                )
                &&
                (
                    SameQ<S,T>
                    ||
                    (Scalar::ComplexQ<T> && SameQ<Scal,typename Scalar::Real<T>>)
                )
                ,
                void
            >
            Dot(
                cref<Tiny::Matrix<m,n,Scal,Int>> M,
                cref<Tiny::Vector<n,  S,   Int>> x,
                mref<Tiny::Vector<m,  T,   Int>> y
            )
            {
                for( Int i = 0; i < m; ++i )
                {
                    T y_i (0);
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        y_i += M[i][j] * x[j];
                    }
                    
                    if constexpr ( addto == Tensors::AddTo )
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
            

            force_inline void Transpose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = A[i][j];
                    }
                }
            }
            
            force_inline Matrix<n,m,Scal,Int> Transpose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                Transpose(B);
                
                return B;
            }

            force_inline void ConjugateTranspose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = Scalar::Conj(A[i][j]);
                    }
                }
            }

            force_inline Matrix<n,m,Scal,Int> ConjugateTranspose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                ConjugateTranspose(B);
                
                return B;
            }
            
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
                        AA += Scalar::AbsSquared(A[i][j]);
                    }
                }
                return std::sqrt(AA);
            }

            
            std::string ToString() const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\t{ ";
                if( (m > 0) && (n > 0) )
                {
                    sout << Tools::ToString(A[0][0]);
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << Tools::ToString(A[0][j]);
                    }
                    for( Int i = 1; i < m; ++i )
                    {
                        sout << " },\n\t{ ";
                        
                        sout << Tools::ToString(A[i][0]);
                        
                        for( Int j = 1; j < n; ++j )
                        {
                            sout << ", " << Tools::ToString(A[i][j]);
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
                            A[i][j] = static_cast<Scal>(0);
                        }
                    }
                }
            }
            
        public:
            
            template<class T>
            force_inline void GivensLeft( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= 2 )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    // Assumes that Scalar::AbsSquared(c) + Scalar::AbsSquared(s) == one.
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
                        const Scal x = A[i][k];
                        const Scal y = A[j][k];
                        
                        A[i][k] =      c    * x + s * y;
                        A[j][k] = - Scalar::Conj(s) * x + c * y;
                    }
                }
            }

            template<class T>
            force_inline void GivensRight( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= 2 )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    // Assumes that Scalar::AbsSquared(c) + Scalar::AbsSquared(s) == one.
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
                        const Scal x = A[k][i];
                        const Scal y = A[k][j];
                        
                        A[k][i] = c * x - Scalar::Conj(s) * y;
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
                        A[i][j] = (i==j) ? static_cast<Scal>(1) : static_cast<Scal>(0);
                    }
                }
            }
            
            force_inline void MakeDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "MakeDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : static_cast<Scal>(0);
                    }
                }
            }
            
            force_inline void SetDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "SetDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    A[i][i] = v[i];
                }
            }
            
            [[nodiscard]] Scal Det() const
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
                    Scal M [n][n];
                    
                    Write( &M[0][0] );
                    
                    Scal sign (one);
                    
                    for(Int k = 0; k < n - 1; ++k )
                    {
                        //Pivot - row swap needed
                        if( M[k][k] == zero )
                        {
                            Int l = 0;
                            for( l = k + 1; l < n; ++l )
                            {
                                if( M[l][k] != zero )
                                {
                                    std::swap_ranges( &M[l][0], &M[l][n], &M[k][0] );
                                    sign = -sign;
                                    break;
                                }
                            }
                            
                            //No entries != 0 found in column k -> det = 0
                            if(l == n)
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
                
                return static_cast<Scal>(0);
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
                        A[i][j] -= two * u[i] * Scalar::Conj(u[j]);
                    }
                }
            }
            
            
            force_inline void SetGivensRotation( const Scal c, const Scal s, const Int i, const Int j )
            {
                static_assert(m==n, "SetGivensRotation is only defined for square matrices.");
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                // Assumes that Scalar::AbsSquared(c) + Scalar::AbsSquared(s) == one.
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
                A[j][i] = -Scalar::Conj(s);
                A[j][j] = c;
            }
            
            void Diagonal( Vector<n,Scal,Int> & v ) const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][i];
                }
            }
            
            [[nodiscard]] Vector<n,Scal,Int>  Diagonal() const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                Vector<n,Scal,Int> v;
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
            
            [[nodiscard]] static std::string ClassName()
            {
                return "Tiny::"+TO_STD_STRING(CLASS)+"<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
            
        };
        
        
    #undef CLASS
        
    } // namespace Tiny
    
    
    
} // namespace Tensors
