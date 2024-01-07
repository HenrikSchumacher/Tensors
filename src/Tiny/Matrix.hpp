#pragma once

#include <initializer_list>

namespace Tensors
{
    namespace Tiny
    {
        template< int n_, typename Scal_, typename Int_> class SelfAdjointMatrix;
        
#define CLASS Matrix
        
        template< int m_, int n_, typename Scal_, typename Int_>
        class Matrix
        {
        public:
            
            using Class_T = Matrix;
            
#include "Tiny_Details.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            static_assert(m>0, "Matrix row count must be positive.");
            static_assert(n>0, "Matrix col count must be positive.");
            
            using ColVector_T = Vector<m,Scal,Int>;
            using RowVector_T = Vector<n,Scal,Int>;
            
            using Vector_T    = Vector<n,Scal,Int>;
            
        protected:
            
            
            // TODO: Switching to std::array<RowVector_T,m> A would simplify several things...
            // TODO: But beware: Tiny::Vector has some nontrivial memory alignment!
            
            alignas(Tools::Alignment) std::array<std::array<Scal,n>,m> A;
            
            
        public:
            
            explicit Matrix( cref<Scal> init )
            :   A {{{init}}}
            {}
            
            template<typename S>
            constexpr Matrix( const std::initializer_list<S[n]> list )
//            :   A (list)
//            {}
            {
                const Int m__ = static_cast<Int>(list.size());
                
                auto iter { list.begin() };
                for( Int i = 0; i < m__; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = static_cast<Scal>((*iter)[j]);
                    }
                    ++iter;
                }
                
                for( Int i = m__; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = 0;
                    }
                }
            }
            
//######################################################
//##                     Access                       ##
//######################################################

#include "Tiny_Details_Matrix.hpp"
#include "Tiny_Details_RectangularMatrix.hpp"
            
            
        public:
            
            void WriteRow( mref<RowVector_T> u, const Int i )
            {
                u.Read( &A[i] );
            }
            
            void WriteCol( mref<ColVector_T> v, const Int j )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][j];
                }
            }
            
//######################################################
//##                  Arithmetic                      ##
//######################################################
         
        public:

            force_inline friend void Plus( const CLASS & x, const CLASS & y, const CLASS & z )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            
            [[nodiscard]] force_inline friend const CLASS operator+( const CLASS & x, const CLASS & y )
            {
                CLASS z;
                
                Plus( x, y, z);
                
                return z;
            }
            
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
            
            template<AddTo_T addto, int K_, typename X_T, typename Y_T >
            force_inline friend void
            Dot(
                cref<Tiny::Matrix<m_,K_,X_T ,Int>> X,
                cref<Tiny::Matrix<K_,n_,Y_T ,Int>> Y,
                mref<Tiny::Matrix<m_,n_,Scal,Int>> Z
            )
            {
                constexpr Int K = K_;
                
                
                // First pass to overwrite (if desired).
                {
                    constexpr Int k = 0;
                    
                    for( Int i = 0; i < m; ++i )
                    {
                        if constexpr ( addto == Tensors::AddTo )
                        {
                            combine_buffers<F_Gen,F_Plus,n>(
                                X.A[i][k], Y[k], one, Z[i]
                            );
                        }
                        else
                        {
                            combine_buffers<F_Gen,F_Zero,n>(
                                X.A[i][k], Y[k], zero, Z[i]
                            );
                        }
                    }
                }
    
                // Now add-in the rest.
                for( Int k = 1; k < K; ++k )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        combine_buffers<F_Gen,F_Plus,n>(
                            X.A[i][k], Y[k], one, Z[i]
                        );
                    }
                }
            }
            
//            template< AddTo_T addto, int K, typename X_T, typename Y_T >
//            force_inline
//            friend
//            typename std::enable_if_t<
//                ( SameQ<X_T,Scal> || (ComplexQ && SameQ<X_T,Real>) )
//                &&
//                ( SameQ<Y_T,Scal> || (ComplexQ && SameQ<Y_T,Real>) )
//                ,
//                void
//            >
//            Dot_old(
//                cref<Tiny::Matrix<m,K,X_T, Int>> X,
//                cref<Tiny::Matrix<K,n,Y_T, Int>> Y,
//                mref<Tiny::Matrix<m,n,Scal,Int>> Z
//            )
//            {
//                // First pass to overwrite (if desired).
//                for( Int i = 0; i < m; ++i )
//                {
//                    for( Int j = 0; j < n; ++j )
//                    {
//                        if constexpr ( addto == Tensors::AddTo )
//                        {
//                            Z[i][j] += X[i][0] * Y[0][j];
//                        }
//                        else
//                        {
//                            Z[i][j] = X[i][0] * Y[0][j];
//                        }
//                    }
//                }
//
//                // Now add-in the rest.
//                for( Int k = 1; k < K; ++k )
//                {
//                    for( Int i = 0; i < m; ++i )
//                    {
//                        for( Int j = 0; j < n; ++j )
//                        {
//                            Z[i][j] += X[i][k] * Y[k][j];
//                        }
//                    }
//                }
//            }
            
            template<AddTo_T addto, typename x_T, typename y_T
            >
            force_inline friend void
            Dot(
                cref<Tiny::Matrix<m_,n_,Scal,Int>> M,
                cref<Tiny::Vector<n_,   x_T, Int>> x,
                mref<Tiny::Vector<m_,   y_T, Int>> y
            )
            {
                if constexpr ( addto == Tensors::AddTo )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        y[i] += dot_buffers<n>( M[i], x.data() );
                    }
                }
                else
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        y[i]  = dot_buffers<n>( M[i], x.data() );
                    }
                }
            }
            
//            template<AddTo_T addto, typename x_T, typename y_T >
//            friend
//            force_inline
//            typename std::enable_if_t<
//                (
//                    SameQ<Scal,x_T>
//                    ||
//                    (Scalar::ComplexQ<x_T> && SameQ<Scal,typename Scalar::Real<x_T>>)
//                )
//                &&
//                (
//                    SameQ<x_T,y_T>
//                    ||
//                    (Scalar::ComplexQ<y_T> && SameQ<Scal,typename Scalar::Real<y_T>>)
//                )
//                ,
//                void
//            >
//            Dot_old(
//                cref<Tiny::Matrix<m,n,Scal,Int>> M,
//                cref<Tiny::Vector<n,  x_T, Int>> x,
//                mref<Tiny::Vector<m,  y_T, Int>> y
//            )
//            {
//                for( Int i = 0; i < m; ++i )
//                {
//                    y_T y_i (0);
//                    
//                    for( Int j = 0; j < n; ++j )
//                    {
//                        y_i += M.A[i][j] * x[j];
//                    }
//                    
//                    if constexpr ( addto == Tensors::AddTo )
//                    {
//                        y[i] += y_i;
//                    }
//                    else
//                    {
//                        y[i] = y_i;
//                    }
//                }
//            }
        
            
            template<Op op>
            void LowerFromUpper()
            {
                if constexpr ( op == Op::Conj )
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            A[i][j] = Conj(A[j][i]);
                        }
                    }
                }
                else
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        for( Int j = 0; j < i; ++j )
                        {
                            A[i][j] = A[j][i];
                        }
                    }
                }
            }
           
        public:

            template<bool upper_triangle_only = false>
            [[nodiscard]] force_inline std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<n,Scal,Int>,
                Matrix<n,n,Scal,Int>
            > AHA() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<n,Scal,Int>,
                    Matrix<n,n,Scal,Int>
                > B;
                
                {
                    constexpr Int k = 0;
                    
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] = Conj_A_ki * A[k][j];
                        }
                    }
                    
                }
                
                for( Int k = 1; k < m; ++k )
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] += Conj_A_ki * A[k][j];
                        }
                    }
                }

                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }
            
            template<bool upper_triangle_only = false>
            [[nodiscard]] force_inline std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<m,Scal,Int>,
                Matrix<m,m,Scal,Int>
            > AAH() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<m,Scal,Int>,
                    Matrix<m,m,Scal,Int>
                > B;

                for( Int i = 0; i < m; ++i )
                {
                    B[i][i] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[i][0] );
                    
                    for( Int j = i + 1; j < m; ++j )
                    {
                        B[i][j] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[j][0] );
                    }
                }
                
                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }

        public:
            
            
            force_inline void Conjugate( CLASS & B ) const
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        B.A[i][j] = Conj(A[i][j]);
                    }
                }
            }

            [[nodiscard]] force_inline CLASS Conjugate() const
            {
                CLASS B;
                
                Conjugate(B);
                
                return B;
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
            
            [[nodiscard]] force_inline Matrix<n,m,Scal,Int> Transpose() const
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
                        B[j][i] = Conj(A[i][j]);
                    }
                }
            }

            [[nodiscard]] force_inline Matrix<n,m,Scal,Int> ConjugateTranspose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                ConjugateTranspose(B);
                
                return B;
            }
            
            
            
            [[nodiscard]] force_inline Real MaxNorm() const
            {
                return norm_max<m*n>( *A[0][0] );
            }
            
            [[nodiscard]] force_inline Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        AA += AbsSquared(A[i][j]);
                    }
                }
                
                return Sqrt(AA);
            }

            
            [[nodiscard]] std::string ToString() const
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
                        if( Abs(A[i][j]) <= threshold )
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
                    
                    // Assumes that AbsSquared(c) + AbsSquared(s) == one.
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
                        A[j][k] = - Conj(s) * x + c * y;
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
                    
                    // Assumes that AbsSquared(c) + AbsSquared(s) == one.
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
                        
                        A[k][i] = c * x - Conj(s) * y;
                        A[k][j] = s * x +    c    * y;
                    }
                }
            }
            
            
            
        public:
           
            constexpr force_inline void SetIdentity()
            {
                static_assert(m==n, "SetIdentity is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = KroneckerDelta<Real>(i,j);
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
                        A[i][j] -= two * u[i] * Conj(u[j]);
                    }
                }
            }
            
            
            force_inline void SetGivensRotation( const Scal c, const Scal s, const Int i, const Int j )
            {
                static_assert(m==n, "SetGivensRotation is only defined for square matrices.");
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                // Assumes that AbsSquared(c) + AbsSquared(s) == one.
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
                A[j][i] = -Conj(s);
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
                return std::string("Tiny::Matrix") + "<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
            
        };
        
        
    #undef CLASS
        
        
        // TODO: Make this more type flexible
        
        template<int m, int K, int n, typename X_T, typename Y_T, typename Int>
        [[nodiscard]] force_inline Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int>
        Dot(
            cref<Tiny::Matrix<m,K,X_T,Int>> X,
            cref<Tiny::Matrix<K,n,Y_T,Int>> Y
        )
        {
            Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int> Z;
         
            Dot<Overwrite>(X,Y,Z);
            
            return Z;
        }
        
    } // namespace Tiny
    
    
    
} // namespace Tensors
