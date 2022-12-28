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
            
            template<
                bool add_to,
                int M, int K, int N,
                typename R, typename S, typename T, typename Int
            >
            friend force_inline
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
                const CLASS<M,K,R,Int> & A,
                const CLASS<K,N,S,Int> & B,
                      CLASS<M,N,T,Int> & C
            )
            {
                // First pass to overwrite (if desired).
                LOOP_UNROLL_FULL
                for( Int i = 0; i < M; ++i )
                {
                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < N; ++j )
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
                LOOP_UNROLL_FULL
                for( Int k = 1; k < K; ++k )
                {
                    LOOP_UNROLL_FULL
                    for( Int i = 0; i < M; ++i )
                    {
                        LOOP_UNROLL_FULL
                        for( Int j = 0; j < N; ++j )
                        {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
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
        

    template<
        bool add_to,
        int M, int N, typename Scalar, typename R, typename S, typename T, typename Int
    >
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
        const Tiny::Matrix<M,N,R,Int> & A,
        const Tiny::Vector<N,  S,Int> & x,
              Tiny::Vector<M,  T,Int> & y
    )
    {
        LOOP_UNROLL_FULL
        for( Int i = 0; i < M; ++i )
        {
            T y_i (0);
            
            LOOP_UNROLL_FULL
            for( Int j = 0; j < N; ++j )
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
    
    
    
} // namespace Tensors

