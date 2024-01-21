#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template< int m_, int n_, typename Scal_, typename Int_, Size_T alignment = CacheLineWidth>
        class MatrixList
        {
        public:
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            static constexpr Size_T Alignment = alignment;
            
            using Tensor_T = Tensor1<Scal,Int,Alignment>;
            
        private:
            
            Int length = 0;
            
            std::array<std::array<Tensor_T,n>,m> A;
//            Tensor_T A [m][n];
            
        public:
            //  The big four and half:
            
            MatrixList() = default;
            
            //Destructor
            ~MatrixList() = default;
            
            explicit MatrixList( const Int length_ )
            :   length(length_)
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = Tensor_T(length_);
                    }
                }
            }
            
            MatrixList( const Int length_, cref<Scal> init )
            :   length(length_)
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = Tensor_T(length_,init);
                    }
                }
            }
            
            // Copy constructor
            MatrixList( const MatrixList & other )
            :   MatrixList( other.length )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j].Read( other.A[i][j].data());
                    }
                }
            }
            
            friend void swap( MatrixList & X, MatrixList & Y )
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                std::swap( X.length, Y.length );
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        swap( X.A[i][j], Y.A[i][j] );
                    }
                }
            }
            
            // Move constructor
            MatrixList( MatrixList && other ) noexcept
            :   MatrixList()
            {
                swap(*this, other);
            }
            
            /* Copy assignment operator */
            MatrixList & operator=( const MatrixList & other )
            {
                if( this != &other )
                {
                    if( (length != other.length) )
                    {
                        // Use the copy constructor.
                        swap( *this, MatrixList(other.length) );
                    }
                    else
                    {
                        for( Int i = 0; i < m; ++i )
                        {
                            for( Int j = 0; j < n; ++j )
                            {
                                A[i][j].Read( other.A[i][j].data());
                            }
                        }
                    }
                }
                return *this;
            }
            
            /* Move assignment operator */
            MatrixList & operator=( MatrixList && other ) noexcept
            {
                if( this == &other )
                {
                    wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
                }
                swap( *this, other );
                return *this;
            }
            
            
            //  Access routines
            
            mptr<Scal> data( const Int i, const Int j )
            {
                return A[i][j].data();
            }
            
            cptr<Scal> data( const Int i, const Int j ) const
            {
                return A[i][j].data();
            }
            
            constexpr void SetZero()
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j].SetZero();
                    }
                }
            }
            
            mref<Tensor_T> operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            const Tensor_T & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            mref<Scal> operator()( const Int i, const Int j, const Int k )
            {
                return A[i][j][k];
            }
            
            cref<Scal> operator()( const Int i, const Int j, const Int k ) const
            {
                return A[i][j][k];
            }
            
            mref<std::array<Tensor_T,n>> operator[]( const Int i )
            {
                return A[i];
            }
            
            cref<std::array<Tensor_T,n>> operator[]( const Int i ) const
            {
                return A[i];
            }
            
            
            template<typename S>
            void Read( const S * const * const * const a )
            {
                //Assuming that a is a list of m x n pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        copy_buffer( a[i][j], &A[i][j], length );
                    }
                }
            }
            
            template<typename S>
            void Write( S * const * const * const a ) const
            {
                //Assuming that a is a list of m pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        copy_buffer( &A[i][j], a[i][j], length );
                    }
                    
                }
            }
            
            template<typename S>
            void Read( const S * const a_ )
            {
                //Assuming that a is a list of size Dimension(1) x m of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++ k)
                {
                    for( Int i = 0; i < m; ++ i)
                    {
                        for( Int j = 0; j < n; ++j )
                        {
                            A[i][j][k] = a_[(k*m+i)*n+j];
                        }
                    }
                }
            }
            
            template<typename S>
            void Write( S * const a ) const
            {
                //Assuming that a is a list of size Dimension(1) x m of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++k )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        for( Int j = 0; j < n; ++j )
                        {
                            a[(k*m+i)*n+j] = A[i][j][k];
                        }
                    }
                }
            }
            
        public:
            
            static constexpr Int Rank()
            {
                return 3;
            }
            
            Int Dimension( const Int k ) const
            {
                switch( k )
                {
                    case 0:
                    {
                        return m;
                    }
                    case 1:
                    {
                        return n;
                    }
                    case 2:
                    {
                        return A[0][0].Dimension(0);
                    }
                    default:
                    {
                        return 0;
                    }
                }
            }
            
            
            [[nodiscard]] static std::string ClassName()
            {
                return "Tiny::MatrixList<"+ToString(m)+","+ToString(n)+","+TypeName<Scal>+","+TypeName<Int>+","+ToString(Alignment)+">";
            }
        };
        
        
    #ifdef LTEMPLATE_H
        
        
        template<int m, int n, typename T, typename I, IS_FLOAT(T)>
        inline mma::TensorRef<mreal> to_MTensorRef( cref<Tiny::MatrixList<m,n,T,I>> A )
        {
            const mint N = A.Dimension(2);
            
            const T * restrict p [m][n];
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    p[i][j] = A.data(i,j);
                }
            }
            
            auto B = mma::makeCube<mreal>( N, m, n );
            
            mptr<mreal> b = B.data();
            
            for( mint k = 0; k < N; ++k )
            {
                for( mint i = 0; i < m; ++i )
                {
                    for( mint j = 0; j < n; ++j )
                    {
                        b[(m * k + i) * n + j] = static_cast<mreal>(p[i][j][k]);
                    }
                }
            }
            
            return B;
        }
        
        template<int m, int n, typename J, typename I, IS_INT(J)>
        inline mma::TensorRef<mint> to_MTensorRef( cref<Tiny::MatrixList<m,n,J,I>> A )
        {
            const mint N = A.Dimension(2);
            
            const J * restrict p [m][n];
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    p[i][j] = A.data(i,j);
                }
            }
            
            auto B = mma::makeCube<mint>( N, m, n );
            
            mptr<mint> b = B.data();
            
            for( mint k = 0; k < N; ++k )
            {
                for( mint i = 0; i < m; ++i )
                {
                    for( mint j = 0; j < n; ++j )
                    {
                        b[(m * k + i) * n + j] = static_cast<mreal>(p[i][j][k]);
                    }
                }
            }
            
            return B;
        }
        
    #endif
    
    } // namespace Tiny
    
} // namespace Tensors
