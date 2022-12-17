#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template< int m_, int n_, typename Scalar_, typename Int_>
        class MatrixList
        {
        public:
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            using Tensor_T = Tensor1<Scalar,Int>;
            
        private:
            
            Int length = 0;
            
            Tensor_T A [m][n];
            
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
            
            MatrixList( const Int length_, const Scalar init )
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
            
            friend void swap(MatrixList &A, MatrixList &B)
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                std::swap( A.length, B.length );
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        swap( A.A[i][j], B.A[i][j] );
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
            
            Scalar * restrict data( const Int i, const Int j )
            {
                return A[i][j].data();
            }
            
            const Scalar * restrict data( const Int i, const Int j ) const
            {
                return A[i][j].data();
            }
            
            void SetZero()
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j].SetZero();
                    }
                }
            }
            
            Tensor_T & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            const Tensor_T & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            Scalar & operator()( const Int i, const Int j, const Int k )
            {
                return A[i][j][k];
            }
            
            const Scalar & operator()( const Int i, const Int j, const Int k ) const
            {
                return A[i][j][k];
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
            
            
            static std::string ClassName()
            {
                return "MatrixList<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
        };
        
        
    #ifdef LTEMPLATE_H
        
        
        template<int m, int n, typename T, typename I, IsFloat(T)>
        inline mma::TensorRef<mreal> to_MTensorRef( const Tiny::MatrixList<m,n,T,I> & A )
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
            
            mreal * restrict const b = B.data();
            
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
        
        template<int m, int n, typename J, typename I, IsInt(J)>
        inline mma::TensorRef<mint> to_MTensorRef( const Tiny::MatrixList<m,n,J,I> & A )
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
            
            mint * restrict const b = B.data();
            
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
