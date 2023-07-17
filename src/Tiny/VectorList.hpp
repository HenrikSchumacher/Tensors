#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
        template< int n_, typename Scal_, typename Int_ >
        class VectorList
        {
        public:
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int n = n_;
            
            using Tensor_T = Tensor1<Scal,Int>;
            
            using Vector_T = Vector<n,Scal,Int>;
            
        private:
            
            Int length = 0;
            
            std::array<Tensor_T,n> v;
            
        public:
            
            //  The big four and half:
            
            VectorList() = default;
            
            ~VectorList() = default;
            
            explicit VectorList( const Int length_ )
            :   length(length_)
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = Tensor_T(length_);
                }
            }
            
            VectorList( const Int length_, const Scal init )
            :   length(length_)
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = Tensor_T(length_,init);
                }
            }
            
            template< typename ExtScal, typename ExtInt >
            explicit VectorList( cptr<ExtScal> a, const ExtInt length_ )
            :   length(length_)
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = Tensor_T( int_cast<Int>(length_) );
                }
                
                Read(a);
            }
            
            // Copy constructor
            VectorList( cref<VectorList> other )
            :   VectorList( other.length )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i].Read( other.v[i].data());
                }
            }
            
            friend void swap(VectorList & A, VectorList & B)
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( A.length, B.length );
                
                for( Int i = 0; i < n; ++i )
                {
                    swap( A.v[i], B.v[i] );
                }
            }
            
            // Move constructor
            VectorList( VectorList && other ) noexcept
            :   VectorList()
            {
                swap(*this, other);
            }
            
            /* Copy assignment operator */
            VectorList & operator=( const VectorList & other )
            {
                if( this != &other )
                {
                    if( (length != other.length) )
                    {
                        // Use the copy constructor.
                        swap( *this, VectorList(other.length) );
                    }
                    else
                    {
                        for( Int i = 0; i < n; ++i )
                        {
                            v[i].Read( other.v[i].data());
                        }
                    }
                }
                return *this;
            }
            
            /* Move assignment operator */
            VectorList & operator=( VectorList && other ) noexcept
            {
                if( this == &other )
                {
                    wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
                }
                swap( *this, other );
                return *this;
            }
            
        private:
            
            void BoundCheck( const Int i ) const
            {
                if( (i < 0) || (i > n) )
                {
                    eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(n) +" [.");
                }
            }
            
        public:
            
            //  Access routines
            
            mptr<Scal> data( const Int i )
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i].data();
            }
            
            cptr<Scal> data( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i].data();
            }
            
            mref<Tensor_T> operator[]( const Int i )
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i];
            }
            
            cref<Tensor_T> operator[]( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i];
            }
            
            mref<Tensor_T> operator()( const Int i )
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i];
            }
            
            cref<Tensor_T> operator()( const Int i ) const
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i];
            }
            
            mref<Scal> operator()( const Int i, const Int k )
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i][k];
            }
            
            cref<Scal> operator()( const Int i, const Int k ) const
            {
#ifdef TOOLS_DEBUG
                BoundCheck(i);
#endif
                return v[i][k];
            }
            
            
            
            void SetZero()
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i].SetZero();
                }
            }
            
            template<typename S>
            void Read( const S * restrict const * restrict const a )
            {
                //Assuming that a is a list of n pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer<VarSize,Sequential>( a[i], &v[i], length );
                }
            }
            
            template<typename S>
            void Write( S * restrict const * restrict const a ) const
            {
                //Assuming that a is a list of n pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer<VarSize,Sequential>( &v[i], a[i], length );
                }
            }
            
            template<typename S>
            void Read( cptr<S> a )
            {
                //Assuming that a is a list of size Dimension(1) x n of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++ k)
                {
                    for( Int i = 0; i < n; ++ i)
                    {
                        v[i][k] = a[n*k+i];
                    }
                }
            }
            
            template<typename S>
            void Write( mptr<S> a ) const
            {
                //Assuming that a is a list of size Dimension(1) x n of vectors in interleaved form.
                
                for( Int k = 0; k < length; ++ k)
                {
                    for( Int i = 0; i < n; ++ i)
                    {
                        a[n*k+i] = v[i][k];
                    }
                }
            }
            
        public:
            
            static constexpr Int Rank()
            {
                return 2;
            }
            
            Int Dimension( const Int k ) const
            {
                switch( k )
                {
                    case 0:
                    {
                        return n;
                    }
                    case 1:
                    {
                        return v[0].Dimension(0);
                    }
                    default:
                    {
                        return 0;
                    }
                }
            }
            
            static std::string ClassName()
            {
                return std::string("VectorList")+"<"+ToString(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
        };
        
        
        
#ifdef LTEMPLATE_H
        
        
        template<int n, typename Scal, typename Int, IS_FLOAT(Scal)>
        inline mma::TensorRef<mreal> to_MTensorRef( cref<VectorList<n,Scal,Int>> A )
        {
            const mint m = A.Dimension(1);
            
            cptr<Scal> p [n];
            
            for( mint j = 0; j < n; ++j )
            {
                p[j] = A.data(j);
            }
            
            auto B = mma::makeMatrix<mreal>( m, n );
            
            mptr<mreal> b = B.data();
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    b[n * i + j] = static_cast<mreal>(p[j][i]);
                }
            }
            
            return B;
        }
        
        template<int n, typename J, typename Int, IS_INT(J)>
        inline mma::TensorRef<mint> to_MTensorRef( cref<VectorList<n,J,Int>> A )
        {
            const mint m = A.Dimension(1);
            
            const J * restrict p [n];
            
            for( mint j = 0; j < n; ++j )
            {
                p[j] = A.data(j);
            }
            
            auto B = mma::makeMatrix<mint>( m, n );
            
            mptr<mint> b = B.data();
            
            for( mint i = 0; i < m; ++i )
            {
                for( mint j = 0; j < n; ++j )
                {
                    b[n * i + j] = static_cast<mint>(p[j][i]);
                }
            }
            
            return B;
        }
        
#endif
        
    } // namespace Tiny
    
} // namespace Tensors
