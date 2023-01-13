#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
        template< int n_, typename Scalar_, typename Int_ >
        class VectorList
        {
        public:
            
#include "Tiny_Constants.hpp"
            
            static constexpr Int n = n_;
            
            using Tensor_T = Tensor1<Scalar,Int>;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
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
            
            VectorList( const Int length_, const Scalar init )
            :   length(length_)
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = Tensor_T(length_,init);
                }
            }
            
            // Copy constructor
            VectorList( const VectorList & other )
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
                    eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(n-1) +" }.");
                }
            }
            
        public:
            
            //  Access routines
            
            mut<Scalar> data( const Int i )
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i].data();
            }
            
            ptr<Scalar> data( const Int i ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i].data();
            }
            
            Tensor_T & operator[]( const Int i )
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i];
            }
            
            const Tensor_T & operator[]( const Int i ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i];
            }
            
            Tensor_T & operator()( const Int i )
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i];
            }
            
            const Tensor_T & operator()( const Int i ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i];
            }
            
            Scalar & operator()( const Int i, const Int k )
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i);
#endif
                return v[i][k];
            }
            
            const Scalar & operator()( const Int i, const Int k ) const
            {
#ifdef TENSORS_BOUND_CHECKS
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
                    copy_buffer( a[i], &v[i], length );
                }
            }
            
            template<typename S>
            void Write( S * restrict const * restrict const a )
            {
                //Assuming that a is a list of n pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < n; ++i )
                {
                    copy_buffer( &v[i], a[i], length );
                }
            }
            
            template<typename S>
            void Read( ptr<S> a )
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
            void Write( mut<S> a )
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
                return "VectorList<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
        };
        
        
        
#ifdef LTEMPLATE_H
        
        
        template<int n, typename Scalar, typename Int, IS_FLOAT(Scalar)>
        inline mma::TensorRef<mreal> to_MTensorRef( const VectorList<n,Scalar,Int> & A )
        {
            const mint m = A.Dimension(1);
            
            ptr<Scalar> p [n];
            
            for( mint j = 0; j < n; ++j )
            {
                p[j] = A.data(j);
            }
            
            auto B = mma::makeMatrix<mreal>( m, n );
            
            mut<mreal>t b = B.data();
            
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
        inline mma::TensorRef<mint> to_MTensorRef( const VectorList<n,J,Int> & A )
        {
            const mint m = A.Dimension(1);
            
            const J * restrict p [n];
            
            for( mint j = 0; j < n; ++j )
            {
                p[j] = A.data(j);
            }
            
            auto B = mma::makeMatrix<mint>( m, n );
            
            mut<mint> b = B.data();
            
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
