#pragma once

namespace Tensors {

#define TENSOR_T Tensor1

    template <typename Scal_, typename Int_>
    class TENSOR_T
    {
        
#include "Tensor_Common.hpp"
        
    protected:
        
        std::array<Int,1> dims = {0}; // dimensions visible to user
        
    public:
        
        explicit TENSOR_T( const Int d0 )
        :   n    { d0 }
        ,   dims { d0 }
        {
            allocate();
        }
        
        TENSOR_T( const Int d0, const Scal init )
        :   TENSOR_T( d0 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( cptr<S> a_, const Int d0 )
        :   TENSOR_T( d0 )
        {
            Read(a_);
        }
        
    private:
        
        void BoundCheck( const Int i ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
        }
        
    public:
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(1);
        }
        

        force_inline mptr<Scal> data( const Int i )
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline cptr<Scal> data( const Int i ) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline mref<Scal> operator()(const Int i)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline cref<Scal> operator()(const Int i) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline mref<Scal> operator[](const Int i)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline cref<Scal> operator[](const Int i) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return a[i];
        }
        

        
        force_inline mref<Scal> First()
        {
#ifdef TOOLS_DEBUG
            BoundCheck(0);
#endif
            return a[0];
        }
        
        force_inline cref<Scal> First() const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(0);
#endif
            return a[0];
        }

        force_inline mref<Scal> Last()
        {
#ifdef TOOLS_DEBUG
            BoundCheck(n-1);
#endif
            return a[n-1];
        }
        
        force_inline cref<Scal> Last() const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(n-1);
#endif
            return a[n-1];
        }
        
        void Resize( const Int m_ )
        {
            const Int m = std::max( static_cast<Int>(0),m_);
            
            TENSOR_T b (m);
            
            if( m <= n )
            {
                b.Read(a);
            }
            else
            {
                Write(b.data());
            }
            
            swap( *this, b );
        }
        
        void Resize( const Int m_, Int thread_count )
        {
            const Int m = std::max( static_cast<Int>(0),m_);
            
            TENSOR_T b (m);
            
            if( m <= n )
            {
                b.Read(a,thread_count);
            }
            else
            {
                Write(b.data(),thread_count);
            }
            
            swap( *this, b );
        }
        
        void Accumulate( const Int thread_count = 1 )
        {
//            for( Int i = 1; i < n; ++i )
//            {
//                a[i] += a[i-1];
//            }
            parallel_accumulate(a, n, thread_count );
        }
        
        Scal Total() const
        {
            Scal sum = static_cast<Scal>(0);
            
            for( Int i = 0; i < n; ++ i )
            {
                sum += a[i];
            }

            return sum;
        }
        
        inline friend Scal Total( const TENSOR_T & x )
        {
            return x.Total();
        }
        
        inline friend Scal Dot( const TENSOR_T & x, const TENSOR_T & y )
        {
            if( x.n != y.m )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return 0;
            }
            
            return dot( x.a, y.a, x.n );
        }
        
        void iota( const Int thread_count = 1 )
        {
            iota_buffer( a, n, thread_count );
        }
        
    public:
        
        std::string ToString( const Int i_begin, const Int i_end ) const
        {
            if( (i_begin >= 0) && ( i_end <= n) )
            {
                return ToString(
                    &a[i_begin],
                    {std::max(int_cast<Size_T>(0),int_cast<Size_T>(i_end-i_begin))}
                );
            }
            else
            {
                return ToString(a,0);
            }
        }
        
        static std::string ClassName()
        {
            return std::string("Tensor1<")+TypeName<Scal>+","+TypeName<Int>+">";
        }
        
        
    }; // Tensor1
    
    template<typename Scal, typename Int>
    Tensor1<Scal,Int> iota( const Int size_ )
    {
        auto v = Tensor1<Scal,Int>(size_);
        
        v.iota();
        
        return v;
    }
    
    template<typename Scal, typename Int, typename S>
    Tensor1<Scal,Int> ToTensor1( mptr<S> a_, const Int d0 )
    {
        Tensor1<Scal,Int> result (d0);

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, typename Int>
    Tensor1<Scal,Int> from_VectorRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor1<Scal,Int>( A.data(), A.dimensions()[0] );
    }
    
    template<typename Scal, typename Int>
    Tensor1<Scal,Int> from_VectorRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor1<Scal,Int>( A.data(), A.dimensions()[0] );
    }
    
#endif
    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
