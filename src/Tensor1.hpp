#pragma once

namespace Tensors
{

#define TENSOR_T Tensor1

    template <typename Scal_, typename Int_, Size_T alignment = DefaultAlignment>
    class TENSOR_T
    {
        
#include "Tensor_Common.hpp"
        
    private:
        
        std::array<Int,1> dims = {0}; // dimensions visible to user
        
    public:
        
        explicit TENSOR_T( const Int d0 )
        :   n    { d0 }
        ,   dims { d0 }
        {
//            logprint("Constuctor of "+ClassName()+" of size "+ToString(d0) );
            allocate();
        }
        
        TENSOR_T( const Int d0, cref<Scal> init )
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
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
#else
            (void)i;
#endif
        }
        
    public:
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(1);
        }
        

        TOOLS_FORCE_INLINE mptr<Scal> data( const Int i )
        {
            BoundCheck(i);
            
            return &a[i];
        }
        
        TOOLS_FORCE_INLINE cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return &a[i];
        }
        
        TOOLS_FORCE_INLINE mref<Scal> operator()(const Int i)
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> operator()(const Int i) const
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        TOOLS_FORCE_INLINE mref<Scal> operator[](const Int i)
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> operator[](const Int i) const
        {
            BoundCheck(i);
            
            return a[i];
        }
        

        
        TOOLS_FORCE_INLINE mref<Scal> First()
        {
            return a[0];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> First() const
        {
            return a[0];
        }

        TOOLS_FORCE_INLINE mref<Scal> Last()
        {
            return a[n-1];
        }
        
        TOOLS_FORCE_INLINE cref<Scal> Last() const
        {
            return a[n-1];
        }
        
        template<bool copy>
        void Resize( const Int m_, const Int thread_count = 1 )
        {
            const Int m = Ramp(m_);
            
            TENSOR_T b (m);
            
            if constexpr ( copy )
            {
                if( m <= n )
                {
                    b.ReadParallel(a,thread_count);
                }
                else
                {
                    WriteParallel(b.data(),thread_count);
                }
            }
            
            swap( *this, b );
        }
        
        template<bool copy>
        void RequireSize( const Int m, const Int thread_count = 1 )
        {
            if( m > n )
            {
                Resize<copy>( m, thread_count );
            }
        }
        
        void Accumulate( const Int thread_count = 1 )
        {
//            for( Int i = 1; i < n; ++i )
//            {
//                a[i] += a[i-1];
//            }
            parallel_accumulate(a, n, thread_count );
        }
        
        Scal Total(const Int thread_count = 1) const
        {
            return total_buffer<VarSize,Parallel>(a,n,thread_count);
        }
        
        inline friend Scal Total( cref<TENSOR_T> x )
        {
            return x.Total();
        }
        
        inline friend Scal Dot( cref<TENSOR_T> x, cref<TENSOR_T> y, const Int thread_count = 1 )
        {
            if( x.n != y.n )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return 0;
            }
            
            return dot_buffers<VarSize,Parallel>(x.a,y.a,x.n,thread_count);
        }
        
        void iota( const Int thread_count = 1 )
        {
            iota_buffer( a, int_cast<Size_T>(n), int_cast<Size_T>(thread_count) );
        }
        
    public:
        
        [[nodiscard]] std::string friend ToString( cref<TENSOR_T> T, const Int i_begin, const Int i_end )
        {
            if( (i_begin >= 0) && ( i_end <= T.n) )
            {
                return ArrayToString(
                    &T.a[i_begin],
                    {Tools::Max(int_cast<Size_T>(0),int_cast<Size_T>(i_end-i_begin))}
                );
            }
            else
            {
                return ArrayToString(T.a,0);
            }
        }

    public:
        
        static std::string ClassName()
        {
            return ct_string("Tensor1")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + to_ct_string(alignment)
                + ">";
        }
        
    }; // Tensor1
    
    template<typename Scal, typename Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> iota( const Int size_ )
    {
        Tensor1<Scal,Int,alignment> v (size_);
        
        v.iota();
        
        return v;
    }
    
    template<typename Scal, typename Int, Size_T alignment = DefaultAlignment, typename S>
    Tensor1<Scal,Int,alignment> ToTensor1( mptr<S> a_, const Int d0 )
    {
        Tensor1<Scal,Int,alignment> result (d0);

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, typename Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> from_VectorRef( cref<mma::TensorRef<mreal>> A )
    {
        return ToTensor1<Scal,Int,alignment>( A.data(), A.dimensions()[0] );
    }
    
    template<typename Scal, typename Int, Size_T alignment = DefaultAlignment>
    Tensor1<Scal,Int,alignment> from_VectorRef( cref<mma::TensorRef<mint>> A )
    {
        return ToTensor1<Scal,Int,alignment>( A.data(), A.dimensions()[0] );
    }
    
#endif
    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
