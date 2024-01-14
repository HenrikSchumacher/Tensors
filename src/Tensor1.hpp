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
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
#endif
        }
        
    public:
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(1);
        }
        

        force_inline mptr<Scal> data( const Int i )
        {
            BoundCheck(i);
            
            return &a[i];
        }
        
        force_inline cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return &a[i];
        }
        
        force_inline mref<Scal> operator()(const Int i)
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        force_inline cref<Scal> operator()(const Int i) const
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        force_inline mref<Scal> operator[](const Int i)
        {
            BoundCheck(i);
            
            return a[i];
        }
        
        force_inline cref<Scal> operator[](const Int i) const
        {
            BoundCheck(i);
            
            return a[i];
        }
        

        
        force_inline mref<Scal> First()
        {
            return a[0];
        }
        
        force_inline cref<Scal> First() const
        {
            return a[0];
        }

        force_inline mref<Scal> Last()
        {
            return a[n-1];
        }
        
        force_inline cref<Scal> Last() const
        {
            return a[n-1];
        }
        
        void Resize( const Int m_, const bool copy = true )
        {
//            ptic(ClassName()+"::Resize(" + ToString(m_) + ")");
            const Int m = Tools::Ramp(m_);
            
            TENSOR_T b (m);
            
            if( copy )
            {
                if( m <= n )
                {
                    b.Read(a);
                }
                else
                {
                    Write(b.data());
                }
            }
            
            swap( *this, b );
//            ptoc(ClassName()+"::Resize(" + ToString(m_) + ")");
        }
        
        void Resize( const Int m_, const Int thread_count, const bool copy = true )
        {
            const Int m = Ramp(m_);
            
            TENSOR_T b (m);
            
            if( copy )
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
        
        void RequireSize( const Int m, const bool copy = false )
        {
            if( m > n )
            {
                Resize( m, copy );
            }
        }
        
        void RequireSize( const Int m, const Int thread_count, const bool copy = false )
        {
            if( m > n )
            {
                Resize( m, thread_count, copy );
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
        
        Scal Total() const
        {
            Scal sum = static_cast<Scal>(0);
            
            for( Int i = 0; i < n; ++ i )
            {
                sum += a[i];
            }

            return sum;
        }
        
        inline friend Scal Total( cref<TENSOR_T> x )
        {
            return x.Total();
        }
        
        inline friend Scal Dot( cref<TENSOR_T> x, cref<TENSOR_T> y )
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
        
        [[nodiscard]] std::string friend ToString( cref<TENSOR_T> T, const Int i_begin, const Int i_end )
        {
            if( (i_begin >= 0) && ( i_end <= T.n) )
            {
                return ToString(
                    &T.a[i_begin],
                    {Tools::Max(int_cast<Size_T>(0),int_cast<Size_T>(i_end-i_begin))}
                );
            }
            else
            {
                return ToString(T.a,0);
            }
        }
        
        static std::string ClassName()
        {
            return std::string("Tensor1<")+TypeName<Scal>+","+TypeName<Int>+","+ToString(alignment)+">";
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
