#pragma once

namespace Tensors
{

#define TENSOR_T Tensor3

    template <typename Scal_, typename Int_, Size_T alignment = DefaultAlignment>
    class TENSOR_T
    {
        
#include "Tensor_Common.hpp"
        
    private:
        
        std::array<Int,3> dims = {0,0,0};     // dimensions visible to user

    public:
        
        TENSOR_T( const Int d0, const Int d1, const Int d2 )
        :   n    { d0 * d1 * d2 }
        ,   dims { d0, d1, d2 }
        {
            allocate();
        }
        
        TENSOR_T( const Int d0, const Int d1, const Int d2, cref<Scal> init )
        :   TENSOR_T( d0, d1, d2 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( cptr<S> a_, const Int d0, const Int d1, const Int d2 )
        :   TENSOR_T( d0, d1, d2 )
        {
            Read(a_);
        }
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(3);
        }
        
    private:
        
        void BoundCheck( const Int i ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
#else
            (void)i;
#endif
        }
        
        void BoundCheck( const Int i, const Int j ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(dims[1]) +" [.");
            }
#else
            (void)i;
            (void)j;
#endif
        }
        
        void BoundCheck( const Int i, const Int j, const Int k ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(dims[1]) +" [.");
            }
            if( (k < 0) || (k > dims[2]) )
            {
                eprint(ClassName()+": third index " + std::to_string(k) + " is out of bounds [ 0, " + std::to_string(dims[2]) +" [.");
            }
#else
            (void)i;
            (void)j;
            (void)k;
#endif
        }
        
    public:
        

        force_inline mptr<Scal> data( const Int i )
        {
            BoundCheck(i);
            
            return &a[i * dims[1] * dims[2]];
        }
        
        force_inline cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return &a[i * dims[1] * dims[2]];
        }

        force_inline mptr<Scal> data( const Int i, const Int j )
        {
            BoundCheck(i,j);
            
            return &a[( i * dims[1] + j ) * dims[2]];
        }
        
        force_inline cptr<Scal> data( const Int i, const Int j ) const
        {
            BoundCheck(i,j);
            
            return &a[( i * dims[1] + j ) * dims[2]];
        }

        
        force_inline mptr<Scal> data( const Int i, const Int j, const Int k)
        {
            BoundCheck(i,j,k);
            
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        
        force_inline mptr<Scal> data( const Int i, const Int j, const Int k) const
        {
            BoundCheck(i,j,k);
            
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        force_inline mref<Scal> operator()( const Int i, const Int j, const Int k)
        {
            BoundCheck(i,j,k);
            
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        force_inline cref<Scal> operator()( const Int i, const Int j, const Int k) const
        {
            BoundCheck(i,j,k);
            
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        template< typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            copy_buffer( data(i), b, dims[1] * dims[2] );
        }
        
        template< typename S>
        void Write( const Int i, const Int j, mptr<Scal> b ) const
        {
            copy_buffer( data(i,j), b, dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, cptr<S> b )
        {
            copy_buffer( b, data(i), dims[1] * dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, const Int j, cptr<S> b )
        {
            copy_buffer( b, data(i,j), dims[2] );
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("Tensor3<")+TypeName<Scal>+","+TypeName<Int>+","+Tools::ToString(alignment)+">";
        }
        
    }; // Tensor3
    
    
    template<typename Scal, typename Int, typename S>
    Tensor3<Scal,Int> ToTensor3( cptr<S> a_, const Int d0, const Int d1, const Int d2 )
    {
        Tensor3<Scal,Int> result ( d0, d1, d2 );

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, typename Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mreal>> A )
    {
        return ToTensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename Scal, typename Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mint>> A )
    {
        return ToTensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
