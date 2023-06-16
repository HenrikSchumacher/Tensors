#pragma once

namespace Tensors {

#define TENSOR_T Tensor3

    template <typename Scal_, typename Int_>
    class TENSOR_T
    {
        
#include "Tensor_Common.hpp"
        
    protected:
        
        std::array<Int,3> dims = {0,0,0};     // dimensions visible to user

    public:
        
        TENSOR_T( const Int d0, const Int d1, const Int d2 )
        :   n    { d0 * d1 * d2 }
        ,   dims { d0, d1, d2 }
        {
            allocate();
        }
        
        TENSOR_T( const Int d0, const Int d1, const Int d2, const Scal init )
        :   TENSOR_T( d0, d1, d2 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( ptr<S> a_, const Int d0, const Int d1, const Int d2 )
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
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
        }
        
        void BoundCheck( const Int i, const Int j ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(dims[1]) +" [.");
            }
        }
        
        void BoundCheck( const Int i, const Int j, const Int k ) const
        {
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
        }
        
    public:
        

        force_inline mut<Scal> data( const Int i )
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return &a[i * dims[1] * dims[2]];
        }
        
        force_inline ptr<Scal> data( const Int i ) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return &a[i * dims[1] * dims[2]];
        }

        force_inline mut<Scal> data( const Int i, const Int j )
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j);
#endif
            return &a[( i * dims[1] + j ) * dims[2]];
        }
        
        force_inline ptr<Scal> data( const Int i, const Int j ) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j);
#endif
            return &a[( i * dims[1] + j ) * dims[2]];
        }

        
        force_inline mut<Scal> data( const Int i, const Int j, const Int k)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j,k);
#endif
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        
        force_inline ptr<Scal> data( const Int i, const Int j, const Int k) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j,k);
#endif
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        force_inline Scal & operator()( const Int i, const Int j, const Int k)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j,k);
#endif
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        force_inline const Scal & operator()( const Int i, const Int j, const Int k) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i,j,k);
#endif
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        template< typename S>
        void Write( const Int i, mut<S> b ) const
        {
            copy_buffer( data(i), b, dims[1] * dims[2] );
        }
        
        template< typename S>
        void Write( const Int i, const Int j, mut<Scal> b ) const
        {
            copy_buffer( data(i,j), b, dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, ptr<S> b )
        {
            copy_buffer( b, data(i), dims[1] * dims[2] );
        }
        
        template< typename S>
        void Read( const Int i, const Int j, ptr<S> b )
        {
            copy_buffer( b, data(i,j), dims[2] );
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("Tensor3<")+TypeName<Scal>+","+TypeName<Int>+">";
        }
        
    }; // Tensor3
    
    
    template<typename Scal, typename Int, typename S>
    Tensor3<Scal,Int> ToTensor3( ptr<S> a_, const Int d0, const Int d1, const Int d2 )
    {
        Tensor3<Scal,Int> result ( d0, d1, d2 );

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, typename Int>
    Tensor3<Scal,Int> from_CubeRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename Scal, typename Int>
    Tensor3<Scal,Int> from_CubeRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
