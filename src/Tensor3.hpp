#pragma once

namespace Tensors {

#define TENSOR_T Tensor3

    template <typename T, typename I>
    class TENSOR_T
    {
        
#include "Tensor_Common.hpp"
        
    protected:
        
        std::array<I,3> dims = {0,0,0};     // dimensions visible to user

    public:
        
        template< typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        TENSOR_T( const J0 d0, const J1 d1, const J2 d2 )
        :   n   { static_cast<I>(d0) * static_cast<I>(d1) * static_cast<I>(d2) }
        ,   dims{static_cast<I>(d0),static_cast<I>(d1),static_cast<I>(d2)}
        {
            allocate();
        }
        
        template<typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2) >
        TENSOR_T( const J0 d0, const J1 d1, const J2 d2, const T init )
        :   TENSOR_T(static_cast<I>(d0),static_cast<I>(d1),static_cast<I>(d2))
        {
            Fill( static_cast<T>(init) );
        }
        
        template<typename J0, typename J1, typename J2, IsInt(J0), IsInt(J1), IsInt(J2)>
        TENSOR_T( const T * a_, const J0 d0, const J1 d1, const J2 d2 )
        :   TENSOR_T(static_cast<I>(d0),static_cast<I>(d1),static_cast<I>(d2))
        {
            Read(a_);
        }
        
        static constexpr I Rank()
        {
            return static_cast<I>(3);
        }
        
    private:
        
        void BoundCheck( const I i ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(dims[0]-1) +" }.");
            }
        }
        
        void BoundCheck( const I i, const I j ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(dims[0]-1) +" }.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(dims[1]-1) +" }.");
            }
        }
        
        void BoundCheck( const I i, const I j, const I k ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(dims[0]-1) +" }.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(dims[1]-1) +" }.");
            }
            if( (k < 0) || (k > dims[2]) )
            {
                eprint(ClassName()+": third index " + std::to_string(k) + " is out of bounds { 0, " + std::to_string(dims[2]-1) +" }.");
            }
        }
        
    public:
        

        T * data( const I i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i * dims[1] * dims[2]];
        }
        
        const T * data( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i * dims[1] * dims[2]];
        }

        T * data( const I i, const I j )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return &a[( i * dims[1] + j ) * dims[2]];
        }
        
        const T * data( const I i, const I j ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return &a[( i * dims[1] + j ) * dims[2]];
        }

        
        T * data( const I i, const I j, const I k)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        
        const T * data( const I i, const I j, const I k) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return &a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        T & operator()( const I i, const I j, const I k)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        const T & operator()( const I i, const I j, const I k) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return a[( i *  dims[1] + j ) * dims[2] + k];
        }
        
        template< typename S>
        void Write( const I i, S * const b ) const
        {
            copy_cast_buffer( data(i), b, dims[1] * dims[2] );
        }
        
        template< typename S>
        void Write( const I i, const I j, T * const b ) const
        {
            copy_cast_buffer( data(i,j), b, dims[2] );
        }
        
        template< typename S>
        void Read( const I i, const S * const b )
        {
            copy_cast_buffer( b, data(i), dims[1] * dims[2] );
        }
        
        template< typename S>
        void Read( const I i, const I j, const S * const b )
        {
            copy_cast_buffer( b, data(i,j), dims[2] );
        }
        
    public:
        
        static std::string ClassName()
        {
            return "Tensor3<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
    }; // Tensor3
    
    
    template<typename T, typename I, typename S, typename J0, typename J1, typename J2, IsInt(I), IsInt(J0), IsInt(J1), IsInt(J2)
    >
    Tensor3<T,I> ToTensor3( const S * a_, const J0 d0, const J1 d1, const J2 d2 )
    {
        Tensor3<T,I> result (static_cast<I>(d0), static_cast<I>(d1), static_cast<I>(d2));

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename T, typename I>
    Tensor3<T,I> from_CubeRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor3<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename T, typename I>
    Tensor3<T,I> from_CubeRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor3<T,I>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
