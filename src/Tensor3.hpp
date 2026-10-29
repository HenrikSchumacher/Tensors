#pragma once

namespace Tensors
{

#define TENSOR_T Tensor3

    
    /*!@brief A generic, contiguous, heap-allocated container that mostly behaves like a tensor of dimensions `{Dim(0),Dim(1),Dim(2)}`.*/
    template <typename Scal_, IntQ Int_, Size_T alignment = DefaultAlignment>
    class Tensor3 final
    {
    public:
        
        static constexpr Int_ rank = 3;
        
#include "Tensor_Common.hpp"
#include "Tensor_Common23.hpp"
        
    public:
        
        template<
        IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( const d0_T d0, const d1_T d1, const d2_T d2 )
        :   n    { int_cast<Int>(ToSize_T(d0) * ToSize_T(d1) * ToSize_T(d2)) }
        ,   dims { int_cast<Int>(d0), int_cast<Int>(d1), int_cast<Int>(d2) }
        {
#ifdef TENSORS_ALLOCATION_LOGS
            logprint(ClassName() + " constructor (size = " + ToString(Size()) + ")");
#endif
            allocate();
        }
        
        template<IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( const d0_T d0, const d1_T d1, const d2_T d2, cref<Scal> init )
        :   Tensor3( d0, d1, d2 )
        {
            Fill( init );
        }
        
        template<typename S, IntQ d0_T, IntQ d1_T, IntQ d2_T>
        Tensor3( cptr<S> a_, const d0_T d0, const d1_T d1, const d2_T d2 )
        :   Tensor3( d0, d1, d2 )
        {
            Read(a_);
        }
        
    protected:
        
        bool OutOfBounds0Q( IntQ auto i ) const
        {
            return (a == nullptr) || std::cmp_less(i,Int{0}) || std::cmp_greater(i,dims[0]);
        }
        
        bool OutOfBounds1Q( IntQ auto i ) const
        {
            return std::cmp_less(i,Int{0}) || std::cmp_greater(i,dims[1]);
        }
        
        bool OutOfBounds2Q( IntQ auto i ) const
        {
            return std::cmp_less(i,Int{0}) || std::cmp_greater(i,dims[2]);
        }
        
    public:
        
        bool OutOfBoundsQ( IntQ auto i ) const
        {
            return OutOfBounds0Q(i);
        }
        
        bool OutOfBoundsQ( IntQ auto i, IntQ auto j ) const
        {
            return OutOfBounds0Q(i) || OutOfBounds1Q(j);
        }
        
        bool OutOfBoundsQ( IntQ auto i, IntQ auto j, IntQ auto k ) const
        {
            return OutOfBoundsQ(i,j) || OutOfBounds1Q(j) || OutOfBounds2Q(k);
        }
        
    protected:
        
        void BoundCheck( IntQ auto i ) const
        {
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( OutOfBounds0Q(i) )
            {
                eprint(ClassName() + ": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
        }
        
        void BoundCheck( IntQ auto i, IntQ auto j ) const
        {
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( OutOfBounds0Q(i) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
            if( OutOfBounds1Q(j) )
            {
                eprint(ClassName()+": second index " + ToString(j) + " is out of bounds [ 0, " + ToString(dims[1]) +" [.");
            }
        }
        
        void BoundCheck( IntQ auto i, IntQ auto j, IntQ auto k ) const
        {
            if( a == nullptr )
            {
                eprint(ClassName() + ": pointer is nullptr.");
            }
            if( OutOfBounds0Q(i) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
            if( OutOfBounds1Q(j) )
            {
                eprint(ClassName()+": second index " + ToString(j) + " is out of bounds [ 0, " + ToString(dims[1]) +" [.");
            }
            if( OutOfBounds2Q(k) )
            {
                eprint(ClassName()+": third index " + ToString(k) + " is out of bounds [ 0, " + ToString(dims[2]) +" [.");
            }
        }
        
    public:
        
        /*!@brief Return mutable pointer to first element in slice `i`.*/
        TOOLS_FORCE_INLINE mptr<Scal> data( IntQ auto i ) noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[static_cast<Int>(i) * dims[1] * dims[2]];
        }
        
        /*!@brief Return immutable pointer to first element in slice `i`.*/
        TOOLS_FORCE_INLINE cptr<Scal> data( IntQ auto i ) const noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[static_cast<Int>(i) * dims[1] * dims[2]];
        }

        /*!@brief Return mutable pointer to first element in slice `{i,j}`.*/
        TOOLS_FORCE_INLINE mptr<Scal> data( IntQ auto i, IntQ auto j ) noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return &a[(static_cast<Int>(i) * dims[1] + static_cast<Int>(j)) * dims[2]];
        }
        
        /*!@brief Return immutable pointer to first element in slice `{i,j}`.*/
        TOOLS_FORCE_INLINE cptr<Scal> data( IntQ auto i, IntQ auto j ) const noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return &a[(static_cast<Int>(i) * dims[1] + static_cast<Int>(j)) * dims[2]];
        }

        /*!@brief Return mutable pointer to element at position `{i,j,k}`.*/
        TOOLS_FORCE_INLINE mptr<Scal> data( IntQ auto i, IntQ auto j, IntQ auto k ) noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return &a[(static_cast<Int>(i) *  dims[1] + static_cast<Int>(j)) * dims[2] + static_cast<Int>(k)];
        }
        
        /*!@brief Return immutable pointer to element at position `{i,j,k}`.*/
        TOOLS_FORCE_INLINE cptr<Scal> data(
            IntQ auto i, IntQ auto j, IntQ auto k
        ) const noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return &a[(static_cast<Int>(i) * dims[1] + static_cast<Int>(j)) * dims[2] + static_cast<Int>(k)];
        }
        
        
        /*!@brief Access element at position `{i,j,k}`.*/
        TOOLS_FORCE_INLINE mref<Scal> operator()( IntQ auto i, IntQ auto j, IntQ auto k ) noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return *data(i,j,k);
        }
        
        /*!@brief Access element at position `{i,j,k}`, read only.*/
        TOOLS_FORCE_INLINE cref<Scal> operator()( IntQ auto i, IntQ auto j, IntQ auto k ) const noexcept
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
            return *data(i,j,k);
        }
        
        auto WriteAccess()
        {
            return [this]( Int i, Int j, Int k ) -> Scal&
            {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j,k);
#endif
                return *this->data(i,j,k);
            };
        }
        
        auto ReadAccess() const
        {
            return [this]( Int i, Int j, Int k ) -> const Scal &
            {
#ifdef TENSORS_BOUND_CHECKS
                BoundCheck(i,j,k);
#endif
                return *this->data(i,j,k);
            };
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
        
        template<typename FillFun_T>
        void FillByFunction( FillFun_T && f )
        {
            for( Int i = 0; i < Dim(0); ++i )
            {
                for( Int j = 0; j < Dim(1); ++j )
                {
                    for( Int k = 0; k < Dim(2); ++k )
                    {
                        this->operator()(i,j,k) = f(i,j,k);
                    }
                }
            }
        }
        
        
        inline friend OutString & operator<<( OutString & s, cref<TENSOR_T> tensor )
        {
            return s.PutCube(tensor.ReadAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2));
        }
        
        inline friend InString & operator>>( InString & s, mref<TENSOR_T> tensor )
        {
            return s.TakeCube(tensor.WriteAccess(), tensor.Dim(0), tensor.Dim(1), tensor.Dim(2));
        }
        
        inline friend std::string ToString( cref<TENSOR_T> tensor, cref<std::string> line_prefix )
        {
            return OutString::FromArray(
                tensor.ReadAccess(),
                tensor.Dim(0), line_prefix + "{\n", ",\n", ("\n" + line_prefix) + "}",
                tensor.Dim(1), line_prefix + " { ", ", ", " }",
                tensor.Dim(2),                "{ ", ", ", " }"
            );
        }

        
    public:
        
        static constexpr std::string ClassName() noexcept
        {
            return std::string("Tensor3")
                + "<" + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + ToString(alignment)
                + ">";
        }
        
    }; // Tensor3
    
    
//    template<typename Scal, IntQ Int, typename S>
//    Tensor3<Scal,Int> ToTensor3( cptr<S> a_, const Int d0, const Int d1, const Int d2 )
//    {
//        Tensor3<Scal,Int> result ( d0, d1, d2 );
//
//        result.Read(a_);
//        
//        return result;
//    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scal, IntQ Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mreal>> A )
    {
        return Tensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
    template<typename Scal, IntQ Int>
    Tensor3<Scal,Int> from_CubeRef( cref<mma::TensorRef<mint>> A )
    {
        return Tensor3<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1], A.dimensions()[2] );
    }
    
#endif

#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
