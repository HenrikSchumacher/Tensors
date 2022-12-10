#pragma once

namespace Tensors {

#define TENSOR_T Tensor1

    template <typename Scalar_, typename Int_>
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
        
        TENSOR_T( const Int d0, const Scalar init )
        :   TENSOR_T( d0 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( const S * a_, const Int d0 )
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
        

        force_inline Scalar * data( const Int i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline const Scalar * data( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline Scalar & operator()(const Int i)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline const Scalar & operator()(const Int i) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline Scalar & operator[](const Int i)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline const Scalar & operator[](const Int i) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        

        
        force_inline Scalar & First()
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(0);
#endif
            return a[0];
        }
        
        force_inline const Scalar & First() const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(0);
#endif
            return a[0];
        }

        force_inline Scalar & Last()
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(n-1);
#endif
            return a[n-1];
        }
        
        force_inline const Scalar & Last() const
        {
#ifdef TENSORS_BOUND_CHECKS
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
        
        void Accumulate( Int thread_count = 1 )
        {
//            for( Int i = 1; i < n; ++i )
//            {
//                a[i] += a[i-1];
//            }
            parallel_accumulate(a, n, thread_count );
        }
        
        Scalar Total() const
        {
            Scalar sum = static_cast<Scalar>(0);

            Scalar * restrict const a_ = a;
            
            #pragma omp simd aligned( a_ : ALIGNMENT ) reduction( + : sum )
            for( Int i = 0; i < n; ++ i )
            {
                sum += a_[i];
            }

            return sum;
        }
        
        inline friend Scalar Total( const TENSOR_T & x )
        {
            return x.Total();
        }
        
        Scalar Dot( const TENSOR_T & y ) const
        {
            Scalar sum = static_cast<Scalar>(0);
            
            const Scalar * restrict const x_a =   a;
            const Scalar * restrict const y_a = y.a;
            
            if( Size() != y.Size() )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return sum;
            }
            const Int n_ = std::min( Size(), y.Size() );

            #pragma omp simd aligned( x_a, y_a : ALIGNMENT ) reduction( + : sum )
            for( Int i = 0; i < n_; ++ i)
            {
                sum += x_a[i] * y_a[i];
            }

            return sum;
        }
        
        inline friend Scalar Dot( const TENSOR_T & x, const TENSOR_T & y )
        {
            return x.Dot(y);
        }
        
        void iota()
        {
            Scalar * restrict const a_ = a;
            
            #pragma omp simd aligned( a_ : ALIGNMENT )
            for( Int i = 0; i < n; ++i )
            {
                a_[i] = static_cast<Scalar>(i);
            }
        }

    public:
        
        inline friend std::ostream & operator<<( std::ostream & s, const TENSOR_T & tensor )
        {
            s << tensor.ToString();
            return s;
        }
        
        std::string ToString( const int p = 16) const
        {
            std::stringstream sout;
            sout << "{ ";
            if( Size() > 0 )
            {
                sout << Tools::ToString(a[0],p);
            }
            for( Int i = 1; i < n; ++i )
            {
                sout << ", " << Tools::ToString(a[i],p);
            }
            sout << " }";
            return sout.str();
        }


        
        std::string ToString( const Int i_begin, const Int i_end, const int p = 16) const
        {
            std::stringstream sout;
            sout << "{ ";
            if( Size() >= i_end )
            {
                sout << Tools::ToString(a[i_begin],p);
            }
            for( Int i = i_begin + 1; i < i_end; ++i )
            {
                sout << ", " << Tools::ToString(a[i],p);
            }
            sout << " }";
            return sout.str();
        }
        
        static std::string ClassName()
        {
            return "Tensor1<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
        }
        
        
    }; // Tensor1
    
    template<typename Scalar, typename Int>
    Tensor1<Scalar,Int> iota( const Int size_ )
    {
        auto v = Tensor1<Scalar,Int>(size_);
        
        v.iota();
        
        return v;
    }
    
    template<typename Scalar, typename Int, typename S>
    Tensor1<Scalar,Int> ToTensor1( const S * a_, const Int d0 )
    {
        Tensor1<Scalar,Int> result (d0);

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename Scalar, typename Int>
    Tensor1<Scalar,Int> from_VectorRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor1<Scalar,Int>( A.data(), A.dimensions()[0] );
    }
    
    template<typename Scalar, typename Int>
    Tensor1<Scalar,Int> from_VectorRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor1<Scalar,Int>( A.data(), A.dimensions()[0] );
    }
    
#endif
    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
