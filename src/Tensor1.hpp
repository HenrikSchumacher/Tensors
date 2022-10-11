#pragma once

namespace Tensors {

#define TENSOR_T Tensor1

    template <typename T, typename I>
    class TENSOR_T
    {

#include "Tensor_Common.hpp"
        
    protected:
        
        std::array<I,1> dims = {0}; // dimensions visible to user
        
    public:
        
        template<typename J, IsInt(J)>
        explicit TENSOR_T( const J d0 )
        :   n    { static_cast<I>(d0) }
        ,   dims { static_cast<I>(d0) }
        {
            allocate();
        }
        
        template<typename S, typename J, IsInt(J)>
        TENSOR_T( const J d0, const S init )
        :   TENSOR_T( static_cast<I>(d0) )
        {
            Fill( static_cast<T>(init) );
        }
        
        template<typename S, typename J, IsInt(J)>
        TENSOR_T( const S * a_, const J d0 )
        :   TENSOR_T( static_cast<I>(d0) )
        {
            Read(a_);
        }
        
    private:
        
        void BoundCheck( const I i ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(dims[0]-1) +" }.");
            }
        }
        
    public:
        
        static constexpr I Rank()
        {
            return static_cast<I>(1);
        }
        

        force_inline T * data( const I i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline const T * data( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i];
        }
        
        force_inline T & operator()(const I i)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline const T & operator()(const I i) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline T & operator[](const I i)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        
        force_inline const T & operator[](const I i) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return a[i];
        }
        

        
        force_inline T & First()
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(0);
#endif
            return a[0];
        }
        
        force_inline const T & First() const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(0);
#endif
            return a[0];
        }

        force_inline T & Last()
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(n-1);
#endif
            return a[n-1];
        }
        
        force_inline const T & Last() const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(n-1);
#endif
            return a[n-1];
        }
        
        void Resize( const I m_ )
        {
            const I m = std::max( static_cast<I>(0),m_);
            
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
        
        
        void Accumulate( I thread_count = 1 )
        {
//            for( I i = 1; i < n; ++i )
//            {
//                a[i] += a[i-1];
//            }
            parallel_accumulate(a, n, thread_count );
        }
        
        force_inline void Scale( const T alpha )
        {
            T * restrict const a_ = a;
            
            #pragma omp simd aligned( a_ : ALIGNMENT )
            for( I i = 0; i < n; ++i )
            {
                a_[i] *= alpha;
            }
        }
        
        inline friend void Scale( TENSOR_T & x, const T alpha )
        {
            x.Scale(alpha);
        }
        
        T Total() const
        {
            T sum = static_cast<T>(0);

            T * restrict const a_ = a;
            
            #pragma omp simd aligned( a_ : ALIGNMENT ) reduction( + : sum )
            for( I i = 0; i < n; ++ i )
            {
                sum += a_[i];
            }

            return sum;
        }
        
        inline friend T Total( const TENSOR_T & x )
        {
            return x.Total();
        }
        
        T Dot( const TENSOR_T & y ) const
        {
            T sum = static_cast<T>(0);
            
            const T * restrict const x_a =   a;
            const T * restrict const y_a = y.a;
            
            if( Size() != y.Size() )
            {
                eprint(ClassName()+"::Dot: Sizes of vectors differ. Doing nothing.");
                return sum;
            }
            const I n_ = std::min( Size(), y.Size() );

            #pragma omp simd aligned( x_a, y_a : ALIGNMENT ) reduction( + : sum )
            for( I i = 0; i < n_; ++ i)
            {
                sum += x_a[i] * y_a[i];
            }

            return sum;
        }
        
        inline friend T Dot( const TENSOR_T & x, const TENSOR_T & y )
        {
            return x.Dot(y);
        }
        
        T Norm() const
        {
            T r2 = 0;
            
            for( I i = 0; i < n; ++i )
            {
                r2 += a[i] * a[i];
            }
            return std::sqrt(r2);
        }
        
        inline friend T Norm( const TENSOR_T & x )
        {
            return x.Norm();
        }
        
        void iota()
        {
            T * restrict const a_ = a;
            
            #pragma omp simd aligned( a_ : ALIGNMENT )
            for( I i = 0; i < n; ++i )
            {
                a_[i] = static_cast<T>(i);
            }
        }

    public:
        
        inline friend std::ostream & operator<<( std::ostream & s, const TENSOR_T & tensor )
        {
            s << tensor.ToString();
            return s;
        }
        
        std::string ToString( const I p = 16) const
        {
            std::stringstream sout;
            sout.precision(p);
            sout << "{ ";
            if( Size() > 0 )
            {
                sout << a[0];
            }
            
            for( I i = 1; i < n; ++i )
            {
                sout << ", " << a[i];
            }
            sout << " }";
            return sout.str();
        }


        
        std::string ToString( const I i_begin, const I i_end, const I p = 16) const
        {
            std::stringstream sout;
            sout.precision(p);
            sout << "{ ";
            if( Size() >= i_end )
            {
                sout << a[i_begin];
            }
            for( I i = i_begin + 1; i < i_end; ++i )
            {
                sout << ", " << a[i];
            }
            sout << " }";
            return sout.str();
        }
        
        static std::string ClassName()
        {
            return "Tensor1<"+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
        
        
    }; // Tensor1
    
    template<typename T, typename I>
    Tensor1<T,I> iota( const I size_ )
    {
        auto v = Tensor1<T,I>(size_);
        
        v.iota();
        
        return v;
    }
    
    template<typename T, typename I, typename S, typename J, IsInt(I), IsInt(J)>
    Tensor1<T,I> ToTensor1( const S * a_, const J d0 )
    {
        Tensor1<T,I> result (static_cast<I>(d0));

        result.Read(a_);
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    template<typename T, typename I>
    Tensor1<T,I> from_VectorRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor1<T,I>( A.data(), A.dimensions()[0] );
    }
    
    template<typename T, typename I>
    Tensor1<T,I> from_VectorRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor1<T,I>( A.data(), A.dimensions()[0] );
    }
    
#endif
    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
} // namespace Tensors
