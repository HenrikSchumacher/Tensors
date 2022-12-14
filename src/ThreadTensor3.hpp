#pragma once

namespace Tensors {

    template <typename Scalar_, typename Int_>
    class ThreadTensor3
    {
        ASSERT_INT(Int_);
        
        using Scalar = Scalar_;
        using Real   = typename ScalarTraits<Scalar_>::Real;
        using Int    = Int_;
        
    private:
        
        Int n = 0;
        std::array<Int,3> dims = {0,0,0};
        std::vector<Tensor2<Scalar,Int>> tensors;
        
    public:
        
        ThreadTensor3() = default;
        
        ThreadTensor3( const Int d0, const Int d1, const Int d2 )
        :   n( d0 * d1 * d2 )
        ,   dims{ d0, d1, d2 }
        ,   tensors( std::vector<Tensor2<Scalar,Int>> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<Scalar,Int>( dims[1], dims[2] );
            }
        }
        
        ThreadTensor3( const Int d0, const Int d1, const Int d2, const Scalar init )
        :   n( d0 * d1 * d2 )
        ,   dims{ d0, d1, d2 }
        ,   tensors( std::vector<Tensor2<Scalar,Int>> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread] = Tensor2<Scalar,Int>( dims[1], dims[2], init );
            }
        }
        
        template<typename S>
        ThreadTensor3( const S * a_, const Int d0, const Int d1, const Int d2 )
        :   ThreadTensor3( d0, d1, d2 )
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].Read( a_ + thread * dims[1] * dims[2]);
            }
        }

        // Copy constructor
        explicit ThreadTensor3( const ThreadTensor3<Scalar,Int> & other )
        :   ThreadTensor3(other.dims)
        {
            print(ClassName()+" copy constructor");
            
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].Read( other[thread].data() );
            }
        }
        
        // Copy constructor
        template<typename S, typename J, IsInt(J)>
        explicit ThreadTensor3( const ThreadTensor3<S,J> & other )
        :   ThreadTensor3(other.dims)
        {
            print(ClassName()+" copy constructor");
            
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].Read( other[thread].data() );
            }
        }
        
//        // Copy constructor
//        ThreadTensor3( const ThreadTensor3 & other )
//        :
//            tensors( std::vector<Tensor2<Scalar,Int>> (other.Dimension(0)) ),
//            n(other.Size())
//        {
//            dims[0] = other.Dimension(0);
//            dims[1] = other.Dimension(1);
//            dims[2] = other.Dimension(2);
//
//            print(ClassName()+" copy constructor");
//
//            const Int thread_count = dims[0];
//
//            #pragma omp parallel for num_threads( thread_count )
//            for( Int thread = 0; thread < thread_count; ++thread )
//            {
//                tensors[thread] = Tensor2<Scalar,Int>( other[thread] );
//            }
//        }
        
        friend void swap(ThreadTensor3 &A, ThreadTensor3 &B) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.tensors, B.tensors);
            swap(A.dims[0], B.dims[0]);
            swap(A.dims[1], B.dims[1]);
            swap(A.dims[2], B.dims[2]);
            swap(A.n , B.n );
        }
        
        // copy-and-swap idiom
        ThreadTensor3 & operator=(ThreadTensor3 B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details
#ifdef BOUND_CHECKS
            print(ClassName()+" copy-and-swap");
#endif
            swap(*this, B);

            return *this;
            
        }
        
        // Move constructor
        ThreadTensor3( ThreadTensor3 && other ) noexcept
        :   ThreadTensor3()
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" move constructor");
#endif
            swap(*this, other);
        }
        
        ~ThreadTensor3(){
#ifdef BOUND_CHECKS
            print("~"+ClassName()+" { " + ToString(dims[0]) + ", " + ToString(dims[1]) + " }" );
#endif
        }
        
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(3);
        }

        
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
        
        force_inline Scalar * data( const Int i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data();
        }
        
        force_inline const Scalar * data( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data();
        }

        force_inline Scalar * data( const Int i, const Int j)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data(j);
        }
        
        force_inline const Scalar * data( const Int i, const Int j) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data(j);
        }
        
        force_inline Scalar * data( const Int i, const Int j, const Int k)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data(j,k);
        }
        
        force_inline const Scalar * data( const Int i, const Int j, const Int k) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i].data(j,k);
        }

        force_inline Scalar & operator()( const Int i, const Int j, const Int k)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i](j,k);
        }
    
        force_inline const Scalar & operator()( const Int i, const Int j, const Int k) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return tensors[i](j,k);
        }
        
        void Fill( const Scalar init )
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].fill( init );
            }
        }
        
        void SetZero()
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].SetZero();
            }
        }

        void Write( Scalar * const b ) const
        {
            const Int thread_count = dims[0];
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                tensors[thread].Write( b + dims[1] * dims[2] * thread );
            }
        }
        
        template<typename S>
        void Write( const Int i, S * const b ) const
        {
            tensors[i].Write( b );
        }
        
        template<typename S>
        void Write( const Int i, const Int j, S * const b ) const
        {
            tensors[i].Write( j, b );
        }
        
        template<typename S>
        void Read( const Int i, const S * const b )
        {
            tensors[i].Read( b );
        }
        
        template<typename S>
        void Read( const Int i, const Int j, const S * const b )
        {
            tensors[i].Read( j, b );
        }
        
    public:
        
        force_inline const Int * Dimensions() const
        {
            return &dims[0];
        }
        
        force_inline Int Dimension( const Int i ) const
        {
            return i < Rank() ? dims[i] : static_cast<Int>(0);
        }
 
        Int Size() const
        {
            return n;
        }
        
        template<typename S, typename J>
        Tensor2<S,J> AddReduce() const
        {
            Tensor2<S,J> B ( dims[1], dims[2] );
            
            AddReduce( B.data(), false );
             
            return B;
        }
        
        template<typename S, typename J>
        void AddReduce( Tensor2<S,J> & B, const bool addto ) const
        {
            AddReduce( B.data(), addto );
        }
        
        template<typename S>
        void AddReduce( S * const B, const bool addto ) const
        {
            if( addto )
            {
                for( Int i = 0; i < dims[0]; ++ i )
                {
                    tensors[i].AddTo( B );
                }
            }
            else
            {
                // Write first slice.
                tensors[0].Write(B);
                
                for( Int i = 1; i < dims[0]; ++ i )
                {
                    tensors[i].AddTo( B );
                }
            }
        }
        
        Int CountNan() const
        {
            Int counter = 0;
            for( Int thread = 0 ; thread < dims[0]; ++thread )
            {
                counter += tensors[thread].CountNan();
            }
            return counter;
        }
        
        
        Tensor2<Scalar,Int> & operator[]( const Int thread )
        {
            return tensors[thread];
        }
        
        const Tensor2<Scalar,Int> & operator[]( const Int thread ) const
        {
            return tensors[thread];
        }
        
    public:
        
        static std::string ClassName()
        {
            return "ThreadTensor3<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    }; // ThreadTensor3
    
    
#ifdef LTEMPLATE_H

    
    template<typename Scalar, typename Int, IsFloat(Scalar)>
    inline mma::TensorRef<mreal> to_MTensorRef( const ThreadTensor3<Scalar,Int> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        
        const Int size_ = A.Dimension(1) * A.Dimension(2);
        
        for( Int thread = 0; thread < A.Dimension(0); ++thread )
        {
            A[thread].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
    template<typename J, typename Int, IsInt(J)>
    inline mma::TensorRef<mint> to_MTensorRef( const ThreadTensor3<J,Int> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        
        const Int size_ = A.Dimension(1) * A.Dimension(2);
        
        for( Int thread = 0; thread < A.Dimension(0); ++thread )
        {
            A[thread].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
#endif
    
} // namespace Tensors
