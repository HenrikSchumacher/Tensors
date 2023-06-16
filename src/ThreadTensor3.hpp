#pragma once

namespace Tensors {

    template <typename Scal_, typename Int_>
    class ThreadTensor3
    {
        ASSERT_INT(Int_);
        
        using Scal = Scal_;
        using Real   = typename Scalar::Real<Scal_>;
        using Int    = Int_;
        
    private:
        
        Int n = 0;
        std::array<Int,3> dims = {0,0,0};
        std::vector<Tensor2<Scal,Int>> tensors;
        
    public:
        
        ThreadTensor3() = default;
        
        ThreadTensor3( const Int d0, const Int d1, const Int d2 )
        :   n( d0 * d1 * d2 )
        ,   dims{ d0, d1, d2 }
        ,   tensors( std::vector<Tensor2<Scal,Int>> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread] = Tensor2<Scal,Int>( dims[1], dims[2] );
                },
                thread_count
            );
        }
        
        ThreadTensor3( const Int d0, const Int d1, const Int d2, const Scal init )
        :   n( d0 * d1 * d2 )
        ,   dims{ d0, d1, d2 }
        ,   tensors( std::vector<Tensor2<Scal,Int>> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread] = Tensor2<Scal,Int>( dims[1], dims[2], init );
                },
                thread_count
            );
        }
        
        template<typename S>
        ThreadTensor3( const S * a_, const Int d0, const Int d1, const Int d2 )
        :   ThreadTensor3( d0, d1, d2 )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].Read( a_ + thread * dims[1] * dims[2]);
                },
                thread_count
            );
        }

        // Copy constructor
        explicit ThreadTensor3( const ThreadTensor3<Scal,Int> & other )
        :   ThreadTensor3( other.dims[0], other.dims[1], other.dims[2] )
        {
            print(ClassName()+" copy constructor");

            const Int thread_count = dims[0];

            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].Read( other[thread].data() );
                },
                thread_count
            );
        }
        
        // Copy constructor
        template<typename S, typename J, IS_INT(J)>
        explicit ThreadTensor3( const ThreadTensor3<S,J> & other )
        :   ThreadTensor3( other.dims[0], other.dims[1], other.dims[2] )
        {
            print(ClassName()+" copy constructor");
            
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].Read( other[thread].data() );
                },
                thread_count
            );
        }
        
//        // Copy constructor
//        ThreadTensor3( const ThreadTensor3 & other )
//        :
//            tensors( std::vector<Tensor2<Scal,Int>> (other.Dimension(0)) ),
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
//            ParallelDo(
//                [=]( const Int thread )
//                {
//                    tensors[thread] = Tensor2<Scal,Int>( other[thread] );
//                },
//                thread_count
//            );
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
        
        force_inline mut<Scal> data( const Int i )
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data();
        }
        
        force_inline ptr<Scal> data( const Int i ) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data();
        }

        force_inline mut<Scal> data( const Int i, const Int j)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data(j);
        }
        
        force_inline ptr<Scal> data( const Int i, const Int j) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data(j);
        }
        
        force_inline mut<Scal> data( const Int i, const Int j, const Int k)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data(j,k);
        }
        
        force_inline ptr<Scal> data( const Int i, const Int j, const Int k) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i].data(j,k);
        }

        force_inline Scal & operator()( const Int i, const Int j, const Int k)
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i](j,k);
        }
    
        force_inline const Scal & operator()( const Int i, const Int j, const Int k) const
        {
#ifdef TOOLS_DEBUG
            BoundCheck(i);
#endif
            return tensors[i](j,k);
        }
        
        void Fill( const Scal init )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].fill( init );
                },
                thread_count
            );
        }
        
        void SetZero()
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].SetZero();
                },
                thread_count
            );
        }

        void Write( mut<Scal> b ) const
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=]( const Int thread )
                {
                    tensors[thread].Write( b + dims[1] * dims[2] * thread );
                },
                thread_count
            );
        }
        
        template<typename S>
        void Write( const Int i, mut<S> b ) const
        {
            tensors[i].Write( b );
        }
        
        template<typename S>
        void Write( const Int i, const Int j, mut<S> b ) const
        {
            tensors[i].Write( j, b );
        }
        
        template<typename S>
        void Read( const Int i, ptr<S> b )
        {
            tensors[i].Read( b );
        }
        
        template<typename S>
        void Read( const Int i, const Int j, ptr<S> b )
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
        void AddReduce( Tensor2<S,J> & B, const bool add_to ) const
        {
            AddReduce( B.data(), add_to );
        }
        
        template<typename S>
        void AddReduce( S * const B, const bool add_to ) const
        {
            if( add_to )
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
        
        
        Tensor2<Scal,Int> & operator[]( const Int thread )
        {
            return tensors[thread];
        }
        
        const Tensor2<Scal,Int> & operator[]( const Int thread ) const
        {
            return tensors[thread];
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("ThreadTensor3")+"<"+TypeName<Scal>+","+TypeName<Int>+">";
        }
        
    }; // ThreadTensor3
    
    
#ifdef LTEMPLATE_H

    
    template<typename Scal, typename Int, IS_FLOAT(Scal)>
    inline mma::TensorRef<mreal> to_MTensorRef( const ThreadTensor3<Scal,Int> & A )
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
    
    template<typename J, typename Int, IS_INT(J)>
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
