#pragma once

namespace Tensors
{

    template <typename Scal_, typename Int_, Size_T alignment = CacheLineWidth>
    class ThreadTensor2
    {
    public:

        static_assert(IntQ<Int_>,"");
        
        using Scal = Scal_;
        using Real = typename Scalar::Real<Scal_>;
        using Int  = Int_;
        
        static constexpr Size_T Alignment = alignment;
        
        using Tensor_T = Tensor1<Scal,Int,Alignment>;
        
        
        
    private:
        
        Int n = 0;
        std::array<Int,2> dims = {0,0};
        std::vector<Tensor_T> tensors;
        
    public:
        
        ThreadTensor2() = default;
        
        ThreadTensor2( const Int d0, const Int d1 )
        :   n( d0 * d1 )
        ,   dims{ d0, d1 }
        ,   tensors( std::vector<Tensor_T> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread] = Tensor_T( dims[1] );
                },
                thread_count
            );
        }
        
        ThreadTensor2( const Int d0, const Int d1, const Scal init )
        :   n( d0 * d1 )
        ,   dims{ d0, d1 }
        ,   tensors( std::vector<Tensor_T> ( d0 ) )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread] = Tensor_T( dims[1], init );
                },
                thread_count
            );
        }
        
        template<typename S>
        ThreadTensor2( const S * a_, const Int d0, const Int d1 )
        :   ThreadTensor2( d0, d1 )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread].Read( a_ + thread * dims[1]);
                },
                thread_count
            );
        }

        // Copy constructor
        explicit ThreadTensor2( const ThreadTensor2<Scal,Int> & other )
        :   ThreadTensor2( other.dims[0], other.dims[1] )
        {
            print(ClassName()+" copy constructor");

            const Int thread_count = dims[0];

            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread].Read( other[thread].data() );
                },
                thread_count
            );
        }
        
        // Copy constructor
        template<typename S, typename J>
        explicit ThreadTensor2( const ThreadTensor2<S,J> & other )
        :   ThreadTensor2( other.dims[0], other.dims[1] )
        {
            static_assert(IntQ<J>,"");
            
            print(ClassName()+" copy constructor");
            
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread].Read( other[thread].data() );
                },
                thread_count
            );
        }
        
        friend void swap( ThreadTensor2 & A, ThreadTensor2 & B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
#ifdef BOUND_CHECKS
            print(ClassName()+" swap");
#endif
            swap(A.tensors, B.tensors);
            swap(A.dims[0], B.dims[0]);
            swap(A.dims[1], B.dims[1]);
            swap(A.n , B.n );
        }
        
        // copy-and-swap idiom
        ThreadTensor2 & operator=(ThreadTensor2 B)
        {
            // see https://stackoverflow.com/a/3279550/8248900 for details
#ifdef BOUND_CHECKS
            print(ClassName()+" copy-and-swap");
#endif
            swap(*this, B);

            return *this;
            
        }
        
        // Move constructor
        ThreadTensor2( ThreadTensor2 && other ) noexcept
        :   ThreadTensor2()
        {
#ifdef BOUND_CHECKS
            print(ClassName()+" move constructor");
#endif
            swap(*this, other);
        }
        
        ~ThreadTensor2(){
#ifdef BOUND_CHECKS
            print("~"+ClassName()+" { " + ToString(dims[0]) + ", " + ToString(dims[1]) + " }" );
#endif
        }
        
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(2);
        }

        
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
        
        void BoundCheck( const Int i, const Int j ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + ToString(i) + " is out of bounds [ 0, " + ToString(dims[0]) +" [.");
            }
            
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + ToString(j) + " is out of bounds [ 0, " + ToString(dims[1]) +" [.");
            }
#else
            (void)i;
            (void)j;
#endif
        }
        
        force_inline mptr<Scal> data( const Int i )
        {
            BoundCheck(i);
            
            return tensors[i].data();
        }
        
        force_inline cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return tensors[i].data();
        }

        force_inline mptr<Scal> data( const Int i, const Int j)
        {
            BoundCheck(i);
            
            return tensors[i].data(j);
        }
        
        force_inline cptr<Scal> data( const Int i, const Int j) const
        {
            BoundCheck(i);
            
            return tensors[i].data(j);
        }
        

        force_inline mref<Scal> operator()( const Int i, const Int j )
        {
            BoundCheck(i);
            
            return tensors[i](j);
        }
    
        force_inline cref<Scal> operator()( const Int i, const Int j) const
        {
            BoundCheck(i);
            
            return tensors[i](j);
        }
        
        void Fill( cref<Scal> init )
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [this,&init]( const Int thread )
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
                [this]( const Int thread )
                {
                    tensors[thread].SetZero();
                },
                thread_count
            );
        }

        void Write( mptr<Scal> b ) const
        {
            const Int thread_count = dims[0];
            
            ParallelDo(
                [=,this]( const Int thread )
                {
                    tensors[thread].Write( b + dims[1] * thread );
                },
                thread_count
            );
        }
        
        template<typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            tensors[i].Write( b );
        }
        
        template<typename S>
        void Write( const Int i, const Int j, mptr<S> b ) const
        {
            tensors[i].Write( j, b );
        }
        
        template<typename S>
        void Read( const Int i, cptr<S> b )
        {
            tensors[i].Read( b );
        }
        
        template<typename S>
        void Read( const Int i, const Int j, cptr<S> b )
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
        Tensor1<S,J> AddReduce() const
        {
            Tensor1<S,J> B ( dims[1] );
            
            AddReduce( B.data(), false );
             
            return B;
        }
        
        template<typename S, typename J, Size_T alignment_>
        void AddReduce( mref<Tensor1<S,J,alignment_>> B, const bool add_to ) const
        {
            AddReduce( B.data(), add_to );
        }
        
        template<typename S>
        void AddReduce( mptr<S> B, const bool add_to ) const
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
        
        Int CountNaNs() const
        {
            Int counter = 0;
            for( Int thread = 0 ; thread < dims[0]; ++thread )
            {
                counter += tensors[thread].CountNaNs();
            }
            return counter;
        }
        
        
        mref<Tensor_T> operator[]( const Int thread )
        {
            return tensors[thread];
        }
        
        cref<Tensor_T> operator[]( const Int thread ) const
        {
            return tensors[thread];
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("ThreadTensor2")+"<"+TypeName<Scal>+","+TypeName<Int>+","+ToString(Alignment)+">";
        }
        
    }; // ThreadTensor2
    
    
#ifdef LTEMPLATE_H

    
    template<typename Scal, typename Int,
        class = typename std::enable_if_t<FloatQ<Scal>>
    >
    inline mma::TensorRef<mreal> to_MTensorRef( cref<ThreadTensor2<Scal,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        
        const Int size_ = A.Dimension(1);
        
        for( Int thread = 0; thread < A.Dimension(0); ++thread )
        {
            A[thread].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
    template<typename J, typename Int,
        class = typename std::enable_if_t<IntQ<J>>
    >
    inline mma::TensorRef<mint> to_MTensorRef( cref<ThreadTensor2<J,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        
        const Int size_ = A.Dimension(1);
        
        for( Int thread = 0; thread < A.Dimension(0); ++thread )
        {
            A[thread].Write( &B.data()[size_ * thread] );
        }
        
        return B;
    }
    
#endif
    
} // namespace Tensors
