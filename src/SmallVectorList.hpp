#pragma once

namespace Tensors {
    
#define CLASS SmallVectorList
    
    template< int M, typename T, typename I>
    class CLASS
    {
    public:
        
        using Tensor_T = Tensor1<T,I>;
        
    private:

        I K = 0;
        
        Tensor_T v [M];

    public:
        
//  The big four and half:
        
        CLASS() = default;
        
        ~CLASS() = default;
        
//        INSERT_MOVE_ASSIGN_CODE(CLASS)
        
        explicit CLASS( const I K_ )
        :   K(K_)
        {
            for( I i = 0; i < M; ++i )
            {
                v[i] = Tensor_T(K_);
            }
        }
        
        CLASS( const I K_, const T init )
        :   K(K_)
        {
            for( I i = 0; i < M; ++i )
            {
                v[i] = Tensor_T(K_,init);
            }
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   CLASS( other.K )
        {
            for( I i = 0; i < M; ++i )
            {
                v[i].Read( other.v[i].data());
            }
        }
        
        friend void swap(CLASS &A, CLASS &B)
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.K, B.K );
            for( I i = 0; i < M; ++i )
            {
                swap( A.v[i], B.v[i] );
            }
        }
        
        // Move constructor
        CLASS( CLASS && other ) noexcept
        :   CLASS()
        {
            swap(*this, other);
        }

        /* Copy assignment operator */
        CLASS & operator=( const CLASS & other )
        {
            if( this != &other )
            {
                if( (K != other.K) )
                {
                    // Use the copy constructor.
                    swap( *this, CLASS(other.K) );
                }
                else
                {
                    for( I i = 0; i < M; ++i )
                    {
                        v[i].Read( other.v[i].data());
                    }
                }
            }
            return *this;
        }

        /* Move assignment operator */
        CLASS & operator=( CLASS && other ) noexcept
        {
            if( this == &other )
            {
                #pragma omp critical
                {
                    wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
                }
            }
            swap( *this, other );
            return *this;
        }
        
    private:
        
        void BoundCheck( const I i ) const
        {
            if( (i < 0) || (i > M) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(M-1) +" }.");
            }
        }
        
    public:

//  Access routines
        
        T * restrict data( const I i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i].data();
        }
        
        const T * restrict data( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i].data();
        }

        
        Tensor_T & operator[]( const I i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i];
        }
        
        const Tensor_T & operator[]( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i];
        }
        
        Tensor_T & operator()( const I i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i];
        }
        
        const Tensor_T & operator()( const I i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i];
        }
    
        T & operator()( const I i, const I k )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i][k];
        }
        
        const T & operator()( const I i, const I k ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return v[i][k];
        }
        
        
        
        void SetZero()
        {
            for( I i = 0; i < M; ++i )
            {
                v[i].SetZero();
            }
        }
        
        template<typename S>
        void Read( const S * restrict const * restrict const a )
        {
            //Assuming that a is a list of M pointers pointing to memory of at least size Dimension(1).
            for( I i = 0; i < M; ++i )
            {
                copy_cast_buffer( a[i], &v[i], K );
            }
        }
        
        template<typename S>
        void Write( S * restrict const * restrict const a )
        {
            //Assuming that a is a list of M pointers pointing to memory of at least size Dimension(1).
            for( I i = 0; i < M; ++i )
            {
                copy_cast_buffer( &v[i], a[i], K );
            }
        }
        
        template<typename S>
        void Read( const S * restrict const a )
        {
            //Assuming that a is a list of size Dimension(1) x M of vectors in interleaved form.
            
            for( I k = 0; k < K; ++ k)
            {
                for( I i = 0; i < M; ++ i)
                {
                    v[i][k] = a[M*k+i];
                }
            }
        }

        template<typename S>
        void Write( S * restrict const a )
        {
            //Assuming that a is a list of size Dimension(1) x M of vectors in interleaved form.
            
            for( I k = 0; k < K; ++ k)
            {
                for( I i = 0; i < M; ++ i)
                {
                    a[M*k+i] = v[i][k];
                }
            }
        }
        
    public:
        
        static constexpr I Rank()
        {
            return 2;
        }
        
        I Dimension( const I k ) const
        {
            switch( k )
            {
                case 0:
                {
                    return M;
                }
                case 1:
                {
                    return v[0].Dimension(0);
                }
                default:
                {
                    return 0;
                }
            }
        }
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+std::to_string(M)+","+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
    };
    
    
    
#ifdef LTEMPLATE_H
    
    
    template<int M, typename T, typename I, IsFloat(T)>
    inline mma::TensorRef<mreal> to_MTensorRef( const CLASS<M,T,I> & A )
    {
        const mint n = A.Dimension(1);
        
        const T * restrict p [M];
        
        for( mint j = 0; j < M; ++j )
        {
            p[j] = A.data(j);
        }
        
        auto B = mma::makeMatrix<mreal>( n, M );
        
        mreal * restrict const b = B.data();
        
        for( mint i = 0; i < n; ++i )
        {
            for( mint j = 0; j < M; ++j )
            {
                b[M * i + j] = static_cast<mreal>(p[j][i]);
            }
        }
        
        return B;
    }

    template<int M, typename J, typename I, IsInt(J)>
    inline mma::TensorRef<mint> to_MTensorRef( const CLASS<M,J,I> & A )
    {
        const mint n = A.Dimension(1);
        
        const J * restrict p [M];
        
        for( mint j = 0; j < M; ++j )
        {
            p[j] = A.data(j);
        }
        
        auto B = mma::makeMatrix<mint>( n, M );
        
        mint * restrict const b = B.data();
        
        for( mint i = 0; i < n; ++i )
        {
            for( mint j = 0; j < M; ++j )
            {
                b[M * i + j] = static_cast<mint>(p[j][i]);
            }
        }
        
        return B;
    }

#endif
    
} // namespace Tensors



#undef CLASS
