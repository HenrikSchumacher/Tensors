#pragma  once

namespace Tensors {
    
#define CLASS SmallMatrixList
    
    template< int M, int N, typename T, typename I>
    class CLASS
    {
    public:
        
        using Tensor_T = Tensor1<T,I>;
        
    private:

        I K = 0;
        
        Tensor_T v [M][N];
        
    public:
//  The big four and half:
        
        CLASS() = default;
        
        //Destructor
        ~CLASS() = default;
        
        explicit CLASS( const I K_ )
        :   K(K_)
        {
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    v[i][j] = Tensor_T(K_);
                }
            }
        }
        
        CLASS( const I K_, const T init )
        :   K(K_)
        {
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    v[i][j] = Tensor_T(K_,init);
                }
            }
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   CLASS( other.K )
        {
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    v[i][j].Read( other.v[i][j].data());
                }
            }
        }
        
        friend void swap(CLASS &A, CLASS &B)
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            std::swap( A.K, B.K );
            
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    swap( A.v[i][j], B.v[i][j] );
                }
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
                        for( I j = 0; j < N; ++j )
                        {
                            v[i][j].Read( other.v[i][j].data());
                        }
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
        
        void BoundCheck( const I i, const I j ) const
        {
            if( (i < 0) || (i > M) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds { 0, " + std::to_string(M-1) +" }.");
            }
            if( (j < 0) || (j > N) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds { 0, " + std::to_string(N-1) +" }.");
            }
        }
        
    public:
//  Access routines
        
        T * data( const I i, const I j )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j].data();
        }
        
        const T * data( const I i, const I j ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j].data();
        }
        
        
        Tensor_T & operator()( const I i, const I j )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j];
        }
        
        const Tensor_T & operator()( const I i, const I j ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j];
        }
    
        T & operator()( const I i, const I j, const I k )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j][k];
        }
        
        const T & operator()( const I i, const I j, const I k ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return v[i][j][k];
        }
        
        
        void SetZero()
        {
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    v[i][j].SetZero();
                }
            }
        }
        
        template<typename S>
        void Read( const S * const * const * const a )
        {
            //Assuming that a is a list of M x N pointers pointing to memory of at least size Dimension(1).
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    copy_cast_buffer( a[i][j], &v[i][j], K );
                }
            }
        }
        
        template<typename S>
        void Write( S * const * const * const a ) const
        {
            //Assuming that a is a list of M pointers pointing to memory of at least size Dimension(1).
            for( I i = 0; i < M; ++i )
            {
                for( I j = 0; j < N; ++j )
                {
                    copy_cast_buffer( &v[i][j], a[i][j], K );
                }
                
            }
        }
        
        template<typename S>
        void Read( const S * const a_ )
        {
            //Assuming that a is a list of size Dimension(1) x M of vectors in interleaved form.
            
            for( I k = 0; k < K; ++ k)
            {
                for( I i = 0; i < M; ++ i)
                {
                    for( I j = 0; j < N; ++j )
                    {
                        v[i][j][k] = a_[(k*M+i)*N+j];
                    }
                }
            }
        }

        template<typename S>
        void Write( S * const a ) const
        {
            //Assuming that a is a list of size Dimension(1) x M of vectors in interleaved form.
            
            for( I k = 0; k < K; ++ k)
            {
                for( I i = 0; i < M; ++ i)
                {
                    for( I j = 0; j < N; ++j )
                    {
                        a[(k*M+i)*N+j] = v[i][j][k];
                    }
                }
            }
        }
        
    public:
        
        static constexpr I Rank()
        {
            return 3;
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
                    return N;
                }
                case 2:
                {
                    return v[0][0].Dimension(0);
                }
                default:
                {
                    return 0;
                }
            }
        }
        
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+std::to_string(M)+","+std::to_string(N)+","+TypeName<T>::Get()+","+TypeName<I>::Get()+">";
        }
    };
    
    
#ifdef LTEMPLATE_H
    
    
    template<int M, int N, typename T, typename I, IsFloat(T), IsInt(I)>
    inline mma::TensorRef<mreal> to_MTensorRef( const CLASS<M,N,T,I> & A )
    {
        const int n = A.Dimension(2);
        
        const T * restrict p [M][N];
        
        for( mint i = 0; i < M; ++i )
        {
            for( mint j = 0; j < N; ++j )
            {
                p[i][j] = A.data(i,j);
            }
        }
        
        auto B = mma::makeCube<mreal>( n, M, N );
        
        mreal * restrict const b = B.data();
        
        for( mint k = 0; k < n; ++k )
        {
            for( mint i = 0; i < M; ++i )
            {
                for( mint j = 0; j < N; ++j )
                {
                    b[(M * k + i) * N + j] = static_cast<mreal>(p[i][j][k]);
                }
            }
        }
        
        return B;
    }

    template<int M, int N, typename J, typename I, IsInt(J), IsInt(I)>
    inline mma::TensorRef<mint> to_MTensorRef( const CLASS<M,N,J,I> & A )
    {
        const mint n = A.Dimension(2);
        
        const J * restrict p [M][N];
        
        for( mint i = 0; i < M; ++i )
        {
            for( mint j = 0; j < N; ++j )
            {
                p[i][j] = A.data(i,j);
            }
        }
        
        auto B = mma::makeCube<mint>( n, M, N );
        
        mint * restrict const b = B.data();
        
        for( mint k = 0; k < n; ++k )
        {
            for( mint i = 0; i < M; ++i )
            {
                for( mint j = 0; j < N; ++j )
                {
                    b[(M * k + i) * N + j] = static_cast<mreal>(p[i][j][k]);
                }
            }
        }
        
        return B;
    }

#endif
    
} // namespace Tensors

#undef CLASS
