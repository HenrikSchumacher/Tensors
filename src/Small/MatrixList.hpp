#pragma once

namespace Small
{
    namespace Small
    {

        template< int M, int N, typename Real, typename Int>
        class MatrixList
        {
        public:
            
            using Tensor_T = Tensor1<Real,Int>;
            
        private:
            
            Int K = 0;
            
            Tensor_T v [M][N];
            
        public:
            //  The big four and half:
            
            MatrixList() = default;
            
            //Destructor
            ~MatrixList() = default;
            
            explicit MatrixList( const Int K_ )
            :   K(K_)
            {
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        v[i][j] = Tensor_T(K_);
                    }
                }
            }
            
            MatrixList( const Int K_, const Real init )
            :   K(K_)
            {
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        v[i][j] = Tensor_T(K_,init);
                    }
                }
            }
            
            // Copy constructor
            MatrixList( const MatrixList & other )
            :   MatrixList( other.K )
            {
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        v[i][j].Read( other.v[i][j].data());
                    }
                }
            }
            
            friend void swap(MatrixList &A, MatrixList &B)
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                std::swap( A.K, B.K );
                
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        swap( A.v[i][j], B.v[i][j] );
                    }
                }
            }
            
            // Move constructor
            MatrixList( MatrixList && other ) noexcept
            :   MatrixList()
            {
                swap(*this, other);
            }
            
            /* Copy assignment operator */
            MatrixList & operator=( const MatrixList & other )
            {
                if( this != &other )
                {
                    if( (K != other.K) )
                    {
                        // Use the copy constructor.
                        swap( *this, MatrixList(other.K) );
                    }
                    else
                    {
                        for( Int i = 0; i < M; ++i )
                        {
                            for( Int j = 0; j < N; ++j )
                            {
                                v[i][j].Read( other.v[i][j].data());
                            }
                        }
                    }
                }
                return *this;
            }
            
            /* Move assignment operator */
            MatrixList & operator=( MatrixList && other ) noexcept
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
            
            
            //  Access routines
            
            Real * restrict data( const Int i, const Int j )
            {
                return v[i][j].data();
            }
            
            const Real * restrict data( const Int i, const Int j ) const
            {
                return v[i][j].data();
            }
            
            void SetZero()
            {
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        v[i][j].SetZero();
                    }
                }
            }
            
            Tensor_T & operator()( const Int i, const Int j )
            {
                return v[i][j];
            }
            
            const Tensor_T & operator()( const Int i, const Int j ) const
            {
                return v[i][j];
            }
            
            Real & operator()( const Int i, const Int j, const Int k )
            {
                return v[i][j][k];
            }
            
            const Real & operator()( const Int i, const Int j, const Int k ) const
            {
                return v[i][j][k];
            }
            
            
            template<typename S>
            void Read( const S * const * const * const a )
            {
                //Assuming that a is a list of M x N pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        copy_cast_buffer( a[i][j], &v[i][j], K );
                    }
                }
            }
            
            template<typename S>
            void Write( S * const * const * const a ) const
            {
                //Assuming that a is a list of M pointers pointing to memory of at least size Dimension(1).
                for( Int i = 0; i < M; ++i )
                {
                    for( Int j = 0; j < N; ++j )
                    {
                        copy_cast_buffer( &v[i][j], a[i][j], K );
                    }
                    
                }
            }
            
            template<typename S>
            void Read( const S * const a_ )
            {
                //Assuming that a is a list of size Dimension(1) x M of vectors in interleaved form.
                
                for( Int k = 0; k < K; ++ k)
                {
                    for( Int i = 0; i < M; ++ i)
                    {
                        for( Int j = 0; j < N; ++j )
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
                
                for( Int k = 0; k < K; ++ k)
                {
                    for( Int i = 0; i < M; ++ i)
                    {
                        for( Int j = 0; j < N; ++j )
                        {
                            a[(k*M+i)*N+j] = v[i][j][k];
                        }
                    }
                }
            }
            
        public:
            
            static constexpr Int Rank()
            {
                return 3;
            }
            
            Int Dimension( const Int k ) const
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
                return "MatrixList<"+std::to_string(M)+","+std::to_string(N)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
            }
        };
        
        
    #ifdef LTEMPLATE_H
        
        
        template<int M, int N, typename T, typename I, IsFloat(T)>
        inline mma::TensorRef<mreal> to_MTensorRef( const Small::MatrixList<M,N,T,I> & A )
        {
            const mint n = A.Dimension(2);
            
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
        
        template<int M, int N, typename J, typename I, IsInt(J)>
        inline mma::TensorRef<mint> to_MTensorRef( const Small::MatrixList<M,N,J,I> & A )
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
    
    } // namespace Small
    
} // namespace Tensors
