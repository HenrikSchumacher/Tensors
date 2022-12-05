#pragma once

namespace Tensors
{
    namespace Small
    {
        template< int m_, int n_, typename Scalar_, typename Int_>
        struct Matrix
        {
        public:
            
            using Scalar = Scalar_;
            using Real   = typename ScalarTraits<Scalar_>::RealType;
            using Int    = Int_;
            
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
            static constexpr Scalar zero            = 0;
            static constexpr Scalar half            = 0.5;
            static constexpr Scalar one             = 1;
            static constexpr Scalar two             = 2;
            static constexpr Scalar three           = 3;
            static constexpr Scalar four            = 4;
            static constexpr Real eps               = std::numeric_limits<Real>::min();
            static constexpr Real infty             = std::numeric_limits<Real>::max();
            
            Matrix() = default;
            
            ~Matrix() = default;
            
            explicit Matrix( const Scalar init )
            :   A {{init}}
            {}
            
            Matrix( const Matrix & other )
            {
                *this = other;
            }
            
        protected:
            
            Scalar A [m][n];
            
        public:
            
            Scalar * restrict data()
            {
                return &A[0][0];
            }
            
            const Scalar * restrict data() const
            {
                return &A[0][0];
            }
            
            void SetZero()
            {
                zerofy_buffer<m * n>( &A[0][0] );
            }
            
            void Fill( const Scalar init )
            {
                fill_buffer<m * n>( &A[0][0], init );
            }
            
            Scalar & operator()( const Int i, const Int j )
            {
                return A[i][j];
            }
            
            const Scalar & operator()( const Int i, const Int j ) const
            {
                return A[i][j];
            }
            
            Scalar * operator[]( const Int i )
            {
                return A[i];
            }
            
            const Scalar * operator[]( const Int i ) const
            {
                return A[i];
            }
            
            friend Matrix operator+( const Matrix & x, const Matrix & y )
            {
                Matrix z;
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            void operator+=( const Matrix & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
            }
            
            void operator-=( const Matrix & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= B.A[i][j];
                    }
                }
            }
            
            void operator*=( const Matrix & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
            }
            
            void operator/=( const Matrix & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] /= B.A[i][j];
                    }
                }
            }
            
            Matrix & operator=( const Matrix & B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = B.A[i][j];
                    }
                }
                return *this;
            }
            
            void Transpose( Matrix<n,m,Scalar,Int> & B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B.A[j][i] = A[i][j];
                    }
                }
            }
            
            void ConjugateTranspose( Matrix<n,m,Scalar,Int> & B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B.A[j][i] = conj(A[i][j]);
                    }
                }
            }
            
            void Conjugate( Matrix<m,n,Scalar,Int> & B ) const
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        B.A[i][j] = conj(A[i][j]);
                    }
                }
            }
            
            Real MaxNorm() const
            {
                Real max = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        max = std::max( max, std::abs(A[i][j]));
                    }
                }
                return max;
            }
            
            Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        AA += abs_squared(A[i][j]);
                    }
                }
                return std::sqrt(AA);
            }
            
            
            void Write( Scalar * target ) const
            {
                copy_buffer<m * n>( &A[0][0], target );
            }
            
            void Read( Scalar const * const source )
            {
                copy_buffer<m * n>( source, &A[0][0] );
            }
            
            std::string ToString( const Int p = 16) const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\t{ ";
                if( (m > 0) && (n > 0) )
                {
                    sout << Tools::ToString(A[0][0],p);
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << Tools::ToString(A[0][j],p);
                    }
                    for( Int i = 1; i < m; ++i )
                    {
                        sout << " },\n\t{ ";
                        
                        sout << Tools::ToString(A[i][0],p);
                        
                        for( Int j = 1; j < n; ++j )
                        {
                            sout << ", " << Tools::ToString(A[i][j],p);
                        }
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            inline friend std::ostream & operator<<( std::ostream & s, const Matrix & M )
            {
                s << M.ToString();
                return s;
            }
            
        public:
            
            static constexpr Int Dimension( const Int i )
            {
                if( i == 0 )
                {
                    return m;
                }
                if( i == 1 )
                {
                    return n;
                }
                return static_cast<Int>(0);
            }
            
            static std::string ClassName()
            {
                return "Matrix<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
        
    } // namespace Small
        
    template< int M, int N, typename Scalar, typename Int >
    void Dot(
        const Small::Matrix<M,N,Scalar,Int> & A,
        const Small::Vector<N,Scalar,Int> & x,
              Small::Vector<M,Scalar,Int> & y
    )
    {
        for( Int i = 0; i < M; ++i )
        {
            Scalar y_i (0);
            
            for( Int j = 0; j < i; ++j )
            {
                y_i += A[j][i] * x[j];
            }
            for( Int j = 0; j < N; ++j )
            {
                y_i += A[i][j] * x[j];
            }
            
            y[i] = y_i;
        }
    }
    
    template< int M, int K, int N, typename Scalar, typename Int >
    void Dot(
        const Small::Matrix<M,K,Scalar,Int> & A,
        const Small::Matrix<K,N,Scalar,Int> & B,
              Small::Matrix<M,N,Scalar,Int> & C
    )
    {
        C.SetZero();
        
        for( Int i = 0; i < M; ++i )
        {
            for( Int k = 0; k < K; ++k )
            {
                for( Int j = 0; j < N; ++j )
                {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
    
} // namespace Tensors

