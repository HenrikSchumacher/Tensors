//######################################################
//##                     Memory                       ##
//######################################################
    
public:

    explicit CLASS( const Scalar init )
    {
        Fill(init);
    }

    force_inline void SetZero()
    {
        if constexpr ( n > 0 )
        {
            setZero<0>();
        }
    }
    
    force_inline void Fill( const Scalar init )
    {
        if constexpr ( n > 0 )
        {
            fill<0>(init);
        }
    }
    
    force_inline void Read( const Scalar * const B )
    {
        if constexpr ( n > 0 )
        {
            read<0>(B);
        }
    }

    force_inline void Read( const Scalar * const B, const Int ldB )
    {
        if constexpr ( n > 0 )
        {
            read<0>(B,ldB);
        }
    }
    
    force_inline void Write( Scalar * B ) const
    {
        if constexpr ( n > 0 )
        {
            write<0>(B);
        }
    }

    force_inline void Write( Scalar * B, const Int ldB ) const
    {
        if constexpr ( n > 0 )
        {
            write<0>(B,ldB);
        }
    }
    
protected:
    
    // Trying to use compile-time loops to unroll these operations.
    
    template<Int k>
    force_inline void setZero()
    {
        zerofy_buffer<n-k>( &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            setZero<k+1>();
        }
    }
    
    template<Int k>
    force_inline void fill( const Scalar init )
    {
        fill_buffer<n-k>( &A[k][k],init );
        
        if constexpr ( n > k+1 )
        {
            fill<k+1>(init);
        }
    }
    
    template<Int k>
    force_inline void read( ptr<Scalar> const B )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            read<k+1>( &B[n+1] );
        }
    }

    template<Int k>
    force_inline void read( ptr<Scalar> B, const Int ldB )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            read<k+1>( &B[ldB+1], ldB );
        }
    }
    
    template<Int k>
    force_inline void write( mut<Scalar> B ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k+1 )
        {
            write<k+1>( &B[n+1] );
        }
    }

    template<Int k>
    force_inline void write( mut<Scalar> B, const Int ldB ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k+1 )
        {
            write<k+1>( &B[ldB+1], ldB );
        }
    }



//######################################################
//##                  Arithmetic                      ##
//######################################################
       
public:

    friend CLASS operator+( const CLASS & x, const CLASS & y )
    {
        CLASS z;
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                z.A[i][j] = x.A[i][j] + y.A[i][j];
            }
        }
        return z;
    }
    
    force_inline void Conjugate( CLASS & B ) const
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                B.A[i][j] = conj(A[i][j]);
            }
        }
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator+=( const CLASS<n,T,Int> & B )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] += B.A[i][j];
            }
        }
        return *this;
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator-=( const CLASS<n,T,Int> & B )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] -= B.A[i][j];
            }
        }
        return *this;
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator*=( const CLASS<n,T,Int> & B )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] *= B.A[i][j];
            }
        }
        return *this;
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator/=( const CLASS<n,T,Int> & B )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] /= B.A[i][j];
            }
        }
        return *this;
    }
    
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator+=( const T lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] += lambda;
            }
        }
        
        return *this;
    }

    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator-=( const T lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] -= lambda;
            }
        }
        
        return *this;
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator*=( const T lambda )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A[i][j] *= lambda;
            }
        }
        
        return *this;
    }


    force_inline Real MaxNorm() const
    {
        Real max = 0;
        
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                max = std::max( max, std::abs(A[i][j]));
            }
        }
        return max;
    }

    force_inline Real FrobeniusNorm() const
    {
        Real AA = 0;
        
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                AA += abs_squared(A[i][j]);
            }
        }
        return std::sqrt(AA);
    }
