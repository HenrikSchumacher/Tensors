//######################################################
//##                     Memory                       ##
//######################################################
    
public:
    
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
    
    force_inline void Read( const Scalar * const source )
    {
        if constexpr ( n > 0 )
        {
            read<0>(source);
        }
    }
    
    force_inline void Write( Scalar * target ) const
    {
        if constexpr ( n > 0 )
        {
            write<0>(target);
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
    force_inline void read( const Scalar * restrict const B )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            read<k+1>( &B[n+1] );
        }
    }
    
    template<Int k>
    force_inline void write( Scalar * restrict const B ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k+1 )
        {
            write<k+1>( &B[n+1] );
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
    
    void Conjugate( CLASS & B ) const
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



    Real MaxNorm() const
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

    Real FrobeniusNorm() const
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
