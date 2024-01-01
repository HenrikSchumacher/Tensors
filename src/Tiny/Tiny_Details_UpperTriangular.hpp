//######################################################
//##                     Memory                       ##
//######################################################
    
public:

    explicit CLASS( cref<Scal> init )
    {
        Fill(init);
    }

    force_inline void SetZero()
    {
        if constexpr ( n > 0 )
        {
            zerofyUpper<0>();
        }
    }

    force_inline void ZerofyUpper()
    {
        if constexpr ( n > 0 )
        {
            zerofyUpper<0>();
        }
    }

    force_inline void ZerofyLower()
    {
        if constexpr ( n > 0 )
        {
            zerofyLower<0>();
        }
    }
    
    force_inline void Fill( cref<Scal> init )
    {
        FillUpper(init);
    }
    

    force_inline void FillUpper( cref<Scal> init )
    {
        if constexpr ( n > 0 )
        {
            fillUpper<0>(init);
        }
    }

    force_inline void FillLower( cref<Scal> init )
    {
        if constexpr ( n > 0 )
        {
            fillLower<0>(init);
        }
    }
    template<typename S>
    force_inline void Read( cptr<S> B )
    {
        if constexpr ( n > 0 )
        {
            read<0>(B);
        }
    }

    template<typename S>
    force_inline void Read( cptr<S> B, const Int ldB )
    {
        if constexpr ( n > 0 )
        {
            read<0>(B,ldB);
        }
    }
    
    template<typename S>
    force_inline void Write( mptr<S> B ) const
    {
        if constexpr ( n > 0 )
        {
            write<0>(B);
        }
    }

    template<typename S>
    force_inline void Write( mptr<S> B, const Int ldB ) const
    {
        if constexpr ( n > 0 )
        {
            write<0>(B,ldB);
        }
    }
    
protected:
    
    // Trying to use compile-time loops to unroll these operations.
    
    template<Int k>
    force_inline void zerofyUpper()
    {
        zerofy_buffer<n-k>( &A[k][k] );
        
        if constexpr ( k+1 < n )
        {
            setZero<k+1>();
        }
    }

    template<Int k>
    force_inline void zerofyLower()
    {
        zerofy_buffer<k>( &A[k][0] );
        
        if constexpr ( k+1 < n )
        {
            zerofyLower<k+1>();
        }
    }
    
    template<Int k>
    force_inline void fillUpper( const Scal init )
    {
        fill_buffer<n-k>( &A[k][k], init );
        
        if constexpr ( k+1 < n )
        {
            fillUpper<k+1>(init);
        }
    }

    template<Int k>
    force_inline void fillLower( const Scal init )
    {
        fill_buffer<k>( &A[k][0], init );
        
        if constexpr ( k+1 < n )
        {
            fillLower<k+1>(init);
        }
    }
    
    template<Int k, typename S>
    force_inline void read( cptr<S> const B )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            read<k+1>( &B[n+1] );
        }
    }

    template<Int k, typename S>
    force_inline void read( cptr<S> B, const Int ldB )
    {
        copy_buffer<n-k>( B, &A[k][k] );
        
        if constexpr ( n > k+1 )
        {
            read<k+1>( &B[ldB+1], ldB );
        }
    }
    
    template<Int k, typename S>
    force_inline void write( mptr<S> B ) const
    {
        copy_buffer<n-k>( &A[k][k], B );
        
        if constexpr ( n > k+1 )
        {
            write<k+1>( &B[n+1] );
        }
    }

    template<Int k, typename S>
    force_inline void write( mptr<S> B, const Int ldB ) const
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

    friend CLASS operator+( cref<CLASS> x, cref<CLASS> y )
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
    
    force_inline void Conjugate( mref<CLASS> B ) const
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                B.A[i][j] = Conj(A[i][j]);
            }
        }
    }
    
    template<class T>
    force_inline
    std::enable_if_t<
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator+=( cref<CLASS<n,T,Int>> B )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator-=( cref<CLASS<n,T,Int>> B )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator*=( cref<CLASS<n,T,Int>> B )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator/=( cref<CLASS<n,T,Int>> B )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator+=( cref<T> lambda )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator-=( cref<T> lambda )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator*=( cref<T> lambda )
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
        
        if constexpr ( Scalar::RealQ<Scal> )
        {
            for( Int i = 0; i < n; ++i )
            {
                for( Int j = i; j < n; ++j )
                {
                    max = Tools::Max( max, Abs(A[i][j]) );
                }
            }
        }
        else
        {
            for( Int i = 0; i < n; ++i )
            {
                for( Int j = i; j < n; ++j )
                {
                    max = Tools::Max( max, AbsSquared(A[i][j]) );
                }
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
                AA += AbsSquared(A[i][j]);
            }
        }
        return Sqrt(AA);
    }
