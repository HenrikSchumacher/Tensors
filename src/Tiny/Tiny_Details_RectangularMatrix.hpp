//######################################################
//##                     Memory                       ##
//######################################################
            
public:
    
    void SetZero()
    {
        zerofy_buffer<m*n>( &A[0][0] );
    }
    
    void Fill( cref<Scal> init )
    {
        fill_buffer<m*n>( &A[0][0], init );
    }


    template<typename T>
    void AddTo( mptr<T> B ) const
    {
        add_to_buffer<m*n>( &A[0][0], B );
    }

//######################################################
//##              Writing to raw buffers              ##
//######################################################

    template<Op op = Op::Id, typename T>
    void Write( mptr<T> B ) const
    {
        if constexpr ( op == Op::Id )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &A[i][0], &B[n*i] );
            }
        }
        else if constexpr ( op == Op::Id )
        {
            for( Int j = 0; j < n; ++j )
            {
                for( Int i = 0; i < m; ++i )
                {
                    B[m * j + i ] = A[i][j];
                }
            }
        }
        else
        {
            eprint(ClassName()+"::Write: No implementation for op available.");
        }
    }

    // Copy stride.
    template<Op op = Op::Id, typename T>
    void Write( mptr<T> B, const Int ld_B ) const
    {
        if constexpr ( op == Op::Id )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &A[i][0], &B[ld_B*i] );
            }
        }
        else if constexpr ( op == Op::Id )
        {
            for( Int j = 0; j < n; ++j )
            {
                for( Int i = 0; i < m; ++i )
                {
                    B[ld_B * j + i ] = A[i][j];
                }
            }
        }
        else
        {
            eprint(ClassName()+"::Write: No implementation for op available.");
        }
    }

    // BLAS-like write-modify method without stride.
    template<Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R, typename S, typename T>
    void Write( cref<R> alpha, cref<S> beta, mptr<T> B) const
    {
        // Writing B = alpha * A + beta * B
        combine_buffers<alpha_flag,beta_flag,m*n>(
            alpha, &A[0][0], beta, B
        );
    }

    // BLAS-like write-modify method with stride.
    template<Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R, typename S, typename T>
    void Write( cref<R> alpha, cref<S> beta, mptr<T> B, const Int ldB ) const
    {
        // Writing B = alpha * A + beta * B
        for( Int i = 0; i < m; ++i )
        {
            combine_buffers<alpha_flag,beta_flag,n>(
                alpha, &A[i][0], beta, &B[ldB*i]
            );
        }
    }

    // Row-scattered write-modify method.
    // Useful in supernodal arithmetic for sparse matrices.
    template<Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R, typename S, typename T>
    void Write( cref<R> alpha, cref<S> beta, mptr<T> B, const Int ldB, cptr<Int> idx ) const
    {
        // Writing B[idx[i]][j] = alpha * A[i][j] + beta * B[idx[i]][j]
        for( Int i = 0; i < m; ++i )
        {
            combine_buffers<alpha_flag,beta_flag,n>(
                alpha, &A[i][0], beta, &B[ldB*idx[i]]
            );
        }
    }


//######################################################
//##             Reading from raw buffers             ##
//######################################################


    // We are extremely generous and provide an extra read method without stride.
    template<Op op = Op::Id, typename T>
    void Read( cptr<T> B )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Id )
        {
            copy_buffer<m*n>( B, &A[0][0] );
        }
        else if constexpr ( op == Op::Trans )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[m*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[m*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = Scalar::Conj(B_j[i]);
                }
            }
        }
    }

    // BLAS-like read-modify method with stride.
    template<Op op = Op::Id, typename T>
    void Read( cptr<T> B, const Int ldB )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Id )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &B[ldB*i], &A[i][0] );
            }
        }
        else if constexpr ( op == Op::Trans )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[ldB*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[ldB*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = Scalar::Conj(B_j[i]);
                }
            }
        }
    }

    // Scattered read-modify method.
    // Useful in supernodal arithmetic for sparse matrices.
    template<Op op = Op::Id, typename T>
    void Read( cptr<T> B, const Int ldB, cptr<Int> idx )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Id )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &B[ldB*idx[i]], &A[i][0] );
            }
        }
        else if constexpr ( op == Op::Trans )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[ldB*idx[j]];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            // TODO: Not sure whether it would be better to  swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                cptr<Scal> B_j = &B[ldB*idx[j]];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = Scalar::Conj(B_j[i]);
                }
            }
        }
    }



//######################################################
//##                  Arithmetic                      ##
//######################################################
         
public:
    
    force_inline friend CLASS operator+( const CLASS & x, const CLASS & y )
    {
        CLASS z;
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                z.A[i][j] = x.A[i][j] + y.A[i][j];
            }
        }
        return z;
    }

    
    template<class T>
    force_inline
    std::enable_if_t<
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator+=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
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
    operator-=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
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
    operator*=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] *= lambda;
            }
        }
        
        return *this;
    }
    
    force_inline void Conjugate( CLASS & B ) const
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                B.A[i][j] = Scalar::Conj(A[i][j]);
            }
        }
    }

    force_inline CLASS Conjugate() const
    {
        CLASS B;
        
        Conjugate(B);
        
        return B;
    }
