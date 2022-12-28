//######################################################
//##                     Memory                       ##
//######################################################
            
public:
    
    void SetZero()
    {
        zerofy_buffer<m*n>( &A[0][0] );
    }
    
    void Fill( const Scalar init )
    {
        fill_buffer<m*n>( &A[0][0], init );
    }

//######################################################
//##              Writing to raw buffers              ##
//######################################################

    template<typename T>
    void Write( T * restrict const target ) const
    {
        copy_buffer<m*n>( &A[0][0], target );
    }

    // Copy stride.
    template<typename T>
    void Write( T * restrict const B, const Int ld_B ) const
    {
        for( Int i = 0; i < m; ++i )
        {
            copy_buffer<n>( &A[i][0], &B[n*i] );
        }
    }

    // BLAS-like write-modify method without stride.
    template<ScalarFlag alpha_flag, ScalarFlag beta_flag, typename R, typename S, typename T>
    void Write( const R alpha, const S beta, T * restrict const B) const
    {
        // Writing B = alpha * A + beta * B
        combine_buffers<m*n, alpha_flag, beta_flag>( alpha, &A[0][0], beta, B );
    }

    // BLAS-like write-modify method with stride.
    template<ScalarFlag alpha_flag, ScalarFlag beta_flag, typename R, typename S, typename T>
    void Write( const R alpha, const S beta, T * restrict const B, const Int ldB ) const
    {
        // Writing B = alpha * A + beta * B
        for( Int i = 0; i < m; ++i )
        {
            combine_buffers<n, alpha_flag, beta_flag>( alpha, &A[i][0], beta, &B[ldB*i] );
        }
    }

    // Row-scattered write-modify method.
    // Useful in supernodal arithmetic for sparse matrices.
    template<ScalarFlag alpha_flag, ScalarFlag beta_flag, typename R, typename S, typename T>
    void Write(
        const R                    alpha,
        const S                    beta,
              T   * restrict const B,
        const Int                  ldB,
        const Int * restrict const idx
    ) const
    {
        // Writing B[idx[i]][j] = alpha * A[i][j] + beta * B[idx[i]][j]
        for( Int i = 0; i < m; ++i )
        {
            combine_buffers<n, alpha_flag, beta_flag>( alpha, &A[i][0], beta, &B[ldB*idx[i]] );
        }
    }


//######################################################
//##             Reading from raw buffers             ##
//######################################################


    // We are extremely generous and provide an extra read method without stride.
    template<Op op = Op::Identity, typename T>
    void Read( const T * restrict const B )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Identity )
        {
            copy_buffer<n>( B, &A[0][0] );
        }
        else if constexpr ( op == Op::Conjugate )
        {
            for( Int i = 0; i < n; ++i )
            {
                const Scalar * restrict const B_i = &B[n*i];
                
                for( Int j = 0; j < m; ++j )
                {
                    A[i][j] = conj(B_i[j]);
                }
            }
        }
        else if constexpr ( op == Op::Transpose )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[m*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjugateTranspose )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[m*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = conj(B_j[i]);
                }
            }
        }
    }

    // BLAS-like read-modify method with stride.
    template<Op op = Op::Identity, typename T>
    void Read( const T * restrict const B, const Int ldB )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Identity )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &B[ldB*i], &A[i][0] );
            }
        }
        else if constexpr ( op == Op::Conjugate )
        {
            for( Int i = 0; i < n; ++i )
            {
                const Scalar * restrict const B_i = &B[ldB*i];
                
                for( Int j = 0; j < m; ++j )
                {
                    A[i][j] = conj(B_i[j]);
                }
            }
        }
        else if constexpr ( op == Op::Transpose )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[ldB*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjugateTranspose )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[ldB*j];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = conj(B_j[i]);
                }
            }
        }
    }

    // Scattered read-modify method.
    // Useful in supernodal arithmetic for sparse matrices.
    template<Op op = Op::Identity, typename T>
    void Read( const T * restrict const B, const Int ldB, const Int * restrict const idx )
    {
        // Reading A = op(B)
        if constexpr ( op == Op::Identity )
        {
            for( Int i = 0; i < m; ++i )
            {
                copy_buffer<n>( &B[ldB*idx[i]], &A[i][0] );
            }
        }
        else if constexpr ( op == Op::Conjugate )
        {
            for( Int i = 0; i < n; ++i )
            {
                const Scalar * restrict const B_i = &B[ldB*idx[i]];
                
                for( Int j = 0; j < m; ++j )
                {
                    A[i][j] = conj(B_i[j]);
                }
            }
        }
        else if constexpr ( op == Op::Transpose )
        {
            // TODO: Not sure whether it would be better to swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[ldB*idx[j]];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = B_j[i];
                }
            }
        }
        else if constexpr ( op == Op::ConjugateTranspose )
        {
            // TODO: Not sure whether it would be better to  swap the two loops here...
            for( Int j = 0; j < n; ++j )
            {
                const Scalar * restrict const B_j = &B[ldB*idx[j]];
                
                for( Int i = 0; i < m; ++i )
                {
                    A[i][j] = conj(B_j[i]);
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
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
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
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
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
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
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
    
    
    force_inline void Transpose( CLASS & B ) const
    {
        for( Int j = 0; j < n; ++j )
        {
            for( Int i = 0; i < m; ++i )
            {
                B.A[j][i] = A[i][j];
            }
        }
    }
    
    force_inline CLASS Transpose() const
    {
        CLASS B;
        
        Transpose(B);
        
        return B;
    }

    force_inline void ConjugateTranspose( CLASS & B ) const
    {
        for( Int j = 0; j < n; ++j )
        {
            for( Int i = 0; i < m; ++i )
            {
                B.A[j][i] = conj(A[i][j]);
            }
        }
    }

    force_inline CLASS ConjugateTranspose() const
    {
        CLASS B;
        
        ConjugateTranspose(B);
        
        return B;
    }
    
    force_inline void Conjugate( CLASS & B ) const
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                B.A[i][j] = conj(A[i][j]);
            }
        }
    }

    force_inline CLASS Conjugate() const
    {
        CLASS B;
        
        Conjugate(B);
        
        return B;
    }
