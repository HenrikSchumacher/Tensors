///######################################################
///##                     Memory                       ##
///######################################################
            
public:

    void SetZero()
    {
        zerofy_buffer<m*n>( &A[0][0] );
    }
    
    template<typename T>
    void Fill( cref<T> init )
    {
        fill_buffer<m*n>( &A[0][0], static_cast<T>(init) );
    }


    template<typename B_T>
    void AddTo( mptr<B_T> B ) const
    {
        add_to_buffer<m*n>( &A[0][0], B );
    }



///######################################################
///##             Reading from raw buffers             ##
///######################################################

    /// BLAS-like read-modify method _with stride_.
    /// Reads the full matrix.
    template<
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void Read(
        cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB
    )
    {
        /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
        /// `opA` can only be `Op::Id` or `Op::Conj`.
        
        constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
        
        if constexpr ( NotTransposedQ(opB) )
        {
            for( Int i = 0; i < m; ++i )
            {
                combine_buffers<beta_flag,alpha_flag,n,Sequential,op,opA>(
                    beta, &B[ldB*i], alpha, &A[i][0]
                );
            }

        }
        else if constexpr ( TransposedQ(opB) )
        {
            // TODO: Compare with reverse ordering of loops.
            
            for( Int j = 0; j < n; ++j )
            {
                mptr<B_T> B_j = &B[ldB * j];
                
                for( Int i = 0; i < m; ++i )
                {
                    combine_scalars<beta_flag,alpha_flag,op,opA>(
                        beta, B_j[i], alpha, A[i][j]
                    );
                }
            }
        }
    }

    /// BLAS-like read-modify method _without stride_.
    template<
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void Read(
        cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B
    )
    {
        /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
        /// `opA` can only be `Op::Id` or `Op::Conj`.

        constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
        
        if constexpr ( NotTransposedQ(opB) )
        {
            combine_buffers<beta_flag,alpha_flag,m*n,Sequential,op,opA>(
                beta, B, alpha, &A[0][0]
            );
        }
        else if constexpr ( TransposedQ(opB) )
        {
            // TODO: Compare with reverse ordering of loops.
            
            for( Int j = 0; j < n; ++j )
            {
                mptr<B_T> B_j = &B[n * j];
                
                for( Int i = 0; i < m; ++i )
                {
                    combine_scalars<alpha_flag,beta_flag,op,opA>(
                        beta, B_j[i], alpha, A[i][j]
                    );
                }
            }
        }
    }

//    /// Basic copy routine _without stride_.
//    template<typename B_T>
//    void Read( cptr<B_T> B)
//    {
//        copy_buffer<m*n>(B,&A[0][0]);
//    }


    /// BLAS-like read-modify method _without stride_.
    template<Op opB = Op::Id, typename B_T>
    void Read( cptr<B_T> B )
    {
        /// Compute `opB(B)` and store it in `A`.
        
        Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
            Scal(0), Scal(1), B
        );
    }

    /// BLAS-like read-modify method _with stride_.
    template<Op opB = Op::Id, typename B_T>
    void Read( cptr<B_T> B, const Int ldB )
    {
        /// Compute `opB(B)` and store it in `A`.
        
        Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
            Scal(0), Scal(1), B, ldB
        );
    }


    /// BLAS-like read-modify method _with stride_.
    /// Reads only the top left portion.
    /// It is meant to read from the right and bottom boundaries of a large matrix.
    template<
        bool chop_m_Q, bool chop_n_Q,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void ReadChopped(
        const Int m_max, const Int n_max,
        cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
    )
    {
        /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
        /// `opA` can only be `Op::Id` or `Op::Conj`.
        
        constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
        
        const Int m_c = Min(m,m_max);
        const Int n_c = Min(n,n_max);

        if constexpr ( NotTransposedQ(opB) )
        {
            for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
            {
                combine_buffers<beta_flag,alpha_flag,COND(chop_n_Q,0,n),Sequential,op,opA>(
                    beta, &B[ldB*i], alpha, &A[i][0], n_c
                );
            }
        }
        else if constexpr ( TransposedQ(opB) )
        {
            // TODO: Compare with reverse ordering of loops.
            
            /// TODO:
            /// I think best performance should be obtained
            /// by making the fixed-size loop the inner loop.
            
            for( Int j = 0; j < COND(chop_n_Q,n_c,n); ++j )
            {
                mptr<B_T> B_j = &B[ldB * j];
                
                for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                {
                    combine_scalars<beta_flag,alpha_flag,op,opA>(
                        beta, B_j[i], alpha, A[i][j]
                    );
                }
            }
        }
    }

//    // Scattered read-modify method.
//    // Might be useful in supernodal arithmetic for sparse matrices.
//    template<Scalar::Flag a_flag, Op op, typename alpha_T, typename B_T>
//    void Read( const alpha_T alpha, cptr<B_T> B, const Int ldB, cptr<Int> idx  )
//    {
//        // Reading A = alpha * op(B)
//
//        if constexpr ( op == Op::Id )
//        {
//            if constexpr ( a_flag == Scalar::Flag::Plus )
//            {
//                for( Int i = 0; i < m; ++i )
//                {
//                    copy_buffer<n>( &B[ldB*idx[i]], &A[i][0] );
//                }
//            }
//        }
//        else if constexpr ( op == Op::Conj )
//        {
//            for( Int i = 0; i < m; ++i )
//            {
//                for( Int j = 0; j < n; ++j )
//                {
//                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B[i][j] );
//                }
//            }
//        }
//        else
//        {
//            // TODO: Not sure whether it would be better to swap the two loops here...
//            for( Int j = 0; j < n; ++j )
//            {
//                cptr<B_T> B_j = &B[ldB*idx[j]];
//
//                for( Int i = 0; i < m; ++i )
//                {
//                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B_j[i] );
//                }
//            }
//        }
//    }

///######################################################
///##              Writing to raw buffers              ##
///######################################################

    /// BLAS-like write-modify method _with stride_.
    template<
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag, 
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void Write(
        cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB
    ) const
    {
        /// Computes `B = alpha * opA(A) + beta * opB(B)`.
        /// `opB` can only be `Op::Id` or `Op::Conj`.
        
        constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
        
        if constexpr ( NotTransposedQ(opA) )
        {
            for( Int i = 0; i < m; ++i )
            {
                combine_buffers<alpha_flag,beta_flag,n,Sequential,op,opB>(
                    alpha, &A[i][0], beta, &B[ldB*i]
                );
            }
        }
        else if constexpr ( TransposedQ(opA) )
        {
            // TODO: Compare with reverse ordering of loops.
            for( Int j = 0; j < n; ++j )
            {
                mptr<B_T> B_j = &B[ldB * j];
                
                for( Int i = 0; i < m; ++i )
                {
                    combine_scalars<alpha_flag,beta_flag,op,opB>(
                        alpha, A[i][j], beta, B_j[i]
                    );
                }
            }
        }
    }



    /// BLAS-like write-modify method _without stride_.
    template<
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void Write(
        cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B
    ) const
    {
        /// Computes `B = alpha * opA(A) + beta * opB(B)`.
        /// `opB` can only be `Op::Id` or `Op::Conj`.

        constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
        
        if constexpr ( NotTransposedQ(opA) )
        {
            /// This is the only reason we use a version without stride:
            /// If the whole matrix is stored contiguously, we can vectorize over
            /// row-ends!
            combine_buffers<alpha_flag,beta_flag,m*n,Sequential,op,opB>(
                alpha, &A[0][0], beta, B
            );
        }
        else if constexpr ( TransposedQ(opA) )
        {
            /// I think that no real optimizations can be made here.
            /// Nonetheless, we let the compiler know that the stride is a
            /// compile-time constant. But since the alignment of B is unknown to
            /// the compiler, this might be quite useless.

            // TODO: Compare with reverse ordering of loops.
            
            for( Int j = 0; j < n; ++j )
            {
                mptr<B_T> B_j = &B[n * j];
                
                for( Int i = 0; i < m; ++i )
                {
                    combine_scalars<alpha_flag,beta_flag,op,opB>(
                        alpha, A[i][j], beta, B_j[i]
                    );
                }
            }
        }
    }


    /// BLAS-like write-modify method _with stride_.
    template<Op opA = Op::Id, typename B_T>
    void Write( mptr<B_T> B, const Int ldB ) const
    {
        /// Compute `B = opA(A)`.

        Write<Scalar::Flag::Plus,Scalar::Flag::Zero,opA,Op::Id>( B_T(1), B_T(0), B, ldB );
    }

    /// BLAS-like write-modify method _without stride_.
    template<Op op = Op::Id, typename B_T>
    void Write( mptr<B_T> B ) const
    {
        /// B = opA(A)
        Write<Scalar::Flag::Plus,Scalar::Flag::Zero,op,Op::Id>( B_T(1), B_T(0), B );
    }

//    /// Row-scattered write-modify method.
//    /// Might be useful in supernodal arithmetic for sparse matrices.
//    template<
//        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
//        typename alpha_T, typename beta_T, typename B_T
//    >
//    void Write( cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB, cptr<Int> idx ) const
//    {
//        
//        // Writing B[idx[i]][j] = alpha * A[i][j] + beta * B[idx[i]][j]
//        for( Int i = 0; i < m; ++i )
//        {
//            combine_buffers<alpha_flag,beta_flag,n>(
//                alpha, &A[i][0], beta, &B[ldB*idx[i]]
//            );
//        }
//    }

    /// BLAS-like write-modify method _with stride_.
    /// Writes only the top left portion.
    /// It is meant to write to the right and bottom boundaries of a large matrix.
    template<
        bool chop_m_Q, bool chop_n_Q,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        Op opA = Op::Id, Op opB = Op::Id,
        typename alpha_T, typename beta_T, typename B_T
    >
    void WriteChopped(
        const Int m_max, const Int n_max,
        cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
    ) const
    {
        /// Computes `B = alpha * opA(A) + beta * opB(B)`.
        /// `opB` can only be `Op::Id` or `Op::Conj`.
        
        constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
        
        const Int m_c = Min( m, m_max );
        const Int n_c = Min( n, n_max );
        
        if constexpr ( NotTransposedQ(opA) )
        {
            for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
            {
                combine_buffers<alpha_flag,beta_flag,COND(chop_n_Q,0,n),Sequential,op,opB>(
                    alpha, &A[i][0], beta, &B[ldB*i], n_c
                );
            }
        }
        else if constexpr ( TransposedQ(opA) )
        {
            // TODO: Compare with reverse ordering of loops.
            for( Int j = 0; j < COND(chop_n_Q,n_c,n); ++j )
            {
                mptr<B_T> B_j = &B[ldB * j];
                
                for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                {
                    combine_scalars<alpha_flag,beta_flag,op,opB>(
                        alpha, A[i][j], beta, B_j[i]
                    );
                }
            }
        }
    }




///######################################################
///##                  Arithmetic                      ##
///######################################################
         
public:

    
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

