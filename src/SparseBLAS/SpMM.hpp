public:

    template<Size_T NRHS = VarSize, typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMM(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha_, cptr<T_in>  X, const Int ldX,
        cref<S_out> beta_,  mptr<T_out> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.

        using alpha_T = std::conditional_t< Scalar::RealQ<R_out>, Scalar::Real<T_out>, T_out>;
        using beta_T  = std::conditional_t< Scalar::RealQ<S_out>, Scalar::Real<T_out>, T_out>;
        
        StaticParameterCheck<alpha_T,T_in,beta_T,T_out>();
        
        const alpha_T alpha = ( rp[m] > 0 ) ? scalar_cast<T_out>(alpha_) : scalar_cast<R_out>(0);
        const beta_T  beta  = scalar_cast<T_out>(beta_);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == static_cast<alpha_T>(0) )
        {
            if( beta == static_cast<beta_T>(0) )
            {
                if( ldY == nrhs )
                {
                    zerofy_buffer<VarSize,Parallel>( Y, m * nrhs, job_ptr.ThreadCount() );
                }
                else
                {
                    ParallelDo(
                        [&]( const Int i )
                        {
                            zerofy_buffer<NRHS,Sequential>( &Y[ldY*i], nrhs );
                        },
                        m, job_ptr.ThreadCount()
                    );
                }
            }
            else if( beta == static_cast<beta_T>(1) )
            {
                // Do nothing.
            }
            else
            {
                if( ldY == nrhs )
                {
                    scale_buffer<VarSize,Parallel>( beta, Y, m * nrhs, job_ptr.ThreadCount() );
                }
                else
                {
                    ParallelDo(
                        [&]( const Int i )
                        {
                            scale_buffer<NRHS,Sequential>( beta, &Y[ldY*i], nrhs );
                        },
                        m, job_ptr.ThreadCount()
                    );
                }
            }
            return;
        }
        
        if( a != nullptr )
        {
            if( alpha == static_cast<alpha_T>(1) )
            {
                if( beta == static_cast<beta_T>(0) )
                {
                    SpMM_impl<Generic,One    ,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<beta_T>(1) )
                {
                    SpMM_impl<Generic,One    ,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else
                {
                    SpMM_impl<Generic,One    ,Generic,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<beta_T>(1) )
                {
                    SpMM_impl<Generic,Generic,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<beta_T>(0) )
                {
                    SpMM_impl<Generic,Generic,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else
                {
                    SpMM_impl<Generic,Generic,Generic,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
            }
        }
        else // a == nullptr
        {
            if( alpha == static_cast<alpha_T>(1) )
            {
                if( beta == static_cast<beta_T>(0) )
                {
                    SpMM_impl<One    ,One    ,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<beta_T>(1) )
                {
                    SpMM_impl<One    ,One    ,One    ,NRHS>(
                    rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else
                {
                    SpMM_impl<One    ,One    ,Generic,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<beta_T>(1) )
                {
                    SpMM_impl<Generic,Generic,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<beta_T>(0) )
                {
                    SpMM_impl<Generic,Generic,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else
                {
                    SpMM_impl<Generic,Generic,Generic,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
            }
        }
    }

private:

    template<Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, Size_T NRHS = VarSize,
        typename R_out, typename T_in, typename S_out, typename T_out
    >
    force_flattening void SpMM_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha,  cptr<T_in>  X, const Int ldX,
        cref<S_out> beta,   mptr<T_out> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {
        std::string tag = std::string(ClassName()+"::SpMM_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +TypeName<R_out>+","
            +TypeName<T_in >+","
            +TypeName<S_out>+","
            +TypeName<T_out>+ ","
            + ( ( NRHS == VarSize ) ? std::string("VarSize") : ToString(NRHS) )
            +">("+ToString(nrhs)+")";
        
        ptic(tag);
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic

        // Treats sparse matrix as a binary matrix if a_flag != Scalar::Flag::Generic.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
        
        using T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<T_in>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        constexpr bool prefetchQ = false;
        
        ParallelDo(
            [&]( const Int thread )
            {
                Tensor1<T,Int> z (nrhs);
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                const LInt last_l = rp[i_end];
                
                const LInt look_ahead = CacheLineWidth / std::max(static_cast<Size_T>(nrhs),NRHS);

                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    if( l_end > l_begin )
                    {
                        // Overwrite for first element in row.
                        {
                            const LInt l = l_begin;
                            const Int  j = ci[l];

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    __builtin_prefetch( &X[ldX * ci[l + look_ahead]], 0, 0 );
                                }
                            }
                            
                            combine_buffers<a_flag,Zero,NRHS,Sequential>(
                              COND(a_flag == Generic,a[l],Scalar::One<T>), &X[ldX * j],
                              Scalar::Zero<T>,                             &z[0],
                              nrhs
                            );
                        }
                        
                        // Add for first entry.
                        for( LInt l = l_begin + static_cast<LInt>(1); l < l_end; ++l )
                        {
                            const Int j = ci[l];

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    __builtin_prefetch( &X[ldX * ci[l + look_ahead]], 0, 0 );
                                }
                            }

                            // Add-in
                            combine_buffers<a_flag,One,NRHS,Sequential>(
                              COND(a_flag == Generic,a[l],Scalar::One<T>), &X[ldX * j],
                              Scalar::One<T>,                              &z[0],
                              nrhs
                            );
                        }

                        // Incorporate the local updates into Y-buffer.
                        combine_buffers<alpha_flag,beta_flag,NRHS,Sequential>(
                            alpha, &z[0],
                            beta,  &Y[ldY * i],
                            nrhs
                        );
                    }
                    else
                    {
                        // Modify the relevant portion of the Y-buffer.
                        if constexpr( beta_flag == Zero )
                        {
                            zerofy_buffer<NRHS,Sequential>( &Y[ldY * i] );
                        }
                        else if constexpr( beta_flag == Generic )
                        {
                            scale_buffer<NRHS,Sequential>( beta, &Y[ldY * i] );
                        }
                        else if constexpr( beta_flag == One )
                        {
                            // Do nothing.
                        }
                    }
                }
            },
            job_ptr.ThreadCount()
        );
        
        ptoc(tag);
    }


