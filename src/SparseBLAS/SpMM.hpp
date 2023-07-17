public:

    template<Size_T NRHS = VarSize, typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMM (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha_, ptr<T_in>  X, const Int ldX,
        const S_out beta,   mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & restrict job_ptr,
        const Int nrhs = NRHS
    )
    {
        StaticParameterCheck<R_out,T_in,S_out,T_out>();
        
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.

        const R_out alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<R_out>(0);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == static_cast<R_out>(0) )
        {
            if( beta == static_cast<S_out>(0) )
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
            else if( beta == static_cast<S_out>(1) )
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
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMM_impl<Generic,One    ,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<S_out>(1) )
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
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_impl<Generic,Generic,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<S_out>(0) )
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
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMM_impl<One    ,One    ,Zero   ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<S_out>(1) )
                {
                    SpMM_impl<One    ,One    ,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else
                {
                    SpMM_impl<One    ,One    ,Generic,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_impl<Generic,Generic,One    ,NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
                }
                else if( beta == static_cast<S_out>(0) )
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

    template<Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, Size_T NRHS,
        typename R_out, typename T_in, typename S_out, typename T_out
    >
    void SpMM_impl(
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha,  ptr<T_in>  X, const Int ldX,
        const S_out beta,   mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & restrict job_ptr,
        const Int nrhs
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
        
        // Only to be called by SpMM which guarantees that the following cases _cannot_ occur:
        //  - a_flag     == Scalar::Flag::Minus
        //  - a_flag     == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Minus
        //  - beta_flag  == Scalar::Flag::Minus

        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
        
        using T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<T_in>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        ParallelDo(
            [&]( const Int thread )
            {
                Tensor1<T,Int> z (nrhs);
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

            //            __builtin_prefetch( &ci[l_end] );
            //
            //            if constexpr ( a_flag )
            //            {
            //                __builtin_prefetch( &a[l_end] );
            //            }

                    if( l_end > l_begin )
                    {
                        // create a local buffer for accumulating the result

                        {
                            const LInt l = l_begin;
                            const Int  j = ci[l];

                            // This pretech frequently caused segfaults!
//                            __builtin_prefetch( &X[ldX * ci[l+1]] );

                            if constexpr ( a_flag == Generic )
                            {
                                combine_buffers<Generic,Zero,NRHS,Sequential>(
                                  a[l],            &X[ldX * j],
                                  Scalar::Zero<T>, &z[0],
                                  nrhs
                                );
                            }
                            else
                            {
                                combine_buffers<One,Zero,NRHS,Sequential>(
                                  Scalar::One<T>,  &X[ldX * j],
                                  Scalar::Zero<T>, &z[0],
                                  nrhs
                                );
                            }
                        }
                        // Remark: l_end-1 is unproblematic here because we have l_end > l_begin and
                        // l_begin and l_end are of the same type LInt .
                        // So if LInt is unsigned, then l_end == 0 cannot occur.
                        for( LInt l = l_begin+1; l < l_end-1; ++l )
                        {
                            const Int j = ci[l];

                            // This pretech frequently caused segfaults!
//                            __builtin_prefetch( &X[ldX * ci[l+1]] );

                            if constexpr ( a_flag == Generic )
                            {
                                combine_buffers<Generic,One,NRHS,Sequential>(
                                  a[l],           &X[ldX * j],
                                  Scalar::One<T>, &z[0],
                                  nrhs
                                );
                            }
                            else
                            {
                                // In caase a is a nullptr...
                                combine_buffers<One,One,NRHS,Sequential>(
                                  Scalar::One<T>, &X[ldX * j],
                                  Scalar::One<T>, &z[0],
                                  nrhs
                                );
                            }
                        }

                        if( l_end > l_begin+1 )
                        {
                            const LInt l = l_end-1;

                            const Int  j = ci[l];

                            if constexpr ( a_flag == Generic)
                            {
                                combine_buffers<Generic,One,NRHS,Sequential>(
                                    a[l],           &X[ldX * j],
                                    Scalar::One<T>, &z[0],
                                    nrhs
                                );
                            }
                            else
                            {
                                // In case a is a nullptr...
                                combine_buffers<One,One,NRHS,Sequential>(
                                    Scalar::One<T>, &X[ldX * j],
                                    Scalar::One<T>, &z[0],
                                    nrhs
                                );
                            }
                        }

                        // incorporate the local updates into Y-buffer

                        combine_buffers<alpha_flag,beta_flag,NRHS,Sequential>(
                            alpha, &z[0],
                            beta,  &Y[ldY * i],
                            nrhs
                        );
                    }
                    else
                    {
                        // zerofy the relevant portion of the Y-buffer
                        zerofy_buffer<NRHS,Sequential>( &Y[ldY * i], nrhs );
                    }
                }
            },
            job_ptr.ThreadCount()
        );
        
        ptoc(tag);
    }


