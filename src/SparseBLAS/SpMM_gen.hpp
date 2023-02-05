public:

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMM_gen (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha_,  ptr<T_in>  X, const Int ldX,
        const S_out beta,    mut<T_out> Y, const Int ldY,
        const Int   cols,
        const JobPointers<Int> & job_ptr
    )
    {
        StaticParameterCheck<R_out,T_in,S_out,T_out>();
        
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_gen_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_gen_impl are instantiated.

        const R_out alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<R_out>(0);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == static_cast<R_out>(0) )
        {
            if( beta == static_cast<S_out>(0) )
            {
                if( ldY == cols )
                {
                    zerofy_buffer( Y, m * cols, job_ptr.ThreadCount() );
                }
                else
                {
                    if( job_ptr.ThreadCount() > 1 )
                    {
                        #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                        for( Int i = 0; i < m; ++ i )
                        {
                            zerofy_buffer( &Y[ldY*i], cols );
                        }
                    }
                    else
                    {
                        for( Int i = 0; i < m; ++ i )
                        {
                            zerofy_buffer( &Y[ldY*i], cols );
                        }
                    }
                }
            }
            else if( beta == static_cast<S_out>(1) )
            {
                // Do nothing.
            }
            else
            {
                if( ldY == cols )
                {
                    scale_buffer( beta, Y, m * cols, job_ptr.ThreadCount() );
                }
                else
                {
                    if( job_ptr.ThreadCount() > 1 )
                    {
                        #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                        for( Int i = 0; i < m; ++ i )
                        {
                            scale_buffer( beta, &Y[ldY*i], cols );
                        }
                    }
                    else
                    {
                        for( Int i = 0; i < m; ++ i )
                        {
                            scale_buffer( beta, &Y[ldY*i], cols );
                        }
                    }
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
                    SpMM_gen_impl<Generic,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
                {
                    SpMM_gen_impl<Generic,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_impl<Generic,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_gen_impl<Generic,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
                {
                    SpMM_gen_impl<Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_impl<Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMM_gen_impl<One,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
                {
                    SpMM_gen_impl<One,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_impl<One,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_gen_impl<Generic,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
                {
                    SpMM_gen_impl<Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_impl<Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
        }
    }

private:

    template<Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        typename R_out, typename T_in, typename S_out, typename T_out
    >
    void SpMM_gen_impl(
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha,  ptr<T_in>  X, const Int ldX,
        const S_out beta,   mut<T_out> Y, const Int ldY,
        const Int   cols,
        const JobPointers<Int> & job_ptr
    )
    {
        std::string tag = std::string(ClassName()+"::SpMM_gen_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +TypeName<R_out>+","
            +TypeName<T_in >+","
            +TypeName<S_out>+","
            +TypeName<T_out>+">";
        
        ptic(tag);
        
        // Only to be called by SpMM_gen which guarantees that the following cases _cannot_ occur:
        //  - a_flag     == Scalar::Flag::Minus
        //  - a_flag     == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Minus
        //  - beta_flag  == Scalar::Flag::Minus

        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
        
        using T = typename std::conditional_t<
            Scalar::IsComplex<Scal> || Scalar::IsComplex<T_in>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        #pragma omp parallel for num_threads( job_ptr.ThreadCount() ) schedule( static )
        for( Int thread = 0; thread < job_ptr.ThreadCount(); ++thread )
        {
            Tensor1<T,Int> z (cols);
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

                        __builtin_prefetch( &X[ldX * ci[l+1]] );

                        if constexpr ( a_flag == Generic )
                        {
                            combine_buffers<Generic,Zero>(
                              a[l],            &X[ldX * j],
                              Scalar::Zero<T>, &z[0],
                              cols
                            );
                        }
                        else
                        {
                            combine_buffers<One,Zero>(
                              Scalar::One<T>,  &X[ldX * j],
                              Scalar::Zero<T>, &z[0],
                              cols
                            );
                        }
                    }
                    // Remark: l_end-1 is unproblematic here because we have l_end > l_begin and
                    // l_begin and l_end are of the same type LInt .
                    // So if LInt is unsigned, then l_end == 0 cannot occur.
                    for( LInt l = l_begin+1; l < l_end-1; ++l )
                    {
                        const Int j = ci[l];

                        __builtin_prefetch( &X[ldX * ci[l+1]] );

                        if constexpr ( a_flag == Generic )
                        {
                            combine_buffers<Generic,One>(
                              a[l],           &X[ldX * j],
                              Scalar::One<T>, &z[0],
                              cols
                            );
                        }
                        else
                        {
                            combine_buffers<One,One>(
                              Scalar::One<T>, &X[ldX * j],
                              Scalar::One<T>, &z[0],
                              cols
                            );
                        }
                    }

                    if( l_end > l_begin+1 )
                    {
                        const LInt l = l_end-1;

                        const Int  j = ci[l];

                        if constexpr ( a_flag == Generic)
                        {
                            combine_buffers<Generic,One>(
                                a[l],           &X[ldX * j],
                                Scalar::One<T>, &z[0],
                                cols
                            );
                        }
                        else
                        {
                            combine_buffers<One,One>(
                                Scalar::One<T>, &X[ldX * j],
                                Scalar::One<T>, &z[0],
                                cols
                            );
                        }
                    }

                    // incorporate the local updates into Y-buffer

                    combine_buffers<alpha_flag,beta_flag>(
                        alpha, &z[0],
                        beta,  &Y[ldY * i],
                        cols
                    );
                }
                else
                {
                    // zerofy the relevant portion of the Y-buffer
                    zerofy_buffer( &Y[ldY * i], cols );
                }
            }
        }
        
        ptoc(tag);
    }

