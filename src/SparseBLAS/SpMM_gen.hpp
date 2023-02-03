public:

    void SpMM_gen (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha_,  ptr<T>     X, const Int ldX,
        const T_out beta,    mut<T_out> Y, const Int ldY,
        const Int   cols,
        const JobPointers<Int> & job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_implementation is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_implementation are instantiated.
        
        const T alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<T>(0);
        
        if( a != nullptr )
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Plus,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Plus,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<true,ScalarFlag::Plus,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else if( alpha == static_cast<T>(0) )
            {
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Zero,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Zero,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<true,ScalarFlag::Zero,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Generic,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<true,ScalarFlag::Generic,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<true,ScalarFlag::Generic,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Plus,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Plus,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<false,ScalarFlag::Plus,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else if( alpha == static_cast<T>(0) )
            {
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Zero,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Zero,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<false,ScalarFlag::Zero,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Generic,ScalarFlag::Plus>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_gen_implementation<false,ScalarFlag::Generic,ScalarFlag::Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
                else
                {
                    SpMM_gen_implementation<false,ScalarFlag::Generic,ScalarFlag::Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
                }
            }
        }
    }

    template<bool a_flag, ScalarFlag alpha_flag, ScalarFlag beta_flag >
    void SpMM_gen_implementation(
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha,  ptr<T_in>  X, const Int ldX,
        const T_out beta,   mut<T_out> Y, const Int ldY,
        const Int   cols,
        const JobPointers<Int> & job_ptr
    )
    {
        // Threats sparse matrix as a binary matrix if a_flag == false.
        // (Implicitly assumes that a == nullptr.)
        // Uses shortcuts if alpha = 0, alpha = 1, beta = 0 or beta = 1.
        // Uses if constexpr to reuse code without runtime overhead.
    //    print("SpMM_gen_implementation<"+ToString(a_flag)+","+","+ToString(alpha_flag)+","+ToString(beta_flag)+">("+ToString(cols)+")");
        if constexpr ( alpha_flag == ScalarFlag::Zero )
        {
            if constexpr ( beta_flag == ScalarFlag::Zero )
            {
                if( ldY == cols )
                {
                    zerofy_buffer( Y, m * cols, job_ptr.ThreadCount() );
                }
                else
                {
                    #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                    for( Int i = 0; i < m; ++ i )
                    {
                        zerofy_buffer( &Y[ldY*i], cols);
                    }
                }
                return;
            }
            else if constexpr ( beta_flag == ScalarFlag::Plus )
            {
                return;
            }
            else if constexpr ( beta_flag == ScalarFlag::Generic )
            {
                if( ldY == cols )
                {
                    scale_buffer( -T_one, Y, m * cols, job_ptr.ThreadCount() );
                }
                else
                {
                    #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                    for( Int i = 0; i < m; ++ i )
                    {
                        scale_buffer( -T_one, &Y[ldY*i], cols);
                    }
                }
                return;
            }
            else
            {
                if( ldY == cols )
                {
                    scale_buffer( beta, Y, m * cols, job_ptr.ThreadCount() );
                }
                else
                {
                    #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                    for( Int i = 0; i < m; ++ i )
                    {
                        scale_buffer( beta, &Y[ldY*i], cols);
                    }
                }
                return;
            }
        }

        #pragma omp parallel for num_threads( job_ptr.ThreadCount() )
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
                        
                        if constexpr ( a_flag )
                        {
                            combine_buffers<ScalarFlag::Generic,ScalarFlag::Zero>(
                                a[l], &X[ldX * j], T_zero, &z[0], cols
                            );
                        }
                        else
                        {
                            combine_buffers<ScalarFlag::Plus,ScalarFlag::Zero>(
                                T_one, &X[ldX * j], T_zero, &z[0], cols
                            );
                        }
                    }
                    for( LInt l = l_begin+1; l < l_end-1; ++l )
                    {
                        const Int j = ci[l];
                        
                        __builtin_prefetch( &X[ldX * ci[l+1]] );
                        
                        if constexpr ( a_flag )
                        {
                            combine_buffers<ScalarFlag::Generic,ScalarFlag::Plus>(
                                a[l], &X[ldX * j], T_one, &z[0], cols
                            );
                        }
                        else
                        {
                            combine_buffers<ScalarFlag::Plus,ScalarFlag::Plus>(
                                T_one, &X[ldX * j], T_one, &z[0], cols
                            );
                        }
                    }
                    
                    if( l_end > l_begin+1 )
                    {
                        const LInt l = l_end-1;
                        
                        const Int j  = ci[l];

                        if constexpr ( a_flag )
                        {
                            combine_buffers<ScalarFlag::Generic,ScalarFlag::Plus>(
                                a[l], &X[ldX * j], T_one, &z[0], cols
                            );
                        }
                        else
                        {
                            combine_buffers<ScalarFlag::Plus,ScalarFlag::Plus>(
                                T_one, &X[ldX * j], T_one, &z[0], cols
                            );
                        }
                    }
                    
                    // incorporate the local updates into Y-buffer
                    
                    combine_buffers<alpha_flag,beta_flag>(
                        alpha, &z[0], beta, &Y[ldY * i], cols
                    );
                }
                else
                {
                    // zerofy the relevant portion of the Y-buffer
                    zerofy_buffer( &Y[ldY * i], cols );
                }
            }
        }
    }
