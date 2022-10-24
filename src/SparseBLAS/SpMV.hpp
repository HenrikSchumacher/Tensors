public:

    void SpMV
    (
        const Int   * restrict const rp,
        const SInt  * restrict const ci,
        const T     * restrict const a,
        const Int                    m,
        const Int                    n,
        const T                      alpha,
        const T_in  * restrict const x,
        const T_out                  beta,
              T_out * restrict const y,
        const JobPointers<Int> & job_ptr
    )
    {
//        T alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<T>(0);
        

        if( rp[m] <= 0 )
        {
            if( beta == static_cast<T_out>(0) )
            {
                wprint(ClassName()+"::SpMV: No nonzeroes found and beta = 0. Overwriting by 0.");
                zerofy_buffer( y, m );
            }
            else
            {
                if( beta == static_cast<T_out>(1) )
                {s
                    wprint(ClassName()+"::SpMV: No nonzeroes found and beta = 1. Doing nothing.");
                }
                else
                {
                    wprint(ClassName()+"::SpMV: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");
                    
                    
                    scale( y, beta, m, job_ptr.Size()-1);
                }
            }
            return;
        }
        
        if( beta == static_cast<T_out>(0) )
        {
            if( alpha == static_cast<T>(0) )
            {
                zerofy_buffer( y, m );
            }
            else
            {
                // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
                #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];

                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        T sum = static_cast<T>(0);

                        const Int l_begin = rp[i  ];
                        const Int l_end   = rp[i+1];
                        
                        __builtin_prefetch( &ci[l_end] );
                        __builtin_prefetch( &a[l_end] );
                    
                        #pragma omp simd reduction( + : sum )
                        for( Int l = l_begin; l < l_end; ++l )
                        {
                            const SInt j = ci[l];
                            
                            sum += a[l] * static_cast<T>(x[j]);
    //                                sum = std::fma(a[l], x[j], sum);
                        }

                        y[i] = static_cast<T_out>(alpha * sum);
                    }
                }
            }
        }
        else
        {
            if( alpha == static_cast<T>(0) )
            {
                scale( y, beta, m, job_ptr.Size()-1);
            }
            else
            {
                #pragma omp parallel for num_threads( job_ptr.Size()-1 )
                for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];

                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        T sum = static_cast<T>(0);

                        const Int l_begin = rp[i  ];
                        const Int l_end   = rp[i+1];
                        
                        __builtin_prefetch( &ci[l_end] );
                        __builtin_prefetch( &a [l_end] );
                    
                        #pragma omp simd reduction( + : sum )
                        for( Int l = l_begin; l < l_end; ++l )
                        {
                            const SInt j = ci[l];
                            
                            sum += a[l] * static_cast<T>(x[j]);
    //                                sum = std::fma(a[l], x[j], sum);
                        }

    //                            y[i] = beta * y[i] + static_cast<T_out>(alpha * sum);
                        y[i] = std::fma(beta, y[i], alpha * sum);
                    }
                }
            }
        }
    }
