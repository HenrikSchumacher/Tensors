protected:

    // Experimental; not really efficient.
    template<int cols>
    void transposed_gemm
    (
        const T alpha,
        Int  const * restrict const rp,
        Int  const * restrict const ci,
        T    const * restrict const a,
        Int  const m,
        Int  const n,
        T_in const * restrict const X,
        const T_out beta,
        T_out      * restrict const Y,
        ThreadTensor3<T_out,Int> & Y_buffer,
        const JobPointers<Int> & job_ptr
    )
    {
    //            ptic(ClassName()+"::gemm<"+ToString(cols)+">");

        Y_buffer.SetZero();

        if( beta == static_cast<T>(0) )
        {
            zerofy_buffer( Y, m * cols );
        }
        else
        {
            Scale( Y, beta, m * cols, job_ptr.Size()-1);
        }

        if( alpha != static_cast<T>(0) )
        {
    //                    logprint("alpha != 0");
            // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
            
            #pragma omp parallel for num_threads( job_ptr.Size()-1 )
            for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
            {
                T_out * restrict const Y_buf = Y_buffer[thread].data();
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const Int l_begin = rp[i  ];
                    const Int l_end   = rp[i+1];
                    
                    // Add the others.
                    for( Int l = l_begin; l < l_end; ++l )
                    {
                        const Int j = ci[l];
                        
                        const T a_i_j = a[l];

                        const T_in  * restrict const X_i = X + cols * i;
                        
                              T_out * restrict const Y_j = Y_buf + cols * j;
                        
                        #pragma omp simd
                        for( Int k = 0; k < cols; ++k )
                        {
                            Y_j[k] += a_i_j * static_cast<T>(X_i[k]);
    //                                Y_j[k] = std::fma(a_i_j ,static_cast<T>(X_i[k]), Y_j[k]);
                        }
                    }
                }
            }
        }
    }


protected:

    template<int cols>
    void symm
    (
        const T alpha,
        Int const * restrict const rp,
        Int const * restrict const ci,
        T const * restrict const a,
        Int const m,
        Int const n,
        T_in  const * restrict const X,
        const T_out beta,
        T_out       * restrict const Y,
              Tensor3<T_out,Int> & Y_buffer,
        const JobPointers<Int> & job_ptr
    )
    {
    //            ptic(ClassName()+"::gemm<"+ToString(cols)+">");

        Y_buffer.SetZero();

        if( alpha != static_cast<T>(0) )
        {
    //                    logprint("alpha != 0");
            // The target buffer Y may contain nan, so we have to _overwrite_ instead of multiply by 0 and add to it!
            
            #pragma omp parallel for num_threads( job_ptr.Size()-1 )
            for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
            {
                T_out * restrict const Y_buf = Y_buffer.data(thread);
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const Int l_begin = rp[i  ];
                    const Int l_end   = rp[i+1];
                    
                    // Add the others.
                    for( Int l = l_begin; l < l_end; ++l )
                    {
                        const Int j = ci[l];
                        
                        const T a_i_j = a[l];

                        const T_in  * restrict const X_i = X + cols * i;
                        const T_in  * restrict const X_j = X + cols * j;
                        
                              T_out * restrict const Y_i = Y_buf + cols * i;
                              T_out * restrict const Y_j = Y_buf + cols * j;
                        
                        #pragma omp simd
                        for( Int k = 0; k < cols; ++k )
                        {
                            Y_i[k] += a_i_j * static_cast<T>(X_j[k]);
                            Y_j[k] += a_i_j * static_cast<T>(X_i[k]);
    //                                Y_i[k] = std::fma(a_i_j, static_cast<T>(X_j[k]), Y_i[k]);
    //                                Y_j[k] = std::fma(a_i_j, static_cast<T>(X_i[k]), Y_i[k]);
                        }
                    }
                }
            }
        }
        
        #pragma omp parallel for num_threads( job_ptr.Size()-1 ) schedule( static )
        for( Int i = 0; i < m; ++i )
        {
            const Int pos = cols*i;
    //                T_out * restrict const T_i = Y + cols * i;

            for( Int k = 0; k < cols; ++k )
            {
                Y[pos+k] *= beta;
            }

            for( Int thread = 0; thread < thread_count; ++thread )
            {
                const T_out * restrict const Y_buf = Y_buffer.data(thread,i);
                
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    Y[pos+k] += Y_buf[k];
                }
            }
        }
    }
