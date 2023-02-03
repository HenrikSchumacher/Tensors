public:

    template<int cols>
    void SpMM_fixed(
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha_, ptr<T_in>  X, const Int ldX,
        const T_out beta,   mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_fixed_implementation is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_fixed_implementation are instantiated.
        
        const T alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<T>(0);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == static_cast<T>(0) )
        {
            if( beta == static_cast<T_out>(0) )
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
                        zerofy_buffer<cols>( &Y[ldY*i] );
                    }
                }
            }
            else if( beta == static_cast<T_out>(1) )
            {
                // Do nothing;
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
                            scale_buffer<cols>( beta, &Y[ldY*i] );
                        }
                    }
                    else
                    {
                        for( Int i = 0; i < m; ++ i )
                        {
                            scale_buffer<cols>( beta, &Y[ldY*i] );
                        }
                    }
                }
            }
            return;
        }
        
        if( a != nullptr )
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_fixed_implementation<cols,Generic,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_fixed_implementation<cols,Generic,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_implementation<cols,Generic,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_fixed_implementation<cols,Generic,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_fixed_implementation<cols,Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_implementation<cols,Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_fixed_implementation<cols,One,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_fixed_implementation<cols,One,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_implementation<cols,One,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_fixed_implementation<cols,One,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_fixed_implementation<cols,One,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_implementation<cols,One,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
    }

private:

    template<Int cols, ScalarFlag a_flag, ScalarFlag alpha_flag, ScalarFlag beta_flag >
    void SpMM_fixed_implementation(
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X, const Int ldX,
        const T_out beta,  mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & job_ptr
    )
    {
        // Only to be called by SpMM_fixed which guarantees that the following cases _cannot_ occur:
        //  - a_flag     == ScalarFlag::Minus
        //  - a_flag     == ScalarFlag::Zero
        //  - alpha_flag == ScalarFlag::Zero
        //  - alpha_flag == ScalarFlag::Minus
        //  - beta_flag  == ScalarFlag::Minus
        
        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        #pragma omp parallel for num_threads( job_ptr.ThreadCount() )
        for( Int thread = 0; thread < job_ptr.ThreadCount(); ++thread )
        {
            const Int i_begin = job_ptr[thread  ];
            const Int i_end   = job_ptr[thread+1];
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                const LInt k_begin = rp[i  ];
                const LInt k_end   = rp[i+1];
                
                if( k_end > k_begin)
                {
                    // Row i has at least one nonzero entry.
                    
                    // create a local buffer for accumulating the result
                    T z [cols] = {};
                    
                    for( LInt k = k_begin; k < k_end-1; ++k )
                    {
                        const Int j = ci[k];
                        
//                        prefetch_range<cols,0,0>( &X[ldX * ci[k+1]] );
                        
                        if constexpr ( a_flag == Generic )
                        {
                            combine_buffers<cols,Generic,One>(
                                a[k], &X[ldX * j], T_one, &z[0]
                            );
                        }
                        else
                        {
                            combine_buffers<cols,One,One>(
                                T_one, &X[ldX * j], T_one, &z[0]
                            );
                        }
                    }
                    
                    // perform last calculation in row without prefetch
                    {
                        const LInt  k = k_end-1;
                        
                        const Int j = ci[k];
                        
                        if constexpr ( a_flag == Generic )
                        {
                            combine_buffers<cols,Generic,One>(
                                a[k], &X[ldX * j], T_one, &z[0]
                            );
                        }
                        else
                        {
                            combine_buffers<cols,One,One>(
                                T_one, &X[ldX * j], T_one, &z[0]
                            );
                        }
                    }
                    
                    // incorporate the local updates into Y-buffer
                    combine_buffers<cols,alpha_flag,beta_flag>(
                        alpha, &z[0], beta, &Y[ldY * i]
                    );
                }
                else
                {
                    // Row i has no nonzero entries. Just zerofy the according row of Y-buffer
                    zerofy_buffer<cols>( &Y[ldY * i] );
                }
            }
        }
    }

