public:

    template<int cols, typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMM_fixed(
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha_, ptr<T_in>  X, const Int ldX,
        const S_out beta,   mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & job_ptr
    )
    {
        StaticParameterCheck<R_out,T_in,S_out,T_out>();
        
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_fixed_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_fixed_impl are instantiated.
        
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
                    #pragma omp parallel for num_threads(job_ptr.ThreadCount())
                    for( Int i = 0; i < m; ++ i )
                    {
                        zerofy_buffer<cols>( &Y[ldY*i] );
                    }
                }
            }
            else if( beta == static_cast<S_out>(1) )
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
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMM_fixed_impl<cols,Generic,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
                {
                    SpMM_fixed_impl<cols,Generic,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_impl<cols,Generic,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_fixed_impl<cols,Generic,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
                {
                    SpMM_fixed_impl<cols,Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_impl<cols,Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMM_fixed_impl<cols,One,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
                {
                    SpMM_fixed_impl<cols,One,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_impl<cols,One,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<S_out>(1) )
                {
                    SpMM_fixed_impl<cols,One,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
                {
                    SpMM_fixed_impl<cols,One,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMM_fixed_impl<cols,One,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
    }

private:

    template<Int cols, Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMM_fixed_impl(
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X, const Int ldX,
        const S_out beta,  mut<T_out> Y, const Int ldY,
        const JobPointers<Int> & job_ptr
    )
    {
        std::string tag = std::string(ClassName()+"::SpMM_fixed_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +TypeName<R_out>+","
            +TypeName<T_in >+","
            +TypeName<S_out>+","
            +TypeName<T_out>+">";
        
        ptic(tag);
        
        // Only to be called by SpMM_fixed which guarantees that the following cases _cannot_ occur:
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
                        
//                        prefetch_buffer<cols,0,0>( &X[ldX * ci[k+1]] );
                        
                        if constexpr ( a_flag == Generic )
                        {
                            combine_buffers<cols,Generic,One>(
                                a[k],                 &X[ldX * j],
                                Scalar::One<T>, &z[0]
                            );
                        }
                        else
                        {
                            combine_buffers<cols,One,One>(
                                Scalar::One<T>, &X[ldX * j],
                                Scalar::One<T>, &z[0]
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
                                a[k],           &X[ldX * j],
                                Scalar::One<T>, &z[0]
                            );
                        }
                        else
                        {
                            combine_buffers<cols,One,One>(
                                Scalar::One<T>, &X[ldX * j],
                                Scalar::One<T>, &z[0]
                            );
                        }
                    }
                    
                    // incorporate the local updates into Y-buffer
                    combine_buffers<cols,alpha_flag,beta_flag>(
                        alpha, &z[0],
                        beta,  &Y[ldY * i]
                    );
                }
                else
                {
                    // Row i has no nonzero entries. Just zerofy the according row of Y-buffer
                    zerofy_buffer<cols>( &Y[ldY * i] );
                }
            }
        }
        
        ptoc(tag);
    }

