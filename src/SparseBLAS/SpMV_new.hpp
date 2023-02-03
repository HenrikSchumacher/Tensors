public:

    void SpMM_fixed(
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha_, ptr<T_in>  X,
        const T_out beta,   mut<T_out> Y,
        const JobPointers<Int> & job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_implementation is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_implementation are instantiated.
        
        const T alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<T>(0);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == static_cast<T>(0) )
        {
            if ( beta == static_cast<T_out>(0) )
            {
                zerofy_buffer( Y, m * cols, job_ptr.ThreadCount() );
            }
            else if ( beta == static_cast<T_out>(1) )
            {
                // Do nothing.
            }
            else
            {
                scale_buffer( beta, Y, m * cols, job_ptr.ThreadCount() );
            }
            return;
        }
        
        if( a != nullptr )
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMV_implementation<Generic,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMV_implementation<Generic,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMV_implementation<Generic,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMV_implementation<Generic,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMV_implementation<Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMV_implementation<Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMV_implementation<One,One,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMV_implementation<One,One,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMV_implementation<One,One,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMV_implementation<One,Generic,One>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMV_implementation<One,Generic,Zero>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
                else
                {
                    SpMV_implementation<One,Generic,Generic>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                }
            }
        }
    }

private:

    template<ScalarFlag a_flag, ScalarFlag alpha_flag, ScalarFlag beta_flag >
    void SpMV_implementation(
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X,
        const T_out beta,  mut<T_out> Y,
        const JobPointers<Int> & job_ptr
    )
    {
        // Only to be called by SpMV which guarantees that the following cases _cannot_ occur:
        
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
                T sum = static_cast<T>(0);

                const LInt l_begin = rp[i  ];
                const LInt l_end   = rp[i+1];
                
                __builtin_prefetch( &ci[l_end] );
                
                if constexpr( a_flag == Generic )
                {
                    __builtin_prefetch( &a[l_end] );
                }
            
                for( LInt l = l_begin; l < l_end; ++l )
                {
                    const Int j = ci[l];
                    
                    if constexpr( a_flag == Generic )
                    {
                        sum += a[l] * static_cast<T>(x[j]);
                    else
                    {
                        sum += static_cast<T>(x[j]);
                    }
                }
                    
                    if constexpr( alpha_flag == One )
                    {
                        y[i] = std::fma(beta, y[i], alpha * sum);
                    }

                    y[i] = alpha * sum + beta * y[i];
//                y[i] = static_cast<T_out>(alpha * sum);
                y[i] = std::fma(beta, y[i], alpha * sum);
            }
        }
    }

