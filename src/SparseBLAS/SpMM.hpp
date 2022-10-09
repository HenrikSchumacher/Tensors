protected:

    template<int cols>
    void SpMM
    (
        const Int   * restrict const rp,
        const Int   * restrict const ci,
        const T     * restrict const a,
        const Int                    m,
        const Int                    n,
        const T                      alpha_,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const JobPointers<Int> & job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_implementation is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_implementation are instantiated.
        
//        print("SpMM<"+ToString(cols)+">, alpha = "+ToString(alpha)+", "+"beta = "+ToString(beta));
        
        const T alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<T>(0);
        
        if( a != nullptr )
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,true,1,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,true,1,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,true,1,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else if( alpha == static_cast<T>(0) )
            {
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,true,0,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,true,0,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,true,0,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,true,-1,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,true,-1,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,true,-1,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<T>(1) )
            {
                if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,false,1,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,false,1,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,false,1,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else if( alpha == static_cast<T>(0) )
            {
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,false,0,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,false,0,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,false,0,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<T_out>(1) )
                {
                    SpMM_implementation<cols,false,-1,1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<T_out>(0) )
                {
                    SpMM_implementation<cols,false,-1,0>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMM_implementation<cols,false,-1,-1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
        }
    }

//    template<Int cols, bool a_flag, int alpha_flag, int beta_flag >
//    void SpMM_implementation(
//        const Int   * restrict const rp,
//        const Int   * restrict const ci,
//        const T     * restrict const a,
//        const Int                    m,
//        const Int                    n,
//        const T                      alpha,
//        const T_in  * restrict const X,
//        const T_out                  beta,
//              T_out * restrict const Y,
//        const JobPointers<Int> & job_ptr
//    )
//    {
//        // Threats sparse matrix as a binary matrix if a_flag == false.
//        // (Implicitly assumes that a == nullptr.)
//        // Uses shortcuts if alpha = 0, alpha = 1, beta = 0 or beta = 1.
//        // Uses if constexpr to reuse code without runtime overhead.
////        prin
//        if constexpr ( alpha_flag == 0 )
//        {
//            if constexpr ( beta_flag == 0 )
//            {
//                zerofy_buffer(Y, m * cols);
//                return;
//            }
//            else if constexpr ( beta_flag == 1 )
//            {
//                return;
//            }
//            else
//            {
//                scale( Y, beta, m, job_ptr.Size()-1);
//                return;
//            }
//        }
//
//        #pragma omp parallel for num_threads( job_ptr.Size()-1 )
//        for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
//        {
//            const Int i_begin = job_ptr[thread  ];
//            const Int i_end   = job_ptr[thread+1];
//
//            for( Int i = i_begin; i < i_end; ++i )
//            {
//                const Int l_begin = rp[i  ];
//                const Int l_end   = rp[i+1];
//
//                __builtin_prefetch( &ci[l_end] );
//
//                if constexpr ( a_flag )
//                {
//                    __builtin_prefetch( &a[l_end] );
//                }
//
//                if( l_end > l_begin)
//                {
//                    // create a local buffer for accumulating the result
//                    T z [cols] = {};
//
//                    for( Int l = l_begin; l < l_end-1; ++l )
//                    {
//                        const Int j = ci[l];
//
//                        __builtin_prefetch( &X[cols * ci[l+1]] );
//
//                        if constexpr ( a_flag )
//                        {
//                            axpbz<cols,-1,1>( a[l], &X[cols * j], T_one, &z[0] );
//                        }
//                        else
//                        {
//                            axpbz<cols,1,1>( T_one, &X[cols * j], T_one, &z[0] );
//                        }
//                    }
//
//                    if( l_end > l_begin+1 )
//                    {
//                        const Int l = l_end-1;
//
//                        const Int j   = ci[l];
//
//                        if constexpr ( a_flag )
//                        {
//                            axpbz<cols,-1,1>( a[l], &X[cols * j], T_one, &z[0] );
//                        }
//                        else
//                        {
//                            axpbz<cols,1,1>( T_one, &X[cols * j], T_one, &z[0] );
//                        }
//                    }
//
//                    // incorporate the local updates into Y-buffer
//                    azpby<cols,alpha_flag,beta_flag>( alpha, &z[0], beta, &Y[cols * i] );
//                }
//                else
//                {
//                    // zerofy the relevant portion of the Y-buffer
//                    azpby<cols,0,0>( alpha, nullptr, beta, &Y[cols * i] );
//                }
//            }
//        }
//    }
//


template<Int cols, bool a_flag, int alpha_flag, int beta_flag >
void SpMM_implementation(
    const Int   * restrict const rp,
    const Int   * restrict const ci,
    const T     * restrict const a,
    const Int                    m,
    const Int                    n,
    const T                      alpha,
    const T_in  * restrict const X,
    const T_out                  beta,
          T_out * restrict const Y,
    const JobPointers<Int> & job_ptr
)
{
    // Threats sparse matrix as a binary matrix if a_flag == false.
    // (Implicitly assumes that a == nullptr.)
    // Uses shortcuts if alpha = 0, alpha = 1, beta = 0 or beta = 1.
    // Uses if constexpr to reuse code without runtime overhead.

//    if constexpr ( beta_flag == 0 )
//    {
//        zerofy_buffer(Y, m * cols);
//    }
//    
    if constexpr ( alpha_flag == 0 )
    {
        if constexpr ( beta_flag == 0 )
        {
            zerofy_buffer(Y, m * cols);
            return;
        }
        else if constexpr ( beta_flag == 1 )
        {
            return;
        }
        else
        {
            scale( Y, beta, m, job_ptr.Size()-1);
            return;
        }
    }

    #pragma omp parallel for num_threads( job_ptr.Size()-1 )
    for( Int thread = 0; thread < job_ptr.Size()-1; ++thread )
    {
        const Int i_begin = job_ptr[thread  ];
        const Int i_end   = job_ptr[thread+1];
        
        for( Int i = i_begin; i < i_end; ++i )
        {
            const Int k_begin = rp[i  ];
            const Int k_end   = rp[i+1];
            
//            __builtin_prefetch( &ci[l_end] );
//
//            if constexpr ( a_flag )
//            {
//                __builtin_prefetch( &a[l_end] );
//            }
            
            if( k_end > k_begin)
            {
                // create a local buffer for accumulating the result
                T z [cols] = {};
                
                for( Int k = k_begin; k < k_end-1; ++k )
                {
                    const Int j = ci[k];
                    
                    prefetch_range<cols,0,0>( &X[cols * ci[k+1]] );
                    
                    if constexpr ( a_flag )
                    {
                        axpbz<cols,-1,1>( a[k], &X[cols * j], T_one, &z[0] );
                    }
                    else
                    {
                        axpbz<cols,1,1>( T_one, &X[cols * j], T_one, &z[0] );
                    }
                }
                
                // perform last calculation in row without prefetch
                {
                    const Int k = k_end-1;
                    
                    const Int j   = ci[k];
                    
                    if constexpr ( a_flag )
                    {
                        axpbz<cols,-1,1>( a[k], &X[cols * j], T_one, &z[0] );
                    }
                    else
                    {
                        axpbz<cols,1,1>( T_one, &X[cols * j], T_one, &z[0] );
                    }
                }
                
                // incorporate the local updates into Y-buffer
                azpby<cols,alpha_flag,beta_flag>( alpha, &z[0], beta, &Y[cols * i] );
            }
            else
            {
                // zerofy the relevant portion of the Y-buffer
                azpby<cols,0,0>( alpha, nullptr, beta, &Y[cols * i] );
            }
        }
    }
}

