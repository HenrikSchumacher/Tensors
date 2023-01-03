//#######################################################################################
//####                               Binary matrices                                #####
//#######################################################################################
        
public:

    void Multiply_BinaryMatrix_Vector
    (
        const LInt   * restrict const rp,
        const Int  * restrict const ci,
        const Int                    m,
        const Int                    n,
        const T                      alpha,
        const T_in  * restrict const x,
        const T_out                  beta,
              T_out * restrict const y
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_BinaryMatrix_Vector(rp,ci,m,n,alpha,x,beta,y,job_ptr);
    }
    
    void Multiply_BinaryMatrix_Vector
    (
        const LInt   * restrict const rp,
        const Int  * restrict const ci,
        const Int                    m,
        const Int                    n,
        const T                      alpha,
        const T_in  * restrict const x,
        const T_out                  beta,
              T_out * restrict const y,
        const JobPointers<Int> & job_ptr
    )
    {
        if( rp[m] <= 0 )
        {
            if( beta == static_cast<T_out>(0) )
            {
                wprint(ClassName()+"::Multiply_BinaryMatrix_Vector: No nonzeroes found and beta = 0. Overwriting by 0.");
                zerofy_buffer( y, m );
            }
            else
            {
                if( beta == static_cast<T_out>(1) )
                {
                    wprint(ClassName()+"::Multiply_BinaryMatrix_Vector: No nonzeroes found and beta = 1. Doing nothing.");
                }
                else
                {
                    wprint(ClassName()+"::Multiply_BinaryMatrix_Vector: No nonzeroes found. Just scaling by beta = "+ToString(beta)+".");
                    
                    scale( y, beta, m, job_ptr.Size()-1);
                }
            }
            goto exit;
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
                    
                        for( Int l = l_begin; l < l_end; ++l )
                        {
                            sum += static_cast<T>(x[ci[l]]);
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
                    
                        for( Int l = l_begin; l < l_end; ++l )
                        {
                            sum += static_cast<T>(x[ci[l]]);
                        }

                        y[i] = beta * y[i] + static_cast<T_out>(alpha * sum);
//                        y[i] = std::fma(beta, y[i], alpha * sum);
                    }
                }
            }
        }

    exit:
        return;
    }
    
    
    void Multiply_BinaryMatrix_DenseMatrix
    (
        const LInt   * restrict const rp,
        const Int  * restrict const ci,
        const Int                    m,
        const Int                    n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const Int                    cols
    )
    {
        Multiply_GeneralMatrix_DenseMatrix(rp,ci,nullptr,m,n,alpha,X,beta,Y,cols);
    }
    
    void Multiply_BinaryMatrix_DenseMatrix
    (
        const LInt   * restrict const rp,
        const Int  * restrict const ci,
        const Int                    m,
        const Int                     n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const Int                    cols,
        const JobPointers<Int> & job_ptr
    )
    {
        Multiply_GeneralMatrix_DenseMatrix(rp,ci,nullptr,m,n,alpha,X,beta,Y,cols,job_ptr);
    }
