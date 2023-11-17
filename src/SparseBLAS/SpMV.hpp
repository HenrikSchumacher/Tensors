public:

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMV(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha_, cptr<T_in>  X,
        cref<S_out> beta_,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMV_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.
        
        using alpha_T = std::conditional_t< Scalar::RealQ<R_out>, Scalar::Real<T_out>, T_out>;
        using beta_T  = std::conditional_t< Scalar::RealQ<S_out>, Scalar::Real<T_out>, T_out>;
        
        StaticParameterCheck<alpha_T,T_in,beta_T,T_out>();
        
        const alpha_T alpha = ( rp[m] > 0 ) ? scalar_cast<T_out>(alpha_) : scalar_cast<R_out>(0);
        const beta_T  beta  = scalar_cast<T_out>(beta_);
        
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == static_cast<alpha_T>(0) )
        {
            if ( beta == static_cast<beta_T>(0) )
            {
                zerofy_buffer( Y, m, job_ptr.ThreadCount() );
            }
            else if ( beta == static_cast<beta_T>(1) )
            {
                // Do nothing.
            }
            else
            {
                scale_buffer( beta, Y, m, job_ptr.ThreadCount() );
            }
            return;
        }
        
        if( a != nullptr )
        {
            if( alpha == static_cast<alpha_T>(1) )
            {
                if( beta == static_cast<beta_T>(0) )
                {
                    SpMV_impl<Generic,One,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<beta_T>(1) )
                {
                    SpMV_impl<Generic,One,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMV_impl<Generic,One,Generic>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<beta_T>(1) )
                {
                    SpMV_impl<Generic,Generic,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<beta_T>(0) )
                {
                    SpMV_impl<Generic,Generic,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMV_impl<Generic,Generic,Generic>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
        }
        else
        {
            if( alpha == static_cast<alpha_T>(1) )
            {
                if( beta == static_cast<beta_T>(0) )
                {
                    SpMV_impl<One,One,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<beta_T>(1) )
                {
                    SpMV_impl<One,One,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMV_impl<One,One,Generic>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
            else
            {
                // general alpha
                if( beta == static_cast<beta_T>(1) )
                {
                    SpMV_impl<One,Generic,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<beta_T>(0) )
                {
                    SpMV_impl<One,Generic,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else
                {
                    SpMV_impl<One,Generic,Generic>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
            }
        }
    }

private:

    template<Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMV_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  x,
        cref<S_out> beta,  mptr<T_out> y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        std::string tag = std::string(ClassName()+"::SpMV_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +TypeName<R_out>+","
            +TypeName<T_in >+","
            +TypeName<S_out>+","
            +TypeName<T_out>+">";
        
        ptic(tag);
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic
        
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
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                const LInt last_l = rp[i_end];
                
                constexpr LInt look_ahead = 1;

                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    T sum ( Scalar::Zero<T> );
                    
                    for( LInt l = l_begin; l < l_end; ++l )
                    {
                        const Int j = ci[l];
    
                        // This prefetch would cause segfaults without the check.
                        if( l + look_ahead < last_l )
                        {
                            prefetch( &x[ci[l + look_ahead]], 0, 0 );
                        }
    
                        if constexpr ( a_flag == Generic )
                        {
                            sum += scalar_cast<T>(a[l]) * scalar_cast<T>(x[j]);
                        }
                        else
                        {
                            sum += scalar_cast<T>(x[j]);
                        }
                    }
                    
                    combine_scalars<alpha_flag,beta_flag>( alpha, sum, beta, y[i] );
                }
            },
            job_ptr.ThreadCount()
        );

        ptoc(tag);
    }

