public:

    template<bool base, typename alpha_T_, typename x_t, typename beta_T_, typename y_t>
    void SpMV(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T_> alpha_, cptr<x_t> X,
        cref< beta_T_> beta_,  mptr<y_t> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        // This is basically a large switch to determine at runtime, which instantiation of SpMV_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.
        
        using alpha_T = std::conditional_t< Scalar::RealQ<alpha_T_>, Scalar::Real<y_t>, y_t>;
        using beta_T  = std::conditional_t< Scalar::RealQ< beta_T_>, Scalar::Real<y_t>, y_t>;
        
        StaticParameterCheck<alpha_T,x_t,beta_T,y_t>();
        
        const alpha_T alpha = ( rp[m] > 0 ) ? scalar_cast<y_t>(alpha_) : scalar_cast<alpha_T_>(0);
        const beta_T  beta  = scalar_cast<y_t>(beta_);
        
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == alpha_T(0) )
        {
            if ( beta == beta_T(0) )
            {
                zerofy_buffer( Y, m, job_ptr.ThreadCount() );
            }
            else if ( beta == beta_T(1) )
            {
                // Do nothing.
            }
            else
            {
                scale_buffer( beta, Y, m, job_ptr.ThreadCount() );
            }
            return;
        }
        
        auto job = [=]<F_T a_flag, F_T alpha_flag, F_T beta_flag>()
        {
            SpMV_impl<a_flag,alpha_flag,beta_flag,base>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        };
        
        if( a != nullptr )
        {
            constexpr F_T a_flag = F_T::Generic;
            
            if( alpha == alpha_T(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                constexpr F_T alpha_flag = F_T::Generic;
                
                // general alpha
                if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
        else
        {
            constexpr F_T a_flag = F_T::Plus;
            
            if( alpha == alpha_T(1) )
            {
                constexpr F_T alpha_flag = F_T::Plus;
                
                if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                // general alpha
                constexpr F_T alpha_flag = F_T::Generic;
                
                if( beta == beta_T(1) )
                {
                    constexpr F_T beta_flag = F_T::Plus;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == beta_T(0) )
                {
                    constexpr F_T beta_flag = F_T::Zero;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = F_T::Generic;
                    job.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
    }

private:

    template<
        Scalar::Flag a_flag, Scalar::Flag alpha_flag, Scalar::Flag beta_flag, bool base = 0,
        typename alpha_T_, typename x_t, typename  beta_T_, typename y_t>
    void SpMV_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T_> alpha, cptr<x_t> x,
        cref< beta_T_> beta,  mptr<y_t> y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        (void)m;
        (void)n;
        
        std::string tag = std::string(ClassName()+"::SpMV_impl<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +ToString(base)+","
            +TypeName<alpha_T_>+","
            +TypeName<x_t >+","
            +TypeName<beta_T_>+","
            +TypeName<y_t>+">";
        
        ptic(tag);
        
        // TODO: Add better check for pointer overlap.
        
        if constexpr ( std::is_same_v<x_t,y_t> )
        {
            if( x == y )
            {
                eprint( tag + ": Input and output pointer coincide. This is not safe. Aborting.");
                
                ptoc(tag);
                
                return;
            }
        }
        
        // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == Plus
        //  - alpha_flag == Plus
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic
        
        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        using T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<x_t>,
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
                        const Int j = ci[l] - base;
    
                        // This prefetch would cause segfaults without the check.
                        if( l + look_ahead < last_l )
                        {
                            prefetch( &x[ci[l + look_ahead] - base], 0, 0 );
                        }
    
                        if constexpr ( a_flag == F_T::Generic )
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

