public:

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void SpMV(
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha_, ptr<T_in>  X,
        const S_out beta,   mut<T_out> Y,
        const JobPointers<Int> & job_ptr
    )
    {
        StaticParameterCheck<R_out,T_in,S_out,T_out>();
        
        // This is basically a large switch to determine at runtime, which instantiation of SpMV_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.
        
        const R_out alpha = ( rp[m] > 0 ) ? alpha_ : static_cast<R_out>(0);
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if ( alpha == static_cast<R_out>(0) )
        {
            if ( beta == static_cast<S_out>(0) )
            {
                zerofy_buffer( Y, m, job_ptr.ThreadCount() );
            }
            else if ( beta == static_cast<S_out>(1) )
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
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMV_impl<Generic,One,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
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
                if( beta == static_cast<S_out>(1) )
                {
                    SpMV_impl<Generic,Generic,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
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
            if( alpha == static_cast<R_out>(1) )
            {
                if( beta == static_cast<S_out>(0) )
                {
                    SpMV_impl<One,One,Zero>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<S_out>(1) )
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
                if( beta == static_cast<S_out>(1) )
                {
                    SpMV_impl<One,Generic,One>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                }
                else if( beta == static_cast<S_out>(0) )
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
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  x,
        const S_out beta,  mut<T_out> y,
        const JobPointers<Int> & job_ptr
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
        
        // Only to be called by SpMV which guarantees that the following cases _cannot_ occur:
        
        //  - a_flag     == Scalar::Flag::Minus
        //  - a_flag     == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Zero
        //  - alpha_flag == Scalar::Flag::Minus
        //  - beta_flag  == Scalar::Flag::Minus
        
        // Treats sparse matrix as a binary matrix if a_flag == false.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.

        using T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<T_in>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        ParallelDo(
            [&]( const Int i )
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
                    }
                    else
                    {
                        sum += static_cast<T>(x[j]);
                    }
                }
                    
                combine_scalars<alpha_flag,beta_flag>( alpha, sum, beta, y[i] );
            },
            job_ptr
        );
        
        ptoc(tag);
    }

