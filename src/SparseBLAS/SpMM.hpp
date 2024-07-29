public:

    template<
        Size_T NRHS = VarSize, bool base,
        typename alpha_T_, typename X_T, typename beta_T_, typename Y_T
    >
    void SpMM(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T_> alpha_, cptr<X_T> X, const Int ldX,
        cref<beta_T_ > beta_,  mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {

        
        // This is basically a large switch to determine at runtime, which instantiation of SpMM_impl is to be invoked.
        // In particular, this implies that all relevant cases of SpMM_impl are instantiated.

        using alpha_T = std::conditional_t<Scalar::RealQ<alpha_T_>,Scalar::Real<Y_T>,Y_T>;
        using beta_T  = std::conditional_t<Scalar::RealQ< beta_T_>,Scalar::Real<Y_T>,Y_T>;
        
        StaticParameterCheck<alpha_T,X_T,beta_T,Y_T>();
        
        const alpha_T alpha = ( rp[m] > 0 ) ? scalar_cast<alpha_T>(alpha_) : scalar_cast<alpha_T>(0);
        const beta_T  beta  = scalar_cast<beta_T>(beta_);
        
        auto dot = [&]<F_T a_flag, F_T alpha_flag, F_T beta_flag>()
        {
            SpMM_impl<a_flag,alpha_flag,beta_flag,NRHS,base>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
        };
        
        // We can exit early if alpha is 0 or if there are no nozeroes in the matrix.
        if( alpha == Scalar::Zero<alpha_T> )
        {
            if( beta == Scalar::Zero<beta_T> )
            {
                if( ldY == nrhs )
                {
                    zerofy_buffer<VarSize,Parallel>( Y, m * nrhs, job_ptr.ThreadCount() );
                }
                else
                {
                    ParallelDo(
                        [&]( const Int i )
                        {
                            zerofy_buffer<NRHS,Seq>( &Y[ldY*i], nrhs );
                        },
                        m, job_ptr.ThreadCount()
                    );
                }
            }
            else if( beta == Scalar::One<beta_T> )
            {
                // Do nothing.
            }
            else
            {
                if( ldY == nrhs )
                {
                    scale_buffer<VarSize,Parallel>( beta, Y, m * nrhs, job_ptr.ThreadCount() );
                }
                else
                {
                    ParallelDo(
                        [&]( const Int i )
                        {
                            scale_buffer<NRHS,Seq>( beta, &Y[ldY*i], nrhs );
                        },
                        m, job_ptr.ThreadCount()
                    );
                }
            }
            return;
        }
        
        if( a != nullptr )
        {
            constexpr F_T a_flag = Generic;
            
            if( alpha == Scalar::One<alpha_T> )
            {
                constexpr F_T alpha_flag = One;
                
                if( beta == Scalar::Zero<beta_T> )
                {
                    constexpr F_T beta_flag = Zero;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == Scalar::One<beta_T> )
                {
                    constexpr F_T beta_flag = Zero;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = Generic;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                constexpr F_T alpha_flag = Generic;
                
                // general alpha
                if( beta == Scalar::One<beta_T> )
                {
                    constexpr F_T beta_flag = One;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == Scalar::Zero<beta_T> )
                {
                    constexpr F_T beta_flag = Zero;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = Generic;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
        else // a == nullptr
        {
            constexpr F_T a_flag = One;
            
            if( alpha == Scalar::One<alpha_T> )
            {
                constexpr F_T alpha_flag = One;
                
                if( beta == Scalar::Zero<beta_T> )
                {
                    constexpr F_T beta_flag = Zero;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == Scalar::One<beta_T> )
                {
                    constexpr F_T beta_flag = One;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = Generic;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
            else
            {
                // general alpha
                
                constexpr F_T alpha_flag = Generic;
                
                if( beta == Scalar::One<beta_T> )
                {
                    constexpr F_T beta_flag = One;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else if( beta == Scalar::Zero<beta_T> )
                {
                    constexpr F_T beta_flag = Zero;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
                else
                {
                    constexpr F_T beta_flag = Generic;
                    dot.template operator()<a_flag,alpha_flag,beta_flag>();
                }
            }
        }
    }

private:
  

    template<
        F_T a_flag, F_T alpha_flag, F_T beta_flag,
        Size_T NRHS = VarSize, bool base = 0,
        typename alpha_T, typename X_T, typename beta_T, typename Y_T
    >
    void SpMM_impl(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<X_T> X, const Int ldX,
        cref< beta_T>  beta, mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr,
        const Int nrhs = NRHS
    )
    {
        using namespace Scalar;
        
        if constexpr( (NRHS > 0) && ( a_flag == F_T::Zero || VectorizableQ<Scal> ) && VectorizableQ<Y_T> )
        {
            SpMM_vec<a_flag,alpha_flag,beta_flag,NRHS,base>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
        }
        else
        {
            std::string tag = std::string(ClassName()+"::SpMM_impl<")
                +ToString(a_flag)+","
                +ToString(alpha_flag)+","
                +ToString(beta_flag)+","
                +TypeName<alpha_T>+","
                +TypeName<X_T>+","
                +TypeName<beta_T>+","
                +TypeName<Y_T>+ ","
                + ( ( NRHS == VarSize ) ? std::string("VarSize") : ToString(NRHS) ) + ","
                + ToString(base)
                +">("+ToString(ldX)+","+ToString(ldY)+","+ToString(nrhs)+")";
            
            ptic(tag);
            
            // Only to be called by SpMM which guarantees that the following cases are the only once to occur:
            //  - a_flag     == Generic
            //  - a_flag     == One
            //  - alpha_flag == One
            //  - alpha_flag == Generic
            //  - beta_flag  == Zero
            //  - beta_flag  == Plus
            //  - beta_flag  == Generic

            // Treats sparse matrix as a binary matrix if a_flag != F_T::Generic.
            // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
            
            // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
            
            using T = typename std::conditional_t<
                Scalar::ComplexQ<Scal> || Scalar::ComplexQ<X_T>,
                typename Scalar::Complex<Scal>,
                typename Scalar::Real<Scal>
            >;
            
            if constexpr ( a_flag == Generic )
            {
                if ( a == nullptr )
                {
                    eprint( tag + ": a_flag == F_T::Generic, but the pointer a is a nullptr. Aborting.");
                    
                    ptoc(tag);
                    
                    return;
                }
            }
            
            constexpr bool prefetchQ = false;
//            constexpr bool prefetchQ = true;
            
            ParallelDo(
                [&]( const Int thread )
                {
                    std::conditional_t<
                        NRHS==VarSize,
                        Tensor1<T,Int>,
                        Tiny::Vector<NRHS,T,Int>
                    > y;
                    
                    if constexpr ( NRHS==VarSize )
                    {
                        y = Tensor1<T,Int>( nrhs );
                    }
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    const LInt last_l = rp[i_end];
                    
                    const LInt look_ahead = int_cast<LInt>(Tools::Max( Size_T(1), (CacheLineWidth / sizeof(Scal)) / Tools::Max(static_cast<Size_T>(nrhs),NRHS) ));
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt l_begin = rp[i  ];
                        const LInt l_end   = rp[i+1];

                        if( l_end > l_begin )
                        {
                            // Overwrite for first element in row.
                            {
                                const LInt l = l_begin;
                                const Int  j = ci[l] - base;

                                if constexpr ( prefetchQ )
                                {
                                    // This prefetch would cause segfaults without the check.
                                    if( l + look_ahead < last_l )
                                    {
                                        prefetch( &X[ldX * (ci[l + look_ahead] - base)], 0, 0 );
                                    }
                                }
                                
                                // We use this `if constexpr` here so that we do not read from a when it is a nullptr
                                if constexpr ( a_flag == Generic )
                                {
                                    combine_buffers<a_flag,Zero,NRHS,Seq>(
                                        a[l],            &X[ldX * j],
                                        Scalar::Zero<T>, &y[0],
                                        nrhs
                                    );
                                }
                                else if constexpr ( a_flag == One )
                                {
                                    combine_buffers<a_flag,Zero,NRHS,Seq>(
                                        Scalar::One<T>,  &X[ldX * j],
                                        Scalar::Zero<T>, &y[0],
                                        nrhs
                                    );
                                }
                            }
                            
                            // Add remaining entries.
                            for( LInt l = l_begin + static_cast<LInt>(1); l < l_end; ++l )
                            {
                                const Int j = ci[l] - base;

                                if constexpr ( prefetchQ )
                                {
                                    // This prefetch would cause segfaults without the check.
                                    if( l + look_ahead < last_l )
                                    {
                                        prefetch( &X[ldX * (ci[l + look_ahead]- base)], 0, 0 );
                                    }
                                }

                                // Add-in
                                if constexpr ( a_flag == Generic )
                                {
                                    combine_buffers<a_flag,One,NRHS,Seq>(
                                        a[l],           &X[ldX * j],
                                        Scalar::One<T>, &y[0],
                                        nrhs
                                    );
                                }
                                else if constexpr ( a_flag == One )
                                {
                                    // We use if constexpr here so that we do not read from a when it is a nullptr
                                    combine_buffers<a_flag,One,NRHS,Seq>(
                                        Scalar::One<T>, &X[ldX * j],
                                        Scalar::One<T>, &y[0],
                                        nrhs
                                    );
                                }
                            }
                            
                            // Incorporate the local updates into Y-buffer.
                            combine_buffers<alpha_flag,beta_flag,NRHS,Seq>(
                                alpha, &y[0],
                                beta,  &Y[ldY * i],
                                nrhs
                            );
                        }
                        else
                        {
                            // Modify the relevant portion of the Y-buffer.
                            if constexpr( beta_flag == Zero )
                            {
                                zerofy_buffer<NRHS,Seq>( &Y[ldY * i] );
                            }
                            else if constexpr( beta_flag == Generic )
                            {
                                scale_buffer<NRHS,Seq>( beta, &Y[ldY * i] );
                            }
                            else if constexpr( beta_flag == One )
                            {
                                // Do nothing.
                            }
                        }
                    }
                },
                job_ptr.ThreadCount()
            );
            
            ptoc(tag);
            
        }
    }

#if defined(__clang__)
    #if( __has_attribute(ext_vector_type) )

    template<
        F_T a_flag, F_T alpha_flag, F_T beta_flag,
        Size_T NRHS = VarSize, bool base,
        typename alpha_T, typename X_T, typename beta_T, typename Y_T
    >
    void SpMM_vec(
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<alpha_T> alpha, cptr<X_T> X, const Int ldX,
        cref<beta_T>  beta,  mptr<Y_T> Y, const Int ldY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        (void)m;
        (void)n;
        
        std::string tag = std::string(ClassName()+"::SpMM_vec<")
            +ToString(a_flag)+","
            +ToString(alpha_flag)+","
            +ToString(beta_flag)+","
            +ToString(NRHS)+","
            +ToString(base)+","
            +TypeName<alpha_T>+","
            +TypeName<X_T>+","
            +TypeName<beta_T>+","
            +TypeName<Y_T>
            +">(" + ToString(ldX) + "," + ToString(ldY) + ")";

        ptic(tag);
        
        
        static_assert(NRHS!=0, "SpMM_vec only implements static size behavior.");
        
        // Only to be called by SpMM_impl which guarantees that the following cases are the only once to occur:
        //  - a_flag     == Generic
        //  - a_flag     == One
        //  - alpha_flag == One
        //  - alpha_flag == Generic
        //  - beta_flag  == Zero
        //  - beta_flag  == Plus
        //  - beta_flag  == Generic

        // Treats sparse matrix as a binary matrix if a_flag != F_T::Generic.
        // (Then it implicitly assumes that a == nullptr and does not attempt to index into it.)
        
        // Uses shortcuts if alpha = 1, beta = 0 or beta = 1.
        
        // TODO: Use real types for alpha, beta, x, a, y if applicable.
        // TODO: Maybe use simple combine_buffer to merge y and z.
        
        using y_T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<X_T>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        using x_T = y_T;
        
        using V_T = vec_T<NRHS,y_T>;
        
        if constexpr ( a_flag == Generic )
        {
            if ( a == nullptr )
            {
                eprint( tag + ": a_flag == F_T::Generic, but the pointer a is a nullptr. Aborting.");
                
                ptoc(tag);
                
                return;
            }
        }
        
        const auto alpha_ = scalar_cast<Y_T>(alpha);
        const auto beta_  = scalar_cast<Y_T>(beta );
        
        constexpr bool prefetchQ = true;
        
        print("SpMM_vec");
             
        ParallelDo(
            [&]( const Int thread )
            {
//                V_T x;
     
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];

                const LInt last_l = rp[i_end];

                const LInt look_ahead = CacheLineWidth / NRHS;
                for( Int i = i_begin; i < i_end; ++i )
                {
                    V_T y ( Scalar::Zero<y_T> );
                    
                    const LInt l_begin = rp[i  ];
                    const LInt l_end   = rp[i+1];

                    if( l_end > l_begin )
                    {
                        // Add for first entry.
                        for( LInt l = l_begin; l < l_end; ++l )
                        {
                            const Int j = ci[l] - base;

                            if constexpr ( prefetchQ )
                            {
                                // This prefetch would cause segfaults without the check.
                                if( l + look_ahead < last_l )
                                {
                                    prefetch( &X[ldX * (ci[l + look_ahead] - base)], 0, 0 );
                                }
                            }

//                            copy_buffer<NRHS>( &X[ldX * j], reinterpret_cast<y_T *>(&x) );
//
//                            if constexpr ( a_flag == Generic )
//                            {
//                                y += a[l] * x;
//                            }
//                            else if constexpr ( a_flag == One )
//                            {
//                                y += x;
//                            }
                            
                            const V_T & x = (*reinterpret_cast<const V_T *>(&X[ldX * j]));
                            
                            if constexpr ( a_flag == Generic )
                            {
                                y += a[l] * x;
                            }
                            else if constexpr ( a_flag == One )
                            {
                                y += x;
                            }
                            
                        }

                        
                        // Incorporate the local updates into Y-buffer.
                                   
//                        combine_buffers_vec<alpha_flag,beta_flag,NRHS>(
//                            alpha_, reinterpret_cast<mptr<y_T>>(&y), 
//                            beta_ , &Y[ldY * i]
//                        );
                        
                        *reinterpret_cast<V_T *>(&Y[ldY * i])
                            =
                            alpha_ * y
                            +
                            beta_  * *reinterpret_cast<V_T *>(&Y[ldY * i]);
                        
                        
    //                    if constexpr ( beta_flag != Zero )
    //                    {
    //                        copy_buffer<NRHS>( &Y[ldY * i], reinterpret_cast<z_T *>(&z) );
    //                    }
    //
    //
    //                    if constexpr ( alpha_flag == One )
    //                    {
    //                        if constexpr ( beta_flag == Zero )
    //                        {
    //                            z = y;
    //                        }
    //                        else if constexpr ( beta_flag == One )
    //                        {
    //                            z += y;
    //                        }
    //                        else if constexpr ( beta_flag == Generic )
    //                        {
    //                            z = y + beta * z;
    //                        }
    //                    }
    //                    else if constexpr( alpha_flag == Generic )
    //                    {
    //                        if constexpr ( beta_flag == Zero )
    //                        {
    //                            z = alpha * y;
    //                        }
    //                        else if constexpr ( beta_flag == One )
    //                        {
    //                            z += alpha * y;
    //                        }
    //                        else if constexpr ( beta_flag == Generic )
    //                        {
    //                            z = alpha * y + beta * z;
    //                        }
    //                    }
    //
    //                    copy_buffer<NRHS>( reinterpret_cast<z_T *>(&z), &Y[ldY * i] );
    //
                    }
                    else
                    {
                        // Modify the relevant portion of the Y-buffer.
                        if constexpr( beta_flag == Zero )
                        {
//                            zerofy_buffer<NRHS,Seq>( &Y[ldY * i] );
                            *reinterpret_cast<V_T *>(&Y[ldY * i]) = 0;
                        }
                        else if constexpr( beta_flag == One )
                        {
                            // Do nothing.
                        }
                        else if constexpr( beta_flag == Generic )
                        {
//                            scale_buffer<NRHS,Seq>( beta, &Y[ldY * i] );
                            *reinterpret_cast<V_T *>(&Y[ldY * i]) *= beta;
                        }
                    }
                }
            },
            job_ptr.ThreadCount()
        );
        
        ptoc(tag);
    }
    #endif
#endif
