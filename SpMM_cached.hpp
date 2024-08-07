#pragma once


namespace Tensors
{
    
    template<
        bool base = 0,
        typename a_T, typename alpha_T, typename X_T, typename beta_T, typename Y_T, typename Int, typename LInt
    >
    void SpMM_cached(
        cptr<LInt> rp, cptr<Int> ci, cptr<a_T> a,
        const Size_T m, const Size_T n, const Size_T nrhs,
        cref<alpha_T> alpha, cptr<X_T> X, const Size_T ldX,
        cref<beta_T>  beta,  mptr<Y_T> Y, const Size_T ldY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        using F_T = Scalar::Flag;
        
        std::string tag = std::string("SpMM_cached")
            + "<" + ToString(base)
            + "," + TypeName<alpha_T>
            + "," + TypeName<X_T>
            + "," + TypeName<beta_T>
            + "," + TypeName<Y_T>
            +">(" + ToString(ldX)
            + "," + ToString(ldY)
            + "," + ToString(nrhs)
            + ")";
        
        ptic(tag);
        
        using T = typename std::conditional_t<
            Scalar::ComplexQ<a_T> || Scalar::ComplexQ<X_T>,
            typename Scalar::Complex<a_T>,
            typename Scalar::Real<a_T>
        >;
        
        const F_T a_flag     = (a == nullptr) ? F_T::Plus : F_T::Generic;
        const F_T alpha_flag = Scalar::ToFlag( alpha );
        const F_T beta_flag  = Scalar::ToFlag( beta );
        
        const Op opx = Op::Id;
        const Op opy = Op::Id;
        
        if ( alpha_flag == F_T::Zero )
        {
            if( beta_flag == F_T::Plus  )
            {
                // Do nothing.
            }
            if( ( ldX == nrhs ) && ( ldY == nrhs ) )
            {
                scale_buffer<VarSize,Parallel>( beta, Y, m * nrhs );
            }
            else
            {
                Do<VarSize,Parallel,Static>(
                    [=]( const Size_T i )
                    {
                        const auto ker_scale = BufferCombiner::GetKernel<Scalar::Real<Y_T>,T,beta_T,Y_T>(
                            F_T::Zero,beta_flag,nrhs,opx,opy
                        );
                        
                        ker_scale( Scalar::Zero<a_T>, nullptr, beta, &Y[ldY * i], nrhs );
                    },
                    m, job_ptr.Size()
                );
            }

            ptoc(tag);
            return;
        }

        
        const auto ker_write = BufferCombiner::GetKernel<a_T,X_T,T,T>(
            a_flag,F_T::Zero,nrhs,opx,opy
        );
        
        const auto ker_add  = BufferCombiner::GetKernel<a_T,X_T,T,T>(
            a_flag,F_T::Plus,nrhs,opx,opy
        );
        
        const auto ker_merge = BufferCombiner::GetKernel<alpha_T,T,beta_T,Y_T>(
            alpha_flag,beta_flag,nrhs,opx,opy
        );
        
        const auto ker_scale = BufferCombiner::GetKernel<Scalar::Real<Y_T>,T,beta_T,Y_T>(
            F_T::Zero,beta_flag,nrhs,opx,opy
        );
        
        constexpr bool prefetchQ = true;
        
        ParallelDo(
            [=,&job_ptr]( const Int thread )
            {
                Tensor1<T,Int> y ( nrhs );

                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                const LInt last_l = rp[i_end];

                const LInt look_ahead = static_cast<LInt>(
                    Tools::Max(
                        Size_T(1),
                        CacheLineWidth
                        /
                        (sizeof(X_T) * nrhs)
                    )
                );
                
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
                            
                            // TODO: This might not work if a == nullptr;
                            ker_write(
                                a[l],            &X[ldX * j],
                                Scalar::Zero<T>, &y[0],
                                nrhs
                            );
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

                            // TODO: This might not work if a == nullptr;
                            // Add-in
                            ker_add (
                                a[l],           &X[ldX * j],
                                Scalar::One<T>, &y[0],
                                nrhs
                            );
                        }
                        
                        // Incorporate the local updates into Y-buffer.
                        ker_merge(
                            alpha, &y[0],
                            beta,  &Y[ldY * i],
                            nrhs
                        );
                    }
                    else
                    {
                        ker_scale(
                            Scalar::Zero<Y_T>, &y[0],
                            beta,              &Y[ldY * i],
                            nrhs
                        );
                    }
                }
            },
            job_ptr.ThreadCount()
        );
        
        ptoc(tag);
        
    }
    
} // namespace Tensors
