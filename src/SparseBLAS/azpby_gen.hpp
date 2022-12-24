protected:

    template<int alpha_flag, int beta_flag>
    force_inline void azpby_gen(
        const T                      alpha,
        const T     * restrict const z,
        const T_out                  beta,
              T_out * restrict const y,
        const Int                    N
    ) const
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                copy_buffer(z,y,N);
            }
            else if constexpr ( beta_flag == 1 )
            {
                add_to_buffer(z,y,N);
            }
            else
            {
                for( Int k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(z[k]) + beta * y[k];
                }
            }
        }
        else if constexpr ( alpha_flag == 0 )
        {
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer(y, N);
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                scale_buffer(beta, y, N);
            }
        }
        else
        {
            // alpha arbitrary;
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(alpha * z[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    y[k] += static_cast<T_out>(alpha * z[k]);
                }
            }
            else
            {
                // general alpha and general beta
                for( Int k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(alpha * z[k]) + beta * y[k];
                }
            }
        }
    }
