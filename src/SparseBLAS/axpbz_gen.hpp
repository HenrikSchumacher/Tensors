protected:

    template<int alpha_flag, int beta_flag>
    force_inline void axpbz_gen(
        const T                     alpha,
        const T_in * restrict const x,
        const T                     beta,
              T    * restrict const z,
              Int                   N
    ) const
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] = static_cast<T>(x[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] += static_cast<T>(x[k]);
                }
            }
            else
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] = static_cast<T>(x[k]) + beta * z[k];
                }
            }
        }
        else if constexpr ( alpha_flag == 0 )
        {
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] = T_zero;
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] *= beta;
                }
            }
        }
        else
        {
            // alpha arbitrary;
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                for( Int k = 0; k < N; ++k )
                {
                    z[k] += alpha * static_cast<T>(x[k]);
                }
            }
            else
            {
                // general alpha and general beta
                for( Int k = 0; k < N; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]) + beta * z[k];
                }
            }
        }
    }
