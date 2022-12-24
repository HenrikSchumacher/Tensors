protected:

    template<Int cols, int alpha_flag, int beta_flag>
    force_inline void axpbz(
        const T                     alpha,
        const T_in * restrict const x,
        const T                     beta,
              T    * restrict const z
    ) const
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                copy_buffer<cols>(x,z);
            }
            else if constexpr ( beta_flag == 1 )
            {
                add_to_buffer<cols>(x,z);
            }
            else
            {
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = static_cast<T>(x[k]) + beta * z[k];
                }
            }
        }
        else if constexpr ( alpha_flag == 0 )
        {
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer<cols>(z);
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                scale_buffer<cols>(beta,z);
            }
        }
        else
        {
            // alpha arbitrary;
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] += alpha * static_cast<T>(x[k]);
                }
            }
            else
            {
                // general alpha and general beta
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]) + beta * z[k];
                }
            }
        }
    }
