protected:

    template<Int cols, int alpha_flag, int beta_flag>
    force_inline void azpby( const T alpha, ptr<T> z, const T_out beta, mut<T_out> y ) const
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                copy_buffer<cols>(z,y);
            }
            else if constexpr ( beta_flag == 1 )
            {
                add_to_buffer<cols>(z,y);
            }
            else
            {
                for( Int k = 0; k < cols; ++k )
                {
                    y[k] = static_cast<T_out>(z[k]) + beta * y[k];
                }
            }
        }
        else if constexpr ( alpha_flag == 0 )
        {
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer<cols>(y);
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                scale_buffer<cols>(y,beta);
            }
        }
        else
        {
            // alpha arbitrary;
            if constexpr ( beta_flag == 0 )
            {
                for( Int k = 0; k < cols; ++k )
                {
                    y[k] = alpha * static_cast<T_out>(z[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                for( Int k = 0; k < cols; ++k )
                {
                    y[k] += alpha * static_cast<T_out>(z[k]);
                }
            }
            else
            {
                // general alpha and general beta
                for( Int k = 0; k < cols; ++k )
                {
                    y[k] = alpha * static_cast<T_out>(z[k]) + beta * y[k];
                }
            }
        }
    }
