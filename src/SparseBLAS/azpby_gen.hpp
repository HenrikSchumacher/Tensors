protected:

    template<int alpha_flag, int beta_flag>
    void azpby_gen(
        const T                      alpha,
        const T     * restrict const z,
        const T_out                  beta,
              T_out * restrict const y,
        const I                      N
    )
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(z[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] += static_cast<T_out>(z[k]);
                }
            }
            else
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(z[k]) + beta * y[k];
                }
            }
        }
        else if constexpr ( alpha_flag == 0 )
        {
            if constexpr ( beta_flag == 0 )
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(0);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] *= beta;
                }
            }
        }
        else
        {
            // alpha arbitrary;
            if constexpr ( beta_flag == 0 )
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(alpha * z[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] += static_cast<T_out>(alpha * z[k]);
                }
            }
            else
            {
                // general alpha and general beta
                #pragma omp simd
                for( I k = 0; k < N; ++k )
                {
                    y[k] = static_cast<T_out>(alpha * z[k]) + beta * y[k];
                }
            }
        }
    }
