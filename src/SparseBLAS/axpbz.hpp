protected:

    template<Int cols, int alpha_flag, int beta_flag>
    void axpbz(
        const T                     alpha,
        const T_in * restrict const x,
        const T                     beta,
              T    * restrict const z
    )
    {
        if constexpr ( alpha_flag == 1 )
        {
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = static_cast<T>(x[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] += static_cast<T>(x[k]);
                }
            }
            else
            {
                #pragma omp simd
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
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = static_cast<T>(0);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
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
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]);
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] += alpha * static_cast<T>(x[k]);
                }
            }
            else
            {
                // general alpha and general beta
                #pragma omp simd
                for( Int k = 0; k < cols; ++k )
                {
                    z[k] = alpha * static_cast<T>(x[k]) + beta * z[k];
                }
            }
        }
    }
