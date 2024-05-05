#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T K, Size_T N, AddTo_T addto, 
            typename A_T, typename B_T, typename C_T
        >
        void fixed_dot_mm_naive( cptr<A_T> A, cptr<B_T> B, mptr<C_T> C )
        {
            // A is of size M X K
            // B is of size K X N
            // C is of size M X N
            
            // All matrices are assumed to be in row-major storage.
            
            if constexpr ( addto == Tensors::AddTo )
            {
                for( Size_T k = 0; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C[N*i+j] += A[K*i+k] * B[N*k+j];
                        }
                    }
                }
            }
            else
            {
                // First pass to overwrite (if desired).
                {
                    constexpr Size_T k = 0;
                    
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C[N*i+j] = A[K*i+k] * B[N*k+j];
                        }
                    }
                    
                }
                // Now add-in the rest.
                for( Size_T k = 1; k < K; ++k )
                {
                    for( Size_T i = 0; i < M; ++i )
                    {
                        for( Size_T j = 0; j < N; ++j )
                        {
                            C[N*i+j] += A[K*i+k] * B[N*k+j];
                        }
                    }
                }
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors

