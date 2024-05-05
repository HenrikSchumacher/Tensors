#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T K, Size_T N, AddTo_T addto, 
            typename A_T, typename B_T, typename C_T
        >
        void fixed_dot_mm_vec( cptr<A_T> A, cptr<B_T> B, mptr<C_T> C )
        {
            // A is of size M X K
            // B is of size K X N
            // C is of size M X N
            
            // All matrices are assumed to be in row-major storage.
            
            // First pass to overwrite (if desired).
            {
                constexpr Size_T k = 0;
                
                for( Size_T i = 0; i < M; ++i )
                {
                    if constexpr ( addto == Tensors::AddTo )
                    {
                        combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,N>(
                            A[K*i+k], &B[N*k], Scalar::One<C_T>, &C[N*i]
                        );
                    }
                    else
                    {
                        combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,N>(
                            A[K*i+k], &B[N*k], Scalar::Zero<C_T>, &C[N*i]
                        );
                    }
                }
            }

            // Now add-in the rest.
            for( Size_T k = 1; k < K; ++k )
            {
                for( Size_T i = 0; i < M; ++i )
                {
                    combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,N>(
                        A[K*i+k], &B[N*k], Scalar::One<C_T>, &C[N*i]
                    );
                }
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors


