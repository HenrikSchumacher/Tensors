#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T K, Size_T N, AddTo_T addto, typename Scal>
        void fixed_dot_mm_vec( cptr<Scal> A, cptr<Scal> B, mptr<Scal> C )
        {

            // First pass to overwrite (if desired).
            {
                constexpr Size_T k = 0;
                
                for( Size_T i = 0; i < M; ++i )
                {
                    if constexpr ( addto == Tensors::AddTo )
                    {
                        combine_buffers<F_Gen,F_Plus,M>(
                            A[K*i+k], &B[N*k], Scalar::One<Scal>, &C[N*i]
                        );
                    }
                    else
                    {
                        combine_buffers<F_Gen,F_Zero,N>(
                            A[K*i+k], &B[N*k], Scalar::Zero<Scal>, &C[N*i]
                        );
                    }
                }
            }

            // Now add-in the rest.
            for( Size_T k = 1; k < K; ++k )
            {
                for( Size_T i = 0; i < M; ++i )
                {
                    combine_buffers<F_Gen,F_Plus,N>(
                        A[K*i+k], &B[N*k], Scalar::One<Scal>, &C[N*i]
                    );
                }
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors


