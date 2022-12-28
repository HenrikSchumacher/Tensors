#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            ScalarFlag alpha_flag, ScalarFlag beta_flag,
            typename Scalar,    typename Int,
            typename Scalar_in, typename Scalar_out
        >
        void gemm(
            const Scalar_out                 alpha,
            const Scalar     * restrict const A,
            const Scalar_in  * restrict const B,
            const Scalar_out                  beta,
                  Scalar_out * restrict const C,
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.Read<opA>(A_);
            B.Read<opB>(B_);
            Dot<0>(A,B,C);
            C.Write<alpha_flag,beta_flag>(alpha,beta,C_);
        }
        
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            ScalarFlag alpha_flag, ScalarFlag beta_flag,
            typename Scalar,    typename Int,
            typename Scalar_in, typename Scalar_out
        >
        void gemm(
            const Scalar_out alpha,
            const Scalar     * restrict const A, const Int ldA,
            const Scalar_in  * restrict const B, const Int ldB,
            const Scalar_out beta,
                  Scalar_out * restrict const C, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.Read<opA>(A_,ldA);
            B.Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            ScalarFlag alpha_flag, ScalarFlag beta_flag,
            typename Scalar,    typename Int,
            typename Scalar_in, typename Scalar_out
        >
        void gemm(
            const Scalar_out alpha,
            const Scalar     * restrict const A, const Int ldA,
            const Scalar_in  * restrict const B, const Int ldB, const Int * restrict const idx,
            const Scalar_out beta,
                  Scalar_out * restrict const C, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Reads from B in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.Read<opA>(A_,ldA);
            B.Read<opB>(B_,ldB,idx);
            Dot<0>(A,B,C);
            C.Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            ScalarFlag alpha_flag, ScalarFlag beta_flag,
            typename Scalar,    typename Int,
            typename Scalar_in, typename Scalar_out
        >
        void gemm(
            const Scalar_out alpha,
            const Scalar     * restrict const A, const Int ldA,
            const Scalar_in  * restrict const B, const Int ldB,
            const Scalar_out beta,
                  Scalar_out * restrict const C, const Int ldC, const Int * restrict const idx
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Writes to C in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.Read<opA>(A_,ldA);
            B.Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC,idx);
        }
        
        
    } // namespace Tiny
    
} // namespace Tensors
