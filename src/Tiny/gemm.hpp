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
            const Scalar_out alpha,
            ptr<Scalar>      A_,
            ptr<Scalar_in>   B_,
            const Scalar_out beta,
            mut<Scalar_out>  C_
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.template Read<opA>(A_);
            B.template Read<opB>(B_);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_);
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
            ptr<Scalar>      A_, const Int ldA,
            ptr<Scalar_in>   B_, const Int ldB,
            const Scalar_out beta,
            mut<Scalar_out>  C_, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
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
            ptr<Scalar>      A_, const Int ldA,
            ptr<Scalar_in>   B_, const Int ldB, ptr<Int> idx,
            const Scalar_out beta,
            mut<Scalar_out>  C_, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Reads from B in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB,idx);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
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
            ptr<Scalar>      A_, const Int ldA,
            ptr<Scalar_in>   B_, const Int ldB,
            const Scalar_out beta,
            mut<Scalar_out>  C_, const Int ldC, ptr<Int> idx
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Writes to C in row-scattered fashion.

            Tiny::Matrix<M,K,Scalar,Int> A;
            Tiny::Matrix<K,N,Scalar,Int> B;
            Tiny::Matrix<M,N,Scalar,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC,idx);
        }
        
        
    } // namespace Tiny
    
} // namespace Tensors
