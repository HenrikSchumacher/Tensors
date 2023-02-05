#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal,    typename Int,
            typename Scal_in, typename Scal_out
        >
        void gemm(
            const Scal_out alpha,
            ptr<Scal>      A_,
            ptr<Scal_in>   B_,
            const Scal_out beta,
            mut<Scal_out>  C_
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scal,Int> A;
            Tiny::Matrix<K,N,Scal,Int> B;
            Tiny::Matrix<M,N,Scal,Int> C;
            
            A.template Read<opA>(A_);
            B.template Read<opB>(B_);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_);
        }
        
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal,    typename Int,
            typename Scal_in, typename Scal_out
        >
        void gemm(
            const Scal_out alpha,
            ptr<Scal>      A_, const Int ldA,
            ptr<Scal_in>   B_, const Int ldB,
            const Scal_out beta,
            mut<Scal_out>  C_, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scal,Int> A;
            Tiny::Matrix<K,N,Scal,Int> B;
            Tiny::Matrix<M,N,Scal,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal,    typename Int,
            typename Scal_in, typename Scal_out
        >
        void gemm(
            const Scal_out alpha,
            ptr<Scal>      A_, const Int ldA,
            ptr<Scal_in>   B_, const Int ldB, ptr<Int> idx,
            const Scal_out beta,
            mut<Scal_out>  C_, const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Reads from B in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scal,Int> A;
            Tiny::Matrix<K,N,Scal,Int> B;
            Tiny::Matrix<M,N,Scal,Int> C;
            
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB,idx);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal,    typename Int,
            typename Scal_in, typename Scal_out
        >
        void gemm(
            const Scal_out alpha,
            ptr<Scal>      A_, const Int ldA,
            ptr<Scal_in>   B_, const Int ldB,
            const Scal_out beta,
            mut<Scal_out>  C_, const Int ldC, ptr<Int> idx
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Writes to C in row-scattered fashion.

            Tiny::Matrix<M,K,Scal,Int> A;
            Tiny::Matrix<K,N,Scal,Int> B;
            Tiny::Matrix<M,N,Scal,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC,idx);
        }
        
        
    } // namespace Tiny
    
} // namespace Tensors
