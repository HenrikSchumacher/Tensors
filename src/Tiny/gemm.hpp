#pragma once

namespace Tensors
{
    namespace Tiny
    {
//        template<
//            const Op opA, const Op opB,
//            int M, int N, int K,
//            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
//            typename Scal_A, typename Scal_B, typename Scal_C
//        >
//        void gemm(
//            const Scal_C alpha,
//            ptr<Scal_A>  A_,
//            ptr<Scal_B>  B_,
//            const Scal_C beta,
//            mut<Scal_C>  C_
//        )
//        {
//            // Computes C = alpha * opA(A) * opB(B) + beta * C.
//            Tiny::Matrix<M,K,Scal_A,int> A;
//            Tiny::Matrix<K,N,Scal_B,int> B;
//            Tiny::Matrix<M,N,Scal_C,int> C;
//            
//            A.template Read<opA>(A_);
//            B.template Read<opB>(B_);
//            Dot<0>(A,B,C);
//            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_);
//        }
        
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, typename Int
        >
        void gemm(
            const Scal_C alpha,
            ptr<Scal_A>  A_,    const Int ldA,
            ptr<Scal_B>  B_,    const Int ldB,
            const Scal_C beta,
            mut<Scal_C>  C_,    const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, typename Int
        >
        void gemm(
            const Scal_C alpha,
            ptr<Scal_A>  A_,    const Int ldA,
            ptr<Scal_B>  B_,    const Int ldB,  ptr<Int> idx,
            const Scal_C beta,
            mut<Scal_C>  C_,    const Int ldC
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Reads from B in row-scattered fashion.
            
            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB,idx);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC);
        }
        
        template<
            const Op opA, const Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal_A, typename Scal_B, typename Scal_C, typename Int
        >
        void gemm(
            const Scal_A alpha,
            ptr<Scal_A>  A_,    const Int ldA,
            ptr<Scal_B>  B_,    const Int ldB,
            const Scal_C beta,
            mut<Scal_C>  C_,    const Int ldC,  ptr<Int> idx
        )
        {
            // Computes C = alpha * opA(A) * opB(B) + beta * C.
            // Writes to C in row-scattered fashion.

            Tiny::Matrix<M,K,Scal_A,Int> A;
            Tiny::Matrix<K,N,Scal_B,Int> B;
            Tiny::Matrix<M,N,Scal_C,Int> C;
            A.template Read<opA>(A_,ldA);
            B.template Read<opB>(B_,ldB);
            Dot<0>(A,B,C);
            C.template Write<alpha_flag,beta_flag>(alpha,beta,C_,ldC,idx);
        }
        
        
    } // namespace Tiny
    
} // namespace Tensors
