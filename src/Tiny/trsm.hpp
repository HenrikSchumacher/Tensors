#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template<
            Side side,
            Triangular uplo,
            Op opA,
            Diagonal diag,
            int N,
            int NRHS,
            ScalarFlag alpha_flag,
            typename Scalar,
            typename Int,
            typename Scalar_in
        >
        void trsm(
            const Scalar_in  alpha,
            const Scalar     * restrict const A_, const Int ldA,
            const Scalar_in  * restrict const B_, const Int ldB
        )
        {
            // Solves opA(A) * C = alpha * B and stores the result in B.
            
            static_assert( side == Side::Left, "Tiny::trsm is not defined for Side::Right.");
            static_assert( uplo == Triangular::Upper, "Tiny::trsm is not defined for Triangular::Lower.");

            Tiny::UpperTriangularMatrix<N,Scalar,Int> A;
            Tiny::Matrix<N,NRHS,Scalar,Int> B;
            A.template Read(A_,ldA);
            B.template Read<Op::Identity>(B_,ldB);
            A.template Solve<opA,diag>(B);
            B.template Write<alpha_flag,ScalarFlag::Zero>(alpha,0,B_,ldB);
        }
        
        template<
            Side side,
            Triangular uplo,
            Op opA,
            Diagonal diag,
            int N,
            int NRHS,
            ScalarFlag alpha_flag,
            typename Scalar,
            typename Int,
            typename Scalar_in
        >
        void trsm(
            const Scalar_in  alpha,
            const Scalar     * restrict const A_,
            const Scalar_in  * restrict const B_
        )
        {
            // Solves opA(A) * C = alpha * B and stores the result in B.
            
            static_assert( side == Side::Left, "Tiny::trsm is not defined for Side::Right.");
            static_assert( uplo == Triangular::Upper, "Tiny::trsm is not defined for Triangular::Lower.");
            
            Tiny::UpperTriangularMatrix<N,Scalar,Int> A;
            Tiny::Matrix<N,NRHS,Scalar,Int> B;
            A.template Read(A_);
            B.template Read<Op::Identity>(B_);
            A.template Solve<opA,diag>(B);
            B.template Write<alpha_flag,ScalarFlag::Zero>(alpha,0,B_);
        }
        
    } // namespace Tiny
    
} // namespace Tensors

