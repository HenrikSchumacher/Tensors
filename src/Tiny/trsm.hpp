#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template<
            Side side, UpLo uplo, Op opA, Diag diag,
            int N, int NRHS,
            Scalar::Flag alpha_flag,
            typename Scal, typename Int, typename Scal_in
        >
        void trsm(
            const Scal_in alpha, ptr<Scal>    A_, const Int ldA,
                                   mut<Scal_in> B_, const Int ldB
        )
        {
            // Solves opA(A) * C = alpha * B and stores the result in B.
            
            static_assert( side == Side::Left, "Tiny::trsm is not defined for Side::Right.");
            static_assert( uplo == UpLo::Upper, "Tiny::trsm is not defined for UpLo::Lower.");

            Tiny::UpperTriangularMatrix<N,Scal,Int> A;
            Tiny::Matrix<N,NRHS,Scal,Int> B;
            A.template Read(A_,ldA);
            B.template Read<Op::Id>(B_,ldB);
            A.template Solve<opA,diag>(B);
            B.template Write<alpha_flag,Scalar::Flag::Zero>(alpha,0,B_,ldB);
        }
        
        template<
            Side side, UpLo uplo, Op opA, Diag diag,
            int N, int NRHS,
            Scalar::Flag alpha_flag,
            typename Scal, typename Int, typename Scal_in
        >
        void trsm( const Scal_in  alpha, ptr<Scal> A_, mut<Scal_in> const B_ )
        {
            // Solves opA(A) * C = alpha * B and stores the result in B.
            
            static_assert( side == Side::Left, "Tiny::trsm is not defined for Side::Right.");
            static_assert( uplo == UpLo::Upper, "Tiny::trsm is not defined for UpLo::Lower.");
            
            Tiny::UpperTriangularMatrix<N,Scal,Int> A;
            Tiny::Matrix<N,NRHS,Scal,Int> B;
            A.template Read(A_);
            B.template Read<Op::Id>(B_);
            A.template Solve<opA,diag>(B);
            B.template Write<alpha_flag,Scalar::Flag::Zero>(alpha,0,B_);
        }
        
    } // namespace Tiny
    
} // namespace Tensors

