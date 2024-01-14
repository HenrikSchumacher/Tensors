template<
//    Scalar::Flag a_flag, Scalar::Flag b_flag,
    typename alpha_T, typename beta_T, 
    typename A_ExT,   typename B_ExT,  typename C_ExT,
    typename I_A,     typename I_B,    typename I_C
>
void gemm(
          cref<alpha_T> alpha, cptr<A_ExT> A, const I_A ldA,
                               cptr<B_ExT> B, const I_B ldB,
          cref<beta_T>  beta,  mptr<C_ExT> C, const I_C ldC
)
{
    ptic(ClassName()+"::gemm");
    
    AP.Read( A, ldA );

    BP.Read( B, ldB );

    DotBlocks();

    CP.Write( alpha, beta, C, ldC );

    ptoc(ClassName()+"::gemm");
}
