#include <iostream>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_DEBUG

#ifdef __APPLE__
/// Use these while on a mac. Don't forget to issue the compiler flag `-framework Accelerate`.
///
    #include "../Accelerate.hpp"
#else
/// This should work for OpenBLAS.
    #include "../OpenBLAS.hpp"
#endif

#include "Tensors.hpp"
#include "BLAS_Wrappers.hpp"
#include "LAPACK_Wrappers.hpp"
#include "Dense.hpp"


using namespace Tools;
using namespace Tensors;

using Real    = Real64;
using Complex = Complex64;

using A_T     = Real;
using B_T     = Real;
using C_T     = Real;

using alpha_T = Real;
using beta_T  = Real;


//using A_T     = Complex;
//using B_T     = Complex;
//using C_T     = Complex;
//
//using alpha_T = Complex;
//using beta_T  = Complex;

using LInt    = long;
using Int     = int_fast32_t;
//using Int     = int64_t;

int main(int argc, const char * argv[])
{
    constexpr Int thread_count = 8;

    std::string home_path = HomeDirectory();
    
    print("");
    print("###############################################################");
    print("###               Test program for DenseBLAS               ####");
    print("###############################################################");
    print("");
    
    
//    constexpr Size_T M = 2048 + 1;
//    constexpr Size_T K = 2048 + 1;
//    constexpr Size_T N = 2048 + 1;

//    constexpr Size_T M = 1048 * 2;
//    constexpr Size_T K = 1048 * 2;
//    constexpr Size_T N = 1048 * 2;
    
    constexpr Size_T M = 1024 * 2 * 2;
    constexpr Size_T K = 1024 * 2 * 2;
    constexpr Size_T N = 1024 * 2 * 2;
    
//    constexpr Size_T m = 16;
//    constexpr Size_T k =  8;
//    constexpr Size_T n =  4;
    
//    constexpr Size_T m = 4;
//    constexpr Size_T k = 4;
//    constexpr Size_T n = 4;
    
//    constexpr Size_T M = 32;
//    constexpr Size_T K = 32;
//    constexpr Size_T N = 32;

    constexpr Size_T m = 16;
    constexpr Size_T k =  8;
    constexpr Size_T n =  4;
    
//    constexpr Size_T m = 8;
//    constexpr Size_T k = 8;
//    constexpr Size_T n = 8;

    
    constexpr Op opA = Op::Id;
    constexpr Op opB = Op::Id;
    constexpr Op opC = Op::Id;
    
//    constexpr Size_T M = 128 * 2;
//    constexpr Size_T K = 128;
//    constexpr Size_T N = 128 /2;
    
//    constexpr Size_T M = 128;
//    constexpr Size_T K = 128;
//    constexpr Size_T N = 128;
//
//    
//    constexpr Size_T m = 16;
//    constexpr Size_T k = 8;
//    constexpr Size_T n = 4;
    
//    constexpr Size_T M = 4 + 1;
//    constexpr Size_T K = 4 + 1;
//    constexpr Size_T N = 4 + 1;
//
//    
//    constexpr Size_T m = 2;
//    constexpr Size_T k = 2;
//    constexpr Size_T n = 2;

  
//    constexpr Size_T M =  4 * 4;
//    constexpr Size_T K =  4 * 4;
//    constexpr Size_T N =  4 * 4;
//    
//    
//    constexpr Size_T m =  4;
//    constexpr Size_T k =  4;
//    constexpr Size_T n =  2;
    
    
//    constexpr Size_T M =  4;
//    constexpr Size_T K =  4;
//    constexpr Size_T N =  4;
//    constexpr Size_T m =  2;
//    constexpr Size_T k =  2;
//    constexpr Size_T n =  2;

    
    Tensor2<A_T,Int> A ( M, N );
    Tensor2<B_T,Int> B ( N, K );
    Tensor2<C_T,Int> C ( M, N );
    Tensor2<C_T,Int> E ( M, N );
    
    A.Random();
    B.Random();
    
//    A.SetZero();
//    B.SetZero();
//    
//    A[2][3]  = 1;
//    B[3][4]  = 1;

    const alpha_T alpha = 1;
    const beta_T  beta  = 0;
    
    tic("gemm_Accelerate");
    BLAS::gemm<Layout::RowMajor,opA,opB>(
        M, N, K,
        alpha, A.data(), K,
               B.data(), N,
        beta,  E.data(), N
    );
    toc("gemm_Accelerate");
    
    print("");

    
    Profiler::Clear( home_path +"/C" );
    
    ptic("Initialize");
//    Dense::BLAS_3<M,N,K,m,n,k,opA,opB,opC,A_T,B_T,C_T,Int> blas3 ( M, N, K, thread_count );
    Dense::BLAS_3<VarSize,VarSize,VarSize,m,n,k,opA,opB,opC,A_T,B_T,C_T,Int> blas3 ( M, N, K, thread_count );
    
    dump(blas3.ClassName());
    ptoc("Initialize");
   
    print("");
    
    tic("blas3.gemm");
    blas3.gemm(
        alpha, A.data(), K,
               B.data(), N,
        beta,  C.data(), N
    );
    toc("blas3.gemm");
    
    tic("blas3.gemm");
    blas3.gemm(
        alpha, A.data(), K,
               B.data(), N,
        beta,  C.data(), N
    );
    toc("blas3.gemm");
    
    tic("blas3.gemm");
    blas3.gemm(
        alpha, A.data(), K,
               B.data(), N,
        beta,  C.data(), N
    );
    toc("blas3.gemm");
    
//    tic("blas3.gemm_recursive");
//    blas3.gemm_recursive(
//        alpha, A.data(), K,
//               B.data(), N,
//        beta,  C.data(), N
//    );
//    toc("blas3.gemm_recursive");
//    
//    tic("blas3.gemm_recursive");
//    blas3.gemm_recursive(
//        alpha, A.data(), K,
//               B.data(), N,
//        beta,  C.data(), N
//    );
//    toc("blas3.gemm_recursive");
    
//    tic("Dense::gemm");
//    Dense::gemm<Layout::RowMajor,opA,opB>(
//        M, N, K,
//        alpha, A.data(), K,
//               B.data(), N,
//        beta,  C.data(), N,
//        thread_count
//    );
//    toc("Dense::gemm");
    
    print("");
    
//    dump(E);
    
    E -= C;
    
    
    dump(E.MaxNorm());
    dump(E.FrobeniusNorm());
//
//    dump(A);
//    dump(blas3.AP);
//    
//    dump(B);
//    dump(blas3.BP);
//    
//    
//    dump(C);
//    dump(blas3.CP);
    
    print("");
    
    return 0;
}
