
#include <iostream>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_DEBUG

#include "Tensors.hpp"
#include "Sparse.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;
using LInt   = int64_t;
using Int    = int32_t;

int main(int argc, const char * argv[])
{
//    print("Hello world!");
    constexpr Int thread_count = 8;
    constexpr Int max_depth = 4;

    const Int n = 49;

    Int perm [49] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
        36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
//
//    Int perm [49] = {13, 10, 3, 0, 17, 6, 20, 18, 11, 2, 16, 14, 7, 5, 12, 1, 19, 4, 8,
//    9, 15, 34, 31, 24, 21, 38, 27, 41, 39, 32, 23, 37, 35, 28, 26, 33,
//        22, 40, 25, 29, 30, 36, 48, 44, 46, 45, 43, 47, 42};
    
//    long long perm [49] = {28, 5, 16, 6, 42, 36, 22, 33, 3, 11, 23, 24, 46, 26, 21, 31, 14, 9, \
//        0, 34, 12, 45, 18, 1, 44, 2, 47, 29, 20, 4, 43, 35, 19, 41, 15, 13, \
//        32, 38, 10, 48, 37, 40, 8, 30, 25, 39, 27, 17, 7};
    

    LInt rp [50] = {0, 4, 10, 16, 22, 31, 40, 46, 55, 64, 70, 74, 80, 89, 95, 104, 113,
        122, 128, 134, 143, 152, 158, 167, 176, 180, 186, 192, 198, 207, 216,
        225, 231, 240, 246, 250, 256, 265, 274, 280, 289, 298, 304, 310, 319,
        328, 337, 346, 355, 361};

    Int ci [361] = {0, 2, 6, 7, 1, 2, 7, 8, 18, 19, 2, 0, 1, 6, 7, 8, 3, 5, 6, 7, 42,
        43, 4, 5, 7, 8, 19, 20, 43, 44, 45, 5, 3, 4, 6, 7, 8, 42, 43, 44, 6,
        0, 2, 3, 5, 7, 7, 0, 1, 2, 3, 4, 5, 6, 8, 8, 1, 2, 4, 5, 7, 18, 19,
        20, 9, 11, 15, 16, 18, 19, 10, 11, 16, 17, 11, 9, 10, 15, 16, 17, 12,
        14, 15, 16, 19, 20, 45, 46, 47, 13, 14, 16, 17, 47, 48, 14, 12, 13,
        15, 16, 17, 46, 47, 48, 15, 9, 11, 12, 14, 16, 18, 19, 20, 16, 9, 10,
        11, 12, 13, 14, 15, 17, 17, 10, 11, 13, 14, 16, 18, 1, 8, 9, 15, 19,
        19, 1, 4, 8, 9, 12, 15, 18, 20, 20, 4, 8, 12, 15, 19, 44, 45, 46, 21,
        23, 27, 28, 42, 43, 22, 23, 28, 29, 39, 40, 43, 44, 45, 23, 21, 22,
        27, 28, 29, 42, 43, 44, 24, 26, 27, 28, 25, 26, 28, 29, 40, 41, 26,
        24, 25, 27, 28, 29, 27, 21, 23, 24, 26, 28, 28, 21, 22, 23, 24, 25,
        26, 27, 29, 29, 22, 23, 25, 26, 28, 39, 40, 41, 30, 32, 36, 37, 39,
        40, 45, 46, 47, 31, 32, 37, 38, 47, 48, 32, 30, 31, 36, 37, 38, 46,
        47, 48, 33, 35, 36, 37, 40, 41, 34, 35, 37, 38, 35, 33, 34, 36, 37,
        38, 36, 30, 32, 33, 35, 37, 39, 40, 41, 37, 30, 31, 32, 33, 34, 35,
        36, 38, 38, 31, 32, 34, 35, 37, 39, 22, 29, 30, 36, 40, 44, 45, 46,
        40, 22, 25, 29, 30, 33, 36, 39, 41, 41, 25, 29, 33, 36, 40, 42, 3, 5,
        21, 23, 43, 43, 3, 4, 5, 21, 22, 23, 42, 44, 44, 4, 5, 20, 22, 23,
        39, 43, 45, 45, 4, 12, 20, 22, 30, 39, 44, 46, 46, 12, 14, 20, 30,
        32, 39, 45, 47, 47, 12, 13, 14, 30, 31, 32, 46, 48, 48, 13, 14, 31,
        32, 47};

    Scal a [361] = {4., -1., -1., -1., 6., -1., -1., -1., -1., -1., 6., -1., -1., -1.,
        -1., -1., 6., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1.,
        -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 6., -1.,
        -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9.,
        -1., -1., -1., -1., -1., -1., -1., -1., 6., -1., -1., -1., -1., -1.,
        4., -1., -1., -1., 6., -1., -1., -1., -1., -1., 9., -1., -1., -1.,
        -1., -1., -1., -1., -1., 6., -1., -1., -1., -1., -1., 9., -1., -1.,
        -1., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1.,
        -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 6., -1., -1., -1.,
        -1., -1., 6., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1.,
        -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 6., -1.,
        -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9.,
        -1., -1., -1., -1., -1., -1., -1., -1., 4., -1., -1., -1., 6., -1.,
        -1., -1., -1., -1., 6., -1., -1., -1., -1., -1., 6., -1., -1., -1.,
        -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9., -1., -1.,
        -1., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1.,
        -1., 6., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1.,
        -1., -1., 6., -1., -1., -1., -1., -1., 4., -1., -1., -1., 6., -1.,
        -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9.,
        -1., -1., -1., -1., -1., -1., -1., -1., 6., -1., -1., -1., -1., -1.,
        9., -1., -1., -1., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1.,
        -1., -1., -1., -1., 6., -1., -1., -1., -1., -1., 6., -1., -1., -1.,
        -1., -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9., -1., -1.,
        -1., -1., -1., -1., -1., -1., 9., -1., -1., -1., -1., -1., -1., -1.,
        -1., 9., -1., -1., -1., -1., -1., -1., -1., -1., 9., -1., -1., -1.,
        -1., -1., -1., -1., -1., 6., -1., -1., -1., -1., -1.};
    
    const Int nrhs = 3;
    
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor2<Scal,Int> X (n,nrhs,0);

    B.Random();
    
    Tensor2<Scal,Int> Y;
    
    Sparse::MatrixCSR<Scal, Int, LInt> A ( &rp[0], &ci[0], &a[0], n, n, thread_count );
    
    Scal reg = 0;
    
    for( Int rep = 0; rep < 1; ++rep )
    {
        Sparse::CholeskyDecomposition<Scal, Int, LInt> chol ( &rp[0], &ci[0], &perm[0], n, thread_count, max_depth );
        
        chol.SN_SymbolicFactorization();
        
        chol.SN_NumericFactorization(&a[0], reg);
        
        chol.Solve(B.data(), X.data(), nrhs);
        
        Y = B;
        
        dump(X.Size());
        dump(X.CountNan());
        
        A.Dot(Scal(-1), X, Scal(1), Y);
        
        dump(Y.MaxNorm());
    }
    
    
    print(ToString(Y));

//    {
//        Tensor2<Real,Int> ZZ  (3,3, 2.);
//        
//        print( ToString( ZZ.data(), {3,3} ) );
//    }
//    
//    
//    {
//        Tensor1<Real,Int> ZZ  (1*2*3*4, 4.);
//        
//        print( ToString( ZZ.data(), {1,2,3,4} ) );
//    
//    }
    
    
    return 0;
}
