#include <iostream>

#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_DEBUG
//#define TOOLS_AGGRESSIVE_INLINING

#ifdef __APPLE__
    #include "../Accelerate.hpp"
#else
    #include "../OpenBLAS.hpp"
#endif

#include "Tensors.hpp"
#include "Sparse.hpp"

#include "../src/Sparse/Metis.hpp"

//#include "../src/CHOLMOD/CholeskyDecomposition.hpp"
//#include "../src/CHOLMOD/ApproximateMinimumDegree.hpp"
//#include "../src/CHOLMOD/NestedDissection.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;

using LInt   = long;
using Int    = int32_t;

int main(int argc, const char * argv[])
{
    //    print("Hello world!");
    constexpr Int thread_count = 8;
    
    const char * homedir = getenv("HOME");
    
    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string home_path ( homedir );
    
    Profiler::Clear( home_path );
    
    print("");
    print("###############################################################");
    print("###     Test program for Sparse::CholeskyDecomposition     ####");
    print("###############################################################");
    print("");
    
    
    std::string path = home_path + "/github/Tensors/SparseMatrices/";
    std::string name = "Spot_4";
//    std::string name = "Spot_0";
    
    const Int nrhs = 32;
    
    Sparse::MatrixCSR<Scal,Int,LInt> A = Sparse::MatrixCSR_FromFile<Scal,Int,LInt>(
        path + name + "_Matrix.txt", thread_count
    );

    dump(A.ThreadCount());
    dump(A.RowCount());
    dump(A.NonzeroCount());
    dump(nrhs);
    
    const Int n = A.RowCount();

    Tensor1<Scal,Int> b (n);
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor1<Scal,Int> x (n,0.);
    Tensor2<Scal,Int> X (n,nrhs,0.);

    b.Random();
    B.Random();

    Tensor1<Scal,Int> y;
    Tensor2<Scal,Int> Y;


    Scal reg = 0;
    
    
    
    // Using a matrix reordering created by TAUCS works splendidly.
    Tensor1<Int,Int> p ( n );
    p.ReadFromFile(path + name + "_Permutation.txt");
    Permutation<Int> perm ( std::move(p), Inverse::False, thread_count );

    
//    tic("Metis");
//    // Corrently, I do not know how to convert metis reordings to ones that are good for parallelization.
//
//    Permutation<Int> perm = Metis<Int>()(
//        A.Outer().data(), A.Inner().data(), A.RowCount(), thread_count
//    );
//    toc("Metis");

    
    
//    tic("CHOLMOD::ApproximateMinimumDegree");
//    Permutation<Int> perm = CHOLMOD::ApproximateMinimumDegree<Int>()(
//        A.Outer().data(), A.Inner().data(), n, thread_count
//    );
//    toc("CHOLMOD::ApproximateMinimumDegree");

//    print("");
//
//    tic("CHOLMOD::NestedDissection");
//    Permutation<Int> perm = CHOLMOD::NestedDissection<int>()(
//        A.Outer().data(), A.Inner().data(), n, thread_count
//    );
//    toc("CHOLMOD::NestedDissection");


    print("");
    print("");


    tic("Cholesky constructor");
    Sparse::CholeskyDecomposition<Scal,Int,LInt> S (
        A.Outer().data(), A.Inner().data(), std::move(perm)
    );
    toc("Cholesky constructor");

    print("");

    tic("Cholesky symbolic");
    S.SymbolicFactorization();
    toc("Cholesky symbolic");

    print("");

    S.AssemblyTree().Traverse_Postordered_Test();

//    print("");
//
//    S.AssemblyTree().Traverse_Preordered_Test();

    print("");

    tic("Cholesky numeric factorization");
//    S.NumericFactorization_LeftLooking(A.Values().data(), reg);
    S.NumericFactorization_Multifrontal(A.Values().data(), reg);
    toc("Cholesky numeric factorization");

    print("");

    x.SetZero();
    tic("Cholesky vector solve");
    S.Solve<Sequential>(b.data(), x.data() );
    toc("Cholesky vector solve");
    y = b;
    A.Dot(Scal(-1), x, Scal(1), y);
    dump(y.MaxNorm());

    print("");

    X.SetZero();
    tic("Cholesky matrix solve");
    S.Solve<Sequential>(B.data(), X.data(), nrhs);
    toc("Cholesky matrix solve");
    Y = B;
    A.Dot(Scal(-1), X, Scal(1), Y);
    dump(Y.MaxNorm());

    print("");

    x.SetZero();
    tic("Cholesky parallel vector solve");
    S.Solve<Parallel>(b.data(), x.data() );
    toc("Cholesky parallel vector solve");
    y = b;
    A.Dot(Scal(-1), x, Scal(1), y);
    dump(y.MaxNorm());

    print("");

    X.SetZero();
    tic("Cholesky parallel matrix solve");
    S.Solve<Parallel>(B.data(), X.data(), nrhs );
    toc("Cholesky parallel matrix solve");
    Y = B;
    A.Dot(Scal(-1), X, Scal(1), Y);
    dump(Y.MaxNorm());
//
//
////    CHOLMOD::CholeskyDecomposition<Scal,Int,LInt> cholmod ( A.Outer().data(), A.Inner().data(), n );
////
////    cholmod.SymbolicFactorization();
////
////    cholmod.NumericFactorization(A.Values().data());
////
////    cholmod.Solve(B.data(), X.data(), nrhs);
////
////    Y = B;
////
////    A.Dot(Scal(-1), X, Scal(1), Y);
////
////    dump(Y.MaxNorm());
//
////    print(ToString(Y));
////
////    {
////        Tensor2<Real,Int> ZZ  (3,3, 2.);
////
////        print( ToString( ZZ.data(), {3,3} ) );
////    }
////
////
////    {
////        Tensor1<Real,Int> ZZ  (1*2*3*4, 4.);
////
////        print( ToString( ZZ.data(), {1,2,3,4} ) );
////
////    }
////


    print("");
    print("");

    {
        SparseMatrixStructure pat = {
            A.RowCount(), A.ColCount(), A.Outer().data(), A.Inner().data(),
            SparseAttributes_t{ false, SparseUpperTriangle, SparseSymmetric, 0, false },
            1
        };

        SparseMatrix_Double AA = { pat, A.Values().data() };

        Tensor2<Scal,Int> BT ( B.Dimension(1), B.Dimension(0));
//        Tensor2<Scal,Int> XT ( X.Dimension(1), X.Dimension(0));

        BT.ReadTransposed(B.data());

        SparseSymbolicFactorOptions opts = _SparseDefaultSymbolicFactorOptions;
        
//        enum class SparseOrder : uint8_t
//        {
//          SparseOrderDefault = 0,
//          SparseOrderUser = 1,
//          SparseOrderAMD = 2,
//          SparseOrderMetis = 3,
//          SparseOrderCOLAMD = 4,
//        );
        opts.orderMethod = SparseOrderDefault;
//        opts.orderMethod = SparseOrderAMD;
//        opts.orderMethod = SparseOrderMetis;
//        opts.orderMethod = SparseOrderUser;
//        opts.order = const_cast<Int*>(perm.GetPermutation().data());
        
        tic("Accelerate symbolic factorization");
        SparseOpaqueSymbolicFactorization Lsym = SparseFactor( 
            SparseFactorizationCholesky, 
            pat,
            opts
        );
        toc("Accelerate symbolic factorization");

        print("");

        tic("Accelerate numeric factorization");
        SparseOpaqueFactorization_Double L = SparseFactor( Lsym, AA );
        toc("Accelerate numeric factorization");

        print("");

        tic("Accelerate Cholesky vector solve");
        SparseSolve( L,
            DenseVector_Double{ b.Dimension(0), b.data() },
            DenseVector_Double{ x.Dimension(0), x.data() }
        );
        toc("Accelerate Cholesky vector solve");
        y = b;
        A.Dot(Scal(-1), x, Scal(1), y);
        dump(y.MaxNorm());

        print("");

        tic("Accelerate Cholesky matrix solve");
        SparseSolve( L,
            DenseMatrix_Double{
                B.Dimension(0), B.Dimension(1), B.Dimension(0), SparseAttributes_t(), BT.data()
            }
        );
        BT.WriteTransposed(X.data());
        toc("Accelerate Cholesky matrix solve");
        Y = B;
        A.Dot(Scal(-1), X, Scal(1), Y);
        dump(Y.MaxNorm());
    }
    
    return 0;
}
