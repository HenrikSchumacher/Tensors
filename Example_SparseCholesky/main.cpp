// Testing the Cholesky solver.

// TODO: Test whether slicing (i.e. the use of leading dimensions) works.

#include <iostream>

//#undef _OPENMP

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

//#include "../src/Sparse/Metis.hpp"
#include "../src/Sparse/ApproximateMinimumDegree.hpp"

#include "../src/CHOLMOD/CholeskyDecomposition.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;

using LInt   = long;
using Int    = int32_t;

int main()
{
    constexpr Int thread_count = 1;

    std::filesystem::path home_dir = HomeDirectory();
    
    Profiler::Clear( home_dir );
    
    print("");
    print("###############################################################");
    print("###     Test program for Sparse::CholeskyDecomposition     ####");
    print("###############################################################");
    print("");
    
    
    std::string name = "Spot_4";
//    std::string name = "Spot_0";
    
    constexpr Int NRHS = 32;
    const     Int nrhs = NRHS;
    
    Sparse::MatrixCSR<Scal,Int,LInt> A;
    
    A.LoadFromFile( home_dir / "github/Tensors/SparseMatrices" / (name + "_Matrix.txt"), thread_count );
    
//    A.LoadFromMatrixMarket( home_dir / "MatrixMarket" / "bcsstk14.mtx", thread_count );

    
//    std::vector< Int> i = { 0, 0, 0, 1, 1, 1, 2, 2, 2 };
//    std::vector< Int> j = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
//    std::vector<Scal> a = { 2.2, 1.1, 0, 1.1, 2.2, 0, 0, 0, 2.2 };
//    
//    Sparse::MatrixCSR<Scal,Int,LInt> A(
//        9, &i[0], &j[0], &a[0], 3, 3, 1, true, false
//    );
    
    const Int n = A.RowCount();
    
    Tensor1<Scal,Int> b (n);
    Tensor1<Scal,Int> x (n);
    
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor2<Scal,Int> X (n,nrhs);

    const Int ldB = nrhs;
    const Int ldX = nrhs;
    const Int ldY = ldB;
    
//    const Scal alpha     = 2.4;
    const Scal alpha     = 1;
    const Scal alpha_inv = Frac<Scal>(1,alpha);
    const Scal beta      = 0;
    
    b.Random();
    B.Random();

    Tensor1<Scal,Int> y;
    Tensor2<Scal,Int> Y;


    Scal reg = 0;
    
    print("");
    
//    // Using a matrix reordering created by TAUCS works splendidly.
//    Tensor1<Int,Int> p ( n );
//    p.ReadFromFile(path / (name + "_Permutation.txt") );
//    Permutation<Int> perm ( std::move(p), Inverse::False, thread_count );

//    
//    tic("Metis");
//    Permutation<Int> perm = Sparse::Metis<Int>()(
//        A.Outer().data(), A.Inner().data(), A.RowCount(), thread_count
//    );
//    toc("Metis");
    
    tic("AMD");
    Permutation<Int> perm = Sparse::ApproximateMinimumDegree<Int>()(
        A.Outer().data(), A.Inner().data(), A.RowCount(), thread_count
    );
    toc("AMD");


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

    print("");

    tic("Cholesky numeric factorization");
//    S.NumericFactorization_LeftLooking( A.Values().data(), reg );
    S.NumericFactorization_Multifrontal(A.Values().data(), reg);
    toc("Cholesky numeric factorization");

    print("");
    print("Sequential solves");
    print("");
    
    x.SetZero();
    tic("Cholesky sequential vector solve -- Tensor1 arguments");
    S.Solve<1,Sequential>( alpha, b.data(), 1, beta, x.data(), 1 );
    toc("Cholesky sequential vector solve -- Tensor1 arguments");
    y = b;
    A.Dot( -alpha_inv, x, Scal(1), y);
    TOOLS_DUMP(y.MaxNorm());
    
    print("");
    
    x.SetZero();
    tic("Cholesky sequential vector solve -- pointer arguments");
    S.Solve<1,Sequential>( alpha, b.data(), 1, beta, x.data(), 1, 1 );
    toc("Cholesky sequential vector solve -- pointer arguments");
    y = b;
    A.Dot<1>( -alpha_inv, x.data(), 1, Scal(1), y.data(), 1);
    TOOLS_DUMP(y.MaxNorm());

    print("");

    X.SetZero();
    tic("Cholesky sequential matrix solve -- Tensor2 arguments");
    S.Solve<VarSize,Sequential>( alpha, B, beta, X );
    toc("Cholesky sequential matrix solve -- Tensor2 arguments");
    Y = B;
    A.Dot<VarSize>( -alpha_inv, X, Scal(1), Y );
    TOOLS_DUMP(Y.MaxNorm());
    
    print("");
    
    X.SetZero();
    tic("Cholesky matrix solve -- pointer arguments");
    S.Solve<NRHS,Sequential>( alpha, B.data(), ldB, beta, X.data(), ldX, nrhs);
    toc("Cholesky matrix solve -- pointer arguments");
    Y = B;
    A.Dot<NRHS>( -alpha_inv, X.data(), ldX, Scal(1), Y.data(), ldY, nrhs );
    TOOLS_DUMP(Y.MaxNorm());

    print("");
    print("Parallel solves");
    print("");

    x.SetZero();
    tic("Cholesky parallel vector solve -- Tensor1 arguments");
    S.Solve<Parallel>( alpha, b, beta, x );
    toc("Cholesky parallel vector solve -- Tensor1 arguments");
    y = b;
    A.Dot( -alpha_inv, x, Scal(1), y);
    TOOLS_DUMP(y.MaxNorm());
    
    print("");

    x.SetZero();
    tic("Cholesky parallel vector solve -- pointer arguments");
    S.Solve<1,Parallel>( alpha, b.data(), 1, beta, x.data(), 1 );
    toc("Cholesky parallel vector solve -- pointer arguments");
    y = b;
    A.Dot( -alpha_inv, x, Scal(1), y);
    TOOLS_DUMP(y.MaxNorm());

    print("");
    
    X.SetZero();
    tic("Cholesky parallel matrix solve -- Tensor2 arguments");
    S.Solve<VarSize,Parallel>( alpha, B, beta, X);
    toc("Cholesky parallel matrix solve -- Tensor2 arguments");
    Y = B;
    A.Dot( -alpha_inv, X, Scal(1), Y );
    TOOLS_DUMP(Y.MaxNorm());
    
    print("");
    
    X.SetZero();
    tic("Cholesky parallel matrix solve -- pointer arguments");
    S.Solve<NRHS,Parallel>( alpha, B.data(), ldB, beta, X.data(), ldX, nrhs);
    toc("Cholesky parallel matrix solve -- pointer arguments");
    Y = B;
    A.Dot<NRHS>( -alpha_inv, X.data(), ldX, Scal(1), Y.data(), ldY, nrhs );
    TOOLS_DUMP(Y.MaxNorm());


    print("");
    print("");

    {
        SparseMatrixStructure pat = {
            A.RowCount(), A.ColCount(), A.Outer().data(), A.Inner().data(),
            SparseAttributes_t{ false, SparseUpperTriangle, SparseSymmetric, 0, false },
            1
        };

        SparseMatrix_Double AA = { pat, A.Values().data() };

        Tensor2<Scal,Int> BT ( B.Dim(1), B.Dim(0));
//        Tensor2<Scal,Int> XT ( X.Dim(1), X.Dim(0));

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
            DenseVector_Double{ b.Dim(0), b.data() },
            DenseVector_Double{ x.Dim(0), x.data() }
        );
        toc("Accelerate Cholesky vector solve");
        y = b;
        A.Dot(Scal(-1), x, Scal(1), y);
        TOOLS_DUMP(y.MaxNorm());

        print("");

        tic("Accelerate Cholesky matrix solve");
        SparseSolve( L,
            DenseMatrix_Double{
                B.Dim(0), B.Dim(1), B.Dim(0), SparseAttributes_t(), BT.data()
            }
        );
        BT.WriteTransposed(X.data());
        toc("Accelerate Cholesky matrix solve");
        Y = B;
        A.Dot(Scal(-1), X, Scal(1), Y);
        TOOLS_DUMP(Y.MaxNorm());
    }
    
    
    
//    TOOLS_DUMP(x[0]);
//    TOOLS_DUMP(x[1]);
//    TOOLS_DUMP(x[2]);
//    TOOLS_DUMP(x[3]);
//    
//    print("");
//    print("");
//    
//    tic("CHOLMOD constructor");
//    CHOLMOD::CholeskyDecomposition<Scal,Int32,Int32> cholmod (
//        A.Outer().data(), A.Inner().data(), A.ColCount()
//    );
//    toc("CHOLMOD constructor");
//    
//    print("");
//    
//    tic("CHOLMOD symbolic factorization");
//    cholmod.SymbolicFactorization();
//    toc("CHOLMOD symbolic factorization");
//    
//    print("");
//    
//    tic("CHOLMOD numeric factorization");
//    cholmod.NumericFactorization(A.Values().data());
//    toc("CHOLMOD numeric factorization");
//
//    print("");
//    print("");
//
//    TOOLS_DUMP(b[0]);
//    TOOLS_DUMP(b[1]);
//    TOOLS_DUMP(b[2]);
//    TOOLS_DUMP(b[3]);
//    x.SetZero();
//    
//    tic("CHOLMOD Cholesky vector solve");
//    cholmod.Solve( b.data(), x.data() );
//    toc("CHOLMOD Cholesky vector solve");
//    
//    TOOLS_DUMP(b[0]);
//    TOOLS_DUMP(b[1]);
//    TOOLS_DUMP(b[2]);
//    TOOLS_DUMP(b[3]);
//    
//    TOOLS_DUMP(x[0]);
//    TOOLS_DUMP(x[1]);
//    TOOLS_DUMP(x[2]);
//    TOOLS_DUMP(x[3]);
//    
//    y = b;
//    A.Dot(Scal(-1), x, Scal(1), y);
//    TOOLS_DUMP(y.MaxNorm());
//
//    print("");
    
    return 0;
}
