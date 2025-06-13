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

#include "../src/Sparse/ApproximateMinimumDegree.hpp"
#include "../src/Sparse/Metis.hpp"

#include "../src/CHOLMOD/CholeskyDecomposition.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;

using LInt   = long;
using Int    = int32_t;

int main()
{
    constexpr Int thread_count = 8;
    
    Profiler::Clear();
    
    print("");
    print("###############################################################");
    print("###     Test program for Sparse::CholeskyDecomposition     ####");
    print("###############################################################");
    print("");
    
    
    Int grid_size = 1024 * 2 * 2;
    Real mass = 1;
    
    // Assembling graph Laplacian + mass matrix on grid of size grid_size x grid_size.
    TripleAggregator<Int,Int,Real,LInt> triples;

    for( Int i = 0; i < grid_size - 1; ++i )
    {
        for( Int j = 0; j < grid_size - 1; ++j )
        {
            Int V = grid_size * i + j;
            Int L = V + grid_size;
            Int R = V + 1;
            
            triples.Push( V, V,  1 );
            triples.Push( V, R, -1 );
            triples.Push( R, V, -1 );
            triples.Push( R, R,  1 );
            
            triples.Push( V, V,  1 );
            triples.Push( V, L, -1 );
            triples.Push( L, V, -1 );
            triples.Push( L, L,  1 );
        }
    }
    
    for( Int i = 0; i < grid_size - 1; ++i )
    {
        Int j = grid_size-1;
        Int V = grid_size * i + j;
        Int L = V + grid_size;
        
        triples.Push( V, V,  1 );
        triples.Push( V, L, -1 );
        triples.Push( L, V, -1 );
        triples.Push( L, L,  1 );
    }
    
    for( Int j = 0; j < grid_size - 1; ++j )
    {
        Int i = grid_size - 1;
        Int V = grid_size * i + j;
        Int R = V + 1;
        
        triples.Push( V, V,  1 );
        triples.Push( V, R, -1 );
        triples.Push( R, V, -1 );
        triples.Push( R, R,  1 );
    }
    
    for( Int i = 0; i < grid_size; ++i )
    {
        for( Int j = 0; j < grid_size; ++j )
        {
            Int V = grid_size * i + j;
            triples.Push( V, V, mass );
        }
    }
    
    tic("Assemble matrix");
    Sparse::MatrixCSR<Scal,Int,LInt> A (
        triples,
        grid_size * grid_size, grid_size * grid_size,
        thread_count, true, false, true
    );
    toc("Assemble matrix");
    
    triples = TripleAggregator<Int,Int,Real,LInt>();
    
    TOOLS_DUMP(A.ProvenInnerSortedQ());
    TOOLS_DUMP(A.ProvenDuplicatedFreeQ());
    TOOLS_DUMP(A.RowCount());
    TOOLS_DUMP(A.ColCount());
    TOOLS_DUMP(A.NonzeroCount());
    
//    auto A_dense = A.ToTensor2();
//    valprint(
//        "A_dense",
//        ToString(A_dense,[](const double x){return ToStringFPGeneral(x);})
//    );
    
//    A.Outer().WriteToFile("/Users/Henrik/A_rp.txt");
//    A.Inner().WriteToFile("/Users/Henrik/A_ci.txt");
//    A.Values().WriteToFile("/Users/Henrik/A_val.txt");
//
//    Tensor2<Real,Int> B( A.RowCount(), A.ColCount(), Real(0) );
//    A.WriteDense(B.data(),grid_size * grid_size);
//
//    B.WriteToFile("/Users/Henrik/b.txt");
        
    constexpr Int NRHS = 4;
    const     Int nrhs = NRHS;
    
    const Int n = A.RowCount();
    
    Tensor1<Scal,Int> b (n);
    Tensor1<Scal,Int> x (n);
    
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor2<Scal,Int> X (n,nrhs);

    const Int ldB = nrhs;
    const Int ldX = nrhs;
    const Int ldY = ldB;
    
    const Scal alpha     = 1;
    const Scal alpha_inv = Frac<Scal>(1,alpha);
    const Scal beta      = 0;
    
    b.Random();
    B.Random();

    Tensor1<Scal,Int> y;
    Tensor2<Scal,Int> Y;


    Scal reg = 0.;
    
    print("");


    print("");
    print("");
    {
        
        
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
        
        S.AssemblyTree().Traverse_PostOrdered_Test();
        
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
        tic("Cholesky sequential matrix solve -- pointer arguments");
        S.Solve<NRHS,Sequential>( alpha, B.data(), ldB, beta, X.data(), ldX, nrhs);
        toc("Cholesky sequential matrix solve -- pointer arguments");
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
    }
#define TEST_ACCELERATE
    
#ifdef TEST_ACCELERATE
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
//        opts.orderMethod = SparseOrderDefault;
        opts.orderMethod = SparseOrderAMD;
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
#endif
    
    return 0;
}
