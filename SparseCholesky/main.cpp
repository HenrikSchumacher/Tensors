#include <iostream>

#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_DEBUG

#define LAPACK_DISABLE_NAN_CHECK
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

//#include <cblas.h>
//#include <lapacke.h>

#include "Tensors.hpp"
#include "Sparse.hpp"

#include "../src/Sparse/Metis.hpp"

#include "../src/CHOLMOD/CholeskyDecomposition.hpp"
#include "../src/CHOLMOD/ApproximateMinimumDegree.hpp"
#include "../src/CHOLMOD/NestedDissection.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;

using LInt   = long;
using Int    = int32_t;

int main(int argc, const char * argv[])
{
//    print("Hello world!");
    constexpr Int thread_count   = 8;
    constexpr Int tree_top_depth = 5;
    
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
    std::string name = "Spot_3";
    
    const Int nrhs = 32;
    
    Sparse::MatrixCSR<Scal, Int, LInt> A = Sparse::MatrixCSR_FromFile<Scal,Int,LInt>(
        path + name + "_Matrix.txt", thread_count
    );
    
    dump(A.RowCount());
    dump(A.NonzeroCount());
    
    const Int n = A.RowCount();
    
    Tensor1<Int,Int> p ( n );
    
    p.ReadFromFile(path + name + "_Permutation.txt");
    
    
    Permutation<Int> perm ( std::move(p), Inverse::False, thread_count );


    
    Tensor1<Scal,Int> b (n);
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor1<Scal,Int> x (n,0.);
    Tensor2<Scal,Int> X (n,nrhs,0.);

    b.Random();
    B.Random();
    
    Tensor1<Scal,Int> y;
    Tensor2<Scal,Int> Y;
    
    
    Scal reg = 0;
    
    
//    Permutation<Int> perm ( perm_array, n, Inverse::False, thread_count );
    
//    Permutation<Int> perm ( n, thread_count );
    
//    Metis_Wrapper()( &rp[0], &ci[0], perm );
    
//    ptic("Metis");
//    Metis()( A.Outer().data(), A.Inner().data(), perm );
//    ptoc("Metis");
//    print( perm.GetPermutation().ToString() );
    
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
        A.Outer().data(), A.Inner().data(), std::move(perm), tree_top_depth
    );
    toc("Cholesky constructor");
    
    print("");
    
    tic("Cholesky symbolic");
    S.SymbolicFactorization();
    toc("Cholesky symbolic");
    
    print("");
    
    tic("Cholesky numeric factorization");
    S.NumericFactorization(A.Values().data(), reg);
    toc("Cholesky numeric factorization");
    
    print("");
    
    x.SetZero();
    tic("Cholesky vector solve");
    S.Solve<false>(b.data(), x.data() );
    toc("Cholesky vector solve");
    y = b;
    A.Dot(Scal(-1), x, Scal(1), y);
    dump(y.MaxNorm());
    
    print("");
    
    X.SetZero();
    tic("Cholesky matrix solve");
    S.Solve<false>(B.data(), X.data(), nrhs);
    toc("Cholesky matrix solve");
    Y = B;
    A.Dot(Scal(-1), X, Scal(1), Y);
    dump(Y.MaxNorm());
    
    print("");
    
    x.SetZero();
    tic("Cholesky parallel vector solve");
    S.Solve<true>(b.data(), x.data() );
    toc("Cholesky parallel vector solve");
    y = b;
    A.Dot(Scal(-1), x, Scal(1), y);
    dump(y.MaxNorm());
    
    print("");
    
    X.SetZero();
    tic("Cholesky parallel matrix solve");
    S.Solve<true>(B.data(), X.data(), nrhs );
    toc("Cholesky parallel matrix solve");
    Y = B;
    A.Dot(Scal(-1), X, Scal(1), Y);
    dump(Y.MaxNorm());
    
    
//    CHOLMOD::CholeskyDecomposition<Scal,Int,LInt> cholmod ( A.Outer().data(), A.Inner().data(), n );
//
//    cholmod.SymbolicFactorization();
//
//    cholmod.NumericFactorization(A.Values().data());
//
//    cholmod.Solve(B.data(), X.data(), nrhs);
//
//    Y = B;
//
//    A.Dot(Scal(-1), X, Scal(1), Y);
//
//    dump(Y.MaxNorm());
    
//    print(ToString(Y));
//
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
//
    
    
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
        
//        unc SparseFactor(
//            _ type: SparseFactorization_t,
//            _ Matrix: SparseMatrixStructure
//        ) -> SparseOpaqueSymbolicFactorization

        tic("Accelerate symbolic factorization");
        SparseOpaqueSymbolicFactorization Lsym = SparseFactor( SparseFactorizationCholesky, pat );
        toc("Accelerate symbolic factorization");
        
//        int * bla = reinterpret_cast<int*>(Lsym.factorization) + 46;
//
//        int dims [1] = {n};
//
//        print( ArrayToString(bla, &dims[0], 1) );
        print("");
        
        tic("Accelerate numeric factorization");
        SparseOpaqueFactorization_Double L = SparseFactor( Lsym, AA );
        toc("Accelerate numeric factorization");

        print("");
        
        tic("Accelerate Cholesky vector solve");
        SparseSolve(
            L,
            DenseVector_Double{ b.Dimension(0), b.data() },
            DenseVector_Double{ x.Dimension(0), x.data() }
        );
        toc("Accelerate Cholesky vector solve");
        y = b;
        A.Dot(Scal(-1), x, Scal(1), y);
        dump(y.MaxNorm());
        
        print("");
        
        tic("Accelerate Cholesky matrix solve");
        SparseSolve(
            L,
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



//SparseMatrix_Double A = {
//       SparseMatrixStructure {
//             n, n, reinterpret_cast<long>(rp.data()), reinterpret_cast<long>(ci.data()),
//             SparseAttributes_t {
//                   false, SparseUpperTriangle, SparseSymmetric, 0, false
//               },
//             1
//         } ,
//       a . data ()
//   };
//
//Tensor2 < Scal, int > BT ( nrhs, n);
//
//BT . ReadTransposed (B . data ());
//
//tic (\" Accelerate Cholesky factorization \");
//SparseOpaqueFactorization_Double L =
//  SparseFactor ( SparseFactorizationCholesky, A );
//toc (\" Accelerate Cholesky factorization \");
//
//tic (\" Accelerate Cholesky solve \");
//SparseSolve (
//       L,
//       DenseMatrix_Double { n, nrhs, n, SparseAttributes_t (),
//     BT . data () }
//   );
//toc (\" Accelerate Cholesky solve \");
//
//auto X = mma::makeMatrix < mreal > ( n, nrhs );
//
//BT . WriteTransposed (X . data ());
