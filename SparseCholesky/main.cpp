#include <iostream>

#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER
#define TOOLS_DEBUG

#define LAPACK_DISABLE_NAN_CHECK
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
//#include <cblas.h>
//#include <lapacke.h>

#include "Tensors.hpp"
#include "MyBLAS.hpp"
#include "Sparse.hpp"

#include "../src/Sparse/Metis.hpp"

#include "../src/CHOLMOD/CholeskyDecomposition.hpp"
#include "../src/CHOLMOD/ApproximateMinimumDegree.hpp"
#include "../src/CHOLMOD/NestedDissection.hpp"

using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;
//using LInt   = int64_t;
//using LInt   = int64_t;
//using Int    = int64_t;

using LInt   = long;
using Int    = int32_t;

int main(int argc, const char * argv[])
{
//    print("Hello world!");
    constexpr Int thread_count = 8;
    constexpr Int tree_top_depth = 5;
    
    const char * homedir = getenv("HOME");

    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
    Profiler::Clear( path );
    
    std::string filename = path + "/github/Tensors/SparseMatrices/Spot_0.txt";
    
    const Int nrhs = 3;
    
//    Sparse::MatrixCSR<Scal, Int, LInt> A ( &rp[0], &ci[0], &a[0], n, n, thread_count );
    
    Sparse::MatrixCSR<Scal, Int, LInt> A = Sparse::MatrixCSR_FromFile<Scal,Int,LInt>(filename, thread_count);
    
    const Int n = A.RowCount();
    
    Tensor2<Scal,Int> B (n,nrhs);
    Tensor2<Scal,Int> X (n,nrhs,0);

    B.Random();
    
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
    
    tic("CHOLMOD::NestedDissection");
    Permutation<Int> perm = CHOLMOD::NestedDissection<int>()(
        A.Outer().data(), A.Inner().data(), n, thread_count
    );
    toc("CHOLMOD::NestedDissection");


    Sparse::CholeskyDecomposition<Scal,Int,LInt> S ( A.Outer().data(), A.Inner().data(), std::move(perm), tree_top_depth );

    tic("My Cholesky factorization");
    S.SymbolicFactorization();

    S.NumericFactorization(A.Values().data(), reg);
    toc("My Cholesky factorization");
    
    tic("My Cholesky solve");
    S.Solve(B.data(), X.data(), nrhs);
    toc("My Cholesky solve");
    
    Y = B;
    
    dump(X.Size());
    dump(X.CountNan());
    
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
    

    
    {
        SparseMatrixStructure pat = {
            A.RowCount(), A.ColCount(), A.Outer().data(), A.Inner().data(),
            SparseAttributes_t{
                false, SparseUpperTriangle, SparseSymmetric, 0, false
            },
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
        
        tic("Accelerate numeric factorization");
        SparseOpaqueFactorization_Double L = SparseFactor( Lsym, AA );
        toc("Accelerate numeric factorization");
        
        tic("Accelerate Cholesky solve");
        SparseSolve(
            L,
            DenseMatrix_Double{ B.Dimension(0), B.Dimension(1), B.Dimension(0), SparseAttributes_t(), BT.data() }
        );
        toc("Accelerate Cholesky solve");
        
        BT.WriteTransposed(X.data());
        
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
