#include <iostream>

#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_DEBUG

#ifdef __APPLE__
    #include "../Accelerate.hpp"
#else
    #include "../OpenBlas.hpp"
#endif

#include "Tensors.hpp"
#include "Sparse.hpp"

#include "../src/Sparse/BlockLUDecomposition.hpp"


using namespace Tools;
using namespace Tensors;

using Scal   = double;
using Real   = Scalar::Real<Scal>;

using LInt   = long;
using Int    = int32_t;

using Solver_T = Sparse::BlockLUDecomposition<Scal,Int,LInt,256> ;
using Block_T  = Solver_T::Block_T;

int main(int argc, const char * argv[])
{
    //    print("Hello world!");
    constexpr Int thread_count   = 8;
    
    const char * homedir = getenv("HOME");
    
    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string home_path ( homedir );
    
    Profiler::Clear( home_path );
    
    print("");
    print("###############################################################");
    print("###     Test program for Sparse::BlockLUDecomposition      ####");
    print("###############################################################");
    print("");
    
    
    std::string path = home_path + "/github/Tensors/SparseMatrices/";
//    std::string name = "Spot_4";
    std::string name = "Spot_2";
    
    const Int nrhs = 32;
    
    Sparse::MatrixCSR<Scal,Int,LInt> A = Sparse::MatrixCSR_FromFile<Scal,Int,LInt>(
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


    print("");
    print("");


    tic("LU constructor");
    Solver_T S (
        A.Outer().data(), A.Inner().data(), n
    );
    toc("LU constructor");

    tic("LU constructor");
    S.LoadValues( A.Value().data() );
    toc("LU constructor");
    
    tic("LU factorize");
    S.SymbolicFactorization();
//    S.Factorize();
    toc("LU factorize");
    
    print("");
    
//    Block_T block;
//    
//    S.WriteBlock(0,0,block.data());
//    
//    dump(block);
    
    print("");
    
    
//    S.WriteBlock(0,2,block.data());
    
//    dump(block);
    
    print("");

    
    return 0;
}
