
#include <iostream>

#define TOOLS_ENABLE_PROFILER
//#define TOOLS_AGGRESSIVE_INLINING

#ifdef __APPLE__
/// Use these while on a mac. Don't forget to issue the compiler flag `-framework Accelerate`.
///
    #include "Accelerate.hpp"
#else
/// This should work for OpenBLAS.
    #include "OpenBLAS.hpp"
#endif

#include "Tensors.hpp"

//using namespace Repulsor;
using namespace Tensors;
using namespace Tools;

using Real    = Real64;
using Complex = Complex64;
using Int     = Int32;
using LInt    = Int64;

//using Int     = Int64;
//using LInt    = Int64;

//using Int     = UInt32;
//using LInt    = UInt64;


int main(int argc, const char * argv[])
{
    
    Profiler::Clear();
    print("");
    
    const Int thread_count = 8;
    
    const Int grid_size = 1024 * 2 * 2;
    
    const Int repetitions = 20;
    
    tic("Sequential assembly of matrix");
    Sparse::MatrixCSR<Complex,Int,LInt> A
    = Sparse::GridLaplacian_Parallel<Complex,Int,LInt>(grid_size,Complex(0,1),thread_count);
    toc("Sequential assembly of matrix");
    
    tic("Parallel assembly of matrix");
    Sparse::MatrixCSR<Complex,Int,LInt> B
    = Sparse::GridLaplacian<Complex,Int,LInt>(grid_size,Complex(0,1),thread_count);
    toc("Parallel assembly of matrix");
    
    TOOLS_DUMP(A.Outer()  == B.Outer() );
    TOOLS_DUMP(A.Inner()  == B.Inner() );
    TOOLS_DUMP(A.Values() == B.Values());
    
//    tic("LoadFromMatrixMarket");
//    A.LoadFromMatrixMarket( 
//        HomeDirectory() / 
//        "Triceratops_Laplacian.mtx"
////        "Triceratops_BiLaplacian.mtx"
////        "Triceratops_BlockClusterMatrix_Far.mtx"
////        "Triceratops_BlockClusterMatrix_Near.mtx"
//        , thread_count
//    );
////    A.LoadFromMatrixMarket( HomeDirectory() / "MatrixMarket" / "bcsstk17.mtx", thread_count );
//    toc("LoadFromMatrixMarket");
    
//    tic("WriteToMatrixMarket");
//    A.WriteToMatrixMarket( HomeDirectory() / "b.mtx" );
//    toc("WriteToMatrixMarket");

    
    TOOLS_DUMP( A.RowCount() );
    TOOLS_DUMP( A.ColCount() );
    TOOLS_DUMP( A.NonzeroCount() );
    TOOLS_DUMP( A.ThreadCount() );
    
//    valprint( "A",
//        ArrayToString(
//            A.ToTensor2().data(),
//            {A.Dim(0),A.Dim(1)},
//            []( Complex x ){ return ToStringFPGeneral(x); }
//        )
//    );
//    
//    valprint( "B",
//        ArrayToString(
//            B.ToTensor2().data(),
//            {B.Dim(0),B.Dim(1)},
//            []( Complex x ){ return ToStringFPGeneral(x); }
//        )
//    );
    
    Sparse::MatrixCSR<Real,Int,LInt> Re_A(
        A.RowCount(), A.ColCount(), A.NonzeroCount(), thread_count
    );
    
    Sparse::MatrixCSR<Real,Int,LInt> Im_A(
        A.RowCount(), A.ColCount(), A.NonzeroCount(), thread_count
    );
    
    Re_A.Outer().Read( A.Outer().data() );
    Im_A.Outer().Read( A.Outer().data() );
    
    Re_A.Inner().Read( A.Inner().data() );
    Im_A.Inner().Read( A.Inner().data() );
    
    for( LInt i = 0; i < A.NonzeroCount(); ++i )
    {
        Re_A.Value(i) = Re( A.Value(i) );
        Im_A.Value(i) = Im( A.Value(i) );
    }

//    TOOLS_DUMP( A.Value(0) );
//    TOOLS_DUMP( A.Value(1) );
//    TOOLS_DUMP( A.Value(2) );
//    
//    TOOLS_DUMP( Re_A.Value(0) );
//    TOOLS_DUMP( Re_A.Value(1) );
//    TOOLS_DUMP( Re_A.Value(2) );
//    
//    TOOLS_DUMP( Im_A.Value(0) );
//    TOOLS_DUMP( Im_A.Value(1) );
//    TOOLS_DUMP( Im_A.Value(2) );
    
    
    print("");

    
//    constexpr Int NRHS =  8;
//    const     Int ldX  = 16;
//    const     Int ldY  = 16;
//    const     Int p    =  0;
    
    constexpr Int NRHS =  8;

    Tensor2<Complex,Int> X ( A.ColCount(), NRHS, 0. );
    Tensor2<Complex,Int> Y ( A.RowCount(), NRHS, 0. );
    
    Tensor2<Real,Int> Re_X ( A.ColCount(), NRHS, 0. );
    Tensor2<Real,Int> Re_Y ( A.RowCount(), NRHS, 0. );

    Tensor2<Real,Int> Im_X ( A.ColCount(), NRHS, 0. );
    Tensor2<Real,Int> Im_Y ( A.RowCount(), NRHS, 0. );
    
    X.Random();
    
    for( Int i = 0; i < A.ColCount(); ++i )
    {
        for( Int j = 0; j < NRHS; ++j )
        {
            Re_X(i,j) = Re( X(i,j) );
            Im_X(i,j) = Im( X(i,j) );
        }
    }
    
    std::string tag_0 = "Complex interleaved dot (" + ToString(repetitions) + " repetitions)";
    tic(tag_0);
    TOOLS_PTIC(tag_0);
    for( Int rep = 0; rep < repetitions; ++rep )
    {
        A.Dot<NRHS>(
            Scalar::One <Complex>, X.data(), NRHS,
            Scalar::Zero<Complex>, Y.data(), NRHS,
            NRHS
        );
    }
    TOOLS_PTOC(tag_0);
    toc(tag_0);
    
    
    std::string tag_1 = "Complex vectorized dot (" + ToString(repetitions) + " repetitions)";
    tic(tag_1);
    TOOLS_PTIC(tag_1);
    for( Int rep = 0; rep < repetitions; ++rep )
    {
        Re_A.Dot<NRHS>(
            Scalar::One <Real>, Re_X.data(), NRHS,
            Scalar::Zero<Real>, Re_Y.data(), NRHS,
            NRHS
        );
        Re_A.Dot<NRHS>(
            Scalar::One <Real>, Im_X.data(), NRHS,
            Scalar::Zero<Real>, Im_Y.data(), NRHS,
            NRHS
        );
        Im_A.Dot<NRHS>(
            -Scalar::One<Real>, Im_X.data(), NRHS,
             Scalar::One<Real>, Re_Y.data(), NRHS,
            NRHS
        );
        Im_A.Dot<NRHS>(
            Scalar::One<Real>, Re_X.data(), NRHS,
            Scalar::One<Real>, Im_Y.data(), NRHS,
            NRHS
        );
    }
    TOOLS_PTOC(tag_1);
    toc(tag_1);
    
    
    Tensor2<Complex,Int> Z ( A.RowCount(), NRHS );
    
    for( Int i = 0; i < A.RowCount(); ++i )
    {
        for( Int j = 0; j < NRHS; ++j )
        {
            Z(i,j) = Complex(Re_Y(i,j),Im_Y(i,j)) - Y(i,j);
        }
    }
    
    valprint("error", Z.MaxNorm() );
    
}

