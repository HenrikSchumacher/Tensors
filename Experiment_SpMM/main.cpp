
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


int main(int argc, const char * argv[])
{
    const Int thread_count = 8;
    
    Sparse::MatrixCSR<Complex,Int,Size_T> A;
    
    
    tic("LoadFromMatrixMarket");
    A.LoadFromMatrixMarket( 
        HomeDirectory() / 
        "Triceratops_Laplacian.mtx"
//        "Triceratops_BiLaplacian.mtx"
//        "Triceratops_BlockClusterMatrix_Far.mtx"
//        "Triceratops_BlockClusterMatrix_Near.mtx"
        , thread_count
    );
//    A.LoadFromMatrixMarket( HomeDirectory() / "MatrixMarket" / "bcsstk17.mtx", thread_count );
    toc("LoadFromMatrixMarket");
    
//    tic("WriteToMatrixMarket");
//    A.WriteToMatrixMarket( HomeDirectory() / "b.mtx" );
//    toc("WriteToMatrixMarket");
    
    Profiler::Clear();
    
    dump( A.RowCount() );
    dump( A.ColCount() );
    
    Sparse::MatrixCSR<Real,Int,Size_T> Re_A(
        A.RowCount(), A.ColCount(), A.NonzeroCount(), thread_count
    );
    
    Sparse::MatrixCSR<Real,Int,Size_T> Im_A(
        A.RowCount(), A.ColCount(), A.NonzeroCount(), thread_count
    );
    
    Re_A.Outer().Read( A.Outer().data() );
    Im_A.Outer().Read( A.Outer().data() );
    
    Re_A.Inner().Read( A.Inner().data() );
    Im_A.Inner().Read( A.Inner().data() );
    
    for( Size_T i = 0; i < A.NonzeroCount(); ++i )
    {
        Re_A.Value(i) = Re( A.Value(i) );
        Im_A.Value(i) = Im( A.Value(i) );
    }
    
    dump( Re_A.Value(0) );
    dump( Re_A.Value(1) );
    dump( Re_A.Value(2) );
    
    dump( Im_A.Value(0) );
    dump( Im_A.Value(1) );
    dump( Im_A.Value(2) );
    
    
    print("");

    
//    constexpr Int NRHS =  8;
//    const     Int ldX  = 16;
//    const     Int ldY  = 16;
//    const     Int p    =  0;
    
    constexpr Int NRHS =  8;

    
    Tensor2<Complex,Int> X ( A.RowCount(), NRHS, 0. );
    Tensor2<Complex,Int> Y ( A.RowCount(), NRHS, 0. );
    Tensor2<Complex,Int> Z ( A.RowCount(), NRHS, 0. );
    
    Tensor2<Real,Int> Re_X ( A.RowCount(), NRHS, 0. );
    Tensor2<Real,Int> Re_Y ( A.RowCount(), NRHS, 0. );
    Tensor2<Real,Int> Re_Z ( A.RowCount(), NRHS, 0. );
    
    Tensor2<Real,Int> Im_X ( A.RowCount(), NRHS, 0. );
    Tensor2<Real,Int> Im_Y ( A.RowCount(), NRHS, 0. );
    Tensor2<Real,Int> Im_Z ( A.RowCount(), NRHS, 0. );
    
    X.Random();
    
    for( Int i = 0; i < A.RowCount(); ++i )
    {
        for( Int j = 0; j < NRHS; ++j )
        {
            Re_X(i,j) = Re( X(i,j) );
            Im_X(i,j) = Im( X(i,j) );
        }
    }
    
    dump( Re_X[0][0] );
    dump( Re_X[0][1] );
    dump( Re_X[0][2] );
    
    dump( Im_X[0][0] );
    dump( Im_X[0][1] );
    dump( Im_X[0][2] );
    
    
    tic("Complex interleaved dot");
    ptic("Complex interleaved dot");
    A.Dot<NRHS>(
        Scalar::One <Complex>, X.data(), NRHS,
        Scalar::Zero<Complex>, Y.data(), NRHS,
        NRHS
    );
    ptoc("Complex interleaved dot");
    toc("Complex interleaved dot");
    
    
    
    tic("Complex vectorized dot");
    ptic("Complex vectorized dot");
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
//    
//    dump( Re_Y[0][0] );
//    dump( Re_Y[0][1] );
//    dump( Re_Y[0][2] );
//    
//    dump( Im_Y[0][0] );
//    dump( Im_Y[0][1] );
//    dump( Im_Y[0][2] );
//    
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
    ptoc("Complex vectorized dot");
    toc("Complex vectorized dot");

    
    dump( Y[0][0] );
    dump( Y[0][1] );
    dump( Y[0][2] );
    
    dump( Complex( Re_Y[0][0], Im_Y[0][0] ) );
    dump( Complex( Re_Y[0][1], Im_Y[0][1] ) );
    dump( Complex( Re_Y[0][2], Im_Y[0][2] ) );
    
}

