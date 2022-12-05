
#include <iostream>

#define EIGEN_NO_DEBUG
//    #define EIGEN_USE_BLAS
//    #define EIGEN_USE_LAPACKE
#include "eigen3/Eigen/Dense"

#include "Tensors.hpp"



using namespace Tools;
using namespace Tensors;

// Some type aliases to make out lives a bit easier.

//HessenbergDecomposition...
//0.306573 s.
//Eigen::HessenbergDecomposition...
//0.317475 s.

using Scalar = std::complex<double>;
//using Scalar = double;
//using Scalar = std::complex<float>;
//using Scalar = float;
using Real   = ScalarTraits<Scalar>::RealType;
using Int    = int32_t;

constexpr Int n = 3;

using E_C_Matrix_T = Eigen::Matrix<Scalar,n,n>;
using E_R_Matrix_T = Eigen::Matrix<Scalar,n,n>;

int main(int argc, const char * argv[])
{

    print("Hello world!");
    
    
    
    const Int reps = 1000000;
    
    dump(n);
    dump(reps);
    dump(TypeName<Scalar>::Get());
//    constexpr Int p = 4;
    
//    constexpr Scalar zero = static_cast<Scalar>(0);
//    constexpr Scalar two  = static_cast<Scalar>(2);
//    constexpr Scalar four = static_cast<Scalar>(4);
    
    Tensor3<Scalar,Int> A_list   (reps,n,n);
    Tensor3<Scalar,Int> U_list   (reps,n,n);
    Tensor3<Scalar,Int> H_list   (reps,n,n);
    Tensor3<Scalar,Int> E_H_list (reps,n,n);
    Tensor3<Scalar,Int> T_list   (reps,n,n);
    
    Small::SelfAdjointMatrix<n,Scalar,Int> A (0);
    
    std::random_device r;
//    std::default_random_engine engine ( r() );
    
    std::default_random_engine engine ( 1 );
    
    std::uniform_real_distribution<Real> unif(static_cast<Real>(-1),static_cast<Real>(1));
    
    for( Int k = 0; k < reps; ++k )
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = i; j < n; ++j )
            {
                A_list(k,i,j) = COND(
                    ScalarTraits<Scalar>::IsComplex,
                    std::complex<Real> ( unif(engine), unif(engine) ),
                    unif(engine);
                );
            }
            A_list(k,i,i) = real(A_list(k,i,i));
        }
    }

    Small::SquareMatrix<n,Scalar,Int> U;
    Small::SquareMatrix<n,Scalar,Int> UH;
    Small::SquareMatrix<n,Scalar,Int> A_mat;
    
    Small::SelfAdjointTridiagonalMatrix<n,Real,Int> T;
    Small::SquareMatrix<n,Scalar,Int> T_mat;
    
    tic("HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        A.Read( A_list.data(rep) );
        A.HessenbergDecomposition(U,T);

        U.Write( U_list.data(rep) );
        T.ToMatrix( T_mat );
        T_mat.Write( H_list.data(rep) );
    }
    toc("HessenbergDecomposition");

    Eigen::HessenbergDecomposition<E_C_Matrix_T> hessenberg;
    
    tic("Eigen::HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        hessenberg.compute(Sigma);
        
        E_R_Matrix_T H_ ( hessenberg.matrixH().real() );
        E_C_Matrix_T U_ ( hessenberg.matrixQ() );
        
        copy_buffer( &H_(0,0), H_list.data(rep), n*n );
        copy_buffer( &U_(0,0), U_list.data(rep), n*n );
    }
    toc("Eigen::HessenbergDecomposition");
    
    tic("HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        A.Read( A_list.data(rep) );
        A.HessenbergDecomposition(U,T);

        U.Write( U_list.data(rep) );
        T.ToMatrix( T_mat );
        T_mat.Write( H_list.data(rep) );
    }
    toc("HessenbergDecomposition");
    
    tic("Eigen::HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        hessenberg.compute(Sigma);
        
        E_R_Matrix_T H_ ( hessenberg.matrixH().real() );
        E_C_Matrix_T U_ ( hessenberg.matrixQ() );
        
        copy_buffer( &H_(0,0), H_list.data(rep), n*n );
        copy_buffer( &U_(0,0), U_list.data(rep), n*n );
    }
    toc("Eigen::HessenbergDecomposition");
    
    
    Eigen::SelfAdjointEigenSolver<E_C_Matrix_T> eigs;
    tic("Eigen::SelfAdjointEigenSolver");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        eigs.compute(Sigma);
    }
    toc("Eigen::SelfAdjointEigenSolver");

    

        
    A.Read( A_list.data(0) );
//    dump(A);
    A.HessenbergDecomposition(U,T);
    
    A.ToMatrix(A_mat);
    T.ToMatrix(T_mat);
    
    U.ConjugateTranspose(UH);
    
    Small::SquareMatrix<n,Scalar,Int> V;
    Small::SquareMatrix<n,Scalar,Int> W;
    
    // W = U . T_mat . UH
    Dot( T_mat, UH, V );
    Dot( U, V, W );
    
    
    W -= A_mat;
    
    dump(W.FrobeniusNorm());
    
    
    Real error = 0;
    
    tic("Check HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        A.Read( A_list.data(rep) );
        A.HessenbergDecomposition(U,T);
        
        A.ToMatrix(A_mat);
        T.ToMatrix(T_mat);
        
        U.ConjugateTranspose(UH);
        
        Small::SquareMatrix<n,Scalar,Int> V;
        Small::SquareMatrix<n,Scalar,Int> W;
        
        // W = U . T_mat . UH
        Dot( T_mat, UH, V );
        Dot( U, V, W );
        
        W -= A_mat;
        
        error = std::max(error, W.FrobeniusNorm());
    }
    toc("Check  HessenbergDecomposition");
    
    dump(error);
//    dump(T.ToString(4));
//    dump(U.ToString(4));
    
    return 0;
}
