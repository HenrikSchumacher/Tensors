
#include <iostream>

#define EIGEN_NO_DEBUG
//    #define EIGEN_USE_BLAS
//    #define EIGEN_USE_LAPACKE
#include "eigen3/Eigen/Dense"

#include "Tensors.hpp"

using namespace Tools;
using namespace Tensors;

//using Scalar = std::complex<double>;
using Scalar = double;
//using Scalar = std::complex<float>;
//using Scalar = float;
using Real   = ScalarTraits<Scalar>::Real;
using Int    = int32_t;

constexpr Int n = 3;

using E_C_Matrix_T = Eigen::Matrix<Scalar,n,n>;
using E_R_Matrix_T = Eigen::Matrix<Scalar,n,n>;

int main(int argc, const char * argv[])
{
    
    const Int reps = 1000000;
//    const Int reps = 10;
    
    dump(n);
    dump(reps);
    dump(TypeName<Scalar>::Get());
    //    constexpr Int p = 4;
    
    //    constexpr Scalar zero = static_cast<Scalar>(0);
    //    constexpr Scalar two  = static_cast<Scalar>(2);
    //    constexpr Scalar four = static_cast<Scalar>(4);
    
    Tensor3<Scalar,Int> A_list    (reps,n,n);
    Tensor3<Scalar,Int> U_list    (reps,n,n);
    Tensor3<Scalar,Int> H_list    (reps,n,n);
    Tensor3<Scalar,Int> T_list    (reps,n,n);
    
    Tensor2<Real,Int> eigs_list (reps,n);
    Tensor3<Real,Int> Q_list    (reps,n,n);
    
    Tiny::Vector      <n,Real,Int> eigs;
    
    Tiny::SelfAdjointMatrix<n,Scalar,Int> A;
    
    std::random_device r;
    std::default_random_engine engine ( r() );
    
    //    std::default_random_engine engine ( 1 );
    
    std::uniform_real_distribution<Real> unif(-1,1);
    
    tic("Randomize");
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
    toc("Randomize");
    
    print("");
    
    Tiny::SquareMatrix<n,Scalar,Int> U;
    Tiny::SquareMatrix<n,Scalar,Int> UH;
    Tiny::SquareMatrix<n,Scalar,Int> A_mat;
    
    Tiny::SelfAdjointTridiagonalMatrix<n,Real,Int> T;
    Tiny::SquareMatrix<n,Scalar,Int> T_mat;
    
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
        
        copy_buffer<n*n>( &H_(0,0), H_list.data(rep) );
        copy_buffer<n*n>( &U_(0,0), U_list.data(rep) );
    }
    toc("Eigen::HessenbergDecomposition");
    
    tic("Eigenvalues");
    for( Int rep = 0; rep < reps; ++rep )
    {
        A.Read( A_list.data(rep) );
        
        A.Eigenvalues(eigs);
        
        eigs.Write( eigs_list.data(rep) );
    }
    toc("Eigenvalues");
    
    tic("Eigen::HessenbergDecomposition");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        hessenberg.compute(Sigma);
        
        E_R_Matrix_T H_ ( hessenberg.matrixH().real() );
        E_C_Matrix_T U_ ( hessenberg.matrixQ() );
        
        copy_buffer<n*n>( &H_(0,0), H_list.data(rep) );
        copy_buffer<n*n>( &U_(0,0), U_list.data(rep) );
    }
    toc("Eigen::HessenbergDecomposition");
    
    tic("Eigensystem");
    for( Int rep = 0; rep < reps; ++rep )
    {
        A.Read( A_list.data(rep) );
        
        A.Eigensystem(U,eigs);
        
        eigs.Write( eigs_list.data(rep) );
        
        U.Write( U_list.data(rep) );
    }
    toc("Eigensystem");
    
    Eigen::SelfAdjointEigenSolver<E_C_Matrix_T> eigen_solver;
    tic("Eigen::SelfAdjointEigenSolver");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        eigen_solver.compute(Sigma,false);
        
        //        auto & v_ = eigen_solver.eigenvalues();
        //        auto & U_ = eigen_solver.eigenvectors();
        //
        //        copy_buffer<n  >( &v_(0)  , eigs_list.data(rep) );
        //        copy_buffer<n*n>( &U_(0,0), U_list.data(rep) );
    }
    toc("Eigen::SelfAdjointEigenSolver");
    
    tic("Eigen::SelfAdjointEigenSolver");
    for( Int rep = 0; rep < reps; ++rep )
    {
        E_C_Matrix_T Sigma ( A_list.data(rep) );
        eigen_solver.compute(Sigma,true);
        
        //        auto & v_ = eigen_solver.eigenvalues();
        //        auto & U_ = eigen_solver.eigenvectors();
        //
        //        copy_buffer<n  >( &v_(0)  , eigs_list.data(rep) );
        //        copy_buffer<n*n>( &U_(0,0), U_list.data(rep) );
    }
    toc("Eigen::SelfAdjointEigenSolver");
    
    A.Read( A_list.data(0) );
    //    dump(A);
    A.HessenbergDecomposition(U,T);
    
    A.ToMatrix(A_mat);
    T.ToMatrix(T_mat);
    
    U.ConjugateTranspose(UH);
    
    Tiny::SquareMatrix<n,Scalar,Int> V;
    Tiny::SquareMatrix<n,Scalar,Int> W;
    
    // W = U . T_mat . UH
    Dot( T_mat, UH, V );
    Dot( U, V, W );
    
    
    W -= A_mat;
    
    dump(W.FrobeniusNorm());
    
    
//    {
//        Real error = 0;
//
//        tic("Check HessenbergDecomposition");
//        for( Int rep = 0; rep < reps; ++rep )
//        {
//            A.Read( A_list.data(rep) );
//            A.HessenbergDecomposition(U,T);
//
//            A.ToMatrix(A_mat);
//            T.ToMatrix(T_mat);
//
//            U.ConjugateTranspose(UH);
//
//            Tiny::SquareMatrix<n,Scalar,Int> V;
//            Tiny::SquareMatrix<n,Scalar,Int> W;
//
//            // W = U . T_mat . UH
//            Dot( T_mat, UH, V );
//            Dot( U, V, W );
//
//            W -= A_mat;
//
//            error = std::max(error, W.FrobeniusNorm());
//        }
//        toc("Check  HessenbergDecomposition");
//
//        dump(error);
//    }
    
    {
        Tiny::SquareMatrix<n,Scalar,Int> U;
        Tiny::SquareMatrix<n,Scalar,Int> UH;
        Tiny::SquareMatrix<n,Scalar,Int> B;
        Tiny::SquareMatrix<n,Scalar,Int> C;
        Tiny::SquareMatrix<n,Real,  Int> D ( static_cast<Real>(0));
        Real error = 0;
        
        tic("Check Eigensystem");
        for( Int rep = 0; rep < reps; ++rep )
        {
            A.Read( A_list.data(rep) );
            
            auto A_mat = A.ToMatrix();
            
            A.Eigensystem(U,eigs);
            
            U.ConjugateTranspose(UH);
            D.SetDiagonal(eigs);
            
            Dot(U,D,B);
            Dot(B,UH,C);
            
            C -= A_mat;
            
            error = std::max( error, C.MaxNorm() );
            
            eigs.Write( eigs_list.data(rep) );
            
            U.Write( U_list.data(rep) );
        }
        toc("Check Eigensystem");
        
        dump(error);
    }
    
    return 0;
}
