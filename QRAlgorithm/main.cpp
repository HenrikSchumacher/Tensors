
#include <iostream>

#define EIGEN_NO_DEBUG
//    #define EIGEN_USE_BLAS
//    #define EIGEN_USE_LAPACKE
#include "eigen3/Eigen/Dense"

#include "Tensors.hpp"

using namespace Tools;
using namespace Tensors;

//using Scal = std::complex<double>;
using Scal = double;
//using Scal = std::complex<float>;
//using Scal = float;
using Real   = Scalar::Real<Scal>;
using Int    = int32_t;

constexpr Int n = 4;

using E_C_Matrix_T = Eigen::Matrix<Scal,n,n>;
using E_R_Matrix_T = Eigen::Matrix<Scal,n,n>;

int main(int argc, const char * argv[])
{
    
//    const Int reps = 1000000;
    const Int reps = 1;
    
    dump(n);
    dump(reps);
    dump(TypeName<Scal>);
    //    constexpr Int p = 4;
    
    //    constexpr Scal zero = static_cast<Scal>(0);
    //    constexpr Scal two  = static_cast<Scal>(2);
    //    constexpr Scal four = static_cast<Scal>(4);
    
    Tensor3<Scal,Int> A_list    (reps,n,n);
    Tensor3<Scal,Int> U_list    (reps,n,n);
    Tensor3<Scal,Int> H_list    (reps,n,n);
    Tensor3<Scal,Int> T_list    (reps,n,n);
    
    Tensor2<Real,Int> eigs_list (reps,n);
    Tensor3<Real,Int> Q_list    (reps,n,n);
    
    Tiny::Vector      <n,Real,Int> eigs;
    
    Tiny::SelfAdjointMatrix<n,Scal,Int> A;
    
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
                A_list(k,i,j) = (
                     Scalar::ComplexQ<Scal> ?
                     std::complex<Real> ( unif(engine), unif(engine) ) :
                     unif(engine);
                 );
            }
            A_list(k,i,i) = Re(A_list(k,i,i));
        }
    }
    toc("Randomize");
    
    print("");
    
    Tiny::Matrix<n,n,Scal,Int> U;
    Tiny::Matrix<n,n,Scal,Int> UH;
    Tiny::Matrix<n,n,Scal,Int> A_mat;
    
    Tiny::SelfAdjointTridiagonalMatrix<n,Real,Int> T;
    Tiny::Matrix<n,n,Scal,Int> T_mat;
    
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
    
    Tiny::Matrix<n,n,Scal,Int> V;
    Tiny::Matrix<n,n,Scal,Int> W;
    
    // W = U . T_mat . UH
    Dot<Overwrite>( T_mat, UH, V );
    Dot<Overwrite>( U, V, W );
    
    
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
//            Tiny::Matrix<n,n,Scal,Int> V;
//            Tiny::Matrix<n,n,Scal,Int> W;
//
//            // W = U . T_mat . UH
//            Dot<Overwrite>( T_mat, UH, V );
//            Dot<Overwrite>( U, V, W );
//
//            W -= A_mat;
//
//            error = Max(error, W.FrobeniusNorm());
//        }
//        toc("Check  HessenbergDecomposition");
//
//        dump(error);
//    }
    
    {
        Tiny::Matrix<n,n,Scal,Int> Id;
        Id.SetIdentity();
        
        Tiny::Matrix<n,n,Scal,Int> U;
        Tiny::Matrix<n,n,Scal,Int> UH;
        Tiny::Matrix<n,n,Scal,Int> B;
        Tiny::Matrix<n,n,Scal,Int> C;
        Tiny::Matrix<n,n,Real,  Int> D ( Scalar::Zero<Real> );
        Real error_0 = 0;
        Real error_1 = 0;
        Real error_2 = 0;
        
        tic("Check Eigensystem");
        for( Int rep = 0; rep < reps; ++rep )
        {
            A.Read( A_list.data(rep) );
            
            auto A_mat = A.ToMatrix();
            
            A.Eigensystem(U,eigs);
            
            U.ConjugateTranspose(UH);
            D.SetDiagonal(eigs);
            
            Dot<Overwrite>(U,D,B);
            Dot<Overwrite>(B,UH,C);
            
            C -= A_mat;
            
            error_0 = Max( error_0, C.MaxNorm() );
            
            Dot<Overwrite>(UH,A_mat,B);
            Dot<Overwrite>(B,U,C);
            C -= D;
            error_1 = Max( error_1, C.MaxNorm() );
            
            Dot<Overwrite>(U,UH,C);
            C -= Id;
            error_2 = Max( error_2, C.MaxNorm() );
            
            eigs.Write( eigs_list.data(rep) );
            
            U.Write( U_list.data(rep) );
        }
        toc("Check Eigensystem");
        
        dump(error_0);
        dump(error_1);
        dump(error_2);
        
        
        A.SetZero();
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] = KroneckerDelta<double>(i,j);
            }
        }
        A[1][1] =  0.0001;
        A[0][1] =  0.00000001;
        A[1][2] = -0.00000001;
        
//        A.Fill(1);
        dump(A);
        
        auto A_mat = A.ToMatrix();
        
        A.Eigensystem(U,eigs,0.0000000000000001);
        dump(eigs);
        
        U.ConjugateTranspose(UH);
        D.SetDiagonal(eigs);

        dump(U);
        
        Dot<Overwrite>(U,D,B);
        Dot<Overwrite>(B,UH,C);
        
        C -= A_mat;
        dump(C.MaxNorm());
        
        Dot<Overwrite>(UH,A_mat,B);
        Dot<Overwrite>(B,U,C);
        C -= D;
        dump(C.MaxNorm());
        
        auto v = C.Diagonal();
//        v /= eigs;
        dump(v);
        
        
        D.SetIdentity();
        Dot<Overwrite>(U,UH,B);
        B -= D;
    
        valprint("orthogonality error",B.MaxNorm());

    }
    
    
    return 0;
}
