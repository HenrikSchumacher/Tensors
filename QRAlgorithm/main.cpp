
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

constexpr Int n = 4;

using E_C_Matrix_T = Eigen::Matrix<Scalar,n,n>;
using E_R_Matrix_T = Eigen::Matrix<Scalar,n,n>;

int main(int argc, const char * argv[])
{
//    const Int reps = 1000000;
    const Int reps = 1;
    
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
    Tensor3<Scalar,Int> T_list   (reps,n,n);
    
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
        
        copy_buffer<n*n>( &H_(0,0), H_list.data(rep) );
        copy_buffer<n*n>( &U_(0,0), U_list.data(rep) );
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
    
    Tiny::SquareMatrix<n,Scalar,Int> V;
    Tiny::SquareMatrix<n,Scalar,Int> W;
    
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
        
        Tiny::SquareMatrix<n,Scalar,Int> V;
        Tiny::SquareMatrix<n,Scalar,Int> W;
        
        // W = U . T_mat . UH
        Dot( T_mat, UH, V );
        Dot( U, V, W );
        
        W -= A_mat;
        
        error = std::max(error, W.FrobeniusNorm());
    }
    toc("Check  HessenbergDecomposition");
    
    dump(error);
    dump(T.ToString(4));
    dump(U.ToString(4));
    
    print("");
    print("");
    print("Test Givens rotations");
    print("");
    {
        Tiny::SquareMatrix<n,Scalar,Int> A;
        Tiny::SquareMatrix<n,Scalar,Int> Q;

        Tiny::SquareMatrix<n,Scalar,Int> AQ_true;
        Tiny::SquareMatrix<n,Scalar,Int> QA_true;
        
        std::uniform_real_distribution<Real> unif(-M_PI,M_PI);


        for( Int i = 0 ; i < n; ++i )
        {
            for( Int j = 0 ; j < n; ++j )
            {
                A[i][j] = unif(engine);
            }
        }
        
//        dump(A);
        
        Real phi = unif(engine);
        Real c = std::cos(phi);
        Real s = std::sin(phi);
        const Int i_ = 0;
        const Int j_ = 2;
        
        Q.SetGivensRotation( c, s, i_, j_ );
        
        Dot(Q,A,QA_true);
        Dot(A,Q,AQ_true);
        
        
        Tiny::SquareMatrix<n,Scalar,Int> QA (A);
        QA.GivensLeft(c, s, i_, j_);
        
        QA-=QA_true;
        
//        dump(QA);
        dump(QA.FrobeniusNorm());
        
        Tiny::SquareMatrix<n,Scalar,Int> AQ (A);
        AQ.GivensRight(c, s, i_, j_);
        
        AQ-=AQ_true;
        
//        dump(AQ);
        dump(AQ.FrobeniusNorm());
    }
    
    {
        Real c = 0;
        Real s = 0;
        
        Tiny::Vector<2,Real,Int> v;
        Tiny::Vector<2,Real,Int> w;
        
        v[0] = 0.23;
        v[1] = -.093;
        
        T.givens( v[0], v[1], c, s );
        
        dump(c);
        dump(s);
        
        
        Tiny::SquareMatrix<2, Real, Int> G;
        
        G.SetGivensRotation(c, -s, 0, 1);
        
        dump(G);
        
        Dot(G,v,w);
        
        dump(w);
        
        
    }
    print("");
    {
        Tiny::SquareMatrix<2, Real, Int> A;
        Tiny::SquareMatrix<2, Real, Int> Q;
        Tiny::SquareMatrix<2, Real, Int> QT;
        
        Tiny::SquareMatrix<2, Real, Int> QA;
        Tiny::SquareMatrix<2, Real, Int> QAQT;
        
        A[0][0] = unif(engine);
        A[1][0] = A[0][1] = unif(engine);
        A[1][1] = unif(engine);
        
        dump(A);
        
        Real c = 0;
        Real s = 0;
        
        T.givens( A[0][0], A[1][1], A[0][1], c, s );
        
        Q.SetGivensRotation(c, s, 0, 1);
        QT.SetGivensRotation(c, -s, 0, 1);
        
        Dot(Q,A,QA);
        Dot(QA,QT,QAQT);
        dump(QAQT);
        
        A.GivensLeft (c,  s, 0, 1);
        A.GivensRight(c, -s, 0, 1);
        
        dump(c);
        dump(s);
        dump(A);

        
    }
    
    print("");
    print("");
    print("Test tridiagonal QR algorithm");
    print("");
    {
        for( Int i = 0; i < n; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] = 0;
            }
        }

        A.Read( A_list.data(0) );
        dump(A);

        A.HessenbergDecomposition(U,T);

        dump(T);
        
        U.ConjugateTranspose(UH);

        Tensors::Tiny::Vector      <n,Real,Int> eigs;
        Tensors::Tiny::SquareMatrix<n,Real,Int> Q;

        T.QRAlgorithm(Q, eigs);

        dump(eigs);


    }
    
    return 0;
}
