
#include <iostream>
#include "Tensors.hpp"

using namespace Tools;
using namespace Tensors;

// Some type aliases to make out lives a bit easier.


using Scalar = std::complex<double>;
//using Scalar = double;
using Real   = ScalarTraits<Scalar>::RealType;
using Int    = int32_t;

int main(int argc, const char * argv[])
{

    print("Hello world!");
    
    constexpr Int n = 6;
    
    constexpr Int p = 4;
    
//    constexpr Scalar zero = static_cast<Scalar>(0);
//    constexpr Scalar two  = static_cast<Scalar>(2);
//    constexpr Scalar four = static_cast<Scalar>(4);
    
    Small::SelfAdjointMatrix<n,Scalar,Int> A (0);
    
    std::random_device r;
//    std::default_random_engine engine ( r() );
    
    std::default_random_engine engine ( 1 );
    
    std::uniform_real_distribution<Real> unif(static_cast<Real>(-1),static_cast<Real>(1));
    
    for( Int i = 0; i < n; ++i )
    {
        for( Int j = i; j < n; ++j )
        {
            A[i][j] = COND( ScalarTraits<Scalar>::IsComplex,
                std::complex<Real> ( unif(engine), unif(engine) ),
                A[i][j] = unif(engine);
            );
        }
        
        A[i][i] = real(A[i][i]);
    }

    Small::SquareMatrix<n,Scalar,Int> U;
    Small::SquareMatrix<n,Scalar,Int> UH;
    Small::SquareMatrix<n,Scalar,Int> A_mat;
    
    Small::SelfAdjointTridiagonalMatrix<n,Scalar,Int> T;
    Small::SquareMatrix<n,Scalar,Int> T_mat;
    
    A.ToMatrix(A_mat);
    
    A.HessenbergDecomposition(U,T);
    
    T.ToMatrix(T_mat);

    U.ConjugateTranspose(UH);
    
    Small::SquareMatrix<n,Scalar,Int> V;
    Small::SquareMatrix<n,Scalar,Int> W;
    
    Dot( T_mat, UH, V );
    Dot( U, V, W );

    W -= A_mat;

    dump(U.ToString(p));
    dump(T_mat.ToString(p));
    dump(W.FrobeniusNorm());
    
    return 0;
}
