
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
    
    constexpr Int n = 4;
    
    constexpr Int p = 16;
    
//    constexpr Scalar zero = static_cast<Scalar>(0);
//    constexpr Scalar two  = static_cast<Scalar>(2);
//    constexpr Scalar four = static_cast<Scalar>(4);
    
    Small::SelfAdjointMatrix<n,Scalar,Int> A (0);
    
    std::random_device r;
//    std::default_random_engine engine ( r() );
    
    std::default_random_engine engine ( 1 );
    
    std::uniform_real_distribution<Real> unif(static_cast<Real>(-1),static_cast<Real>(1));
    
    dump(ScalarTraits<Scalar>::IsComplex);
    
    for( Int i = 0; i < n; ++i )
    {
        for( Int j = i; j < n; ++j )
        {
            A[i][j] =  COND( ScalarTraits<Scalar>::IsComplex,
                std::complex<Real> ( unif(engine), unif(engine) ),
                A[i][j] = unif(engine);
            );
        }
    }

    
    Small::SquareMatrix                <n, Scalar, Int> U;
    Small::SelfAdjointTridiagonalMatrix<n, Scalar, Int> T;
    A.Tridiagonalize(U,T);

    
    return 0;
}
