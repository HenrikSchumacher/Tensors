#pragma once

namespace Tensors
{
    // cf. CBLAS_TRANSPOSE
    enum class Op : unsigned char
    {
        Identity           = 111,
        Transpose          = 112,
        ConjugateTranspose = 113
//        Conjugate          = 'J'
    };
    
    // cf. CBLAS_LAYOUT
    enum class Layout : unsigned char
    {
        RowMajor = 101,
        ColMajor = 102
    };
    
    // cf. CBLAS_UPLO
    enum class Triangular : unsigned char
    {
        Upper = 121,
        Lower = 122
    };
    
    // cf. CBLAS_DIAG
    enum class Diagonal : unsigned char
    {
        Generic = 131,
        Unit    = 132
    };
    
    // cf. CBLAS_SIDE
    enum class Side : unsigned char
    {
        Left  = 141,
        Right = 142
    };
}
