#pragma once

namespace Tensors
{
    enum class Triangular : bool
    {
        Upper = true,
        Lower = false
    };
    
    enum class Side : bool
    {
        Left  = true,
        Right = false
    };
    
    enum class Layout : bool
    {
        RowMajor  = true,
        ColMajor = false
    };
    
    enum class Op : char
    {
        Identity           = 111,
        Transpose          = 112,
        ConjugateTranspose = 113
//        Conjugate          = 'J'
    };
}
