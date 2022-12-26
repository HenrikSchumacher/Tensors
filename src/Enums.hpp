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
        Identity  = 'N',
        Transpose = 'T',
        ConjugateTranspose = 'C'
    };
}
