#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        constexpr int Dynamic = -1;
        
        static constexpr Op O_Id        = Op::Id;
        static constexpr Op O_Trans     = Op::Trans;
        static constexpr Op O_Conj      = Op::Conj;
        static constexpr Op O_ConjTrans = Op::ConjTrans;
        
        using Flag = Scalar::Flag;
        
        static constexpr Flag F_Plus    = Flag::Plus;
        static constexpr Flag F_Minus   = Flag::Minus;
        static constexpr Flag F_Gen     = Flag::Generic;
        static constexpr Flag F_Zero    = Flag::Zero;
    }
}

#include "src/BLAS.hpp"
#include "src/LAPACK.hpp"

//#include "src/MyBLAS/GEMM.hpp"
//#include "src/MyBLAS/TRSM.hpp"


