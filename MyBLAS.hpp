#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        constexpr int Dynamic = -1;
    }
}

//#include "Tiny.hpp"
#include "src/BLAS_Wrappers.hpp"

#include "src/MyBLAS/GEMM.hpp"
#include "src/MyBLAS/TRSM.hpp"
