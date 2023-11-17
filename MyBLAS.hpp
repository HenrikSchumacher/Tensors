#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        constexpr int Dynamic = -1;
    }
}

#include "src/BLAS.hpp"
#include "src/LAPACK.hpp"

#include "src/MyBLAS/GEMM.hpp"
#include "src/MyBLAS/TRSM.hpp"


