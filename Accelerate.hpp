#pragma once

#include "Base.hpp"

#if defined(TENSORS_ILP64)

    #define ACCELERATE_LAPACK_ILP64

    #define blasint __LAPACK_int

    #define lapack_int __LAPACK_int

#else

    #ifdef ACCELERATE_LAPACK_ILP64
        #undef ACCELERATE_LAPACK_ILP64
    #endif

    #define blasint __LAPACK_int

    #define lapack_int __LAPACK_int

#endif

#define LAPACK_DISABLE_NAN_CHECK
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace Tensors
{
    constexpr bool AppleAccelerateQ = true;
    constexpr bool OpenBLASQ        = false;
}

#include "BLAS_Wrappers.hpp"
#include "LAPACK_Wrappers.hpp"
