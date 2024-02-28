#pragma once

#include "Base.hpp"

#if defined(TENSORS_ILP64)

    #define OPENBLAS_USE64BITINT

    #define LAPACK_ILP64

#else

    #if defined(OPENBLAS_USE64BITINT)
        #undef OPENBLAS_USE64BITINT
    #endif

    #if defined(LAPACK_ILP64)
        #undef LAPACK_ILP64
    #endif

#endif

#define LAPACK_DISABLE_NAN_CHECK

#include <cblas.h>
#include <lapack.h>

namespace Tensors
{
    constexpr bool AppleAccelerateQ = false;
    constexpr bool OpenBLASQ        = true;
}

#include "BLAS_Wrappers.hpp"
#include "LAPACK_Wrappers.hpp"
