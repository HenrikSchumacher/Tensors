#pragma once

#include <complex>

#if defined(TENSORS_ILP64)

    #define ACCELERATE_LAPACK_ILP64

namespace Tensors
{
    namespace BLAS
    {
        
        using Int = Int64;
        
    } // namespace BLAS
}

#else

    #ifndef ACCELERATE_LAPACK_ILP64
        #undef ACCELERATE_LAPACK_ILP64
    #endif

namespace Tensors
{
    namespace LAPACK
    {
        
        using Int = Int32;
        
    } // namespace BLAS
}

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
