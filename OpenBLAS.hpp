#pragma once

#include <complex>

#if defined(TENSORS_ILP64)

    #define OPENBLAS_USE64BITINT

    #define LAPACK_ILP64

#else

    #if defined(OPENBLAS_USE64BITINT)
        #undef OPENBLAS_USE64BITINT
    #endfif

    #if defined(LAPACK_ILP64)
        #undef LAPACK_ILP64
    #endfif

#endif

#define openblas_complex_double std::complex<double>
#define openblas_complex_float  std::complex<float>

#define LAPACK_DISABLE_NAN_CHECK
#define lapack_complex_double std::complex<double>
#define lapack_complex_float  std::complex<float>

#include <cblas.h>
#include <lapack.h>

namespace Tensors
{
    constexpr bool AppleAccelerateQ = false;
    constexpr bool OpenBLASQ        = true;
    
    namespace BLAS
    {
        
        using Int = blasint;
        
    } // namespace BLAS
    
    namespace LAPACK
    {
        
        using Int = lapack_int;
        
    } // namespace LAPACK
}

#include "BLAS_Wrappers.hpp"
#include "LAPACK_Wrappers.hpp"
