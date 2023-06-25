#pragma once

namespace Tensors
{
    namespace MyBLAS
    {
        constexpr int Dynamic = -1;
    }
}

//#include <complex>
//#undef  lapack_complex_float
//#define lapack_complex_float  std::complex<float>
//#undef  lapack_complex_double
//#define lapack_complex_double std::complex<double>

//#include "Tiny.hpp"
#include "src/BLAS.hpp"
#include "src/LAPACK.hpp"

#include "src/MyBLAS/GEMM.hpp"
#include "src/MyBLAS/TRSM.hpp"


