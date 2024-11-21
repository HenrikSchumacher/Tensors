#pragma once

namespace Tensors
{
    namespace BLAS
    {
        // This namespace is to provide wrappers for some BLAS routines.
#ifdef blasint
        using Int = blasint;
#else
    #if defined(TENSORS_ILP64)
        using Int = std::int64_t;
    #else
        using Int = std::int32_t;
    #endif
#endif
        
#ifdef lapack_complex_double
        using ComplexDouble = lapack_complex_double;
#else
        using ComplexDouble = std::complex<double>;
#endif
        
#ifdef lapack_complex_float
        using ComplexFloat  = lapack_complex_float;
#else
        using ComplexFloat  = std::complex<float>;
#endif
        
        inline double * to_BLAS( double * x )
        {
            return x;
        }
        
        inline const double * to_BLAS( const double * x )
        {
            return x;
        }
        
        inline float * to_BLAS( float * x )
        {
            return x;
        }
        
        inline const float * to_BLAS( const float * x )
        {
            return x;
        }
        
        inline ComplexDouble * to_BLAS( std::complex<double> * z )
        {
            return reinterpret_cast<ComplexDouble*>(z);
        }
        
        inline const ComplexDouble * to_BLAS( const std::complex<double> * z )
        {
            return reinterpret_cast<const ComplexDouble*>(z);
        }
        
        inline ComplexFloat * to_BLAS( std::complex<float> * z )
        {
            return reinterpret_cast<ComplexFloat*>(z);
        }
        
        inline const ComplexFloat * to_BLAS( const std::complex<float> * z )
        {
            return reinterpret_cast<const ComplexFloat*>(z);
        }
        
        constexpr CBLAS_ORDER to_BLAS( Layout layout )
        {
            if ( layout == Layout::RowMajor )
            {
                return CblasRowMajor;
            }
            else // if ( layout == Layout::RowMajor )
            {
                return CblasColMajor;
            }
        }
        
        constexpr CBLAS_TRANSPOSE to_BLAS( Op op )
        {
            if ( op == Op::Id )
            {
                return CblasNoTrans;
            }
            else if ( op == Op::Trans )
            {
                return CblasTrans;
            }
            else // if ( op == Op::ConjTrans )
            {
                return CblasConjTrans;
            }
        }
        
        constexpr CBLAS_UPLO to_BLAS( UpLo uplo )
        {
            if ( uplo == UpLo::Upper )
            {
                return CblasUpper;
            }
            else // if ( uplo == UpLo::Lower )
            {
                return CblasLower;
            }
        }
        
        constexpr CBLAS_DIAG to_BLAS( Diag diag )
        {
            if ( diag == Diag::NonUnit )
            {
                return CblasNonUnit;
            }
            else // if ( diag == Diagonal::Unit )
            {
                return CblasUnit;
            }
        }
        
        constexpr CBLAS_SIDE to_BLAS( Side side )
        {
            if ( side == Side::Left )
            {
                return CblasLeft;
            }
            else // if ( side == Side::Right )
            {
                return CblasRight;
            }
        }
    }
}

#include "src/BLAS/scal.hpp"
#include "src/BLAS/dot.hpp"
#include "src/BLAS/axpy.hpp"
#include "src/BLAS/copy.hpp"

#include "src/BLAS/gemv.hpp"
#include "src/BLAS/gemm.hpp"

#include "src/BLAS/trsv.hpp"
#include "src/BLAS/trsm.hpp"

#include "src/BLAS/ger.hpp"
#include "src/BLAS/her.hpp"
#include "src/BLAS/herk.hpp"
