#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        using Int = int;
        
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

        inline double * to_LAPACK( double * x )
        {
            return x;
        }
        
        inline const double * to_LAPACK( const double * x )
        {
            return x;
        }
        
        inline float * to_LAPACK( float * x )
        {
            return x;
        }
        
        inline const float * to_LAPACK( const float * x )
        {
            return x;
        }
        
        inline ComplexDouble * to_LAPACK( std::complex<double> * z )
        {
            return reinterpret_cast<ComplexDouble*>(z);
        }
        
        inline const ComplexDouble * to_LAPACK( const std::complex<double> * z )
        {
            return reinterpret_cast<const ComplexDouble*>(z);
        }
        
        inline ComplexFloat * to_LAPACK( std::complex<float> * z )
        {
            return reinterpret_cast<ComplexFloat*>(z);
        }
        
        inline const ComplexFloat * to_LAPACK( const std::complex<float> * z )
        {
            return reinterpret_cast<const ComplexFloat*>(z);
        }
        
        
        
//#ifndef __complex__
//    using COMPLEX_DOUBLE = struct{ double real; double imag; };
//    using COMPLEX_FLOAT  = struct{ float  real; float  imag; };
//#else
//    using COMPLEX_DOUBLE = __complex__ double;
//    using COMPLEX_FLOAT  = __complex__ float;
//#endif
        
        // This namespace is to provide wrappers for some LAPACK routines.
        
        constexpr int to_LAPACK( Layout layout )
        {
            if ( layout == Layout::RowMajor )
            {
                return 101; // LAPACK_ROW_MAJOR
            }
            else // if ( layout == Layout::RowMajor )
            {
                return 102; // LAPACK_COL_MAJOR
            }
        }
        
        constexpr char to_LAPACK( Op op )
        {
            if ( op == Op::Id )
            {
                return 'N';
            }
            else if ( op == Op::Trans )
            {
                return 'T';
            }
            else // if ( op == Op::ConjTranspose )
            {
                return 'C';
            }
        }
        
        constexpr char to_LAPACK( UpLo uplo )
        {
            if ( uplo == UpLo::Upper )
            {
                return 'U';
            }
            else // if ( uplo == UpLo::Lower )
            {
                return 'L';
            }
        }
        
    }
}

#include "src/LAPACK/potrf.hpp"
#include "src/LAPACK/getrf.hpp"

#include "src/LAPACK/SelfAdjointEigensolver.hpp"

