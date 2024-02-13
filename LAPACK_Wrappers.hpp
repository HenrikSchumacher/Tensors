#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        
#ifndef __complex__
    using COMPLEX_DOUBLE = struct{ double real; double imag; };
    using COMPLEX_FLOAT  = struct{ float  real; float  imag; };
#else
    using COMPLEX_DOUBLE = __complex__ double;
    using COMPLEX_FLOAT  = __complex__ float;
#endif
        
        
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

#include "src/LAPACK/SelfAdjointEigensolver.hpp"
