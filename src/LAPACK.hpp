#pragma once

namespace Tensors
{
    namespace LAPACK
    {
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

#include "LAPACK/potrf.hpp"

