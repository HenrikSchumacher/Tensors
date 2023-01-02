#pragma once

namespace Tensors
{
    namespace LAPACK_Wrappers
    {
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
        
        constexpr char to_LAPACK( Triangular uplo )
        {
            if ( uplo == Triangular::Upper )
            {
                return 'U';
            }
            else // if ( uplo == Triangular::Lower )
            {
                return 'L';
            }
        }
        
    }
}

#include "LAPACK_Wrappers/potrf.hpp"

