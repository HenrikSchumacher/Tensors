#pragma once

namespace Tensors
{
    namespace BLAS
    {
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



#include "BLAS/gemv.hpp"
#include "BLAS/gemm.hpp"

#include "BLAS/trsv.hpp"
#include "BLAS/trsm.hpp"

#include "BLAS/her.hpp"
#include "BLAS/herk.hpp"
