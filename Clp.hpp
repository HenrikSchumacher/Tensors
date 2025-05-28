#pragma once


// We only provide some convenience routines to convert between Sparse::MatrixCSR and COIN-OR's

// To use this you need to install CLC (see https://github.com/coin-or/Clp) and link against libClp and libCoinUtils.

// I recommend using this workflow:
//
//  - git-clone the CoinUtils repository https://github.com/coin-or/CoinUtils
//  - cd into the new directory
//  - run
//          ./configure -C
//          make
//          make test
//          make install
//  - You can delete the cloned repository now.
//  - git-clone CoinUtils https://github.com/coin-or/Clp
//  - cd into the new directory
//  - run
//          ./configure -C
//          make
//          make test
//          make install
//  - You can delete the cloned repository now.
//  - On macos the header files should be located at /user/local/include/coin-or and the library files should be in /usr/local/lib. I guess the paths will be the same under Linux.
//
//
// WARNING: The homebrew installation does not seem to work.


#include "ClpSimplex.hpp"
#include "ClpSimplexDual.hpp"
//#include "CoinHelperFunctions.hpp"


namespace Tensors
{
    CoinPackedMatrix MatrixCSR_to_CoinPackedMatrix(
        const Sparse::MatrixCSR<double,int,CoinBigIndex> & A
    )
    {
        // https://coin-or.github.io/Clp/Doxygen/classClpPackedMatrix.html
        CoinPackedMatrix B(
            false, A.RowCount(), A.ColCount(), A.NonzeroCount(),
            A.Values().data(), A.Inner().data(), A.Outer().data(), nullptr
        );
        
        return B;
    }

    Sparse::MatrixCSR<double,int,CoinBigIndex> MatrixCSR_to_CoinPackedMatrix(
        const CoinPackedMatrix & A, int thread_count = 1
    )
    {
        if( A.isColOrdered() )
        {
            return Sparse::MatrixCSR<double,int,CoinBigIndex>(
                A.getVectorStarts(), A.getIndices(), A.getElements(),
                A.getNumCols(), A.getNumRows(), thread_count
            ).Transpose();
        }
        else
        {
            return Sparse::MatrixCSR<double,int,CoinBigIndex>(
                A.getVectorStarts(), A.getIndices(), A.getElements(),
                A.getNumRows(), A.getNumCols(), thread_count
            );
        }
    }
    
} // namespace Tensors
