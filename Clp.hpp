#pragma once


// We only provide some convenience routines to convert between Sparse::MatrixCSR and COIN-OR's

// To use this you need to install CLC (see https://github.com/coin-or/Clp) and link against libClp and libCoinUtils.
// I recommend using coinbrew:
//
//  wget https://raw.githubusercontent.com/coin-or/coinbrew/master/coinbrew
//  chmod u+x coinbrew
//  ./coinbrew fetch Clp@master
//  ./coinbrew build Clp
//
// The homebrew installation does not seem to work.

// TODO: Find out how to compile COIN-OR with 64 bit integers.
#define COIN_BIG_INDEX 1


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
