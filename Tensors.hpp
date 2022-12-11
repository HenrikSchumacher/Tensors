#ifndef SCHUMACHER_TENSORS_HPP
    #define SCHUMACHER_TENSORS_HPP

    #include <iostream>
    #include <random>
    #include <cstring>
//    #include <array>
    #include <complex>

    #include "Tools/Tools.hpp"

    namespace Tensors
    {
        using namespace Tools;
    }

    #include "src/Tensor1.hpp"
    #include "src/Tensor2.hpp"
    #include "src/Tensor3.hpp"
    #include "src/ThreadTensor3.hpp"

    #include "src/Tiny/Vector.hpp"
    #include "src/Tiny/VectorList.hpp"
    #include "src/Tiny/Matrix.hpp"
    #include "src/Tiny/SquareMatrix.hpp"
    #include "src/Tiny/UpperTriangularMatrix.hpp"
    #include "src/Tiny/SelfAdjointTridiagonalMatrix.hpp"
    #include "src/Tiny/SelfAdjointMatrix.hpp"
    #include "src/Tiny/MatrixList.hpp"

    #include "src/AssemblyCounters.hpp"
    #include "src/Aggregator.hpp"
    #include "src/PairAggregator.hpp"
    #include "src/TripleAggregator.hpp"

    #include "src/TwoArrayQuickSort.hpp"
    #include "src/TimSort.hpp"

    #include "src/DenseBLAS.hpp"

    #include "src/SparseBLAS.hpp"
    #include "src/Sparse/SparsePatternCSR.hpp"
    #include "src/Sparse/SparseBinaryMatrixCSR.hpp"
//    #include "src/Sparse/BinaryMatrixVBSR.hpp"
    #include "src/Sparse/SparseMatrixCSR.hpp"


    #include "src/Sparse/SparseKernelMatrixCSR.hpp"
    #include "src/Sparse/DiagonalKernelMatrix.hpp"

    #include "src/BlockKernels/BlockKernel_fixed.hpp"
    #include "src/BlockKernels/DenseBlockKernel_fixed.hpp"
    #include "src/BlockKernels/ArrowheadBlockKernel_fixed.hpp"
    #include "src/BlockKernels/ScalarBlockKernel_fixed.hpp"

//    #include "src/BlockKernels/BlockKernel_RM.hpp"
//    #include "src/BlockKernels/DenseBlockKernel_RM.hpp"


    #include "src/LUDecomposition.hpp"
    #include "src/CholeskyDecomposition.hpp"

//    #include "src/SparseCholeskyDecomposition/EliminationTree.hpp"
//    #include "src/SparseCholeskyDecomposition/UniteSortedBuffers.hpp"
//    #include "src/SparseCholeskyDecomposition.hpp"

#endif
