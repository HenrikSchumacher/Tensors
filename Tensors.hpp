#ifndef SCHUMACHER_TENSORS_HPP
    #define SCHUMACHER_TENSORS_HPP

    #include <iostream>
    #include <random>
    #include <cstring>
    #include <array>
    

//    #include <Accelerate/Accelerate.h>
//    #include <cblas.h>
//    #include <lapacke.h>

//    #define EIGEN_USE_BLAS
//    #define EIGEN_USE_LAPACKE
//    #include <eigen3/Eigen/Dense>

    #include "Tools/Tools.hpp"

    namespace Tensors {
        
        using namespace Tools;
        
    }

    #include "src/Tensor1.hpp"
    #include "src/Tensor2.hpp"
    #include "src/Tensor3.hpp"
    #include "src/ThreadTensor3.hpp"

    #include "src/SmallVector.hpp"
    //#include "src/SmallMatrix.hpp"
    #include "src/SmallVectorList.hpp"
    #include "src/SmallSquareMatrix.hpp"
    //#include "src/SmallSymmetricMatrix.hpp"
    #include "src/SmallMatrixList.hpp"

    #include "src/PairAggregator.hpp"
    #include "src/TripleAggregator.hpp"

    #include "src/TwoArrayQuickSort.hpp"
    #include "src/TimSort.hpp"

    #include "src/DenseBLAS.hpp"

    #include "src/SparseBLAS.hpp"
    #include "src/SparsityPatternCSR.hpp"
    #include "src/SparseBinaryMatrixCSR.hpp"
//    #include "src/SparseBinaryMatrixVBSR.hpp"
    #include "src/SparseMatrixCSR.hpp"

//    #include "src/BlockKernels_old/BlockKernel.hpp"
//    #include "src/BlockKernels_old/SquareBlockKernel.hpp"
//    #include "src/BlockKernels_old/ScalarBlockKernel.hpp"
//    #include "src/BlockKernels_old/LowRankSquareBlockKernel.hpp"
//    #include "src/BlockKernels_old/DenseSquareBlockKernel.hpp"
//    #include "src/BlockKernels_old/DenseSquareBlockKernel_BLAS.hpp"
//    #include "src/BlockKernels_old/TP_ExperimentalBlockKernel.hpp"
//    #include "src/BlockKernels_old/DenseSquareBlockKernel_Eigen.hpp"


    #include "src/SparseKernelMatrixCSR.hpp"
    #include "src/DiagonalKernelMatrix.hpp"

    #include "src/BlockKernels/BlockKernel_fixed.hpp"
    #include "src/BlockKernels/DenseBlockKernel_fixed.hpp"
    #include "src/BlockKernels/ArrowheadBlockKernel_fixed.hpp"

    #include "src/BlockKernels/BlockKernel_RM.hpp"
    #include "src/BlockKernels/DenseBlockKernel_RM.hpp"

#endif
