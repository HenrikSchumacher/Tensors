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

    #include "src/Debugging.hpp"
    #include "src/Enums.hpp"

    #include "src/Tensor1.hpp"
    #include "src/Tensor2.hpp"
    #include "src/Tensor3.hpp"
    #include "src/ThreadTensor2.hpp"
    #include "src/ThreadTensor3.hpp"

    #include "src/Sorting/TwoArrayQuickSort.hpp"
//    #include "src/Sorting/TimSort.hpp"

    #include "src/Permutation.hpp"

    #include "Tiny.hpp"

    #include "src/AssemblyCounters.hpp"
    #include "src/Aggregator.hpp"
    #include "src/PairAggregator.hpp"
    #include "src/TripleAggregator.hpp"



    #include "src/SparseBLAS.hpp"
    #include "src/Sparse/PatternCSR.hpp"
    #include "src/Sparse/BinaryMatrixCSR.hpp"
//    #include "src/Sparse/BinaryMatrixVBSR.hpp"
    #include "src/Sparse/MatrixCSR.hpp"


    #include "src/Sparse/KernelMatrixCSR.hpp"
    #include "src/Sparse/DiagonalKernelMatrix.hpp"

    #include "src/BlockKernels/BlockKernel_fixed.hpp"
    #include "src/BlockKernels/DenseBlockKernel_fixed.hpp"
    #include "src/BlockKernels/ArrowheadBlockKernel_fixed.hpp"
    #include "src/BlockKernels/ScalarBlockKernel_fixed.hpp"

//    #include "src/BlockKernels/BlockKernel_RM.hpp"
//    #include "src/BlockKernels/DenseBlockKernel_RM.hpp"


#endif
