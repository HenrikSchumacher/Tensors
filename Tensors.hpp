#pragma once

#include "Base.hpp"

#include "src/ThreadTensor2.hpp"
#include "src/ThreadTensor3.hpp"

//#include "src/Sorting/TwoArrayQuickSort.hpp"
#include "src/Sorting/TwoArraySort.hpp"
#include "src/Sorting/ThreeArraySort.hpp"
#include "src/Sorting/NextPermutation.hpp"
//#include "src/Sorting/MergeSort.hpp"

#include "Tiny.hpp"
#include "src/TriangleIndexing.hpp"

#include "src/AssemblyCounters.hpp"
#include "src/Aggregator.hpp"
#include "src/PairAggregator.hpp"
#include "src/TripleAggregator.hpp"


#include "src/Sparse/Permutation.hpp"
#include "src/SparseBLAS.hpp"
#include "src/Sparse/PatternCSR.hpp"
#include "src/Sparse/BinaryMatrixCSR.hpp"
#include "src/Sparse/MatrixCSR.hpp"

#include "src/Sparse/KernelMatrixCSR.hpp"
#include "src/Sparse/DiagonalKernelMatrix.hpp"

#include "src/BlockKernels/BlockKernel_fixed.hpp"
#include "src/BlockKernels/BlockKernel_fixed_2.hpp"
#include "src/BlockKernels/DenseBlockKernel_fixed.hpp"
#include "src/BlockKernels/ArrowheadBlockKernel_fixed.hpp"
#include "src/BlockKernels/ScalarBlockKernel_fixed.hpp"
