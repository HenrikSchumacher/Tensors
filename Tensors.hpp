#pragma once

#include <iostream>
#include <random>
#include <cstring>
#include <array>

//#include <Accelerate/Accelerate.h>

#include "Tools/Tools.hpp"

namespace Tensors {
    
    using namespace Tools;
    
}

#include "src/Tensor1.hpp"
#include "src/Tensor2.hpp"

#include "src/Tensor3.hpp"
#include "src/ThreadTensor3.hpp"

#include "src/SmallVector.hpp"
#include "src/SmallMatrix.hpp"
#include "src/SmallVectorList.hpp"
#include "src/SmallMatrixList.hpp"

#include "src/TwoArrayQuickSort.hpp"
#include "src/TimSort.hpp"

#include "src/DenseBLAS.hpp"

#include "src/SparseBLAS.hpp"
#include "src/SparseCSR.hpp"
#include "src/SparseBinaryMatrixCSR.hpp"
#include "src/SparseBinaryMatrixVBSR.hpp"
#include "src/SparseMatrixCSR.hpp"

#include "src/SparseBlockBLAS.hpp"
#include "src/SparseMatrixBSR.hpp"
