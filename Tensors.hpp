#ifndef SCHUMACHER_TENSORS_HPP
    #define SCHUMACHER_TENSORS_HPP

    #include <iostream>
    #include <random>
    #include <cstring>
//    #include <array>
    #include <complex>

//    #include <Accelerate/Accelerate.h>
//    #include <cblas.h>
//    #include <lapacke.h>

//    #define EIGEN_NO_DEBUG
//    #define EIGEN_USE_BLAS
//    #define EIGEN_USE_LAPACKE
//    #include "eigen3/Eigen/Dense"
//
    #include "Tools/Tools.hpp"

    namespace Tensors
    {
        
        using namespace Tools;
        
        template<typename T>
        force_inline T conj( const T x )
        {
            return x;
        }
        
        template<typename T>
        force_inline T real( const T x )
        {
            return x;
        }
        
        template<typename T>
        force_inline constexpr T imag( const T x )
        {
            return static_cast<T>(0);
        }
        
        template<typename T>
        force_inline T abs_squared( const T x )
        {
            return  x * x;
        }
        
//        template<typename T>
//        force_inline T abs_squared( const std::complex<T> z )
//        {
//            return  norm(z);
//        }
        
        template<typename T>
        force_inline T abs_squared( const std::complex<T> z )
        {
            const T x = real(z);
            const T y = imag(z);
            return  x * x + y * y;
        }
        
        template <typename T>
        struct ScalarTraits
        {
            using RealType = T;
            
            static constexpr bool IsComplex = false;
        };

        template <typename T>
        struct ScalarTraits<std::complex<T>>
        {
            using RealType = T;
            
            static constexpr bool IsComplex = true;
        };

    }

    #include "src/Tensor1.hpp"
    #include "src/Tensor2.hpp"
    #include "src/Tensor3.hpp"
    #include "src/ThreadTensor3.hpp"

    #include "src/Tiny/Vector.hpp"
    #include "src/Tiny/VectorList.hpp"
    #include "src/Tiny/Matrix.hpp"
    #include "src/Tiny/SquareMatrix.hpp"
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
    #include "src/SparsityPatternCSR.hpp"
    #include "src/SparseBinaryMatrixCSR.hpp"
    #include "src/SparseBinaryMatrixVBSR.hpp"
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
    #include "src/BlockKernels/ScalarBlockKernel_fixed.hpp"

//    #include "src/BlockKernels/BlockKernel_RM.hpp"
//    #include "src/BlockKernels/DenseBlockKernel_RM.hpp"


    #include "src/LUDecomposition.hpp"
    #include "src/CholeskyDecomposition.hpp"

    #include "src/SparseCholeskyDecomposition/EliminationTree.hpp"
    #include "src/SparseCholeskyDecomposition/UniteSortedBuffers.hpp"
    #include "src/SparseCholeskyDecomposition.hpp"

#endif
