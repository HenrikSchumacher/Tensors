#pragma once

namespace Tensors
{
    template<typename T, typename I, typename T_in, typename T_out>
    class SparseBLAS
    {
        ASSERT_INT(I);

    public:
        
        static constexpr T T_one  = 1;
        static constexpr T T_zero = 0;
        static constexpr T T_two  = 2;
        
        SparseBLAS()
        {
//            ptic("SparseBLAS()");
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                thread_count = static_cast<I>(omp_get_num_threads());
            }
//            ptoc("SparseBLAS()");

        };
        
        explicit SparseBLAS( const I thread_count_ )
        : thread_count(thread_count_)
        {
//            ptic("SparseBLAS()");
//            ptoc("SparseBLAS()");
        };
        
        ~SparseBLAS() = default;
        
    protected:
        
        I thread_count = 1;
        
    protected:
        
        void scale( T_out * restrict const y, const T_out beta, const I size, const I thread_count_ )
        {
            #pragma omp parallel for simd num_threads( thread_count_ ) schedule( static )
            for( I i = 0; i < size; ++i )
            {
                y[i] *= beta;
            }
        }


#include "SparseBLAS/axpbz.hpp"
#include "SparseBLAS/axpbz_gen.hpp"
#include "SparseBLAS/azpby.hpp"
#include "SparseBLAS/azpby_gen.hpp"
//#include "SpMV.hpp"
#include "SparseBLAS/SpMM.hpp"
#include "SparseBLAS/SpMM_gen.hpp"
#include "SparseBLAS/Multiply_BinaryMatrix.hpp"
#include "SparseBLAS/Multiply_GeneralMatrix.hpp"
        
    public:
        
        static std::string ClassName()
        {
            return "SparseBLAS<"+TypeName<T>::Get()+","+TypeName<I>::Get()+","+TypeName<T_in>::Get()+","+TypeName<T_out>::Get()+">";
        }
        

    }; // SparseBLAS
    
    
} // namespace Tensors


