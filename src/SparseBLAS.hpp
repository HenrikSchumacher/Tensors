#pragma once

namespace Tensors
{
    template<typename T, typename Int, typename LInt, typename T_in, typename T_out>
    class SparseBLAS
    {
        ASSERT_INT(Int);

    public:
        
        static constexpr T T_zero = 0;
        static constexpr T T_one  = 1;
        static constexpr T T_two  = 2;
        
        SparseBLAS()
        {
//            ptic("SparseBLAS()");
            
            #pragma omp parallel
            {
                // cppcheck-suppress [useInitializationList]
                thread_count = static_cast<Int>(omp_get_num_threads());
            }
//            ptoc("SparseBLAS()");

        };
        
        explicit SparseBLAS( const Int thread_count_ )
        : thread_count(thread_count_)
        {
//            ptic("SparseBLAS()");
//            ptoc("SparseBLAS()");
        };
        
        ~SparseBLAS() = default;
        
    private:
        
        static constexpr ScalarFlag Generic = ScalarFlag::Generic;
        static constexpr ScalarFlag One     = ScalarFlag::Plus;
        static constexpr ScalarFlag Zero    = ScalarFlag::Zero;
        
    protected:
        
        Int thread_count = 1;
        
    protected:
        
        void scale( mut<T_out> y, const T_out beta, const Int size, const Int thread_count_ )
        {
            scale_buffer( beta, y, size, thread_count_ );
        }

#include "SparseBLAS/SpMV.hpp"
#include "SparseBLAS/SpMM_fixed.hpp"
#include "SparseBLAS/SpMM_gen.hpp"
#include "SparseBLAS/Multiply_Vector.hpp"
#include "SparseBLAS/Multiply_DenseMatrix.hpp"
        
    public:
        
        static std::string ClassName()
        {
            return "SparseBLAS<"+TypeName<T>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+","+TypeName<T_in>::Get()+","+TypeName<T_out>::Get()+">";
        }
        

    }; // SparseBLAS
    
    
} // namespace Tensors


