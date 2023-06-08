#pragma once

namespace Tensors
{
    template<typename Scal, typename Int, typename LInt>
    class SparseBLAS
    {
        ASSERT_INT(Int);

    public:
        
        explicit SparseBLAS( const Int thread_count_ )
        :   thread_count(thread_count_)
        {
//            ptic("SparseBLAS()");
//            ptoc("SparseBLAS()");
        };
        
        ~SparseBLAS() = default;
        
    private:
        
        static constexpr typename Scalar::Flag Generic = Scalar::Flag::Generic;
        static constexpr typename Scalar::Flag One     = Scalar::Flag::Plus;
        static constexpr typename Scalar::Flag Zero    = Scalar::Flag::Zero;
        
    protected:
        
        Int thread_count = 1;
        
    protected:
        
        template<typename R_out, typename T_in, typename S_out, typename T_out>
        static constexpr void StaticParameterCheck()
        {
            static_assert(
                Scalar::IsComplex<T_out> || Scalar::IsReal<Scal>,
                "Template argument T_out is real, but Scalar is complex."
            );
            static_assert(
                Scalar::IsComplex<T_out> || Scalar::IsReal<R_out>,
                "Template argument T_out is real, but R_out is complex."
            );
            static_assert(
                Scalar::IsComplex<T_out> || Scalar::IsReal<T_in>,
                "Template argument T_out is real, but T_in is complex."
            );
            static_assert(
                Scalar::IsComplex<T_out> || Scalar::IsReal<S_out>,
                "Template argument T_out is real, but S_out is complex."
            );
                          
            static_assert(
                Scalar::Prec<S_out> == Scalar::Prec<T_out>,
                "Precision of template parameter S_out does not coincide with T_out's."
            );
            static_assert(
                Scalar::Prec<R_out> == Scalar::Prec<T_out>,
                "Precision of template parameter R_out does not coincide with T_out's."
            );
        }
        
#include "SparseBLAS/SpMV.hpp"
#include "SparseBLAS/SpMM_fixed.hpp"
#include "SparseBLAS/SpMM_gen.hpp"
#include "SparseBLAS/Multiply_Vector.hpp"
#include "SparseBLAS/Multiply_DenseMatrix.hpp"
        
    public:
        
        static std::string ClassName()
        {
            return std::string("SparseBLAS<")+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
        }
        

    }; // SparseBLAS
    
    
} // namespace Tensors


