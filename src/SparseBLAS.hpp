#pragma once

namespace Tensors
{
    template<typename Scal, typename Int, typename LInt>
    class SparseBLAS
    {
        static_assert(IntQ<Int>,"");

    public:
        
        explicit SparseBLAS()
        {
//            ptic("SparseBLAS()");
//            ptoc("SparseBLAS()");
        };
        
        ~SparseBLAS() = default;
        
    private:
        
        using F_T = Scalar::Flag;
        
    protected:
        
        template<typename alpha_T, typename X_T, typename beta_T, typename Y_T>
        static constexpr void StaticParameterCheck()
        {
            static_assert(
                Scalar::ComplexQ<Y_T> || Scalar::RealQ<Scal>,
                "Template argument Y_T is real, but Scal is complex."
            );
            static_assert(
                Scalar::ComplexQ<Y_T> || Scalar::RealQ<alpha_T>,
                "Template argument Y_T is real, but alpha_T is complex."
            );
            static_assert(
                Scalar::ComplexQ<Y_T> || Scalar::RealQ<X_T>,
                "Template argument Y_T is real, but X_T is complex."
            );
            static_assert(
                Scalar::ComplexQ<Y_T> || Scalar::RealQ<beta_T>,
                "Template argument Y_T is real, but beta_T is complex."
            );
                          
            static_assert(
                Scalar::Prec<beta_T> == Scalar::Prec<Y_T>,
                "Precision of template parameter beta_T does not coincide with the one of Y_T."
            );
            static_assert(
                Scalar::Prec<alpha_T> == Scalar::Prec<Y_T>,
                "Precision of template parameter alpha_T does not coincide with the one of Y_T."
            );
        }
        
#include "SparseBLAS/SpMV.hpp"
#include "SparseBLAS/SpMV_Transposed.hpp"
#include "SparseBLAS/SpMM.hpp"
#include "SparseBLAS/Multiply_Vector.hpp"
#include "SparseBLAS/Multiply_DenseMatrix.hpp"
        
    public:
        
        static std::string ClassName()
        {
            return std::string("SparseBLAS<")+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
        }
        

    }; // SparseBLAS
    
    
} // namespace Tensors


