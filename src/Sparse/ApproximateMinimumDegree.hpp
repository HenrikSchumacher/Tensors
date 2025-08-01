#pragma once

#include <amd.h>

#define TENSORS_HAS_AMD

namespace Tensors
{
    namespace Sparse
    {
        template<typename Int, bool base = 0>
        class ApproximateMinimumDegree final
        {
        public:
            
            // Default constuctor
            ApproximateMinimumDegree() = default;
            // Destructor
            ~ApproximateMinimumDegree() = default;
            // Copy constructor
            ApproximateMinimumDegree( const ApproximateMinimumDegree & other ) = default;
            // Copy assignment operator
            ApproximateMinimumDegree & operator=( const ApproximateMinimumDegree & other ) = default;
            // Move constructor
            ApproximateMinimumDegree( ApproximateMinimumDegree && other ) = default;
            // Move assignment operator
            ApproximateMinimumDegree & operator=( ApproximateMinimumDegree && other ) = default;
                        
            template<typename I_1, typename I_2, typename I_3, typename I_4>
            Permutation<Int> operator()(
                mptr<I_1> rp_, mptr<I_2> ci_, const I_3 n_, const I_4 final_thread_count = 1
            )
            {
                TOOLS_PTIC(ClassName()+": Preprocessing");
                const Int64 n = n_;
                
                Tensor1<Int64,Int64> rp ( rp_, n + 1  );
                Tensor1<Int64,Int64> ci ( rp_[n] );
                
                for( Int64 k = 0; k < rp[n]; ++k )
                {
                    ci[k] = ci_[k] - base;
                }
                
                Tensor1<Int64,Int64> perm = iota<Int64,Int64>(n);

                TOOLS_PTOC(ClassName()+": Preprocessing");
                
                TOOLS_PTIC(ClassName()+": amd_l_order");
                
//                Tiny::Vector<AMD_CONTROL,double,Int64> control;
//                Tiny::Vector<AMD_INFO   ,double,Int64> info;
                
//                int status = amd_l_order(
//                    n, rp.data(), ci.data(), perm.data(), &control[0], &info[0]
//                );
                
                int status = amd_l_order(
                    n, rp.data(), ci.data(), perm.data(), nullptr, nullptr
                );
                
                if( status == AMD_OUT_OF_MEMORY )
                {
                    eprint("ApproximateMinimumDegree: Out of memory.");
                }
                else if( status == AMD_INVALID )
                {
                    eprint("ApproximateMinimumDegree: Inputs are invalid.");
                }
                else if( status == AMD_OK_BUT_JUMBLED )
                {
                    wprint("ApproximateMinimumDegree: Inputs have unordered column indices, but they are otherwise correct.");
                }
                
                TOOLS_PTOC(ClassName()+": amd_l_order");
                
                return Permutation<Int>( perm.data(), static_cast<Int>(n), Inverse::False, Int(final_thread_count) );
            }
            
        public:
            
            static std::string ClassName()
            {
                return std::string("ApproximateMinimumDegree")+ "<" + TypeName<Int> + "," + ToString(base) + ">";
            }
            
        }; // class ApproximateMinimumDegree
        
    } // namespace Sparse
    
} // namespace Tensors

