#pragma once

#include <amd.h>


namespace Tensors
{
    namespace CHOLMOD
    {
        template<typename Int_>
        class ApproximateMinimumDegree
        {
            
        public:
            
            static_assert(
                std::is_same_v<Int_,Int32> || std::is_same_v<Int_,Int64>,
                "ApproximateMinimumDegree supports only 32 and 64 bit signed integers."
            );
            
            using  Int = Int_;
            
            
        protected:
            
            Tiny::Vector<AMD_CONTROL,double,Int> ctrl;
            Tiny::Vector<AMD_INFO,double,Int>    info;
            
        public:
            
            ApproximateMinimumDegree() = default;
            
            ~ApproximateMinimumDegree() = default;
            
            template<typename LInt>
            int operator()( ptr<LInt> rp_, ptr<Int> ci, const Int n, mut<Int> perm )
            {
                ptic(ClassName()+"::operator()");
                Tensor1<Int,Int> rp_buffer;
                
                const Int * rp_ptr;
                
                
                int status = 0;
                
                if constexpr ( std::is_same_v<Int,LInt>  )
                {
                    rp_ptr = rp_;
                }
                else
                {
                    wprint(ClassName()+": converting row pointers.");
                    rp_buffer = Tensor1<Int,Int>(n);
                    rp_ptr = rp_buffer.data();
                }
                
                if constexpr ( std::is_same_v<Int,Int32>  )
                {
                    status = amd_order( n, rp_ptr, ci, perm, nullptr, info.data() );
                }
                else
                {
                    status = amd_l_order( n, rp_ptr, ci, perm, nullptr, info.data() );
                }
                
                //            PrintInfo();
                
                ptoc(ClassName()+"::operator()");
                
                return status;
            }
            
            template<typename LInt>
            Permutation<Int> operator()( ptr<LInt> rp, ptr<Int> ci, const Int n, const Int thread_count )
            {
                Tensor1<Int,Int> perm ( n );
                
                (void)this->operator()( rp, ci, n, perm.data() );
                
                Permutation<Int> result ( std::move(perm), Inverse::False, thread_count );
                
                return result;
            }
            
            
            
        public:
            
            void PrintInfo()
            {
                //            dump(info);
                valprint("AMD status                      ", info[AMD_STATUS]);
                valprint("Size of input matrix            ", info[AMD_N]);
                valprint("Degree of symmetry              ", info[AMD_SYMMETRY]);
                valprint("Number of diagonal entries      ", info[AMD_NZDIAG]);
                valprint("Number of nonzeroes in A + A'   ", info[AMD_NZ_A_PLUS_AT]);
                valprint("Number of dense rows            ", info[AMD_NDENSE]);
                valprint("Memory used (Bytes)             ", info[AMD_MEMORY]);
                //            valprint("Number of garbage collections   ", info[AMD_NCMPA]);
                valprint("Number of nonzeroes in L        ", info[AMD_LNZ]);
                //            valprint("Number of divide operations     ", info[AMD_NDIV]);
                //            valprint("Number of mult-subt pairs       ", info[AMD_NMULTSUBS_LU]);
                valprint("Max. no of nonzeros per row in L", info[AMD_DMAX]);
            }
            
            
            
            std::string ClassName() const
            {
                return std::string("CHOLMOD::ApproximateMinimumDegree")+"<"+ TypeName<Int> + ">";
            }
            
            
        }; // class ApproximateMinimumDegree
        
    } // namespace CHOLMOD
    
} // namespace Tensors


