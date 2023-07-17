#pragma once

#include <amd.h>


namespace Tensors
{
    namespace CHOLMOD
    {
        template<typename Int_>
        class NestedDissection
        {
            
        public:
            
            static_assert(
                std::is_same_v<Int_,Int32>,
                "CHOLMOD::NestedDissection uses Metis which supports only 32 bit signed integers."
            );
            
            using  Int = Int_;
            
            
        protected:
            
            static constexpr int itype = CHOLMOD_INT;
            
        public:
            
            NestedDissection() = default;
            
            ~NestedDissection() = default;
            
            template<typename Int1, typename Int2, typename Int3, typename Int4>
            Int64 operator()( cptr<Int1> rp, cptr<Int2> ci, const Int3 n_, mptr<Int4> perm )
            {
                ptic(ClassName()+"::operator()");
                
                const Int n = int_cast<Int>(n_);
                
                cholmod_common c ;
                
                cholmod_start (&c);
                
                c.itype = itype;
                
                cholmod_sparse * A = nullptr;
                
                ptic(ClassName()+": Allocating A");
                A = cholmod_allocate_sparse(
                    static_cast<Size_T>(n_),
                    static_cast<Size_T>(n_),
                    static_cast<Size_T>(rp[n_]),
                    true, // Assume CSR-compliant sorting.
                    true, // Assume packed.
                    0,    // Assume full matrix is provided, not only a triangle.
                    CHOLMOD_SINGLE,
                    &c
                );
                
                if( A == nullptr )
                {
                    eprint(ClassName()+" failed to allocate A.");
                    return -1;
                }
                ptoc(ClassName()+": Allocating A");
                
                Tensor1<Int,Int> perm_buffer;
                Int * perm_ptr = nullptr;
                
                if constexpr( std::is_same_v<Int,Int4> )
                {
                    perm_ptr = perm;
                }
                else
                {
                    perm_buffer = Tensor1<Int,Int>( n );
                    perm_ptr    = perm_buffer.data();
                }
                
                
                ptic(ClassName()+": Copying pattern of A");
                copy_buffer( rp, reinterpret_cast<Int*>(A->p), int_cast<Size_T>(A->nrow + 1) );
                copy_buffer( ci, reinterpret_cast<Int*>(A->i), int_cast<Size_T>(A->nzmax   ) );
                ptoc(ClassName()+": Copying pattern of A");

                
                Tensor1<Int,Int> fset = iota<Int>( n );
                Tensor1<Int,Int> Cparent ( n );
                Tensor1<Int,Int> Cmember ( n );
                
                Int64 status = cholmod_nested_dissection(
                    A, fset.data(), A->nrow, perm_ptr, Cparent.data(), Cmember.data(), &c
                );
                
                if( status == CHOLMOD_NOT_INSTALLED )
                {
                    eprint(ClassName()+" nested dissection module not installed.");
                }
                
                if constexpr( std::is_same_v<Int,Int4> )
                {
                    // Do nothing.
                }
                else
                {
                    perm_buffer.Write(perm);
                }
                
                ptoc(ClassName()+"::operator()");
                
                
                return status;
            }
            
            template<typename Int_out, typename LInt1, typename Int2>
            Permutation<Int_out> operator()( cptr<LInt1> rp, cptr<Int2> ci, const Int_out n, const Int_out thread_count )
            {
                Tensor1<Int,Int> perm ( int_cast<Int>(n) );
                
                (void)this->operator()( rp, ci, n, perm.data() );
                
                Permutation<Int_out> result ( perm.data(), n, Inverse::False, thread_count );
                
                return result;
            }
            
            
            
        public:
            
//            void PrintInfo()
//            {
//                //            dump(info);
//                valprint("AMD status                      ", info[AMD_STATUS]);
//                valprint("Size of input matrix            ", info[AMD_N]);
//                valprint("Degree of symmetry              ", info[AMD_SYMMETRY]);
//                valprint("Number of diagonal entries      ", info[AMD_NZDIAG]);
//                valprint("Number of nonzeroes in A + A'   ", info[AMD_NZ_A_PLUS_AT]);
//                valprint("Number of dense rows            ", info[AMD_NDENSE]);
//                valprint("Memory used (Bytes)             ", info[AMD_MEMORY]);
//                //            valprint("Number of garbage collections   ", info[AMD_NCMPA]);
//                valprint("Number of nonzeroes in L        ", info[AMD_LNZ]);
//                //            valprint("Number of divide operations     ", info[AMD_NDIV]);
//                //            valprint("Number of mult-subt pairs       ", info[AMD_NMULTSUBS_LU]);
//                valprint("Max. no of nonzeros per row in L", info[AMD_DMAX]);
//            }
            
            
            
            std::string ClassName() const
            {
                return std::string("CHOLMOD::NestedDissection")+"<"+ TypeName<Int> + ">";
            }
            
            
        }; // class NestedDissection
        
    } // namespace CHOLMOD
    
} // namespace Tensors



