#pragma once

#include <scotch.h>

namespace Tensors
{

    class Scotch
    {
        
    public:
        
        Scotch() = default;
        
        ~Scotch() = default;
        
        
        template<typename ExtInt1, typename ExtInt2, typename Int>
        void operator()(
            mptr<ExtInt1> rp, mptr<ExtInt2> ci, mref<Permutation<Int>> perm
        )
        {
            print( TypeName<idx_t> );
            
            idx_t opts [METIS_NOPTIONS] = {};

            METIS_SetDefaultOptions(&opts[0]);
            
            opts[METIS_OPTION_NUMBERING] = 0;
            
            idx_t num_flag = 0;
            
            idx_t n = perm.Size();
            
            Tensor1<idx_t,idx_t> rp_buffer;
            Tensor1<idx_t,idx_t> ci_buffer;
            Tensor1<idx_t,idx_t> perm_buffer;
            Tensor1<idx_t,idx_t> iperm_buffer;
            
            idx_t * rp_ptr    = nullptr;
            idx_t * ci_ptr    = nullptr;
            idx_t * perm_ptr  = nullptr;
            idx_t * iperm_ptr = nullptr;
            
            if constexpr ( !SameQ<idx_t,ExtInt1> )
            {
                rp_buffer = Tensor1<idx_t,idx_t>( rp, n + 1 );
                rp_ptr    = rp_buffer.data();
            }
            else
            {
                rp_ptr = rp;
            }
            
            if constexpr ( !SameQ<idx_t,ExtInt2>)
            {
                ci_buffer = Tensor1<idx_t,idx_t>( ci, rp[n] );
                ci_ptr    = ci_buffer.data();
            }
            else
            {
                ci_ptr    = ci;
            }
            
            if constexpr ( !SameQ<idx_t,Int>)
            {
                perm_buffer  = iota<idx_t>( n );
                iperm_buffer = iota<idx_t>( n );
                perm_ptr  = perm_buffer.data();
                iperm_ptr = iperm_buffer.data();
            }
            else
            {
                perm_ptr  = perm.GetPermutation().data();
                iperm_ptr = perm.GetInversePermutation().data();
            }
            
            
            ptic("METIS_NodeND");
            METIS_NodeND(&n, rp_ptr, ci_ptr, &num_flag, &opts[0], perm_ptr, iperm_ptr );
            ptic("METIS_NodeND");
            
            if constexpr ( !SameQ<idx_t,Int> )
            {
                 perm_buffer.Write( perm.GetPermutation().data() );
                iperm_buffer.Write( perm.GetInversePermutation().data() );
            }
        }
    }; // class Scotch

    
} // namespace Tensors
