#pragma once

extern "C" {
    #include <metis.h>
}

namespace Tensors
{
    class Metis
    {
        
    public:
        
        Metis() = default;
        
        ~Metis() = default;
        
        
        template<typename ExtInt1, typename ExtInt2, typename Int>
        void operator()(
            mptr<ExtInt1> rp, mptr<ExtInt2> ci, Permutation<Int> & perm
        )
        {
            idx_t opts [METIS_NOPTIONS] = {};

            METIS_SetDefaultOptions(&opts[0]);
            
            opts[METIS_OPTION_NUMBERING] = 0;
            
//            opts[METIS_OPTION_SEED]      = std::random_device()();
            opts[METIS_OPTION_SEED]      = 0;
            
//            opts[METIS_OPTION_CCORDER]   = 0;
            
//            opts[METIS_OPTION_DBGLVL]    = 256;
            
            idx_t n = perm.Size();
            
            Tensor1<idx_t,idx_t> rp_buffer;
            Tensor1<idx_t,idx_t> ci_buffer;
            Tensor1<idx_t,idx_t> perm_buffer;
            Tensor1<idx_t,idx_t> iperm_buffer;
            
            idx_t * rp_ptr    = nullptr;
            idx_t * ci_ptr    = nullptr;
            idx_t * perm_ptr  = nullptr;
            idx_t * iperm_ptr = nullptr;
            
            if constexpr ( !std::is_same_v<idx_t,ExtInt1> )
            {
                rp_buffer = Tensor1<idx_t,idx_t>( n + 1 );
                rp_buffer.Read(rp);
                rp_ptr    = rp_buffer.data();
            }
            else
            {
                rp_ptr = rp;
            }
            
            if constexpr ( !std::is_same_v<idx_t,ExtInt2>)
            {
                ci_buffer = Tensor1<idx_t,idx_t>( rp[n] );
                ci_buffer.Read(ci);
                ci_ptr    = ci_buffer.data();
            }
            else
            {
                ci_ptr    = ci;
            }
            
            if constexpr ( !std::is_same_v<idx_t,Int>)
            {
                perm_buffer  = Tensor1<idx_t,idx_t>( n );
                iperm_buffer = Tensor1<idx_t,idx_t>( n );
                 perm_ptr = perm_buffer.data();
                iperm_ptr = iperm_buffer.data();
            }
            else
            {
                 perm_ptr = perm.GetPermutation().data();
                iperm_ptr = perm.GetInversePermutation().data();
            }
            
//            dump( rp_ptr );
//            dump( ci_ptr );
//            dump( perm_ptr );
//            dump( iperm_ptr );
            
            
            ptic("METIS_NodeND");
            METIS_NodeND(&n, rp_ptr, ci_ptr, nullptr, &opts[0], perm_ptr, iperm_ptr );
            ptoc("METIS_NodeND");
            
            if constexpr ( !std::is_same_v<idx_t,Int> )
            {
                 perm_buffer.Write( perm.GetPermutation().data() );
                iperm_buffer.Write( perm.GetInversePermutation().data() );
            }
        }
    }; // class Metis
    
} // namespace Tensors
