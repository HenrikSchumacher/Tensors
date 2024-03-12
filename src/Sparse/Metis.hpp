#pragma once


extern "C" {
    #include <metis.h>
}

namespace Tensors
{
    
    template<typename I_0>
    class Metis
    {
    public:
        
        using Int  = idx_t;
        
        Metis() = default;
        
        ~Metis() = default;
        
        
        template<typename I_1, typename I_2, typename I_3, typename I_4>
        Permutation<I_0> operator()(
            mptr<I_1> rp_, mptr<I_2> ci_, const I_3 n_, const I_4 final_thread_count = 1
        )
        {
//            Int opts [METIS_NOPTIONS] = {};

//            METIS_SetDefaultOptions(&opts[0]);
            
//            opts[METIS_OPTION_NUMBERING] = 0;
            
            Int n = static_cast<Int>(n_);
            
            Tensor1<Int,Int> rp    ( static_cast<Int>( n + 1  ) );
            Tensor1<Int,Int> ci    ( static_cast<Int>( rp_[n] ) );
            Tensor1<Int,Int> perm  ( static_cast<Int>( n      ) );
            Tensor1<Int,Int> iperm ( static_cast<Int>( n      ) );
            
            Int nnz_counter = 0;
            
            rp[0] = 0;
            
            // We need to eliminate the diagonal entries.
            for( Int i = 0; i < n; ++i )
            {
                const Int k_begin = static_cast<Int>(rp_[i    ]);
                const Int k_end   = static_cast<Int>(rp_[i + 1]);
                
                for( Int k = k_begin; k < k_end; ++k )
                {
                    const Int j = static_cast<Int>(ci_[k]);
                    
                    if( i != j )
                    {
                        ci[nnz_counter] = j;
                        ++nnz_counter;
                    }
                }
                
                rp[i+1] = nnz_counter;
            }
            
            ptic("METIS_NodeND");
            (void)METIS_NodeND(
                &n, rp.data(), ci.data(), nullptr, nullptr, perm.data(), iperm.data()
            );
            ptoc("METIS_NodeND");
            
            return Permutation<I_0>( perm.data(), n, Inverse::False, I_0(1) );
        }
    }; // class Metis
    
} // namespace Tensors
