#pragma once

namespace Tensors
{
    template<
        Size_T N = VarSize, typename C = std::less<>,
        typename T, typename Int
    >
    void Ordering( ptr<T> key, mut<Int> perm, Int n, C comp = C() )
    {
        ASSERT_INT  (Int);
        
        if( n <= Scalar::Zero<Int> )
        {
            return;
        }
        
        if( n == Scalar::One<Int> )
        {
            perm[0] = 0;
            return;
        }
        
        iota_buffer<N,Sequential>( perm, n );
        
        Sort( perm, &perm[n],
            [=]( Int i, Int j ) -> bool
            {
                return comp( key[i], key[j] );
            }
        );
    }

}


