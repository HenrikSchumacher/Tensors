#pragma once

namespace Tensors
{
    
    template<typename Int>
    static constexpr Int isqrt( const Int k )
    {
        ASSERT_INT(Int);
        
        return static_cast<Int>( std::floor(std::sqrt(static_cast<double>(k))) );
    }
    
    
    template<Size_T n, typename Int>
    static constexpr Int tri_i( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
     
        ASSERT_INT(Int);
        
        Int kk = n * (n + 1) / 2 - 1 - k;
        Int K = ( isqrt( 8 * kk + 1 ) - 1 ) / 2;
        return n-1-K;
    }
    
    template<Size_T n, typename Int>
    static constexpr Int tri_j( const Int k )
    {
        // https://stackoverflow.com/a/244550/8248900
        
        ASSERT_INT(Int);
        
        Int kk = (n * (n + 1)) / 2 - 1 - k;
        Int K = ( isqrt( 8 * kk + 1 ) - 1 ) / 2;
        return n - 1 - kk + (K * (K + 1)) /2;

    }
    
    template<Size_T n, typename Int>
    static constexpr Int lin_k( const Int i, const Int j )
    {
        // https://stackoverflow.com/a/244550/8248900
        
        return ( i <= j ) ?
            j + ( i * (2 * n - 1 - i ) ) / 2
            :
            i + ( j * (2 * n - 1 - j ) ) / 2;
    }

    
} // namespace Tensors
