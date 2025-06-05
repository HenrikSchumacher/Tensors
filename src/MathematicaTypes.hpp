#pragma once

namespace mma
{
    template<typename T>
    bool HasTypeQ = Tools::IntQ<T>
                  || (Tools::FloatQ<T> && Tools::Scalar::RealQ<T>)
                  || (Tools::FloatQ<T> && Tools::Scalar::ComplexQ<T>);
    
    template< typename T, class = typename std::enable_if_t<HasTypeQ<T>>>
    using Type = std::conditional_t<
        Tools::IntQ<T>,
        mint,
        std::conditional_t<
            Tools::FloatQ<T> && Tools::Scalar::RealQ<T>,
            mreal,
            mcomplex
        >
    >;
} // namespace mma
