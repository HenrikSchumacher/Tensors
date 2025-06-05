#pragma once

namespace mma
{
    template<typename T>
    constexpr bool HasTypeQ = (Tools::IntQ<T>
        || (Tools::Scalar::FloatQ<T> && Tools::Scalar::RealQ<T>)
        || (Tools::Scalar::FloatQ<T> && Tools::Scalar::ComplexQ<T>) );
    
    template< typename T, class = typename std::enable_if_t<HasTypeQ<T>>>
    using Type = std::conditional_t<
        Tools::IntQ<T>,
        mint,
        std::conditional_t<
            Tools::Scalar::FloatQ<T> && Tools::Scalar::RealQ<T>,
            mreal,
            std::complex<double>
        >
    >;
} // namespace mma
