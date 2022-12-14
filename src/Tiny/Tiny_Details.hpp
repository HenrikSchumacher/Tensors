public:

#include "Tiny_Constants.hpp"

    CLASS() = default;

    ~CLASS() = default;

    CLASS(std::nullptr_t) = delete;

    explicit CLASS( const Scalar * a )
    {
        Read(a);
    }

    // Copy assignment operator
    CLASS & operator=( CLASS other )
    {
        // copy-and-swap idiom
        // see https://stackoverflow.com/a/3279550/8248900 for details
        swap(*this, other);

        return *this;
    }

    /* Move constructor */
    CLASS( CLASS && other ) noexcept
    :   CLASS()
    {
        swap(*this, other);
    }

    template<class T>
    force_inline 
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator/=( const T lambda )
    {
        return (*this) *= ( static_cast<T>(1)/lambda );
    }

    inline friend std::ostream & operator<<( std::ostream & s, const CLASS & M )
    {
        s << M.ToString();
        return s;
    }
