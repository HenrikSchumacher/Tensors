public:

#include "Tiny_Constants.hpp"

    CLASS() = default;

    ~CLASS() = default;

    CLASS(std::nullptr_t) = delete;

    explicit CLASS( const Scal * a )
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
        SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
        CLASS &
    >
    operator/=( const T lambda )
    {
        return (*this) *= ( static_cast<T>(1)/lambda );
    }

    inline friend std::ostream & operator<<( std::ostream & s, cref<CLASS> M )
    {
        s << ToString(M);
        return s;
    }
