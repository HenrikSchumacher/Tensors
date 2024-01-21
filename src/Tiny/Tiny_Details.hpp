public:

#include "Tiny_Constants.hpp"

    template<class T>
    force_inline mref<Class_T> operator/=( const T lambda )
    {
        return (*this) *= ( scalar_cast<Scal>(Inv<T>(lambda)) );
    }

    inline friend std::ostream & operator<<( std::ostream & s, cref<Class_T> M )
    {
        s << ToString(M);
        return s;
    }
