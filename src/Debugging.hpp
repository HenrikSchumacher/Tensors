#pragma once

namespace Tensors
{
    template <typename Int1, typename Int0>
    force_inline constexpr Int1 int_cast( const Int0 n )
    {
        ASSERT_INT(Int1);
        ASSERT_INT(Int0);
#ifdef TENSORS_BOUND_CHECKS
        if( (n < std::numeric_limits<Int1>::lowest()) ||  (n > std::numeric_limits<Int1>::max()) )
        {
            eprint("int_cast reports integer overflow.");
        }
#endif
        return static_cast<Int1>(n);
    }
    
    template<typename T>
    force_inline void assert_positive( const T x )
    {
#ifdef TENSORS_BOUND_CHECKS
        if( x <= 0 )
        {
            eprint(std::string("assert_positive failed in function in ")+std::string(__FILE__)+" at line "+ ToString(__LINE__)+".");
        }
#endif
    }
//
//#ifdef TENSORS_BOUND_CHECKS
//    #define assert_positive(x,str)                                                          \
//    if( x <= 0 )                                                                            \
//    {                                                                                       \
//        eprint( "assert_positive failed for variable "+std::string(#x)+" at "+str+"." );    \
//    }                                                                                       \
//    else (void)0
//#else
//
//    #define assert_positive(x,str)
//#endif
    
} // namespace Tensors
