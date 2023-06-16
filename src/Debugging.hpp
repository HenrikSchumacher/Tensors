#pragma once

namespace Tensors
{
    template <typename Int1, typename Int0>
    force_inline constexpr Int1 int_cast( const Int0 n )
    {
        ASSERT_INT(Int1);
        ASSERT_INT(Int0);
        
#ifdef TOOLS_DEBUG
        
        if constexpr ( !std::numeric_limits<Int1>::is_signed && std::numeric_limits<Int0>::is_signed)
        {
            if( n < static_cast<Int0>(0) )
            {
                eprint(std::string("int_cast<") + TypeName<Int1> + "> reports integer underflow n < 0 for n = " + ToString(n) + " of type " + TypeName<Int0> + ".");
            }
        }
        else
        {
            if constexpr ( std::numeric_limits<Int0>::lowest() < std::numeric_limits<Int1>::lowest() )
            {
                if( n < std::numeric_limits<Int1>::lowest() )
                {
                    eprint(std::string("int_cast<") + TypeName<Int1> + "> reports integer underflow n < " + ToString(std::numeric_limits<Int1>::lowest()) + " for n = " + ToString(n) + " of type " + TypeName<Int0> + ".");
                }
            }
        }

        if constexpr ( std::numeric_limits<Int0>::max() > std::numeric_limits<Int1>::max() )
        {
            if( n > std::numeric_limits<Int1>::max() )
            {
                eprint(std::string("int_cast<") + TypeName<Int1> + "> reports integer overflow n > " + ToString(std::numeric_limits<Int1>::max()) + "  for n = " + ToString(n) + " of type " + TypeName<Int0> + ".");
            }
        }
        
#endif
        return static_cast<Int1>(n);
    }
    
    template<typename T>
    force_inline void assert_positive( const T x )
    {
#ifdef TOOLS_DEBUG
        if constexpr ( std::numeric_limits<T>::is_signed )
        {
            if( x <= static_cast<T>(0) )
            {
                eprint(std::string("assert_positive failed in function in ") + std::string(__FILE__) + " at line "+ ToString(__LINE__)+".");
            }
        }
#endif
    }
    
} // namespace Tensors
