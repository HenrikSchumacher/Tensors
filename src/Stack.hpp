#pragma once

namespace Tensors
{
    
    template<typename Entry_T_, typename Int_>
    class Stack
    {
        static_assert(IntQ<Int_>, "");
        
    public:
        
        using Entry_T     = Entry_T_;
        using Int         = Int_;
        
        Stack()
        :   a ( 1 )
        {
            a[0] = Entry_T();
        }
        
        Stack( Int max_size )
        :   a ( max_size + 1 )
        {
            a[0] = Entry_T();
        }
        
        ~Stack() = default;
        
        // Copy constructor
        Stack( const Stack & other )
        :   a   ( other.a   )
        ,   ptr ( other.ptr )
        {}
        
        inline friend void swap( Stack & A, Stack & B) noexcept
        {
            using std::swap;
            
            if( &A == &B )
            {
                wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
            }
            else
            {
                swap( A.a  , B.a   );
                swap( A.ptr, B.ptr );
            }
        }
        
        // Copy-assignment operator
        Stack & operator=( Stack other ) noexcept
        {
            swap(*this, other);
            
            return *this;
        }
        
        
        // Move constructor
        Stack( Stack && other ) noexcept
        :   Stack()
        {
            swap(*this, other);
        }
        
        
        Int Size() const
        {
            return a.Size()-1;
        }
        
        void Reset()
        {
            ptr = 0;
        }
        
        void Reset( const Int minimum_size)
        {
            a.template RequireSize<false>(minimum_size);
            ptr = 0;
        }
        
        void Push( cref<Entry_T> value )
        {
            a[++ptr] = value;
        }
        
        void Push( Entry_T && value )
        {
            a[++ptr] = std::move(value);
        }
        
        cref<Entry_T> Top() const
        {
            return a[ptr];
        }
        
        Entry_T Pop()
        {
            Entry_T r ( std::move(a[ptr--]) );
            
//            ptr = (ptr > 0) ? (ptr - 1) : 0;
            
            return r;
        }
        
        bool EmptyQ() const
        {
            return ptr <= 0;
        }
        
        Int ElementCount() const
        {
            return ptr;
        }
        
        bool HasRoom( const Int element_count ) const
        {
            return ( (ptr + element_count) < Size() );
        }
        
        std::string String() const
        {
            return ArrayToString( &a[1], {ptr} );
        }
        
        friend std::string ToString( const Stack & stack )
        {
            return stack.String();
        }
        
        Tensor1<Entry_T,Int> & GetTensor()
        {
            return a;
        }
        
    private:
        
        Tensor1<Entry_T,Int> a;
        
        Int ptr = 0;
        
    public:
        
        static std::string ClassName()
        {
            return std::string("Stack")
            + "<" + TypeName<Entry_T>
            + "," + TypeName<Int>
            + ">";
        }
        
        
    }; // class Stack
    
    
//    template<Size_T max_element_count_, typename Entry_T_, typename Int_>
//    class FixedSizePairStack
//    {
//        static_assert(IntQ<Int_>, "");
//
//    public:
//        
//        using Entry_T = Entry_T_;
//        using Int     = Int_;
//        
//        static constexpr Int max_element_count = static_cast<Int>(max_element_count_);
//        static constexpr Int actual_element_count = max_element_count + 1;
//        
////        using Pair_T      = std::array<Entry_T,2>;
//        using Pair_T      = std::pair<Entry_T,Entry_T>;
////        using Container_T = std::array<Pair_T,actual_size>;
//        
//        using Container_T = Tiny::Matrix<actual_element_count,2,Entry_T,Int>;
//        
//        
//        FixedSizePairStack()
//        {
//            a[0][0] = Entry_T();
//            a[0][1] = Entry_T();
//             
////            a[0] = { Entry_T(), Entry_T() };
//        }
//        
//        ~FixedSizePairStack() = default;
//        
//        // Copy constructor
//        FixedSizePairStack( const FixedSizePairStack & other )
////        :   a   ( other.a   )
//        :   ptr ( other.ptr )
//        {
//            std::copy<actual_element_count * 2>( &other.a[0][0], &a[0][0] );
//        }
//        
//        inline friend void swap( FixedSizePairStack & A, FixedSizePairStack & B) noexcept
//        {
//            using std::swap;
//            
//            if( &A == &B )
//            {
//                wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
//            }
//            else
//            {
////                swap( A.a  , B.a   );
//                
//                std::swap_ranges(
//                    &A.a[0][0],
//                    &A.a[0][0] + 2 * actual_element_count,
//                    &B.a[0][0]
//                );
//                swap( A.ptr, B.ptr );
//            }
//        }
//        
//        // Copy-assignment operator
//        FixedSizePairStack & operator=( FixedSizePairStack other ) noexcept
//        {
//            swap(*this, other);
//            
//            return *this;
//        }
//        
//        
//        // Move constructor
//        FixedSizePairStack( FixedSizePairStack && other ) noexcept
//        :   FixedSizePairStack()
//        {
//            swap(*this, other);
//        }
//        
//        
//        static constexpr Int Size()
//        {
//            return max_element_count;
//        }
//        
//        void Reset()
//        {
//            ptr = 0;
//        }
//        
//        inline void Push( cref<Entry_T> val_0, cref<Entry_T> val_1 )
//        {
//            ++ptr;
//            a[ptr][0] = val_0;
//            a[ptr][1] = val_1;
//        }
//
//        Pair_T Top() const
//        {
//            return { a[ptr][0], a[ptr][1] };
//        }
//        
//        inline Pair_T Pop()
//        {
//            Pair_T r { a[ptr][0], a[ptr][1] };
//            --ptr;
//            return r;
//        }
//        
//        bool EmptyQ() const
//        {
//            return ptr <= 0;
//        }
//        
//        Int ElementCount() const
//        {
//            return ptr;
//        }
//         
//        template<Int element_count>
//        bool HasRoom() const
//        {
//            static_assert( element_count <= actual_element_count, "" );
//            
//            return ( ptr < actual_element_count - element_count );
//        }
//        
//        bool HasRoom( const Int element_count ) const
//        {
//            return ( (ptr + element_count) < actual_element_count );
//        }
//        
//        std::string String() const
//        {
//            return ArrayToString( &a[1][0], {ptr,2} );
//        }
//        
//        friend std::string ToString( const FixedSizePairStack & stack )
//        {
//            return stack.String();
//        }
//        
////        cref<Container_T> GetCcontainer() const
////        {
////            return a;
////        }
//        
//    private:
//        
//        
//        Entry_T a [actual_element_count][2];
//        
//        Int ptr = 0;
//        
//    public:
//        
//        static std::string ClassName()
//        {
//            return std::string("FixedSizePairStack")
//            + "<" + TypeName<Entry_T>
//            + "," + TypeName<Int>
//            + ">";
//        }
//        
//        
//    }; // class FixedSizePairStack

} // namespace Tensors
