#pragma once

namespace Tensors
{
    
    template<typename Entry_T_, typename Int_>
    class Stack
    {
        static_assert(IntQ<Int_>, "");
        
    public:
        
        using Entry_T = Entry_T_;
        using Int     = Int_;
        
        Stack()
        :   a ( 1 )
        {
            a[0] = 0;
        }
        
        Stack( Int max_size )
        :   a ( max_size + 1 )
        {
            a[0] = 0;
        }
        
        ~Stack() = default;
        
        // Copy constructor
        Stack( const Stack & other )
        :   a   ( other.a   )
        ,   ptr ( other.ptr )
        {}
        
        // Move constructor
        Stack( Stack && other ) noexcept
        :   Stack()
        {
            swap(*this, other);
        }
        
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
        
        
        Int Size() const
        {
            a.Size()-1;
        }
        
        void Reset()
        {
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
        
        cptr<Entry_T> Top() const
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
            return std::string("Queue")
            + "<" + TypeName<Entry_T>
            + "," + TypeName<Int>
            + ">";
        }
        
        
    }; // class Stack
    
} // namespace Tensors
