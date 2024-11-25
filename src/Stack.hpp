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
        
        void Reset( const Int minimum_size)
        {
            a.RequireSize<false>(minimum_size);
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
    
    
    template<Size_T max_size_, typename X_T_, typename y_T_, typename Int_>
    class PairStack
    {
        static_assert(IntQ<Int_>, "");
        
    public:
        
        using X_T = X_T_;
        using Y_T = y_T_;
        using Int = Int_;
        
        static constexpr Int max_size  = static_cast<Int>(max_size_);
        
        PairStack()
        {
            X[0] = X_T();
            Y[0] = Y_T();
        }
        
        PairStack( Int max_size )
//        :   a { max_size + 1, 2 }
        {
            X[0] = X_T();
            Y[0] = Y_T();
        }
        
        ~PairStack() = default;
        
        // Copy constructor
        PairStack( const PairStack & other )
        :   X   ( other.X )
        ,   Y   ( other.Y )
        ,   ptr ( other.ptr )
        {}
        
        // Move constructor
        PairStack( PairStack && other ) noexcept
        :   PairStack()
        {
            swap(*this, other);
        }
        
        inline friend void swap( PairStack & A, PairStack & B) noexcept
        {
            using std::swap;
            
            if( &A == &B )
            {
                wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
            }
            else
            {
                swap( A.X  , B.X   );
                swap( A.Y  , B.Y   );
                swap( A.ptr      , B.ptr       );
            }
        }
        
        
        force_inline Int Size() const
        {
            max_size;
        }
        
        force_inline void Reset()
        {
            ptr = 0;
        }
        
//        void Reset( const Int minimum_size)
//        {
//            a.RequireSize<false>(minimum_size);
//            ptr = 0;
//        }
        
        force_inline void Push( cref<X_T> x, cref<Y_T> y )
        {
            ++ptr;
            X[ptr] = x;
            Y[ptr] = y;
        }
        
        force_inline void Push( X_T && x, Y_T && y )
        {
            ++ptr;
            X[ptr] = std::move(x);
            Y[ptr] = std::move(y);
        }
        
        force_inline std::pair<cref<X_T>,cref<Y_T>> Top() const
        {
            return std::pair(X[ptr],Y[ptr]);
        }
        
        force_inline std::pair<X_T,Y_T> Pop()
        {
            std::pair<X_T,Y_T> r ( std::move(X[ptr]), std::move(Y[ptr]) );
            
            ptr--;
            
//            ptr = (ptr > 0) ? (ptr - 1) : 0;
            
            return r;
        }
        
        force_inline bool EmptyQ() const
        {
            return ptr <= 0;
        }
        
        force_inline Int ElementCount() const
        {
            return ptr;
        }
        
        
//        std::string String() const
//        {
//            return ArrayToString( &a[1][0], {max_size,2} );
//        }
        
//        friend std::string ToString( const PairStack & stack )
//        {
//            return stack.String();
//        }
        
//        Tensor2<Entry_T,Int> & GetTensor()
//        {
//            return a;
//        }
        
    private:
        
        std::array<X_T,max_size+1> X;
        std::array<Y_T,max_size+1> Y;
        
        Int ptr = 0;
        
    public:
        
        static std::string ClassName()
        {
            return std::string("Stack")
            + "<" + TypeName<X_T>
            + "<" + TypeName<Y_T>
            + "," + TypeName<Int>
            + ">";
        }
        
        
    }; // class Stack

} // namespace Tensors
