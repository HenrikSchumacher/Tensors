#pragma once

namespace Tensors
{
    template<typename T_0, typename T_1, typename Int, int BUFFER_CAPACITY = 128>
    class alignas(OBJECT_ALIGNMENT) PairAggregator
    {
        ASSERT_INT(Int);

        using Container_0_T = Tensor1<T_0,Int>;
        using Container_1_T = Tensor1<T_1,Int>;

        Int current_size = static_cast<Int>(0);
        Int capacity     = static_cast<Int>(1);

        Int current_buffer_size = static_cast<Int>(0);
        std::array<T_0,BUFFER_CAPACITY> buffer_0;
        std::array<T_0,BUFFER_CAPACITY> buffer_1;
        
        Container_0_T container_0 {static_cast<Int>(BUFFER_CAPACITY)};
        Container_1_T container_1 {static_cast<Int>(BUFFER_CAPACITY)};

    public:

        PairAggregator() = default;

        ~PairAggregator() = default;

        PairAggregator( const Int n )
        :   current_size ( static_cast<Int>(0)             )
        ,   capacity     ( std::max(static_cast<Int>(BUFFER_CAPACITY),n) )
        ,   container_0  ( std::max(static_cast<Int>(BUFFER_CAPACITY),n) )
        ,   container_1  ( std::max(static_cast<Int>(BUFFER_CAPACITY),n) )
        {}

        // Copy contructor
        PairAggregator( const PairAggregator & other )
        :   current_size ( other.current_size              )
        ,   capacity     ( other.capacity                  )
        ,   buffer_0     ( other.buffer_0                  )
        ,   buffer_1     ( other.buffer_1                  )
        ,   container_0  ( other.container_0               )
        ,   container_1  ( other.container_1               )
        {}

        friend void swap ( PairAggregator & A, PairAggregator & B ) noexcept
        {
            using std::swap;
            
            swap( A.current_size, B.current_size );
            swap( A.capacity,     B.capacity     );
            swap( A.buffer_0,     B.buffer_0     );
            swap( A.buffer_1,     B.buffer_1     );
            swap( A.container_0,  B.container_0  );
            swap( A.container_1,  B.container_1  );
        }

        // Move constructor
        PairAggregator( PairAggregator && other ) noexcept
        :   PairAggregator()
        {
            swap(*this, other);
        }

        // Move assignment operator
        PairAggregator & operator=( PairAggregator && other ) noexcept
        {
            if( this != &other )
            {
                swap( *this, other );
            }
            return *this;
        }



        Int Size() const
        {
            return current_size;
        }

        void Push( const T_0 a, const T_1 b )
        {
            if( current_buffer_size >= BUFFER_CAPACITY )
            {
                FlushBuffer();
            }

            buffer_0[current_buffer_size] = a;
            buffer_1[current_buffer_size] = b;
            ++current_buffer_size;
        }

        Container_0_T& Get_0()
        {
            return container_0;
        }

        const Container_0_T & Get_0() const
        {
            return container_0;
        }

        Container_1_T & Get_1()
        {
            return container_1;
        }

        const Container_1_T & Get_1() const
        {
            return container_1;
        }

    public:

        Int Capacity() const
        {
            return capacity;
        }
        
        void RequireCapacity( const Int new_capacity )
        {
            if( new_capacity > capacity)
            {
                Container_0_T new_container_0 (new_capacity);
                Container_1_T new_container_1 (new_capacity);
                
                copy_buffer( container_0.data(), new_container_0.data(), capacity );
                copy_buffer( container_1.data(), new_container_1.data(), capacity );
                
                using std::swap;
                swap( container_0, new_container_0 );
                swap( container_1, new_container_1 );
                
                capacity = new_capacity;
            }
        }
        
        void Finalize()
        {
            if( current_buffer_size > 0 )
            {
                RequireCapacity( current_size + current_buffer_size );
                
                copy_buffer( buffer_0.data(), &container_0.data()[current_size], current_buffer_size );
                copy_buffer( buffer_1.data(), &container_1.data()[current_size], current_buffer_size );
                
                current_size += current_buffer_size;
                current_buffer_size = 0;
            }
        }
        
    protected:
        
        void FlushBuffer()
        {
            if( capacity < current_size + BUFFER_CAPACITY )
            {
                Expand();
            }
            
            copy_buffer( buffer_0.data(), &container_0.data()[current_size], BUFFER_CAPACITY );
            copy_buffer( buffer_1.data(), &container_1.data()[current_size], BUFFER_CAPACITY );
            
            current_size += BUFFER_CAPACITY;
            current_buffer_size = 0;
        }
        
        void Expand()
        {
            RequireCapacity( static_cast<Int>(2) * capacity );
        }
    };

//    template<typename T_0, typename T_1, typename Int>
//    class alignas(OBJECT_ALIGNMENT) PairAggregator
//    {
//        ASSERT_INT(Int);
//
//        using Container_0_T = std::vector<T_0>;
//        using Container_1_T = std::vector<T_1>;
//
//        Container_0_T container_0;
//        Container_1_T container_1;
//
//    public:
//
//        PairAggregator() = default;
//
//        ~PairAggregator() = default;
//
//        PairAggregator( const Int n )
//        {
//            container_0.reserve(std::max(static_cast<Int>(1),n));
//            container_1.reserve(std::max(static_cast<Int>(1),n));
//        }
//
//        PairAggregator( const PairAggregator & other )
//        :   container_0 ( other.container_0 )
//        ,   container_1 ( other.container_1 )
//        {}
//
//        friend void swap ( PairAggregator & A, PairAggregator & B ) noexcept
//        {
//            using std::swap;
//            swap( A.container_0,  B.container_0  );
//            swap( A.container_1,  B.container_1  );
//        }
//
//        // Move constructor
//        PairAggregator( PairAggregator && other ) noexcept
//        :   PairAggregator()
//        {
//            swap(*this, other);
//        }
//
//        // Move assignment operator
//        PairAggregator & operator=( PairAggregator && other ) noexcept
//        {
//            if( this != &other )
//            {
//                swap( *this, other );
//            }
//            return *this;
//        }
//
//        Int Size() const
//        {
//            return container_0.size();
//        }
//
//        void Push( const T_0 a, const T_1 b )
//        {
//            container_0.push_back(a);
//            container_1.push_back(b);
//        }
//
//        Container_0_T& Get_0()
//        {
//            return container_0;
//        }
//
//        const Container_0_T & Get_0() const
//        {
//            return container_0;
//        }
//
//        Container_1_T & Get_1()
//        {
//            return container_1;
//        }
//
//        const Container_1_T & Get_1() const
//        {
//            return container_1;
//        }
//    };

    
} // namespace Tensors
