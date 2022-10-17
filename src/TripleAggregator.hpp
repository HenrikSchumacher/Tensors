#pragma once

namespace Tensors
{
    template<typename T_0, typename T_1, typename T_2, typename Int, int BUFFER_CAPACITY = 128>
    class alignas(OBJECT_ALIGNMENT) TripleAggregator
    {
        ASSERT_INT(Int);

        using Container_0_T = Tensor1<T_0,Int>;
        using Container_1_T = Tensor1<T_1,Int>;
        using Container_2_T = Tensor1<T_2,Int>;

        Int current_size = static_cast<Int>(0);
        Int capacity     = static_cast<Int>(1);

        Int current_buffer_size = static_cast<Int>(0);
        std::array<T_0,BUFFER_CAPACITY> buffer_0;
        std::array<T_0,BUFFER_CAPACITY> buffer_1;
        std::array<T_2,BUFFER_CAPACITY> buffer_2;
        
        Container_0_T container_0 {static_cast<Int>(BUFFER_CAPACITY)};
        Container_1_T container_1 {static_cast<Int>(BUFFER_CAPACITY)};
        Container_2_T container_2 {static_cast<Int>(BUFFER_CAPACITY)};

    public:

        TripleAggregator() = default;

        ~TripleAggregator() = default;

        TripleAggregator( const Int n )
        :   current_size ( static_cast<Int>(0)             )
        ,   capacity     ( std::max(static_cast<Int>(BUFFER_CAPACITY),n) )
        ,   container_0  ( capacity )
        ,   container_1  ( capacity )
        ,   container_2  ( capacity )
        {}

        // Copy contructor
        TripleAggregator( const TripleAggregator & other )
        :   current_size ( other.current_size              )
        ,   capacity     ( other.capacity                  )
        ,   buffer_0     ( other.buffer_0                  )
        ,   buffer_1     ( other.buffer_1                  )
        ,   buffer_2     ( other.buffer_2                  )
        ,   container_0  ( other.container_0               )
        ,   container_1  ( other.container_1               )
        ,   container_2  ( other.container_2               )
        {}

        friend void swap ( TripleAggregator & A, TripleAggregator & B ) noexcept
        {
            using std::swap;
            
            swap( A.current_size, B.current_size );
            swap( A.capacity,     B.capacity     );
            swap( A.buffer_0,     B.buffer_0     );
            swap( A.buffer_1,     B.buffer_1     );
            swap( A.buffer_2,     B.buffer_2     );
            swap( A.container_0,  B.container_0  );
            swap( A.container_1,  B.container_1  );
            swap( A.container_2,  B.container_2  );
        }

        // Move constructor
        TripleAggregator( TripleAggregator && other ) noexcept
        :   TripleAggregator()
        {
            swap(*this, other);
        }

        // Move assignment operator
        TripleAggregator & operator=( TripleAggregator && other ) noexcept
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

        void force_inline Push( const T_0 a, const T_1 b, const T_2 c )
        {
            if( current_buffer_size >= BUFFER_CAPACITY )
            {
                FlushBuffer();
            }

            buffer_0[current_buffer_size] = a;
            buffer_1[current_buffer_size] = b;
            buffer_2[current_buffer_size] = c;
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
        
        Container_2_T & Get_2()
        {
            return container_2;
        }

        const Container_2_T & Get_2() const
        {
            return container_2;
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
                Container_2_T new_container_2 (new_capacity);
                
                copy_buffer( container_0.data(), new_container_0.data(), capacity );
                copy_buffer( container_1.data(), new_container_1.data(), capacity );
                copy_buffer( container_2.data(), new_container_2.data(), capacity );
                
                using std::swap;
                swap( container_0, new_container_0 );
                swap( container_1, new_container_1 );
                swap( container_2, new_container_2 );
                
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
                copy_buffer( buffer_2.data(), &container_2.data()[current_size], current_buffer_size );
                
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
            copy_buffer( buffer_2.data(), &container_2.data()[current_size], BUFFER_CAPACITY );
            
            current_size += BUFFER_CAPACITY;
            current_buffer_size = 0;
        }
        
        void Expand()
        {
            RequireCapacity( static_cast<Int>(2) * capacity );
        }
    };

    
} // namespace Tensors
