#pragma once

namespace Tensors
{
    template<typename T_0, typename T_1, typename LInt, int BUFFER_CAP = 128>
    class alignas(OBJECT_ALIGNMENT) PairAggregator
    {
        ASSERT_INT(LInt);

        using Container_0_T = Tensor1<T_0,LInt>;
        using Container_1_T = Tensor1<T_1,LInt>;

        mutable LInt current_size = static_cast<LInt>(0);
        mutable LInt capacity     = static_cast<LInt>(1);

        mutable LInt current_buffer_size = static_cast<LInt>(0);
        mutable std::array<T_0,BUFFER_CAP> buffer_0;
        mutable std::array<T_1,BUFFER_CAP> buffer_1;
        
        mutable Container_0_T container_0 {static_cast<LInt>(BUFFER_CAP)};
        mutable Container_1_T container_1 {static_cast<LInt>(BUFFER_CAP)};

    public:

        PairAggregator() = default;

        ~PairAggregator() = default;

        explicit PairAggregator( const LInt n )
        :   current_size ( static_cast<LInt>(0)             )
        ,   capacity     ( std::max(static_cast<LInt>(BUFFER_CAP),n) )
        ,   container_0  ( std::max(static_cast<LInt>(BUFFER_CAP),n) )
        ,   container_1  ( std::max(static_cast<LInt>(BUFFER_CAP),n) )
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



        LInt Size() const
        {
            return current_size;
        }

        void Push( const T_0 a, const T_1 b )
        {
            if( current_buffer_size >= BUFFER_CAP )
            {
                FlushBuffer();
            }

            buffer_0[current_buffer_size] = a;
            buffer_1[current_buffer_size] = b;
            ++current_buffer_size;
        }

        void Clear()
        {
            current_size        = 0;
            current_buffer_size = 0;
        }
        
        Container_0_T & Get_0()
        {
            Finalize();
            return container_0;
        }
        
        const Container_0_T & Get_0() const
        {
            Finalize();
            return container_0;
        }

        Container_1_T & Get_1()
        {
            Finalize();
            return container_1;
        }
        
        const Container_1_T & Get_1() const
        {
            Finalize();
            return container_1;
        }

    public:

        LInt Capacity() const
        {
            return capacity;
        }
        
        void RequireCapacity( const LInt new_capacity ) const
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
        
        void Finalize() const
        {
            if( current_buffer_size > 0 )
            {
                RequireCapacity( current_size + current_buffer_size );
                
                copy_buffer( buffer_0.data(), container_0.data(current_size), current_buffer_size );
                copy_buffer( buffer_1.data(), container_1.data(current_size), current_buffer_size );
                
                current_size += current_buffer_size;
                current_buffer_size = 0;
            }
        }
        
    protected:
        
        void FlushBuffer() const
        {
            if( capacity < current_size + BUFFER_CAP )
            {
                Expand();
            }
            
            copy_buffer<BUFFER_CAP>( buffer_0.data(), &container_0.data()[current_size] );
            copy_buffer<BUFFER_CAP>( buffer_1.data(), &container_1.data()[current_size] );
            
            current_size += BUFFER_CAP;
            current_buffer_size = 0;
        }
        
        void Expand() const
        {
            RequireCapacity( static_cast<LInt>(2) * capacity );
        }
    };
    
} // namespace Tensors
