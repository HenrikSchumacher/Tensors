#pragma once

namespace Tensors
{
    // TODO: It could be helpful to track min and max of the three containers (at least of the first two) as these might be needed for a later assembly.
    template<typename T_0, typename T_1, typename T_2, typename LInt, int BUFFER_CAP = 128>
    class alignas(ObjectAlignment) TripleAggregator
    {
        static_assert(IntQ<LInt>,"");
        
        // LInt -- an integer type capable of storing the number of triples to aggregate.
        
        using Container_0_T = Tensor1<T_0,LInt>;
        using Container_1_T = Tensor1<T_1,LInt>;
        using Container_2_T = Tensor1<T_2,LInt>;

        mutable LInt current_size = LInt(0);
        mutable LInt capacity     = LInt(BUFFER_CAP);

        mutable LInt current_buffer_size = LInt(0);
        mutable std::array<T_0,BUFFER_CAP> buffer_0;
        mutable std::array<T_0,BUFFER_CAP> buffer_1;
        mutable std::array<T_2,BUFFER_CAP> buffer_2;
        
        mutable Container_0_T container_0 {static_cast<LInt>(BUFFER_CAP)};
        mutable Container_1_T container_1 {static_cast<LInt>(BUFFER_CAP)};
        mutable Container_2_T container_2 {static_cast<LInt>(BUFFER_CAP)};

    public:

        TripleAggregator() = default;

        ~TripleAggregator() = default;

        explicit TripleAggregator( const LInt n )
        :   current_size ( LInt(0)             )
        ,   capacity     ( Tools::Max(static_cast<LInt>(BUFFER_CAP),n) )
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



        LInt Size() const
        {
            return current_size + current_buffer_size;
        }

        template<typename Ext_T_0, typename Ext_T_1, typename Ext_T_2>
        void Push( const Ext_T_0 a, const Ext_T_1 b, const Ext_T_2 c )
        {
            if( current_buffer_size >= BUFFER_CAP )
            {
                FlushBuffer();
            }

            const Size_T cbs = static_cast<Size_T>(current_buffer_size);
            
            buffer_0[cbs] = static_cast<T_0>(a);
            buffer_1[cbs] = static_cast<T_1>(b);
            buffer_2[cbs] = static_cast<T_2>(c);
            ++current_buffer_size;
        }

        mref<Container_0_T> Get_0()
        {
            Finalize();
            return container_0;
        }

        cref<Container_0_T> Get_0() const
        {
            Finalize();
            return container_0;
        }

        mref<Container_1_T> Get_1()
        {
            Finalize();
            return container_1;
        }

        cref<Container_1_T> Get_1() const
        {
            Finalize();
            return container_1;
        }
        
        mref<Container_2_T> Get_2()
        {
            Finalize();
            return container_2;
        }

        cref<Container_2_T> Get_2() const
        {
            Finalize();
            return container_2;
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
        
        void Finalize() const
        {
            if( current_buffer_size > LInt(0) )
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
            if( capacity < current_size + BUFFER_CAP )
            {
                Expand();
            }
            
            copy_buffer( buffer_0.data(), &container_0.data()[current_size], BUFFER_CAP );
            copy_buffer( buffer_1.data(), &container_1.data()[current_size], BUFFER_CAP );
            copy_buffer( buffer_2.data(), &container_2.data()[current_size], BUFFER_CAP );
            
            current_size += BUFFER_CAP;
            current_buffer_size = 0;
        }
        
        void Expand()
        {
            RequireCapacity( LInt(2) * capacity );
        }
    };

    
} // namespace Tensors
