#pragma once

namespace Tensors
{
    template<typename T_0, typename LInt>
    class alignas(OBJECT_ALIGNMENT) Aggregator
    {
        // A dynamically growing version of Tensor1 that allows pushing of several elements at once.
        
        ASSERT_INT(LInt);

        using Container_0_T = Tensor1<T_0,LInt>;

        LInt current_size = static_cast<LInt>(0);
        LInt capacity     = static_cast<LInt>(1);
        
        Container_0_T container_0 {capacity};

    public:

        Aggregator() = default;

        ~Aggregator() = default;

        explicit Aggregator( const LInt n )
        :   current_size ( static_cast<LInt>(0)             )
        ,   capacity     ( std::max(static_cast<LInt>(1),n) )
        ,   container_0  ( std::max(static_cast<LInt>(1),n) )
        {}

        // Copy contructor
        Aggregator( const Aggregator & other )
        :   current_size ( other.current_size              )
        ,   capacity     ( other.capacity                  )
        ,   container_0  ( other.container_0               )
        {}

        friend void swap ( Aggregator & A, Aggregator & B ) noexcept
        {
            using std::swap;
            
            swap( A.current_size, B.current_size );
            swap( A.capacity,     B.capacity     );
            swap( A.container_0,  B.container_0  );
        }

        // Move constructor
        Aggregator( Aggregator && other ) noexcept
        :   Aggregator()
        {
            swap(*this, other);
        }

        // Move assignment operator
        Aggregator & operator=( Aggregator && other ) noexcept
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

        force_inline void Push( const T_0 a )
        {
            if( current_size >= capacity )
            {
                Expand();
            }

            container_0[current_size++] = a;
        }
        
        force_inline void Push( const T_0 * restrict const a, const LInt n )
        {
            if( current_size + n >= capacity )
            {
                RequireCapacity( std::max( current_size + n, static_cast<LInt>(2) * capacity ) );
            }

            copy_buffer( a, &container_0[current_size], n );
            current_size += n;
        }

        Container_0_T & Get()
        {
            ShrinkToFit();
            
            return container_0;
        }

        const Container_0_T & Get() const
        {
            ShrinkToFit();
            
            return container_0;
        }

    public:

        LInt Capacity() const
        {
            return capacity;
        }
        
        void RequireCapacity( const LInt new_capacity )
        {
            if( new_capacity > capacity)
            {
                Container_0_T new_container_0 (new_capacity);
                
                copy_buffer( container_0.data(), new_container_0.data(), capacity );
                
                using std::swap;
                swap( container_0, new_container_0 );
                
                capacity = new_capacity;
            }
        }
        
        force_inline T_0 & operator[]( const LInt i )
        {
            return container_0[i];
        }
        
        force_inline const T_0 & operator[]( const LInt i ) const
        {
            return container_0[i];
        }
        
        void ShrinkToFit()
        {
            Container_0_T new_container_0 ( container_0.data(), current_size );
            
            using std::swap;
            swap( container_0, new_container_0 );
        }
        
    protected:
        
        void Expand()
        {
            RequireCapacity( static_cast<LInt>(2) * capacity );
        }
    };
    
} // namespace Tensors
