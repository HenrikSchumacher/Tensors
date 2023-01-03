#pragma once

namespace Tensors
{
    template<typename T_0, typename LInt>
    class alignas(OBJECT_ALIGNMENT) Aggregator
    {
        // A dynamically growing version of Tensor1 that allows pushing of several elements at once.
        // It also allows decently fast access with the operator[].
        
        // Setting thread_count higher than 1 can speed up copy operations -- but only if sufficiently many OpenMP are free and only if there is sufficient RAM bandwidth.
        // Better set thread_count = 1 if you want to use more than one Aggregator at a time.
        
        
        // TODO: It might be a better idea to use a std::vector of Tensor1s and then grow it.
        // TODO: There will be some overhead for indexing into this nested data structure, though.
        // TODO: And using &operator[i] is asking for trouble...
        
        ASSERT_INT(LInt);

        using Container_0_T = Tensor1<T_0,LInt>;

        LInt current_size   = static_cast<LInt>(0);
        LInt capacity       = static_cast<LInt>(1);
        
        Container_0_T container_0 {capacity};

        size_t thread_count = 1;
        
    public:

        Aggregator() = default;

        ~Aggregator() = default;

        explicit Aggregator( const LInt n )
        :   current_size ( static_cast<LInt>(0)             )
        ,   capacity     ( std::max(static_cast<LInt>(1),n) )
        ,   container_0  ( std::max(static_cast<LInt>(1),n) )
        ,   thread_count ( 1                                )
        {}
        
        explicit Aggregator( const LInt n, const size_t thread_count_ )
        :   current_size ( static_cast<LInt>(0)             )
        ,   capacity     ( std::max(static_cast<LInt>(1),n) )
        ,   container_0  ( std::max(static_cast<LInt>(1),n) )
        ,   thread_count ( thread_count_                    )
        {}

        // Copy contructor
        Aggregator( const Aggregator & other )
        :   current_size ( other.current_size               )
        ,   capacity     ( other.capacity                   )
        ,   container_0  ( other.container_0                )
        ,   thread_count ( thread_count                     )
        {}

        friend void swap ( Aggregator & A, Aggregator & B ) noexcept
        {
            using std::swap;
            
            swap( A.current_size, B.current_size );
            swap( A.capacity,     B.capacity     );
            swap( A.container_0,  B.container_0  );
            swap( A.thread_count, B.thread_count );
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
        
        force_inline void Push( ptr<T_0> a, const LInt n )
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
            ptic(ClassName()+"::RequireCapacity");
            if( new_capacity > capacity)
            {
                Container_0_T new_container_0 (new_capacity);
                
                copy_buffer( container_0.data(), new_container_0.data(), capacity, thread_count );
                
                using std::swap;
                swap( container_0, new_container_0 );
                
                capacity = new_capacity;
            }
            ptoc(ClassName()+"::RequireCapacity");
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
            ptic(ClassName()+"::ShrinkToFit");
//            Container_0_T new_container_0 ( container_0.data(), current_size );
            
            Container_0_T new_container_0 ( current_size );
            new_container_0.Read( container_0.data(), thread_count );
            
            using std::swap;
            swap( container_0, new_container_0 );
            ptoc(ClassName()+"::ShrinkToFit");
        }
        
    protected:
        
        void Expand()
        {
            RequireCapacity( static_cast<LInt>(2) * capacity );
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Aggregator<"+TypeName<T_0>::Get()+","+TypeName<LInt>::Get()+">";
        }
    };
    
} // namespace Tensors
