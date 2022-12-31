#pragma once

#define CLASS EliminationTree

namespace Tensors
{
    template<typename Int>
    class CLASS
    {
    public:
        
        CLASS() = default;
        
        ~CLASS() = default;
        
        explicit CLASS( Tensor1<Int,Int> && parents_ )
        {
            ptic("EliminationTree");
            
            std::swap( parents, parents_ );
            
            const Int n = parents.Size()-1;
            
            Tensor1<Int,Int> id (n);
            
            for( Int i = 0; i < n; ++i )
            {
                id[i] = i;
            }
            
            Int thread_count  = 1;
            Int list_count  = 1;
            Int entry_counts [1] = {n};
            
            Int * parent_data = parents.data();
            Int * id_data = id.data();
            
            A = SparseBinaryMatrixCSR<Int,Int> (
                &parent_data,
                &id_data,
                &entry_counts[0],
                list_count, n+1, n, thread_count, false, 0
            );
            
            ptoc("EliminationTree");
        }
        
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   parents ( other.parents )
        ,   A       ( other.A       )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.parents, B.parents );
            swap( A.A,       B.A       );
        }
        
        // Copy assignment operator
        CLASS & operator=(CLASS other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept : CLASS()
        {
            swap(*this, other);
        }
        
        
        const Tensor1<Int,Int> & Parents() const
        {
            return parents;
        }
        
        const Tensor1<Int,Int> & ChildPointers() const
        {
            return A.Outer();
        }
        
        const Tensor1<Int,Int> & ChildIndices() const
        {
            return A.Inner();
        }
        
        force_inline Int VertexCount() const
        {
            return parents.Size()-static_cast<Int>(1);
        }
        
        force_inline Int ChildCount( const Int i ) const
        {
            // Returns number of children of child i.
            return A.Outer()[i+1]-A.Outer()[i];
        }
        
        force_inline Int Child( const Int i, const Int k ) const
        {
            // Returns k-th child of node i.
            return A.Inner()[A.Outer()[i]+k];
        }
        
        Tensor1<Int,Int> PostOrdering() const
        {
            Tensor1<Int, Int> p       ( VertexCount()+1 );
            Tensor1<Int, Int> stack   ( VertexCount()+1 );
            Tensor1<bool,Int> visited ( VertexCount()+1, false );
            
            const Int * restrict const child_ptr = A.Outer().data();
            const Int * restrict const child_idx = A.Inner().data();
            
            Int ptr = 0;
            stack[0]   = VertexCount(); // I use VertexCount() as root node because 0 is already an ordinary node and I do not want to force usage of signed integers.
            visited[0] = false;
            
            Int counter = 0;
            
            while( ptr >= 0 )
            {
                Int node = stack[ptr];
                
                if( visited[ptr] )
                {
                    visited[ptr--] = false;
                    p[counter++] = node;
                }
                else
                {
                    visited[ptr] = true;
                    
                    const Int k_begin = child_ptr[node  ];
                    const Int k_end   = child_ptr[node+1];
                    
                    for( Int k = k_end; k --> k_begin; )
                    {
                        stack[++ptr] = child_idx[k];
                    }
                }
            }
            
            return p;
        }
        
        
        Tensor1<Int,Int> DescendantCounts() const
        {
            // Computed for each vertex i the number of its descendants.
            // This can be used to determine fundamental supernodes.
            // c.f. Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations.
            
            const Int n = VertexCount();
            
            Tensor1<Int,Int> descendant_counts (n,1);
            
            for( Int i = 0; i < n; ++i )
            {
                descendant_counts[parents[i]] += descendant_counts[i];
            }
            
            return descendant_counts;
        }
        
    protected:
        
        Tensor1<Int,Int> parents;
        
//        Tensor1<Int,Int> descendant_counts;
        
        
        SparseBinaryMatrixCSR<Int,Int> A;
        
    };
}

#undef CLASS
