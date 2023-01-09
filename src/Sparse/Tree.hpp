#pragma once

namespace Tensors
{
    template<typename Int>
    class Tree
    {
    public:
        
        // TODO: Parallel post-order traverals with templated parameters for
        // -- Previsit routine
        // -- Postvisit routine
        // (-- Revisit)
        // -- Leafvisit routine
        
        Tree() = default;
        
        ~Tree() = default;
        
        explicit Tree( Tensor1<Int,Int> & parents_, const Int thread_count_ = 1 )
        :   n                 ( parents_.Size()+1 )
        // We use and additional virtual vertex as root.
        ,   root              ( n-1               )
        ,   thread_count      ( thread_count_     )
        ,   descendant_counts ( n                 )
        {
            ptic(ClassName());
            
//            std::swap( parents, parents_ );
            
            parents = parents_;
            
//            ptic("Adjacency matrix");
            
            // Next we build the adjacency matrix A of the tree.

            Int list_count       = 1;
            Int entry_counts [1] = {n-1};
            
            post = Permutation<Int> ( n-1, thread_count );

            mut<Int> p = post.Scratch().data();
            
            const Int * idx_data = parents.data();
            const Int * jdx_data = post.GetPermutation().data(); // This is a iota.
            
            A = SparseBinaryMatrixCSR<Int,Int> (
                &idx_data,
                &jdx_data,
                &entry_counts[0],
                list_count, n, n, thread_count, false, 0
            );
//            ptoc("Adjacency matrix");
            
            // Compute postordering and descendant counts. Also check whether tree is already postordered.
            
            // TODO: Can be parallelized via a bit of limited depth DFS, and then several parallel DFS.
//            ptic("Postorder traversal");
            Tensor1<Int, Int> stack   (2*n+2 );
            Tensor1<bool,Int> visited (2*n+2, false );
            
//            Int i      = 0; // stack pointer
//            stack  [0] = root;
//            visited[0] = false;
            
            Int i = -1; // stack pointer

            // Push the children of root onto stack. root itself is not to be processed.
            for( Int k = ChildPointer(root+1); k --> ChildPointer(root); )
            {
                ++i;
                stack[i] = ChildIndex(k);
            }
            
            Int counter = 0;
            
            // post order traversal of the tree
            while( i >= 0 )
            {
                const Int node = stack[i];
                
                const Int k_begin = ChildPointer(node  );
                const Int k_end   = ChildPointer(node+1);
                
                if( !visited[i] )
                {
                    // The first time we visit this node we mark it as visited
                    visited[i] = true;
                    
                    // Pushing the children in reverse order onto the stack.
                    for( Int k = k_end; k --> k_begin; )
                    {
                        stack[++i] = ChildIndex(k);
                    }
                }
                else
                {
                    // Visiting the node for the second time.
                    // We are moving in direction towards the root.
                    // Hence all children have already been visited.
                    
                    // Popping current node from the stack.
                    visited[i--] = false;
                    
                    p[counter] = node;
//                    postordered   = postordered && (counter == node);
                    ++counter;
                    
                    Int sum = 1; // The node itself is also counted as its own descendant.
                    
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        sum += descendant_counts[ChildIndex(k)];
                    }

                    descendant_counts[node] = sum;
                }
            }
            
            descendant_counts[root] = 1;
            for( Int k = ChildPointer(root+1); k --> ChildPointer(root); )
            {
                descendant_counts[root] += descendant_counts[ChildIndex(k)];
            }

            
            // Now post.Scratch() contains the post ordering.
            
            post.SwapScratch();
            
            // Now post.GetPermutation() contains the post ordering.

//            ptoc("Postorder traversal");

            ptoc(ClassName());
        }
        
        
    
        // Copy constructor
        Tree( const Tree & other )
        :   n                 ( other.n                 )
        ,   root              ( other.root              )
        ,   thread_count      ( other.thread_count      )
        ,   post              ( other.post              )
        ,   descendant_counts ( other.descendant_counts )
        ,   parents           ( other.parents           )
        ,   A                 ( other.A                 )
        {}
        
        // We could also simply use the implicitly created copy constructor.
        
        friend void swap (Tree &A, Tree &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.n,                 B.n                 );
            swap( A.root,              B.root              );
            swap( A.thread_count,      B.thread_count      );
            swap( A.post,              B.post              );
            swap( A.descendant_counts, B.descendant_counts );
            swap( A.parents,           B.parents           );
            swap( A.A,                 B.A                 );
        }
        
        // Copy assignment operator
        Tree & operator=(Tree other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        Tree( Tree && other ) noexcept : Tree()
        {
            swap(*this, other);
        }
        
        
        const Tensor1<Int,Int> & Parents() const
        {
            return parents;
        }
        
        const Permutation<Int> & PostOrdering() const
        {
            return post;
        }
        
        const Tensor1<Int,Int> & DescendantCounts() const
        {
            return descendant_counts;
        }
        
        Int DescendantCount( const Int i ) const
        {
            return descendant_counts[i];
        }
        
        const Tensor1<Int,Int> & ChildPointers() const
        {
            return A.Outer();
        }
        
        Int ChildPointer( const Int i ) const
        {
            return A.Outer(i);
        }
        
        const Tensor1<Int,Int> & ChildIndices() const
        {
            return A.Inner();
        }
        
        Int ChildIndex( const Int k ) const
        {
            return A.Inner(k);
        }
        
        force_inline Int VertexCount() const
        {
            return n;
        }
        
        force_inline Int ChildCount( const Int i ) const
        {
            // Returns number of children of child i.
            return ChildPointer(i+1)-ChildPointer(i);
        }
        
        
        force_inline Int Child( const Int i, const Int k ) const
        {
            // Returns k-th child of node i.
            return A.ChildIndex(ChildPointer(i)+k);
        }

        
        bool PostOrdered() const
        {
            bool postordered = true;
            
            #pragma omp parallel for num_threads( thread_count ) reduction( && : postordered )
            for( Int i = 0; i < n-1; ++i )
            {
                const Int p_i = parents[i];
                
                postordered = postordered && (i < p_i) && (i >= p_i + 1 - DescendantCount(p_i) );
            }
            
            return postordered;
        }
        
        Int Root() const
        {
            return root;
        }
        
    protected:
        
        Int n;
        
        Int root;
        
        Int thread_count;
        
        Permutation<Int> post;
        
        Tensor1<Int,Int> descendant_counts;
    
        Tensor1<Int,Int> parents;
        
        SparseBinaryMatrixCSR<Int,Int> A;
      
//
//        bool postordered = true;
        
    public:
        
        std::string ClassName() const
        {
            return "Tree<"+TypeName<Int>::Get()+">";
        }
    };
}
