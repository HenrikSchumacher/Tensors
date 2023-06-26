#pragma once

// TODO: Traverse_Preordered_Parallel.

// TODO: Improve load balancing.

// TODO: Would be great to have allow nodes in the top levels of the tree to use more than one thread...

namespace Tensors
{
    template<typename Int>
    class Tree
    {
    protected:
        
        Int n; // The number of vertices, _including_ the virtual root vertex.
        
        Int thread_count;

        Tensor1<Int,Int> parents;
        
        Tensor1<double,Int> costs;
        Tensor1<double,Int> acc_costs;
        
        Sparse::BinaryMatrixCSR<Int,Int> A;       // The adjacency matrix of the directed graph.
        
        Tensor1<Int,Int> descendant_counts;
        
        Permutation<Int> post;                  // To store the postordering.
        
        Sparse::BinaryMatrixCSR<Int,Int> levels;       // The adjacency matrix of the directed graph.

        static constexpr Int zero = 0;
        static constexpr Int one  = 1;

    public:
        
        Tree() = default;
        
        ~Tree() = default;
        
        explicit Tree(
            Tensor1<Int,Int> && parents_,
            const Int thread_count_ = 1
        )
        :   n                 ( parents_.Size()+1   )
        // We use an additional virtual vertex as root.
        ,   thread_count      ( thread_count_       )
        ,   parents           ( std::move(parents_) )
        ,   costs             ( n, 1.               )
        ,   descendant_counts ( n                   )
        {
            ptic(ClassName());
            Init();
            ptoc(ClassName());
        }
        
        Tree(
            Tensor1<   Int,Int> && parents_,
            Tensor1<double,Int> && costs_,
            const Int thread_count_ = 1
        )
        :   n                 ( parents_.Size()+1   )
        // We use an additional virtual vertex as root.
        ,   thread_count      ( thread_count_       )
        ,   parents           ( std::move(parents_) )
        ,   costs             ( std::move(costs_)   )
        ,   descendant_counts ( n                   )
        {
            ptic(ClassName());
            Init();
            ptoc(ClassName());
        }
        
        
    protected:
        
        void Init()
        {
            // Next we build the adjacency matrix A of the tree.

            post = Permutation<Int> ( n-1, thread_count );

            mut<Int> p = post.Scratch().data();
            
            {
                Int list_count       = 1;
                Int entry_counts [1] = {n-1};
                
                const Int * idx_data = parents.data();
                const Int * jdx_data = post.GetPermutation().data(); // This is a iota.
                
                A = Sparse::BinaryMatrixCSR<Int,Int> (
                    &idx_data,
                    &jdx_data,
                    &entry_counts[0],
                    list_count, n, n, thread_count, false, 0
                );
            }
            
            // Compute postordering and descendant counts.
            Int counter = 0;
            Int tree_top_depth = 0;
            
            Tensor1<Int, Int> node_to_depth (n, one);
            
            node_to_depth[Root()] = 0;

            acc_costs = costs;
            
            Traversal_Postordered_Sequential(
                [this, &node_to_depth, &tree_top_depth]( const Int node )
                {
                    const Int depth = node_to_depth[node]+1;
                    
                    tree_top_depth = std::max(tree_top_depth, depth);

                    for( Int k = ChildPointer(node); k < ChildPointer(node+1); ++k )
                    {
                        node_to_depth[ChildIndex(k)] = depth;
                    }
                }
                ,
                [this, &counter, p]( const Int node )
                {
                    p[counter++] = node;

                    Int sum = 1; // The node itself is also counted as its own descendant.
                    double cost = 0.; // costs[node] has already been filled.
                    
                    for( Int k = ChildPointer(node); k < ChildPointer(node+1); ++k )
                    {
                        const Int child = ChildIndex(k);
                        sum  += descendant_counts[child];
                        cost += acc_costs[child];
                    }
                    descendant_counts[node]  = sum;
                    acc_costs        [node] += cost;
                }
                ,
                []( const Int node ) {}
            );
            
            descendant_counts[Root()] = 1;
            
            {
                Int list_count       = 1;
                Int entry_counts [1] = {n-1};
                                
                const Int * idx_data = node_to_depth.data();
                const Int * jdx_data = post.GetPermutation().data(); // This is still a iota.
                
                levels = Sparse::BinaryMatrixCSR<Int,Int> (
                    &idx_data,
                    &jdx_data,
                    &entry_counts[0],
                    list_count, tree_top_depth+1, n, thread_count, true, 0
                );
            }
            
            for( Int k = ChildPointer(Root()+1); k --> ChildPointer(Root()); )
            {
                const Int child = ChildIndex(k);
                descendant_counts[Root()] += descendant_counts[child];
                acc_costs[Root()]         += acc_costs        [child];
            }

            // Now post.Scratch() contains the post ordering.
            
            post.SwapScratch( Inverse::False );
            
            // Now post.GetPermutation() contains the post ordering.

//            print( post.GetPermutation().ToString() );
        }
        
    public:
        
        // Copy constructor
        Tree( const Tree & other )
        :   n                 ( other.n                 )
        ,   thread_count      ( other.thread_count      )
        ,   A                 ( other.A                 )
        ,   parents           ( other.parents           )
        ,   descendant_counts ( other.descendant_counts )
        ,   post              ( other.post              )
        ,   costs             ( other.costs             )
        ,   acc_costs         ( other.acc_costs         )
        ,   levels            ( other.levels            )
        {}
        
        // We could also simply use the implicitly created copy constructor.
        
        friend void swap (Tree &X, Tree &Y ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( X.n,                 Y.n                 );
            swap( X.thread_count,      Y.thread_count      );
            swap( X.parents,           Y.parents           );
            swap( X.A,                 Y.A                 );
            swap( X.descendant_counts, Y.descendant_counts );
            swap( X.post,              Y.post              );
            swap( X.costs,             Y.costs             );
            swap( X.acc_costs,         Y.acc_costs         );
            swap( X.levels,            Y.levels            );
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
        
        force_inline Int DescendantCount( const Int i ) const
        {
            return descendant_counts[i];
        }
        
        const Tensor1<Int,Int> & ChildPointers() const
        {
            return A.Outer();
        }
        
        force_inline Int ChildPointer( const Int i ) const
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
        
        const Tensor1<Int,Int> & LevelPointers() const
        {
            return levels.Outer();
        }
        
        force_inline Int LevelPointer( const Int i ) const
        {
            return levels.Outer(i);
        }
        
        const Tensor1<Int,Int> & LevelIndices() const
        {
            return levels.Inner();
        }
        
        Int LevelIndex( const Int k ) const
        {
            return levels.Inner(k);
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

        const Tensor1<double,Int> & Costs() const
        {
            return costs;
        }
        
        const Tensor1<double,Int> & AccumulatedCosts() const
        {
            return acc_costs;
        }
        
        bool PostOrderedQ() const
        {
            return ParallelDoReduce(
                [=]( const Int i ) -> bool
                {
                    const Int p_i = parents[i];

                    return (i < p_i) && (i >= p_i + one - DescendantCount(p_i) );
                },
                AndReducer(),
                true,
                zero, n-1, thread_count
            );
        }
        
        
        Int Root() const
        {
            return n-1;
        }
        
        // Sequential postorder traversal. Meant to be used only for initialization and for reference purposes.
        template<class Lambda_PreVisit, class Lambda_PostVisit, class Lambda_LeafVisit>
        void Traversal_Postordered_Sequential(
            Lambda_PreVisit  pre_visit,
            Lambda_PostVisit post_visit,
            Lambda_LeafVisit leaf_visit,
            const Int tree_top_depth = std::numeric_limits<Int>::max()
        )
        {
            ptic(ClassName()+"::Traversal_Postordered_Sequential");
            
            Tensor1<Int, Int> stack   ( n );
            Tensor1<Int, Int> depth   ( n );
            Tensor1<bool,Int> visited ( n, false );
            
            Int i = -1; // stack pointer

            // Push the children of the root onto stack. The root itself is not to be processed.
            for( Int k = ChildPointer(Root()+1); k --> ChildPointer(Root()); )
            {
                ++i;
                stack[i]   = ChildIndex(k);
                depth[i]   = 0;
            }
            
            // post order traversal of the tree
            while( i >= zero )
            {
                const Int node    = stack[i];
                const Int d       = depth[i];
                const Int k_begin = ChildPointer(node  );
                const Int k_end   = ChildPointer(node+1);
                
                if( !visited[i] && (d < tree_top_depth) && (k_begin < k_end) )
                {
                    // We visit this node for the first time.
                    pre_visit(node);
                    
                    visited[i] = true;
                    
                    // Pushing the children in reverse order onto the stack.
                    for( Int k = k_end; k --> k_begin; )
                    {
                        stack[++i] = ChildIndex(k);
                        depth[i]   = d+1;
                    }
                }
                else
                {
                    // Visiting the node for the second time.
                    // We are moving in direction towards the root.
                    // Hence all children have already been visited.
                    
                    // Popping current node from the stack.
                    visited[i--] = false;

                    // things to be done when node is a leaf.
                    if (k_begin == k_end)
                    {
                        leaf_visit(node);
                    }
                    
                    post_visit(node);
                }
            }
            ptoc(ClassName()+"::Traversal_Postordered_Sequential");
        }
        
#include "Tree/Traverse_Preordered.hpp"
#include "Tree/Traverse_Postordered.hpp"
        
                 
    public:
        
        std::string ClassName() const
        {
            return std::string("Tree")+"<"+TypeName<Int>+">";
        }
    };
}
