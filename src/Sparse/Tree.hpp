#pragma once

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
        
        Sparse::BinaryMatrixCSR<Int,Int> A;       // The adjacency matrix of the directed graph.
        
        Tensor1<Int,Int> descendant_counts;
        
        Permutation<Int> post;                  // To store the postordering.

        Tensor1<double,Int> costs;
        
        Sparse::BinaryMatrixCSR<Int,Int> levels;       // The adjacency matrix of the directed graph.

        static constexpr Int zero = 0;
        static constexpr Int one  = 1;

    public:
        
        Tree() = default;
        
        ~Tree() = default;
        
        explicit Tree( Tensor1<Int,Int> && parents_, const Int thread_count_ = 1 )
        :   n                 ( parents_.Size()+1   )
        // We use an additional virtual vertex as root.
        ,   thread_count      ( thread_count_       )
        ,   parents           ( std::move(parents_) )
        ,   descendant_counts ( n                   )
        {
            ptic(ClassName());

            // Next we build the adjacency matrix A of the tree.

            post = Permutation<Int> ( n-1, thread_count );

            dump( post.TrivialQ() );
            
            mut<Int> p = post.Scratch().data();
            
            {
                Int list_count       = 1;
                Int entry_counts [1] = {n-1};
                
                dump( post.TrivialQ() );
                
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

            
            Traversal_DFS_Sequential(
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
                    for( Int k = ChildPointer(node); k < ChildPointer(node+1); ++k )
                    {
                        sum += descendant_counts[ChildIndex(k)];
                    }
                    descendant_counts[node] = sum;
                }
                ,
                [](Int node) {}
            );
            
            descendant_counts[Root()] = 1;
            
            {
                Int list_count       = 1;
                Int entry_counts [1] = {n-1};
                
                dump( post.TrivialQ() );
                
                const Int * idx_data = node_to_depth.data();
                const Int * jdx_data = post.GetPermutation().data(); // This is still a iota.
                
                levels = Sparse::BinaryMatrixCSR<Int,Int> (
                    &idx_data,
                    &jdx_data,
                    &entry_counts[0],
                    list_count, tree_top_depth+1, n, thread_count, true, 0
                );
                
                dump( post.TrivialQ() );
            }
            
            for( Int k = ChildPointer(Root()+1); k --> ChildPointer(Root()); )
            {
                descendant_counts[Root()] += descendant_counts[ChildIndex(k)];
            }
            dump( post.TrivialQ() );
            // Now post.Scratch() contains the post ordering.
            
            post.SwapScratch( Inverse::False );
            
            // Now post.GetPermutation() contains the post ordering.

            dump( post.TrivialQ() );
//            print( post.GetPermutation().ToString() );
            
            ptoc(ClassName());
        }
        
        
    
        // Copy constructor
        Tree( const Tree & other )
        :   n                 ( other.n                 )
        ,   thread_count      ( other.thread_count      )
        ,   A                 ( other.A                 )
        ,   parents           ( other.parents           )
        ,   descendant_counts ( other.descendant_counts )
        ,   post              ( other.post              )
        ,   costs             ( other.costs             )
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
        
//        bool PostOrderedQ() const
//        {
//            return ParallelDoReduce(
//                [=]( const Int i ) -> bool
//                {
//                    const Int p_i = parents[i];
//
//                    return (i < p_i) && (i >= p_i + one - DescendantCount(p_i) );
//                },
//                AndReducer(),
//                true,
//                zero, n-1, thread_count
//            );
//        }
        
        bool PostOrderedQ() const
        {
            return ParallelDoReduce(
                [=]( const Int i ) -> bool
                {
                    const Int p_i = parents[i];
                 
                    return (i < p_i);
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
        
        template<class Lambda_PerVisit, class Lambda_PostVisit, class Lambda_LeafVisit>
        void Traversal_DFS_Sequential(
            Lambda_PerVisit  pre_visit,
            Lambda_PostVisit post_visit,
            Lambda_LeafVisit leaf_visit,
            const Int tree_top_depth = std::numeric_limits<Int>::max()
        )
        {
            ptic(ClassName()+"::Traversal_DFS_Sequential");
            
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
            
            ptoc(ClassName()+"::Traversal_DFS_Sequential");
        }
        
        
        
        
        
        template<class Worker_T>
        void Traverse_DFS_Postordered( Worker_T & worker, Int node )
        {
            // Applies ker to node and its descendants in postorder.
            // Worker can be a class that has operator( Int node ) defined or simply a lambda.
            
            // This routine assumes that PostOrderedQ() evaluates to true so that the decendants lie contiguously directly before node.
            
            const Int desc_begin = (node+1) - DescendantCount(node);
            const Int desc_end   = (node+1);  // Apply worker also to yourself.
            
            for( Int desc = desc_begin; desc < desc_end; ++desc )
            {
                worker(desc);
            }
        }
        
        template<class Worker_T>
        void Traverse_DFS_Parallel(
            std::vector<std::unique_ptr<Worker_T>> & workers,
            Int tree_top_depth_
        )
        {
            if( !PostOrderedQ() )
            {
                eprint(ClassName()+"::Traverse_DFS_Parallel requires postordered tree! Doing nothing.");
                return;
            }
            
            Int tree_top_depth = std::min( zero, tree_top_depth_ );
            
            const Int min_subtree_count = 4 * thread_count;
            
            dump(min_subtree_count);
            
            while(
                ( tree_top_depth+1 < levels.RowCount() )
                &&
                ( levels.NonzeroCount(tree_top_depth) < min_subtree_count )
            )
            {
                dump(levels.NonzeroCount(tree_top_depth));
                ++tree_top_depth;
            }
            
            dump(tree_top_depth);
            dump(levels.NonzeroCount(tree_top_depth));
            

            ptic(ClassName()+"::Traverse_DFS_Parallel");
                 
            std::string tag = "Apply " + workers[0]->ClassName() + " to level";
            
            ptic(tag+" <= "+ToString(tree_top_depth)+")");
            {
                const Int k_begin = LevelPointer(tree_top_depth    );
                const Int k_end   = LevelPointer(tree_top_depth + 1);
                
                const Int use_threads = std::min( thread_count, k_end - k_begin );
                
                ParallelDo_Dynamic(
                    [=,&workers]( const Int thread, const Int k )
                    {
                        const Time start_time = Clock::now();
                        
                        Worker_T & worker = *workers[thread];
                        
                        const Int node = LevelIndex(k);
                        
                        Traverse_DFS_Postordered( worker, node );
                        
                        const Time stop_time = Clock::now();
                        logprint(
                            ClassName()+"::Traverse_DFS_Parallel: Worker " + ToString(thread) + " required " +
                                 ToString(Tools::Duration(start_time,stop_time)) +
                                " s for traversing the subtree at node " + ToString(node) + "."
                        );
                    },
                    k_begin, k_end, Scalar::One<Int>,
                    use_threads
                );
            }
            
            ptoc(tag+" <= "+ToString(tree_top_depth)+")");

            
            for( Int d = tree_top_depth; d --> Scalar::One<Int> ; ) // Don't process the root node!
            {
                ptic(tag+" = "+ToString(d)+")");
                
                const Int k_begin = LevelPointer(d  );
                const Int k_end   = LevelPointer(d+1);
                
                const Int use_threads = std::min( thread_count, k_end - k_begin );
                
                ParallelDo_Dynamic(
                    [=,&workers]( const Int thread, const Int k )
                    {
                        const Time start_time = Clock::now();
                        
                        Worker_T & worker = *workers[thread];
                        
                        const Int node = LevelIndex(k);
                        
                        worker(node);
                        
                        const Time stop_time = Clock::now();
                        logprint(
                            ClassName()+"::Traverse_DFS_Parallel: Worker " + ToString(thread) + " required " +
                                 ToString(Tools::Duration(start_time,stop_time)) +
                                " s for completing node " + ToString(node) + "."
                        );
                    },
                    k_begin, k_end, Scalar::One<Int>,
                    use_threads
                );
                
                ptoc(tag+" = "+ToString(d)+")");
            }
            
            ptoc(ClassName()+"::Traverse_DFS_Parallel");
        }
                 
    public:
        
        std::string ClassName() const
        {
            return std::string("Tree")+"<"+TypeName<Int>+">";
        }
    };
}
