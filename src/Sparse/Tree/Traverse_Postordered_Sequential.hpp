public:

    // Sequential postorder traversal. Meant to be used only for initialization and for reference purposes.
        template<class Lambda_PreVisit, class Lambda_PostVisit, class Lambda_LeafVisit>
        void Traverse_Postordered_Sequential(
            Lambda_PreVisit  pre_visit,
            Lambda_PostVisit post_visit,
            Lambda_LeafVisit leaf_visit,
            const Int tree_top_depth = std::numeric_limits<Int>::max()
        )
        {
            ptic(ClassName()+"::Traverse_Postordered_Sequential");
            
            Tensor1<Int, Int> stack   ( n );
            Tensor1<Int, Int> depth   ( n );
            Tensor1<bool,Int> visited ( n, false );

            Int i = zero; // stack pointer
            
            stack[i] = Root();
            depth[i] = zero;
            
//            Int i = -1; // stack pointer
//
//            // Push the children of the root onto stack. The root itself is not to be processed.
//            for( Int k = ChildPointer(Root()+1); k --> ChildPointer(Root()); )
//            {
//                ++i;
//                stack[i]   = ChildIndex(k);
//                depth[i]   = 0;
//            }
            
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
            
            ptoc(ClassName()+"::Traverse_Postordered_Sequential");
        }
