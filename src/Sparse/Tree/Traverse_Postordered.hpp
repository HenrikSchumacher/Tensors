
//###########################################################################################
//####          Traversal_Postorder
//###########################################################################################

public:

    template<class Worker_T>
    void Traverse_Descendants_Postordered( mref<Worker_T> worker, const Int node ) const
    {
        debug_print(ClassName() + "::Traverse_Descendants_Postordered ( node = " + ToString(node) + " ) begins.");

        // Applies ker to the descendants of the node _and the node itself_ in postorder.

        // This is to guarantee that all children of node are processed on the same thread to avoid write-conflicts in the case they attempt to write to some of their common parent's memory.
        
        // Worker can be a class that has operator( Int node ) defined or simply a lambda.
        
        // This routine assumes that PostOrderedQ() evaluates to true so that the decendants lie contiguously directly before node.
        
        const Int desc_begin =  node - DescendantCount(node);
        const Int desc_end   =  node + 1;  // Apply worker also to yourself.

        
        for( Int desc = desc_begin; desc < desc_end; ++desc )
        {
            worker(desc);
        }
        
        debug_print(ClassName() + "::Traverse_Descendants_Postordered ( node = " + ToString(node) + " ) ends.");
    }


//    template<class Worker_T>
//    void Traverse_Children_Postordered( Worker_T & worker, const Int node ) const
//    {
//        debug_print(ClassName() + "::Traverse_Children_Postordered ( node = " + ToString(node) + " ) begins.");
//
//        // Applies ker to the direct children of the node in postorder.
//        // CAUTION: Does not apply ker to the node itself!
//        // This is to guarantee that all children of node are processed on the same thread to avoid write-conflicts in the case they attempt to write to some of their common parent's memory.
//
//
//        const Int child_begin =  ChildPointer(node     );
//        const Int child_end   =  ChildPointer(node + 1 );
//
//        for( Int k = child_begin; k < child_end; ++k )
//        {
//            const Int child = ChildIndex(k);
//
//            debug_assert(
//                parents[child] == node,
//                "Node " + ToString(child) + " is not the child of node " + ToString(node)+ "."
//            );
//
//            worker(child);
//        }
//
//        debug_print(ClassName() + "::Traverse_Children_Postordered ( node = " + ToString(node) + " ) ends.");
//    }

    template<Parallel_T parQ = Parallel, class Worker_T>
    void Traverse_Postordered( mref<std::vector<std::unique_ptr<Worker_T>>> workers ) const
    {
        std::string tag = ClassName() + "::Traverse_Postordered<" + (parQ == Parallel ? "Parallel" : "Sequential") + ">";
        
        if( !PostOrderedQ() )
        {
            eprint(tag+" requires postordered tree! Doing nothing.");
            return;
        }
        
        ptic(tag);        
        
        std::string tag_1 = "Apply worker " + workers[0]->ClassName() + " to level";
        
        const Int target_split_level = static_cast<Int>(tree_top_levels.size()-1);
        
        // Process the subtrees, but not their roots!
        // (That is to be done by these roots' parents!)
        {
            const Int k_begin = 0;
            const Int k_end   = static_cast<Int>(subtrees.size());
            
            const Int use_threads = (parQ == Parallel) ? Min( thread_count, k_end - k_begin ) : 1;
            
            ptic(tag_1 + " <= "+ToString(target_split_level)+"; using " + ToString(use_threads) + " threads.");
            
            ParallelDo_Dynamic(
                [=,this,&workers]( const Int thread, const Int k )
                {
                    const Time start_time = Clock::now();

                    mref<Worker_T> worker = *workers[thread];

                    const Int node = subtrees[k];

                    Traverse_Descendants_Postordered( worker, node );

                    const Time stop_time = Clock::now();
                    
                    pprint(
                        tag + ": Worker " + ToString(thread) + " required " +
                             ToString(Tools::Duration(start_time,stop_time)) +
                            " s for the " + ToString(DescendantCount(node)) + " descendants of node " + ToString(node) + "."
                    );
                },
                k_begin, k_end, Scalar::One<Int>, use_threads
            );
            
            ptoc(tag_1 + " <= "+ToString(target_split_level)+"; using " + ToString(use_threads) + " threads.");
        }
        

        for( Int d = static_cast<Int>(tree_top_levels.size()); d --> Scalar::One<Int> ; )
        {
            const Int k_begin = 0;
            const Int k_end   = static_cast<Int>(tree_top_levels[d].size());

            const Int use_threads = parQ == Parallel ? Min( thread_count, k_end - k_begin ) : one;

            ptic(tag_1 + " = " + ToString(d) + "; using " + ToString(use_threads) + " threads.");
            
            ParallelDo_Dynamic(
                [=,this,&workers]( const Int thread, const Int k )
                {
                    const Time start_time = Clock::now();

                    mref<Worker_T> worker = *workers[thread];

                    const Int node = tree_top_levels[d][k];

                    worker( node );

                    const Time stop_time = Clock::now();
                    pprint(
                        tag + ": Worker " + ToString(thread) + " required " +
                             ToString(Tools::Duration(start_time,stop_time)) +
                            " s for completing node " + ToString(node) + "."
                    );
                },
                k_begin, k_end, Scalar::One<Int>, use_threads
            );


            ptoc(tag_1 + " = "+ToString(d)+"; using " + ToString(use_threads) + " threads.");

        } // for( Int d = target_split_level; d --> Scalar::One<Int> ; )
        
        
//        
//        
//        
//        // Process the root node's children. (Merge all the subtrees)
//        // We can do this in parallel, because these subtrees have pairwise disjoint index sets.
//        // (Often, there is just a single subtree.)
//        {
//            const Int k_begin = ChildPointer( Root()     );
//            const Int k_end   = ChildPointer( Root() + 1 );
//    
//
//            const Int use_threads = parQ == Parallel ? Min( thread_count, k_end - k_begin ) : one;
//    
//            ParallelDo_Dynamic(
//                [=,this,&workers]( const Int thread, const Int k )
//                {
//                    const Time start_time = Clock::now();
//    
//                    mref<Worker_T> worker = *workers[thread];
//    
//                    const Int node = ChildIndex(k);
//    
//                    worker( node );
//    
//                    const Time stop_time = Clock::now();
//                    logprint(
//                        tag + ": Worker " + ToString(thread) + " required " +
//                             ToString(Tools::Duration(start_time,stop_time)) +
//                            " s for completing node " + ToString(node) + "."
//                    );
//                },
//                k_begin, k_end, Scalar::One<Int>,
//                use_threads
//            );
//    
//        }


        ptoc(tag);
    }
