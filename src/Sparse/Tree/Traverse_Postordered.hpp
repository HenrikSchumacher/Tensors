
//###########################################################################################
//####          Traversal_Postorder
//###########################################################################################


template<class Worker_T>
void Traverse_Children_Postordered( Worker_T & worker, Int node )
{
    // Applies ker to the descendants of the node in postorder.
    // CAUTION: Does not apply ker to the node itself!
    // This is to guarantee that all children of node are processed on the same thread to avoid write-conflicts in the case they attempt to write to some of their common parent's memory.
    
    // Worker can be a class that has operator( Int node ) defined or simply a lambda.
    
    // This routine assumes that PostOrderedQ() evaluates to true so that the decendants lie contiguously directly before node.
    
    const Int desc_begin = (node+1) - DescendantCount(node);
    const Int desc_end   =  node;  // Apply worker _not_ to yourself.
    
//    const Int desc_end   = (node+1);  // Apply worker also to yourself.
    
    for( Int desc = desc_begin; desc < desc_end; ++desc )
    {
        worker(desc);
    }
}


template<Parallel_T parQ = Parallel, class Worker_T>
void Traverse_Postordered(
    std::vector<std::unique_ptr<Worker_T>> & workers,
    Int tree_top_depth_
)
{
    std::string tag = ClassName() + "::Traverse_Postordered<" + (parQ == Parallel ? "par" : "seq") + ">";
    if( !PostOrderedQ() )
    {
        eprint(tag+" requires postordered tree! Doing nothing.");
        return;
    }
    
    Int tree_top_depth = std::min( one, tree_top_depth_ );
    
    const Int min_subtree_count = 4 * thread_count;
    
//            dump(min_subtree_count);
    
    while(
        ( tree_top_depth+1 < levels.RowCount() )
        &&
        ( levels.NonzeroCount(tree_top_depth) < min_subtree_count )
    )
    {
//                dump(levels.NonzeroCount(tree_top_depth));
        ++tree_top_depth;
    }
    
//            dump(tree_top_depth);
//            dump(levels.NonzeroCount(tree_top_depth));
    

    ptic(tag);
         
    
    
    std::string tag_1 = "Apply " + workers[0]->ClassName() + " to level";
    
    
    logprint(levels.Outer().ToString());
    logprint(levels.Inner().ToString());
    
    // Process the subtrees at lowest level, but not their roots!
    // (That is to be done by these roots' parents!)
    {
        const Int k_begin = LevelPointer(tree_top_depth     );
        const Int k_end   = LevelPointer(tree_top_depth + 1 );
        
        const Int use_threads = parQ == Parallel ? std::min( thread_count, k_end - k_begin ) : 1;
        
        ptic(tag_1 + " <= "+ToString(tree_top_depth)+"; using " + ToString(use_threads) + " threads.");
        
        // DEBUG_THIS
        // The error must be here.
        ParallelDo_Dynamic(
            [=,&workers]( const Int thread, const Int k )
            {
                const Time start_time = Clock::now();

                Worker_T & worker = *workers[thread];

                const Int node = LevelIndex(k);

                Traverse_Children_Postordered( worker, node );

                const Time stop_time = Clock::now();
                logprint(
                    tag+": Worker " + ToString(thread) + " required " +
                         ToString(Tools::Duration(start_time,stop_time)) +
                        " s for traversing the subtree at node " + ToString(node) + "."
                );
            },
            k_begin, k_end, Scalar::One<Int>, use_threads
        );
        
        ptoc(tag_1 + " <= "+ToString(tree_top_depth)+"; using " + ToString(use_threads) + " threads.");
    }
    
    

    
    for( Int d = tree_top_depth; d --> Scalar::One<Int> ; )
    {
        const Int k_begin = LevelPointer(d  );
        const Int k_end   = LevelPointer(d+1);
        
        const Int use_threads = parQ == Parallel ? std::min( thread_count, k_end - k_begin ) : one;
        
        ptic(tag_1 + " = "+ToString(d)+"; using " + ToString(use_threads) + " threads.");
        
        ParallelDo_Dynamic(
            [=,&workers]( const Int thread, const Int k )
            {
                const Time start_time = Clock::now();
                
                Worker_T & worker = *workers[thread];
                
                const Int node = LevelIndex(k);
                
                worker(node);
                
                const Time stop_time = Clock::now();
                logprint(
                    tag + ": Worker " + ToString(thread) + " required " +
                         ToString(Tools::Duration(start_time,stop_time)) +
                        " s for completing node " + ToString(node) + "."
                );
            },
            k_begin, k_end, Scalar::One<Int>,
            use_threads
        );
        
        
        // Process the root node's children. (Merge all the subtrees)
        {
            const Int thread = zero;
            
            const Time start_time = Clock::now();
            
            Worker_T & worker = *workers[thread];
            
            const Int node = Root();
            
            Traverse_Children_Postordered( worker, node );
            
            const Time stop_time = Clock::now();
            logprint(
                tag + ": Worker " + ToString(thread) + " required " +
                     ToString(Tools::Duration(start_time,stop_time)) +
                    " s for completing node " + ToString(node) + "."
            );
            
        }
        ptoc(tag_1 + " = "+ToString(d)+"; using " + ToString(use_threads) + " threads.");
    }
    
    ptoc(tag);
}
