public:

    template<class Worker_T>
    void Traverse_Descendants_PreOrdered( mref<Worker_T> worker, const Int node ) const
    {
        TOOLS_DEBUG_PRINT(ClassName()+"::Traverse_Descendants_PreOrdered ( node = " + ToString(node) + " ) begins.");

        // Applies ker to the descendants of the node _and the node itself_ in postorder.
        // This is to guarantee that all children of node are processed on the same thread to avoid write-conflicts in the case they attempt to write to some of their common parent's memory.
        
        // Worker can be a class that has operator( Int node ) defined or simply a lambda.
        
        // This routine assumes that PostOrderedQ() evaluates to true so that the decendants lie contiguously directly before node.
        
        const Int desc_begin =  node - DescendantCount(node);
        const Int desc_end   =  node + 1;  // Apply worker also to yourself.

        
        for( Int desc = desc_end; desc --> desc_begin; )
        {
            worker(desc);
        }
        
        TOOLS_DEBUG_PRINT(ClassName()+"::Traverse_Descendants_PreOrdered ( node = " + ToString(node) + " ) ends.");
    }


    template<Parallel_T parQ = Parallel, class Worker_T>
    void Traverse_PreOrdered( std::vector<std::unique_ptr<Worker_T>> & workers ) const
    {
        std::string tag = ClassName()+"::Traverse_PreOrdered<" + (parQ == Parallel ? "Parallel" : "Sequential") + ">";
        if( !PostOrderedQ() )
        {
            eprint(tag+" requires postordered tree! Doing nothing.");
            return;
        }
        
        TOOLS_PTIC(tag);
        
//        std::string tag_1 = "Apply worker " + workers[0]->ClassName() + " to level";
//        
//        const Int target_split_level = static_cast<Int>(tree_top_levels.size()-1);

        for( Size_T d = Size_T(1); d < tree_top_levels.size(); ++d )
        {
            const Size_T k_begin = 0;
            const Size_T k_end   = tree_top_levels[d].size();

            const Size_T use_threads = parQ == Parallel ? Min( ToSize_T(thread_count), k_end - k_begin ) : Size_T(1);

//            TOOLS_PTIC(tag_1 + " = " + ToString(d) + "; using " + ToString(use_threads) + " threads.");
            
            ParallelDo_Dynamic(
                [=,this,&workers]( const Size_T thread, const Size_T k )
                {
//                    const Time start_time = Clock::now();

                    Worker_T & worker = *workers[thread];

                    const Int node = tree_top_levels[d][k];

                    worker( node );

//                    const Time stop_time = Clock::now();
//                    logprint(
//                        tag + ": Worker " + ToString(thread) + " required " +
//                             ToString(Tools::Duration(start_time,stop_time)) +
//                            " s for completing node " + ToString(node) + " and its direct children."
//                    );
                },
                k_begin, k_end, Size_T(1), use_threads
            );


//            TOOLS_PTOC(tag_1 + " = "+ToString(d)+"; using " + ToString(use_threads) + " threads.");

        } // for( Int d = target_split_level; d --> Scalar::One<Int> ; )


        // Process the subtrees, but not their roots!
        // (That is to be done by these roots' parents!)
        {
            const Size_T k_begin = 0;
            const Size_T k_end   = subtrees.size();
            
            const Size_T use_threads = (parQ == Parallel) ? Min( ToSize_T(thread_count), k_end - k_begin ) : Size_T(1);
            
//            TOOLS_PTIC(tag_1 + " <= "+ToString(target_split_level)+"; using " + ToString(use_threads) + " threads.");
            
            ParallelDo_Dynamic(
                [=,this,&workers]( const Size_T thread, const Size_T k )
                {
//                    const Time start_time = Clock::now();

                    Worker_T & worker = *workers[thread];

                    const Int node = subtrees[k];

                    Traverse_Descendants_PreOrdered( worker, node );

//                    const Time stop_time = Clock::now();
//                    
//                    logprint(
//                        tag + ": Worker " + ToString(thread) + " required " +
//                             ToString(Tools::Duration(start_time,stop_time)) +
//                            " s for the " + ToString(DescendantCount(node)) + " descendants of node " + ToString(node) + "."
//                    );
                },
                k_begin, k_end, Size_T(1), use_threads
            );
            
//            TOOLS_PTOC(tag_1 + " <= "+ToString(target_split_level)+"; using " + ToString(use_threads) + " threads.");
        }
        TOOLS_PTOC(tag);
    }
