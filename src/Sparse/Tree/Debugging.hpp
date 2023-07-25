public:

    class DebugWorker
    {
    public:
     
        cref<Tree<Int>> tree = nullptr;
        
        DebugWorker( const Tree<Int> & tree_ )
        :   tree( tree_ )
        {}
        
        void operator()( const Int node )
        {
            tree.IncrementChecklist( node );
        }
        
        std::string ClassName()
        {
            return "DebugWorker";
        }
    };

    void AllocateCheckList() const
    {
//        print(ClassName() + "::AllocateCheckList");
        check_list = Tensor1<Int,Int> ( n, 0 );
    }

    void IncrementChecklist( const Int node ) const
    {
        if( node >= check_list.Size() )
        {
            eprint(ClassName()+"::IncrementChecklist: check_list is too short (size  = " + ToString(check_list.Size()) + ").");
        }
        else
        {
            ++check_list[node];
        }
    }

    bool PrintCheckList() const
    {
        std::vector<Int> not_visited;
        std::vector<Int> multiply_visited;
        for( Int node = 0; node < check_list.Size()-1; ++node )
        {
            if( check_list[node] < 1 )
            {
                not_visited.push_back(node);
            }
            
            if( check_list[node] > 1 )
            {
                multiply_visited.push_back(node);
            }
        }
        
        if( not_visited.size() > 0 )
        {
            eprint(ClassName()+"::PrintCheckList: There are are " + ToString(not_visited.size()) + " unvisited nodes.");
            
            dump(not_visited);
        }
        
        if( multiply_visited.size() > 0 )
        {
            eprint(ClassName()+"::PrintCheckList: There are are " + ToString(multiply_visited.size()) + " multiply visited nodes.");
            
            dump(multiply_visited);
        }
        
        return (not_visited.size() == Size_T(0)) && (multiply_visited.size() == Size_T(0));
    }

    Tensor1<Int,Int> & CheckList() const
    {
        return check_list;
    }






bool Traverse_Postordered_Test() const
{
    ptic(ClassName()+"::Traverse_Postordered_Test");
    AllocateCheckList();

    std::vector<std::unique_ptr<DebugWorker>> workers (thread_count );
    
    ParallelDo(
        [this,&workers]( const Int thread )
        {
            workers[thread] = std::make_unique<DebugWorker>( *this );
        },
        thread_count
    );
    
    Traverse_Postordered( workers );
    
    bool succeededQ = PrintCheckList();
    
    if( succeededQ )
    {
        print(ClassName()+"::Traverse_Postordered_Test succeeded.");
        logprint(ClassName()+"::Traverse_Postordered_Test succeeded.");
    }
    else
    {
        eprint(ClassName()+"::Traverse_Postordered_Test failed.");
    }
    
    ptoc(ClassName()+"::Traverse_Postordered_Test");
    
    return succeededQ;
}



template<Parallel_T parQ = Parallel, class Worker_T>
void Traverse_Preordered( std::vector<std::unique_ptr<Worker_T>> & workers ) const
{
    std::string tag = ClassName() + "::Traverse_Preordered<" + (parQ == Parallel ? "Parallel" : "Sequential") + ">";
    if( !PostOrderedQ() )
    {
        eprint(tag+" requires postordered tree! Doing nothing.");
        return;
    }
    
    ptic(tag);
    
    std::string tag_1 = "Apply worker " + workers[0]->ClassName() + " to level";
    
    const Int target_split_level = static_cast<Int>(tree_top_levels.size()-1);

    for( Int d = Scalar::One<Int>; d < static_cast<Int>(tree_top_levels.size()); ++d )
    {
        const Int k_begin = 0;
        const Int k_end   = static_cast<Int>(tree_top_levels[d].size());

        const Int use_threads = parQ == Parallel ? Min( thread_count, k_end - k_begin ) : one;

        ptic(tag_1 + " = " + ToString(d) + "; using " + ToString(use_threads) + " threads.");
        
        ParallelDo_Dynamic(
            [=,this,&workers]( const Int thread, const Int k )
            {
                const Time start_time = Clock::now();

                Worker_T & worker = *workers[thread];

                const Int node = tree_top_levels[d][k];

                worker( node );

                const Time stop_time = Clock::now();
                logprint(
                    tag + ": Worker " + ToString(thread) + " required " +
                         ToString(Tools::Duration(start_time,stop_time)) +
                        " s for completing node " + ToString(node) + " and its direct children."
                );
            },
            k_begin, k_end, Scalar::One<Int>, use_threads
        );


        ptoc(tag_1 + " = "+ToString(d)+"; using " + ToString(use_threads) + " threads.");

    } // for( Int d = target_split_level; d --> Scalar::One<Int> ; )


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

                Worker_T & worker = *workers[thread];

                const Int node = subtrees[k];

                Traverse_Descendants_Preordered( worker, node );

                const Time stop_time = Clock::now();
                
                logprint(
                    tag + ": Worker " + ToString(thread) + " required " +
                         ToString(Tools::Duration(start_time,stop_time)) +
                        " s for the " + ToString(DescendantCount(node)) + " descendants of node " + ToString(node) + "."
                );
            },
            k_begin, k_end, Scalar::One<Int>, use_threads
        );
        
        ptoc(tag_1 + " <= "+ToString(target_split_level)+"; using " + ToString(use_threads) + " threads.");
    }
    ptoc(tag);
}
