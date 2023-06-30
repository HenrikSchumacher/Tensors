public:

    class DebugWorker
    {
    public:
     
        const Tree<Int> & tree = nullptr;
        
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
