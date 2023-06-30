//###########################################################################################
//####          Traversal_Preorder
//###########################################################################################

public:

template<class Worker_T>
void Traverse_Descendants_Preordered( Worker_T & worker, const Int node ) const
{
    debug_print(ClassName() + "::Traverse_Descendants_Preordered ( node = " + ToString(node) + " ) begins.");

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
    
    debug_print(ClassName() + "::Traverse_Descendants_Preordered ( node = " + ToString(node) + " ) ends.");
}

//template<class Worker_T>
//void Traverse_Children_Preordered( Worker_T & worker, const Int node ) const
//{
//    debug_print(ClassName() + "::Traverse_Children_Preordered ( node = " + ToString(node) + " ) begins.");
//
//    // Applies ker to the direct children of the node in postorder.
//    // CAUTION: Does not apply ker to the node itself!
//    // This is to guarantee that all children of node are processed on the same thread to avoid write-conflicts in the case they attempt to write to some of their common parent's memory.
//
//
//    const Int child_begin =  ChildPointer(node     );
//    const Int child_end   =  ChildPointer(node + 1 );
//
//    for( Int k = child_end; k --> child_begin; )
//    {
//        const Int child = ChildIndex(k);
//
//        debug_assert(
//            parents[child] == node,
//            "Node " + ToString(child) + " is not the child of node " + ToString(node)+ "."
//        );
//
//        worker(child);
//    }
//
//    debug_print(ClassName() + "::Traverse_Children_Preordered ( node = " + ToString(node) + " ) ends.");
//}


bool Traverse_Preordered_Test() const
{
    ptic(ClassName()+"::Traverse_Preordered_Test");
    AllocateCheckList();

    std::vector<std::unique_ptr<DebugWorker>> workers (thread_count );
    
    ParallelDo(
        [this,&workers]( const Int thread )
        {
            workers[thread] = std::make_unique<DebugWorker>( *this );
        },
        thread_count
    );
    
    Traverse_Preordered( workers );
    
    bool succeededQ = PrintCheckList();
    
    if( succeededQ )
    {
        print(ClassName()+"::Traverse_Preordered_Test succeeded.");
        logprint(ClassName()+"::Traverse_Preordered_Test succeeded.");
    }
    else
    {
        eprint(ClassName()+"::Traverse_Preordered_Test failed.");
    }
    
    ptoc(ClassName()+"::Traverse_Preordered_Test");
    
    return succeededQ;
}
