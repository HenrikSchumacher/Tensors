#pragma once

//###########################################################################################
//####          Public interface for parallel solve routines
//###########################################################################################

public:

    template<Op op = Op::Id, typename ExtScal>
    void SolveParallel( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs = ione )
    {
        std::string tag = ClassName()+"::SolveParallel ("+ToString(nrhs)+")";
        static_assert(
            Scalar::IsReal<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, nrhs );
        
        // TODO: The `LowerSolver` does not seem to work correctly. Why is that?
        
        if constexpr ( op == Op::Id )
        {
            if ( nrhs == ione )
            {
//                SN_LowerSolve_Sequential<false>( nrhs );
//                SN_UpperSolve_Sequential<false>( nrhs );

                SN_LowerSolve_Parallel<false>( nrhs );
                SN_UpperSolve_Parallel<false>( nrhs );
            }
            else
            {
//                SN_LowerSolve_Sequential<true>( nrhs );
//                SN_UpperSolve_Sequential<true>( nrhs );

                SN_LowerSolve_Parallel<true>( nrhs );
                SN_UpperSolve_Parallel<true>( nrhs );
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            if ( nrhs == ione )
            {
//                SN_UpperSolve_Sequential<false>( nrhs );
//                SN_LowerSolve_Sequential<false>( nrhs );
            
                SN_UpperSolve_Parallel<false>( nrhs );
                SN_LowerSolve_Parallel<false>( nrhs );
            }
            else
            {
//                SN_UpperSolve_Sequential<true>( nrhs );
//                SN_LowerSolve_Sequential<true>( nrhs );
            
                SN_UpperSolve_Parallel<true>( nrhs );
                SN_LowerSolve_Parallel<true>( nrhs );
            }
            
        }
        
        WriteSolution( X_, nrhs );
        
        ptoc(tag);
    }
//###########################################################################################
//####          Supernodal back substitution, parallel
//###########################################################################################

protected:

    template<bool mult_rhs>
    void SN_UpperSolve_Parallel( const Int nrhs )
    {
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        using Solver_T = UpperSolver<mult_rhs,Scal,Int,LInt>;
        
        const std::string tag = mult_rhs
            ?
            ClassName() + "SN_UpperSolve_Parallel (" + ToString(nrhs)+ ")"
            :
            ClassName() + "SN_UpperSolve_Parallel";
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_UpperSolve_Parallel: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list (thread_count);
        
        ParallelDo(
            [&F_list,this,nrhs]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in preorder
        
        aTree.Traverse_Preorder_Parallel( F_list, tree_top_depth );
        
        ptoc(tag);
    }

    template<bool mult_rhs>
    void SN_LowerSolve_Parallel( const Int nrhs )
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        using Solver_T = LowerSolver<mult_rhs,Scal,Int,LInt>;
        
        const std::string tag = mult_rhs
            ?
            ClassName() + "SN_LowerSolve_Parallel (" + ToString(nrhs)+ ")"
            :
            ClassName() + "SN_LowerSolve_Parallel";
        
        ptic(tag);

        
        if( !SN_factorized )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list (thread_count);
        
        ParallelDo(
            [&F_list,this,nrhs]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in postorder
        aTree.Traverse_Postorder_Parallel( F_list, tree_top_depth );
        
        ptoc(tag);
    }
