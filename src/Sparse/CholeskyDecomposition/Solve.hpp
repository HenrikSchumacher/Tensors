#pragma once

//###########################################################################################
//####          Public interface for solve routines
//###########################################################################################
    
public:
    
    template<bool parallelQ = false, Op op = Op::Id, typename ExtScal>
    void Solve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs_ = ione )
    {
        const std::string tag = ClassName() + "::Solve<"
            + (parallelQ ? "par" : "seq") + ","
            + ( op==Op::Id ? "N" : (op==Op::Trans ? "T" : (op==Op::ConjTrans ? "H" : "N/A" ) ) ) + ","
            + TypeName<ExtScal>
            + "> ( " + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, nrhs_ );
        
        // TODO: The `LowerSolver` does not seem to work correctly. Why is that?
        
        if( nrhs == ione )
        {
            SN_Solve<false,parallelQ,op>();
        }
        else
        {
            SN_Solve<true, parallelQ,op>();
        }
        
        WriteSolution( X_ );
        
        ptoc(tag);
    }


//###########################################################################################
//####          Supernodal back substitution, both parallel and sequential
//###########################################################################################

protected:


    template<bool mult_rhsQ, bool parallelQ, Op op = Op::Id>
    void SN_Solve()
    {
        if constexpr ( op == Op::Id )
        {
            SN_LowerSolve<mult_rhsQ,parallelQ>();
            SN_UpperSolve<mult_rhsQ,parallelQ>();
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            SN_UpperSolve<mult_rhsQ,parallelQ>();
            SN_LowerSolve<mult_rhsQ,parallelQ>();
        }
    }


    template<bool mult_rhsQ, bool parallelQ>
    void SN_UpperSolve()
    {
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        using Solver_T = UpperSolver<mult_rhsQ,Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_UpperSolve<" + ToString(mult_rhsQ) + "," + (parallelQ ? "par" : "seq") + "> ( " + ToString(nrhs)+ " )";
        
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
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in preorder
        
        aTree.template Traverse_Preordered<parallelQ>( F_list, tree_top_depth );
        
        ptoc(tag);
    }

    template<bool mult_rhsQ, bool parallelQ>
    void SN_LowerSolve()
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        using Solver_T = LowerSolver<mult_rhsQ,Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_LowerSolve<" + ToString(mult_rhsQ) + "," + (parallelQ ? "par" : "seq") + "> ( " + ToString(nrhs)+ " )";
        
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
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<parallelQ>( F_list, tree_top_depth );
        
        ptoc(tag);
    }
