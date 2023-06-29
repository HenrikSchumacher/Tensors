#pragma once

//###########################################################################################
//####          Public interface for solve routines
//###########################################################################################
    
public:
    
    template<Parallel_T parQ = Sequential, Op op = Op::Id, typename ExtScal>
    void Solve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs_ = ione )
    {
        const std::string tag = ClassName() + "::Solve<"
            + (parQ == Parallel ? "par" : "seq") + ","
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
            SN_Solve<false,parQ,op>();
        }
        else
        {
            SN_Solve<true, parQ,op>();
        }
        
        WriteSolution( X_ );
        
        ptoc(tag);
    }


//###########################################################################################
//####          Supernodal back substitution, both parallel and sequential
//###########################################################################################

protected:


    template<bool mult_rhsQ, Parallel_T parQ = Sequential, Op op = Op::Id>
    void SN_Solve()
    {
        if constexpr ( op == Op::Id )
        {
            SN_LowerSolve<mult_rhsQ,parQ>();
            SN_UpperSolve<mult_rhsQ,parQ>();
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            SN_UpperSolve<mult_rhsQ,parQ>();
            SN_LowerSolve<mult_rhsQ,parQ>();
        }
    }


    template<bool mult_rhsQ, Parallel_T parQ = Sequential>
    void SN_UpperSolve()
    {
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        using Solver_T = UpperSolver<mult_rhsQ,Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_UpperSolve<" + ToString(mult_rhsQ) + "," + (parQ == Parallel ? "par" : "seq") + "> ( " + ToString(nrhs)+ " )";
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list (thread_count);
        
        Do<VarSize,parQ>(
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count, thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in preorder
        
        aTree.template Traverse_Preordered<parQ>( F_list, tree_top_depth );
        
        ptoc(tag);
    }

    template<bool mult_rhsQ, Parallel_T parQ = Sequential>
    void SN_LowerSolve()
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
//        using Solver_T = LowerSolver<mult_rhsQ,COND(parQ == Parallel,true,false),Scal,Int,LInt>;
        using Solver_T = LowerSolver<mult_rhsQ,false,Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_LowerSolve<" + ToString(mult_rhsQ) + "," + (parQ == Parallel ? "par" : "seq") + "> ( " + ToString(nrhs)+ " )";
        
        ptic(tag);

        
        if( !SN_factorized )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list (thread_count);
        
        Do<VarSize,parQ>(
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            thread_count, thread_count
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<parQ>( F_list, tree_top_depth );
        
        ptoc(tag);
    }
