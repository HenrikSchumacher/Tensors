public:

    template<Parallel_T parQ = Sequential, Op op = Op::Id, typename ExtScal>
    void Solve( 
        cptr<ExtScal> B,     const Int ldB,
        mptr<ExtScal> X_out, const Int ldX,
        const Int nrhs_ = ione 
    )
    {
        const std::string tag = ClassName() + "::Solve<"
            + (parQ == Parallel ? "Parallel" : "Sequential") + ","
            + ( op==Op::Id ? "N" : (op==Op::Trans ? "T" : (op==Op::ConjTrans ? "H" : "N/A" ) ) ) + ","
            + TypeName<ExtScal>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > 1 )
            {
                SN_Solve<false,parQ,op>();
            }
            else
            {
                SN_Solve<false,Sequential,op>();
            }
        }
        else
        {
            if( thread_count > 1 )
            {
                SN_Solve<true,parQ,op>();
            }
            else
            {
                SN_Solve<true,Sequential,op>();
            }
        }
        
        WriteSolution( X_out, ldX );
        
        ptoc(tag);
    }

    template<Parallel_T parQ = Sequential, Op op = Op::Id, typename ExtScal>
    void Solve(
        cptr<ExtScal> B,
        mptr<ExtScal> X_out,
        const Int nrhs_ = ione
    )
    {
        Solve<parQ,op>( B, nrhs_, X_out, nrhs_, nrhs_ );
    }


    template<Parallel_T parQ = Sequential, typename ExtScal>
    void UpperSolve( 
        cptr<ExtScal> B,     const Int ldB,
        mptr<ExtScal> X_out, const Int ldX,
        const Int nrhs_ = ione )
    {
        const std::string tag = ClassName() + "::UpperSolve<"
            + (parQ == Parallel ? "Parallel" : "Sequential") + ","
            + TypeName<ExtScal>
            + "> ( " + ToString(nrhs_) + " )";

        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > 1 )
            {
                SN_UpperSolve<false,parQ>();
            }
            else
            {
                SN_UpperSolve<false,Sequential>();
            }
        }
        else
        {
            if( thread_count > 1 )
            {
                SN_UpperSolve<true,parQ>();
            }
            else
            {
                SN_UpperSolve<true,Sequential>();
            }
        }
        
        WriteSolution( X_out, ldX );
        
        ptoc(tag);
    }

    template<Parallel_T parQ = Sequential, typename ExtScal>
    void UpperSolve(
        cptr<ExtScal> B,
        mptr<ExtScal> X_out,
        const Int nrhs_ = ione 
    )
    {
        UpperSolve<parQ>( B, nrhs_, X_out, nrhs_, nrhs_ );
    }

    template<Parallel_T parQ = Sequential, typename ExtScal>
    void LowerSolve( 
        cptr<ExtScal> B,     const Int ldB,
        mptr<ExtScal> X_out, const Int ldX,
        const Int nrhs_ = ione
    )
    {
        const std::string tag = ClassName() + "::LowerSolve<"
            + (parQ == Parallel ? "Parallel" : "Sequential") + ","
            + TypeName<ExtScal>
            + "> ( " + ToString(nrhs_) + " )";
        
        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, ldB, nrhs_ );
        
        if( nrhs == ione )
        {
            if( thread_count > 1 )
            {
                SN_LowerSolve<false,parQ>();
            }
            else
            {
                SN_LowerSolve<false,Sequential>();
            }
        }
        else
        {
            if( thread_count > 1 )
            {
                SN_LowerSolve<true,parQ>();
            }
            else
            {
                SN_LowerSolve<true,Sequential>();
            }
        }
        
        WriteSolution( X_out, ldX );
        
        ptoc(tag);
    }

    template<Parallel_T parQ = Sequential, typename ExtScal>
    void LowerSolve(
        cptr<ExtScal> B,
        mptr<ExtScal> X_out,
        const Int nrhs_ = ione
    )
    {
        LowerSolve<parQ>( B, nrhs_, X_out, nrhs_, nrhs_ );
    }

//####################################################################################
//####          Supernodal back substitution, both parallel and sequential
//####################################################################################

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
        // Assumes that B has size n x nrhs.
        
        using Solver_T = UpperSolver<mult_rhsQ,Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_UpperSolve<" + ToString(mult_rhsQ) + "," + (parQ == Parallel ? "Parallel" : "Sequential") + "> ( " + ToString(nrhs)+ " )";
        
        ptic(tag);
        
        if( !NumericallyFactorizedQ() )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        const Int use_threads = ( parQ == Parallel) ? thread_count : 1;
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list ( use_threads );
        
        Do<VarSize,parQ>(
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            use_threads, use_threads
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in preorder
        aTree.template Traverse_Preordered<parQ>( F_list );
        
        ptoc(tag);
    }

    template<bool mult_rhsQ, Parallel_T parQ = Sequential>
    void SN_LowerSolve()
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x nrhs.
        
        // Use locks if run in parallel.
        using Solver_T = LowerSolver<mult_rhsQ,( parQ == Parallel ? true : false),Scal,Int,LInt>;
        
        const std::string tag = ClassName() + "::SN_LowerSolve<" + ToString(mult_rhsQ) + "," + (parQ == Parallel ? "Parallel" : "Sequential") + "> ( " + ToString(nrhs)+ " )";
        
        ptic(tag);
        
        if( !NumericallyFactorizedQ() )
        {
            eprint(tag+": Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        const Int use_threads = ( parQ == Parallel) ? thread_count : 1;
        
        ptic("Initialize solvers");
        std::vector<std::unique_ptr<Solver_T>> F_list ( use_threads );
        
        Do<VarSize,parQ>(
            [&F_list,this]( const Int thread )
            {
                F_list[thread] = std::make_unique<Solver_T>(*this, nrhs );
            },
            use_threads, use_threads
        );
        ptoc("Initialize solvers");
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<parQ>( F_list );
        
        ptoc(tag);
    }
