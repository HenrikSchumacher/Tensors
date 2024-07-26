public:

// TODO: Overloads for UpperSolve

// TODO: Overloads for LowerSolve

//####################################################################################
//####          Solve
//####################################################################################


    // The most general Solve routine that is exposed to the user.
    // It loads the inputs, runs the actual solver, and writes the outputs.

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id, typename a_T, typename B_T, typename b_T, typename X_T
    >
    void Solve(
        cref<a_T> alpha, cptr<B_T> B,     const Int ldB,
        cref<b_T> beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName() + "::Solve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                ptoc(tag);
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
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

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );

        ptoc(tag);
    }

    // Evaluate X = op(A)^{-1} . B
    template<
        Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id,
        typename B_T, typename X_T
    >
    void Solve(
        cptr<B_T> B,
        mptr<X_T> X_out,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        Solve<NRHS,parQ,op>(
            Scalar::One <X_T>, B,     nrhs_,
            Scalar::Zero<X_T>, X_out, nrhs_,
            nrhs_
        );
    }

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id, typename a_T, typename B_T, typename b_T, typename X_T,
        typename B_I, typename X_I
    >
    void Solve(
        cref<a_T> alpha, cref<Tensor2<B_T,B_I>> B,
        cref<b_T> beta,  mref<Tensor2<X_T,X_I>> X_out
    )
    {
        
        if ( B.Dimension(0) < RowCount() )
        {
            eprint( ClassName() + "::Solve: B.Dimension(0) < RowCount(). Aborting");
            return;
        }
        
        if ( X_out.Dimension(0) < RowCount() )
        {
            eprint( ClassName() + "::Solve: X.Dimension(0) < RowCount(). Aborting");
            return;
        }
        
        if ( B.Dimension(1) != X_out.Dimension(1) )
        {
            eprint( ClassName() + "::Solve: B.Dimension(1) != B.Dimension(1).");
            return;
        }
        
        const Int ldB = int_cast<Int>(B.Dimension(1));
        const Int ldX = int_cast<Int>(X_out.Dimension(1));
        
        Solve<NRHS,parQ,op>(
            alpha, B.data(),     ldB,
            beta,  X_out.data(), ldX,
            ldX
        );
    }

    // Evaluate X = alpha * op(A)^{-1} . B + beta * X
    template<Parallel_T parQ = Sequential, Op op = Op::Id, typename a_T, typename B_T, typename b_T, typename X_T,
        typename B_I, typename X_I
    >
    void Solve(
        cref<a_T> alpha, cref<Tensor1<B_T,B_I>> B,
        cref<b_T> beta,  mref<Tensor1<X_T,X_I>> X_out
    )
    {
        
        if ( B.Dimension(0) < RowCount() )
        {
            eprint( ClassName() + "::Solve: B.Dimension(0) < RowCount(). Aborting");
            return;
        }
        
        if ( X_out.Dimension(0) < RowCount() )
        {
            eprint( ClassName() + "::Solve: X.Dimension(0) < RowCount(). Aborting");
            return;
        }
        
        constexpr Int NRHS = 1;
        
        Solve<NRHS,parQ,op>(
            alpha, B.data(),     NRHS,
            beta,  X_out.data(), NRHS
        );
    }


//####################################################################################
//####          UpperSolve
//####################################################################################

    // Evaluate X = alpha * op(U)^{-1} . B + beta * X
    template<Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id, typename a_T, typename B_T, typename b_T, typename X_T
    >
    void UpperSolve(
        cref<a_T> alpha, cptr<B_T> B,     const Int ldB,
        cref<b_T> beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ( (NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName() + "::UpperSolve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "UpperSolve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                ptoc(tag);
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
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

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );

        ptoc(tag);
    }

//####################################################################################
//####          LowerSolve
//####################################################################################

    // Evaluate X = alpha * op(L)^{-1} . B + beta * X
    template<Size_T NRHS = VarSize, Parallel_T parQ = Sequential, Op op = Op::Id, typename a_T, typename B_T, typename b_T, typename X_T
    >
    void LowerSolve(
        cref<a_T> alpha, cptr<B_T> B,     const Int ldB,
        cref<b_T> beta,  mptr<X_T> X_out, const Int ldX,
        const Int nrhs_ = ((NRHS > VarSize) ? NRHS : ione )
    )
    {
        const std::string tag = ClassName() + "::LowerSolve"
            + "<" + ToString(VarSize)
            + "," + ToString(parQ)
            + "," + ToString(op)
            + "," + TypeName<a_T>
            + "," + TypeName<B_T>
            + "," + TypeName<b_T>
            + "," + TypeName<X_T>
            + "> ( " + ToString(ldB) + "," + ToString(ldX) + "," + ToString(nrhs_) + " )";
        
        static_assert(
            Scalar::RealQ<Scal> || op != Op::Trans,
            "LowerSolve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        
        // No problem if X_out and B overlap, since we load B into X anyways.
        
        if constexpr ( NRHS > VarSize )
        {
            if( nrhs_ != NRHS )
            {
                eprint( tag + ": (NRHS > VarSize) && (nrhs_ != NRHS). Aborting.");
                ptoc(tag);
                return;
            }
        }
        
        ReadRightHandSide<NRHS>( B, ldB, nrhs_ );
        
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

        WriteSolution<NRHS>( alpha, beta, X_out, ldX );

        ptoc(tag);
    }

//####################################################################################
//####          Supernodal back substitution, both parallel and sequential
//####################################################################################

protected:

    // These are the internal solve commands.


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
        std::vector<std::unique_ptr<Solver_T>> F_list ( 
            int_cast<Size_T>(use_threads)
        );
        
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
