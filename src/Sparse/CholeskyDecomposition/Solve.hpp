#pragma once

//###########################################################################################
//####          Public interface for solve routines
//###########################################################################################
    
public:
    
    template<Op op = Op::Id, typename ExtScal>
    void Solve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs = ione )
    {
        std::string tag = ClassName()+"::Solve ("+ToString(nrhs)+")";
        static_assert(
            Scalar::IsReal<Scal> || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        ptic(tag);
        // No problem if X_ and B overlap, since we load B into X anyways.
        
        ReadRightHandSide( B, nrhs );
        
        if constexpr ( op == Op::Id )
        {
            if( nrhs == ione )
            {
                SN_LowerSolve_Sequential<false>( nrhs );
                SN_UpperSolve_Sequential<false>( nrhs );
            }
            else
            {
                SN_LowerSolve_Sequential<true>( nrhs );
                SN_UpperSolve_Sequential<true>( nrhs );
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            if( nrhs == ione )
            {
                SN_UpperSolve_Sequential<false>( nrhs );
                SN_LowerSolve_Sequential<false>( nrhs );
            }
            else
            {
                SN_UpperSolve_Sequential<true>( nrhs );
                SN_LowerSolve_Sequential<true>( nrhs );
            }
        }
        
        WriteSolution( X_, nrhs );
        
        ptoc(tag);
    }
    

//    template<typename ExtScal>
//    void UpperSolve( ptr<ExtScal> b, mut<ExtScal> x )
//    {
//        // No problem if x and b overlap, since we load b into X anyways.
//        ReadRightHandSide(b);
//
//        SN_UpperSolve_Sequential();
//
//        WriteSolution(x);
//    }
//
//    template<typename ExtScal>
//    void UpperSolve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs )
//    {
//        // No problem if X_ and B overlap, since we load B into X anyways.
//        ReadRightHandSide( B, nrhs );
//
//        SN_UpperSolve_Sequential( nrhs );
//
//        WriteSolution( X_, nrhs );
//    }
//
//    template<typename ExtScal>
//    void LowerSolve( ptr<ExtScal> b, mut<ExtScal> x )
//    {
//        // No problem if x and b overlap, since we load b into X anyways.
//        ReadRightHandSide(b);
//
//        SN_LowerSolve_Sequential();
////                SN_LowerSolve_Parallel();
//
//        WriteSolution(x);
//    }
//
//    template<typename ExtScal>
//    void LowerSolve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs )
//    {
//        // No problem if X_ and B overlap, since we load B into X anyways.
//        ReadRightHandSide( B, nrhs );
//
//        SN_LowerSolve_Sequential( nrhs );
////                SN_LowerSolve_Parallel(nrhs);
//
//        WriteSolution( X_, nrhs );
//    }

//###########################################################################################
//####          Supernodal back substitution, sequential
//###########################################################################################

protected:

    template<bool mult_rhs>
    void SN_UpperSolve_Sequential( const Int nrhs )
    {
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        const std::string tag = mult_rhs
            ?
            ClassName() + "SN_UpperSolve_Sequential (" + ToString(nrhs)+ ")"
            :
            ClassName() + "SN_UpperSolve_Sequential";
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_UpperSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        UpperSolver<mult_rhs,Scal,Int,LInt> worker ( *this, nrhs );
        
        for( Int s = SN_count; s --> 0; )
        {
            worker(s);
        }
        
        ptoc(tag);
    }

    template<bool mult_rhs>
    void SN_LowerSolve_Sequential( const Int nrhs )
    {
        // Solves L * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        const std::string tag = mult_rhs
            ?
            ClassName() + "SN_LowerSolve_Sequential (" + ToString(nrhs)+ ")"
            :
            ClassName() + "SN_LowerSolve_Sequential";
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_LowerSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        LowerSolver<mult_rhs,Scal,Int,LInt> worker ( *this, nrhs );
        
        for( Int s = 0; s < SN_count; ++s )
        {
            worker(s);
        }
        
        ptoc(tag);
    }
