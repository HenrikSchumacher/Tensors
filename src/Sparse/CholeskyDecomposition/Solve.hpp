#pragma once

//###########################################################################################
//####          Public interface for solve routines
//###########################################################################################
    
public:
    
    
    template<Op op = Op::Id, typename ExtScal>
    void Solve( ptr<ExtScal> b, mut<ExtScal> x )
    {
        ptic(ClassName()+"::Solve");

        static_assert(
            (!Scalar::IsComplex<Scal>) || op != Op::Trans,
            "Solve with Op::Trans not implemented for scalar of complex type."
        );
        
        // No problem if x and b overlap, since we load b into X anyways.
        ReadRightHandSide(b);
        
        if constexpr ( op == Op::Id )
        {
            SN_LowerSolve_Sequential();
            SN_UpperSolve_Sequential();
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            SN_UpperSolve_Sequential();
            SN_LowerSolve_Sequential();
        }

        WriteSolution(x);

        ptoc(ClassName()+"::Solve");
    }
    
    template<Op op = Op::Id, typename ExtScal>
    void Solve( ptr<ExtScal> B, mut<ExtScal> X_, const Int nrhs )
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
            SN_LowerSolve_Sequential( nrhs );
            SN_UpperSolve_Sequential( nrhs );
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            SN_UpperSolve_Sequential( nrhs );
            SN_LowerSolve_Sequential( nrhs );
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

    void SN_UpperSolve_Sequential( const Int nrhs )
    {
        const std::string tag = ClassName() + "SN_UpperSolve_Sequential (" + ToString(nrhs)+ ")";
        
        ptic(tag);
        // Solves U * X = B and stores the result back into B.
        // Assumes that B has size n x rhs_count.
        
        if( nrhs == ione )
        {
            SN_UpperSolve_Sequential();
            ptoc(tag);
            return;
        }
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_UpperSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        for( Int sn = SN_count; sn --> 0; )
        {
            const Int n_0 = SN_rp[sn+1] - SN_rp[sn];
        
            assert_positive(n_0);
            
            const LInt l_begin = SN_outer[sn  ];
            const LInt l_end   = SN_outer[sn+1];
            
            const Int n_1 = int_cast<Int>(l_end - l_begin);
            
            // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
            ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[sn]];
            
            // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
            ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[sn]];
            
            // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
            mut<Scal> X_0 = &X[nrhs * SN_rp[sn]];
            
            // X_1 is the part of X that interacts with U_1, size = n_1 x rhs_count.
            mut<Scal> X_1 = X_scratch.data();
            
            // Load the already computed values into X_1.
            for( Int j = 0; j < n_1; ++j )
            {
                copy_buffer( &X[nrhs * SN_inner[l_begin+j]], &X_1[nrhs * j], nrhs );
            }

            if( n_0 == ione )
            {
                if( n_1 > izero )
                {
                    // Compute X_0 -= U_1 * X_1

                    //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                    //  X_1 is a matrix of size n_1 x nrhs.
                    //  X_0 is a matrix of size 1 x nrhs; we can interpret it as vector of size nrhs.

                    // Hence we can compute X_0^T -= X_1^T * U_1^T via gemv instead:
                    BLAS::gemv<Layout::RowMajor,Op::Trans>(
                        n_1, nrhs,
                        -one, X_1, nrhs,
                              U_1, 1,        // XXX Problem: We need Scalar::Conj(U_1)!
                         one, X_0, 1
                    );
                }

                // Triangle solve U_0 * X_0 = B while overwriting X_0.
                // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                scale_buffer( Scalar::Inv<Scal>(U_0[0]), X_0, nrhs );
            }
            else // using BLAS3 routines.
            {
                if( n_1 > izero )
                {
                    // Compute X_0 -= U_1 * X_1
                    BLAS::gemm<Layout::RowMajor,Op::Id,Op::Id>(
                        // XX Op::Id -> Op::ConjugateTranspose
                        n_0, nrhs, n_1,
                        -one, U_1, n_1,      // XXX n_1 -> n_0
                              X_1, nrhs,
                         one, X_0, nrhs
                    );
                }
                // Triangle solve U_0 * X_0 = B while overwriting X_0.
                BLAS::trsm<Layout::RowMajor,
                    Side::Left, UpLo::Upper, Op::Id, Diag::NonUnit
                >(
                    n_0, nrhs,
                    one, U_0, n_0,
                         X_0, nrhs
                );
            }
        }
        
        ptoc(tag);
    }
    
    void SN_UpperSolve_Sequential()
    {
        // Solves U * X = X and stores the result back into X.
        // Assumes that X has size n.
        
        const std::string tag = ClassName() + "SN_UpperSolve_Sequential";
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_UpperSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        for( Int sn = SN_count; sn --> 0; )
        {
            const Int n_0 = SN_rp[sn+1] - SN_rp[sn];
            
            const LInt l_begin = SN_outer[sn  ];
            const LInt l_end   = SN_outer[sn+1];

            const Int n_1 = int_cast<Int>(l_end - l_begin);

            // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
            ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[sn]];

            // U_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
            ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[sn]];

            // x_0 is the part of x that interacts with U_0, size = n_0.
            mut<Scal> x_0 = &X[SN_rp[sn]];


            if( n_0 == one )
            {
                Scal U_1x_1 = 0;

                if( n_1 > izero )
                {
                    // Compute X_0 -= U_1 * X_1
                    //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                    //  x_1 is a vector of size n_1.
                    //  x_0 is a matrix of size 1 x 1; we can interpret it as vector of size 1.

                    // Hence we can compute X_0 -= U_1 * X_1 via a simple dot product.

                    for( Int j = 0; j < n_1; ++j )
                    {
                        U_1x_1 += U_1[j] * X[SN_inner[l_begin+j]]; // XXX Scalar::Conj(U_1[j])
                    }
                }

                // Triangle solve U_0 * X_0 = B while overwriting X_0.
                // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                x_0[0] = (x_0[0] - U_1x_1) / U_0[0];
            }
            else // using BLAS2 routines.
            {
                if( n_1 > izero )
                {
                    // x_1 is the part of x that interacts with U_1, size = n_1.
                    mut<Scal> x_1 = X_scratch.data();

                    // Load the already computed values into x_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        x_1[j] = X[SN_inner[l_begin+j]];
                    }

                    // Compute x_0 -= U_1 * x_1
                    BLAS::gemv<Layout::RowMajor,Op::Id>(// XXX Op::Id -> Op::ConjTrans
                        n_0, n_1,
                        -one, U_1, n_1, // XXX n_1 -> n_0
                              x_1, 1,
                         one, x_0, 1
                    );
                }

                // Triangle solve U_0 * x_0 = B while overwriting x_0.
                BLAS::trsv<Layout::RowMajor,UpLo::Upper,Op::Id,Diag::NonUnit>(
                    n_0, U_0, n_0, x_0, 1
                );
            }
        }
        
        ptoc(tag);
    }
    
    void SN_LowerSolve_Sequential( const Int nrhs )
    {
        const std::string tag = ClassName() + "SN_LowerSolve_Sequential (" + ToString(nrhs) + ")";
        
        ptic(tag);
        // Solves L * X = X and stores the result back into X.
        // Assumes that X has size n x rhs_count.
     
        if( nrhs == ione )
        {
            SN_LowerSolve_Sequential();
            ptoc(tag);
            return;
        }
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_LowerSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            
            ptoc(tag);
            return;
        }
        
        for( Int sn = 0; sn < SN_count; ++sn )
        {
            const Int n_0 = SN_rp[sn+1] - SN_rp[sn];
            
            assert_positive(n_0);
            
            const LInt l_begin = SN_outer[sn  ];
            const LInt l_end   = SN_outer[sn+1];
            
            const Int n_1 = int_cast<Int>(l_end - l_begin);
            
            // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
            ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[sn]];
            
            // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
            ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[sn]];
            
            // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
            mut<Scal> X_0 = &X[nrhs * SN_rp[sn]];
            
            // X_1 is the part of X that interacts with U_1, size = n_1 x rhs_count.
            mut<Scal> X_1 = X_scratch.data();

            if( n_0 == ione )
            {
                // Triangle solve U_0 * X_0 = B while overwriting X_0.
                // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                scale_buffer( Scalar::Inv<Scal>(U_0[0]), X_0, nrhs );
                if( n_1 > izero )
                {
                    // Compute X_1 = - U_1^H * X_0

                    //  U_1 is a matrix of size 1   x n_1.
                    //  X_1 is a matrix of size n_1 x nrhs.
                    //  X_0 is a matrix of size 1   x nrhs.

                    for( LInt i = 0; i < int_cast<LInt>(n_1); ++i )
                    {
                        const Scal factor = - Scalar::Conj(U_1[i]); // XXX Scalar::Conj(U_1[i])-> U_1[i]
                        for( LInt j = 0; j < int_cast<LInt>(nrhs); ++j )
                        {
                            X_1[nrhs*i+j] = factor * X_0[j];
                        }
                    }
                }
            }
            else // using BLAS3 routines.
            {
                // Triangle solve U_0^H * X_0 = B_0 while overwriting X_0.
                BLAS::trsm<
                    Layout::RowMajor, Side::Left,
                    UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                >(
                    n_0, nrhs,
                    one, U_0, n_0,
                         X_0, nrhs
                );
                
                if( n_1 > izero )
                {
                    // Compute X_1 = - U_1^H * X_0
                    BLAS::gemm<Layout::RowMajor, Op::ConjTrans, Op::Id>(
                       //XXX Op::ConjTrans -> Op::Id?
                        n_1, nrhs, n_0, // ???
                        -one, U_1, n_1, // n_1 -> n_0
                              X_0, nrhs,
                        zero, X_1, nrhs
                    );
                }
            }

            // Add X_1 into B_1
            for( Int j = 0; j < n_1; ++j )
            {
                add_to_buffer( &X_1[nrhs * j], &X[nrhs * SN_inner[l_begin+j]], nrhs );
            }
        }
        ptoc(tag);
    }
    
    
    void SN_LowerSolve_Sequential()
    {
        const std::string tag = ClassName() + "SN_LowerSolve_Sequential";
        // Solves L * x = X and stores the result back into X.
        // Assumes that X has size n.
        
        ptic(tag);
        
        if( !SN_factorized )
        {
            eprint(ClassName()+"::SN_LowerSolve_Sequential: Nonzero values of matrix have not been passed, yet. Aborting.");
            return;
        }
        
        for( Int sn = 0; sn < SN_count; ++sn )
        {
            const Int n_0 = SN_rp[sn+1] - SN_rp[sn];
            
            const LInt l_begin = SN_outer[sn  ];
            const LInt l_end   = SN_outer[sn+1];

            const Int n_1 = int_cast<Int>(l_end - l_begin);

            // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
            ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[sn]];

            // U_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
            ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[sn]];

            // x_0 is the part of x that interacts with U_0, size = n_0.
            mut<Scal> x_0 = &X[SN_rp[sn]];
            
            if( n_0 == ione )
            {
                // Triangle solve U_0 * x_0 = b_0 while overwriting x_0.
                // Since U_0 is a 1 x 1 matrix, it suffices to just scale x_0.
                x_0[0] /= U_0[0];

                if( n_1 > izero )
                {
                    // Compute x_1 = - U_1^H * x_0
                    // x_1 is a vector of size n_1.
                    // U_1 is a matrix of size 1 x n_1
                    // x_0 is a vector of size 1.
                    
                    // Add x_1 into b_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        X[SN_inner[l_begin+j]] -= Scalar::Conj(U_1[j]) * x_0[0];
                    }   // XXX Scalar::Conj(U_1[j]) -> U_1[j]
                }
            }
            else // using BLAS2 routines.
            {
                // Triangle solve U_0^H * x_0 = b_0 while overwriting x_0.
                BLAS::trsv<
                    Layout::RowMajor, UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                >( n_0, U_0, n_0, x_0, 1 );
                
                if( n_1 > izero )
                {
                    // x_1 is the part of x that interacts with U_1, size = n_1.
                    mut<Scal> x_1 = X_scratch.data();
                    
                    // Compute x_1 = - U_1^H * x_0
                    BLAS::gemv<Layout::RowMajor, Op::ConjTrans>(
                        n_0, n_1,             // XXX Op::ConjTrans -> Op::Trans
                        -one, U_1, n_1, // XXX n_1 -> n_0
                                            x_0, 1,
                        zero, x_1, 1
                    );
                    
                    // Add x_1 into b_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        X[SN_inner[l_begin+j]] += x_1[j];
                    }
                }
            }
        }
        
        ptoc(tag);
    }

//###########################################################################################
//####          Supernodal back substitution, parallel
//###########################################################################################

protected:

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
                SN_LowerSolve_Sequential();
//                SN_UpperSolve_Sequential();

//                SN_LowerSolve_Parallel<false>( nrhs );
                SN_UpperSolve_Parallel<false>( nrhs );
            }
            else
            {
                SN_LowerSolve_Sequential( nrhs );
//                SN_UpperSolve_Sequential( nrhs );

//                SN_LowerSolve_Parallel<true>( nrhs );
                SN_UpperSolve_Parallel<true>( nrhs );
            }
        }
        else if constexpr ( op == Op::ConjTrans )
        {
            if ( nrhs == ione )
            {
//                SN_UpperSolve_Sequential();
                SN_LowerSolve_Sequential();
            
                SN_UpperSolve_Parallel<false>( nrhs );
//                SN_LowerSolve_Parallel<false>( nrhs );
            }
            else
            {
//                SN_UpperSolve_Sequential( nrhs );
                SN_LowerSolve_Sequential( nrhs );
            
                SN_UpperSolve_Parallel<true>( nrhs );
//                SN_LowerSolve_Parallel<true>( nrhs );
            }
            
        }
        
        WriteSolution( X_, nrhs );
        
        ptoc(tag);
    }
