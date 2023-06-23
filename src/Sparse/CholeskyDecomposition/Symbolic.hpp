#pragma once

//###########################################################################################
//####          Supernodal symbolic factorization
//###########################################################################################
        
public:
    
    void SN_SymbolicFactorization()
    {
        // Compute supernodal symbolic factorization with so-called _fundamental supernodes_.
        // See Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        // https://www.osti.gov/servlets/purl/6756314
        
        // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

        if( !SN_initialized )
        {
            ptic(ClassName()+"::SN_SymbolicFactorization");

            (void)PostOrdering();

            // temporary arrays
            Tensor1<Int,Int> row (n);// An array to aggregate columnn indices of row of U.
            Tensor1<Int,Int> row_scratch (n); // Some scratch space for UniteSortedBuffers.
            Int n_1;  // Holds the current number of indices in row.

            // TODO: Fix this to work with singed integers.
            Tensor1<Int,Int> prev_col_nz(n,-1);

            // i-th row of U belongs to supernode row_to_SN[i].
            row_to_SN   = Tensor1< Int,Int> (n);
            
            // Holds the current number of supernodes.
            SN_count    = 0;
            // Pointers from supernodes to their starting rows.
            SN_rp       = Tensor1< Int,Int> (n+2);
            
            // Pointers from supernodes to their starting position in SN_inner.
            SN_outer    = Tensor1<LInt,Int> (n+2);
            SN_outer[0] = 0;
            
            // To be filled with the column indices of super nodes.
            // Will later be moved to SN_inner.
            Aggregator<Int,LInt> SN_inner_agg ( 2 * A.NonzeroCount(), thread_count );
            
            // Start first supernode.
            SN_rp[0]     = 0;
            row_to_SN[0] = 0;
            SN_count     = 0;
            
            // TODO: Should be parallelizable by processing subtrees of elimination tree in parallel.
            // TODO: Even better: Build aTree first (we need only to knoe the fundamental rows for that). Then collect SN_inner by tranversing aTree in parellel.
            // TODO: --> Can we precompute somehow the size of SN_inner_agg? That would greatly help to reduce copy ops and to schedule its generation.
            
            ptic("Main loop");
            for( Int i = 1; i < n+1; ++i ) // Traverse rows.
            {
                if( IsFundamental( i, prev_col_nz ) )
                {
                    // i is going to be the first node of the newly created fundamental supernode.
                    // However, we do not now at the moment how long the supernode is going to be.
                    
                    // Instead building the new supernode, we first have to finish the current supernode.
                    // Get first row in current supernode.
                    const Int i_0 = SN_rp[SN_count];

                    // The nonzero pattern of upper(A) belongs definitely to the pattern of U.
                    // We have to find all nonzero columns j of row i_0 of A such that j > i-1,
                    // because that will be the ones belonging to the rectangular part.

                    // We know that A.Inner(A.Diag(i_0)) == i_0 < i. Hence we can start the search here:
                    {
                        LInt k = A.Diag(i_0) + 1;
                        
                        const LInt k_end = A.Outer(i_0+1);
                        
                        while( (A.Inner(k) < i) && (k < k_end) ) { ++k; }
                        
                        n_1 = int_cast<Int>(k_end - k);
                        copy_buffer( &A.Inner(k), row.data(), n_1 );
                    }
                    
                    // Next, we have to merge the column indices of the children of i_0 into row.
                    const Int l_begin = eTree.ChildPointer(i_0  );
                    const Int l_end   = eTree.ChildPointer(i_0+1);
                    
                    // Traverse all children of i_0 in the eTree. Most of the time it's zero, one or two children. Seldomly it's more.
                    for( Int l = l_begin; l < l_end; ++l )
                    {
                        const Int j = eTree.ChildIndex(l);
                        // We have to merge the column indices of child j that are greater than i into U_row.
                        // This is the supernode where we find the j-th row of U.
                        const Int k = row_to_SN[j];
                        
                        // Notice that because of j < i, we only have to consider the reactangular part of this supernode.
                        
                              LInt a = SN_outer[k  ];
                        const LInt b = SN_outer[k+1];
                        
                        // Only consider column indices of j-th row of U that are greater than last row i-1 in current supernode.
                        while( (SN_inner_agg[a] < i) && (a < b) ) { ++a; }
                        
                        if( a < b )
                        {
                            n_1 = UniteSortedBuffers(
                                row.data(),       n_1,
                                &SN_inner_agg[a], int_cast<Int>(b - a),
                                row_scratch.data()
                            );
                            swap( row, row_scratch );
                        }
                    }
                    
                    // Now row is ready to be pushed into SN_inner.
                    SN_inner_agg.Push( row.data(), n_1 );
                    
                    // Start new supernode.
                    ++SN_count;
                    
                    SN_outer[SN_count] = SN_inner_agg.Size();
                    SN_rp   [SN_count] = i; // row i does not belong to previous supernode.
                }
                else
                {
                    // Continue supernode.
                }
                
                // Remember where to find i-th row.
                if( i < n )
                {
                    row_to_SN[i] = SN_count;
                }
                
            } // for( Int i = 0; i < n+1; ++i )
            ptoc("Main loop");

            ptic("Finalization");
            

            pdump(SN_count);
            
            SN_rp.Resize( SN_count+1 );
            SN_outer.Resize( SN_count+1 );
            
            ptic("SN_inner_agg.Get()");
            SN_inner = std::move(SN_inner_agg.Get());

            SN_inner_agg = Aggregator<Int, LInt>(0);
            ptoc("SN_inner_agg.Get()");

            SN_Allocate();

            CreateAssemblyTree();
            
            ptoc("Finalization");
            
            SN_initialized = true;
            
            ptoc(ClassName()+"::SN_SymbolicFactorization");
        }
    }


protected:
    
    void SN_Allocate()
    {
        ptic(ClassName()+"::SN_Allocate");
        SN_tri_ptr = Tensor1<LInt,Int> (SN_count+1);
        SN_tri_ptr[0] = 0;
        
        SN_rec_ptr = Tensor1<LInt,Int> (SN_count+1);
        SN_rec_ptr[0] = 0;
        
        max_n_0 = 0;
        max_n_1 = 0;
        
        for( Int k = 0; k < SN_count; ++k )
        {
            // Warning: Taking differences of potentially signed numbers.
            // Should not be of concern because negative numbers appear here only if something went wrong upstream.
            const Int n_0 =                  SN_rp   [k+1] - SN_rp   [k];
            const Int n_1 = int_cast<Int>(SN_outer[k+1] - SN_outer[k]);

            max_n_0 = std::max( max_n_0, n_0 );
            max_n_1 = std::max( max_n_1, n_1 );

            SN_tri_ptr[k+1] = SN_tri_ptr[k] + n_0 * n_0;
            SN_rec_ptr[k+1] = SN_rec_ptr[k] + n_0 * n_1;
        }
        
        pdump(max_n_0);
        pdump(max_n_1);
        
        // Allocating memory for the nonzero values of the factorization.
        
        SN_tri_val = Tensor1<Scal,LInt> (SN_tri_ptr[SN_count]);
        SN_rec_val = Tensor1<Scal,LInt> (SN_rec_ptr[SN_count]);
        
        logvalprint("triangle_nnz ", SN_tri_val.Size());
        logvalprint("rectangle_nnz", SN_rec_val.Size());
        
        ptoc(ClassName()+"::SN_Allocate");
    }
    
    bool IsFundamental( const Int i, Tensor1<Int,Int> & prev_col_nz )
    {
        // Using Theorem 2.3 and Corollary 3.2 in
        //
        //     Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations,
        //     https://www.osti.gov/servlets/purl/6756314
        //
        // to determine whether a new fundamental supernode starts at node u.
        
        bool is_fundamental = ( i == n ); // We make virtual root vertes fundamental, so that the main loop finishes off the last nonvirtual supernode correctly.
        
        is_fundamental = is_fundamental || ( eTree.ChildCount(i) > 1);
        
        if( !is_fundamental )
        {
            const Int threshold = i - eTree.DescendantCount(i) + 1;

            const LInt k_begin = A.Diag(i)+1; // exclude diagonal entry
            const LInt k_end   = A.Outer(i+1);
            
            for( LInt k = k_begin; k < k_end; ++k )
            {
                const Int j = A.Inner(k);
                const Int l = prev_col_nz[j];
                
                prev_col_nz[j] = i;
                
                is_fundamental = is_fundamental || ( l < threshold );
            }
        }
        
        return is_fundamental;
    }
