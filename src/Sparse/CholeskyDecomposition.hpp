#pragma once

#include <unordered_map>
#include <unordered_set>

#define LAPACK_DISABLE_NAN_CHECK

//#include <Accelerate/Accelerate.h>
#include <cblas.h>
#include <lapacke.h>

#include "../../MyBLAS.hpp"

#include "SuperNodalCholeskyFactorizer.hpp"

// Super helpful literature:
// Stewart - Building an Old-Fashioned Sparse Solver

// TODO: https://arxiv.org/pdf/1711.08446.pdf: On Computing Min-Degree Elimination Orderings

// TODO: Find supernodes: https://www.osti.gov/servlets/purl/6756314

// TODO: https://hal.inria.fr/hal-01114413/document

// TODO: https://www.jstor.org/stable/2132786 !!


// TODO: Try whether postordering leads to better supernodes!
// cf. Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations

namespace Tensors
{
    namespace Sparse
    {   
        template<typename Scalar_, typename Int_, typename LInt_>
        class CholeskyDecomposition
        {
        public:
            
            using Scalar    = Scalar_;
            using Real      = typename ScalarTraits<Scalar_>::Real;
            using Int       = Int_;
            using LInt      = LInt_;
            
            using SparseMatrix_T = SparseBinaryMatrixCSR<Int,LInt>;

            using Factorizer = SupernodalCholeskyFactorizer<Scalar,Int,LInt>;
            
            friend class SupernodalCholeskyFactorizer<Scalar,Int,LInt>;
            
        protected:
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;
            
            const Int n = 0;
            const Int thread_count = 1;
            const UpLo uplo  = UpLo::Upper;
            
            SparseMatrix_T A_lo;
            SparseMatrix_T A_up;
            
            SparseMatrix_T L;
            SparseMatrix_T U;
            
            Tensor1<Int,Int> p; // Row    permutation;
            Tensor1<Int,Int> q; // Column permutation;
            
            // elimination tree
            bool eTree_initialized = false;
            Tree<Int> eTree;
            
            // assembly three
            bool aTree_initialized = false;
            Tree<Int> aTree;
            
            //Supernode data:
            
            bool SN_initialized = false;
            
            // Number of supernodes.
            Int SN_count = 0;
            
            // Pointers from supernodes to their rows.
            // k-th supernode has rows [ SN_rp[k],...,SN_rp[k+1] [
            Tensor1<   Int, Int> SN_rp;
            // Pointers from supernodes to their starting position in SN_inner.
            Tensor1<  LInt, Int> SN_outer;
            // The column indices of rectangular part of the supernodes.
            Tensor1<   Int,LInt> SN_inner;
            // Hence k-th supernode has the following column indices:
            // triangular  part = [ SN_rp[k],SN_rp[k]+1,...,SN_rp[k+1] [
            // rectangular part = [
            //                      SN_inner[j  ],
            //                      SN_inner[j+1],
            //                      SN_inner[j+2],
            //                      ...,
            //                      SN_inner[SN_outer[k+1]]
            //                    [
            // where j = SN_outer[k].

//            // column indices of i-th row of U can be found in SN_inner in the half-open interval
//            // [ U_begin[i],...,U_end[i] [
//            Tensor1<   Int,LInt> U_begin;
//            Tensor1<   Int,LInt> U_end;
            
            // i-th row of U belongs to supernode row_to_SN[i].
            Tensor1<   Int, Int> row_to_SN;
            // Hence the column indices of U for row i can are:
            // triangular  part = [ i,i+1,...,SN_rp[row_to_SN[i]+1] [
            // rectangular part = [
            //                      SN_inner[j  ],
            //                      SN_inner[j+1],
            //                      SN_inner[j+2],
            //                      ...,
            //                      SN_inner[SN_outer[row_to_SN[i]+1]]
            //                    [
            // where j = SN_outer[row_to_SN[i]].
            
            // Values of triangular part of k-th supernode is stored in
            // [ SN_tri_val[SN_tri_ptr[k]],...,SN_tri_val[SN_tri_ptr[k]+1] [
            Tensor1<  LInt, Int> SN_tri_ptr;
            Tensor1<Scalar,LInt> SN_tri_val;
            
            // Values of rectangular part of k-th supernode is stored in
            // [ SN_rec_val[SN_rec_ptr[k]],...,SN_rec_val[SN_rec_ptr[k]+1] [
            Tensor1<  LInt, Int> SN_rec_ptr;
            Tensor1<Scalar,LInt> SN_rec_val;
            
            // Maximal size of triangular part of supernodes.
            Int max_n_0 = 0;
            // Maximal size of rectangular part of supernodes.
            Int max_n_1 = 0;
            
        public:
            
            CholeskyDecomposition() = default;
            
            ~CholeskyDecomposition() = default;
            
            CholeskyDecomposition(
                const LInt * restrict const outer_,
                const  Int * restrict const inner_,
                const  Int n_,
                const  Int thread_count_,
                const UpLo uplo_ = UpLo::Upper
            )
            :   n ( n_ )
            ,   thread_count( thread_count_ )
            ,   uplo( uplo_ )
            {
                if( uplo == UpLo::Upper)
                {
                    // TODO: Is there a way to avoid this copy?
                    A_up = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                    
                    // TODO: Is there a way to avoid this copy?
                    A_lo = A_up.Transpose();
                }
                else
                {
                    // TODO: Is there a way to avoid this copy?
                    A_lo = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                    
                    // TODO: Is there a way to avoid this copy?
                    A_up = A_lo.Transpose();
                }
                
                // TODO: What if I want to submit a full symmetric matrix pattern, not only a triangular part?
                
//                FactorizeSymbolically();
            }
            
            const Tree<Int> & EliminationTree()
            {
                // TODO: How can this be parallelized?
                
                // TODO: read Kumar, Kumar, Basu - A parallel algorithm for elimination tree computation and symbolic factorization

                
                if( ! eTree_initialized )
                {
                    ptic(ClassName()+"::EliminationTree");
                    
                    // See Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                    
                    // I want to make it possible to use unsigned integer types for Int.
                    // Hence using -1 as "no_element" is not an option.
                    // We have to use something else instead of 0 to mark empty places.
                    const Int no_element = n;
                    
                    Tensor1<Int,Int> parents ( n, no_element );
                    
                    // A vector for path compression.
                    Tensor1<Int,Int> a ( n, no_element );
                    
                    const LInt * restrict const A_outer = A_lo.Outer().data();
                    const  Int * restrict const A_inner = A_lo.Inner().data();
                    
                    for( Int k = 1; k < n; ++k )
                    {
                        // We need visit all i < k with A_ik != 0.
                        const LInt l_begin = A_outer[k  ];
                        const LInt l_end   = A_outer[k+1]-1;
                        
                        for( LInt l = l_begin; l < l_end; ++l )
                        {
                            Int i = A_inner[l];
                            
                            while( ( i != no_element ) && ( i < k ) )
                            {
                                Int j = a[i];
                                
                                a[i] = k;
                                
                                if( j == no_element )
                                {
                                    parents[i] = k;
                                }
                                i = j;
                            }
                        }
                    }
                    
                    eTree = Tree<Int> ( std::move(parents), n-1, thread_count );
                    
                    eTree_initialized = true;
                    
                    ptoc(ClassName()+"::EliminationTree");
                }
                
                return eTree;
            }

            
        protected:
            
            void CreateAssemblyTree()
            {
                ptic(ClassName()+"::CreateAssemblyTree");
                
                const Tensor1<Int,Int> & parents = EliminationTree().Parents();
                
                Tensor1<Int,Int> SN_parents ( SN_count );
                
                for( Int k = 0; k < SN_count-1; ++k )
                {
                    Int last_row = SN_rp[k+1]-1;

                    Int last_rows_parent = parents[last_row];

                    SN_parents[k] = (last_rows_parent<n) ? row_to_SN[last_rows_parent] : SN_count;
                }
                
                SN_parents[SN_count-1] = SN_count;
                
                aTree = Tree<Int> ( std::move(SN_parents), SN_count-1, thread_count );
                
                ptoc(ClassName()+"::CreateAssemblyTree");
            }
            
        public:
            
            const Tree<Int> & AssemblyTree()
            {
                SN_FactorizeSymbolically();
                
                return aTree;
            }
            
            
            void FactorizeSymbolically()
            {
                // Non-supernodal way to perform symbolic analysis.
                // Only meant as reference! In practice rather use the supernodal version SN_FactorizeSymbolically.
                
                
                // This is Algorithm 4.2 from  Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                
                ptic(ClassName()+"::FactorizeSymbolically");
                
                const LInt * restrict const A_rp      = A_up.Outer().data();
                const  Int * restrict const A_ci      = A_up.Inner().data();
                
                const  Int * restrict const child_ptr = EliminationTree().ChildPointers().data();
                const  Int * restrict const child_idx = EliminationTree().ChildIndices().data();
                
                Tensor1<Int,Int> U_i    ( n );  // An array to aggregate the rows of U.
                Tensor1<Int,Int> buffer ( n );  // Some scratch space for UniteSortedBuffers.
                Int row_counter;                // Holds the current number of indices in U_i.

                // To be filled with the row pointers of U.
                Tensor1<LInt,Int> U_rp (n+1);
                U_rp[0] = 0;
                
                // To be filled with the column indices of U.
                Aggregator<Int,LInt> U_ci ( 2 * A_up.NonzeroCount() );

                
                for( Int i = 0; i < n; ++i ) // Traverse rows.
                {
                    // The nonzero pattern of A_up belongs definitely to the pattern of U.
                    row_counter = A_rp[i+1] - A_rp[i];
                    copy_buffer( &A_ci[A_rp[i]], U_i.data(), row_counter );
                    
                    const Int l_begin = child_ptr[i  ];
                    const Int l_end   = child_ptr[i+1];

                    
                    // Traverse all children of i in the eTree. Most of the time it's a single child or no one at all. Sometimes it's two or more.
                    for( Int l = l_begin; l < l_end; ++l )
                    {
                        const Int j = child_idx[l];
                        
                        // Merge row pointers of child j into U_i
                        const LInt _begin = U_rp[j]+1;  // This excludes U_ci[U_rp[j]] == j.
                        const LInt _end   = U_rp[j+1];
                        
                        if( _end > _begin )
                        {
                            row_counter = UniteSortedBuffers(
                                U_i.data(),    row_counter,
                                &U_ci[_begin], static_cast<Int>(_end - _begin),
                                buffer.data()
                            );
                            swap( U_i, buffer );
                        }
                        
                    }
                    
                    // Copy U_i to i-th row of U.
                    U_ci.Push( U_i.data(), row_counter );
                    U_rp[i+1] = U_ci.Size();

                } // for( Int i = 0; i < n; ++i )
                
                
                U = SparseMatrix_T( std::move(U_rp), std::move(U_ci.Get()), n, n, thread_count );

                ptoc(ClassName()+"::FactorizeSymbolically");
            }
            
            template< Int RHS_COUNT, bool unitDiag = false>
            void U_Solve_Sequential_0(
                const Scalar * restrict const b,
                      Scalar * restrict const x
            )
            {
                U.SolveUpperTriangular_Sequential_0<RHS_COUNT,unitDiag>(b,x);
            }
            
//###########################################################################################
//####          Supernodal symbolic factorization
//###########################################################################################
            
            void SN_FactorizeSymbolically()
            {
                // Compute supernodal symbolic factorization with so-called _fundamental supernodes_.
                // See Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations.
                
                
                // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

                if( !SN_initialized )
                {
                    ptic(ClassName()+"::SN_FactorizeSymbolically");
                    
                    if( !EliminationTree().PostOrdered() )
                    {
                        eprint(ClassName()+"::SN_FactorizeSymbolically requires postordering!");
                    }
                    
                    const LInt * restrict const A_rp      = A_up.Outer().data();
                    const  Int * restrict const A_ci      = A_up.Inner().data();
                    
                    const  Int * restrict const child_ptr = EliminationTree().ChildPointers().data();
                    const  Int * restrict const child_idx = EliminationTree().ChildIndices().data();
        
                    // temporary arrays
                    Tensor1<Int,Int> row        (n);// An array to aggregate the columnn indices of a row of U.
                    Tensor1<Int,Int> row_buffer (n);// Some scratch space for UniteSortedBuffers.
                    Int row_counter;                // Holds the current number of indices in row.
        
                    Tensor1<Int,Int> prev_col_nz(n,-1);
                    Tensor1<Int,Int> descendant_counts  = EliminationTree().DescendantCounts();
    
                    // i-th row of U belongs to supernode row_to_SN[i].
                    row_to_SN = Tensor1< Int,Int> (n);
                    
                    // Holds the current number of supernodes.
                    SN_count = 0;
                    // Pointers from supernodes to their starting rows.
                    SN_rp    = Tensor1< Int,Int> (n+1);
                    
                    // Pointers from supernodes to their starting position in SN_inner.
                    SN_outer = Tensor1<LInt,Int> (n+1);
                    SN_outer[0]  = 0;
                    
                    // To be filled with the column indices of super nodes.
                    // Will later be moved to SN_inner.
                    Aggregator<Int,LInt> SN_inner_agg ( 4 * A_up.NonzeroCount(), thread_count );
                    
                    
                    // Start first supernode.
                    SN_rp[0]     = 0;
                    row_to_SN[0] = 0;
                    SN_count     = 0;
                    
                    // TODO: Should be parallelizable by processing subtrees of elimination tree in parallel.
                    ptic("Main loop");
                    for( Int i = 1; i < n+1; ++i ) // Traverse rows.
                    {
                        // Using Theorem 2.3 and Corollary 3.2 in
                        //
                        //     Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations
                        //
                        // to determine whether a new fundamental supernode starts at node u.
                        
                        bool is_fundamental = ( i == n );
                        
                        is_fundamental = is_fundamental || ( child_ptr[i+1] - child_ptr[i] > 1);
                        
                        if( !is_fundamental )
                        {
                            const Int threshold = i - descendant_counts[i] + 1;

                            const Int k_begin = A_rp[i]+1; // exclude diagonal entry
                            const Int k_end   = A_rp[i+1];
                            
                            for( Int k = k_begin; k < k_end; ++k )
                            {
                                const Int j = A_ci[k];
                                const Int l = prev_col_nz[j];
                                
                                prev_col_nz[j] = i;
                                
                                is_fundamental = is_fundamental || ( l < threshold );
                            }
                        }
                        
                        if( is_fundamental )
                        {
                            // i is going to be the first node of the newly created fundamental supernode.
                            // However, we do not now at the moment how long the supernode is going to be.
                            
                            // Instead building the new supernode, we first have to finish current supernode.
                            // Get first row in current supernode.
                            const Int i_0 = SN_rp[SN_count];

                            // The nonzero pattern of A_up belongs definitely to the pattern of U.
                            // We have to find all nonzero columns j of row i_0 of A such that j > i-1,
                            // because that will be the ones belonging to the rectangular part.

                            // We know that A_ci[A_rp[i_0]] == i_0 < i. Hence we can start the search here:
                            {
                                Int k = A_rp[i_0] + 1;
                                
                                const Int k_end = A_rp[i_0+1];
                                
                                while( A_ci[k] < i && k < k_end )
                                {
                                    ++k;
                                }
                                
                                row_counter = k_end - k;
                                copy_buffer( &A_ci[k], row.data(), row_counter );
                            }
                            
                            // Next, we have to merge the column indices of the children of i_0 into row.
                            const Int l_begin = child_ptr[i_0  ];
                            const Int l_end   = child_ptr[i_0+1];
        
                            // Traverse all children of i_0 in the eTree. Most of the time it's one or two children. Seldomly it's more.
                            for( Int l = l_begin; l < l_end; ++l )
                            {
                                const Int j = child_idx[l];
                                // We have to merge the column indices of child j that are greater than i into U_row.
                                // This is the supernode where we find the j-th row of U.
                                const Int k = row_to_SN[j];
                                
                                // Notice that because of j < i, we only have to consider the reactangular part of this supernode.
                                
                                      LInt a = SN_outer[k  ];
                                const LInt b = SN_outer[k+1];
                                
                                // Only consider column indices of j-th row of U that are greater than last row i-1 in current supernode.
                                while( (SN_inner_agg[a] < i) && (a < b) )
                                {
                                    ++a;
                                }
                                
                                if( a < b )
                                {
                                    row_counter = UniteSortedBuffers(
                                        row.data(),       row_counter,
                                        &SN_inner_agg[a], static_cast<Int>(b - a),
                                        row_buffer.data()
                                    );
                                    swap( row, row_buffer );
                                }
                            }
                            
                            // Now row is ready to be pushed into SN_inner.
                            SN_inner_agg.Push( row.data(), row_counter );
                            
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
                        const Int n_1 = static_cast<Int>(SN_outer[k+1] - SN_outer[k]);

                        max_n_0 = std::max( max_n_0, n_0 );
                        max_n_1 = std::max( max_n_1, n_1 );

                        SN_tri_ptr[k+1] = SN_tri_ptr[k] + n_0 * n_0;
                        SN_rec_ptr[k+1] = SN_rec_ptr[k] + n_0 * n_1;
                    }

                    
                    pdump(max_n_0);
                    pdump(max_n_1);
                    
                    // Allocating memory for the nonzero values of the factorization.
                    
                    SN_tri_val = Tensor1<Scalar, LInt> (SN_tri_ptr[SN_count]);
                    SN_rec_val = Tensor1<Scalar, LInt> (SN_rec_ptr[SN_count]);
                    
                    logvalprint("triangle_nnz ", SN_tri_val.Size());
                    logvalprint("rectangle_nnz", SN_rec_val.Size());
                    
                    CreateAssemblyTree();
                    
                    ptoc("Finalization");
                    
                    ptoc(ClassName()+"::SN_FactorizeSymbolically");
                    
                    SN_initialized = true;
                }
            }
            
            
//###########################################################################################
//####          Supernodal numeric factorization
//###########################################################################################
            
            void SN_FactorizeNumerically_Parallel(
                const Scalar * restrict const A_val,
                const Int max_depth
            )
            {
                ptic(ClassName()+"::SN_FactorizeNumerically_Parallel");

                
                ptic("Postorder traversal of assembly tree");
                const Int root = AssemblyTree().Root();

                const Int * restrict const child_ptr   = AssemblyTree().ChildPointers().data();
                const Int * restrict const child_idx   = AssemblyTree().ChildIndices().data();
                const Int * restrict const desc_counts = AssemblyTree().DescendantCounts().data();
                
                std::vector<std::vector<Int>> levels (max_depth+1);
                
                Tensor1<Int, Int> stack   ( 2*max_depth+1 );
                Tensor1<Int, Int> depth   ( 2*max_depth+1 );
                Tensor1<bool,Int> visited ( 2*max_depth+1, false );
                
                Int ptr    = 0;
                stack  [0] = root;
                depth  [0] = 0;
                visited[0] = false;
                
                // post order traversal of the tree
                while( ptr >= 0 )
                {
                    const Int node = stack[ptr];
                    const Int d    = depth[ptr];
                    
                    if( !visited[ptr] && d < max_depth )
                    {
                        // The first time we visit this node we mark it as visited
                        visited[ptr] = true;
                        
                        const Int k_begin = child_ptr[node  ];
                        const Int k_end   = child_ptr[node+1];
                        
                        // Pushing the children in reverse order onto the stack.
                        for( Int k = k_end; k --> k_begin; )
                        {
                            stack[++ptr] = child_idx[k];
                            depth[ptr]   = d+1;
                        }
                    }
                    else
                    {
                        // Visiting the node for the second time.
                        // We are moving in direction towards the root.
                        // Hence all children have already been visited.
                        
                        // Popping current node from the stack.
                        visited[ptr--] = false;
                        
                        levels[d].push_back(node);
                    }
                }
                
                ptoc("Postorder traversal of assembly tree");
                
//                for( Int s : levels[max_depth] )
//                {
//                    valprint("|T("+ToString(s)+")|",desc_counts[s]);
//                }
                
                ptic("Parallel treatment of subtrees");
                
                ptic("Zerofy buffers.");
                zerofy_buffer( SN_tri_val.data(), SN_tri_ptr.Last(), thread_count );
                zerofy_buffer( SN_rec_val.data(), SN_rec_ptr.Last(), thread_count );
                ptoc("Zerofy buffers.");
                
                ptic("Initialize factorizers");
                std::vector<std::unique_ptr<Factorizer>> SN_list ( thread_count);
                
                #pragma omp parallel for num_threads( thread_count ) schedule(static)
                for( Int thread = 0; thread < thread_count; ++thread )
                {
//                    dump(thread);
//                    dump(omp_get_thread_num());
                    SN_list[thread] = std::make_unique<Factorizer>(*this, A_val);
                }
                ptoc("Initialize factorizers");
                
                for( Int d = max_depth+1; d -->0 ; )
                {
                    ptic("Parallel treatment of subtrees (depth = "+ToString(d)+")");
//                    valprint("level["+ToString(d)+"]",levels[d]);
                    #pragma omp parallel for num_threads( thread_count ) schedule(dynamic)
                    for( size_t r = 0; r < levels[d].size(); ++r )
                    {
                        const Int thread = omp_get_thread_num();

                        Factorizer & SN = *SN_list[thread];

                        const Int s = levels[d][r];

                        if( d == max_depth )
                        {
                            const Int t_begin = (s+1) - desc_counts[s] ;
                            const Int t_end   = (s+1);  // Factorize also yourself.
                            
                            for( Int t = t_begin; t < t_end; ++t )
                            {
                                SN.Factorize(t);
                            }
                        }
                        else
                        {
                            SN.Factorize(s);
                        }
                    }
                    ptoc("Parallel treatment of subtrees (depth = "+ToString(d)+")");
                }
                
                ptoc("Parallel treatment of subtrees");
                
                ptoc(ClassName()+"::SN_FactorizeNumerically_Parallel");
                
            }
            
            void SN_FactorizeNumerically_Sequential( const Scalar * restrict const A_val )
            {
                // TODO: Make the function accept a root node s_0 of the AssemblyTree.
                // TODO: It shall then do the factorization of the full subtree.
                
                ptic(ClassName()+"::SN_FactorizeNumerically_Sequential");

                if( !AssemblyTree().PostOrdered() )
                {
                    eprint(ClassName()+"::SN_FactorizeNumerically_Sequential requires postordered assembly tree!");
                    return;
                }
//                else
//                {
//                    print("Well done! The AssemblyTree is postordered.");
//                }
                
                ptic("SetZero");
                SN_tri_val.SetZero();
                SN_rec_val.SetZero();
                ptoc("SetZero");
                
                SupernodalCholeskyFactorizer<Scalar,Int,LInt> SN ( *this, A_val );

                for( Int s = 0; s < SN_count; ++s )
                {
                    SN.Factorize(s);
                }

                ptoc(ClassName()+"::SN_FactorizeNumerically_Sequential");
            }
            
//###########################################################################################
//####          Supernodal back substitution
//###########################################################################################

            void SN_UpperSolve_Sequential( Scalar * restrict const B, const Int nrhs )
            {
                ptic("SN_UpperSolve_Sequential");
                // Solves U * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    SN_UpperSolve_Sequential(B);
                    ptoc("SN_UpperSolve_Sequential");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = SN_count; k --> 0; )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];
                    
                    const Int n_1 = static_cast<Int>(l_end - l_begin);
                    
                    // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const U_0 = &SN_tri_val[SN_tri_ptr[k]];
                    
                    // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const U_1 = &SN_rec_val[SN_rec_ptr[k]];
                    
                    // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
                          Scalar * restrict const X_0 = &B[nrhs * SN_rp[k]];
                    
                    // X_1 is the part of X that interacts with U_1, size = n_1 x rhs_count.
                          Scalar * restrict const X_1 = X_buffer.data();
                    
                    // Load the already computed values into X_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        copy_buffer( &B[nrhs * SN_inner[l_begin+j]], &X_1[nrhs * j], nrhs );
                    }

                    if( n_0 == 1 )
                    {
                        if( n_1 > 0 )
                        {
                            // Compute X_0 -= U_1 * X_1

                            //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1 x nrhs; we can interpret it as vector of size nrhs.

                            // Hence we can compute X_0^T -= X_1^T * U_1^T via gemv instead:
                            BLAS_Wrappers::gemv<Layout::RowMajor,Op::Trans>(
                                n_1, nrhs,
                                Scalar(-1), X_1, nrhs,
                                            U_1, 1,        // XXX Problem: We need conj(U_1)!
                                Scalar( 1), X_0, 1
                            );
                        }

                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        scale_buffer( one / U_0[0], X_0, nrhs );
                    }
                    else // using BLAS3 routines.
                    {
                        if( n_1 > 0 )
                        {
                            // Compute X_0 -= U_1 * X_1
                            BLAS_Wrappers::gemm<Layout::RowMajor,Op::Id,Op::Id>(
                                // XX Op::Id -> Op::ConjugateTranspose
                                n_0, nrhs, n_1,
                                Scalar(-1), U_1, n_1,      // XXX n_1 -> n_0
                                            X_1, nrhs,
                                Scalar( 1), X_0, nrhs
                            );
                        }
                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        BLAS_Wrappers::trsm<Layout::RowMajor,
                            Side::Left, UpLo::Upper, Op::Id, Diag::NonUnit
                        >(
                            n_0, nrhs,
                            Scalar(1), U_0, n_0,
                                       X_0, nrhs
                        );
                    }
                }
                ptoc("SN_UpperSolve_Sequential");
            }
            
            void SN_UpperSolve_Sequential( Scalar * restrict const b )
            {
                // Solves U * x = b and stores the result back into b.
                // Assumes that b has size n.

                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor1<Scalar,Int> x_buffer ( n );
                
                for( Int k = SN_count; k --> 0; )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];

                    const Int n_1 = static_cast<Int>(l_end - l_begin);

                    // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const U_0 = &SN_tri_val[SN_tri_ptr[k]];

                    // U_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const U_1 = &SN_rec_val[SN_rec_ptr[k]];

                    // x_0 is the part of x that interacts with U_0, size = n_0.
                          Scalar * restrict const x_0 = &b[SN_rp[k]];


                    if( n_0 == 1 )
                    {
                        Scalar U_1x_1 = 0;

                        if( n_1 > 0 )
                        {
                            // Compute X_0 -= U_1 * X_1
                            //  U_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  x_1 is a vector of size n_1.
                            //  x_0 is a matrix of size 1 x 1; we can interpret it as vector of size 1.

                            // Hence we can compute X_0 -= U_1 * X_1 via a simple dot product.

                            for( Int j = 0; j < n_1; ++j )
                            {
                                U_1x_1 += U_1[j] * b[SN_inner[l_begin+j]]; // XXX conj(U_1[j])
                            }
                        }

                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        x_0[0] = (x_0[0] - U_1x_1) / U_0[0];
                    }
                    else // using BLAS2 routines.
                    {
                        if( n_1 > 0 )
                        {
                            // x_1 is the part of x that interacts with U_1, size = n_1.
                            Scalar * restrict const x_1 = x_buffer.data();

                            // Load the already computed values into X_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                x_1[j] = b[SN_inner[l_begin+j]];
                            }

                            // Compute x_0 -= U_1 * x_1
                            BLAS_Wrappers::gemv<Layout::RowMajor,Op::Id>(// XXX Op::Id -> Op::ConjTrans
                                n_0, n_1,
                                Scalar(-1), U_1, n_1, // XXX n_1 -> n_0
                                            x_1, 1,
                                Scalar( 1), x_0, 1
                            );
                        }

                        // Triangle solve U_0 * x_0 = B while overwriting x_0.
                        BLAS_Wrappers::trsv<Layout::RowMajor,UpLo::Upper,Op::Id,Diag::NonUnit>(
                            n_0, U_0, n_0, x_0, 1
                        );
                    }
                   
                }
            }
            
            void SN_LowerSolve_Sequential( Scalar * restrict const B, const Int nrhs )
            {
                ptic("SN_LowerSolve_Sequential");
                // Solves L * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    SN_LowerSolve_Sequential(B);
                    ptoc("SN_LowerSolve_Sequential");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = 0; k< SN_count; ++k )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];
                    
                    const Int n_1 = static_cast<Int>(l_end - l_begin);
                    
                    // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const U_0 = &SN_tri_val[SN_tri_ptr[k]];
                    
                    // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const U_1 = &SN_rec_val[SN_rec_ptr[k]];
                    
                    // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
                          Scalar * restrict const X_0 = &B[nrhs * SN_rp[k]];
                    
                    // X_1 is the part of X that interacts with U_1, size = n_1 x rhs_count.
                          Scalar * restrict const X_1 = X_buffer.data();


                    if( n_0 == 1 )
                    {
                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        scale_buffer( one / U_0[0], X_0, nrhs );
                        
                        if( n_1 > 0 )
                        {
                            // Compute X_1 = - U_1^H * X_0

                            //  U_1 is a matrix of size 1   x n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1   x nrhs.

                            for( Int i = 0; i < n_1; ++i )
                            {
                                const Scalar factor = - conj(U_1[i]); // XXX conj(U_1[i])-> U_1[i]
                                for( Int j = 0; j < nrhs; ++j )
                                {
                                    X_1[nrhs*i+j] = factor * X_0[j];
                                }
                            }
                        }
                    }
                    else // using BLAS3 routines.
                    {

                        // Triangle solve U_0^H * X_0 = B_0 while overwriting X_0.
                        BLAS_Wrappers::trsm<
                            Layout::RowMajor, Side::Left,
                            UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                        >(
                            n_0, nrhs,
                            Scalar(1), U_0, n_0,
                                       X_0, nrhs
                        );
                        
                        if( n_1 > 0 )
                        {
                            // Compute X_1 = - U_1^H * X_0
                            BLAS_Wrappers::gemm<
                                Layout::RowMajor, Op::ConjTrans, Op::Id
                            >(
                               //XXX Op::ConjTrans -> Op::Id?
                                n_1, nrhs, n_0, // ???
                                Scalar(-1), U_1, n_1, // n_1 -> n_0
                                            X_0, nrhs,
                                Scalar( 0), X_1, nrhs
                            );
                        }
                    }
                    
                    // Add X_1 into B_1
                    for( Int j = 0; j < n_1; ++j )
                    {
                        add_to_buffer( &X_1[nrhs * j], &B[nrhs * SN_inner[l_begin+j]], nrhs );
                    }
                }
                ptoc("SN_LowerSolve_Sequential");
            }
            
            
            void SN_LowerSolve_Sequential( Scalar * restrict const b )
            {
                // Solves L * x = b and stores the result back into b.
                // Assumes that b has size n.

                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor1<Scalar,Int> x_buffer ( n );
                
                for( Int k = 0; k < SN_count; ++k )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];

                    const Int n_1 = static_cast<Int>(l_end - l_begin);

                    // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const U_0 = &SN_tri_val[SN_tri_ptr[k]];

                    // U_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const U_1 = &SN_rec_val[SN_rec_ptr[k]];

                    // x_0 is the part of x that interacts with U_0, size = n_0.
                          Scalar * restrict const x_0 = &b[SN_rp[k]];
                    
                    if( n_0 == 1 )
                    {
                        // Triangle solve U_0 * x_0 = b_0 while overwriting x_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale x_0.
                        x_0[0] /= U_0[0];

                        if( n_1 > 0 )
                        {
                            // Compute x_1 = - U_1^H * x_0
                            // x_1 is a vector of size n_1.
                            // U_1 is a matrix of size 1 x n_1
                            // x_0 is a vector of size 1.
                            
                            // Add x_1 into b_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                b[SN_inner[l_begin+j]] -= conj(U_1[j]) * x_0[0];
                            }   // XXX conj(U_1[j]) -> U_1[j]
                        }
                    }
                    else // using BLAS2 routines.
                    {
                        // Triangle solve U_0^H * x_0 = b_0 while overwriting x_0.
                        BLAS_Wrappers::trsv<
                            Layout::RowMajor, UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                        >( n_0, U_0, n_0, x_0, 1 );
                        
                        if( n_1 > 0 )
                        {
                            // x_1 is the part of x that interacts with U_1, size = n_1.
                            Scalar * restrict const x_1 = x_buffer.data();
                            
                            // Compute x_1 = - U_1^H * x_0
                            BLAS_Wrappers::gemv<
                                Layout::RowMajor, Op::ConjTrans // XXX Op::ConjTrans -> Op::Trans
                            >(
                                n_0, n_1,
                                Scalar(-1), U_1, n_1, // XXX n_1 -> n_0
                                            x_0, 1,
                                Scalar( 0), x_1, 1
                            );
                            
                            // Add x_1 into b_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                b[SN_inner[l_begin+j]] += x_1[j];
                            }
                        }
                    }
                   
                }
            }
            
            void SN_Solve_Sequential( Scalar * restrict const B, const Int nrhs )
            {
                SN_LowerSolve_Sequential( B, nrhs );
                SN_UpperSolve_Sequential( B, nrhs );
            }
            
            void SN_Solve_Sequential( Scalar * restrict const b )
            {
                SN_LowerSolve_Sequential( b );
                SN_UpperSolve_Sequential( b );
            }
            
            
//###########################################################################################
//####          Supernodes to matrix
//###########################################################################################
            
            void SN_ReconstructU()
            {
                Tensor1<LInt,Int> U_rp (n+1);
                U_rp[0] = 0;
                
                for( Int k = 0; k < SN_count; ++k )
                {
                    const Int i_begin = SN_rp[k  ];
                    const Int i_end   = SN_rp[k+1];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];
                    
//                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = static_cast<Int>(l_end - l_begin);
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        U_rp[i+1] = U_rp[i] + (i_end-i) + n_1;
                    }
                }
                
                valprint("nnz",U_rp.Last());
                
                Tensor1<Int,LInt> U_ci (U_rp.Last());
                
                for( Int k = 0; k < SN_count; ++k )
                {
                    const Int i_begin = SN_rp[k  ];
                    const Int i_end   = SN_rp[k+1];
                    
                    const LInt l_begin = SN_outer[k  ];
                    const LInt l_end   = SN_outer[k+1];
                    
                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = static_cast<Int>(l_end - l_begin);

                    const Int start = U_rp[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        U_ci[start + delta] = i;
                    }
                    
                    copy_buffer( &SN_inner[l_begin], &U_ci[U_rp[i_begin]+n_0], n_1 );
                    
                    #pragma omp parallel for num_threads( thread_count )
                    for( Int i = i_begin+1; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        copy_buffer( &U_ci[start+delta], &U_ci[U_rp[i]], (i_end-i) + n_1 );
                    }
                }
                
                U = SparseMatrix_T( std::move(U_rp), std::move(U_ci), n, n, thread_count );
            }
            
//###########################################################################################
//####          Get routines
//###########################################################################################

            Int RowCount() const
            {
                return n;
            }
            
            Int ColCount() const
            {
                return n;
            }
            
            const SparseMatrix_T & GetL() const
            {
                return L;
            }
            
            const SparseMatrix_T & GetU() const
            {
                return U;
            }
            
            const SparseMatrix_T & GetLowerTriangleOfA() const
            {
                return A_lo;
            }
            
            const SparseMatrix_T & GetUpperTriangleOfA() const
            {
                return A_up;
            }
            
            
            Int SN_Count() const
            {
                return SN_count;
            }
            
            const Tensor1<Int,Int> & SN_RowPointers() const
            {
                return SN_rp;
            }
            
            const Tensor1<LInt,Int> & SN_Outer() const
            {
                return SN_outer;
            }
            
            const Tensor1<Int,LInt> & SN_Inner() const
            {
                return SN_inner;
            }
            
            const Tensor1<LInt,Int> & SN_TrianglePointers() const
            {
                return SN_tri_ptr;
            }
            
            
            Tensor1<Scalar,LInt> & SN_TriangleValues()
            {
                return SN_tri_val;
            }
            
            const Tensor1<Scalar,LInt> & SN_TriangleValues() const
            {
                return SN_tri_val;
            }
            
            const Tensor1<LInt,Int> & SN_RectanglePointers() const
            {
                return SN_rec_ptr;
            }
            
            Tensor1<Scalar,LInt> & SN_RectangleValues()
            {
                return SN_rec_val;
            }
            
            const Tensor1<Scalar,LInt> & SN_RectangleValues() const
            {
                return SN_rec_val;
            }
            
            const Tensor1<Int,Int> & RowToSN() const
            {
                return row_to_SN;
            }
            
            std::string ClassName()
            {
                return "Sparse::CholeskyDecomposition<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
            }
            
            
        }; // class CholeskyDecomposition
        
    } // namespace Sparse
    
} // namespace Tensors
