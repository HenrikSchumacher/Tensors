#pragma once

#include <unordered_map>
#include <unordered_set>


#include <Accelerate/Accelerate.h>

#include "../../MyBLAS.hpp"

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
//    template<
//        int M, int N, int K,
//        ScalarFlag alpha_flag, ScalarFlag beta_flag,
//        typename Scalar, typename LInt
//    >
//    void gemm_input_scattered(
//        const int M_,
//        const int N_,
//        const int K_,
//        const Scalar alpha, const Scalar * restrict A,
//                            const Scalar * restrict B, const LInt * restrict const idx,
//        const Scalar beta,        Scalar * restrict C
//    )
//    {
//        constexpr int Dyn = MyBLAS::Dynamic;
//
//        if constexpr ( beta_flag == ScalarFlag::Zero )
//        {
//            zerofy_buffer( C, M_*N_ );
//        }
//        else if constexpr ( beta_flag == ScalarFlag::Plus )
//        {
//            // Do nothing.
//        }
//        else
//        {
//            scale_buffer( beta, C, M_*N_ );
//        }
//
//        if constexpr ( alpha_flag == ScalarFlag::Plus )
//        {
//            for( int k = 0; k < COND(K==Dyn,K_,K); ++k )
//            {
//                const Scalar * restrict B_k = &B[COND(N==Dyn,N_,N)*idx[k]];
//
//                for( int i = 0; i < COND(M==Dyn,M_,M); ++i )
//                {
//                    const Scalar A_ik = A[COND(K==Dyn,K_,K)*i+k];
//
//                    for( int j = 0; j < COND(N==Dyn,N_,N); ++j )
//                    {
//                        C[COND(N==Dyn,N_,N)*i+j] += A_ik * B_k[j];
//                    }
//                }
//            }
//        }
//        else if constexpr ( alpha_flag == ScalarFlag::Minus )
//        {
//            for( int k = 0; k < COND(K==Dyn,K_,K); ++k )
//            {
//                const Scalar * restrict B_k = &B[COND(N==Dyn,N_,N)*idx[k]];
//
//                for( int i = 0; i < COND(M==Dyn,M_,M); ++i )
//                {
//                    const Scalar A_ik = A[COND(K==Dyn,K_,K)*i+k];
//
//                    for( int j = 0; j < COND(N==Dyn,N_,N); ++j )
//                    {
//                        C[COND(N==Dyn,N_,N)*i+j] -= A_ik * B_k[j];
//                    }
//                }
//            }
//        }
//        else if constexpr ( alpha_flag == ScalarFlag::Generic )
//        {
//            for( int k = 0; k < COND(K==Dyn,K_,K); ++k )
//            {
//                const Scalar * restrict B_k = &B[COND(N==Dyn,N_,N)*idx[k]];
//
//                for( int i = 0; i < COND(M==Dyn,M_,M); ++i )
//                {
//                    const Scalar A_ik = alpha * A[COND(K==Dyn,K_,K)*i+k];
//
//                    for( int j = 0; j < COND(N==Dyn,N_,N); ++j )
//                    {
//                        C[COND(N==Dyn,N_,N)*i+j] += A_ik * B_k[j];
//                    }
//                }
//            }
//        }
//        else if constexpr ( alpha_flag == ScalarFlag::Zero )
//        {
//            // Do nothing.
//        }
//    }

    namespace Sparse
    {
        
        template<typename Scalar, typename Int, typename LInt>
        class CholeskyDecomposition
        {
        public:
            
            using SparseMatrix_T = SparseBinaryMatrixCSR<Int,LInt>;
            
            using List_T = SortedList<Int,Int>;
            
        protected:
            
            static constexpr Scalar zero = 0;
            static constexpr Scalar one  = 1;
            
            const Int n = 0;
            const Int thread_count = 1;
            const Triangular uplo  = Triangular::Upper;
            
            SparseMatrix_T A_lo;
            SparseMatrix_T A_up;
            
            SparseMatrix_T L;
            SparseMatrix_T U;
            
            Tensor1<Int,Int> p; // Row    permutation;
            Tensor1<Int,Int> q; // Column permutation;
            
            bool eTree_initialized = false;
            EliminationTree<Int> eTree;
            
            //Supernode data:
            
            // Number of supernodes.
            Int SN_count = 0;
            
            // Pointers from supernodes to their rows.
            // k-th supernode has rows [ SN_rp[k],...,SN_rp[k+1] [
            Tensor1<  LInt, Int> SN_rp;
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
            // [ SN_tri_vals[SN_tri_ptr[k]],...,SN_tri_vals[SN_tri_ptr[k]+1] [
            Tensor1<  LInt, Int> SN_tri_ptr;
            Tensor1<Scalar,LInt> SN_tri_vals;
            
            // Values of rectangular part of k-th supernode is stored in
            // [ SN_rec_vals[SN_rec_ptr[k]],...,SN_rec_vals[SN_rec_ptr[k]+1] [
            Tensor1<  LInt, Int> SN_rec_ptr;
            Tensor1<Scalar,LInt> SN_rec_vals;
            
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
                const Triangular uplo_ = Triangular::Upper
            )
            :   n ( n_ )
            ,   thread_count( thread_count_ )
            ,   uplo( uplo_ )
            {
                if( uplo == Triangular::Upper)
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
                
//                FactorizeSymbolically();
                
                //            FactorizeSymbolically2();
            }
            
            const EliminationTree<Int> & GetEliminationTree()
            {
                if( ! eTree_initialized )
                {
                    ptic(ClassName()+"::GetEliminationTree");
                    
                    // See Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                    
                    Tensor1<Int,Int> parents ( n, n );
                    Tensor1<Int,Int> buffer  ( n, n );
                    
                    const LInt * restrict const A_outer = A_lo.Outer().data();
                    const  Int * restrict const A_inner = A_lo.Inner().data();
                    
//                    tic("Main loop");
                    for( Int i = 1; i < n; ++i )
                    {
                        const LInt k_begin = A_outer[i  ];
                        const LInt k_end   = A_outer[i+1];
                        
                        for( LInt k = k_begin; k < k_end; ++k )
                        {
                            Int j = A_inner[k];
                            
                            while( j != n && j < i)
                            {
                                Int j_temp = buffer[j];
                                
                                buffer[j] = i;
                                
                                if( j_temp == n )
                                {
                                    parents[j] = i;
                                }
                                j = j_temp;
                            }
                        }
                    }
//                    toc("Main loop");
                    
                    eTree = EliminationTree<Int> ( std::move(parents) );
                    
                    eTree_initialized = true;
                    
                    ptoc(ClassName()+"::GetEliminationTree");
                }
                
                return eTree;
            }
            
            
            void FactorizeSymbolically()
            {
                // This is Algorithm 4.2 from  Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                
                ptic(ClassName()+"::FactorizeSymbolically");
                
                const LInt * restrict const A_rp      = A_up.Outer().data();
                const  Int * restrict const A_ci      = A_up.Inner().data();
                
                const  Int * restrict const child_ptr = GetEliminationTree().ChildPointers().data();
                const  Int * restrict const child_idx = GetEliminationTree().ChildIndices().data();
                
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
                        const Int _begin = U_rp[j]+1;  // This excludes U_ci[U_rp[j]] == j.
                        const Int _end   = U_rp[j+1];
                        
                        if( _end > _begin )
                        {
                            row_counter = UniteSortedBuffers(
                                U_i.data(),    row_counter,
                                &U_ci[_begin], _end - _begin,
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
                
                // TODO: This requires that A is postordered, i.e., that
                // TODO: GetEliminationTree().PostOrdering() == [0,...,n[.
                
                auto p = GetEliminationTree().PostOrdering();
                
                bool postordered = true;
                
                for( Int i = 0; i < n; ++i )
                {
                    if( i != p[i] )
                    {
                        postordered =false;
                        break;
                    }
                }
                
                if( !postordered )
                {
                    eprint(ClassName()+"::SN_FactorizeSymbolically requires postordering!");
                }
                
                
                // We avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

                tic(ClassName()+"::SN_FactorizeSymbolically");

                const LInt * restrict const A_rp      = A_up.Outer().data();
                const  Int * restrict const A_ci      = A_up.Inner().data();
                
                const  Int * restrict const child_ptr = GetEliminationTree().ChildPointers().data();
                const  Int * restrict const child_idx = GetEliminationTree().ChildIndices().data();
    
                // temporary arrays
                Tensor1<Int,Int> row        (n);// An array to aggregate the columnn indices of a row of U.
                Tensor1<Int,Int> row_buffer (n);// Some scratch space for UniteSortedBuffers.
                Int row_counter;                // Holds the current number of indices in row.
    
                Tensor1<Int,Int> prev_col_nz(n,0);
                Tensor1<Int,Int> descendant_counts  = GetEliminationTree().DescendantCounts();
//
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
                Aggregator<Int,LInt> SN_inner_agg ( 2 * A_up.NonzeroCount() );
                
                SN_rp[0]     = 0;
                row_to_SN[0] = 0;
                SN_count     = 0;
                
                tic("Main loop");
                for( Int i = 0; i < n+1; ++i ) // Traverse rows.
                {
                    // Using Theorem 2.3 and Corollary 3.2 in
                    //
                    //     Liu, Ng, Peyton - On Finding Supernodes for Sparse Matrix Computations
                    //
                    // to determine whether a new fundamental supernode starts at node u.
                    
                    bool is_fundamental = ( i == n );

//                    if( i == n )
//                    {
//                        print("i==n");
//                    }
                    
                    is_fundamental = is_fundamental || ( child_ptr[i+1] - child_ptr[i] > 1);
                    
//                    if( child_ptr[i+1] - child_ptr[i] > 1 )
//                    {
//                        print("child_ptr[i+1] - child_ptr[i] > 1");
//                    }
                    
                    if( !is_fundamental )
                    {
                        const Int threshold = i+1 - descendant_counts[i] + 1;

                        const Int k_begin = A_rp[i]+1; // exclude diagonal entry
                        const Int k_end   = A_rp[i+1];
                        
                        for( Int k = k_begin; k < k_end; ++k )
                        {
                            const Int j = A_ci[k];
                            const Int l = prev_col_nz[j];
                            
                            if( l < threshold )
                            {
                                is_fundamental = true;
                            }

//                            is_fundamental = is_fundamental || ( l < threshold );

                            prev_col_nz[j] = i+1;
                        }
                    }
                    
                    is_fundamental = is_fundamental && ( 0 < i );
                    
                    if( is_fundamental )
                    {

                        // i is going to be the first node of the newly created fundamental supernode.
                        // However, we do not now at the moment how long the supernode is going to be.
                        
                        // Instead building the new supernode, we first have to finish current supernode.
                        // Get first row in current supernode.
                        const Int i_0 = SN_rp[SN_count];
//                        dump(SN_count);
//                        dump(i_0);
//                        dump(i);
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
                            while( SN_inner_agg[a] < i && a < b )
                            {
                                ++a;
                            }
                            
                            if( a < b )
                            {
                                row_counter = UniteSortedBuffers(
                                    row.data(),       row_counter,
                                    &SN_inner_agg[a], b - a,
                                    row_buffer.data()
                                );
                                swap( row, row_buffer );
                            }
                        }
                        
                        // Now row is ready to be pushed into SN_inner.
//                        dump(ToString( row.data(), row_counter, 4));
                        SN_inner_agg.Push( row.data(), row_counter );
                        
                        // Start new supernode.
                        ++SN_count;
                        
                        SN_outer[SN_count] = SN_inner_agg.Size();
                        SN_rp[SN_count] = i; // row i does not belong to previous supernode.
                    }
                    else
                    {
                        // Continue supernode -- do nothing!
                    }
                    
                    // Remember where to find i-th row.
                    row_to_SN[i] = SN_count;
                    
                } // for( Int i = 0; i < n+1; ++i )
                toc("Main loop");
                
                dump(SN_count);
                
                SN_rp.Resize(SN_count+1);
                SN_outer.Resize(SN_count+1);
                
//                dump(SN_rp);
//                dump(SN_outer);
                
                SN_inner = std::move(SN_inner_agg.Get());

//                dump(SN_inner);
//                dump(row_to_SN);
                
                SN_tri_ptr = Tensor1<LInt,Int> (SN_count+1);
                SN_tri_ptr[0] = 0;
                
                SN_rec_ptr = Tensor1<LInt,Int> (SN_count+1);
                SN_rec_ptr[0] = 0;
                
                max_n_0 = 0;
                max_n_1 = 0;
                
                tic("Computing number of nonzeroes");
                for( Int k = 0; k < SN_count; ++k )
                {
                    // Warning: Taking differences of potentially signed numbers.
                    // Should not be of concern because negative numbers appear here only of sumething went wrong upstream.
                    const LInt n_0 = SN_rp[k+1]    - SN_rp[k];
                    const LInt n_1 = SN_outer[k+1] - SN_outer[k];

                    max_n_0 = std::max( max_n_0, n_0 );
                    max_n_1 = std::max( max_n_1, n_1 );
                    
                    SN_tri_ptr[k+1] = SN_tri_ptr[k] + n_0 * n_0;
                    SN_rec_ptr[k+1] = SN_rec_ptr[k] + n_0 * n_1;
                }
                toc("Computing number of nonzeroes");
                
                dump(max_n_0);
                dump(max_n_1);
                
                tic("Allocating nonzero values");
                // Allocating memory for the nonzero values of the factorization.
                
                // TODO: Filling with 0 is not really needed.
                SN_tri_vals = Tensor1<Scalar, LInt> (SN_tri_ptr[SN_count],0);
                SN_rec_vals = Tensor1<Scalar, LInt> (SN_rec_ptr[SN_count],0);
                
                valprint("triangle_nnz ", SN_tri_vals.Size());
                valprint("rectangle_nnz", SN_rec_vals.Size());
                
                toc("Allocating nonzero values");
            
                
                toc(ClassName()+"::SN_FactorizeSymbolically");
            }
            
            
            void SN_ReconstructU()
            {
                Tensor1<LInt,Int> U_rp (n+1);
                U_rp[0] = 0;
                
                
                for( Int k = 0; k < SN_count; ++k )
                {
                    const Int i_begin = SN_rp[k  ];
                    const Int i_end   = SN_rp[k+1];
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    
//                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = l_end - l_begin;
                    
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
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    
                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = l_end - l_begin;

                    const Int start = U_rp[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        U_ci[start + delta] = i;
                    }
                    
                    copy_buffer( &SN_inner[l_begin], &U_ci[U_rp[i_begin]+n_0], n_1 );
                    
                    for( Int i = i_begin+1; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        copy_buffer( &U_ci[start+delta], &U_ci[U_rp[i]], (i_end-i) + n_1 );
                    }
                }
                
                U = SparseMatrix_T( std::move(U_rp), std::move(U_ci), n, n, thread_count );
            }
            
            
//###########################################################################################
//####          Supernodal numeric factorization
//###########################################################################################
            
//            void SN_FactorizeNumerically()
//            {
//                tic(ClassName()+"::SN_FactorizeNumerically");
//
//
//                Aggregator<Int,Int> I_pos (max_n_0);
//                Aggregator<Int,Int> J_pos (max_n_1);
//
//                Aggregator<Int,Int> K_pos (max_n_0);
//                Aggregator<Int,Int> L_pos (max_n_1);
//
//                toc(ClassName()+"::SN_FactorizeNumerically");
//            }
//
//            void SN_Intersect(
//                const Int i, const Int k,
//                Tensor1<Int,Int> & I_pos, Int & I_ctr,
//                Tensor1<Int,Int> & J_pos, Int & J_ctr,
//                Tensor1<Int,Int> & K_pos, Int & K_ctr,
//                Tensor1<Int,Int> & L_pos, Int & L_ctr
//            )
//            {
//                // Computes the intersecting column indices of i-th and k-th supernode
//
//                // i-th supernode has triangular  part I = [i,i+1,...,i+n_0[
//                //                and rectangular part J = [SN_inner[SN_outer[i]],...,[
//                // i-th supernode has triangular part [i,i+1,...,i+n_0[
//
//                tic(ClassName()+"::SN_FactorizeNumerically");
//
//                I_ctr = 0;
//                J_ctr = 0;
//                K_ctr = 0;
//                L_ctr = 0;
//
//                Aggregator<Int,Int> posI (max_n_0);
//                Aggregator<Int,Int> posJ (max_n_1);
//                Aggregator<Int,Int> posK (max_n_0);
//                Aggregator<Int,Int> posL (max_n_1);
//
//                toc(ClassName()+"::SN_FactorizeNumerically");
//            }
            
//###########################################################################################
//####          Supernodal back substitution
//###########################################################################################
            
//            template<int nrhs>
//            void SN_UpperSolve_Sequential( Scalar * restrict const b )
//            {
//                tic("SN_UpperSolve_Sequential<"+ToString(nrhs)+">");
//                // Solves U * x = b and stores the result back into b.
//                // Assumes that b has size n x rhs_count.
//
//                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
//                Tensor2<Scalar,Int> x_buffer ( n, nrhs );
//
//                for( Int k = SN_count; k --> 0; )
//                {
//                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
//
//                    const Int l_begin = SN_outer[k  ];
//                    const Int l_end   = SN_outer[k+1];
//
//                    const Int n_1 = l_end - l_begin;
//
//                    // A_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
//                    const Scalar * restrict const A_0 = &SN_tri_vals[SN_tri_ptr[k]];
//
//                    // A_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
//                    const Scalar * restrict const A_1 = &SN_rec_vals[SN_rec_ptr[k]];
//
//                    // x_0 is the part of x that interacts with A_0, size = n_0 x rhs_count.
//                          Scalar * restrict const x_0 = &b[nrhs * SN_rp[k]];
//
//                    // x_1 is the part of x that interacts with A_1, size = n_1 x rhs_count.
//                          Scalar * restrict const x_1 = x_buffer.data();
//
//                    if( n_1 > 0 )
//                    {
//                        // Load the already computed values into x_1.
//                        for( Int j = 0; j < n_1; ++j )
//                        {
//                            copy_buffer<nrhs>( &b[ nrhs * SN_inner[l_begin+j]], &x_1[nrhs * j] );
//                        }
//
//                        // Compute x_0 -= A_1 * x_1
//
////                        MyBLAS::GEMM<
////                            Op::Identity, Op::Identity,
////                            -1, nrhs, -1,
////                            ScalarFlag::Minus, ScalarFlag::Plus, Scalar
////                        >()(
////                            n_0, nrhs, n_1,
////                           -one, A_1, n_1,
////                                 x_1, nrhs,
////                            one, x_0, nrhs
////                        );
//
//
//                        if constexpr ( nrhs == 1 )
//                        {
//                            BLAS_Wrappers::gemv(
//                                CblasRowMajor, CblasNoTrans, n_0, n_1,
//                               -one, A_1, n_1,
//                                     x_1, nrhs,
//                                one, x_0, nrhs
//                            );
//                        }
//                        else
//                        {
//                            BLAS_Wrappers::gemm(
//                                CblasRowMajor, CblasNoTrans, CblasNoTrans, n_0, nrhs, n_1,
//                               -one, A_1, n_1,
//                                x_1, nrhs,
//                                one, x_0, nrhs
//                            );
//                        }
//                    }
//
//                    // Triangle solve A_0 * x_0 = b while overwriting x_0.
//                    if( n_0 == 1 )
//                    {
//                        scale_buffer<nrhs>( one / A_0[0], x_0 );
//                    }
//                    else
//                    {
////                        TriangularSolve<nrhs,CblasUpper,CblasNonUnit>( n_0, A_0, x_0 );
////
////
//                        MyBLAS::TRSM<
//                            Side::Left,
//                            Triangular::Upper,
//                            Op::Identity,
//                            Diagonal::Generic,
//                            MyBLAS::Dynamic,
//                            nrhs,
//                            ScalarFlag::Plus,
//                            Scalar
//                        >()(n_0, nrhs, A_0, x_0);
//                    }
//                }
//                toc("SN_UpperSolve_Sequential<"+ToString(nrhs)+">");
//            }
//
//            template<int nrhs_lo, int nrhs_hi>
//            void SN_UpperSolve_Sequential( Scalar * restrict const b, const int nrhs )
//            {
//                if constexpr (nrhs_lo == nrhs_hi )
//                {
//                    U_Solve_Sequential_SN<nrhs_lo>(b);
//                }
//                else
//                {
//                    const int nrhs_mid = nrhs_lo + (nrhs_hi - nrhs_lo)/2;
//                    if( nrhs == nrhs_mid )
//                    {
//                        U_Solve_Sequential_SN<nrhs_mid>(b);
//                    }
//                    else if( nrhs < nrhs_mid )
//                    {
//                        U_Solve_Sequential_SN<nrhs_lo,nrhs_mid-1>(b,nrhs);
//                    }
//                    else
//                    {
//                        U_Solve_Sequential_SN<nrhs_mid+1,nrhs_hi>(b,nrhs);
//                    }
//                }
//            }
            
            void SN_UpperSolve_Sequential( Scalar * restrict const B, const Int nrhs )
            {
                tic("SN_UpperSolve_Sequential_SN");
                // Solves U * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    SN_UpperSolve_Sequential(B);
                    toc("SN_UpperSolve_Sequential_SN");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = SN_count; k --> 0; )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    
                    const Int n_1 = l_end - l_begin;
                    
                    // A_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const A_0 = &SN_tri_vals[SN_tri_ptr[k]];
                    
                    // A_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const A_1 = &SN_rec_vals[SN_rec_ptr[k]];
                    
                    // X_0 is the part of X that interacts with A_0, size = n_0 x rhs_count.
                          Scalar * restrict const X_0 = &B[nrhs * SN_rp[k]];
                    
                    // X_1 is the part of X that interacts with A_1, size = n_1 x rhs_count.
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
                            // Compute X_0 -= A_1 * X_1

                            //  A_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1 x nrhs; we can interpret it as vector of size nrhs.

                            // Hence we can compute X_0^T -= X_1^T * A_1^T via gemv instead:
                            BLAS_Wrappers::gemv(
                                CblasRowMajor, CblasTrans,
                                n_1, nrhs,
                               -one, X_1, nrhs,
                                     A_1, 1,
                                one, X_0, 1
                            );
                        }

                        // Triangle solve A_0 * X_0 = B while overwriting X_0.
                        // Since A_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        scale_buffer( one / A_0[0], X_0, nrhs );
                    }
                    else // using BLAS3 routines.
                    {
                        if( n_1 > 0 )
                        {
                            // Compute X_0 -= A_1 * X_1
                            BLAS_Wrappers::gemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                n_0, nrhs, n_1,
                               -one, A_1, n_1,
                                     X_1, nrhs,
                                one, X_0, nrhs
                            );
                        }
                        // Triangle solve A_0 * X_0 = B while overwriting X_0.
                        BLAS_Wrappers::trsm(
                            CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                            n_0, nrhs,
                            one, A_0, n_0,
                                 X_0, nrhs
                        );
                    }
                }
                toc("SN_UpperSolve_Sequential");
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
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];

                    const Int n_1 = l_end - l_begin;

                    // A_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const A_0 = &SN_tri_vals[SN_tri_ptr[k]];

                    // A_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const A_1 = &SN_rec_vals[SN_rec_ptr[k]];

                    // x_0 is the part of x that interacts with A_0, size = n_0.
                          Scalar * restrict const x_0 = &b[SN_rp[k]];


                    if( n_0 == 1 )
                    {
                        Scalar A_1x_1 = 0;

                        if( n_1 > 0 )
                        {
                            // Compute X_0 -= A_1 * X_1
                            //  A_1 is a matrix of size 1 x n_1; we can interpret it as vector of size n_1.
                            //  x_1 is a vector of size n_1.
                            //  x_0 is a matrix of size 1 x 1; we can interpret it as vector of size 1.

                            // Hence we can compute X_0 -= A_1 * X_1 via a simple dot product.

                            for( Int j = 0; j < n_1; ++j )
                            {
                                A_1x_1 += A_1[j] * b[SN_inner[l_begin+j]];
                            }
                        }

                        // Triangle solve A_0 * X_0 = B while overwriting X_0.
                        // Since A_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        x_0[0] = (x_0[0] - A_1x_1) / A_0[0];
                    }
                    else // using BLAS2 routines.
                    {
                        if( n_1 > 0 )
                        {
                            // x_1 is the part of x that interacts with A_1, size = n_1.
                            Scalar * restrict const x_1 = x_buffer.data();

                            // Load the already computed values into X_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                x_1[j] = b[SN_inner[l_begin+j]];
                            }

                            // Compute x_0 -= A_1 * x_1
                            BLAS_Wrappers::gemv(
                                CblasRowMajor, CblasNoTrans,
                                n_0, n_1,
                               -one, A_1, n_1,
                                     x_1, 1,
                                one, x_0, 1
                            );
                        }

                        // Triangle solve A_0 * x_0 = B while overwriting x_0.
                        BLAS_Wrappers::trsv(
                            CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                            n_0, A_0, n_0, x_0, 1
                        );
                    }
                   
                }
            }
            
            void SN_LowerSolve_Sequential( Scalar * restrict const B, const Int nrhs )
            {
                tic("SN_LowerSolve_Sequential");
                // Solves L * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    SN_LowerSolve_Sequential(B);
                    toc("SN_LowerSolve_Sequential");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = 0; k< SN_count; ++k )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    
                    const Int n_1 = l_end - l_begin;
                    
                    // A_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const A_0 = &SN_tri_vals[SN_tri_ptr[k]];
                    
                    // A_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const A_1 = &SN_rec_vals[SN_rec_ptr[k]];
                    
                    // X_0 is the part of X that interacts with A_0, size = n_0 x rhs_count.
                          Scalar * restrict const X_0 = &B[nrhs * SN_rp[k]];
                    
                    // X_1 is the part of X that interacts with A_1, size = n_1 x rhs_count.
                          Scalar * restrict const X_1 = X_buffer.data();


                    if( n_0 == 1 )
                    {
                        // Triangle solve A_0 * X_0 = B while overwriting X_0.
                        // Since A_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        scale_buffer( one / A_0[0], X_0, nrhs );
                        
                        if( n_1 > 0 )
                        {
                            // Compute X_1 = - A_1^H * X_0

                            //  A_1 is a matrix of size 1   x n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1   x nrhs.

                            for( Int i = 0; i < n_1; ++i )
                            {
                                const Scalar factor = - conj(A_1[i]);
                                for( Int j = 0; j < nrhs; ++j )
                                {
                                    X_1[nrhs*i+j] = factor * X_0[j];
                                }
                            }
                        }
                    }
                    else // using BLAS3 routines.
                    {

                        // Triangle solve A_0^H * X_0 = B_0 while overwriting X_0.
                        BLAS_Wrappers::trsm(
                            CblasRowMajor, CblasLeft, CblasUpper, CblasConjTrans, CblasNonUnit,
                            n_0, nrhs,
                            one, A_0, n_0,
                                 X_0, nrhs
                        );
                        
                        if( n_1 > 0 )
                        {
                            // Compute X_1 = - A_1^H * X_0
                            BLAS_Wrappers::gemm(
                                CblasRowMajor, CblasConjTrans, CblasNoTrans,
                                n_1, nrhs, n_0, // ???
                                -one, A_1, n_1,
                                      X_0, nrhs,
                                zero, X_1, nrhs
                            );
                        }
                    }
                    
                    // Add X_1 into B_1
                    for( Int j = 0; j < n_1; ++j )
                    {
                        add_to_buffer( &X_1[nrhs * j], &B[nrhs * SN_inner[l_begin+j]], nrhs );
                    }
                }
                toc("SN_LowerSolve_Sequential");
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
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];

                    const Int n_1 = l_end - l_begin;

                    // A_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                    const Scalar * restrict const A_0 = &SN_tri_vals[SN_tri_ptr[k]];

                    // A_0 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                    const Scalar * restrict const A_1 = &SN_rec_vals[SN_rec_ptr[k]];

                    // x_0 is the part of x that interacts with A_0, size = n_0.
                          Scalar * restrict const x_0 = &b[SN_rp[k]];
                    
                    if( n_0 == 1 )
                    {
                        // Triangle solve A_0 * x_0 = b_0 while overwriting x_0.
                        // Since A_0 is a 1 x 1 matrix, it suffices to just scale x_0.
                        x_0[0] /= A_0[0];

                        if( n_1 > 0 )
                        {
                            // Compute x_1 = - A_1^H * x_0
                            // x_1 is a vector of size n_1.
                            // A_1 is a matrix of size 1 x n_1
                            // x_0 is a vector of size 1.
                            
                            // Add x_1 into b_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                b[SN_inner[l_begin+j]] -= conj(A_1[j]) * x_0[0];
                            }
                        }
                    }
                    else // using BLAS2 routines.
                    {
                        // Triangle solve A_0^H * x_0 = b_0 while overwriting x_0.
                        BLAS_Wrappers::trsv(
                            CblasRowMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            n_0, A_0, n_0, x_0, 1
                        );
                        
                        if( n_1 > 0 )
                        {
                            // x_1 is the part of x that interacts with A_1, size = n_1.
                                  Scalar * restrict const x_1 = x_buffer.data();
                            
                            // Compute x_1 = - A_1^H * x_0
                            BLAS_Wrappers::gemv(
                                CblasRowMajor, CblasConjTrans,
                                n_0, n_1,
                                -one, A_1, n_1,
                                      x_0, 1,
                                zero, x_1, 1
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
            
            const Tensor1<LInt,Int> & SN_RowPointers() const
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
                return SN_tri_vals;
            }
            
            const Tensor1<Scalar,LInt> & SN_TriangleValues() const
            {
                return SN_tri_vals;
            }
            
            const Tensor1<LInt,Int> & SN_RectanglePointers() const
            {
                return SN_rec_ptr;
            }
            
            Tensor1<Scalar,LInt> & SN_RectangleValues()
            {
                return SN_rec_vals;
            }
            
            const Tensor1<Scalar,LInt> & SN_RectangleValues() const
            {
                return SN_rec_vals;
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
