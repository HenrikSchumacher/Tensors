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
            // The column indices of the supernodes.
            // k-th supernode has column indices [ SN_inner[SN_outer[k]],...,SN_innerSN_outer[k+1]] [
            Tensor1<   Int,LInt> SN_inner;
            
            // column indices of i-th row of U can be found in SN_inner in the half-open interval
            // [ U_begin[i],...,U_end[i] [
            Tensor1<   Int,LInt> U_begin;
            Tensor1<   Int,LInt> U_end;
            
            // Values of triangular part of k-th supernode is stored in
            // [ SN_tri_vals[SN_tri_ptr[k]],...,SN_tri_vals[SN_tri_ptr[k]+1] [
            Tensor1<  LInt, Int> SN_tri_ptr;
            Tensor1<Scalar,LInt> SN_tri_vals;
            
            // Values of rectangular part of k-th supernode is stored in
            // [ SN_rec_vals[SN_rec_ptr[k]],...,SN_rec_vals[SN_rec_ptr[k]+1] [
            Tensor1<  LInt, Int>   SN_rec_ptr;
            Tensor1<Scalar,LInt> SN_rec_vals;
            
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
                    tic("Initialize A_lo");
                    A_up = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                    toc("Initialize A_lo");
                    
                    // TODO: Is there a way to avoid this copy?
                    tic("Transpose");
                    A_lo = A_up.Transpose();
                    toc("Transpose");
                }
                else
                {
                    // TODO: Is there a way to avoid this copy?
                    tic("Initialize A_lo");
                    A_lo = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                    toc("Initialize A_lo");
                    
                    // TODO: Is there a way to avoid this copy?
                    tic("Transpose");
                    A_up = A_lo.Transpose();
                    toc("Transpose");
                }
                
//                FactorizeSymbolically();
                
                //            FactorizeSymbolically2();
            }
            
            const EliminationTree<Int> & GetEliminationTree()
            {
                if( ! eTree_initialized )
                {
                    tic(ClassName()+"::GetEliminationTree");
                    
                    // See Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                    
                    Tensor1<Int,Int> parents ( n, n );
                    Tensor1<Int,Int> buffer  ( n, n );
                    
                    const LInt * restrict const A_outer = A_lo.Outer().data();
                    const  Int * restrict const A_inner = A_lo.Inner().data();
                    
                    tic("Main loop");
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
                    toc("Main loop");
                    
                    eTree = EliminationTree<Int> ( std::move(parents) );
                    
                    eTree_initialized = true;
                    
                    toc(ClassName()+"::GetEliminationTree");
                }
                
                return eTree;
            }
            
            
            void FactorizeSymbolically()
            {
                // This is Algorithm 4.2 from  Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                
                tic(ClassName()+"::FactorizeSymbolically");
                
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

                toc(ClassName()+"::FactorizeSymbolically");
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
//####          Supernodal factorization
//###########################################################################################
            
            void FactorizeSymbolically_SN()
            {
                // This is Algorithm 4.3 from  Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
                
                // However, we reversed it: Instead of children "supplementing" their parents, the parents just pull from their children. This allows us to use less temporary memory.
                
                // Also we avoid storing the sparsity pattern of U in CSR format. Instead, we remember where we can find U's column indices of the i-th row within the row pointers SN_inner of the supernodes.

                tic(ClassName()+"::FactorizeSymbolically_SN");

                const LInt * restrict const A_rp      = A_up.Outer().data();
                const  Int * restrict const A_ci      = A_up.Inner().data();

                const  Int * restrict const parents   = GetEliminationTree().Parents().data();
                const  Int * restrict const child_ptr = GetEliminationTree().ChildPointers().data();
                const  Int * restrict const child_idx = GetEliminationTree().ChildIndices().data();

                Tensor1<Int,Int> U_i    ( n );  // An array to aggregate the rows of U.
                Tensor1<Int,Int> buffer ( n );  // Some scratch space for UniteSortedBuffers.
                Int row_counter;                // Holds the current number of indices in U_i.
                
                // column indices of i-th row of U can be found in SN_inner in the half-open interval
                // [ U_begin[i],...,U_end[i] [
                U_begin = Tensor1<LInt,Int> (n);
                U_end   = Tensor1<LInt,Int> (n);
                
                // Holds the current number of supernodes.
                SN_count = 0;
                // Pointers from supernodes to their starting rows.
                SN_rp    = Tensor1<LInt,Int> (n+1);
                
                // Pointers from supernodes to their starting position in SN_inner.
                SN_outer = Tensor1<LInt,Int> (n+1);
                SN_outer[0]  = 0;
                
                // To be filled with the column indices of super nodes.
                // Will later be moved to SN_inner.
                Aggregator<Int,LInt> SN_inner_agg ( 2 * A_up.NonzeroCount() );

                // TODO: So far we store all column indices of a supernode.
                // TODO: But this is redundant, because we know that the indices that belonging to the
                // TODO: triangle part are contiguous; so we actually need to store only
                // TODO:  - the starting row
                // TODO:  - the column index after the triangle, or equivalently, the supernode rowcount
                // TODO:    (these two are already stored in SN_rp)
                // TODO:  - the (scattered) column indices for the rectangular part -> SN_inner
                // TODO:  - the length of the rectangular part -> SN_outer
                // TODO:  - for each row its supernode (this might allow us to replace U_begin and U_end).
                // TODO:  - SN_tri_ptr and SN_rec_ptr as before.
                
                
                // The first row needs some special treatment because it does not have any predecessor.
                {
                    const Int i = 0;
                    // The nonzero pattern of A_up belongs definitely to the pattern of U.
                    row_counter = A_rp[i+1] - A_rp[i];
                    copy_buffer( &A_ci[A_rp[i]], U_i.data(), row_counter );
                    
                    // No children to travers for first row.
                    
                    // start first supernode
                    SN_rp[SN_count] = i;
                    ++SN_count;
                    // Copy U_i to new supernode.
                    SN_inner_agg.Push( U_i.data(), row_counter );
                    
                    SN_outer[SN_count] = SN_inner_agg.Size();
                    
                    // Remember where to find U_i within SN_inner_agg.
                    U_begin[i] = 0;
                    U_end  [i] = row_counter;
                }
                
                tic("Main loop");
                for( Int i = 1; i < n; ++i ) // Traverse rows.
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
                        
                        // We have to merge row pointers of child j  (different from j) into U_i

                        // We have to look up U_j \ {j} in SN_inner_agg in the interval [a,...,b[:
                        const Int a = U_begin[j]+1;  // This excludes the first entry j.
                        const Int b = U_end  [j]  ;
                        
                        if( b > a )
                        {
                            row_counter = UniteSortedBuffers(
                                U_i.data(),    row_counter,
                                &SN_inner_agg[a], b - a,
                                buffer.data()
                            );
                            swap( U_i, buffer );
                        }
                        
                    }
                    
                    const Int a = U_begin[i-1];
                    const Int b = U_end  [i-1];
                    
                    if( (i == parents[i-1]) && (b - a == row_counter + 1) )
                    {
                        // continue supernode
                        
                        // Remember where to find U_i within SN_inner_agg.
                        U_begin[i] = a+1;
                        U_end  [i] = b;
                    }
                    else
                    {
                        // start new supernode
                        SN_rp[SN_count] = i; // row i does not belong to previous supernode.
                        ++SN_count;
                        
                        // Copy U_i to new supernode.
                        SN_inner_agg.Push( U_i.data(), row_counter );
                        
                        SN_outer[SN_count] = SN_inner_agg.Size();
                        
                        // Remember where to find U_i within SN_inner_agg.
                        U_begin[i] = b;
                        U_end[i]   = U_begin[i] + row_counter;
                    }
                    
                } // for( Int i = 0; i < n; ++i )
                toc("Main loop");
                
                // finish last supernode
                SN_rp[SN_count] = n; // row n does not belong to previous supernode.
                
                dump(SN_count);
                
                SN_rp.Resize(SN_count+1);
                SN_outer.Resize(SN_count+1);
                SN_inner = std::move(SN_inner_agg.Get());

                SN_tri_ptr = Tensor1<LInt,Int> (SN_count+1);
                SN_tri_ptr[0] = 0;
                
                SN_rec_ptr = Tensor1<LInt,Int> (SN_count+1);
                SN_rec_ptr[0] = 0;
                
                tic("Computing number of nonzeroes");
                for( Int k = 0; k < SN_count; ++k )
                {
                    // Warning: Taking differences of potentially signed numbers.
                    // Should not be of concern because negative numbers appear here only of sumething went wron upstream.
                    const LInt SN_cols = SN_rp[k+1]    - SN_rp[k];
                    const LInt SN_rows = SN_outer[k+1] - SN_outer[k];
                    
                    SN_tri_ptr[k+1] = SN_tri_ptr[k] + SN_cols * SN_cols;
                    SN_rec_ptr[k+1] = SN_rec_ptr[k] + SN_cols * (SN_rows - SN_cols);
                }
                toc("Computing number of nonzeroes");
                
                tic("Allocating nonzero values");
                // Allocating memory for the nonzero values of the factorization.
                
                // TODO: Filling with 0 is not really needed.
                SN_tri_vals = Tensor1<Scalar, LInt> (SN_tri_ptr[SN_count],0);
                SN_rec_vals = Tensor1<Scalar, LInt> (SN_rec_ptr[SN_count],0);
                toc("Allocating nonzero values");
                
                toc(ClassName()+"::FactorizeSymbolically_SN");
            }
            
            
            
            
            
            
//###########################################################################################
//####          Supernodal back substitution
//###########################################################################################
            
//            template<int nrhs>
//            void U_Solve_Sequential_SN( Scalar * restrict const b )
//            {
//                tic("U_Solve_Sequential_SN<"+ToString(nrhs)+">");
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
//                    const Int l_mid   = l_begin + n_0;
//
//                    const Int n_1 = l_end - l_mid;
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
//                            copy_buffer<nrhs>( &b[ nrhs * SN_inner[l_mid+j]], &x_1[nrhs * j] );
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
//                toc("U_Solve_Sequential_SN<"+ToString(nrhs)+">");
//            }
//
//            template<int nrhs_lo, int nrhs_hi>
//            void U_Solve_Sequential_SN( Scalar * restrict const b, const int nrhs )
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
            
            void U_Solve_Sequential_SN( Scalar * restrict const B, const Int nrhs )
            {
                tic("U_Solve_Sequential_SN");
                // Solves U * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    U_Solve_Sequential_SN(B);
                    toc("U_Solve_Sequential_SN");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = SN_count; k --> 0; )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    const Int l_mid   = l_begin + n_0;
                    
                    const Int n_1 = l_end - l_mid;
                    
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
                        copy_buffer( &B[nrhs * SN_inner[l_mid+j]], &X_1[nrhs * j], nrhs );
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
                toc("U_Solve_Sequential_SN");
            }
            
            void U_Solve_Sequential_SN( Scalar * restrict const b )
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
                    const Int l_mid   = l_begin + n_0;

                    const Int n_1 = l_end - l_mid;

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
                                A_1x_1 += A_1[j] * b[SN_inner[l_mid+j]];
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
                                x_1[j] = b[SN_inner[l_mid+j]];
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
            
            void L_Solve_Sequential_SN( Scalar * restrict const B, const Int nrhs )
            {
                tic("L_Solve_Sequential_SN");
                // Solves L * X = B and stores the result back into B.
                // Assumes that B has size n x rhs_count.
             
                if( nrhs == 1 )
                {
                    L_Solve_Sequential_SN(B);
                    toc("L_Solve_Sequential_SN");
                    return;
                }
                
                // Some scratch space to read parts of x that belong to a supernode's rectangular part.
                Tensor2<Scalar,Int> X_buffer ( n, nrhs );
                
                for( Int k = 0; k< SN_count; ++k )
                {
                    const Int n_0 = SN_rp[k+1] - SN_rp[k];
                    
                    const Int l_begin = SN_outer[k  ];
                    const Int l_end   = SN_outer[k+1];
                    const Int l_mid   = l_begin + n_0;
                    
                    const Int n_1 = l_end - l_mid;
                    
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
                        add_to_buffer( &X_1[nrhs * j], &B[nrhs * SN_inner[l_mid+j]], nrhs );
                    }
                }
                toc("L_Solve_Sequential_SN");
            }
            
            
            void L_Solve_Sequential_SN( Scalar * restrict const b )
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
                    const Int l_mid   = l_begin + n_0;

                    const Int n_1 = l_end - l_mid;

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
                                b[SN_inner[l_mid+j]] -= conj(A_1[j]) * x_0[0];
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
                                b[SN_inner[l_mid+j]] += x_1[j];
                            }
                        }
                    }
                   
                }
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
            
            const Tensor1<LInt,Int> & U_Begin() const
            {
                return U_begin;
            }
            
            const Tensor1<LInt,Int> & U_End() const
            {
                return U_end;
            }
            
            std::string ClassName()
            {
                return "Sparse::CholeskyDecomposition<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
            }
            
            
        }; // class CholeskyDecomposition
        
    } // namespace Sparse
        
} // namespace Tensors
