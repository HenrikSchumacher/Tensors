#pragma once

//How to use:

// Under macos simply include the following and link `-framework Accelerate`:
//
//  #define LAPACK_DISABLE_NAN_CHECK
//  #define ACCELERATE_NEW_LAPACK
//  #include <Accelerate/Accelerate.h>
//
// Alternatively (and on other operating systems), include the following and link `-lcblas -llapack`:
//
//  #include <cblas.h>
//  #include <lapack.h>
//

#include "../BLAS.hpp"
#include "../LAPACK.hpp"

#include "CholeskyDecomposition/Factorizer.hpp"
#include "CholeskyDecomposition/UpperSolver.hpp"
#include "CholeskyDecomposition/LowerSolver.hpp"

// Priority I:

// TODO: Improve scheduling for parallel factorization.
// DONE: - What to do if top of the tree is not a binary tree?
// DONE: - What to do in case of a forest?
// TODO: - Estimate work to do in subtrees.
// TODO: - Reorder `subrees` in `Tree` based on this cost estimate.

// TODO: Parallelize symbolic factorization.
// TODO:     --> Build aTree first and traverse it in parallel to determine SN_inner.

// Priority II:

// TODO: Add arguments for leading dimensions.
// TODO: Add multiplication and add-into possibilities.

// TODO: Compute nested dissection --> Metis, Scotch. Parallel versions? MT-Metis?

// TODO: Speed up supernode update in factorization phase.
//           --> transpose U_0 and U_1 to reduce scatter_reads/scatter_adds.
//           --> employ Tiny::BLAS kernels. --> Does not seem to be helpful...
//           --> is there a way to skip unrelevant descendants?
//           --> fetching updates from descendants can be done in parallel

// TODO: Do we really have to build _two_ EliminationTrees?

// Priority III:
// TODO: hierarchical low-rank factorization of supernodes?

// TODO: Allow the user to supply only upper or lower triangle of matrix.

// TODO: Optional iterative refinement?

// Priority IV:
// TODO: hierarchical low-rank factorization of supernodes?

// TODO: incomplete factorization?

// TODO: Maybe load linear combination of matrices A (with sub-pattern, of course) during factorization?

// TODO: parallelize potrf + trsm of large supernodes.
//           --> not a good idea if Apple Accelerate is used?!?


// Super helpful literature:
// Stewart - Building an Old-Fashioned Sparse Solver. http://hdl.handle.net/1903/1312

// TODO: On Computing Min-Degree Elimination Orderings. https://arxiv.org/pdf/1711.08446.pdf:

// TODO: Kayaaslan, Ucar - Reducing elimination tree height for parallel LU factorization of sparse unsymmetric matrices. https://hal.inria.fr/hal-01114413/document

// TODO: Liu - The Multifrontal Method for Sparse Matrix Solution: Theory and Practice. https://www.jstor.org/stable/2132786 !!

namespace Tensors
{
    namespace Sparse
    {   
        template<typename Scal_, typename Int_, typename LInt_>
        class CholeskyDecomposition : public CachedObject
        {
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
            using BinaryMatrix_T     = Sparse::BinaryMatrixCSR<Int,LInt>;
            using Matrix_T           = Sparse::MatrixCSR<Scal,Int,LInt>;
            
            using Factorizer         = CholeskyFactorizer<Scal,Int,LInt>;
            
            friend Factorizer;
            friend class UpperSolver<false,Scal,Int,LInt>;
            friend class UpperSolver<true, Scal,Int,LInt>;
            friend class LowerSolver<false,false,Scal,Int,LInt>;
            friend class LowerSolver<true, false,Scal,Int,LInt>;
            friend class LowerSolver<false,true, Scal,Int,LInt>;
            friend class LowerSolver<true, true, Scal,Int,LInt>;
            
            using VectorContainer_T = Tensor1<Scal,LInt>;

            
        protected:
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;
            
            const Int n = 0;
            const Int thread_count = 1;
            
            Permutation<Int>  perm;           // row and column permutation the nonzeros of the matrix.
            
            BinaryMatrix_T A;
            
            Tensor1<LInt,LInt> A_inner_perm; // permutation of the nonzero values. Needed for reading in.
            
            Tensor1<Scal,LInt> A_val;
            Scal reg = 0;

//            Matrix_T L;
//            Matrix_T U;
            
            // elimination tree
            bool eTree_initialized = false;
            Tree<Int> eTree;
            
            // assembly three
            Tree<Int> aTree;
            
            // Supernode data:
            
            bool SN_initialized = false;
            bool SN_factorized  = false;
            
            // Number of supernodes.
            Int SN_count = 0;
            
            // Pointers from supernodes to their rows.
            // k-th supernode has rows [ SN_rp[k],SN_rp[k]+1,...,SN_rp[k+1] [
            Tensor1<   Int, Int> SN_rp;
            // Pointers from supernodes to their starting position in SN_inner.
            Tensor1<  LInt, Int> SN_outer;
            // The column indices of rectangular part of the supernodes.
            Tensor1<   Int,LInt> SN_inner;
            
            // Hence k-th supernode has the following column indices:
            // triangular  part = [ i_begin, i_begin+1,...,i_end [
            // rectangular part = [
            //                      SN_inner[j_begin  ],
            //                      SN_inner[j_begin+1],
            //                      SN_inner[j_begin+2],
            //                      ...,
            //                      SN_inner[j_end]
            //                    [
            // where i_begin = SN_rp[k],
            //       i_end   = SN_rp[k+1],
            //       j_begin = SN_outer[k], and
            //       j_end   = SN_outer[SN_outer[k+1]].
            
            // i-th row of U belongs to supernode row_to_SN[i].
            Tensor1<   Int, Int> row_to_SN;
            
            // Hence the column indices of U for row i can are:
            // triangular  part = [ i_begin,i_begin+1,...,i_end [
            // rectangular part = [
            //                      SN_inner[j_begin  ],
            //                      SN_inner[j_begin+1],
            //                      SN_inner[j_begin+2],
            //                      ...,
            //                      SN_inner[j_end]
            //                    [
            // where i_begin = SN_rp[row_to_SN[i]] = i,
            //       i_end   = SN_rp[row_to_SN[i]+1],
            //       j_begin = SN_outer[row_to_SN[i]  ], and
            //       j_end   = SN_outer[row_to_SN[i]+1].
            
            // Values of triangular part of k-th supernode is stored in
            // [ SN_tri_val[SN_tri_ptr[k]],...,SN_tri_val[SN_tri_ptr[k]+1] [
            Tensor1<LInt, Int> SN_tri_ptr;
            Tensor1<Scal,LInt> SN_tri_val;
            
            // Values of rectangular part of k-th supernode is stored in
            // [ SN_rec_val[SN_rec_ptr[k]],...,SN_rec_val[SN_rec_ptr[k]+1] [
            Tensor1<LInt, Int> SN_rec_ptr;
            Tensor1<Scal,LInt> SN_rec_val;
            
            // Maximal size of triangular part of supernodes.
            Int max_n_0 = 0;
            // Maximal size of rectangular part of supernodes.
            Int max_n_1 = 0;
            
            Int nrhs = 1;
            
            // Stores right hand side / solution during the solve phase.
            VectorContainer_T X;
            
            // Stores right hand side / solution during the solve phase.
            // Some scratch space to read parts of X that belong to a supernode's rectangular part.
            VectorContainer_T X_scratch;
            // TODO: If I want to parallelize the solve phase, I have to provide each thread with its own X_scratch.
            
            std::vector<std::mutex> row_mutexes;
            
        public:
            
            CholeskyDecomposition() = default;
            
            ~CholeskyDecomposition() = default;
            
            template<typename ExtLInt, typename ExtInt>
            CholeskyDecomposition(
                cptr<ExtLInt> outer_,
                cptr<ExtInt>  inner_,
                Int n_, Int thread_count_
            )
            :   n               ( Max( izero, n_)                    )
            ,   thread_count    ( Max( ione, thread_count_)          )
            ,   perm            ( n_, thread_count                   ) // use identity permutation
            ,   A               ( outer_, inner_, n, n, thread_count )
            ,   A_inner_perm    ( A.Permute( perm, perm )            )
            ,   A_val           ( outer_[n]                          )
            {
                Init();
            }
            
            template<typename ExtLInt, typename ExtInt, typename ExtInt2>
            CholeskyDecomposition(
                cptr<ExtLInt> outer_,
                cptr<ExtInt>  inner_,
                cptr<ExtInt2> p_,
                Int n_, Int thread_count_
            )
            :   n               ( Max( izero, n_)                     )
            ,   thread_count    ( Max( ione, thread_count_)           )
            ,   perm            ( p_, n, Inverse::False, thread_count )
            ,   A               ( outer_, inner_, n, n, thread_count  )
            ,   A_inner_perm    ( A.Permute( perm, perm )             )
            ,   A_val           ( outer_[n]                           )
            {
                Init();
            }
            
            template<typename ExtLInt, typename ExtInt>
            CholeskyDecomposition(
                cptr<ExtLInt> outer_,
                cptr<ExtInt>  inner_,
                Permutation<Int> && perm_
            )
            :   n               ( Max( izero, perm_.Size() )                 )
            ,   thread_count    ( Max( ione, perm_.ThreadCount())            )
            ,   perm            ( std::move( perm_)                          )
            ,   A               ( outer_, inner_, n, n, perm.ThreadCount()   )
            ,   A_inner_perm    ( A.Permute( perm, perm )                    )
            ,   A_val           ( outer_[n]                                  )
            {
                Init();
            }
            
            
        protected:
            
            void Init()
            {
                ptic(ClassName());
                if( n <= izero )
                {
                    eprint(ClassName()+": Size n = "+ToString(n)+" of matrix is <= 0.");
                }
                                
                A.RequireDiag();
                
                CheckDiagonal();
                
                
                row_mutexes = std::vector<std::mutex> ( n );
                
                ptoc(ClassName());
            }
            
            
        public:
            
            void CheckDiagonal() const
            {
                ptic(ClassName()+"CheckDiagonal()");
                
                A.RequireDiag();
                
                bool okay = true;
                
                for( Int i = 0; i < n; ++i )
                {
                    okay = okay && (A.Inner(A.Diag(i)) == i);
                }
                
                if( !okay )
                {
                    eprint(ClassName()+"::PostOrdering: Diagonal of input matrix is not marked as nonzero.");
                }
                
                ptoc(ClassName()+"CheckDiagonal()");
            }
            
        protected:
            
            cref<Tensor1<Int,Int>> PostOrdering()
            {
                cref<Permutation<Int>> post = EliminationTree().PostOrdering();
                
                if( !EliminationTree().PostOrderedQ() )
                {
                    ptic(ClassName()+"::PostOrdering");
                    
                    perm.Compose( post, Compose::Post );
                    
                    
                    // `post` will reorder the inner indices; hence, we have to reorder also `A_inner_perm`;
                    // Otherwise, `Factorizer` will read the wrong nonzero values.
                    {
                        Tensor1<LInt,LInt> inner_perm_perm = A.Permute( post, post );
                        
                        
                        // A_inner_perm.Compose( std::move(A.Permute( post, post )), Compose::Post );
                        
                        cptr<LInt> p = A_inner_perm.data();
                        mptr<LInt> q = inner_perm_perm.data();
                        
                        ParallelDo(
                            [p,q]( const LInt i )
                            {
                               q[i] = p[q[i]];
                            },
                            A_inner_perm.Size(), static_cast<LInt>(thread_count)
                        );
                        
                        swap(A_inner_perm,inner_perm_perm);
                    }
                    
                    
                    A.RequireDiag();
                    
                    CheckDiagonal();
                    
                    eTree_initialized = false;
                    SN_initialized    = false;
                    SN_factorized     = false;
                    
                    // TODO:  Is there a cheaper way to generate the correct tree,
                    // TODO:  e.g., by permuting the old tree?
                    (void)EliminationTree();
                    
                    ptoc(ClassName()+"::PostOrdering");
                }
                
                return EliminationTree().PostOrdering().GetPermutation();
            }
            
#include "CholeskyDecomposition/EliminationTree.hpp"
#include "CholeskyDecomposition/AssemblyTree.hpp"
#include "CholeskyDecomposition/InputOutput.hpp"
#include "CholeskyDecomposition/Symbolic.hpp"
#include "CholeskyDecomposition/Numeric.hpp"
#include "CholeskyDecomposition/Solve.hpp"
    
        public:
            
            
            std::string ClassName() const
            {
                return std::string("Sparse::CholeskyDecomposition")+"<"+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }

            
        }; // class CholeskyDecomposition
        
    } // namespace Sparse
    
} // namespace Tensors





// =========================================================
// DONE: Reordering in the solve phase.
//          --> Copy-cast during pre- and post-permutation.
//          --> ReadRightHandSide, WriteSolution

// DONE: Load A + eps * Id during factorization.

// DONE:Call SymbolicFactorization, NumericFactorization,... when dependent routines are called.

// DONE: Currently, EliminationTree breaks down if the matrix is reducible.
//           --> What we need is an EliminationForest!
//           --> Maybe it just suffices to append a virtual root (that is not to be factorized).



// DONE: Automatically determine postordering and apply it!

// DONE: Allow the user to supply a permutation.

// DONE: Parallelized, abstract postorder traversal of Tree

// DONE: Specialization of the cases m_0 = 1 and n_0 = 1.

// DONE: Return permutation (as sparse matrices) so that they can be checked. (< 2023-06-25)

// DONE: Return factors (as sparse matrices) so that they can be checked. (< 2023-06-25)

// DONE: Parallelize upper solve phase. (2023-06-26)

// DONE: Parallelize lower solve phase. (2032-07-30)

// DONE: User interface for lower/upper solves. (2032-07-30)

// DONE: A_inner_perm seems to be a bit wasteful; we neither need the inverse permutation nor scratch. (2032-07-30)
