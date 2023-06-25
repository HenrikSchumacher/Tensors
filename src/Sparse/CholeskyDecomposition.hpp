#pragma once

#include <unordered_map>
#include <unordered_set>

//#include <Accelerate/Accelerate.h>
//#include <cblas.h>
//#include <lapacke.h>

//#include "../../MyBLAS.hpp"

#include "CholeskyDecomposition/Factorizer.hpp"

//#include "Metis_Wrapper.hpp"

// Priority I+++:

// TODO: Numeric factorization is incorrect when using tree_top_depth > 1!

// Priority I:

// TODO: Parallelize symbolic factorization.
// TODO:     --> Build aTree first and traverse it in parallel to determine SN_inner.

// TODO: Parallelize solve phases.

// Priority II:

// TODO: Compute nested dissection --> Metis, Scotch. Parallel versions? MT-Metis?

// TODO: Speed up supernode update in factorization phase.
//           --> transpose U_0 and U_1 to reduce scatter_reads/scatter_adds.
//           --> employ Tiny::BLAS kernels. --> Does not seem to be helpful...
//           --> is there a way to skip unrelevant descendants?

// TODO: Return permutation and factors (as sparse matrices) so that they can be checked.

// TODO: incomplete factorization?


// Priority III:
// TODO: hierarchical low-rank factorization of supernodes?

// TODO: Improve scheduling for parallel factorization.
// TODO: - What to do if top of the tree is not a binary tree?
// TODO: - What to do in case of a forest?
// TODO: - Estimate work to do in subtrees.

// Priority IV:
// TODO: Maybe load linear combination of matrices A (with sub-pattern, of course) during factorization?

// TODO: parallelize update of supernodes with many descendants.
//           --> fetching updates from descendants can be done in parallel

// TODO: parallelize potrf + trsm of large supernodes.
//           --> fetching updates from descendants can be done in parallel


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
            
            using BinaryMatrix_T = Sparse::BinaryMatrixCSR<Int,LInt>;
            using Matrix_T       = Sparse::MatrixCSR<Scal,Int,LInt>;

            friend class CholeskyFactorizer<Scal,Int,LInt>;
            
            using Factorizer = CholeskyFactorizer<Scal,Int,LInt>;

//            using VectorContainer_T = Tensor1<Scal,Int>;
            using VectorContainer_T = Tensor1<Scal,LInt>;

            
        protected:
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;
            
            const Int n = 0;
            const Int thread_count = 1;
            const Int tree_top_depth = 0;
            
            Permutation<Int>  perm;           // row and column permutation the nonzeros of the matrix.
            
            BinaryMatrix_T A;
            
            Permutation<LInt> A_inner_perm;   // permutation of the nonzero values. Needed for reading in
            
            Tensor1<Scal,LInt> A_val;
            Scal reg = 0;

            Matrix_T L;
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
            
            
            // Stores right hand side / solution during the solve phase.
            VectorContainer_T X;
            
            // Stores right hand side / solution during the solve phase.
            // Some scratch space to read parts of X that belong to a supernode's rectangular part.
            VectorContainer_T X_scratch;
            // TODO: If I want to parallelize the solve phase, I have to provide each thread with its own X_scratch.
            
        public:
            
            CholeskyDecomposition() = default;
            
            ~CholeskyDecomposition() = default;
            
            template<typename ExtLInt, typename ExtInt>
            CholeskyDecomposition(
                ptr<ExtLInt> outer_, ptr<ExtInt> inner_,
                Int n_, Int thread_count_, Int tree_top_depth_
            )
            :   n               ( std::max( izero, n_)  )
            ,   thread_count    ( std::max( ione, thread_count_)    )
            ,   tree_top_depth  ( std::max( ione, tree_top_depth_)  )
            ,   perm            ( n_, thread_count                   ) // use identity permutation
            ,   A               ( outer_, inner_, n, n, thread_count )
            ,   A_inner_perm    ( A.Permute( perm, perm )            )
            ,   A_val           ( outer_[n]                          )
            {
                Init();
            }
            
            template<typename ExtLInt, typename ExtInt, typename ExtInt2>
            CholeskyDecomposition(
                ptr<ExtLInt> outer_, ptr<ExtInt> inner_, ptr<ExtInt2> p_,
                Int n_, Int thread_count_, Int tree_top_depth_
            )
            :   n               ( std::max( izero, n_)     )
            ,   thread_count    ( std::max( ione, thread_count_)       )
            ,   tree_top_depth  ( std::max( ione, tree_top_depth_)     )
            ,   perm            ( p_, n, Inverse::False, thread_count   )
            ,   A               ( outer_, inner_, n, n, thread_count    )
            ,   A_inner_perm    ( A.Permute( perm, perm )               )
            ,   A_val           ( outer_[n]                             )
            {
                Init();
            }
            
            template<typename ExtLInt, typename ExtInt>
            CholeskyDecomposition(
                ptr<ExtLInt> outer_, ptr<ExtInt> inner_, Permutation<Int> && perm_,
                Int tree_top_depth_
            )
            :   n               ( std::max( izero, perm_.Size() )            )
            ,   thread_count    ( std::max( ione, perm_.ThreadCount())       )
            ,   tree_top_depth  ( std::max( ione, tree_top_depth_)           )
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
            
            const Tensor1<Int,Int> & PostOrdering()
            {
                auto & post = EliminationTree().PostOrdering();
                
                if( !EliminationTree().PostOrderedQ() )
                {
                    ptic(ClassName()+"::PostOrdering");
                    
                    perm.Compose( post, Compose::Post );
                    
                    A_inner_perm.Compose( std::move(A.Permute( post, post )), Compose::Post );
                    
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
            
//###########################################################################################
//####          Elimination tree
//###########################################################################################
            
        public:
            
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
                    
                    ptr<LInt> A_diag  = A.Diag().data();
                    ptr<LInt> A_outer = A.Outer().data();
                    ptr<Int>  A_inner = A.Inner().data();

                    for( Int k = 1; k < n; ++k )
                    {
                        // We need visit all i < k with A_ik != 0.
                        const LInt l_begin = A_outer[k];
                        const LInt l_end   = A_diag [k];
//                        const LInt l_end   = A_diag[k+1]-1;
                        
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

                    eTree = Tree<Int> ( std::move(parents), thread_count );
                    
//                    eTree = Tree<Int> ( parents, thread_count );

                    eTree_initialized = true;
                    
                    ptoc(ClassName()+"::EliminationTree");
                }
                
                return eTree;
            }

//###########################################################################################
//####          Assembly tree
//###########################################################################################
            
        protected:
            
            
        public:
            
            const Tree<Int> & AssemblyTree()
            {
                SymbolicFactorization();
                
                return aTree;
            }
            
            void CreateAssemblyTree()
            {
                ptic(ClassName()+"::CreateAssemblyTree");

                const Tensor1<Int,Int> & parents = EliminationTree().Parents();

                Tensor1<Int,Int> SN_parents ( SN_count );

                for( Int k = 0; k < SN_count-1; ++k )
                {
                    // Thus subtraction is safe as each supernode has at least one row.
                    Int last_row = SN_rp[k+1]-1;

                    Int last_rows_parent = parents[last_row];

                    SN_parents[k] = (last_rows_parent<n) ? row_to_SN[last_rows_parent] : SN_count;
                }
                
                SN_parents[SN_count-1] = SN_count;
                

                aTree = Tree<Int> ( std::move(SN_parents), thread_count );
//                aTree = Tree<Int> ( SN_parents, thread_count );

                ptoc(ClassName()+"::CreateAssemblyTree");
            }
        
            
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
