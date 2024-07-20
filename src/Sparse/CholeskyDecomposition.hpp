#pragma once

#include "../../BLAS_Wrappers.hpp"
#include "../../LAPACK_Wrappers.hpp"

#include "CholeskyDecomposition/LeftLooking.hpp"
#include "CholeskyDecomposition/Multifrontal.hpp"
#include "CholeskyDecomposition/UpperSolver.hpp"
#include "CholeskyDecomposition/LowerSolver.hpp"

// Priority I:

// TODO: Compute AMD reordering. Often it works very well!

// TODO: LowerSolve seems to be unneccessary slow for nrhs = 1 and thread_count = 1.

// TODO: Improve scheduling for parallel factorization.
// TODO: - What to do if top of the tree is not a binary tree?
// DONE: - What to do in case of a forest?
// TODO: - Estimate work to do in subtrees.
// TODO: - Reorder `subrees` in `Tree` based on this cost estimate.

// Priority II:

// TODO: Parallelize symbolic factorization.
// TODO: - Build aTree first and traverse it in parallel to determine SN_inner.

// TODO: Do we really have to build _two_ EliminationTrees?
// TODO: Can we parallelize EliminationTrees? E.g., build it for chunks of rows and then merge the trees somehow?

// TODO: Improve solve phase:
// TODO: - trsm/trsv like interface
// TODO: - Fixed size arithmetic?
// TODO: - leading dimensions
// TODO: - multiplication and add-into

// TODO: Compute nested dissection --> Metis, Scotch. Parallel versions? MT-Metis?


// Priority III:

// TODO: hierarchical low-rank factorization of supernodes?
//      --> Superfast Multifrontal Method for Large Structured Linear Systems of Equations
//
// TODO: Iterative refinement -> CG solver?

// Priority IV:
// TODO: incomplete factorization?

// TODO: Maybe load linear combination of matrices A (with sub-pattern, of course) during factorization?

// TODO: parallelize potrf + trsm + herk of large supernodes.
// DONT: not helpful if Apple Accelerate is used?!?


// TODO: Speed up supernode update in left-looking factorization phase.
// TODO: - transpose U_0 and U_1 to reduce scatter_reads/scatter_adds.
// DONT: - fetching updates from descendants can be done in parallel


// DON'Ts:

// DONT: Allow the user to supply only upper or lower triangle of matrix.
//           --> I think this feature is seldomly used and creates some bad incentives.
//           --> iterative refinement would need the whole matrix anyways to be fast.

// DONT: employ Tiny::BLAS kernels. --> Does not seem to be helpful...
//           --> Did not help; at least on M1, the BLAS kernels are fast also for small sizes.


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
        class alignas( ObjectAlignment ) CholeskyDecomposition : public CachedObject
        {
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
            using BinaryMatrix_T     = Sparse::BinaryMatrixCSR<Int,LInt>;
            using Matrix_T           = Sparse::MatrixCSR<Scal,Int,LInt>;
            using Tree_T             = Tree<Int>;
            using Permutation_T      = Permutation<Int>;
            using Factorizer_LL_T    = CholeskyFactorizer_LeftLooking<Scal,Int,LInt>;
            
            using Factorizer_MF_T    = CholeskyFactorizer_Multifrontal<Scal,Int,LInt>;
            
            using Update_T           = Tensor2<Scal,Int>;
//            using Update_T           = Scal *;
            
            friend Factorizer_LL_T;
            friend Factorizer_MF_T;
            
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
            
            Int n = 0;
            Int thread_count = 1;
            
            Permutation_T perm;          // row and column permutation of the nonzeros of the matrix.
            
            BinaryMatrix_T A;
            
            Tensor1<LInt,LInt> A_inner_perm; // permutation of the nonzero values. Needed for reading in.
            
            Tensor1<Scal,LInt> A_val;
            Scal reg = 0;

//            Matrix_T L;
//            Matrix_T U;
            
            // elimination tree
            bool eTree_initialized = false;
            Tree_T eTree;
            
            // assembly tree
            Tree_T aTree;
            
            // Supernode data:
            
            Int  amalgamation_threshold = 4;
            bool SN_initialized = false;
            bool SN_factorized  = false;
            signed char SN_strategy = 0;
            

            
            // Number of supernodes.
            Int SN_count = 0;
            
            // Pointers from supernodes to their rows.
            // k-th supernode has rows [ SN_rp[k],SN_rp[k]+1,...,SN_rp[k+1] [
            Tensor1< Int, Int> SN_rp;
            // Pointers from supernodes to their starting position in SN_inner.
            Tensor1<LInt, Int> SN_outer;
            // The column indices of rectangular part of the supernodes.
            Tensor1< Int,LInt> SN_inner;
            
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
            Tensor1< Int, Int> row_to_SN;
            
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
            
            // Values of triangular part of s-th supernode is stored in
            // [ SN_tri_val[SN_tri_ptr[s]],...,SN_tri_val[SN_tri_ptr[s]+1] [
            Tensor1<LInt, Int> SN_tri_ptr;
            Tensor1<Scal,LInt> SN_tri_val;
            
            // Values of rectangular part of s-th supernode is stored in
            // [ SN_rec_val[SN_rec_ptr[s]],...,SN_rec_val[SN_rec_ptr[s]+1] [
            Tensor1<LInt, Int> SN_rec_ptr;
            Tensor1<Scal,LInt> SN_rec_val;
            
            std::vector<Update_T> SN_updates;
            
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
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                Permutation_T && perm_
            )
            :   n               ( Max( izero, perm_.Size() )      )
            ,   thread_count    ( Max( ione, perm_.ThreadCount()) )
            {
                std::string tag = ClassName()+"( "+ TypeName<ExtLInt> + "*, "+ TypeName<ExtInt> + "*,  Permutation<" + TypeName<Int>+ "> )";
                
                ptic(tag);
                
                perm = Permutation_T( std::move( perm_) );
                
                A = BinaryMatrix_T( A_outer, A_inner, n, n, thread_count );
                
                A_val = Tensor1<Scal,LInt>( A.NonzeroCount() );
                
                
                // The matrix reordering is parallelized.
                // But when run single-threaded, it is better to avoid it.
                
//                if( thread_count == 1 )
//                {
//                    Tensor1<Int,Int> parents ( n );
//                    
//                    (void)PermutedEliminationTreeParents(
//                        n, A_outer, A_inner,
//                        perm.GetPermutation().data(),
//                        perm.GetInversePermutation().data(),
//                        parents.data()
//                    );
//                    
//                    eTree = Tree<Int>( std::move(parents), thread_count );
//                    
//                    if( eTree.PostOrderedQ() )
//                    {
//                        eTree_initialized = true;
//                    }
//                    else
//                    {
//                        perm.Compose( eTree.PostOrdering(), Compose::Post );
//                        
//                        eTree = Tree<Int>();
//                    }
//                }
                
                A_inner_perm = A.Permute( perm, perm );
                
                Init();
                
                ptoc(tag);
            }
            
            // This is the constructor that will most likely to be used in practice.
            // p is supposed to be a vector of size n_ containing a fill-in reducing permutation of [0,...,n[.
            template<typename ExtLInt, typename ExtInt, typename ExtInt2>
            CholeskyDecomposition(
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                cptr<ExtInt2> p,
                Int n_, Int thread_count_
            )
            :   CholeskyDecomposition(
                    A_outer, A_inner, Permutation_T( p, n_, Inverse::False, thread_count_ )
                )
            {}

            
            
            // Constructor if the user has applied a fill-in
            // reducing permutation to the matrix already.
            template<typename ExtLInt, typename ExtInt>
            CholeskyDecomposition(
                cptr<ExtLInt> A_outer,
                cptr<ExtInt>  A_inner,
                Int n_, Int thread_count_
            )
            :   n               ( n_                        )
            ,   thread_count    ( Max( ione, thread_count_) )
            {
                std::string tag = ClassName()+"( "+ TypeName<ExtLInt> + "*, "+ TypeName<ExtInt> + "*, " + TypeName<Int>+ ", " + TypeName<Int>+ " )";
                
                ptic(tag);
                
                A = BinaryMatrix_T( A_outer, A_inner, n, n, thread_count );
                
                A_val = Tensor1<Scal,LInt>( A.NonzeroCount() );
                
//                if( thread_count > 1 )
//                {
                    perm = Permutation_T( n_, thread_count ); // use identity permutation
                        
                    A_inner_perm = Tensor1<LInt,LInt>( A.NonzeroCount() );

                    A_inner_perm.iota( thread_count );
//                }
//                else
//                {
//                    Tensor1<Int,Int> parents ( n );
//                    
//                    (void)EliminationTreeParents( n, A_outer, A_inner, parents.data() );
//                    
//                    eTree = Tree<Int>( std::move(parents), thread_count );
//                    
//                    if( eTree.PostOrderedQ() )
//                    {
//                        perm = Permutation_T( n_, thread_count ); // use identity permutation
//                        
//                        A_inner_perm = Tensor1<LInt,LInt>( A.NonzeroCount() );
//                        
//                        A_inner_perm.iota( thread_count );
//                        
//                        eTree_initialized = true;
//                    }
//                    else
//                    {
//                        perm = eTree.PostOrdering();
//                        
//                        A_inner_perm = A.Permute( perm, perm );
//                        
//                        eTree = Tree<Int>();
//                    }
//                }
                
                Init();
                
                ptoc(tag);
            }

            
            
            
            /* Copy constructor */
            CholeskyDecomposition( const CholeskyDecomposition & other )
            :   CachedObject        ( other                     )
            ,   n                   ( other.n                   )
            ,   thread_count        ( other.thread_count        )
            ,   perm                ( other.perm                )
            ,   A                   ( other.A                   )
            ,   A_inner_perm        ( other.A_inner_perm        )
            ,   A_val               ( other.A_val               )
            ,   eTree_initialized   ( other.eTree_initialized   )
            ,   eTree               ( other.eTree               )
            ,   aTree               ( other.aTree               )
            ,   SN_initialized      ( other.SN_initialized      )
            ,   SN_count            ( other.SN_count            )
            ,   SN_rp               ( other.SN_rp               )
            ,   SN_outer            ( other.SN_outer            )
            ,   SN_inner            ( other.SN_inner            )
            ,   row_to_SN           ( other.row_to_SN           )
            ,   SN_tri_ptr          ( other.SN_tri_ptr          )
            ,   SN_tri_val          ( other.SN_tri_val          )
            ,   SN_rec_ptr          ( other.SN_rec_ptr          )
            ,   SN_rec_val          ( other.SN_rec_val          )
            ,   SN_updates          ( other.SN_updates          )
            ,   max_n_0             ( other.max_n_0             )
            ,   max_n_1             ( other.max_n_1             )
            ,   nrhs                ( other.nrhs                )
            ,   X                   ( other.X                   )
            ,   X_scratch           ( other.X_scratch           )
            ,   row_mutexes         ( n                         )
            {
                Init();
            }
            
            /* Swap function */
            friend void swap (CholeskyDecomposition & A_, CholeskyDecomposition & B_ ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( static_cast<CachedObject&>(A_), static_cast<CachedObject&>(B_) );
                swap( A_.n,                  B_.n                     );
                swap( A_.thread_count,       B_.thread_count          );
                swap( A_.A,                  B_.A                     );
                swap( A_.A_inner_perm,       B_.A_inner_perm          );
                swap( A_.A_val,              B_.A_val                 );
                swap( A_.eTree_initialized,  B_.eTree_initialized     );
                swap( A_.eTree,              B_.eTree                 );
                swap( A_.aTree,              B_.aTree                 );
                swap( A_.SN_initialized,     B_.SN_initialized        );
                swap( A_.SN_count,           B_.SN_count              );
                swap( A_.SN_rp,              B_.SN_rp                 );
                swap( A_.SN_outer,           B_.SN_outer              );
                swap( A_.SN_inner,           B_.SN_inner              );
                swap( A_.row_to_SN,          B_.row_to_SN             );
                swap( A_.SN_tri_ptr,         B_.SN_tri_ptr            );
                swap( A_.SN_tri_val,         B_.SN_tri_val            );
                swap( A_.SN_rec_ptr,         B_.SN_rec_ptr            );
                swap( A_.SN_rec_val,         B_.SN_rec_val            );
                swap( A_.SN_updates,         B_.SN_updates            );
                swap( A_.max_n_0,            B_.max_n_0               );
                swap( A_.max_n_1,            B_.max_n_1               );
                swap( A_.nrhs,               B_.nrhs                  );
                swap( A_.X,                  B_.X                     );
                swap( A_.X_scratch,          B_.X_scratch             );
                swap( A_.row_mutexes,        B_.row_mutexes           );
            }
            
            
            /* Copy assignment operator */
            CholeskyDecomposition & operator=( CholeskyDecomposition other )
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                swap(*this, other);

                return *this;
            }
            
//            /* Move constructor */
//            CholeskyDecomposition( CholeskyDecomposition && other ) noexcept
//            {
//                swap(*this, other);
//            }
            
//            /* Move assignment operator */
//            CholeskyDecomposition & operator=( CholeskyDecomposition && other ) noexcept
//            {
//                if( this == &other )
//                {
//                    wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
//                }
//                swap( *this, other );
//                return *this;
//            }
            
            
            
        protected:
            
            void Init()
            {
                ptic(ClassName()+"::Init");
                if( n <= izero )
                {
                    eprint(ClassName()+": Size n = "+ToString(n)+" of matrix is <= 0.");
                }
                                
                A.RequireDiag();
                
                CheckDiagonal();
                
                row_mutexes = std::vector<std::mutex> ( n );
                
                ptoc(ClassName()+"::Init");
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
                cref<Permutation_T> post = EliminationTree().PostOrdering();
                
                if( !EliminationTree().PostOrderedQ() )
                {
                    ptic(ClassName()+"::PostOrdering");
                    
                    perm.Compose( post, Compose::Post );
                    
                    
                    // `post` will reorder the inner indices; hence, we have to reorder also `A_inner_perm`;
                    // Otherwise, `Factorizer_LL_T` will read the wrong nonzero values.
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
            
            bool NumericallyFactorizedQ() const
            {
                return SN_factorized;
            }
            
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

// DONE: Skip some unrelevant descendants in left-looking factorization.
