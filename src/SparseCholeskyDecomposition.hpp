#pragma once

#include <unordered_map>
#include <unordered_set>

// Super helpful literature:
// Stewart - Building an Old-Fashioned Sparse Solver

// TODO: https://arxiv.org/pdf/1711.08446.pdf: On Computing Min-Degree Elimination Orderings

// TODO: Find supernodes: https://www.osti.gov/servlets/purl/6756314

// TODO: https://hal.inria.fr/hal-01114413/document

// TODO: https://www.jstor.org/stable/2132786 !!

namespace Tensors
{
    enum class Triangular : bool
    {
        Upper = true,
        Lower = false
    };
    
    
    template<typename Scalar, typename Int, typename LInt>
    class SparseCholeskyDecomposition
    {
    public:
        
        using SparseMatrix_T = SparseBinaryMatrixCSR<Int,LInt>;
        
        using List_T = SortedList<Int,Int>;

        
    protected:
        
        const Int n = 0;
        const Int thread_count = 1;
        const Triangular uplo  = Triangular::Upper;
        
        Tensor1<Int,Int> eTree;
        
        SparseMatrix_T A_lo;
        SparseMatrix_T A_up;
        
        SparseMatrix_T L;
        SparseMatrix_T U;
        
        
    public:
        
        SparseCholeskyDecomposition() = default;
        
        ~SparseCholeskyDecomposition() = default;
        
        SparseCholeskyDecomposition(
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
                tic("Initialize A_lo");
                
                A_up = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                toc("Initialize A_lo");
                
                tic("Transpose");
                A_lo = A_up.Transpose();
                toc("Transpose");
            }
            else
            {
                tic("Initialize A_lo");
                
                A_lo = SparseMatrix_T( outer_, inner_, n, n, thread_count );
                toc("Initialize A_lo");
                
                tic("Transpose");
                A_up = A_lo.Transpose();
                toc("Transpose");
            }

            FactorizeSymbolically();
            
//            FactorizeSymbolically2();
        }
        
        void FactorizeSymbolically()
        {
            // This is Aglorithm 8.1 from  Liu - The Role of Elimination Trees in Sparse Factorizations
            
            // TODO: Read about parallel variant in :
            //
            // Tarjan and Yannakakis - Simple linear-time algorithms to test chordality of graphs, test acyclicity of hypergraphs, and selectively reduce acyclic hypergraphs
            tic("FactorizeSymbolically");

            tic("Prepare");

            const LInt * restrict const A_rp = A_up.Outer().data();
            const  Int * restrict const A_ci = A_up.Inner().data();

            Tensor1<Int,Int> U_i    ( n );  // An array to aggregate the rows of U.
            Tensor1<Int,Int> buffer ( n );  // Some scratch space for UniteSortedBuffers.
            Int row_counter;                // Holds the current number of indices in U_i.

            eTree = Tensor1<Int,Int>( n, n );

            constexpr Int child_threshold = 2;
            Tensor2<Int,Int> child_info (n, child_threshold+1, 0);      // First entry in row is no. of children; followed by the children.
            std::unordered_map<Int,std::vector<Int>> children_galore;   // If there are mire than child_threshold children, we push them here.

            Tensor1<LInt,Int> U_rp (n+1);                          // To be filled with the row pointers of U.
            U_rp[0] = 0;
            Aggregator<Int,LInt> U_ci ( 2 * A_up.NonzeroCount() ); // To be filled with the column indices of U.

            toc("Prepare");

            tic("Main loop");
            for( Int i = 0; i < n; ++i )
            {
                // The nonzero pattern of A_up belongs definitely to the pattern of U.
                row_counter = A_rp[i+1] - A_rp[i];
                copy_buffer( &A_ci[A_rp[i]], U_i.data(), row_counter );

                // Traverse all children of i in the eTree. Most of the time it's a single child or no one at all. Sometimes it's two or more.
                for( Int l = 0; l < child_info(i,0); ++l )
                {
                    const Int j = ( l < child_threshold ) ? child_info(i,l+1) : children_galore[i][l-child_threshold];

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

                // Update the eTree.
                if( row_counter > 1 )       // U_i is sorted, thus U_i[0] = i. But i cannot be parent if i.
                {
                    const Int j = U_i[1];    // First row entry strictly right of column i.
                    eTree[i] = j;

                    const Int child_count = child_info(j,0);

                    if( child_count < child_threshold )
                    {
                        child_info(j, child_count+1 ) = i;
                    }
                    else
                    {
                        children_galore[j].push_back(i);
                    }
                    ++child_info(j,0);
                }
            } // for( Int i = 0; i < n; ++i )

            toc("Main loop");

            // rows_to_cols encodes now G_{n}(A); we are done.


            tic("Create U");
            U = SparseMatrix_T( std::move(U_rp), std::move(U_ci.Get()), n, n, thread_count );
            toc("Create U");

            toc("FactorizeSymbolically");
        }
        
        
//        void FactorizeSymbolically2()
//        {
//            This is Aglorithm 8.1 from  Liu - The Role of Elimination Trees in Sparse Factorizations
//            tic("FactorizeSymbolically2");
//
//            tic("Prepare");
//
//            const LInt * restrict const A_rp = A_up.Outer().data();
//            const  Int * restrict const A_ci = A_up.Inner().data();
//
//            Tensor1<Int,Int> U_i    ( n );
//            Tensor1<Int,Int> buffer ( n );
//
//            Int row_counter;
//
//
//
//            eTree = Tensor1<Int,Int>( n, n );
//            std::vector<std::vector<Int>> children (n);
//
//            // Idea of using "baby_sitter" is taken from Stewart - Building an Old-Fashioned Sparse Solver
              // It is also mentioned as CLIST on page 159 of  Liu - The Role of Elimination Trees in Sparse Factorizations
//            const Int no_child = n + 1;
//            Tensor1<Int,Int> baby_sitter ( n, no_child );
//
//            Tensor1<LInt,Int> U_rp ( n + 1 );
//            U_rp[0] = 0;
//
//            std::vector<Int> U_ci (A_up.NonzeroCount() );
//
//            toc("Prepare");
//
//            tic("Main loop");
//            for( Int i = 0; i < n; ++i )
//            {
//                row_counter = A_rp[i+1] - A_rp[i];
//                copy_buffer( &A_ci[A_rp[i]], U_i.data(), row_counter );
//
//                // Use baby_sitter to visit all children of i.
//                {
//                    Int j = baby_sitter[i];
//
//                    while( j != no_child )
//                    {
//                        // Merge row pointers of child j into U_i
//                        const Int _begin = U_rp[j]+1;  // This excludes U_ci[U_rp[j]] == j.
//                        const Int _end   = U_rp[j+1];
//                        if( _end > _begin )
//                        {
//                            row_counter = UniteSortedBuffers(
//                                U_i.data(),    row_counter,
//                                &U_ci[_begin], _end - _begin,
//                                buffer.data()
//                            );
//
//                            std::swap( U_i, buffer );
//                        }
//
//                        const Int j_temp = baby_sitter[j];
//                        baby_sitter[j] = no_child;
//                        j = j_temp;
//                    }
//                }
//
//                // Copy U_i to i-th row of U.
//                U_rp[i+1] = U_rp[i] + row_counter;
//                U_ci.resize( U_rp[i+1] );
//                copy_buffer( U_i.data(), &U_ci[U_rp[i]], row_counter );
//
//                // Update baby_sitter and eTree.
//                if( row_counter > 1 )
//                {
//                    Int j = U_i[1]; // j is parent of i.
//                    eTree[i] = j;
//
//                    Int j_temp;
//                    // traverse children of j until a free place is found.
//                    while( j != no_child )
//                    {
//                        j_temp = j;
//                        j = baby_sitter[j_temp];
//                    }
//
//                    baby_sitter[j_temp] = i;
//                }
//            }
//            toc("Main loop");
//
//            // rows_to_cols encodes now G_{n}(A); we are done.
//
//            tic("Initialize U");
//            U = SparseMatrix_T( &U_rp[0], &U_ci[0], n, n, thread_count );
//
//            toc("Initialize U");
//
//            toc("FactorizeSymbolically2");
//        }
        
        void FactorizeSymbolically_Old()
        {
            tic("FactorizeSymbolically_Old");
            
            std::vector<List_T> L_cols (n);
            
            const LInt * restrict const cp = A_up.Outer().data();
            const  Int * restrict const ri = A_up.Inner().data();
            
            tic("Reserve");
            for( Int i = 0; i < n; ++i )
            {
                const LInt k_begin = cp[i  ];
                const LInt k_end   = cp[i+1];
                
                L_cols[i].Reserve(2 *(k_end-k_begin));
            }
            toc("Reserve");
            
            LInt L_nnz = 0;
            
            
            List_T nonempty_rows;
            nonempty_rows.Reserve(n);
            
            
            tic("Main loop");
            for( Int j = 0; j < n; ++j )
            {
//                dump(j);
//                std::unordered_map<Int,bool> s_rows;
                
                const LInt k_begin = cp[j  ];
                const LInt k_end   = cp[j+1];
                
                // Push all nonzero rows in A_lo's column j onto s_rows.
                for( LInt k = k_begin; k < k_end; ++k )
                {
//                    s_rows[ri[k]] = true;
                    const Int i = ri[k];
                    L_cols[i].PushBack(j);
                    ++L_nnz;
                    nonempty_rows.Insert(i);
                }
            
                const List_T & u = L_cols[j];
                
                if( !u.Empty() )
                {
                    const Int row_begin = nonempty_rows.FindPosition(j);
                    const Int row_end   = nonempty_rows.Size();
                    
                    #pragma omp parallel for num_threads( thread_count ) reduction( + : L_nnz ) schedule( static )
                    for( Int row = row_begin; row < row_end; ++row )
                    {
                        const Int i = nonempty_rows[row];
                        
                        if( IntersectingQ( u, L_cols[i] ) )
                        {
                            // L_cols[j] and L_cols[i] intersect. Add position {i,j} as fill-in.
                            if( L_cols[i].Max() != j ) // // We know already that L_cols[i] is nonempty, so using .back() is safe.
                            {
                                L_cols[i].PushBack(j);
                                ++L_nnz;
                            }
                        }
                    }
                }
//
//                // Now s_rows contains all nonzero rows of the j-th column of L.
////                print("Write fill-in.");
//
//
////                std::stringstream s;
////                s << "s_rows = { ";
//                for( auto row : s_rows )
//                {
//                    const Int i = row.first;
////                    s << i <<", ";
////                    valprint("L_cols["+ToString(i)+"]",L_cols[i]);
//                    L_cols[i].push_back(j);
////                    valprint("L_cols["+ToString(i)+"]",L_cols[i]);
//                }
////                s <<" }";
////                print(s.str());
            }
            toc("Main loop");
            
            tic("Initialize L");
            L = SparseMatrix_T( n, n, L_nnz, thread_count );
            
//            print("Create row pointers of L.");
            {
                LInt * restrict const outer = L.Outer().data();
                
                outer[0] = 0;
                
                for( Int i = 0; i < n; ++i )
                {
                    outer[i+1] = outer[i] + L_cols[i].Size();
                }
            }
            
//            print("Copy column indices into L.");
            {
                const LInt * restrict const outer = L.Outer().data();
                       Int * restrict const inner = L.Inner().data();
                
                for( Int i = 0; i < n; ++i )
                {
                    LInt i_nnz = outer[i+1] - outer[i];
                    
                    copy_buffer( &L_cols[i][0], &inner[outer[i]], i_nnz );
                }
            }
            toc("Initialize L");
            
            toc("FactorizeSymbolically_Old");
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
        
        const Tensor1<Int,Int> & GetEliminationTree() const
        {
            return eTree;
        }
        
        void FillGraph()
        {
            tic("FillGraph");
            
            tic("Prepare");
            
            const LInt * restrict const rp = A_up.Outer().data();
            const  Int * restrict const ci = A_up.Inner().data();
            
            std::vector< List_T > rows_to_cols (n);
            
            for( Int i = 0; i < n; ++i )
            {
                const Int k_begin = rp[i  ];
                const Int k_end   = rp[i+1];
                
                const Int len = k_end-k_begin;
                
                rows_to_cols[i] = List_T( &ci[k_begin], len, 2 * len );
            }
            
            LInt U_nnz = 0;
            toc("Prepare");
            
            // rows_to_cols encodes the adjacency graph G_{0}(A) of A.
            // Each rows_to_cols[i] stores the neighbors j >= i.
            
            // Recursively build the eliminated graph G_{k+1}(A) from G_{k}(A).
            // The vertices of G_{k+1}(A) are always 0,...,n-1.
            // The set of edges of G_{k+1}(A) is the union of
            //      1. edges of G_{k}(A) and
            //      2. all {i,j}, where k < i < j and {k,i} and {k,j} are edges of G_{k}(A).
            
            tic("Main loop");
            for( Int k = 0; k < n; ++k )
            {
                // rows_to_cols encodes now G_{k}(A).
    
                List_T & cols_k = rows_to_cols[k];
                
//                cols_k.Sort();
//                cols_k.DeleteDuplicates();
                
                // cols_k contains all the neighbors j >= k in a sorted list.
                
                const Int col_count = cols_k.Size();
                
                // TODO: This loop could be parallelized. Not sure whether that would be helpful.
                #pragma omp parallel for schedule( static )
                for( Int a = 1; a < col_count; ++a )
                {
                    const Int i = cols_k[a];
                    // This cols_k is ordered, and the first entry is always k, this guarantees that i > k.
                    
                    List_T & cols_i = rows_to_cols[i];
                    
                    // TODO: We insert several increasing j. There must be a shortcut to this!
                    // TODO: cols_i.Insert( cols_k, a+1, col_count );
                    for( Int b = a+1; b < col_count; ++b )
                    {
                        const Int j = cols_k[b];
                        
                        // This cols_k is ordered, this guarantees that k < i < j
                        cols_i.Insert(j);
                    }
                }
                
                // cols_k should be finished by now. It is even sorted already.
                
                U_nnz += cols_k.Size();
                
                // rows_to_cols encodes now G_{k+1}(A).
            }
            toc("Main loop");
            
            // rows_to_cols encodes now G_{n}(A); we are done.
            
            tic("Initialize U");
            U = SparseMatrix_T( n, n, U_nnz, thread_count );
            
//            print("Create row pointers of U.");
            {
                LInt * restrict const outer = U.Outer().data();
                
                outer[0] = 0;
                
                for( Int i = 0; i < n; ++i )
                {
                    outer[i+1] = outer[i] + rows_to_cols[i].Size();
                }
            }
            
//            print("Copy column indices into U.");
            {
                const LInt * restrict const outer = U.Outer().data();
                       Int * restrict const inner = U.Inner().data();
                
                for( Int i = 0; i < n; ++i )
                {
                    LInt i_nnz = outer[i+1] - outer[i];
                    
                    copy_buffer( &rows_to_cols[i][0], &inner[outer[i]], i_nnz );
                }
            }
            toc("Initialize U");
            
            toc("FillGraph");
        }
        
        void FillGraph2()
        {
            tic("FillGraph2");
            
            tic("Prepare");
            
            const LInt * restrict const rp = A_up.Outer().data();
            const  Int * restrict const ci = A_up.Inner().data();
            
            std::vector< std::unordered_set<Int> > rows_to_cols (n);
            
            Tensor1<LInt,Int> U_rp (n+1);
            U_rp[0] = 0;
            
            std::vector<Int>  U_ci (A_up.NonzeroCount(), -1 );
            
            
            for( Int i = 0; i < n; ++i )
            {
                rows_to_cols[i] = std::unordered_set<Int>( &ci[rp[i]], &ci[rp[i+1]] );
            }
            
            toc("Prepare");
            
            // rows_to_cols encodes the adjacency graph G_{0}(A) of A.
            // Each rows_to_cols[i] stores the neighbors j >= i.
            
            // Recursively build the eliminated graph G_{k+1}(A) from G_{k}(A).
            // The vertices of G_{k+1}(A) are always 0,...,n-1.
            // The set of edges of G_{k+1}(A) is the union of
            //      1. edges of G_{k}(A) and
            //      2. all {i,j}, where k < i < j and {k,i} and {k,j} are edges of G_{k}(A).
            
            tic("Main loop");
            for( Int k = 0; k < n; ++k )
            {
                // rows_to_cols encodes now G_{k}(A).
//                dump(k);
                std::unordered_set<Int> & cols_k = rows_to_cols[k];

                const LInt _begin = U_rp[k];
                const LInt _end   = _begin + cols_k.size();
                U_rp[k+1] = _end;
                
                U_ci.resize(_end);
                
                std::copy( cols_k.begin(), cols_k.end(), U_ci.begin() + _begin );
                
                std::sort( U_ci.begin() + _begin, U_ci.begin() + _end );
                
                const Int * restrict const cols_k_ = &U_ci[_begin];
                // cols_k_ should be finished by now.
                // It is even sorted already?
                
//                // We may release the memory, but it is probably not a good idea.
//                cols_k = std::unordered_set<Int>();
                
                // cols_k_ contains all the neighbors j >= k in a sorted list.
                const Int col_count = _end - _begin;
                
                // TODO: This loop could be parallelized. Not sure whether that's worth it.
//                #pragma omp parallel for schedule(static)
                for( Int a = 2; a < col_count; ++a )
                {
                    // Since cols_k_ is ordered, and the first entry is always k, this guarantees that i > k.
                    const Int i = cols_k_[a-1];
//                    dump(i);
                    rows_to_cols[i].insert( &cols_k_[a], &cols_k_[col_count] );
                }
                
                // rows_to_cols encodes now G_{k+1}(A).
            }
            toc("Main loop");
            
            // rows_to_cols encodes now G_{n}(A); we are done.
            
            tic("Initialize U");
            // TODO: A constructor that moves U_rp and U_ci would be great here.
            U = SparseMatrix_T( &U_rp[0], &U_ci[0], n, n, thread_count );
            
            toc("Initialize U");
            
            toc("FillGraph2");
        }
        
        
        
        
    }; // class SparseCholeskyDecomposition
    
    
} // namespace Tensors
