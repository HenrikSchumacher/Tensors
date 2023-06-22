#pragma once
            
//###########################################################################################
//####          "Naive" symbolic factorization (for checking correctness)
//###########################################################################################
        public:
            
//            void SymbolicFactorization()
//            {
//                // Non-supernodal way to perform symbolic analysis.
//                // Only meant as reference! In practice rather use the supernodal version SN_SymbolicFactorization.
//
//                // This is Algorithm 4.2 from  Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
//
//                ptic(ClassName()+"::SymbolicFactorization");
//
//                Tensor1<Int,Int> row ( n );  // An array to aggregate the rows of U.
//
//                Tensor1<Int,Int> buffer ( n );  // Some scratch space for UniteSortedBuffers.
//                Int row_counter;                // Holds the current number of indices in row.
//
//                // To be filled with the row pointers of U.
//                Tensor1<LInt,Int> U_rp (n+1);
//                U_rp[0] = 0;
//
//                // To be filled with the column indices of U.
//                Aggregator<Int,LInt> U_ci ( 2 * A.NonzeroCount() );
//                for( Int i = 0; i < n; ++i ) // Traverse rows.
//                {
//                    // The nonzero pattern of upper(A) belongs definitely to the pattern of U.
//                    row_counter = A.Outer(i+1) - A.Diag(i);
//                    copy_buffer( &A.Inner(A.Diag(i)), row.data(), row_counter );
//
//                    const Int l_begin = eTree.ChildPointer(i  );
//                    const Int l_end   = eTree.ChildPointer(i+1);
//
//                    // Traverse all children of i in the eTree. Most of the time it's a single child or no one at all. Sometimes it's two or more.
//                    for( Int l = l_begin; l < l_end; ++l )
//                    {
//                        const Int j = eTree.ChildIndex(l);
//
//                        // Merge row pointers of child j into row
//                        const LInt _begin = U_rp[j]+1;  // This excludes U_ci[U_rp[j]] == j.
//                        const LInt _end   = U_rp[j+1];
//
//                        if( _end > _begin )
//                        {
//                            row_counter = UniteSortedBuffers(
//                                row.data(),    row_counter,
//                                &U_ci[_begin], int_cast<Int>(_end - _begin),
//                                buffer.data()
//                            );
//                            swap( row, buffer );
//                        }
//
//                    }
//
//                    // Copy row to i-th row of U.
//                    U_ci.Push( row.data(), row_counter );
//                    U_rp[i+1] = U_ci.Size();
//
//                } // for( Int i = 0; i < n; ++i )
//
//                Tensor1<Scal,LInt> U_val ( U_ci.Size() );
//
//                U = Matrix_T( std::move(U_rp), std::move(U_ci.Get()), std::move(U_val), n, n, thread_count );
//
//                ptoc(ClassName()+"::SymbolicFactorization");
//            }
//
//            template< Int RHS_COUNT, bool unitDiag = false>
//            void U_Solve_Sequential_0( ptr<Scal> b,  mut<Scal> x )
//            {
//                U.SolveUpperTriangular_Sequential_0<RHS_COUNT,unitDiag>(b,x);
//            }
