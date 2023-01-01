            
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
            
