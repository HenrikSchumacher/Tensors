// Intra-supernode cholesky factorization
// ======================================


//                for( Int k = 0; k < n_0; ++k )
//                {
//                    const Real diag ( std::sqrt( std::abs( A_0[ (n_0+1) * k ] ) ) );
//
//                    A_0[(n_0+1)*k] = diag;
//
//                    const Real diag_inv ( static_cast<Real>(1)/diag );
//
//                    scale_buffer( diag_inv, &A_0[(n_0+1)*k+1], n_0-k-1 );
//
//                    // TODO: Replace by column-wise scaling.
//                    scale_buffer( diag_inv, &A_1[n_1*k], n_1 ); // XXX
//
//                    for( Int i = k+1; i < n_0; ++i )
//                    {
//                        const Scalar a = -A_0[ n_0 * k + i ] ;
//
//                        for( Int j = i; j < n_0; ++j )
//                        {
//                            A_0[n_0 * i + j] += a * A_0[n_0 * k + j];
//                        }
//
//                        for( Int j = 0; j < n_1; ++j )
//                        {
//                            A_1[n_1 * i + j] += a * A_1[n_1 * k + j];  // XXX
//                        }
//
//
//                    }
//                }
//
//                for( Int k = 0; k < n_0; ++k )
//                {
//                    const Real diag ( std::sqrt( std::abs( A_0[ (n_0+1) * k ] ) ) );
//
//                    A_0[(n_0+1)*k] = diag;
//
//                    const Real diag_inv ( static_cast<Real>(1)/diag );
//
//                    scale_buffer( diag_inv, &A_0[(n_0+1)*k+1], n_0-k-1 );
//
//                    // TODO: Replace by column-wise scaling.
//                    scale_buffer( diag_inv, &A_1[n_1*k], n_1 ); // XXX
//
//                    for( Int i = k+1; i < n_0; ++i )
//                    {
//                        const Scalar a = -A_0[ n_0 * k + i ] ;
//
//                        for( Int j = i; j < n_0; ++j )
//                        {
//                            A_0[n_0 * i + j] += a * A_0[n_0 * k + j];
//                        }
//
//                        for( Int j = 0; j < n_1; ++j )
//                        {
//                            A_1[n_1 * i + j] += a * A_1[n_1 * k + j];  // XXX
//                        }
//                    }
//                }


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
////                            Op::Identity, Op::Id,
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
//                            UpLo::Upper,
//                            Op::Id,
//                            Diagonal::NonUnit,
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
            






//// Update triangular block A_0.
//if( IL_ctr > 0 )
//{
//    // TODO: Add specializations for m_0 == 1!
//    // TODO: Add specializations for IL_ctr == 1! --> her
//
//    // Col-scatter-read t_rec[:,IL_pos] into B_0,
//    // where B_0 is a matrix of size m_0 x IL_ctr;
//    for( Int i = 0; i < m_0; ++i )
//    {
//        scatter_read( &t_rec[m_1 * i], &B_0[IL_ctr * i], IL_pos, IL_ctr );
//    }
//
////                        if( IL_ctr > 1 )
//    if constexpr ( true )
//    {
//        // Do C_0 = upper(B_0^H * B_0),
//        // where C_0 is an upper triangular matrix of size IL_ctr x IL_ctr.
//
//        BLAS_Wrappers::herk(
//            CblasRowMajor, CblasUpper, CblasConjTrans,
//            IL_ctr, m_0,
//            one,  B_0, IL_ctr,
//            zero, C_0, IL_ctr
//        );
//
//        // Row-col-scatter-subtract C_0 into A_0,
//        // where A_0 is an upper triangular matrix of size n_0 x n_0.
//        for( Int i = 0; i < IL_ctr; ++i )
//        {
//            for( Int j = i; j < IL_ctr; ++j )
//            {
//                A_0[ n_0 * II_pos[i] + II_pos[j] ] -= C_0[ IL_ctr * i + j];
//            }
//        }
//    }
//    else
//    {
//        Scalar sum = 0;
//        for( Int i = 0; i < m_0; ++i )
//        {
//            sum += abs_squared(B_0[i]);
//        }
//
//        //Write directly to A_0.
//
//        A_0[ (n_0+1) * II_pos[0]] -= sum;
//    }
//}
//
//// Update rectangular block A_1.
//if( (IL_ctr > 0) && (JL_ctr > 0) )
//{
//    // TODO: Add specialization for m_0 == 1.
//    // TODO: Add specialization for IL_ctr == 1.
//    // TODO: Add specialization for JL_ctr == 1! ->  gemv or even scalar update.
//
//    // Col-scatter-read t_rec[:,JL_pos] from B_1,
//    // where B_1 is a matrix of size m_0 x JL_ctr.
//    for( Int i = 0; i < m_0; ++i )
//    {
//        scatter_read( &t_rec[m_1 * i], &B_1[JL_ctr * i], JL_pos, JL_ctr ); // XXX
//    }
//
//
////                        if( IL_ctr > 1 )
//    if constexpr ( true )
//    {
////                            if( JL_ctr > 1 )
//        if constexpr ( true )
//        {
//            // Do C_1 = B_0^H * B_1,
//            // where C_1 is a matrix of size IL_ctr x JL_ctr.
//            BLAS_Wrappers::gemm(
//                CblasRowMajor, CblasConjTrans, CblasNoTrans, // XXX
//                IL_ctr, JL_ctr, m_0,
//                one,  B_0, IL_ctr,
//                      B_1, JL_ctr,
//                zero, C_1, JL_ctr
//            );
//
//            // Row-col-scatter-subtract C_1 from A_1,
//            // where A_1 is a matrix of size n_0 x n_1.
//            for( Int i = 0; i < IL_ctr; ++i )
//            {
//                for( Int j = 0; j < JL_ctr; ++j )
//                {
//                    A_1[n_1 * II_pos[i] + JJ_pos[j] ] -= C_1[JL_ctr * i + j]; // XXX
//                }
//            }
//        }
//        else
//        {
//            // JL_ctr == 1
//
//            // Do C_1 = B_0^H * B_1,
//            // where C_1 is a matrix of size IL_ctr x 1.
//
//            BLAS_Wrappers::gemv(
//                CblasRowMajor, CblasConjTrans, // XXX
//                m_0, IL_ctr,
//                one,  B_0, IL_ctr,
//                      B_1, 1,
//                zero, C_1, 1
//            );
//
//            // Row-col-scatter-subtract C_1 from A_1,
//            // where A_1 is a matrix of size n_0 x n_1.
//            for( Int i = 0; i < IL_ctr; ++i )
//            {
//                A_1[n_1 * II_pos[i] + JJ_pos[0] ] -= C_1[JL_ctr * i]; // XXX
//            }
//        }
//    }
//    else
//    {
//        // IL_ctr == 1
//
//        if( JL_ctr > 1 )
//        {
//            // Do C_1 = B_0^H * B_1,
//            // where C_1 is a matrix of size 1 x JL_ctr.
//
//            // This is equivalent to
//            // C_1^H = B_1^H * B_0^H
//            // <=>
//            // conj(C_1) = B_1^H * B_0,
//            // where C_1 is a vector of size JL_ctr.
//
//            BLAS_Wrappers::gemv(
//                CblasRowMajor, CblasConjTrans, // XXX
//                m_0, JL_ctr,
//                one,  B_1, JL_ctr,
//                      B_0, 1,
//                zero, C_1, 1
//            );
//
//            // Row-col-scatter-subtract conj(C_1) from A_1,
//            // where A_1 is a matrix of size n_0 x n_1.
//
//            Scalar * restrict const A_1_ = &A_1[n_1 * II_pos[0]];
//
//            for( Int j = 0; j < JL_ctr; ++j )
//            {
//                A_1_[JJ_pos[j]] -= conj(C_1[j]); // XXX
//            }
//        }
//        else
//        {
//            // IL_ctr == 1 and JL_ctr == 1
//
//            // Do C_1 = B_0^H * B_1,
//            // where C_1 is a matrix of size 1 x 1.
//
//            Scalar sum = 0;
//            for( Int i = 0; i < m_0; ++i )
//            {
//                sum += conj(B_0[i]) * B_1[i];
//            }
//
//            A_1[n_1 * II_pos[0] + JJ_pos[0] ] -= sum; // XXX
//        }
//    }
//}
//}
