#pragma once

namespace Tensors
{
    namespace Sparse
    {
        
        template<typename Scalar_, typename Int_, typename LInt_> class CholeskyDecomposition;
        
        template<typename Scalar_, typename Int_, typename LInt_>
        class alignas(OBJECT_ALIGNMENT) CholeskyFactorizer
        {
            // Performs a left-looking factorization.
            
            ASSERT_INT(Int_);
            ASSERT_INT(LInt_)
            
        public:
            
            using Scalar    = Scalar_;
            using Real      = typename ScalarTraits<Scalar_>::Real;
            using Int       = Int_;
            using LInt      = LInt_;
            
            using SparseMatrix_T = Sparse::BinaryMatrixCSR<Int,LInt>;
            
        protected:
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;

            const Int n;
            
            // shared data
            ptr<LInt>       A_diag; // position to the diagonal element of A in each row.
            ptr<LInt>       A_rp;   // row pointers of upper triangle of A
            ptr<Int>        A_ci;   // column indices of upper triangle of A
            ptr<Scalar>     A_val;  // values of upper triangle of A
            
            Scalar          reg;     // Regularization parameter for the diagonal.
            
            ptr<Int>        child_ptr;
            ptr<Int>        child_idx;
            
            ptr<Int>        SN_rp;
            ptr<LInt>       SN_outer;
            ptr<Int>        SN_inner;
            
            ptr<LInt>       SN_tri_ptr;
            mut<Scalar>     SN_tri_val;
            ptr<LInt>       SN_rec_ptr;
            mut<Scalar>     SN_rec_val;
            
            ptr<Int>         desc_counts;
            
            
            // local data
            
            // Working space for intersection calculations.
            Tensor1<Int,Int> II_pos_buffer;
            Tensor1<Int,Int> IL_pos_buffer;
            Tensor1<Int,Int> JJ_pos_buffer;
            Tensor1<Int,Int> JL_pos_buffer;
            
            mut<Int> II_pos = nullptr;
            mut<Int> IL_pos = nullptr;
            mut<Int> JJ_pos = nullptr;
            mut<Int> JL_pos = nullptr;
            
            Int IL_len = 0;
            Int JL_len = 0;
            
            // Working space for BLAS3 routines.
            Tensor1<Scalar,Int> B_0_buffer; // part that updates U_0
            Tensor1<Scalar,Int> B_1_buffer; // part that updates U_1
            Tensor1<Scalar,Int> C_0_buffer; // scattered subblock of U_0
            Tensor1<Scalar,Int> C_1_buffer; // scattered subblock of U_1
            
            mut<Scalar> B_0 = nullptr;
            mut<Scalar> B_1 = nullptr;
            mut<Scalar> C_0 = nullptr;
            mut<Scalar> C_1 = nullptr;

            
            // Monitors.
            Int intersec = 0;
            Int empty_intersec_undetected = 0;
            Int empty_intersec_detected   = 0;
            
            Int IL_len_small = 0;
            Int JL_len_small = 0;
            Int IL_len_and_JL_len_small = 0;

            
            Time start_time;
//            Time stop_time;
            float intersec_time = 0;
            float scatter_time  = 0;
            float gemm_time     = 0;
            float herk_time     = 0;
            float chol_time     = 0;
            
        public:
            
//            ~CholeskyFactorizer() = default;
            
            ~CholeskyFactorizer()
            {
//                tic("~CholeskyFactorizer");
//                dump(intersec);
//                dump(empty_intersec_undetected);
//                dump(empty_intersec_detected);
//                valprint("non_empty_intersec",intersec-empty_intersec_undetected-empty_intersec_detected);
//                dump(IL_len_small);
//                dump(JL_len_small);
//                dump(IL_len_and_JL_len_small);
                
//                dump(intersec_time);
//                dump(scatter_time);
//                dump(herk_time);
//                dump(gemm_time);
//                dump(chol_time);
//                toc("~CholeskyFactorizer");
            }
            
            CholeskyFactorizer(
                CholeskyDecomposition<Scalar,Int,LInt> & chol
            )
            // shared data
            :   n               ( chol.n                                        )
            ,   A_diag          ( chol.A.Diag().data()                          )
            ,   A_rp            ( chol.A.Outer().data()                         )
            ,   A_ci            ( chol.A.Inner().data()                         )
            ,   A_val           ( chol.A_val.data()                             )
            ,   reg             ( chol.reg                                      )
            ,   child_ptr       ( chol.AssemblyTree().ChildPointers().data()    )
            ,   child_idx       ( chol.AssemblyTree().ChildIndices().data()     )
            ,   SN_rp           ( chol.SN_rp.data()                             )
            ,   SN_outer        ( chol.SN_outer.data()                          )
            ,   SN_inner        ( chol.SN_inner.data()                          )
            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data()                        )
            ,   SN_tri_val      ( chol.SN_tri_val.data()                        )
            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data()                        )
            ,   SN_rec_val      ( chol.SN_rec_val.data()                        )
            ,   desc_counts     ( chol.AssemblyTree().DescendantCounts().data() )
            
            // local data
            ,   II_pos_buffer   ( chol.max_n_0                                  )
            ,   IL_pos_buffer   ( chol.max_n_0                                  )
            ,   JJ_pos_buffer   ( chol.max_n_1                                  )
            ,   JL_pos_buffer   ( chol.max_n_1                                  )
            ,   II_pos          ( II_pos_buffer.data()                          )
            ,   IL_pos          ( IL_pos_buffer.data()                          )
            ,   JJ_pos          ( JJ_pos_buffer.data()                          )
            ,   JL_pos          ( JL_pos_buffer.data()                          )
            ,   B_0_buffer      ( chol.max_n_0 * chol.max_n_0                   )
            ,   B_1_buffer      ( chol.max_n_0 * chol.max_n_1                   )
            ,   C_0_buffer      ( chol.max_n_0 * chol.max_n_0                   )
            ,   C_1_buffer      ( chol.max_n_0 * chol.max_n_1                   )
            ,   B_0             ( B_0_buffer.data()                             )
            ,   B_1             ( B_1_buffer.data()                             )
            ,   C_0             ( C_0_buffer.data()                             )
            ,   C_1             ( C_1_buffer.data()                             )
            {
//                valprint(
//                    "scratch memory of type "+TypeName<Int>::Get(),
//                    2 * (chol.max_n_0 + chol.max_n_1)
//                );
//
//
//                valprint(
//                    "scratch memory of type "+TypeName<Scalar>::Get(),
//                    2 * (chol.max_n_0 * chol.max_n_0 + chol.max_n_0 * chol.max_n_1)
//                );
            }
            
        protected:
            
//            void _tic()
//            {
//                start_time = Clock::now();
//            }
//
//            float _toc()
//            {
//                return Duration( start_time, Clock::now() );
//            }
            
            void _tic()
            {
            }

            float _toc()
            {
                return 0;
            }
            
            
            
        public:
                
            // Factorization routine.
            void operator()( const Int s )
            {
//                valprint("factorizing supernode",s);
                
                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const LInt l_begin = SN_outer[s  ];
                const LInt l_end   = SN_outer[s+1];
                
                const  Int n_0     = i_end - i_begin;
                const  Int n_1     = int_cast<Int>(l_end - l_begin);
                
                assert_positive(n_0);
                
                if( n_0 <= 0 )
                {
                    eprint("n_0<=0");
                    dump(s);
                    dump(SN_rp[s  ]);
                    dump(SN_rp[s+1]);
                }
                
                // U_0 is interpreted as an upper triangular matrix of size n_0 x n_0.
                // U_1 is interpreted as a  rectangular      matrix of size n_0 x n_1.
                mut<Scalar> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                mut<Scalar> U_1 = &SN_rec_val[SN_rec_ptr[s]];
                
                FetchFromA( i_begin, i_end, l_begin, l_end, U_0, U_1);
                
                // We have to fetch the row updates of all descendants of s.
                // We assume that the descendants of s are already factorized.
                // Since aTree is postordered, we can exploit that all descendants of s lie
                // contiguously in memory directly before s.
                const Int t_begin = (s + 1) - desc_counts[s] ;
                const Int t_end   = s;
                
                // TODO: This can be parallelized by adding into local buffers V_0 and V_1 and reducing them into U_0 and U_1.
                FetchFromDescendants( s, t_begin, t_end, n_0, n_1, U_0, U_1 );
                
                FactorizeSupernode( n_0, n_1, U_0, U_1 );
            }
            
        protected:
            
            void FetchFromA(
                Int i_begin, Int i_end, LInt l_begin, LInt l_end, mut<Scalar> U_0, mut<Scalar> U_1
            )
            {
                // Read the values of A into U_0 and U_1.

                const  Int n_0 = i_end - i_begin;
                const  Int n_1 = int_cast<Int>(l_end - l_begin);
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt k_begin = A_diag[i  ];
                    const LInt k_end   = A_rp  [i+1];
                    
                    LInt k;
                    
                    {
                        k = k_begin;
                        
                        const Int j = A_ci [k];
                        
                        U_0[n_0 * (i-i_begin) + (j-i_begin)] = static_cast<Scalar>(A_val[k]+reg);
                        
                        ++k;
                    }
                    for( ; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        if( j < i_end )
                        {
                            // j belongs to the triangular part
                            U_0[n_0 * (i-i_begin) + (j-i_begin)] = static_cast<Scalar>(A_val[k]);
                        }
                        else
                        {
                            break;
                        }
                    }
                    
                    // From now on the insertion position must be in the rectangular part.
                    LInt l     = l_begin;
                    Int col_l = SN_inner[l];
                    
                    // Continue the loop where the previous one was aborted.
                    for( ; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        // Find the position l of j in SN_inner, then write into U_1[l - l_begin].
                        // Remark: We don't need a check l < l_end here, because we know that we find an l.
                        while( col_l < j )
                        {
                            col_l = SN_inner[++l];
                        }
                        U_1[n_1 * (i - i_begin) + (l - l_begin)] = static_cast<Scalar>(A_val[k]);
                        // XXX conj-transpose here.
                    }
                }
            }
            
            void FetchFromDescendants(
                const Int s,                             // the supernode into which to fetch
                const Int t_begin, const Int t_end,     // the range of descendants
                const Int n_0, const Int n_1, mut<Scalar> U_0, mut<Scalar> U_1
            )
            {
                // Incorporate the row updates from descendants [ t_begin,...,t_end [ into U_0 and U_1.
                
                for( Int t = t_begin; t < t_end; ++t )
                {
                    // Compute the intersection of supernode s with t.
                    ComputeIntersection( s, t );
                    // TODO: I experience quite many empty intersects. Is there a way to avoid them?
                    
//                    CheckIntersection( s, t );
                    
                    if( (IL_len <= 0) && (JL_len <= 0) )
                    {
                        continue;
                    }

                    const Int m_0 = SN_rp[t+1] - SN_rp[t];
                    const Int m_1 = int_cast<Int>(SN_outer[t+1] - SN_outer[t]);

                    ptr<Scalar> t_rec = &SN_rec_val[SN_rec_ptr[t]];
                    // t_rec is interpreted as a rectangular matrix of size m_0 x m_1.

                    // TODO: Maybe we should transpose U_0 and U_1 etc. to reduce amount of scattered-reads and adds...
                    // TODO: Then U_0 and U_1 would be ColMajor and we could use BLAS and LAPACK without the C-layers CBLAS or LAPACKE.
                    // TODO: This is attractive because Apple Accelerate does not ship LAPACKE!
                    
//                    constexpr Int threshold = 16;
//
//                    if( (0 < IL_len) && (IL_len <= threshold) )
//                    {
//                        ++IL_len_small;
//                    }
//
//                    if( (0< JL_len) && (JL_len <= threshold) )
//                    {
//                        ++JL_len_small;
//                    }
//
//                    if( (0 < IL_len) && (IL_len <= threshold) && (0< JL_len) && (JL_len <= threshold) )
//                    {
//                        ++IL_len_and_JL_len_small;
//                    }
                    
                    
                    if( m_0 > 1 )
                    {
                        // Update triangular block U_0.
                        if( IL_len > 0 )
                        {
                            _tic();
                            // Col-scatter-read t_rec[:,IL_pos] into B_0,
                            // where B_0 is a matrix of size m_0 x IL_len;
                            
                            for( Int i = 0; i < m_0; ++i )
                            {
                                scatter_read( &t_rec[m_1 * i], &B_0[IL_len * i], IL_pos, IL_len );
                            }
                            
    //                        for( Int i = 0; i < IL_len; ++i )
    //                        {
    //                            copy_buffer( &t_rec[m_0 * IL_pos[i]], &B_0[m_0 * i], m_0 );
    //                        }
                            
                            scatter_time += _toc();
                            
                            // Do C_0 = - upper(B_0^H * B_0),
                            // where C_0 is an upper triangular matrix of size IL_len x IL_len.
                            
                            _tic();
                            BLAS_Wrappers::herk<Layout::RowMajor,UpLo::Upper,Op::ConjTrans>(
                                IL_len, m_0,
                                Real(-1), B_0, IL_len,
                                Real( 0), C_0, IL_len
                            );
                            herk_time += _toc();
                            _tic();

                            // Row-col-scatter-add C_0 into U_0,
                            // where U_0 is an upper triangular matrix of size n_0 x n_0.
                            
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0[ n_0 * II_pos[i] + II_pos[j] ] += C_0[ IL_len * i + j ];
                                }
                            }
                            
                            scatter_time += _toc();
                        }

                        // Update rectangular block U_1.
                        if( (IL_len > 0) && (JL_len > 0) )
                        {
                            // TODO: Add specialization for IL_len == 1.
                            // TODO: Add specialization for JL_len == 1! ->  gemv or even scalar update.

                            _tic();
                            // Col-scatter-read t_rec[:,JL_pos] from B_1,
                            // where B_1 is a matrix of size m_0 x JL_len.
                            
                            for( Int i = 0; i < m_0; ++i )
                            {
                                scatter_read( &t_rec[m_1 * i], &B_1[JL_len * i], JL_pos, JL_len ); // XXX
                            }
                            
                            // This is how the "transposed" version would look like.
    //                        for( Int j = 0; j < JL_len; ++j )
    //                        {
    //                            copy_buffer( &t_rec[m_0 * JL_pos[j]], &B_1[m_0 * j], m_0 );
    //                        }
                            
                            scatter_time += _toc();
                            
                            // Do C_1 = - B_0^H * B_1,
                            // where C_1 is a matrix of size IL_len x JL_len.

                            _tic();
                            if( JL_len > 1)
                            {
                                BLAS_Wrappers::gemm<Layout::RowMajor,Op::ConjTrans,Op::Id>(// XXX
                                    IL_len, JL_len, m_0,
                                    Scalar(-1), B_0, IL_len,
                                                B_1, JL_len,
                                    Scalar( 0), C_1, JL_len
                                );
                            }
                            else // JL_len > 1 -- But this specialization does not seem to make a big difference.
                            {
                                BLAS_Wrappers::gemv<Layout::RowMajor,Op::ConjTrans>(// XXX
                                    m_0, IL_len,
                                    Scalar(-1), B_0, IL_len,
                                                B_1, 1,
                                    Scalar( 0), C_1, 1
                                );
                            }
                            gemm_time += _toc();

                            _tic();
                            // Row-col-scatter-add C_1 into U_1,
                            // where U_1 is a matrix of size n_0 x n_1.
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                for( Int j = 0; j < JL_len; ++j )
                                {
                                    U_1[n_1 * II_pos[i] + JJ_pos[j]] += C_1[JL_len * i + j]; // XXX
                                }
                            }
                            scatter_time += _toc();
                        }
                    }
                    else // m_0 == 1
                    {
                        // In this case we have to form only (scattered) outer products of vectors, which are basically BLAS2 routines. Unfortunately, ?ger, ?syr, ?her, etc. add into the target matrix without the option to zero it out first. Hence we simply write these double loops ourselves. Since this is BLAS2 and not BLAS3, and since IL_len and JL_len are often not particularly long, there isn't that much room for optimization anyways. And since we cannot use ?ger / ?her, we fuse the scattered read/write operations directly into these loops.
                        
                        for( Int j = 0; j < IL_len; ++j )
                        {
                            B_0[j] = t_rec[IL_pos[j]];
                        }

                        for( Int j = 0; j < JL_len; ++j )
                        {
                            B_1[j] = t_rec[JL_pos[j]];
                        }
                        
                        if( JL_len > 0 )
                        {
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                const Scalar factor = - conj(t_rec[IL_pos[i]]);

                                const Int i_ = II_pos[i];
                                mut<Scalar> U_0_ = &U_0[n_0 * i_];
                                mut<Scalar> U_1_ = &U_1[n_1 * i_];
                                
                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0_[II_pos[j]] += factor * B_0[j];
                                }

                                for( Int j = 0; j < JL_len; ++j )
                                {
                                    U_1_[JJ_pos[j]] += factor * B_1[j];
                                }
                            }
                        }
                        else
                        {
                            for( Int i = 0; i < IL_len; ++i )
                            {
                                const Scalar factor = - conj(t_rec[IL_pos[i]]);

                                const Int i_ = II_pos[i];
                                mut<Scalar> U_0_ = &U_0[n_0 * i_];
                                
                                for( Int j = i; j < IL_len; ++j )
                                {
                                    U_0_[II_pos[j]] += factor * B_0[j];
                                }
                            }
                        }
                        
                    } // if( m_0 > 1 )
                        
                } // for( Int t = t_begin; t < t_end; ++t )
            }
            
            void FactorizeSupernode( Int n_0, Int n_1, mut<Scalar> U_0, mut<Scalar> U_1 )
            {
                _tic();
                
                if( n_0 > 1 )
                {
                    // Cholesky factorization of U_0
                    (void)LAPACK_Wrappers::potrf<Layout::RowMajor,UpLo::Upper>( n_0, U_0, n_0);

                    // Triangular solve U_1 = U_0^{-H} U_1.
                    if( n_1 > 1 )
                    {
                        BLAS_Wrappers::trsm<Layout::RowMajor,
                            Side::Left, UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                        >(
                            n_0, n_1, Scalar(1), U_0, n_0, U_1, n_1
                        );
                    }
                    else if( n_1 == 1 )
                    {
                        BLAS_Wrappers::trsv<Layout::RowMajor, UpLo::Upper, Op::ConjTrans, Diag::NonUnit>(
                            n_0, U_0, n_0, U_1, 1
                        );
                    }
                    else // n_1 == 0
                    {
                        // Do nothing.
                    }
                }
                else
                {
                    U_0[0] = std::sqrt(std::abs(U_0[0]));
                    scale_buffer(static_cast<Scalar>(1)/U_0[0], U_1, n_1);
                }
                
                chol_time += _toc();
            }
            
        protected:
            
            force_inline void scatter_read( ptr<Scalar> x, mut<Scalar> y, ptr<Int> idx, Int N )
            {
                for( ; N --> 0; ) { y[N] = x[idx[N]]; }
            }
            
            force_inline void scatter_add( ptr<Scalar> x, mut<Scalar> y, ptr<Int> idx, Int N )
            {
                for( ; N --> 0; ) { y[idx[N]] += x[N]; }
            }
            
            void ComputeIntersection( const Int s, const Int t )
            {
//                _tic();
                
                // Compute the intersecting column indices of s-th and t-th supernode.
                // We assume that t < s.
                
                // s-th supernode has triangular part I = [SN_rp[s],SN_rp[s]+1,...,SN_rp[s+1][
                // and rectangular part J = [SN_inner[SN_outer[s]],[SN_inner[SN_outer[s]+1],...,[
                // t-th supernode has triangular part K = [SN_rp[t],SN_rp[t]+1,...,SN_rp[t+1][
                // and rectangular part L = [SN_inner[SN_outer[t]],[SN_inner[SN_outer[t]+1],...,[
                
                // We have to compute
                // - the positions II_pos of I \cap L in I,
                // - the positions IL_pos of I \cap L in L,
                // - the positions JJ_pos of J \cap L in J,
                // - the positions JL_pos of J \cap L in L.

                // On return the numbers IL_len, JL_len contain the lengths of the respective lists.

                IL_len = 0;
                JL_len = 0;
                
                ++intersec;
                
                const  Int i_begin = SN_rp[s  ];
                const LInt l_end   = SN_outer[t+1];
                
                // quick check whether the convex hulls of index ranges overlap.
                if(
                    (
                       (SN_inner[l_end-1] < i_begin) // l_end > l_begin, bc. s would not be a child of t otherwise.
                    )
//                    ||
//                    (
//                        (j_begin < j_end)
//                        &&
//                        (SN_inner[j_end-1] < SN_inner[l_begin]) // Impossible due to postordering?
//                    )
                )
                {
//                    // Debugging
//                    ++empty_intersec_detected;
                    return;
                }
                
                // Go through I and L in ascending order and collect intersection indices.

                const  Int i_end   = SN_rp[s+1];
                
                const LInt j_begin = SN_outer[s  ];
                const LInt j_end   = SN_outer[s+1];
                
                const LInt l_begin = SN_outer[t  ];
                
                 Int i = i_begin;
                LInt l = l_begin;
                
                Int L_l = SN_inner[l];
                
                while( (i < i_end) && (l < l_end) )
                {
                    if( i < L_l )
                    {
                        ++i;
                    }
                    else if( L_l < i )
                    {
                        L_l = SN_inner[++l];
                    }
                    else // i == L_l
                    {
                        II_pos[IL_len] = int_cast<Int>(i-i_begin);
                        IL_pos[IL_len] = int_cast<Int>(l-l_begin);
                        ++IL_len;
                        ++i;
                        L_l = SN_inner[++l];
                    }
                }
                
                // Go through J and L in ascending order and collect intersection indices.
                
                LInt j = j_begin;
//                LInt l = l_begin;         // We can continue with l where it were before...
                
                Int J_j = SN_inner[j];
//                Int L_l = SN_inner[l];    // ... and thus, we can keep the old L_l, too.
                
                while( (j < j_end) && (l < l_end) )
                {
                    if( J_j < L_l )
                    {
                        J_j = SN_inner[++j];
                    }
                    else if( L_l < J_j )
                    {
                        L_l = SN_inner[++l];
                    }
                    else // J_j == L_l
                    {
                        JJ_pos[JL_len] = int_cast<Int>(j-j_begin);
                        JL_pos[JL_len] = int_cast<Int>(l-l_begin);
                        ++JL_len;
                        J_j = SN_inner[++j];
                        L_l = SN_inner[++l];
                    }
                }
//                intersec_time += _toc();
//
//                // Debugging
//                if( (IL_len == 0) && (JL_len == 0) )
//                {
//                    ++empty_intersec_undetected;
//                }
            }
            
            
            void CheckIntersection( const Int s, const Int t ) const
            {
                // Beware: This test only checks whether all _found_ pairs match!
                // Beware: It does not check, whether _all_ pairs are found!
                bool okay = true;

                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const  Int j_begin = SN_outer[s  ];
                const  Int j_end   = SN_outer[s+1];
                
                const LInt l_begin = SN_outer[t  ];
                const LInt l_end   = SN_outer[t+1];

                const Int n_0 = i_end - i_begin;
                const Int n_1 = j_end - j_begin;
                
                const Int m_1 = l_end - l_begin;
                
                for( Int i = 0; i < IL_len; ++i )
                {
                    okay = okay && ( i_begin + II_pos[i] == SN_inner[ l_begin + IL_pos[i] ] );
                }
                
                if( !okay )
                {
                    eprint("bug in II_pos + IL_pos");
                    
                    dump(n_0);
                    dump(m_1);
                    
                    Tensor1<Int,Int> a (n_0);
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        a[i-i_begin] = i;
                    }
                    
//                    valprint( "I", ToString( &a[0], n_0 , 16 ) );
//                    valprint( "L", ToString( &SN_inner[l_begin], m_1 , 16 ) );
//                    
//                    valprint( "II_pos", ToString(II_pos,IL_len,16) );
//                    valprint( "IL_pos", ToString(IL_pos,IL_len,16) );
                }
                
                for( Int j = 0; j < JL_len; ++j )
                {
                    okay = okay && ( SN_inner[j_begin + JJ_pos[j]] == SN_inner[l_begin + JL_pos[j]] );
                }
                
                if( !okay )
                {
                    eprint("bug in JJ_pos + JL_pos");
                    
                    dump(n_1);
                    dump(m_1);
                    
//                    valprint( "J", ToString( &SN_inner[j_begin], n_1 , 16 ) );
//                    valprint( "L", ToString( &SN_inner[l_begin], m_1 , 16 ) );
//                    
//                    valprint( "JJ_pos", ToString(JJ_pos,JL_len,16) );
//                    valprint( "JL_pos", ToString(JL_pos,JL_len,16) );
                }
            }
            
        public:
            
            std::string ClassName() const
            {
                return "Sparse::CholeskyFactorizer<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
            }
            
        }; // class CholeskyFactorizer
        
        
    } // namespace Sparse
    
} // namespace Tensors
