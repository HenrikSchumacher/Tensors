#pragma once

namespace Tensors
{
    namespace Sparse
    {
        
        template<typename Scalar_, typename Int_, typename LInt_> class CholeskyDecomposition;
        
        template<typename Scalar_, typename Int_, typename LInt_>
        class SupernodalCholeskyFactorizer
        {
            ASSERT_ARITHMETIC(Scalar_);
            ASSERT_INT(Int_);
            ASSERT_INT(LInt_)
        public:
            
            using Scalar    = Scalar_;
            using Real      = typename ScalarTraits<Scalar_>::Real;
            using Int       = Int_;
            using LInt      = LInt_;
            
            using SparseMatrix_T = SparseBinaryMatrixCSR<Int,LInt>;
            
        protected:
            
            static constexpr Scalar zero = 0;
            static constexpr Scalar one  = 1;

            const Int n;
            
            
            // shared data
            const   LInt * restrict const A_rp;  // row pointers of upper triangle of A
            const    Int * restrict const A_ci;  // column indices of upper triangle of A
            const Scalar * restrict const A_val; // nonzero values upper triangle of A
            
            const    Int * restrict const child_ptr;
            const    Int * restrict const child_idx;
            
            const    Int * restrict const SN_rp;
            const   LInt * restrict const SN_outer;
            const    Int * restrict const SN_inner;
            
            const   LInt * restrict const SN_tri_ptr;
                  Scalar * restrict const SN_tri_val;
            const   LInt * restrict const SN_rec_ptr;
                  Scalar * restrict const SN_rec_val;
            
            const    Int * restrict const desc_counts;
            
            
            // local data
            
            // Working space for intersection calculations.
            Tensor1<Int,Int> II_pos_buffer;
            Tensor1<Int,Int> IL_pos_buffer;
            Tensor1<Int,Int> JJ_pos_buffer;
            Tensor1<Int,Int> JL_pos_buffer;
            
            Int * restrict const II_pos = nullptr;
            Int * restrict const IL_pos = nullptr;
            Int * restrict const JJ_pos = nullptr;
            Int * restrict const JL_pos = nullptr;
            
            
            Int IL_ctr = 0;
            Int JL_ctr = 0;
            
            // Working space for BLAS3 routines.
            Tensor1<Scalar,Int> B_0_buffer; // part that updates A_0
            Tensor1<Scalar,Int> B_1_buffer; // part that updates A_1
            Tensor1<Scalar,Int> C_0_buffer; // scattered subblock of A_0
            Tensor1<Scalar,Int> C_1_buffer; // scattered subblock of A_1

            Scalar * restrict B_0 = nullptr;
            Scalar * restrict B_1 = nullptr;
            Scalar * restrict C_0 = nullptr;
            Scalar * restrict C_1 = nullptr;

            
            // Monitors.
            Int intersec = 0;
            Int empty_intersec = 0;
            
        public:
            
//            ~SupernodalCholeskyFactorizer() = default;
            
            ~SupernodalCholeskyFactorizer()
            {
                dump(intersec);
                dump(empty_intersec);
            }
            
            SupernodalCholeskyFactorizer(
                CholeskyDecomposition<Scalar,Int,LInt> & chol,
                const Scalar * restrict A_val_
            )
            // shared data
            :   n               ( chol.n                                        )
            ,   A_rp            ( chol.A_up.Outer().data()                      )
            ,   A_ci            ( chol.A_up.Inner().data()                      )
            ,   A_val           ( A_val_                                        )
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
                
                valprint(
                    "scratch memory of type "+TypeName<Int>::Get(),
                    2 * (chol.max_n_0 + chol.max_n_1)
                );

                
                valprint(
                    "scratch memory of type "+TypeName<Scalar>::Get(),
                    2 * (chol.max_n_0 * chol.max_n_0 + chol.max_n_0 * chol.max_n_1)
                );
                
            }
            
            
        public:
                
            void Factorize( const Int s )
            {
//                dump(s);
                
                const Int i_begin = SN_rp[s  ];
                const Int i_end   = SN_rp[s+1];
                
                const LInt l_begin = SN_outer[s  ];
                const LInt l_end   = SN_outer[s+1];
                
                const Int n_0 = i_end - i_begin;
                const Int n_1 = static_cast<Int>(l_end - l_begin);
                                
//                dump(n_0);
//                dump(n_1);
                
                // A_0 is interpreted as an upper triangular matrix of size n_0 x n_0.
                // A_1 is interpreted as a  rectangular      matrix of size n_0 x n_1.
                Scalar * restrict const A_0 = &SN_tri_val[SN_tri_ptr[s]];
                Scalar * restrict const A_1 = &SN_rec_val[SN_rec_ptr[s]];
                
                
                // Read the values of A into A_0 and A_1.
                // ======================================
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt k_begin = A_rp[i  ];
                    const LInt k_end   = A_rp[i+1];
                    
                    Int k;
                    for( k = k_begin; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        if( j < i_end )
                        {
                            // j belongs to the triangular part
                            A_0[ n_0 * (i - i_begin) + (j - i_begin) ] = A_val[k];
                        }
                        else
                        {
                            break;
                        }
                    }
                    
                    // From now on the insertion position must be in the rectangular part.
                    Int l     = l_begin;
                    Int col_l = SN_inner[l];
                    
                    // continue the loop where the previous one was aborted.
                    for( ; k < k_end; ++k )
                    {
                        const Int j = A_ci [k];
                        
                        // Find the position l of j in SN_inner, then write into A_1[l - l_begin].
                        // Remark: We don't need a check l < l_end here, because we know that we find an l.
                        while( col_l < j )
                        {
                            col_l = SN_inner[++l];
                        }
                        A_1[ n_1 * (i - i_begin) + (l - l_begin) ] = A_val[k]; // XXX conj-transpose here.
                    }
                }
                
                // Until here everything seems to be okay.
                
                
                // Fetch row updates of all descendants of s.
                // ==========================================

                // Using a postordering guarantees, that the descendants of s are already computed.
                // Moreover we can exploit that all children of s lie contiguously directly before s.
                const Int t_begin = (s + 1) - desc_counts[s] ;
                const Int t_end   = s;

                for( Int t = t_begin; t < t_end; ++t )
                {
//                    dump(t);

                    const Int m_0 = SN_rp   [t+1] - SN_rp   [t];
                    const Int m_1 = SN_outer[t+1] - SN_outer[t];
//
//                    dump(m_0);
//                    dump(m_1);


                    // Compute the intersection of supernode s with t.
                    Intersection( s, t );

                    // TODO: Comment this out!
                    CheckIntersection( s, t );


                    const Scalar * restrict const t_rec = &SN_rec_val[SN_rec_ptr[t]];
                    // t_rec is interpreted as a rectangular matrix of size m_0 x m_1.

                    // TODO: Maybe transpose t_rec to reduce amount of scattered-reads and adds...

                    // Update triangular block A_0.
                    if( IL_ctr > 0 )
                    {
                        // TODO: Add specializations for m_0 == 1!
                        // TODO: Add specializations for IL_ctr == 1! --> her

                        // Col-scatter-read t_rec[:,IL_pos] into B_0,
                        // where B_0 is a matrix of size m_0 x IL_ctr;
                        for( Int i = 0; i < m_0; ++i )
                        {
                            scatter_read( &t_rec[m_1 * i], &B_0[IL_ctr * i], IL_pos, IL_ctr );
                        }
                        
                        // Do C_0 = upper(B_0^H * B_0),
                        // where C_0 is an upper triangular matrix of size IL_ctr x IL_ctr.
                        BLAS_Wrappers::herk(
                            CblasRowMajor, CblasUpper, CblasConjTrans,
                            IL_ctr, m_0,
                            one,  B_0, IL_ctr,
                            zero, C_0, IL_ctr
                        );

                        // Row-col-scatter-subtract C_0 into A_0,
                        // where A_0 is an upper triangular matrix of size n_0 x n_0.
                        for( Int i = 0; i < IL_ctr; ++i )
                        {
                            for( Int j = i; j < IL_ctr; ++j )
                            {
                                A_0[ n_0 * II_pos[i] + II_pos[j] ] -= C_0[ IL_ctr * i + j];
                            }
                        }
                    }

                    // Update rectangular block A_1.
                    if( (IL_ctr > 0) && (JL_ctr > 0) )
                    {
                        // TODO: Add specialization for m_0 == 1.
                        // TODO: Add specialization for IL_ctr == 1.
                        // TODO: Add specialization for JL_ctr == 1! ->  gemv or even scalar update.

                        // Col-scatter-read t_rec[:,JL_pos] from B_1,
                        // where B_1 is a matrix of size m_0 x JL_ctr.
                        for( Int i = 0; i < m_0; ++i )
                        {
                            scatter_read( &t_rec[m_1 * i], &B_1[JL_ctr * i], JL_pos, JL_ctr ); // XXX
                        }
                        
                        // Do C_1 = B_0^H * B_1,
                        // where C_1 is a matrix of size IL_ctr x JL_ctr.
                        BLAS_Wrappers::gemm(
                            CblasRowMajor, CblasConjTrans, CblasNoTrans, // XXX
                            IL_ctr, JL_ctr, m_0,
                            one,  B_0, IL_ctr,
                                  B_1, JL_ctr,
                            zero, C_1, JL_ctr
                        );

                        // Row-col-scatter-subtract C_1 from A_1,
                        // where A_1 is a matrix of size n_0 x n_1.
                        for( Int i = 0; i < IL_ctr; ++i )
                        {
                            for( Int j = 0; j < JL_ctr; ++j )
                            {
                                A_1[ n_1 * II_pos[i] + JJ_pos[j] ] -= C_1[ JL_ctr * i + j]; // XXX
                            }
                        }
                    }
                }
                
                
                // TODO: Add specializations for n_0 == 1?
                
                // Do the intra-supernodal row operations.
                // =======================================
                for( Int k = 0; k < n_0; ++k )
                {
                    const Real diag ( std::sqrt( std::abs( A_0[ (n_0+1) * k ] ) ) );

                    A_0[(n_0+1)*k] = diag;

                    const Real diag_inv ( static_cast<Real>(1)/diag );

                    scale_buffer( diag_inv, &A_0[(n_0+1)*k+1], n_0-k-1 );

                    // TODO: Replace by column-wise scaling.
                    scale_buffer( diag_inv, &A_1[n_1*k], n_1 ); // XXX
                    
                    for( Int i = k+1; i < n_0; ++i )
                    {
                        const Scalar a = A_0[ n_0 * k + i ] ;
                        
//                        combine_buffers<ScalarFlag::Generic, ScalarFlag::Plus>(
//                            -a, &A_0[n_0 * i], one, &A_0[n_0 * k], n_0-i
//                        );
                        
                        for( Int j = i; j < n_0; ++j )
                        {
                            A_0[ n_0 * i + j ] -= a * A_0[ n_0 * k + j ];
                        }
                        
                        // TODO: Replace by column-wise operation.
                        for( Int j = 0; j < n_1; ++j )
                        {
                            A_1[ n_1 * i + j ] -= a * A_1[ n_1 * k + j ];  // XXX
                        }
                    }
                }
            }
            
            
        protected:
            
            void CheckIntersection( const Int s, const Int t ) const
            {
                // Beware: This test only checks whether all found pairs match.
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
                
                
                for( Int i = 0; i < IL_ctr; ++i )
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
                    
                    valprint( "I", ToString( &a[0], n_0 , 16 ) );
                    valprint( "L", ToString( &SN_inner[l_begin], m_1 , 16 ) );
                    
                    valprint( "II_pos", ToString(II_pos,IL_ctr,16) );
                    valprint( "IL_pos", ToString(IL_pos,IL_ctr,16) );
                }
                
                for( Int j = 0; j < JL_ctr; ++j )
                {
                    okay = okay && ( SN_inner[j_begin + JJ_pos[j]] == SN_inner[ l_begin + JL_pos[j] ] );
                }
                
                if( !okay )
                {
                    eprint("bug in JJ_pos + JL_pos");
                    
                    dump(n_1);
                    dump(m_1);
                    
                    valprint( "J", ToString( &SN_inner[j_begin], n_1 , 16 ) );
                    valprint( "L", ToString( &SN_inner[l_begin], m_1 , 16 ) );
                    
                    valprint( "JJ_pos", ToString(JJ_pos,JL_ctr,16) );
                    valprint( "JL_pos", ToString(JL_pos,JL_ctr,16) );
                }
            }
            
            void Intersection( const Int s, const Int t )
            {
//                tic(ClassName()+"::Intersection");
                // Computes the intersecting column indices of s-th and t-th supernode
                
                // We assume that t < s.
                if( t >= s )
                {
                    eprint(ClassName()+"::Intersection: t >= s, but t < s is required.");
                }

                
                // s-th supernode has triangular part I = [SN_rp[s],SN_rp[s]+1,...,SN_rp[s+1][
                // and rectangular part J = [SN_inner[SN_outer[s]],[SN_inner[SN_outer[s]+1],...,[
                // t-th supernode has triangular part K = [SN_rp[t],SN_rp[t]+1,...,SN_rp[t+1][
                // and rectangular part L = [SN_inner[SN_outer[t]],[SN_inner[SN_outer[t]+1],...,[
                
                // We have to compute
                // - the positions II_pos of I \cap L in I,
                // - the positions IL_pos of I \cap L in L,
                // - the positions JJ_pos of J \cap L in J,
                // - the positions JL_pos of J \cap L in L.

                // On return the numbers IL_ctr, JL_ctr contain the lengths of the respective lists.

                // Go through I and L in ascending order and collect intersection indices.
                const  Int i_begin = SN_rp[s  ];
                const  Int i_end   = SN_rp[s+1];
                
                const LInt j_begin = SN_outer[s  ];
                const LInt j_end   = SN_outer[s+1];
                
                const LInt l_begin = SN_outer[t  ];
                const LInt l_end   = SN_outer[t+1];
                
                 Int i = i_begin;
                LInt l = l_begin;
                
                Int L_l = SN_inner[l];
                
                IL_ctr = 0;
                
                while( (i < i_end) && (l < l_end) )
                {
                    if( i < L_l )
                    {
                        ++i;
                    }
                    else if( i > L_l )
                    {
                        L_l = SN_inner[++l];
                    }
                    else // i == L_l
                    {
                        II_pos[IL_ctr] = static_cast<Int>(i-i_begin);
                        IL_pos[IL_ctr] = static_cast<Int>(l-l_begin);
                        ++IL_ctr;
                        ++i;
                        L_l = SN_inner[++l];
                    }
                }
                
                // Go through J and L in ascending order and collect intersection indices.
                
                LInt j = j_begin;
//                LInt l = l_begin;         // We can continue with l where it were before...
                
                Int J_j = SN_inner[j];
//                Int L_l = SN_inner[l];    // ... and thus, we can keep the old L_l, too.
                JL_ctr = 0;
                
                while( (j < j_end) && (l < l_end) )
                {
                    if( J_j < L_l )
                    {
                        J_j = SN_inner[++j];
                        
                    }
                    else if( J_j > L_l )
                    {
                        L_l = SN_inner[++l];
                    }
                    else // J_j == L_l
                    {
                        JJ_pos[JL_ctr] = static_cast<Int>(j-j_begin);
                        JL_pos[JL_ctr] = static_cast<Int>(l-l_begin);
                        ++JL_ctr;
                        J_j = SN_inner[++j];
                        L_l = SN_inner[++l];
                    }
                }
                
                ++intersec;
                
                if( (IL_ctr == 0) && (JL_ctr == 0) )
                {
                    ++empty_intersec;
                }

//                print("Intersection between supernodes s = "+ToString(s)+" and t = "+ToString(t)+":");
//                valprint("node s",ToString( &SN_inner[j_begin], j_end - j_begin, 16 ) );
//                valprint("node t",ToString( &SN_inner[l_begin], l_end - l_begin, 16 ) );
//                valprint("II_pos",ToString( II_pos, IL_ctr, 16 ) );
//                valprint("IL_pos",ToString( IL_pos, IL_ctr, 16 ) );
//                valprint("JJ_pos",ToString( JJ_pos, JL_ctr, 16 ) );
//                valprint("JL_pos",ToString( JL_pos, JL_ctr, 16 ) );


                    
//                if( IL_ctr == 0 && JL_ctr == 0)
//                {
//                    wprint("Empty intersection detected for supernodes s = "+ToString(s)+" and t = "+ToString(t)+".");
//
//                    valprint("node s",ToString( &SN_inner[j_begin], j_end - j_begin, 16 ) );
//                    valprint("node t",ToString( &SN_inner[l_begin], l_end - l_begin, 16 ) );
//                }
                
//                toc(ClassName()+"::Intersection");
            }
            
            
            void scatter_read(
                const Scalar * restrict const x,
                      Scalar * restrict const y,
                const Int    * restrict const idx,
                const Int n
            )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y[i] = x[idx[i]];
                }
            }
//
//            template<typename Scalar,typename Int>
//            void scatter_write(
//                const Scalar * restrict const x,
//                      Scalar * restrict const y,
//                const Int    * restrict const idx,
//                const size_t n
//            )
//            {
//                for( size_t i = 0; i < n; ++i )
//                {
//                    y[idx[i]] = x[i];
//                }
//            }
            
        public:
            
            std::string ClassName() const
            {
                return "Sparse::SupernodalCholeskyFactorizer<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
            }
            
        }; // class SupernodalCholeskyFactorizer
        
        
    } // namespace Sparse
    
} // namespace Tensors
