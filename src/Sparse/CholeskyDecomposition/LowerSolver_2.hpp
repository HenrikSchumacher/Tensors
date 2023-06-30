#pragma once

namespace Tensors
{
    namespace Sparse
    {
        // TODO: The `LowerSolver` does not seem to work correctly in parallel.
        // TODO: Why is that? Running it single-thread does work!
        
        template<typename Scal_, typename Int_, typename LInt_> class CholeskyDecomposition;
        
        template<bool mult_rhs, bool lockedQ, typename Scal_, typename Int_, typename LInt_>
        class alignas(OBJECT_ALIGNMENT) LowerSolver
        {
            ASSERT_INT(Int_);
            ASSERT_INT(LInt_)
            
        public:
            
            using Scal = Scal_;
            using Real = typename Scalar::Real<Scal_>;
            using Int  = Int_;
            using LInt = LInt_;
            
        protected:
            
            static constexpr Int izero = 0;
            static constexpr Int ione  = 1;
            
            static constexpr Real zero = 0;
            static constexpr Real one  = 1;

            const Int nrhs    = 0;
            const Int max_n_1 = 0;
            
            ptr<Int>  SN_child_ptr;
            ptr<Int>  SN_child_idx;
            
            ptr<Int>  SN_rp;
            ptr<LInt> SN_outer;
            ptr<Int>  SN_inner;
            
            ptr<LInt> SN_tri_ptr;
            mut<Scal> SN_tri_val;
            ptr<LInt> SN_rec_ptr;
            mut<Scal> SN_rec_val;
            
            // On in put: the right hand side; on output: the solution.
            mut<Scal> X;
            
            // Working space for BLAS routines.
            Tensor1<Scal,Int> X_1_buffer;
            
            // X_1 is the part of X that interacts with U_1, size = n_0 x n_1.
            mut<Scal> X_1;
//
            // x_1 is the part of x that interacts with U_1, size = n_1.
            mut<Scal> x_1;
            
            Tensor1<Scal,Int> Zero_buffer;
            ptr<Scal> Zero;
            
            std::vector<std::mutex> & row_mutexes;
            
        public:
            
            ~LowerSolver() = default;
            
            LowerSolver(
                CholeskyDecomposition<Scal,Int,LInt> & chol, const Int nrhs_
            )
            :   nrhs            ( nrhs_                  )
            ,   max_n_1         ( chol.max_n_1           )
            ,   SN_child_ptr    ( chol.AssemblyTree().ChildPointers().data() )
            ,   SN_child_idx    ( chol.AssemblyTree().ChildIndices().data()  )
            ,   SN_rp           ( chol.SN_rp.data()      )
            ,   SN_outer        ( chol.SN_outer.data()   )
            ,   SN_inner        ( chol.SN_inner.data()   )
            ,   SN_tri_ptr      ( chol.SN_tri_ptr.data() )
            ,   SN_tri_val      ( chol.SN_tri_val.data() )
            ,   SN_rec_ptr      ( chol.SN_rec_ptr.data() )
            ,   SN_rec_val      ( chol.SN_rec_val.data() )
            ,   X               ( chol.X.data()          )
            ,   X_1_buffer      ( max_n_1 * nrhs         )
            ,   X_1             ( X_1_buffer.data()      )
            ,   x_1             ( X_1_buffer.data()      )
            ,   Zero_buffer     ( max_n_1 * nrhs, zero   )
            ,   Zero            ( Zero_buffer.data()     )
            ,   row_mutexes     ( chol.row_mutexes       )
            {}
            
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
                
            
            void FetchFromChild( const Int c )
            {
                const Int n_0 = SN_rp[c+1] - SN_rp[c];
            
                assert_positive(n_0);
                
                const LInt l_begin = SN_outer[c  ];
                const LInt l_end   = SN_outer[c+1];
                
                const Int n_1 = int_cast<Int>(l_end - l_begin);
                
                if( n_1 <= ione )
                {
                    return;
                }
                
                // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[c]];
                
                if constexpr ( mult_rhs )
                {
                    // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
                    ptr<Scal> X_0 = &X[nrhs * SN_rp[c]];

                    if( n_0 == ione )
                    {
                        // Use BLAS 2 routine.
                        
                        // Compute X_1 = - U_1^H * X_0

                        //  U_1 is a matrix of size 1   x n_1.
                        //  X_1 is a matrix of size n_1 x nrhs.
                        //  X_0 is a matrix of size 1   x nrhs.

                        // TODO: It is really a shame, that we have to zerofy X_1 and cannot overwrite it.
                        
//                            zerofy_buffer(X_1, n_1 * nrhs );
                        
                        // Most curiously, copying is faster than zeroing out - just because we can use BLAS.
//                            BLAS::copy( n_1 * nrhs, Zero, 1, X_1, 1 );
                        
//                            BLAS::scal( n_1 * nrhs, zero, X_1, 1 );
//
//                            BLAS::ger<Layout::RowMajor,Op::Conj,Op::Id>(
//                                n_1, nrhs, -one, U_1, 1, X_0, 1, X_1, nrhs
//                            );
                        
                        for( LInt i = 0; i < int_cast<LInt>(n_1); ++i )
                        {
                            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero>(
                                - Scalar::Conj(U_1[i]), X_0, Scalar::Zero<Scal>, &X_1[nrhs*i], nrhs
                            );
                        }
                    }
                    else // n_0 > ione; use BLAS 3 routine.
                    {
                        // Compute X_1 = - U_1^H * X_0
                        BLAS::gemm<Layout::RowMajor,Op::ConjTrans,Op::Id>(
                            //XXX Op::ConjTrans -> Op::Id?
                            n_1, nrhs, n_0, // ???
                            -one, U_1, n_1, // n_1 -> n_0
                                  X_0, nrhs,
                            zero, X_1, nrhs
                        );
                    }

                    
                    // Scatter-add X_1 into B_1.
                    for( Int j = 0; j < n_1; ++j )
                    {
                        Int row = SN_inner[l_begin+j];
                        
                        if constexpr( lockedQ )
                        {
                            const std::lock_guard<std::mutex> lock ( row_mutexes[row] );
                            
                            add_to_buffer( &X_1[nrhs * j], &X[nrhs * row], nrhs );
                        }
                        else
                        {
                            add_to_buffer( &X_1[nrhs * j], &X[nrhs * row], nrhs );
                        }
                    }
                    
                }
                else // mult_rhs == false
                {
                    // x_0 is the part of x that interacts with U_0, size = n_0.
                    mut<Scal> x_0 = &X[SN_rp[c]];
                    
                    if( n_0 == ione )
                    {
                        // Use BLAS 1 routine.
                        
                        // Compute x_1 = - U_1^H * x_0
                        // x_1 is a vector of size n_1.
                        // U_1 is a matrix of size 1 x n_1
                        // x_0 is a vector of size 1.
                        
                        
                        // TODO: Use combine_scatter_write.
                        
                        // Add x_1 into b_1.
                        for( Int j = 0; j < n_1; ++j )
                        {
                            Int row = SN_inner[l_begin+j];
                            
                            if constexpr( lockedQ )
                            {
                                const std::lock_guard<std::mutex> lock ( row_mutexes[row] );
                                
                                X[row] -= Scalar::Conj(U_1[j]) * x_0[0];
                            }
                            else
                            {
                                X[row] -= Scalar::Conj(U_1[j]) * x_0[0];
                            }
                        }
                    }
                    else
                    {
                        // n_0 > ione; use BLAS 2 routine.
                        
                        // Compute x_1 = - U_1^H * x_0
                        BLAS::gemv<Layout::RowMajor,Op::ConjTrans>(
                            n_0, n_1,
                            -one, U_1, n_1,
                                  x_0, 1,
                            zero, x_1, 1
                        );
                        
                        // TODO: Use combine_scatter_write.
                        // Scatter-add x_1 into b_1.
                        for( Int j = 0; j < n_1; ++j )
                        {
                            Int row = SN_inner[l_begin+j];
                            
                            if constexpr( lockedQ )
                            {
                                const std::lock_guard<std::mutex> lock ( row_mutexes[row] );
                                
                                X[row] += x_1[j];
                            }
                            else
                            {
                                X[row] += x_1[j];
                            }
                        }
                    }
                }
            }
            
            // Solver routine.
            void operator()( const Int s )
            {
                
                // Perform the children's reactangular block updates.
                // Because there might be write conflicts, we let their parent (this node) manage this.
                const Int k_begin = SN_child_ptr[s    ];
                const Int k_end   = SN_child_ptr[s + 1];
                
                for( Int k = k_begin; k < k_end; ++k )
                {
                    const Int c = SN_child_idx[k];
                    
                    FetchFromChild(c);
                }
                
                // Now we are ready to read the triangle block and to do the lower triangular solve.
                
                const Int n_0 = SN_rp[s+1] - SN_rp[s];
            
                assert_positive(n_0);
                
                // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];

                
                if constexpr ( mult_rhs )
                {
                    // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
                    mut<Scal> X_0 = &X[nrhs * SN_rp[s]];
                    
                    if( n_0 == ione )
                    {
                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        
                        scale_buffer( Scalar::Inv<Scal>(U_0[0]), X_0, nrhs );
                        
//                        BLAS::scal( nrhs, Scalar::Inv<Scal>(U_0[0]), X_0, 1 );
                    }
                    else // n_0 > ione; using BLAS3 routines.
                    {
                        // Triangle solve U_0^H * X_0 = B_0 while overwriting X_0.
                        BLAS::trsm<Layout::RowMajor,Side::Left,UpLo::Upper,Op::ConjTrans,Diag::NonUnit>(
                            n_0, nrhs,
                            one, U_0, n_0,
                                 X_0, nrhs
                        );
                    }
                }
                else // mult_rhs == false
                {
                    // x_0 is the part of x that interacts with U_0, size = n_0.
                    mut<Scal> x_0 = &X[SN_rp[s]];
                    
                    if( n_0 == ione )
                    {
                        // Triangle solve U_0 * x_0 = b_0 while overwriting x_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale x_0.
                        x_0[0] /= U_0[0];
                    }
                    else // using BLAS2 routines.
                    {
                        // Triangle solve U_0^H * x_0 = b_0 while overwriting x_0.
                        BLAS::trsv<Layout::RowMajor, UpLo::Upper,Op::ConjTrans,Diag::NonUnit>(
                            n_0, U_0, n_0, x_0, 1
                        );
                    }
                }
            }

    
        public:
            
            std::string ClassName() const
            {
                return std::string("Sparse::LowerSolver")+"<"+ToString(mult_rhs)+
                ","+TypeName<Scal>+","+TypeName<Int>+","+TypeName<LInt>+">";
            }
            
        }; // class UpperSolve
        
        
    } // namespace Sparse
    
} // namespace Tensors


