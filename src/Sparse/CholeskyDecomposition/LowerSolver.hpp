#pragma once

namespace Tensors
{
    namespace Sparse
    {
        // TODO: The `LowerSolver` does not seem to work correctly in parallel. Why is that? Running it single-thread does work.
        
        template<typename Scal_, typename Int_, typename LInt_> class CholeskyDecomposition;
        
        template<bool mult_rhs, typename Scal_, typename Int_, typename LInt_>
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
            
            ptr<Int>  SN_rp;
            ptr<LInt> SN_outer;
            ptr<Int>  SN_inner;
            
            ptr<LInt> SN_tri_ptr;
            mut<Scal> SN_tri_val;
            ptr<LInt> SN_rec_ptr;
            mut<Scal> SN_rec_val;
            
            // On in put: the right hand side; on output: the solution.
            mut<Scal> X;
            
            // Working space for BLAS3 routines.
            Tensor1<Scal,Int> X_1_buffer;
            
            // X_1 is the part of X that interacts with U_1, size = n_0 x n_1.
            mut<Scal> X_1;
//
            // x_1 is the part of x that interacts with U_1, size = n_1.
            mut<Scal> x_1;
            
        public:
            
            ~LowerSolver() = default;
            
            LowerSolver(
                CholeskyDecomposition<Scal,Int,LInt> & chol, const Int nrhs_
            )
            :   nrhs            ( nrhs_                  )
            ,   max_n_1         ( chol.max_n_1           )
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
                
            // Solver routine.
            void operator()( const Int s )
            {
                // X_0 is the part of X that interacts with U_0, size = n_0 x rhs_count.
                mut<Scal> X_0 = &X[nrhs * SN_rp[s]];
                
                const Int n_0 = SN_rp[s+1] - SN_rp[s];
            
                assert_positive(n_0);
                
                const LInt l_begin = SN_outer[s  ];
                const LInt l_end   = SN_outer[s+1];
                
                const Int n_1 = int_cast<Int>(l_end - l_begin);
                
                // U_0 is the triangular part of U that belongs to the supernode, size = n_0 x n_0
                ptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                
                // U_1 is the rectangular part of U that belongs to the supernode, size = n_0 x n_1
                ptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[s]];

                
                if constexpr ( mult_rhs )
                {
                    if( n_0 == ione )
                    {
                        // Triangle solve U_0 * X_0 = B while overwriting X_0.
                        // Since U_0 is a 1 x 1 matrix, it suffices to just scale X_0.
                        scale_buffer( Scalar::Inv<Scal>(U_0[0]), X_0, nrhs );
                        if( n_1 > izero )
                        {
                            // Compute X_1 = - U_1^H * X_0

                            //  U_1 is a matrix of size 1   x n_1.
                            //  X_1 is a matrix of size n_1 x nrhs.
                            //  X_0 is a matrix of size 1   x nrhs.

                            for( LInt i = 0; i < int_cast<LInt>(n_1); ++i )
                            {
                                const Scal factor = - Scalar::Conj(U_1[i]); // XXX Scalar::Conj(U_1[i])-> U_1[i]
                                for( LInt j = 0; j < int_cast<LInt>(nrhs); ++j )
                                {
                                    X_1[nrhs*i+j] = factor * X_0[j];
                                }
                            }
                        }
                    }
                    else // using BLAS3 routines.
                    {
                        // Triangle solve U_0^H * X_0 = B_0 while overwriting X_0.
                        BLAS::trsm<
                            Layout::RowMajor, Side::Left,
                            UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                        >(
                            n_0, nrhs,
                            one, U_0, n_0,
                                 X_0, nrhs
                        );
                        
                        if( n_1 > izero )
                        {
                            // Compute X_1 = - U_1^H * X_0
                            BLAS::gemm<Layout::RowMajor, Op::ConjTrans, Op::Id>(
                               //XXX Op::ConjTrans -> Op::Id?
                                n_1, nrhs, n_0, // ???
                                -one, U_1, n_1, // n_1 -> n_0
                                      X_0, nrhs,
                                zero, X_1, nrhs
                            );
                        }
                    }

                    // Add X_1 into B_1
                    for( Int j = 0; j < n_1; ++j )
                    {
                        add_to_buffer( &X_1[nrhs * j], &X[nrhs * SN_inner[l_begin+j]], nrhs );
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

                        if( n_1 > izero )
                        {
                            // Compute x_1 = - U_1^H * x_0
                            // x_1 is a vector of size n_1.
                            // U_1 is a matrix of size 1 x n_1
                            // x_0 is a vector of size 1.
                            
                            // Add x_1 into b_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                X[SN_inner[l_begin+j]] -= Scalar::Conj(U_1[j]) * x_0[0];
                            }   // XXX Scalar::Conj(U_1[j]) -> U_1[j]
                        }
                    }
                    else // using BLAS2 routines.
                    {
                        // Triangle solve U_0^H * x_0 = b_0 while overwriting x_0.
                        BLAS::trsv<
                            Layout::RowMajor, UpLo::Upper, Op::ConjTrans, Diag::NonUnit
                        >( n_0, U_0, n_0, x_0, 1 );
                        
                        if( n_1 > izero )
                        {
                            // Compute x_1 = - U_1^H * x_0
                            BLAS::gemv<Layout::RowMajor, Op::ConjTrans>(
                                n_0, n_1,             // XXX Op::ConjTrans -> Op::Trans
                                -one, U_1, n_1, // XXX n_1 -> n_0
                                                    x_0, 1,
                                zero, x_1, 1
                            );
                            
                            // Add x_1 into b_1.
                            for( Int j = 0; j < n_1; ++j )
                            {
                                X[SN_inner[l_begin+j]] += x_1[j];
                            }
                        }
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


