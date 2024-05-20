#pragma once

namespace Tensors
{
    namespace LAPACK
    {
        
        template<Layout layout, UpLo uplo, typename Scal>
        class SelfAdjointEigensolver
        {
            
        public:
            
            SelfAdjointEigensolver() = default;
            
            ~SelfAdjointEigensolver() = default;
            
        private:
            
            static constexpr char flag = to_LAPACK(
                layout == Layout::ColMajor
                ? uplo
                : ( uplo == UpLo::Upper ) ? UpLo::Lower : UpLo::Upper
            );
            
            Int n;
            
            Int ldA;
            
            Int lwork;
            
            Tensor1<Scalar::Real<Scal>,Int> eigs;
            
            Tensor2<Scal,Int> A;
            
            Tensor1<Scal,Int> work;
            
            Tensor1<Scalar::Real<Scal>,Int> rwork;
            
        public:
            
            template< typename I0, typename I1>
            Int Eigenvalues( const I0 n_, mptr<Scal> A_, const I1 ldA_, mptr<Scalar::Real<Scal>> eigs_ )
            {
                static_assert(IntQ<I0>,"");
                static_assert(IntQ<I1>,"");
                
                n    = int_cast<Int>(n_);
                ldA  = int_cast<Int>(ldA_);
                
                Int info = Prepare( 'N', A_ );
                
                info = heev( 'N');
                
                eigs.Write(eigs_);
                
                return info;
            }
            
            template< typename S, typename I0, typename I1, typename I2>
            Int Eigensystem( const I0 n_, mptr<S> A_, const I1 ldA_, mptr<Scalar::Real<S>> eigs_, mptr<S> Q_, const I2 ldQ )
            {
                // Returns Q and eigs such that ConjugateTranspose(Q) * A * Q == Diagona(eigs);
                
                static_assert(IntQ<I0>,"");
                static_assert(IntQ<I1>,"");
                
                n    = int_cast<Int>(n_);
                ldA  = int_cast<Int>(ldA_);
                
                Int info = Prepare( 'V', A_ );
                
                info = heev( 'V' );
                
                eigs.Write(eigs_);
                
                if constexpr ( layout == Layout::RowMajor )
                {
                    A.template Write<Op::ConjTrans>( Q_, int_cast<Int>(ldQ) );
                }
                else
                {
                    A.template Write<Op::Id>( Q_, int_cast<Int>(ldQ) );
                }
                
                return info;
            }
            
            
        private:
            
            
            template< typename S>
            force_inline int Prepare( char job, mptr<S> A_ )
            {
                assert_positive(n);
                assert_positive(ldA);
                
                work.RequireSize( 1 );
                work[0] = 0;
                
                if constexpr ( Scalar::ComplexQ<Scal> )
                {
                    rwork.RequireSize( 3 * n - 2 );
                }
                
                int info;
                
                lwork = -1;
                
                // Request the required work space from LAPACK
                eigs.RequireSize( n );
                
                A.RequireSize( n, n );
                
                A.Read( A_, ldA );
                
                info = heev( job );
                
                lwork = static_cast<int>(Re(work[0]));
                                
                work.RequireSize( lwork );
                
                return info;
            }
            
            force_inline int heev( char job  )
            {
                Int info = 0;
                
                auto * A_     = to_LAPACK(A.data());
                auto * eigs_  = to_LAPACK(eigs.data());
                auto * work_  = to_LAPACK(work.data());
                auto * rwork_ = to_LAPACK(rwork.data());
                                
                if constexpr ( SameQ<Scal,double> )
                {
                    #ifdef LAPACK_dsyev
                        LAPACK_dsyev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #else
                        dsyev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,float> )
                {
                    #ifdef LAPACK_ssyev
                        LAPACK_ssyev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #else
                        ssyev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,std::complex<double>> )
                {
                    #ifdef LAPACK_zheev
                        LAPACK_zheev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #else
                        zheev_      ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #endif
                }
                else if constexpr ( SameQ<Scal,std::complex<float>> )
                {
                    #ifdef LAPACK_cheev
                        LAPACK_cheev( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #else
                        cheev       ( &job, &flag, &n, A_, &ldA, eigs_, work_, &lwork, rwork_, &info );
                    #endif
                }
                else
                {
                    eprint("heev not defined for scalar type " + TypeName<Scal> );
                }

                return info;
            }
        };
        
        
        
    } // namespace LAPACK
    
} // namespace Tensors

