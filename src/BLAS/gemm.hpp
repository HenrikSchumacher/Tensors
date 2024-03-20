#pragma once

#include "Tiny.hpp"

namespace Tensors
{
    namespace BLAS
    {
        template<
            Layout layout, Op opA, Op opB,
            typename Scal, typename I0, typename I1, typename I2, typename I3, typename I4, typename I5
        >
        force_inline void gemm(
            const I0 m_, const I1 n_, const I2 k_,
            cref<Scal> alpha, cptr<Scal> A_, const I3 ldA_,
                              cptr<Scal> B_, const I4 ldB_,
            cref<Scal> beta,  mptr<Scal> C_, const I5 ldC_
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            ASSERT_INT(I4);
            ASSERT_INT(I5);
            
            Int m    = int_cast<Int>(m_);
            Int n    = int_cast<Int>(n_);
            Int k    = int_cast<Int>(k_);
            Int ldA  = int_cast<Int>(ldA_);
            Int ldB  = int_cast<Int>(ldB_);
            Int ldC  = int_cast<Int>(ldC_);
            
            auto * A = to_BLAS(A_);
            auto * B = to_BLAS(B_);
            auto * C = to_BLAS(C_);
            
//                logprint( std::string("BLASS::gemm ( ") + ToString(m) + "," + ToString(n) + "," + ToString(k) + " )");
                
            assert_positive(m);
            assert_positive(n);
            assert_positive(k);
            assert_positive(ldA);
            assert_positive(ldB);
            assert_positive(ldC);

            if constexpr ( SameQ<Scal,double> )
            {
                return cblas_dgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,float> )
            {
                return cblas_sgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,std::complex<double>> )
            {
                return cblas_zgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, to_BLAS(&alpha), A, ldA, B, ldB, to_BLAS(&beta), C, ldC
                );
            }
            else if constexpr ( SameQ<Scal,std::complex<float>> )
            {
                return cblas_cgemm(
                    to_BLAS(layout), to_BLAS(opA), to_BLAS(opB),
                    m, n, k, to_BLAS(&alpha), A, ldA, B, ldB, to_BLAS(&beta), C, ldC
                );
            }
            else
            {
                eprint("gemm not defined for scalar type " + TypeName<Scal> );
            }
        }
        
        template<
            Layout layout, Op opA, Op opB,
            Int M = 4, Int N = 4, Int K = 16,
            typename Scal, typename I0, typename I1, typename I2, typename I3, typename I4, typename I5
        >
        force_inline void gemm_hybrid(
            const I0 m, const I1 n, const I2 k,
            cref<Scal> alpha, cptr<Scal> A, const I3 ldA,
                              cptr<Scal> B, const I4 ldB,
            cref<Scal> beta,  mptr<Scal> C, const I5 ldC
        )
        {
            ASSERT_INT(I0);
            ASSERT_INT(I1);
            ASSERT_INT(I2);
            ASSERT_INT(I3);
            ASSERT_INT(I4);
            ASSERT_INT(I5);
            
            if constexpr ( (0 < M) || (0 < N) || (0 < K) )
            {
                gemm<layout,opA,opB>( m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
            }
            else
            {
                if( (m <= M) && (n <= N) && (k <= K) )
                {
                    Tiny::Matrix<M,K,Scal,Int> a;
                    Tiny::Matrix<K,N,Scal,Int> b;
                    
                    a.SetZero();
                    b.SetZero();
                    
                    for( Int i = 0; i < m; ++i )
                    {
                        copy_buffer( &A[k * i], &a[i][0], k );
                    }
                    
                    for( Int i = 0; i < k; ++i )
                    {
                        copy_buffer( &B[n * i], &b[i][0], n );
                    }
                    
                    Tiny::Matrix<M,N,Scal,Int> c = Dot(a,b);
                    
                    
                    for( Int i = 0; i < m; ++i )
                    {
                        copy_buffer( &c[i][0], &C[n * i], n );
                    }
                    
    //                for( Int l = 0; l < k; ++l )
    //                {
    //                    for( Int i = 0; i < m; ++i )
    //                    {
    //                        for( Int j = 0; j < n; ++j )
    //                        {
    //                            C[n*i+j] += A[k*i+l] * B[n*l+j];
    //                        }
    //                    }
    //                }
                }
                else
                {
                    gemm<layout,opA,opB>( m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC );
                }
            }
            
        }
        
    } // namespace BLAS
    
} // namespace Tensors
