#pragma once

namespace Tensors
{
    
    namespace MyBLAS
    {
        template<
            Op opA, Op opB,
            int M, int N, int K,
            Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
            typename Scal
        >
        class GEMM
        {
        public:
            
            static constexpr int MaxM = 16;
            static constexpr int MaxN = 16;
            static constexpr int MaxK = 16;
            
            GEMM() = default;
            
            ~GEMM() = default;
            
            void operator()(
                const int m, const int n, const int k,
                cref<Scal> alpha, cptr<Scal> A, const int ldA,
                                  cptr<Scal> B, const int ldB,
                cref<Scal> beta,  mptr<Scal> C, const int ldC
            )
            {
                if constexpr( (1 <= M) && (M<=MaxM) && (1 <= N) && (N<=MaxN) && (1 <= K) && (K<=MaxK) )
                {
                    Tiny::gemm<opA,opB,M,N,K,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                }
                else if constexpr( (1 <= N) && (N<=MaxN) )
                {
                    gemm_N_<N>(m,k,alpha,A,ldA,B,ldB,beta,C,ldC);
                }
//                else if constexpr( (1 <= M) && (M<=MaxM) )
//                {
//                    gemmM__<M>(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
//                }
//                else if constexpr( (1 <= K) && (K<=MaxK) )
//                {
//                    gemm__K<K>(m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC);
//                }
                else
                {
                    if constexpr ( N == 1 )
                    {
                        Tensors::BLAS::gemv<Layout::RowMajor, opA>(
                            m,k,alpha,A,ldA,B,ldB,beta,C,ldC
                        );
                    }
                    else
                    {
                        Tensors::BLAS::gemm<Layout::RowMajor, opA, opB>(
                            m,n,k,alpha,A,ldA,B,ldB,beta,C,ldC
                        );
                    }
                }
            }
            
            
        protected:
            
            template<int N_>
            void gemm_N_(
                const int m, const int k,
                cref<Scal> alpha, cptr<Scal> A, const int ldA,
                                  cptr<Scal> B, const int ldB,
                cref<Scal> beta,  mptr<Scal> C, const int ldC
            )
            {
                switch( m )
                {
                    case 1:
                    {
                        gemmMN_<1,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 2:
                    {
                        gemmMN_<2,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 3:
                    {
                        gemmMN_<3,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 4:
                    {
                        gemmMN_<4,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 5:
                    {
                        gemmMN_<5,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 6:
                    {
                        gemmMN_<6,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 7:
                    {
                        gemmMN_<7,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 8:
                    {
                        gemmMN_<8,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 9:
                    {
                        gemmMN_<9,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 10:
                    {
                        gemmMN_<10,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 11:
                    {
                        gemmMN_<11,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 12:
                    {
                        gemmMN_<12,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 13:
                    {
                        gemmMN_<13,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 14:
                    {
                        gemmMN_<14,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 15:
                    {
                        gemmMN_<15,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 16:
                    {
                        gemmMN_<16,N_>(k,alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    default:
                    {
                        if constexpr ( N_ == 1 )
                        {
                            Tensors::BLAS::gemv<Layout::RowMajor,opA>(
                                m,k,alpha,A,ldA,B,ldB,beta,C,ldC
                            );
                        }
                        else
                        {
                            Tensors::BLAS::gemm<Layout::RowMajor,opA,opB>(
                                m,N_,k,alpha,A,ldA,B,ldB,beta,C,ldC
                            );
                        }
                    }
                }
            }
            
            template<int M_, int N_>
            void gemmMN_(
                const int k,
                cref<Scal> alpha, cptr<Scal> A, const int ldA,
                                  cptr<Scal> B, const int ldB,
                cref<Scal> beta,  mptr<Scal> C, const int ldC
            )
            {
                switch( k )
                {
                    case 1:
                    {
                        Tiny::gemm<opA,opB,M_,N_,1,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 2:
                    {
                        Tiny::gemm<opA,opB,M_,N_,2,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 3:
                    {
                        Tiny::gemm<opA,opB,M_,N_,3,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 4:
                    {
                        Tiny::gemm<opA,opB,M_,N_,4,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 5:
                    {
                        Tiny::gemm<opA,opB,M_,N_,5,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 6:
                    {
                        Tiny::gemm<opA,opB,M_,N_,6,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 7:
                    {
                        Tiny::gemm<opA,opB,M_,N_,7,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 8:
                    {
                        Tiny::gemm<opA,opB,M_,N_,8,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 9:
                    {
                        Tiny::gemm<opA,opB,M_,N_,9,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 10:
                    {
                        Tiny::gemm<opA,opB,M_,N_,10,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 11:
                    {
                        Tiny::gemm<opA,opB,M_,N_,11,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 12:
                    {
                        Tiny::gemm<opA,opB,M_,N_,12,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 13:
                    {
                        Tiny::gemm<opA,opB,M_,N_,13,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 14:
                    {
                        Tiny::gemm<opA,opB,M_,N_,14,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 15:
                    {
                        Tiny::gemm<opA,opB,M_,N_,15,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    case 16:
                    {
                        Tiny::gemm<opA,opB,M_,N_,16,alpha_flag,beta_flag>(alpha,A,ldA,B,ldB,beta,C,ldC);
                        return;
                    }
                    default:
                    {
                        if constexpr ( N_ == 1 )
                        {
                            Tensors::BLAS::gemv<Layout::RowMajor,opA>(
                                M_,k,alpha,A,ldA,B,ldB,beta,C,ldC
                            );
                        }
                        else
                        {
                            Tensors::BLAS::gemm<Layout::RowMajor,opA,opB>(
                                M_,N_,k,alpha,A,ldA,B,ldB,beta,C,ldC
                            );
                        }
                    }
                }
            }
        }; // class GEMM
        
    } // namespace MyBLAS
    
} // namespace Tensors
