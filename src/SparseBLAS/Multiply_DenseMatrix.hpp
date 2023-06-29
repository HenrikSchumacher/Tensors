//#######################################################################################
//####                              General matrices                                #####
//#######################################################################################

public:
    
//    // Calling this NRHS <= 0 will use generic implementation.
//    // Calling with NRHS > 0  will uses dimension-dependent implementation
//    template<Int NRHS, typename R_out, typename T_in, typename S_out, typename T_out>
//    void Multiply_DenseMatrix
//    (
//        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
//        const R_out alpha, ptr<T_in>  X, const Int ldX,
//        const S_out beta,  mut<T_out> Y, const Int ldY,
//        const Int   nrhs,
//        const JobPointers<Int> & job_ptr
//    )
//    {
//        StaticParameterCheck<R_out,T_in,S_out,T_out>();
//
//        if( nrhs == 1 )
//        {
//            if( ldX == 1 && ldY == 1)
//            {
//                SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
//            }
//            else
//            {
//                SpMM_fixed<1>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
//            }
//
//            return;
//        }
//
//        if constexpr ( NRHS == VarSize )
//        {
//            SpMM_gen(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,nrhs,job_ptr);
//        }
//        else
//        {
//            if( NRHS == nrhs )
//            {
//                SpMM_fixed<NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
//            }
//            else
//            {
//                SpMM_gen(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,nrhs,job_ptr);
//            }
//        }
//    }

public:

    // Calling this NRHS <= 0 will use generic implementation.
    // Calling with NRHS > 0  will uses dimension-dependent implementation
    template<Int NRHS = VarSize, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_DenseMatrix
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X, const Int ldX,
        const S_out beta,  mut<T_out> Y, const Int ldY,
        const Int   nrhs,
        const JobPointers<Int> & job_ptr
    )
    {
        StaticParameterCheck<R_out,T_in,S_out,T_out>();

        if( nrhs == Scalar::One<Int> )
        {
            if( ldX == Scalar::One<Int> && ldY == Scalar::One<Int> )
            {
                SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
            }
            else
            {
                SpMM<1>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs);
            }

            return;
        }
        else
        {
            SpMM<NRHS>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr,nrhs );
        }
    }
