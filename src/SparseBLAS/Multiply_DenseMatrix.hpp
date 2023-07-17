//#######################################################################################
//####                              General matrices                                #####
//#######################################################################################

public:

    // Calling this NRHS <= 0 will use generic implementation.
    // Calling with NRHS > 0  will uses dimension-dependent implementation
    template<Int NRHS = VarSize, typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_DenseMatrix
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  X, const Int ldX,
        cref<S_out> beta,  mptr<T_out> Y, const Int ldY,
        const Int   nrhs,
        cref<JobPointers<Int>> job_ptr
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
