public:

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  X,
        cref<S_out> beta,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  X, const Int incX,
        cref<S_out> beta,  mptr<T_out> Y, const Int incY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( (incX == 1) && (incY == 1) )
        {
            SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        }
        else
        {
            SpMM<1>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
        }
    }


    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector_Transposed
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  X,
        cref<S_out> beta,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        SpMV_Transposed(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector_Transposed
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        cref<R_out> alpha, cptr<T_in>  X, const Int incX,
        cref<S_out> beta,  mptr<T_out> Y, const Int incY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( (incX == 1) && (incY == 1) )
        {
            SpMV_Transpose(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        }
        else
        {
            SpMM_Transposed<1>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
        }
    }
