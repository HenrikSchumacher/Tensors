public:

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X,
        const S_out beta,  mptr<T_out> Y,
        cref<JobPointers<Int>> job_ptr
    )
    {
        SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        cptr<LInt> rp, cptr<Int> ci, cptr<Scal> a, const Int m, const Int n,
        const R_out alpha, cptr<T_in>  X, const Int incX,
        const S_out beta,  mptr<T_out> Y, const Int incY,
        cref<JobPointers<Int>> job_ptr
    )
    {
        if( incX == 1 && incY == 1)
        {
            SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
        }
        else
        {
            SpMM<1>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
        }
    }
