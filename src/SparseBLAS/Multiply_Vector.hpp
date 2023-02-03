public:
    
    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X,
        const T_out beta,  mut<T_out> Y
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_Vector(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X,
        const T_out beta,  mut<T_out> Y,
        const JobPointers<Int> & job_ptr
    )
    {
        SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X, const Int incX,
        const T_out beta,  mut<T_out> Y, const Int incY
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_GeneralMatrix_Vector(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
    }

    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X, const Int incX,
        const T_out beta,  mut<T_out> Y, const Int incY,
        const JobPointers<Int> & job_ptr
    )
    {
        SpMM<1>(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
    }
