public:
    
    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X,
        const S_out beta,  mut<T_out> Y
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_Vector(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X,
        const S_out beta,  mut<T_out> Y,
        const JobPointers<Int> & job_ptr
    )
    {
        SpMV(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X, const Int incX,
        const S_out beta,  mut<T_out> Y, const Int incY
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_Vector(rp,ci,a,m,n,alpha,X,incX,beta,Y,incY,job_ptr);
    }

    template<typename R_out, typename T_in, typename S_out, typename T_out>
    void Multiply_Vector
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<Scal> a, const Int m, const Int n,
        const R_out alpha, ptr<T_in>  X, const Int incX,
        const S_out beta,  mut<T_out> Y, const Int incY,
        const JobPointers<Int> & job_ptr
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
