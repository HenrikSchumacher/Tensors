//#######################################################################################
//####                              General matrices                                #####
//#######################################################################################


public:
    
    void Multiply_DenseMatrix
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X, const Int ldX,
        const T_out beta,  mut<T_out> Y, const Int ldY,
        const Int   cols
    )
    {
        const JobPointers<Int> job_ptr (m,rp,thread_count,false);
        
        Multiply_DenseMatrix(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
    }
    
    void Multiply_DenseMatrix
    (
        ptr<LInt> rp, ptr<Int> ci, ptr<T> a, const Int m, const Int n,
        const T     alpha, ptr<T_in>  X, const Int ldX,
        const T_out beta,  mut<T_out> Y, const Int ldY,
        const Int   cols,
        const JobPointers<Int> & job_ptr
    )
    {
        switch( cols )
        {
            case 1:
            {
                SpMM_fixed<1>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 2:
            {
                SpMM_fixed<2>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 3:
            {
                SpMM_fixed<3>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 4:
            {
                SpMM_fixed<4>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 5:
            {
                SpMM_fixed<5>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 6:
            {
                SpMM_fixed<6>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 7:
            {
                SpMM_fixed<7>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 8:
            {
                SpMM_fixed<8>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 9:
            {
                SpMM_fixed<9>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 10:
            {
                SpMM_fixed<10>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 11:
            {
                SpMM_fixed<11>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 12:
            {
                SpMM_fixed<12>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 13:
            {
                SpMM_fixed<13>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 14:
            {
                SpMM_fixed<14>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 15:
            {
                SpMM_fixed<15>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 16:
            {
                SpMM_fixed<16>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 17:
            {
                SpMM_fixed<17>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 18:
            {
                SpMM_fixed<18>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 19:
            {
                SpMM_fixed<19>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 20:
            {
                SpMM_fixed<20>(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,job_ptr);
                break;
            }
            case 0:
            {
                return;
            }
            default:
            {
                SpMM_gen(rp,ci,a,m,n,alpha,X,ldX,beta,Y,ldY,cols,job_ptr);
            }
        }
    }
