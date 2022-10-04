//#######################################################################################
//####                              General matrices                                #####
//#######################################################################################


public:
    
    void Multiply_GeneralMatrix_Vector
    (
        const I     * restrict const rp,
        const I     * restrict const ci,
        const T     * restrict const a,
        const I                      m,
        const I                      n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y
    )
    {
        const JobPointers<I> job_ptr (m,rp,thread_count,false);
        
        Multiply_GeneralMatrix_Vector(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }

    void Multiply_GeneralMatrix_Vector
    (
        const I     * restrict const rp,
        const I     * restrict const ci,
        const T     * restrict const a,
        const I                      m,
        const I                      n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const JobPointers<I> & job_ptr
    )
    {
        SpMM<1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
    }
    
    void Multiply_GeneralMatrix_DenseMatrix
    (
        const I     * restrict const rp,
        const I     * restrict const ci,
        const T     * restrict const a,
        const I                      m,
        const I                      n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const I                      cols
    )
    {
        const JobPointers<I> job_ptr (m,rp,thread_count,false);
        
        Multiply_GeneralMatrix_DenseMatrix(rp,ci,a,m,n,alpha,X,beta,Y,cols,job_ptr);
    }
    
    void Multiply_GeneralMatrix_DenseMatrix
    (
        const I     * restrict const rp,
        const I     * restrict const ci,
        const T     * restrict const a,
        const I                      m,
        const I                      n,
        const T                      alpha,
        const T_in  * restrict const X,
        const T_out                  beta,
              T_out * restrict const Y,
        const I                      cols,
        const JobPointers<I> & job_ptr
    )
    {        
        switch( cols )
        {
            case 1:
            {
                SpMM<1>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 2:
            {
                SpMM<2>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 3:
            {
                SpMM<3>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 4:
            {
                SpMM<4>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 5:
            {
                SpMM<5>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 6:
            {
                SpMM<6>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 7:
            {
                SpMM<7>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 8:
            {
                SpMM<8>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 9:
            {
                SpMM<9>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 10:
            {
                SpMM<10>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 11:
            {
                SpMM<11>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 12:
            {
                SpMM<12>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 13:
            {
                SpMM<13>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 14:
            {
                SpMM<14>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 15:
            {
                SpMM<15>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 16:
            {
                SpMM<16>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 17:
            {
                SpMM<17>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 18:
            {
                SpMM<18>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 19:
            {
                SpMM<19>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            case 0:
            {
                return;
            }
            case 20:
            {
                SpMM<20>(rp,ci,a,m,n,alpha,X,beta,Y,job_ptr);
                break;
            }
            default:
            {
//                wprint(ClassName()+"::Multiply_GeneralMatrix_DenseMatrix: falling back to SpMM_gen for cols = "+ToString(cols)+".");
                SpMM_gen(rp,ci,a,m,n,alpha,X,beta,Y,cols,job_ptr);
            }
        }
    }
