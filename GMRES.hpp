#pragma once

#include "MyBLAS.hpp"

namespace Tensors
{
    template<size_t eq_count_, typename Scal_, typename Int_, Side side>
    class GMRES
    {
        
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int K = int_cast<Int>(eq_count_);
        
        static constexpr Int gram_schmidt_counts = 2;
        
        using Vector_T = Tiny::Vector<K,Scal,Int>;
        
        using RealVector_T = Tiny::Vector<K,Real,Int>;

    protected:
        
        const Int n;
        const Int max_iter;
        const Int thread_count;
        
        JobPointers<Int> job_ptr;
        
        Tensor3<Scal,Int> Q;
        Tensor3<Scal,Int> H;
        
        Tensor2<Scal,Int> cs;
        Tensor2<Scal,Int> sn;
        Tensor2<Scal,Int> beta;
        
        Tensor2<Scal,Int> x;
        Tensor2<Scal,Int> z;

        ThreadTensor2<Scal,Int>      reduction_buffer;
        ThreadTensor2<Real,Int> real_reduction_buffer;
        
        RealVector_T TOL;
        RealVector_T b_norms;
        RealVector_T r_norms;
        RealVector_T q_norms;
        Vector_T h;
        
        Int iter = 0;
        Int restarts = 0;
        
    public:
        
        GMRES() = delete;
        
        GMRES( const Int n_, const Int max_iter_, const Int thread_count_ )
        :   n               ( n_ )
        ,   max_iter        ( std::min(max_iter_,n) )
        ,   thread_count    ( thread_count_ )
        ,   job_ptr         ( n, thread_count )
        ,   Q               ( max_iter + 1, n, K )
        ,   H               ( max_iter + 1, max_iter, K )
        ,   cs              ( max_iter,     K )
        ,   sn              ( max_iter,     K )
        ,   beta            ( max_iter + 1, K )
        ,   x               ( n, K )
        ,   z               ( n, K )
        ,        reduction_buffer ( thread_count, K )
        ,   real_reduction_buffer ( thread_count, K )
        {}
        
        
        ~GMRES() = default;
        
        
        template<typename Operator_T, typename Preconditioner_T>
        bool operator()(
            Operator_T       & A,
            Preconditioner_T & P,
            ptr<Scal> b_in,       const Int ldb,
            mut<Scal> x_inout,    const Int ldx,
            const Real relative_tolerance,
            const Int  max_restarts
        )
        {
            ptic(ClassName()+": Compute norm of right hand side.");
            // Compute norms of b.
            x.Read( b_in, ldx, thread_count );
            
            if constexpr ( side == Side::Left )
            {
                P( x.data(), z.data() );
                
            }
            else
            {
                swap(x,z);
            }
            
            ComputeNorms( z.data(), b_norms );
            
            TOL = b_norms;
            TOL *= relative_tolerance;
            
            if( TOL.Max() <= Scalar::Zero<Scal> )
            {
                x.Write( x_inout, ldx, thread_count );
               
                return true;
            }
            
            ptoc(ClassName()+": Compute norm of right hand side.");
            
            restarts = 0;
            bool succeeded = false;
            
            while( !succeeded && (restarts < max_restarts) )
            {
                succeeded = Solve( A, P, b_in, ldb, x_inout, ldx, relative_tolerance );
                ++restarts;
            }
            
            return succeeded;
        }
        
    protected:
        
        template<typename Operator_T, typename Preconditioner_T>
        bool Solve(
            Operator_T       & A,
            Preconditioner_T & P,
            ptr<Scal> b_in,       const Int ldb,
            mut<Scal> x_inout,    const Int ldx,
            const Real relative_tolerance
        )
        {
            ptic(ClassName()+"::Solve");
            
            h.Fill(Scalar::One<Scal>);
            
            x.Read( x_inout, ldx, thread_count );
            
            if constexpr ( side == Side::Left )
            {
                A( x.data(), z.data() );
                
                // z = A.x - b;
                MulAdd<Scalar::Flag::Minus>( z.data(), b_in, h );
                
                // Q[0] = P.(A.x-b)
                P( z.data(), Q.data(0) );
            }
            else
            {
                // Q[0] = A.x-b
                A( x.data(), Q.data(0) );
                MulAdd<Scalar::Flag::Minus>( Q.data(0), b_in, h );
            }
            
            // Residual norms
            ComputeNorms( Q.data(0), r_norms );
            
            // Normalize Q[0]
            InverseScale( Q.data(0), r_norms );
            
            // Initialize beta
            beta.SetZero();
            r_norms.Write( &beta[0][0] );
            
            H.SetZero();
            
            
            iter = 0;
            
            bool succeeded = CheckResiduals();
            
            while( !succeeded && iter < max_iter )
            {
                ArnoldiStep( A, P );
                
                ApplyGivensRotations();
                
                ++iter;
                
                succeeded = CheckResiduals();
            }
            
            ptic(ClassName()+": Solve least squares system.");
            
            // TODO: y = H[1..iter][1..iter] \ beta[1..iter];
            Tensor2<Scal,Int> H_mat    (iter,iter);
            Tensor1<Scal,Int> beta_vec (iter);
            Tensor2<Scal,Int> y        (iter,K);
            
            for( Int k = 0; k < K; ++k )
            {
                for( Int i = 0; i < iter; ++i )
                {
                    beta_vec[i] = beta(i,k);
                    
                    for( Int j = 0; j < iter; ++j )
                    {
                        H_mat(i,j) = H(i,j,k);
                    }
                }
                
                // Solve H_mat.y = beta_vec.
                BLAS_Wrappers::trsv<
                    Layout::RowMajor, UpLo::Upper, Op::Id, Diag::NonUnit
                >( iter, H_mat.data(), iter, beta_vec.data(), static_cast<Int>(1) );
                
                for( Int j = 0; j < iter; ++j )
                {
                    y[j][k] = beta_vec[j];
                }
            }
            
            ptoc(ClassName()+": Solve least squares system.");
            
            ptic(ClassName()+": Synthesize solution.");
            // z = y * Q;
            z.SetZero();
            
            ParallelDo(
                [this]( const Int i )
                {
                    for( Int j = 0; j < iter; ++j )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            z(i,k) += y(j,k) * Q(j,i,k);
                        }
                    }
                },
                job_ptr
            );
            
            if constexpr( side == Side::Left )
            {
                // x = z
                swap(x,z);
            }
            else
            {
                // x = P[z]
                P( z.data(), x.data() );
            }
            
            //x_inout -= x
            combine_buffers<Scalar::Flag::Minus,Scalar::Flag::Plus>(
                -Scalar::One<Real>, x.data(), Scalar::One<Real>, x_inout, n * K, thread_count
            );
            
            ptoc(ClassName()+": Synthesize solution.");
            
            ptoc(ClassName()+"::Solve");
            
            return succeeded;
        }
        
        
    protected:
        
        template<typename Operator_T, typename Preconditioner_T>
        void ArnoldiStep( Operator_T & A, Preconditioner_T & P )
        {
            ptic(ClassName()+"::ArnoldiStep");
            
            mut<Scal> q = Q.data(iter+1);
                        
            ptic(ClassName()+"::Apply Operators");
            if constexpr( side == Side::Left )
            {
                // z = A.Q[iter]
                A( Q.data(iter), z.data() );
                // Q[iter+1] = P.A.Q[iter]
                P( z.data(), q );
            }
            else
            {
                // z = P.Q[iter]
                P( Q.data(iter), z.data() );
                // Q[iter+1] = A.P.Q[iter]
                A( z.data(), q );
            }
            ptoc(ClassName()+"::Apply Operators");

            // Several runs of Gram-Schmidt algorithm.
            // Rumor has it that Kahan's "twice is enough" statement states that gram_schmidt_counts does not need to be greater then 2.
            // But gram_schmidt_counts = 1 seems to produce good GMRES solutions, even if Q is not perfectly orthogonalized.
            ptic(ClassName()+" Gram-Schmidt");
            
            for( Int gs_iter = 0; gs_iter < gram_schmidt_counts; ++ gs_iter)
            {
                for( Int i = 0; i < iter+1; ++ i )
                {
                
                    // h = Q[i] . Q[iter+1];
                    ComputeScalarProducts( Q.data(i), q, h );
                    
                    // H[i] += h;
                    for( Int k = 0; k < K; ++ k )
                    {
                        H(i,iter,k) += h[k];
                    }
                    
                    // Q[iter+1] -= Q[i] * h;
                    MulAdd<Scalar::Flag::Minus>( q, Q.data(i), h );
                }
            }
            ptoc(ClassName()+" Gram-Schmidt");
            
            // Residual norms
            ComputeNorms( q, q_norms );
            
            // H[iter+1][iter] += q_norms;
            for( Int k = 0; k < K; ++ k )
            {
                H(iter+1,iter,k) = q_norms[k];
            }
            
            // q /= q_norms;
            InverseScale( q, q_norms );
            
            ptoc(ClassName()+"::ArnoldiStep");
        }
        
        void ApplyGivensRotations()
        {
            ptic(ClassName()+"::ApplyGivensRotations");
            
            for( Int i = 0; i < iter; ++ i )
            {
                for( Int k = 0; k < K; ++k )
                {
                    const Scal xi  = H(i  ,iter,k);
                    const Scal eta = H(i+1,iter,k);
                    
                    const Scal cos = cs[i][k];
                    const Scal sin = sn[i][k];
                    
                    H(i  ,iter,k) =  cos * xi + sin * eta;
                    H(i+1,iter,k) = -Scalar::Conj(sin) * xi + cos * eta;
                }
            }
            {
                for( Int k = 0; k < K; ++k )
                {
                    const Scal xi  = H(iter  ,iter,k);
                    const Scal eta = H(iter+1,iter,k);
                    
                    Scal cos;
                    Scal sin;
                    
                    const Real r = std::sqrt( Scalar::AbsSquared(xi) + Scalar::AbsSquared(eta) );
                    
                    if( abs(xi) <= Scalar::eps<Scal> * r )
                    {
                        cos = Scalar::Zero<Scal>;
                        sin = Scalar::One<Scal>;
                    }
                    else
                    {
                        const Real r_inv = Scalar::Inv(r);
                        
                        cos =  std::abs(xi) * r_inv;
                        sin = (xi / std::abs(xi)) * Scalar::Conj(eta) * r_inv;
                    }
                    
                    cs[iter][k] = cos;
                    sn[iter][k] = sin;

                    H(iter  ,iter,k) = cos * xi + sin * eta;
                    H(iter+1,iter,k) = Scalar::Zero<Scal>;
                                                                      
                    beta[iter+1][k] = -Scalar::Conj(sin) * beta[iter][k];
                    beta[iter  ][k] =  cos * beta[iter][k];
                }
            }
            
            ptoc(ClassName()+"::ApplyGivensRotations");
        }
        
        void ComputeNorms( ptr<Scal> v, RealVector_T & norms )
        {
            ParallelDo(
                [this,v,&norms]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    RealVector_T sums;
                    
                    sums.SetZero();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            sums[k] += Scalar::AbsSquared(v[K * i + k]);
                        }
                    }
                    
                    sums.Write( &real_reduction_buffer[thread][0] );
                },
                thread_count
            );
            
            real_reduction_buffer.AddReduce( norms.data(), false );
            
            for( Int k = 0; k < K; ++k )
            {
                norms[k] = std::sqrt( norms[k] );
            }
        }
        
        void ComputeScalarProducts( ptr<Scal> v, ptr<Scal> w, Vector_T & dots )
        {
            ParallelDo(
                [this,v,w,&dots]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    Vector_T sums;
                    
                    sums.SetZero();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            sums[k] += Scalar::Conj(v[K * i + k]) * w[K * i + k];
                        }
                    }
                    
                    sums.Write( &reduction_buffer[thread][0] );
                },
                thread_count
            );
            
            reduction_buffer.AddReduce( dots.data(), false );
        }
        
        template<Scalar::Flag flag>
        void MulAdd( mut<Scal> v, ptr<Scal> w, const Vector_T & factors )
        {
            ParallelDo(
                [this,v,w,&factors]( const Int i )
                {
                    for( Int k = 0; k < K; ++k )
                    {
                        if constexpr ( flag == Scalar::Flag::Minus )
                        {
                            v[K * i + k] -= w[K * i + k] * factors[k];
                        }
                        else
                        {
                            v[K * i + k] += w[K * i + k] * factors[k];
                        }
                    }
                },
                job_ptr
            );
        }
        
        void InverseScale( mut<Scal> q, const RealVector_T & factors )
        {
            ParallelDo(
                [this,q,&factors]( const Int thread )
                {
                    RealVector_T factors_inv;
                    
                    for( Int k = 0; k < K; ++k )
                    {
                        factors_inv[k] = Scalar::Inv( factors[k] );
                    }
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            q[K * i + k] *= factors_inv[k];
                        }
                    }
                },
                thread_count
            );
        }
        
    public:
        
        Int IterationCount() const
        {
            return iter;
        }
        
        Int RestartCount() const
        {
            return restarts;
        }
        
        RealVector_T Residuals() const
        {
            RealVector_T res;
            
            for( Int k = 0; k < K; ++k )
            {
                res[k] = std::abs(beta[iter][k]);
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res;
            
            for( Int k = 0; k < K; ++k )
            {
                res[k] = std::abs(beta[iter][k]) / b_norms[k];
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succeeded = true;
            for( Int k = 0; k < K; ++k )
            {
                succeeded = succeeded && (std::abs(beta[iter][k]) <= TOL[k]);
            }
            
            return succeeded;
        }
        
        const Tensor3<Scal,Int> & GetBasis() const
        {
            return Q;
        }
        
        const Tensor3<Scal,Int> & GetHessenbergMatrix() const
        {
            return H;
        }
        
        std::string ClassName() const
        {
            return std::string(
                "GMRES<"+ToString(K)+","+TypeName<Scal>+","+TypeName<Int>+","+COND(side==Side::Left,"Left","Right")+">"
            );
        }
    }; // class GMRES
        
        
    
        
} // namespace Tensors
