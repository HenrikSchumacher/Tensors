#pragma once

#include "BLAS_Wrappers.hpp"

namespace Tensors
{
    // NRHS_ = number of right hand sides.
    // If you know this at compile time, then enter it in the template.
    // If you don't know this at compile time, then use NRHS_ = VarSize (==0) and specify
    // the nrhs_ in the constructor.
    
    // Scal_ is the floating point type that is used internally.
    
    template<Size_T NRHS_,typename Scal_,typename Int_, Side side,
        bool A_verboseQ = true, bool P_verboseQ = true
    >
    class GMRES
    {
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int NRHS = int_cast<Int>(NRHS_);
        
        static constexpr Int gram_schmidt_counts = 2;
        
        using Vector_T     = Tensor1<Scal,Int>;
        
        using RealVector_T = Tensor1<Real,Int>;
        
        using F_T = Scalar::Flag;

    protected:
        
        const Int n;
        const Int max_iter;
        const Int nrhs;
        const Int thread_count;
        
        JobPointers<Int> job_ptr;
        
        Tensor3<Scal,Int> Q;
        Tensor3<Scal,Int> H;
        
        Tensor2<Scal,Int> cs;
        Tensor2<Scal,Int> sn;
        Tensor2<Scal,Int> beta;
        
        Tensor2<Scal,Int> x;
        Tensor2<Scal,Int> z;

        ThreadTensor2<Scal,Int>      red_buf;
        ThreadTensor2<Real,Int> real_red_buf;
        
        RealVector_T TOL;
        RealVector_T b_norms;
        RealVector_T r_norms;
        RealVector_T q_norms;
        Vector_T h;
        
        Int iter = 0;
        Int restarts = 0;
        Int max_restarts = 30;
        Real relative_tolerance = 0.00001;
        
        bool use_initial_guessQ = false;
        bool succeeded          = false;
        
    public:
        
        GMRES() = delete;
        
        GMRES(
            const Int n_,
            const Int max_iter_,
            const Size_T nrhs_ = NRHS,
            const Size_T thread_count_ = 1
        )
        :   n               ( n_                                    )
        ,   max_iter        ( Min(max_iter_,n+1)                    )
        ,   nrhs            ( (NRHS > VarSize) ? NRHS : nrhs_       )
        ,   thread_count    ( static_cast<Int>(thread_count_)       )
        ,   job_ptr         ( n, thread_count                       )
        ,   Q               ( max_iter + 1, n, nrhs                 )
        ,   H               ( max_iter + 1, max_iter, nrhs          )
        ,   cs              ( max_iter,     nrhs                    )
        ,   sn              ( max_iter,     nrhs                    )
        ,   beta            ( max_iter + 1, nrhs                    )
        ,   x               ( n,            nrhs                    )
        ,   z               ( n,            nrhs                    )
        ,        red_buf    ( thread_count, nrhs                    )
        ,   real_red_buf    ( thread_count, nrhs                    )
        ,   TOL             ( nrhs                                  )
        ,   b_norms         ( nrhs                                  )
        ,   r_norms         ( nrhs                                  )
        ,   q_norms         ( nrhs                                  )
        ,   h               ( nrhs                                  )
        {}
        
        
        ~GMRES() = default;
        
        
        // Computes X_inout <- a * A^{-1} . B_in + b * X_inout via CG method.
        // Uses P as precondition, i.e., P should be a proxy of A^{-1}.
        
        template<
            typename Operator_T, typename Preconditioner_T,
            typename a_T, typename B_T, typename b_T, typename X_T
        >
        bool operator()(
            mref<Operator_T>       A,
            mref<Preconditioner_T> P,
            const a_T a, cptr<B_T> B_in,    const Int ldB,
            const b_T b, mptr<X_T> X_inout, const Int ldX,
            const Real relative_tolerance_,
            const Int  max_restarts_,
            // We force the use to actively request that the initial guess is used,
            // because this is a common source of bugs.
            const bool use_initial_guessQ_ = false
        )
        {
            // `A` and `P` must be a functions or lambda with prototypes
            //
            // `A( cptr<Scal> x, mptr<Scal> y)`
            //
            // and
            //
            // `P( cptr<Scal> x, mptr<Scal> y)`
            //
            // They may return something, but the returned value is ignored.
            
            std::string tag = ClassName() + "::operator<" + TypeName<B_T> + "," + TypeName<X_T> + ">()";
            
            ptic(tag);
            
            relative_tolerance = relative_tolerance_;
            max_restarts       = max_restarts_;
            use_initial_guessQ = use_initial_guessQ_;
            
            if( use_initial_guessQ && (b != b_T(0)) )
            {
                wprint( tag + ": use_initial_guessQ == true and b != 0. Typically, this does not make sense." );
            }
            
            ptic(ClassName()+": Compute norm of right hand side.");
            
            // Compute norms of b.
            x.Read( B_in, ldX, thread_count );
            
            if constexpr ( side == Side::Left )
            {
                ApplyPreconditioner(P,x.data(),z.data());
            }
            else
            {
                // TODO: Not sure whether this makes sense.
                swap(x,z);
            }
            
            ComputeNorms( z.data(), b_norms );
            
            TOL = b_norms;
            
            TOL *= relative_tolerance;
            
            ptoc(ClassName()+": Compute norm of right hand side.");
            
            iter = 0;
            restarts = 0;
            succeeded = false;
            
            if( b_norms.CountNaNs() > 0 )
            {
                eprint(tag + ": Right-hand side contains NaNs. Doing nothing.");
                
                succeeded = false;
                
                logprint( Stats() );
                
                ptoc(tag);
                
                return succeeded;
            }
            
            if( b_norms.Max() <= 0 )
            {
                ParallelDo(
                    [X_inout,ldX,this]( const Int i )
                    {
                        zerofy_buffer<NRHS>( &X_inout[ldX * i], nrhs );
                    },
                    n, thread_count
                );
               
                succeeded = true;
                
                wprint(tag + ": Right-hand side is 0. Returning zero vector.");
                
                logprint( Stats() );
                
                ptoc(tag);
                
                return succeeded;
            }
            
            while( !succeeded && (restarts <= max_restarts) )
            {
                succeeded = Solve( 
                    A, P, a, B_in, ldB, b, X_inout, ldX, relative_tolerance,
                    use_initial_guessQ || (restarts > 0)
                );
                ++restarts;
            }
            
            // We have to correct this for the statistics.
            --restarts;

            logprint( Stats() );
            
            ptoc(tag);
            
            return succeeded;
        }
        
    protected:
        
        template<
            typename Operator_T, typename Preconditioner_T,
            typename a_T, typename B_T, typename b_T, typename X_T
        >
        bool Solve(
            mref<Operator_T>       A,
            mref<Preconditioner_T> P,
            const a_T a, cptr<B_T> B_in,    const Int ldB,
            const b_T b, mptr<X_T> X_inout, const Int ldX,
            const Real relative_tolerance,
            bool use_initial_guessQ
        )
        {
            std::string tag = ClassName() + "::Solve<" + TypeName<B_T> + "," + TypeName<X_T> + ">";
            
//            ptic(tag);
            
            h.SetZero();
            
            // TODO: Remove the redudant computations in the case of a restart.
            
            if( use_initial_guessQ )
            {
                logprint( tag + ": Using X_inout as initial guess." );
                x.Read( X_inout, ldX, thread_count );
            }
            else
            {
                // Not neccessary anymore.
//                x.SetZero( thread_count );
            }
            
            if constexpr ( side == Side::Left )
            {
                // z = A.x - b;
                if( use_initial_guessQ )
                {
                    // z = A.x - b;
                    ApplyOperator(A,x.data(),z.data());
                    
                    combine_matrices<F_T::Minus,F_T::Plus,VarSize,NRHS,Parallel>
                    (
                        -Scalar::One<Scal>, B_in,     ldB,
                         Scalar::One<Scal>, z.data(), nrhs,
                        n, nrhs, thread_count
                    );
                }
                else
                {
                    // z = -b
                    combine_matrices<F_T::Minus,F_T::Zero,VarSize,NRHS,Parallel>
                    (
                        -Scalar::One<Scal>, B_in,     ldB,
                        Scalar::Zero<Scal>, z.data(), nrhs,
                        n, nrhs, thread_count
                    );
                }
                
                // Q[0] = P.( A.x - b )
                ApplyPreconditioner(P,z.data(),Q.data(0));
            }
            else
            {
                // TODO: Test this!
                wprint(tag + ": Right preconditioner is untested. Please double-check you results.");
                
                // Q[0] = A.x - b
                if( use_initial_guessQ )
                {
                    // Q[0] = A.x - b
                    ApplyOperator(A,x.data(),Q.data(0));
                    
                    combine_matrices<F_T::Minus,F_T::Plus,VarSize,NRHS,Parallel>
                    (
                        -Scalar::One<Scal>, B_in,      ldB,
                         Scalar::One<Scal>, Q.data(0), nrhs,
                        n, nrhs, thread_count
                    );
                }
                else
                {
                    // Q[0] = -b
                    combine_matrices<F_T::Minus,F_T::Zero,VarSize,NRHS,Parallel>
                    (
                        -Scalar::One<Scal>, B_in,      ldB,
                        Scalar::Zero<Scal>, Q.data(0), nrhs,
                        n, nrhs, thread_count
                    );
                }
            }
            
            // Residual norms
            ComputeNorms( Q.data(0), r_norms );
            
            // TODO: What happens here if some entries of r are exactly 0?
            // Normalize Q[0]
            InverseScale( Q.data(0), r_norms );
            
            // Initialize beta
            beta.SetZero();
            
            r_norms.Write( &beta[0][0] );
            
            H.SetZero();
            
            
            iter = 0;
            
            bool succ = CheckResiduals();
            
            if( succ )
            {
                return succ;
            }
            
            while( !succ && iter < max_iter )
            {
                ArnoldiStep( A, P );
                
                ApplyGivensRotations();
                
                ++iter;
                
                succ = CheckResiduals();
            }
            
            ptic(ClassName()+": Solve least squares system.");
            
            Tensor2<Scal,Int> H_mat    (iter,iter);
            Tensor1<Scal,Int> beta_vec (iter);
            Tensor2<Scal,Int> y        (iter,nrhs);
            
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
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
                BLAS::trsv<Layout::RowMajor, UpLo::Upper, Op::Id, Diag::NonUnit>(
                    iter, H_mat.data(), iter, beta_vec.data(), Int(1)
                );
                
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
                [&]( const Int i )
                {
                    mptr<Scal> z_i = z.data(i);
                    
                    for( Int j = 0; j < iter; ++j )
                    {
                        cptr<Scal> q_j_i = Q.data(j,i);
                        cptr<Scal> y_j   = y.data(j);
                        
                        for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                        {
                            z_i[k] += y_j[k] * q_j_i[k];
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
                ApplyPreconditioner(P,z.data(),x.data());
            }
            
            
            
            
            if( use_initial_guessQ )
            {
                // solution = X_inout - x.
                
                // We return a * solution + b * X_inout = -a * x + (b+1) * X_inout
                combine_matrices_auto<VarSize,NRHS,Parallel>(
                    -scalar_cast<X_T>(a),    x.data(), nrhs,
                    scalar_cast<X_T>(b + 1), X_inout,  ldX,
                    n, nrhs, thread_count
                );
            }
            else
            {
                // solution = 0 - x.
                
                // We return a * solution + b * X_inout = -a * x + 0
                combine_matrices_auto<VarSize,NRHS,Parallel>(
                    -scalar_cast<X_T>(a), x.data(), nrhs,
                    scalar_cast<X_T>(b), X_inout,  ldX,
                    n, nrhs, thread_count
                );
            }

            
            ptoc(ClassName()+": Synthesize solution.");
            
//            ptoc(tag);
            
            return succ;
        }
        
        
    protected:
        
        template<typename Operator_T>
        void ApplyOperator(
            mref<Operator_T> A, cptr<Scal> X, mptr<Scal> Y
        )
        {
            if constexpr ( A_verboseQ )
            {
                ptic(ClassName()+ "::ApplyOperator");
            }
            
            (void)A( X, Y );
            
            if constexpr ( A_verboseQ )
            {
                ptoc(ClassName()+ "::ApplyOperator");
            }
        }
        
        template<typename Preconditioner_T>
        void ApplyPreconditioner(
            mref<Preconditioner_T> P, cptr<Scal> X, mptr<Scal> Y
        )
        {
            if constexpr ( P_verboseQ )
            {
                ptic(ClassName()+ "::ApplyPreconditioner");
            }
            
            (void)P( X, Y );
            
            if constexpr ( P_verboseQ )
            {
                ptoc(ClassName()+ "::ApplyPreconditioner");
            }
        }
        
        template<typename Operator_T, typename Preconditioner_T>
        void ArnoldiStep( mref<Operator_T> A, mref<Preconditioner_T> P )
        {
            ptic(ClassName()+"::ArnoldiStep");
            
            // Pivot element
            mptr<Scal> q_p = Q.data(iter+1);
                        
            if constexpr( side == Side::Left )
            {
                // z = A.Q[iter]
                ApplyOperator      (A,Q.data(iter),z.data());
                
                // Q[iter+1] = P.A.Q[iter]
                ApplyPreconditioner(P,z.data(),q_p);            }
            else
            {
                // z = P.Q[iter]
                ApplyPreconditioner(P,Q.data(iter),z.data());
                
                // Q[iter+1] = A.P.Q[iter]
                ApplyOperator      (A,z.data(),q_p);
            }

            // Several runs of Gram-Schmidt algorithm.
            // Rumor has it that Kahan's "twice is enough" statement states that gram_schmidt_counts does not need to be greater then 2.
            // But gram_schmidt_counts = 1 seems to produce good GMRES solutions, even if Q is not perfectly orthogonalized.
            ptic(ClassName()+" Gram-Schmidt");
            
            for( Int gs_iter = 0; gs_iter < gram_schmidt_counts; ++ gs_iter)
            {
                for( Int i = 0; i < iter+1; ++ i )
                {
                    mptr<Scal> q_i = Q.data(i);
                    
                    // h = Q[i] . Q[iter+1];
                    ComputeScalarProducts( q_i, q_p, h );
                    
                    // H[i] += h;
                    for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++ k )
                    {
                        H(i,iter,k) += h[k];
                    }
                    
                    // Q[iter+1] -= Q[i] * h;
                    ParallelDo(
                        [this,q_i,q_p]( const Int j )
                        {
                            for( Int k = 0; k < ((NRHS>VarSize) ? NRHS : nrhs); ++k )
                            {
                                mptr<Scal> q_p_j = &q_p[nrhs * j];
                                cptr<Scal> q_i_j = &q_i[nrhs * j];
                                
                                q_p_j[k] -= q_i_j[k] * h[k];
                            }
                        },
                        job_ptr
                    );
                }
            }
            ptoc(ClassName()+" Gram-Schmidt");
            
            // Residual norms
            ComputeNorms( q_p, q_norms );
            
            // H[iter+1][iter] += ||q||;
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++ k )
            {
                H(iter+1,iter,k) = q_norms[k];
            }
            
            // q /= ||q||;
            InverseScale( q_p, q_norms );
            
            ptoc(ClassName()+"::ArnoldiStep");
        }
        
        void ApplyGivensRotations()
        {
            ptic(ClassName()+"::ApplyGivensRotations");
            
            for( Int i = 0; i < iter; ++ i )
            {
                for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                {
                    const Scal xi  = H(i  ,iter,k);
                    const Scal eta = H(i+1,iter,k);
                    
                    const Scal cos = cs[i][k];
                    const Scal sin = sn[i][k];
                    
                    H(i  ,iter,k) =  cos * xi + sin * eta;
                    H(i+1,iter,k) = -Conj(sin) * xi + cos * eta;
                }
            }
            
            {
                for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                {
                    const Scal xi  = H(iter  ,iter,k);
                    const Scal eta = H(iter+1,iter,k);
                    
                    Scal cos;
                    Scal sin;
                    
                    const Real r = Sqrt( AbsSquared(xi) + AbsSquared(eta) );
                    
                    if( Abs(xi) <= Scalar::eps<Scal> * r )
                    {
                        cos = Scalar::Zero<Scal>;
                        sin = Scalar::One<Scal>;
                    }
                    else
                    {
                        const Real r_inv = Inv(r);
                        
                        cos = Abs(xi) * r_inv;
                        sin = (xi / Abs(xi)) * Conj(eta) * r_inv;
                    }
                    
                    cs[iter][k] = cos;
                    sn[iter][k] = sin;

                    H(iter  ,iter,k) = cos * xi + sin * eta;
                    H(iter+1,iter,k) = Scalar::Zero<Scal>;
                                                                      
                    beta[iter+1][k] = -Conj(sin) * beta[iter][k];
                    beta[iter  ][k] =  cos * beta[iter][k];
                }
            }
            
            ptoc(ClassName()+"::ApplyGivensRotations");
        }
        
        void ComputeNorms( cptr<Scal> v, mref<RealVector_T> norms )
        {
//            ptic(ClassName()+"::ComputeNorms");
            
            ParallelDo(
                [this,v]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    RealVector_T sums ( nrhs );
                    
                    sums.SetZero();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                        {
                            sums[k] += AbsSquared(v[(NRHS>VarSize ? NRHS : nrhs) * i + k]);
                        }
                    }
                    
                    sums.Write( &real_red_buf[thread][0] );
                },
                thread_count
            );
            
            real_red_buf.AddReduce( norms.data(), false );
            
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
            {
                norms[k] = Sqrt( norms[k] );
            }
            
//            ptoc(ClassName()+"::ComputeNorms");
        }
        
        void ComputeScalarProducts( cptr<Scal> v, cptr<Scal> w, mref<Vector_T> dots )
        {
//            ptic(ClassName()+"::ComputeScalarProducts");
            
            ParallelDo(
                [this,v,w]( const Int thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    Vector_T sums ( nrhs );
                    
                    sums.SetZero();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                        {
                            sums[k] += Conj(v[(NRHS>VarSize ? NRHS : nrhs) * i + k]) * w[(NRHS>VarSize ? NRHS : nrhs) * i + k];
                        }
                    }
                    
                    sums.Write( &red_buf[thread][0] );
                },
                thread_count
            );
            
            red_buf.AddReduce( dots.data(), false );
            
//            ptoc(ClassName()+"::ComputeScalarProducts");
        }
        
        void InverseScale( mptr<Scal> q, const RealVector_T & factors )
        {
//            ptic(ClassName()+"::InverseScale");
            
            ParallelDo(
                [this,q,&factors]( const Int thread )
                {
                    RealVector_T factors_inv ( nrhs );
                    
                    for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                    {
                        factors_inv[k] = Inv( factors[k] );
                    }
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
                        {
                            q[(NRHS>VarSize ? NRHS : nrhs) * i + k] *= factors_inv[k];
                        }
                    }
                },
                thread_count
            );
            
//            ptoc(ClassName()+"::InverseScale");
        }
        
    public:
        
        Int ThreadCount() const
        {
            return thread_count;
        }
        
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
            RealVector_T res ( nrhs );
            
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
            {
                res[k] = std::abs(beta[iter][k]);
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res ( nrhs );
            
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
            {
                res[k] = Abs(beta[iter][k]) / b_norms[k];
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succ = true;
            
            for( Int k = 0; k < (NRHS>VarSize ? NRHS : nrhs); ++k )
            {
                succ = succ && (Abs(beta[iter][k]) <= TOL[k]);
            }
            
            return succ;
        }
        
        const Tensor3<Scal,Int> & GetBasis() const
        {
            return Q;
        }
        
        const Tensor3<Scal,Int> & GetHessenbergMatrix() const
        {
            return H;
        }
        
        std::string Stats() const
        {
            std:: stringstream s;
            
            s
            << "\n==== " + ClassName() + " Stats ====" << "\n\n"
            << " n                  = " << n << "\n"
            << " nrhs               = " << nrhs << "\n"
            << " restarts           = " << restarts << "\n"
            << " max_restarts       = " << max_restarts << "\n"
            << " iter               = " << iter     << "\n"
            << " max_iter           = " << max_iter << "\n"
            << " relative_tolerance = " << relative_tolerance << "\n"
            << " use_initial_guessQ = " << use_initial_guessQ << "\n"
            << "\n==== " + ClassName() + " Stats ====\n" << std::endl;
            
            
            s << " beta             = " << ArrayToString( beta.data(), {iter,nrhs} ) << "\n";
            s << " TOL              = " << ToString(TOL) << "\n";
            s << " b_norms          = " << ToString(b_norms) << "\n";
            
            s << " relative residuals = " << ArrayToString( RelativeResiduals().data(), {nrhs} ) << "\n";
            
            
            return s.str();
        }
        
        std::string ClassName() const
        {
            return std::string(
                std::string("GMRES")
                    + "<" + (NRHS <= VarSize ? std::string("VarSize") : ToString(NRHS) )
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + (side==Side::Left ? "Left" : "Right")
                    +">(" + ToString(nrhs)
                    + ")"
            );
        }
    }; // class GMRES
        
        
    
        
} // namespace Tensors
