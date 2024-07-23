#pragma once

#include "Tensors.hpp"

namespace Tensors
{
    // EQ_COUNT = number of right hand sides.
    // If you know this and compile time, then enter it in the template.
    // If you don't know this at compile time, then use EQ_COUNT = VarSize (==0) and specify
    // the eq_count_ in the constructor.
    
    // Scal_ is the floating point type that is used internally.
    
    template<Size_T EQ_COUNT, typename Scal_, typename Int_,
        bool A_verboseQ = true, bool P_verboseQ = true
    >
    class ConjugateGradient
    {
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int EQ = int_cast<Int>(EQ_COUNT);
        
        using RealVector_T = Tensor1<Real,Int>;

    protected:
        
        const Int n;
        const Int max_iter;
        const Int eq;
        const Int thread_count;
        
        Tensor2<Scal,Int> r;
        Tensor2<Scal,Int> u;
        Tensor2<Scal,Int> p;
        Tensor2<Scal,Int> x;
        Tensor2<Scal,Int> z;
        
        ThreadTensor2<Real,Int> reduction_buffer;
        
        RealVector_T TOL;
        RealVector_T b_squared_norms;
        
        RealVector_T alpha;
        RealVector_T beta;
        RealVector_T rho;
        RealVector_T rho_old;
        
        JobPointers<Int> job_ptr;
        
        Int iter = 0;
        
    public:
        
        ConjugateGradient() = delete;
        
        ConjugateGradient(
            const Int n_,
            const Int max_iter_,
            const Size_T eq_count_ = EQ,
            const Size_T thread_count_ = 1
        )
        :   n               ( n_                                    )
        ,   max_iter        ( Min(max_iter_,n)                      )
        ,   eq              ( ( EQ > VarSize ? EQ : static_cast<Int>(eq_count_ ) )  )
        ,   thread_count    ( static_cast<Int>(thread_count_)       )
        ,   r               ( n, eq                                 )
        ,   u               ( n, eq                                 )
        ,   p               ( n, eq                                 )
        ,   x               ( n, eq                                 )
        ,   z               ( n, eq                                 )
        ,   reduction_buffer( thread_count, eq                      )
        ,   TOL             ( eq                                    )
        ,   b_squared_norms ( eq                                    )
        ,   alpha           ( eq                                    )
        ,   beta            ( eq                                    )
        ,   rho             ( eq                                    )
        ,   rho_old         ( eq                                    )
        ,   job_ptr         ( n, thread_count                       )
        {}
        
        
        ~ConjugateGradient() = default;
        
        template<
            typename Operator_T, typename Preconditioner_T,
            typename b_T,        typename x_T
        >
        bool operator()(
            mref<Operator_T>       A,
            mref<Preconditioner_T> P,
            cptr<b_T> b_in,    const Int ldb,
            mptr<x_T> x_inout, const Int ldx,
            const Real relative_tolerance
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
            
            std::string tag = ClassName() + "::operator<" + TypeName<b_T> + "," + TypeName<x_T> + ">()";
            
            ptic(tag);
            
            iter = 0;
            bool succeeded = false;
            
            // r = b
            r.Read( b_in, ldx, thread_count );
            
            ptic(ClassName()+": Compute norm of right hand side.");
            
            // z = P.b
            ApplyPreconditioner(P,r,z);
            
            // rho = r.z
            ComputeScalarProducts( r, z, rho );
            
            Real factor = relative_tolerance * relative_tolerance;
            for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
            {
                b_squared_norms[k] = std::abs(rho[k]);
                TOL[k] = b_squared_norms[k] * factor;
            }
            
            ptoc(ClassName()+": Compute norm of right hand side.");
            
            if( TOL.Max() <= Scalar::Zero<Scal> )
            {
                ParallelDo(
                    [x_inout,ldx,this]( const Int i )
                    {
                        zerofy_buffer<EQ>( &x_inout[ldx * i], eq );
                    },
                    n, thread_count
                );
                
                wprint(tag + ": Right-hand side is 0. Returning zero vector.");
                
                logvalprint( tag + " iter"      , iter      );
                logvalprint( tag + " succeeded" , succeeded );
                
                ptoc(tag);
                
                return succeeded;
            }
            
            x.Read( x_inout, ldx, thread_count );


            
            // TODO: The Frobenius norm is good for detecting NaNs, but it could produce an overflow...
            
            const Real x_norm = x.FrobeniusNorm( thread_count);
            
            if( NaNQ(x_norm) )
            {
                wprint( tag + ": NaN detected in x_inout. Treating input as zero (skipping first operator multiplication).");
            }
            
            if( x_norm <= Scalar::Zero<Real>)
            {
                logprint( tag + ": Input x_inout is zero.");
            }
            
            // r = b - A.x
            if( (!NaNQ(x_norm)) && (x_norm > Scalar::Zero<Real>) )
            {
                logprint( tag + ": Input x_inout is nonzero. Using it as initial guess." );
                
                // u = A.x
                ApplyOperator(A,x,u);
    
                // r = b - A.x
                ParallelDo(
                    [this]( const Int i )
                    {
                        mptr<Scal> r_i = r.data(i);
                        cptr<Scal> u_i = u.data(i);
                        
                        for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                        {
//                            r[i][k] -= u[i][k];
                            r_i[k] -= u_i[k];
                        }
                    },
                    n, thread_count
                );
            }
            
            // z = P.r
            ApplyPreconditioner(P,r,z);
            
            p.Read( z.data(), eq, thread_count );
            
            // rho = r.z
            ComputeScalarProducts( r, z, rho );
            
            succeeded = CheckResiduals();

            while( !succeeded && (iter < max_iter ) )
            {
                // u = A.p
                ApplyOperator(A,p,u);
                
                // alpha = rho / (p.u);
                ComputeScalarProducts( p, u, alpha );
                for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                {
                    alpha[k] = rho[k] / alpha[k];
                }
                
                // x = x + alpha p;
                // r = r - alpha u;
                
                ParallelDo(
                    [this]( const Int i )
                    {
                        mptr<Scal> x_i = x.data(i);
                        mptr<Scal> r_i = r.data(i);
                        
                        cptr<Scal> p_i = p.data(i);
                        cptr<Scal> u_i = u.data(i);
                        
                        for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                        {
//                            x[i][k] += alpha[k] * p[i][k];
//                            r[i][k] -= alpha[k] * u[i][k];
                            
                            x_i[k] += alpha[k] * p_i[k];
                            r_i[k] -= alpha[k] * u_i[k];
                        }
                    },
                    n, thread_count
                );
                
                // z = P.r;
                ApplyPreconditioner(P,r,z);
                
                // rho_old = rho
                swap( rho_old, rho );
                
                // rho = r.z;
                ComputeScalarProducts( r, z, rho );
                
                // beta = rho / rho_old;
                for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                {
                    beta[k] = rho[k] / rho_old[k];
                }
                
                // TODO: Put this at the start of the while loop, and only for iter > 0?
                // p = z + beta p;
                ParallelDo(
                    [this]( const Int i )
                    {
                        mptr<Scal> p_i = p.data(i);
                        cptr<Scal> z_i = z.data(i);
                        
                        for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                        {
//                            p[i][k] = z[i][k] + beta[k] * p[i][k];
                            
                            p_i[k] = z_i[k] + beta[k] * p_i[k];
                        }
                    },
                    n, thread_count
                );
                
                succeeded = CheckResiduals();
                ++iter;
            }
            
            x.Write( x_inout, ldx, thread_count );
            
            logvalprint( tag + " iter"      , iter      );
            logvalprint( tag + " succeeded" , succeeded );
            
            ptoc(tag);
            
            return succeeded;
        }
        
        
    protected:
        
        template<typename Operator_T>
        void ApplyOperator(
            mref<Operator_T> A, cref<Tensor2<Scal,Int>> X, mref<Tensor2<Scal,Int>> Y
        )
        {
            if constexpr ( A_verboseQ )
            {
                ptic(ClassName()+ "::ApplyOperator");
            }
            
            (void)A( X.data(), Y.data() );
            
            if constexpr ( A_verboseQ )
            {
                ptoc(ClassName()+ "::ApplyOperator");
            }
        }
        
        template<typename Preconditioner_T>
        void ApplyPreconditioner(
            mref<Preconditioner_T> P, cref<Tensor2<Scal,Int>> X, mref<Tensor2<Scal,Int>> Y
        )
        {
            if constexpr ( P_verboseQ )
            {
                ptic(ClassName()+ "::ApplyPreconditioner");
            }
            
            (void)P( X.data(), Y.data() );
            
            
            if constexpr ( P_verboseQ )
            {
                ptoc(ClassName()+ "::ApplyPreconditioner");
            }
        }
        
        void ComputeScalarProducts(
            mref<Tensor2<Scal,Int>> v,
            mref<Tensor2<Scal,Int>> w,
            mref<RealVector_T> dots
        )
        {
            ParallelDo(
                [this,&v,&w,&dots]( const Int thread )
                {
                    auto & sums = reduction_buffer[thread];
                    
                    sums.SetZero();
                    
                    const Int i_begin = job_ptr[thread    ];
                    const Int i_end   = job_ptr[thread + 1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        cptr<Scal> v_i = v.data(i);
                        cptr<Scal> w_i = w.data(i);
                        
                        for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
                        {
                            // We know that all scalar products that we compute have to be real-valued.
//                            sums[k] += Re(Conj(v[i][k]) * w[i][k]);
                            
                            if constexpr ( Scalar::ComplexQ<Scal> )
                            {
                                sums[k] += Re(v_i[k]) * Re(w_i[k]) + Im(v_i[k]) * Im(w_i[k]) ;
                            }
                            else
                            {
                                sums[k] += v_i[k] * w_i[k];
                            }
                        }
                    }
                },
                thread_count
            );
            
            reduction_buffer.AddReduce( dots.data(), false );
        }
        
    public:
        
        Int IterationCount() const
        {
            return iter;
        }
        
        RealVector_T Residuals() const
        {
            RealVector_T res (eq);
            
            for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) );
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res(eq);
            
            for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) / b_squared_norms[k] );
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succeeded = true;
            for( Int k = 0; k < (EQ>VarSize ? EQ : eq); ++k )
            {
                succeeded = succeeded && ( Abs(rho[k]) <= TOL[k]);
            }
            
            return succeeded;
        }
        
        std::string ClassName() const
        {
            return std::string( "ConjugateGradient")
                + "<" + ToString(EQ)
                + "," + TypeName<Scal>
                + "," + TypeName<Int>
                + "," + ToString(A_verboseQ)
                + "," + ToString(P_verboseQ)
                + "> (" + ToString(eq) + ")";
        }
        
    }; // class ConjugateGradient
        
        
    
        
} // namespace Tensors


