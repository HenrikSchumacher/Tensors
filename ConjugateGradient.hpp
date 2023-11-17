#pragma once

#include "Tensors.hpp"
#include "src/BLAS.hpp"

namespace Tensors
{
    // eq_count_ = number of right hand sides.
    // If you know this and compile time, then enter it in the template.
    // If you don't know this at compile time, then use eq_count_ = VarSize (==0) and specify
    // the eq_count__ in the constructor.
    template<Size_T eq_count_, typename Scal_, typename Int_>
    class ConjugateGradient
    {
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int EQ = int_cast<Int>(eq_count_);
        
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
            const Size_T eq_count__ = EQ,
            const Size_T thread_count_ = 1
        )
        :   n               ( n_                                    )
        ,   max_iter        ( Min(max_iter_,n)                      )
        ,   eq              ( COND( EQ > VarSize, EQ, eq_count__ )  )
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
        
        template<typename Operator_T, typename Preconditioner_T>
        bool operator()(
            mref<Operator_T>       A,
            mref<Preconditioner_T> P,
            cptr<Scal> b_in,    const Int ldb,
            mptr<Scal> x_inout, const Int ldx,
            const Real relative_tolerance
        )
        {
            // r = b
            r.Read( b_in, ldx, thread_count );
            
            // z = P.b
            ptic(ClassName()+ ": Apply preconditioner");
            P( r.data(), z.data() );
            ptoc(ClassName()+ ": Apply preconditioner");
            
            // rho = r.z
            ComputeScalarProducts( r.data(), z.data(), rho );
            
            Real factor = relative_tolerance * relative_tolerance;
            for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
            {
                b_squared_norms[k] = std::abs(rho[k]);
                TOL[k] = b_squared_norms[k] * factor;
            }
            
            if( TOL.Max() <= Scalar::Zero<Scal> )
            {
                r.Write( x_inout, ldx, thread_count );
               
                return true;
            }
            
            x.Read( x_inout, ldx, thread_count );

            // u = A.x
            ptic(ClassName()+ ": Apply operator");
            A( x.data(), u.data() );
            ptoc(ClassName()+ ": Apply operator");
            // r = b - A.x
            
            ParallelDo(
                [this]( const Int i )
                {
                    for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                    {
                        r[i][k] -= u[i][k];
                    }
                },
                n, thread_count
            );
            
            // z = P.r
            ptic(ClassName()+ ": Apply preconditioner");
            P( r.data(), z.data() );
            ptoc(ClassName()+ ": Apply preconditioner");
            
            p.Read( z.data(), eq, thread_count );
            
            // rho = r.z
            ComputeScalarProducts( r.data(), z.data(), rho );
            
            iter = 0;
            bool succeeded = CheckResiduals();

            while( !succeeded && (iter < max_iter ) )
            {
                // u = A.p
                ptic(ClassName()+ ": Apply operator");
                A( p.data(), u.data() );
                ptoc(ClassName()+ ": Apply operator");
                
                // alpha = rho / (p.u);
                ComputeScalarProducts( p.data(), u.data(), alpha );
                for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                {
                    alpha[k] = rho[k] / alpha[k];
                }
                
                // x = x + alpha p;
                // r = r - alpha u;
                
                ParallelDo(
                    [this]( const Int i )
                    {
                        for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                        {
                            x[i][k] += alpha[k] * p[i][k];
                            r[i][k] -= alpha[k] * u[i][k];
                        }
                    },
                    n, thread_count
                );
                
                // z = P.r;
                ptic(ClassName()+ ": Apply preconditioner");
                P( r.data(), z.data() );
                ptoc(ClassName()+ ": Apply preconditioner");
                
                // rho_old = rho
                swap( rho_old, rho );
                
                // rho = r.z;
                ComputeScalarProducts( r.data(), z.data(), rho );
                
                // beta = rho / rho_old;
                for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                {
                    beta[k] = rho[k] / rho_old[k];
                }
                
                // TODO: Put this at the start of the while loop, and only for iter > 0?
                // p = z + beta p;
                ParallelDo(
                    [this]( const Int i )
                    {
                        for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                        {
                            p[i][k] = z[i][k] + beta[k] * p[i][k];
                        }
                    },
                    n, thread_count
                );
                
                succeeded = CheckResiduals();
                ++iter;
            }
            
            x.Write(x_inout);
            
            return succeeded;
        }
        
        
    protected:
        
        void ComputeScalarProducts( cptr<Scal> v, cptr<Scal> w, mref<RealVector_T> dots )
        {
            ParallelDo(
                [this,v,w]( const Int thread )
                {
                    RealVector_T sums ( eq );
                    
                    sums.SetZero();
                    
                    const Int i_begin = job_ptr[thread    ];
                    const Int i_end   = job_ptr[thread + 1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
                        {
                            // We know that all scalar products that we compute have to be real-valued.
                            sums[k] += Re(Conj(v[COND(EQ>VarSize,EQ,eq) * i + k]) * w[COND(EQ>VarSize,EQ,eq) * i + k]);
                        }
                    }
                    
                    sums.Write( reduction_buffer.data(thread) );
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
            
            for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) );
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res(eq);
            
            for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
            {
                res[k] = Sqrt( Abs(rho[k]) / b_squared_norms[k] );
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succeeded = true;
            for( Int k = 0; k < COND(EQ>VarSize,EQ,eq); ++k )
            {
                succeeded = succeeded && ( Abs(rho[k]) <= TOL[k]);
            }
            
            return succeeded;
        }
        
        std::string ClassName() const
        {
            return std::string(
                "ConjugateGradient<"+ToString(EQ)+","+TypeName<Scal>+","+TypeName<Int>+"> ( " + ToString(eq) + ")"
            );
        }
        
    }; // class ConjugateGradient
        
        
    
        
} // namespace Tensors

