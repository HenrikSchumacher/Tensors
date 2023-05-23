#pragma once

#include "Tensors.hpp"

namespace Tensors
{
    template<size_t eq_count_, typename Scal_, typename Int_>
    class ConjugateGradient
    {
        
    public:
        
        using Scal     = Scal_;
        using Real     = Scalar::Real<Scal>;
        using Int      = Int_;
        
        static constexpr Int K = int_cast<Int>(eq_count_);
        
        using RealVector_T = Tiny::Vector<K,Real,Int>;

    protected:
        
        const Int n;
        const Int max_iter;
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
        
        Int iter = 0;
        
    public:
        
        ConjugateGradient() = delete;
        
        ConjugateGradient( const Int n_, const Int max_iter_, const Int thread_count_ )
        :   n               ( n_ )
        ,   max_iter        ( std::min(max_iter_,n) )
        ,   thread_count    ( thread_count_ )
        ,   r               ( n, K )
        ,   u               ( n, K )
        ,   p               ( n, K )
        ,   x               ( n, K )
        ,   z               ( n, K )
        ,   reduction_buffer( thread_count, K )
        {}
        
        ~ConjugateGradient() = default;
        
        template<typename Operator_T, typename Preconditioner_T>
        bool operator()(
            Operator_T       & A,
            Preconditioner_T & P,
            ptr<Scal> b_in,       const Int ldb,
            mut<Scal> x_inout,    const Int ldx,
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
            for( Int k = 0; k < K; ++k )
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
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                const Int i_begin = JobPointer( n, thread_count, thread   );
                const Int i_end   = JobPointer( n, thread_count, thread+1 );
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    for( Int k = 0; k < K; ++k )
                    {
                        r[i][k] -= u[i][k];
                    }
                }
            }
            
            // z = P.r
            ptic(ClassName()+ ": Apply preconditioner");
            P( r.data(), z.data() );
            ptoc(ClassName()+ ": Apply preconditioner");
            
            p.Read( z.data(), K, thread_count );
            
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
                for( Int k = 0; k < K; ++k )
                {
                    alpha[k] = rho[k] / alpha[k];
                }
                
                // x = x + alpha p;
                // r = r - alpha u;
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = JobPointer( n, thread_count, thread   );
                    const Int i_end   = JobPointer( n, thread_count, thread+1 );
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            x[i][k] += alpha[k] * p[i][k];
                            r[i][k] -= alpha[k] * u[i][k];
                        }
                    }
                }
                
                // z = P.r;
                ptic(ClassName()+ ": Apply preconditioner");
                P( r.data(), z.data() );
                ptoc(ClassName()+ ": Apply preconditioner");
                
                // rho_old = rho
                swap( rho_old, rho );
                
                // rho = r.z;
                ComputeScalarProducts( r.data(), z.data(), rho );
                
                // beta = rho / rho_old;
                for( Int k = 0; k < K; ++k )
                {
                    beta[k] = rho[k] / rho_old[k];
                }
                
                // TODO: Put this at the start of the while loop, and only for iter > 0?
                // p = z + beta p;
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = JobPointer( n, thread_count, thread   );
                    const Int i_end   = JobPointer( n, thread_count, thread+1 );
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        for( Int k = 0; k < K; ++k )
                        {
                            p[i][k] = z[i][k] + beta[k] * p[i][k];
                        }
                    }
                }
                
                succeeded = CheckResiduals();
                ++iter;
            }
            
            x.Write(x_inout);
            
            return succeeded;
        }
        
        
    protected:
        
        void ComputeScalarProducts( ptr<Scal> v, ptr<Scal> w, RealVector_T & dots )
        {
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                const Int i_begin = JobPointer( n, thread_count, thread   );
                const Int i_end   = JobPointer( n, thread_count, thread+1 );
                
                RealVector_T sums;
                
                sums.SetZero();
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    for( Int k = 0; k < K; ++k )
                    {
                        // We know that all scalar products that we compute have to be real-valued.
                        sums[k] += Scalar::Re(Scalar::Conj(v[K * i + k]) * w[K * i + k]);
                    }
                }
                
                sums.Write( reduction_buffer.data(thread) );
            }
            
            reduction_buffer.AddReduce( dots.data(), false );
        }
        
    public:
        
        Int IterationCount() const
        {
            return iter;
        }
        
        RealVector_T Residuals() const
        {
            RealVector_T res;
            
            for( Int k = 0; k < K; ++k )
            {
                res[k] = std::sqrt( std::abs(rho[k]) );
            }
            
            return res;
        }
        
        RealVector_T RelativeResiduals() const
        {
            RealVector_T res;
            
            for( Int k = 0; k < K; ++k )
            {
                res[k] = std::sqrt( std::abs(rho[k]) / b_squared_norms[k] );
            }
            return res;
        }
        
        bool CheckResiduals() const
        {
            bool succeeded = true;
            for( Int k = 0; k < K; ++k )
            {
                succeeded = succeeded && ( std::abs(rho[k]) <= TOL[k]);
            }
            
            return succeeded;
        }
        
        std::string ClassName() const
        {
            return std::string(
                "ConjugateGradient<"+ToString(K)+","+TypeName<Scal>+","+TypeName<Int>+">"
            );
        }
        
    }; // class ConjugateGradient
        
        
    
        
} // namespace Tensors

