#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
#define CLASS SelfAdjointTridiagonalMatrix
        
        template< int n_, typename Scal_, typename Int_ >
        class CLASS
        {
            // Uses only upper triangle.
            
        public:
            
#include "Tiny_Details.hpp"
            
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scal,Int>;
            
        protected:
            
            std::array<Real,n>   diag;  //the main diagonal (should actually only have real values on it.
            std::array<Scal,n> upper; //upper diagonal
            
        
        public:

//######################################################
//##                     Memory                       ##
//######################################################

            explicit CLASS( const Scal init )
            :   diag  { init }
            ,   upper { init }
            {}
            
            
            force_inline void SetZero()
            {
                zerofy_buffer<n  >( &diag[0]  );
                zerofy_buffer<n-1>( &upper[0] );
            }
            
//######################################################
//##                     Access                       ##
//######################################################

            
            force_inline Real & Diag( const Int i )
            {
                return diag[i];
            }
            
            force_inline const Real & Diag( const Int i ) const
            {
                return diag[i];
            }
            
            force_inline Scal & Upper( const Int i )
            {
                return upper[i];
            }
            
            force_inline const Scal & Upper( const Int i ) const
            {
                return upper[i];
            }
            
            force_inline Scal Lower( const Int i )
            {
                return Scalar::Conj(upper[i]);
            }

            
            force_inline friend CLASS operator+( const CLASS & x, const CLASS & y )
            {
                CLASS z;
                for( Int i = 0; i < n; ++i )
                {
                    z.diag[i] = x.diag[i] + y.diag[i];
                }
                for( Int i = 0; i < n-1; ++i )
                {
                    z.upper[i] = x.upper[i] + y.upper[i];
                }
                
                return z;
            }
            
            force_inline void operator+=( const CLASS & B )
            {
                add_to_buffer<n>  ( &B.diag[0],  &diag[0]       );
                add_to_buffer<n-1>( &B.upper[0], &diag.upper[0] );
            }
            
            force_inline void Dot( const Vector_T & x, Vector_T & y ) const
            {
                if constexpr ( n >= 1 )
                {
                    y[0] = diag[0] * x[0];
                }
                else if constexpr ( n > 1 )
                {
                    y[0] = diag[0] * x[0] + upper[0] * x[1];
                }
                
                for( Int i = 1; i < n-2; ++ i )
                {
                    y[i] = upper[i-1] * x[i-1] + diag[i] * x[i] + upper[i] * x[i+1];
                }
                
                if constexpr ( n >= 2 )
                {
                    y[n-1] = diag[n-1] * x[n-1];
                }
                else if constexpr  ( n > 2 )
                {
                    y[n-1] = upper[n-2] * x[n-2] + diag[n-1] * x[n-1];
                }
            }
            
            std::string ToString() const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\tdiag  = { ";
                
                sout << Tools::ToString(diag[0]);
                for( Int j = 1; j < n; ++j )
                {
                    sout << ", " << Tools::ToString(diag[j]);
                }
                
                sout << " },\n\tupper = { ";
                
                if( n > 1 )
                {
                    sout << Tools::ToString(upper[0]);
                    
                    for( Int j = 1; j < n-1; ++j )
                    {
                        sout << ", " << Tools::ToString(upper[j]);
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            template<typename T = Scal>
            void ToMatrix( Matrix<n,n,T,Int> & B ) const
            {
                if( n <= 0 )
                {
                    return;
                }
                
                B.SetZero();
                
                for( Int i = 0; i < n-1; ++i )
                {
                    B[i][i]     = static_cast<T>(diag[i]);
                    B[i  ][i+1] = static_cast<T>(upper[i]);
                    B[i+1][i  ] = static_cast<T>(Scalar::Conj(upper[i]));
                }
                B[n-1][n-1] = static_cast<T>(diag[n-1]);
            }
        
        public:
            
            template<typename S>
            force_inline std::enable_if_t<
                std::is_same_v<S, Scal> && !Scalar::ComplexQ<Scal>,
                void
            >
            QRAlgorithm(
                Matrix<n,n,S,Int> & Q,
                Vector<n,  S,Int> & eigs,
                const Real tol = eps_sqrt,
                const Int max_iter = 4
            )
            {
                // Computes a unitary matrix Q and and a vector eigs such that
                // Q . A . Q^T = diag(eigs).
                // Caution: Performs computations in-place, so it modifies A.
                
                Q.SetIdentity();
                
                if constexpr ( n == 1 )
                {
                    eigs[0] = diag[0];
                }
                
                if constexpr ( n == 2 )
                {
                    qr_algorithm_2x2(Q);
                    
                    eigs[0] = diag[0];
                    eigs[1] = diag[1];
                }
                
                if constexpr ( n >= 3 )
                {
                    qr_algorithm(Q,n,tol,0,max_iter);
                    
                    eigs.Read( &diag[0] );
                }
            }
            
            template<typename S>
            force_inline std::enable_if_t<
                std::is_same_v<S, Scal> && !Scalar::ComplexQ<Scal>,
                void
            >
            QRAlgorithm(
                Vector<n,S,Int> & eigs,
                const Real tol = std::sqrt(eps),
                const Int max_iter = 4
            )
            {
                // Computes the eigenvalues only.
                // Caution: Performs computations in-place, so it modifies A.
                
                if constexpr ( n == 1 )
                {
                    eigs[0] = diag[0];
                }
                
                if constexpr ( n == 2 )
                {
                    qr_algorithm_2x2();
                    
                    eigs[0] = diag[0];
                    eigs[1] = diag[1];
                }
                
                if constexpr ( n >= 3 )
                {
                    qr_algorithm(n,tol,0,max_iter);
                    
                    eigs.Read( &diag[0] );
                }
            }
            
        public:
            

            force_inline std::enable_if_t< !Scalar::ComplexQ<Scal>, void >
            qr_algorithm_2x2( Matrix<n,n,Real,Int> & Q )
            {
                const Real a_0 = diag[0];
                const Real a_1 = diag[1];
                const Real b_0 = upper[0];
                
                // Find c, s such that
                //
                //    /         \  /            \  /         \
                //    |  c  -s  |  |  a_0  b_0  |  |   c  s  |
                //    |  s   c  |  |  b_0  a_1  |  |  -s  c  |
                //    \         /  \            /  \         /
                //
                //  is diagonal.
                if( std::abs(b_0) > eps * ( std::abs(a_0)+std::abs(a_1) ) )
                {
                    const Real d = a_0 - a_1;
                    const Real delta = std::sqrt( d * d + four * b_0 * b_0 );
                    
                    const Real s_ = (d + delta) / (two * b_0);
                    const Real factor = one / std::sqrt(one + s_ * s_);
                    
                    const Real c = factor;
                    const Real s = s_ * factor;
                    
                    diag[0]  = half * (a_0 + a_1 - delta);
                    diag[1]  = half * (a_0 + a_1 + delta);
                    upper[0] = 0;
                    
                    // Multiply Q by Givens(c,s,0,1) from the right.
                    Q.GivensRight(c,s,0,1);
                }
            }
            
            force_inline std::enable_if_t< !Scalar::ComplexQ<Scal>, void >
            qr_algorithm_2x2()
            {
                const Real a_0 = diag[0];
                const Real a_1 = diag[1];
                const Real b_0 = upper[0];
                
                // Find c, s such that
                //
                //    /         \  /            \  /         \
                //    |  c  -s  |  |  a_0  b_0  |  |   c  s  |
                //    |  s   c  |  |  b_0  a_1  |  |  -s  c  |
                //    \         /  \            /  \         /
                //
                //  is diagonal.
                if( std::abs(b_0) > eps * ( std::abs(a_0)+std::abs(a_1) ) )
                {
                    const Real d = a_0 - a_1;
                    const Real delta = std::sqrt( d * d + four * b_0 * b_0 );
                    
                    diag[0]  = half * (a_0 + a_1 - delta);
                    diag[1]  = half * (a_0 + a_1 + delta);
                    upper[0] = 0;
                }
            }
//
            force_inline std::enable_if_t< !Scalar::ComplexQ<Scal>, void >
            qr_algorithm(
                Matrix<n,n,Real,Int> & Q,
                const Int m,
                const Real tol,
                const Int iter,
                const Int max_iter
            )
            {
                // Performs the implicit QR algorithm in the block A[begin..end-1][begin..end-1].
                // Assumes that m > 2 (otherwise we would have called qr_algorithm_2x2 or stopped.
                
                // Compute Wilkinson’s shift mu
                const Real a = diag[m-1];
                const Real b = upper[m-2];
                
                const Real d_ = half * (diag[m-2] - a);
                
                const Real mu = ( std::abs(d_) <= eps * ( std::abs(diag[m-2]) + std::abs(a) ) )
                    ?
                    a - std::abs(b)
                    :
                    a - b * b / ( d_ + (d_ >= zero ? one : -one ) * std::sqrt( d_ * d_ + b * b ) );
                
                // Implicit QR step begins here
                
                Real x = diag[0] - mu;
                Real y = upper[0];
                
                { // k = 0
                    
                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real d = diag[0] - diag[1];
                        const Real z = ( two * c * upper[0] + d * s ) * s;
                        diag[0] -= z;
                        diag[1] += z;
                        upper[0] = d * c * s + (c * c - s * s) * upper[0];
                        x = upper[0];
                        y = - s * upper[1];
                        upper[1] *= c;
                        
                        // Multiply Q by Givens(c,s,k,k+1) from the right.
                        Q.GivensRight(c,s,0,1);
                    }
                }
                
                for( Int k = 1; k < m-2; ++k )
                {
                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[k] - diag[k+1];
                        const Real z = ( two * c * upper[k] + d * s ) * s;
                        diag[k  ] -= z;
                        diag[k+1] += z;
                        upper[k] = d * c * s + (c * c - s * s) * upper[k];
                        x = upper[k];
                        //                    if( k > 0 )
                        //                    {
                        upper[k-1] = w;
                        //                    if( k < m-2 )
                        //                    {
                        y = - s * upper[k+1];
                        upper[k+1] *= c;
                        //                    }
                        
                        // Multiply Q by Givens(c,s,k,k+1) from the right.
                        Q.GivensRight(c,s,k,k+1);
                    }
                }
                
                {   // k = m-2
                    
                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[m-2] - diag[m-1];
                        const Real z = ( two * c * upper[m-2] + d * s ) * s;
                        diag[m-2] -= z;
                        diag[m-1] += z;
                        upper[m-2] = d * c * s + (c * c - s * s) * upper[m-2];
                        x = upper[m-2];
                        upper[m-3] = w;
                        
                        // Multiply Q by Givens(c,s,k,k+1) from the right.
                        Q.GivensRight(c,s,m-2,m-1);
                    }
                }
                // Implicit QR step ends here
                

                
                // Check for possible deflation and do the next iteration.
                if(
                   (iter < max_iter)
                   &&
                   (std::abs(upper[m-2]) > tol * (std::abs(diag[m-2])+std::abs(diag[m-1])))
                )
                {
                    qr_algorithm( Q, m, tol, iter+1, max_iter );
                }
                else
                {
                    if ( m == 3 )
                    {
                        qr_algorithm_2x2( Q );
                    }
                    else
                    {
                        qr_algorithm( Q, m-1, tol, 0, max_iter );
                    }
                }
            }
            
            
            force_inline std::enable_if_t< !Scalar::ComplexQ<Scal>, void >
            qr_algorithm(
                const Int m,
                const Real tol,
                const Int iter,
                const Int max_iter
            )
            {
                if( m <= 1 )
                {
                    return;
                }
                // Performs the implicit QR algorithm in the block A[begin..end-1][begin..end-1].
                // Assumes that m > 2 (otherwise we would have called qr_algorithm_2x2 or stopped.
                
                // Compute Wilkinson’s shift mu
                const Real a = diag[m-1];
                const Real b = upper[m-2];
                
                const Real d_ = half * (diag[m-2] - a);
                
                const Real mu = ( std::abs(d_) <= eps * ( std::abs(diag[m-2]) + std::abs(a) ) )
                    ?
                    a - std::abs(b)
                    :
                    a - b * b / ( d_ + (d_ >= zero ? one : -one ) * std::sqrt( d_ * d_ + b * b ) );
                
                // Implicit QR step begins here
                
                Real x = diag[0] - mu;
                Real y = upper[0];
                
                { // k = 0

                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real d = diag[0] - diag[1];
                        const Real z = ( two * c * upper[0] + d * s ) * s;
                        diag[0] -= z;
                        diag[1] += z;
                        upper[0] = d * c * s + (c * c - s * s) * upper[0];
                        x = upper[0];
                        y = - s * upper[1];
                        upper[1] *= c;
                    }
                }
                
                for( Int k = 1; k < m-2; ++k )
                {
                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[k] - diag[k+1];
                        const Real z = ( two * c * upper[k] + d * s ) * s;
                        diag[k  ] -= z;
                        diag[k+1] += z;
                        upper[k] = d * c * s + (c * c - s * s) * upper[k];
                        x = upper[k];
                        //                    if( k > 0 )
                        //                    {
                        upper[k-1] = w;
                        //                    if( k < m-2 )
                        //                    {
                        y = - s * upper[k+1];
                        upper[k+1] *= c;
                        //                    }
                    }
                }
                
                {   // k = m-2
                    
                    // Find c, s such that
                    //
                    //    /         \  /     \   /       \
                    //    |  c  -s  |  |  x  | = |  rho  |
                    //    |  s   c  |  |  y  |   |   0   |
                    //    \         /  \     /   \       /
                    //
                    
                    const Real rho = std::sqrt(x * x + y * y);
                    
                    if( rho > eps * diag[0] )
                    {
                        const Real rho_inv = one / rho;
                        const Real c =   x * rho_inv;
                        const Real s = - y * rho_inv;
                        
                        // Apply Givens rotation to tridiagonal matrix.
                        const Real w = c * x - s * y;
                        const Real d = diag[m-2] - diag[m-1];
                        const Real z = ( two * c * upper[m-2] + d * s ) * s;
                        diag[m-2] -= z;
                        diag[m-1] += z;
                        upper[m-2] = d * c * s + (c * c - s * s) * upper[m-2];
                        x = upper[m-2];
                        upper[m-3] = w;
                    }
                }
                // Implicit QR step ends here
                

                
                // Check for possible deflation and do the next iteration.
                
                if( std::abs(upper[m-2]) > tol * (std::abs(diag[m-2])+std::abs(diag[m-1])) )
                {
                    if( iter < max_iter )
                    {
                        qr_algorithm( m, tol, iter+1, max_iter );
                    }
                    else
                    {
                        // We give up to improve this eigenvalue. Go to the next one.
                        if ( m == 3 )
                        {
                            qr_algorithm_2x2();
                        }
                        else
                        {
                            qr_algorithm( m-1, tol, 0, max_iter );
                        }
                    }
                }
                else
                {
                    if ( m == 3 )
                    {
                        qr_algorithm_2x2();
                    }
                    else
                    {
                        qr_algorithm(m-1, tol, 0, max_iter );
                    }
                }
            }
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "Tiny::"+TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
            
        };
        
#undef CLASS
        
    } // namespace Tiny
    
} // namespace Tensors
