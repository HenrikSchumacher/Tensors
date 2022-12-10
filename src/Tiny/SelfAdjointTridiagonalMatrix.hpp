#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
#define CLASS SelfAdjointTridiagonalMatrix
        
        template< int n_, typename Scalar_, typename Int_ >
        class CLASS
        {
            // Uses only upper triangle.
            
        public:
            
#include "Tiny_Details.hpp"
            
            static constexpr Int n = n_;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
        protected:
            
            std::array<Real,n>   diag;  //the main diagonal (should actually only have real values on it.
            std::array<Scalar,n> upper; //upper diagonal
            
        
        public:

//######################################################
//##                     Memory                       ##
//######################################################

            explicit CLASS( const Scalar init )
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
            
            force_inline Scalar & Upper( const Int i )
            {
                return upper[i];
            }
            
            force_inline const Scalar & Upper( const Int i ) const
            {
                return upper[i];
            }
            
            force_inline Scalar Lower( const Int i )
            {
                return conj(upper[i]);
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
            
            std::string ToString( const int p = 16) const
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\tdiag  = { ";
                
                sout << Tools::ToString(diag[0],p);
                for( Int j = 1; j < n; ++j )
                {
                    sout << ", " << Tools::ToString(diag[j],p);
                }
                
                sout << " },\n\tupper = { ";
                
                if( n > 1 )
                {
                    sout << Tools::ToString(upper[0],p);
                    
                    for( Int j = 1; j < n-1; ++j )
                    {
                        sout << ", " << Tools::ToString(upper[j],p);
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
            
            template<typename T = Scalar>
            force_inline void ToMatrix( SquareMatrix<n,T,Int> & B ) const
            {
                B.SetZero();
                
                for( Int i = 0; i < n-1; ++i )
                {
                    B[i][i]     = static_cast<T>(diag[i]);
                    B[i  ][i+1] = static_cast<T>(upper[i]);
                    B[i+1][i  ] = static_cast<T>(conj(upper[i]));
                }
                B[n-1][n-1] = static_cast<T>(diag[n-1]);
            }
        
        public:
            
            template<typename S>
            force_inline std::enable_if_t<
                std::is_same_v<S, Scalar> && !ScalarTraits<Scalar>::IsComplex,
                void
            >
            QRAlgorithm( SquareMatrix<n,S,Int> & Q, Vector<n,Real,Int> & eigs )
            {
                // Computes a unitary matrix Q and and a vector eigs such that
                // Q . diag(eigs) . Q^T = A.
                
                Q.SetIdentity();
                
                if constexpr ( n == 1 )
                {
                    qr_algorithm<1>(Q);
                    
                    eigs[0] = diag[0];
                }
                
                if constexpr ( n == 2 )
                {
                    qr_algorithm<0,2>(Q);
                    
                    eigs[0] = diag[0];
                    eigs[1] = diag[1];
                }
                
                if constexpr ( n >= 3 )
                {
                    // In order to leave the current matrix intact, we do a copy and process that one in-place.
                    SelfAdjointTridiagonalMatrix<n,Real,Int> T;
                    
                    for( Int i = 0; i < n; ++i )
                    {
                        T.Diag(i) = real( Diag(i)  );
                    }
                    
                    for( Int i = 0; i < n-1; ++i )
                    {
                        T.Upper(i) = real( Upper(i)  );
                    }
                    
                    T.template qr_algorithm<0,n>(Q);
                    
                    eigs.Read( &T.Diag(0) );
                }
            }
            
        public:
            
            //Checked
            force_inline void givens( const Real a_0, const Real a_1, const Real b_0, Real & c, Real & s )
            {
                // Find c, s such that
                //
                //    /         \  /            \  /         \
                //    |  c  -s  |  |  a_0  b_0  |  |   c  s  |
                //    |  s   c  |  |  b_0  a_1  |  |  -s  c  |
                //    \         /  \            /  \         /
                //
                //  is diagonal;
                
                if( std::abs(b_0) < eps * ( std::abs(a_0)+std::abs(a_1) ) )
                {
                    c = one;
                    s = zero;
                }
                else
                {
                    const Real d = a_0 - a_1;
                    
                    const Real d_squared = d * d;
                    
                    const Real delta_squared = d_squared + four * b_0 * b_0;
                    
                    const Real delta = std::sqrt(delta_squared);
                    
                    const Real s_ = (d + delta) / (two * b_0);
                    
                    const Real factor = one / std::sqrt(one + s_ * s_);
                    
                    c = factor;
                    
                    s = s_ * factor;
                }
                
            }
            
            //Checked
            force_inline void givens( const Real x, const Real y, Real & c, Real & s )
            {
                // Find c, s such that
                //
                //    /         \  /     \  /       \
                //    |   c  s  |  |  x  |  |  rho  |
                //    |  -s  c  |  |  y  |  |   0   |
                //    \         /  \     /  \       /
                //
                //  is diagonal;
                
                const Real rho_inv = one / std::sqrt(x * x + y * y);
                
                c = x * rho_inv;
                s = y * rho_inv;
            }

            
//            template<Int begin, Int end>
//            force_inline std::enable_if_t< !ScalarTraits<Scalar>::IsComplex, void >
//            qr_algorithm( SquareMatrix<n,Real,Int> & Q )
//            {
//                // Performs the implicit QR algorithm in the block A[begin..end-1][begin..end-1].
//                dump(begin);
//                dump(end);
//                dump(*this);
//                
//                constexpr Real tol = 128 * eps;
//                
//                if constexpr ( end == begin+1 )
//                {
//                    return;
//                }
//                
//                // Compute Wilkinson’s shift
//                const Real a = diag[end-1];
//                const Real b = upper[end-2];
//                
//                Real d_ = half * (diag[end-2] - a);
//                
//                Real shift = ( std::abs(d_) <= eps )
//                    ?
//                    a - std::abs(b)
//                    :
//                    a - b * b /( d_ + (d_<zero ? -one : one ) * std::sqrt( d_ * d_ + b * b ) );
//
//                // Implicit QR step begins here
//                
//                Real x = diag[begin] - shift;
//                Real y = upper[begin];
//                
//                dump(x);
//                dump(y);
//                
//                Real c = 0;
//                Real s = 0;
//                
//                for( Int k = begin; k < end-1; ++k )
//                {
//                    dump(k);
//                    
//                    if constexpr ( end > begin + 2 )    // At least 3-dim
//                    {
//                        givens( x, y, c, s );
//                    }
//                    else // 2-dimensional case.
//                    {
//                        givens( diag[begin], diag[begin+1], upper[begin], c, s );
//                    }
//                    
//                    dump(c);
//                    dump(s);
//                    
//                    Real w = c * x - s * y;
//                    Real d = diag[k] - diag[k+1];
//                    Real z = ( two * c * upper[k] + d * s ) * s;
//                    
//                    diag[k  ] -= z;
//                    diag[k+1] += z;
//                    upper[k] = d * c * s + (c * c - s * s) * upper[k];
//                    
//                    x = upper[k];
//                    
//                    dump(upper[k]);
//                    
//                    if( k > 0 )
//                    {
//                        upper[k-1] = w;
//                        dump(upper[k-1]);
//                    }
//                    if( k < end-2 )
//                    {
//                        y = - s * upper[k+1];
//                        upper[k+1] *= c;
//                        dump(upper[k+1]);
//                    }
//                    
//                    // Multiply Q by Givens(c,-s,k,k+1) from the right.
//                    Q.GivensRight(c,-s,k,k+1);
//                }
//                
//                // Implicit QR step ends here
//                
//                dump(*this);
//                
//                // Check for possible deflation and do the next iteration.
//                if( std::abs(upper[end-2]) < tol * (std::abs(diag[end-2])+std::abs(diag[end-1])) )
//                {
//                    if constexpr ( end > begin +1 )
//                    {
//                        qr_algorithm<begin,end-1>( Q );
//                    }
//                }
//                else
//                {
//                    qr_algorithm<begin,end>( Q );
//                }
//            }
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
#undef CLASS
        
    } // namespace Tiny
    
} // namespace Tensors
