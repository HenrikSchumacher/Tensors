#pragma once

namespace Tensors
{
    namespace Tiny
    {
#define CLASS Vector
        
        template<int n_, typename Scal_, typename Int_> class VectorList;
        
        template< int n_, typename Scal_, typename Int_>
        class CLASS
        {
            // Very slim vector type of fixed length, with basic arithmetic operations.
            
#include "Tiny_Details.hpp"
            
            static constexpr Int n = n_;

            template<typename S>
            CLASS( const VectorList<n,S,Int> & v_list, const Int k )
            {
                Read(v_list, k);
            }
            
            template<typename S>
            CLASS( const S * buffer, const Int k )
            {
                Read( buffer[n * k], k );
            }
            
            template<typename S>
            CLASS( const std::initializer_list<S> & w )
            {
                const Int m = int_cast<Int>(w.size());
                
                if(m > n )
                {
                    eprint(TO_STD_STRING(CLASS)+": Length of initializer list must not exceed length of n");
                }
                else
                {
                    ptr<S> w_ = &(*w.begin());
                    
                    for( Int i = 0; i < m; ++i )
                    {
                        v[i] = scalar_cast<Scal>(w_[i]);
                    }
                    for( Int i = m; i < n; ++i )
                    {
                        v[i] = Scalar::Zero<Scal>;
                    }
                }
            }
            
            
            
        protected:
            
            std::array<Scal,n> v;
            
//######################################################
//##                     Memory                       ##
//######################################################
            
        public:
            
            explicit CLASS( const Scal init )
            :   v {{init}}
            {}
            
            // Copy constructor
            CLASS( const CLASS & other )
            {
                Read( &other.v[0] );
            }

            friend void swap( CLASS & A, CLASS & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap( A.v, B.v );
            }
            
        public:
            
            void SetZero()
            {
                zerofy_buffer<n>( &v[0] );
            }
            
            void Fill( const Scal init )
            {
                fill_buffer<n>( &v[0], init );
            }
            
            template<typename T>
            void Write( T * const target ) const
            {
                copy_buffer<n>( &v[0], target );
            }
            
            template<typename T>
            void Read( T const * const source )
            {
                copy_buffer<n>( source, &v[0] );
            }
            
            template<typename S>
            void Read( const VectorList<n,S,Int> & source, const Int k )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = static_cast<Scal>(source[i][k]);
                }
            }
            
            template<typename S>
            void Write( VectorList<n,S,Int> & target, const Int k ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    target[i][k] = static_cast<S>(v[i]);
                }
            }
            
//######################################################
//##                     Access                       ##
//######################################################
            
        public:
            
            Scal * data()
            {
                return &v[0];
            }
            
            const Scal * data() const
            {
                return &v[0];
            }
            
            Scal & operator[]( const Int i )
            {
                return v[i];
            }
            
            const Scal & operator[]( const Int i ) const
            {
                return v[i];
            }
            
            Scal & operator()( Int i )
            {
                return v[i];
            }
            
            const Scal & operator()( const Int i ) const
            {
                return v[i];
            }
            
//######################################################
//##                  Artihmethic                     ##
//######################################################
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator+=( const Tiny::Vector<n,T,Int> & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator-=( const Tiny::Vector<n,T,Int> & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] -= s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator*=( const Tiny::Vector<n,T,Int> & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator/=( const Tiny::Vector<n,T,Int> & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] /= s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator+=( const T & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += s;
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator-=( const T & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] -= s;
                }
                return *this;
            }
            
            template<class T>
            force_inline
            std::enable_if_t<
                std::is_same_v<T,Scal> || (Scalar::IsComplex<Scal> && std::is_same_v<T,Real>),
                CLASS &
            >
            operator*=( const T & s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= s;
                }
                return *this;
            }
            
            force_inline Real Norm() const
            {
                Real r = 0;
                for( Int i = 0; i < n; ++i )
                {
                    r += abs_squared(v[i]);
                }
                return std::sqrt( r );
            }
            
            force_inline friend Real Norm( const CLASS & u )
            {
                return u.Norm();
            }
            
            force_inline void Normalize()
            {
                *this *= (static_cast<Scal>(1) / Norm());
            }
            
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<std::is_same_v<Real,Dummy>,Real> Min() const
            {
                if constexpr ( n > 0 )
                {
                    Real m = v[0];
                    for( Int i = 1; i < n; ++i )
                    {
                        m = std::min(m,v[i]);
                    }
                    return m;
                }
                else
                {
                    return std::numeric_limits<Real>::max();
                }
            }

            
            force_inline friend Scal Dot( const CLASS & x, const CLASS & y )
            {
                Scal r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += x.v[i] * y.v[i];
                }
                return r;
            }
            
            force_inline friend Scal InnerProduct( const CLASS & x, const CLASS & y )
            {
                Scal r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += conj(x.v[i]) * y.v[i];
                }
                return r;
            }
            
            
            force_inline friend Real AngleBetweenUnitVectors( const CLASS & u, const CLASS & w )
            {
                Real a = 0;
                Real b = 0;
                
                for( int i = 0; i < n; ++i )
                {
                    a += real( conj(u[i]-w[i]) * (u[i]-w[i]) );
                    b += real( conj(u[i]+w[i]) * (u[i]+w[i]) );
                }
                
                return static_cast<Real>(2) * atan( std::sqrt(a/b) );
            }
            
            force_inline friend Real Angle( const CLASS & x, const CLASS & y )
            {
                CLASS u = x;
                CLASS w = y;
                
                u.Normalize();
                w.Normalize();
                
                return AngleBetweenUnitVectors(u,w);
            }

            
            force_inline friend void Plus( const CLASS & x, const CLASS & y, CLASS & z )
            {
                for( Int i = 0; i < n; ++i )
                {
                    z.v[i] = x.v[i] + y.v[i];
                }
            }
   
            force_inline friend void Times( const Scal scale, const CLASS & x, CLASS & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] = scale * x.v[i];
                }
            }
            
            force_inline friend void axpy( const Scal alpha, const CLASS & x, CLASS & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] += alpha * x.v[i];
                }
            }

            
            std::string ToString( const int p = 16) const
            {
                std::stringstream sout;
                sout << "{ ";
                sout << ToString(v[0],p);
                for( Int i = 1; i < n; ++i )
                {
                    sout << ", " << ToString(v[i],p);
                }
                sout << " }";
                return sout.str();
            }
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return TO_STD_STRING(CLASS)+"<"+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
        };
        
#undef CLASS
        
    } // namespace Tiny
    
} // namespace Tensors
