#pragma once

namespace Tensors
{
    namespace Tiny
    {
        
        template< int n_, typename Scalar_, typename Int_>
        struct Vector
        {
            // Very slim vector type of fixed length, with basic arithmetic operations.
            
            using Scalar = Scalar_;
            using Real   = typename ScalarTraits<Scalar_>::RealType;
            using Int    = Int_;
            
            static constexpr Int n = n_;
            
            static constexpr Scalar zero            = 0;
            static constexpr Scalar half            = 0.5;
            static constexpr Scalar one             = 1;
            static constexpr Scalar two             = 2;
            static constexpr Scalar three           = 3;
            static constexpr Scalar four            = 4;
            static constexpr Real eps               = std::numeric_limits<Real>::min();
            static constexpr Real infty             = std::numeric_limits<Real>::max();
            
            std::array<Scalar,n> v;
            
            Vector() = default;

            explicit Vector( const Scalar init )
            :   v { init }
            {}
            
            explicit Vector( const Scalar * restrict const v_ )
            {
                Read(v_);
            }
            
            ~Vector() = default;
            
            // Copy constructor.
            Vector( const Vector & other )
            {
                Read(&other.v[0]);
            }
            
            friend void swap( Vector & A, Vector & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap(A.v,B.v);
            }
            
            // Copy assignment.
            Vector & operator=( const Vector & x )
            {
                Read(&x.v[0]);
                return *this;
            }

            /* Move constructor */
            Vector( Vector && other ) noexcept
            :   Vector()
            {
                swap(*this, other);
            }
            
            
            Scalar * data()
            {
                return &v[0];
            }
            
            const Scalar * data() const
            {
                return &v[0];
            }
            
            void SetZero()
            {
                zerofy_buffer<n>( &v[0] );
            }
            
            Scalar & operator[]( const Int i )
            {
                return v[i];
            }
            
            const Scalar & operator[]( const Int i ) const
            {
                return v[i];
            }
            
            Scalar & operator()( Int i )
            {
                return v[i];
            }
            
            const Scalar & operator()( const Int i ) const
            {
                return v[i];
            }
            
            void operator+=( const Vector & x )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += x.v[i];
                }
            }
            
            void operator*=( const Vector & x )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= x.v[i];
                }
            }

            void operator+=( const Scalar add )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += add;
                }
            }
            
            void operator-=( const Scalar add )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] -= add;
                }
            }
            
            void operator*=( const Scalar scale )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= scale;
                }
            }
            
            void operator/=( const Scalar scale )
            {
                (*this) *= (one/scale);
            }
            
            Real Norm() const
            {
                Real r = 0;
                for( Int i = 0; i < n; ++i )
                {
                    r += real(conj(v[i]) * v[i]);
                }
                return std::sqrt( r );
            }
            
            friend Real Norm( const Vector & v )
            {
                return v.Norm();
            }
            
            void Normalize()
            {
                *this *= (static_cast<Scalar>(1) / Norm());
            }
            
            

            
            friend Scalar Dot( const Vector & x, const Vector & y )
            {
                Scalar r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += x.v[i] * y.v[i];
                }
                return r;
            }
            
            friend Scalar InnerProduct( const Vector & x, const Vector & y )
            {
                Scalar r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += conj(x.v[i]) * y.v[i];
                }
                return r;
            }
            
            
            friend Real AngleBetweenUnitVectors( const Vector & u, const Vector & w )
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
            
            friend Real Angle( const Vector & x, const Vector & y )
            {
                Vector u = x;
                Vector w = y;
                
                u.Normalize();
                w.Normalize();
                
                return AngleBetweenUnitVectors(u,w);
            }

            
            friend void Plus( const Vector & x, const Vector & y, Vector & z )
            {
                for( Int i = 0; i < n; ++i )
                {
                    z.v[i] = x.v[i] + y.v[i];
                }
            }
   
            friend void Times( const Scalar scale, const Vector & x, Vector & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] = scale * x.v[i];
                }
            }
            
            friend void axpy( const Scalar alpha, const Vector & x, Vector & y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] += alpha * x.v[i];
                }
            }
            
            //            friend Vector operator+( const Vector & x, const Vector & y )
            //            {
            //                Vector z;
            //                for(Int i = 0; i < n; ++i )
            //                {
            //                    z(i) = x(i) + y(i);
            //                }
            //                return z;
            //            }
            
            template<typename S>
            void Read( const S * const a_ )
            {
                copy_cast_buffer<n>( a_, &v[0] );
            }
            
            template<typename S>
            void Write( S * a_ ) const
            {
                copy_cast_buffer<n>( &v[0], a_ );
            }
            
            std::string ToString( const int p = 16) const
            {
                std::stringstream sout;
                sout << "{ ";
                sout << Tools::ToString(v[0],p);
                for( Int i = 1; i < n; ++i )
                {
                    sout << ", " << Tools::ToString(v[i],p);
                }
                sout << " }";
                return sout.str();
            }
            
        public:
            
            inline friend std::ostream & operator<<( std::ostream & s, const Vector & v )
            {
                s << v.ToString();
                return s;
            }
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "Vector<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
        };
    } // namespace Tiny
    
} // namespace Tensors
