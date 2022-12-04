#pragma once

namespace Tensors
{
    namespace Small
    {
        
        template< int n_, typename Scalar_, typename Int_>
        struct Vector
        {
            
            using Scalar = Scalar_;
            using Real   = typename ScalarTraits<Scalar_>::RealType;
            using Int    = Int_;
            
            static constexpr Int n = n_;
            // Very slim vector type of fixed length, with basic arithmetic operations.
            
            Scalar v [n];
            
            Vector() = default;

            explicit Vector( const Scalar init )
            :   v { init }
            {}
            
            ~Vector() = default;
            
            Vector( const Vector & other )
            {
                Read(&other.v[0]);
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
                zerofy_buffer( &v[0], n );
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
            
            void operator*=( const Scalar scale )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= scale;
                }
            }
            
            Vector & operator=( const Vector & x )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = x.v[i];
                }
                return *this;
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
            
            void Scale( const Scalar scale )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] *= scale;
                }
            }
            
            void Normalize()
            {
                Scale( static_cast<Scalar>(1) / Norm() );
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
                copy_cast_buffer( a_, &v[0], n );
            }
            
            template<typename S>
            void Write( S * a_ ) const
            {
                copy_cast_buffer( &v[0], a_, n );
            }
            
            std::string ToString( const Int p = 16) const
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
    } // namespace Small
    
} // namespace Tensors
