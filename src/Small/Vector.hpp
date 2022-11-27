#pragma once

namespace Tensors
{
    namespace Small
    {
        
        template< int AmbDim, typename Real, typename Int>
        struct Vector
        {
            // Very slim vector type of fixed length, with basic arithmetic operations.
            
            Real v [AmbDim] = {};
            
            Vector() = default;
            
            ~Vector() = default;
            
            Vector( const Vector & other )
            {
                Read(&other.v[0]);
            }
            
            Real * data()
            {
                return &v[0];
            }
            
            const Real * data() const
            {
                return &v[0];
            }
            
            void SetZero()
            {
                zerofy_buffer( &v[0], AmbDim );
            }
            
            void rand()
            {
                std::uniform_real_distribution<Real> unif(-1,1);
                std::default_random_engine re{static_cast<unsigned int>(time(0))};
                
                for( Int i = 0; i < AmbDim; ++i )
                {
                    v[ i ] = unif(re);
                }
            }
            
            Real & operator[]( const Int i )
            {
                return v[i];
            }
            
            const Real & operator[]( const Int i ) const
            {
                return v[i];
            }
            
            Real & operator()( Int i )
            {
                return v[i];
            }
            
            const Real & operator()( const Int i ) const
            {
                return v[i];
            }
            
            void operator+=( const Vector & x )
            {
                for(Int i = 0; i < AmbDim; ++i )
                {
                    v[i] += x.v[i];
                }
            }
            
            void operator*=( const Real scale )
            {
                for(Int i = 0; i < AmbDim; ++i )
                {
                    v[i] *= scale;
                }
            }
            
            Vector & operator=( const Vector & x )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    v[i] = x.v[i];
                }
                return *this;
            }
            
            Real Norm() const
            {
                Real r = v[0] * v[0];
                for( Int i = 1; i < AmbDim; ++i )
                {
                    r += v[i] * v[i];
                }
                return std::sqrt(r);
            }
            
            
            friend Real Dot( const Vector & x, const Vector & y )
            {
                Real r = x.v[0] * y.v[0];
                for( Int i = 1; i < AmbDim; ++i )
                {
                    r += x.v[i] * y.v[i];
                }
                return r;
            }
            
            friend void Plus( const Vector & x, const Vector & y, Vector & z )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    z.v[i] = x.v[i] + y.v[i];
                }
            }
            
            friend void Times( const Real scale, const Vector & x, Vector & y )
            {
                for( Int i = 0; i < AmbDim; ++i )
                {
                    y.v[i] = scale * x.v[i];
                }
            }
            
            //            friend Vector operator+( const Vector & x, const Vector & y )
            //            {
            //                Vector z;
            //                for(Int i = 0; i < AmbDim; ++i )
            //                {
            //                    z(i) = x(i) + y(i);
            //                }
            //                return z;
            //            }
            
            template<typename S>
            void Read( const S * const a_ )
            {
                copy_cast_buffer( a_, &v[0], AmbDim );
            }
            
            template<typename S>
            void Write( S * a_ ) const
            {
                copy_cast_buffer( &v[0], a_, AmbDim );
            }
            
            std::string ToString( const Int n = 16) const
            {
                std::stringstream sout;
                sout.precision(n);
                sout << "{ ";
                sout << v[0];
                for( Int i = 1; i < AmbDim; ++i )
                {
                    sout << ", " << v[i];
                }
                sout << " }";
                return sout.str();
            }
            
            //        #pragma omp declare reduction( + : Vector : omp_out+=omp_in )
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return AmbDim;
            }
            
            static std::string ClassName()
            {
                return "Vector<"+std::to_string(AmbDim)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
            }
        };
    } // namespace Small
    
} // namespace Tensors