#pragma once

namespace Tensors
{
    namespace Tiny
    {
#define CLASS Vector
        
        template<int n_, typename Scal_, typename Int_, Size_T alignment> class VectorList;
        
        template< int n_, typename Scal_, typename Int_>
        class CLASS
        {
            // Very slim vector type of fixed length, with basic arithmetic operations.
            
#include "Tiny_Details.hpp"
            
            static constexpr Int n = n_;

            template<typename S, Size_T alignment>
            CLASS( cref<VectorList<n,S,Int,alignment>> v_list, const Int k )
            {
                Read(v_list, k);
            }
            
            template<typename S>
            CLASS( cref<Tensor2<S,Int>> matrix, const Int k )
            {
                Read(matrix.data(k));
            }
            
            template<typename S>
            CLASS( cptr<S> matrix, const Int k )
            {
                Read( &matrix[n * k] );
            }
            
            template<typename S>
            CLASS( cref<std::initializer_list<S>> w )
            {
                const Int m = int_cast<Int>(w.size());
                
                if(m > n)
                {
                    eprint(TO_STD_STRING(CLASS)+": Length of initializer list must not exceed length of n");
                }
                else
                {
                    cptr<S> w_ = &(*w.begin());
                    
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
            
            alignas(Tools::Alignment) std::array<Scal,n> v;
            
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
            
            void Fill( cref<Scal> init )
            {
                fill_buffer<n>( &v[0], init );
            }
            
            template<typename T>
            void Write( mptr<T> target ) const
            {
                copy_buffer<n>( &v[0], target );
            }
            
            template<typename T>
            void Read( cptr<T> source )
            {
                copy_buffer<n>( source, &v[0] );
            }
            
            template<typename S, Size_T alignment>
            void Read( cref<VectorList<n,S,Int,alignment>> source, const Int k )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = static_cast<Scal>(source[i][k]);
                }
            }
            
            template<typename S>
            void Read( cref<Tensor2<S,Int>> source, const Int k )
            {
                Real( source.data(k) );
            }
            
            template<typename S, Size_T alignment>
            void Write( mref<VectorList<n,S,Int,alignment>> target, const Int k ) const
            {
                for( Int i = 0; i < n; ++i )
                {
                    target[i][k] = static_cast<S>(v[i]);
                }
            }
            
            template<typename S>
            void Write( mref<Tensor2<S,Int>> source, const Int k ) const
            {
                Write( source.data(k) );
            }
            
            
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op self_op, Op target_op,
                typename R, typename S, typename T
            >
            void CombineInto( cref<R> alpha, cref<S> beta, mptr<T> target ) const
            {
                combine_buffers<alpha_flag,beta_flag,self_op,target_op,n>(
                    alpha, data(), beta, target
                );
            }
            
            template<typename T >
            void AddTo( mptr<T> target ) const
            {
                add_to_buffer<n>(data(), target);
            }
            
//######################################################
//##                     Access                       ##
//######################################################
            
        public:
            
            mptr<Scal> data()
            {
                return &v[0];
            }
            
            cptr<Scal> data() const
            {
                return &v[0];
            }
            
            mref<Scal> operator[]( const Int i )
            {
                return v[i];
            }
            
            cref<Scal> operator[]( const Int i ) const
            {
                return v[i];
            }
            
            mref<Scal> operator()( Int i )
            {
                return v[i];
            }
            
            cref<Scal> operator()( const Int i ) const
            {
                return v[i];
            }
            
//######################################################
//##                  Artihmethic                     ##
//######################################################
            
            template<class T>
            force_inline
            std::enable_if_t<
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator+=( cref<Tiny::Vector<n,T,Int>> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator-=( cref<Tiny::Vector<n,T,Int>> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator*=( cref<Tiny::Vector<n,T,Int>> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator/=( cref<Tiny::Vector<n,T,Int>> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator+=( cref<T> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator-=( cref<T> s )
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
                SameQ<T,Scal> || (Scalar::ComplexQ<Scal> && SameQ<T,Real>),
                CLASS &
            >
            operator*=( cref<T> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= s;
                }
                return *this;
            }

            force_inline Real SquaredNorm() const
            {
                Real r = 0;
                for( Int i = 0; i < n; ++i )
                {
                    r += AbsSquared(v[i]);
                }
                return r;
            }
            
            force_inline Real Norm() const
            {
                return Sqrt( SquaredNorm() );
            }
            
            force_inline friend Real Norm( cref<CLASS> u )
            {
                return u.Norm();
            }
            
            force_inline void Normalize()
            {
                *this *= (static_cast<Scal>(1) / Norm());
            }
            
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Min() const
            {
                if constexpr ( n > 0 )
                {
                    Real m = v[0];
                    for( Int i = 1; i < n; ++i )
                    {
                        m = Min(m,v[i]);
                    }
                    return m;
                }
                else
                {
                    return std::numeric_limits<Real>::max();
                }
            }

            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Max() const
            {
                if constexpr ( n > 0 )
                {
                    Real m = v[0];
                    for( Int i = 1; i < n; ++i )
                    {
                        m = Tools::Max(m,v[i]);
                    }
                    return m;
                }
                else
                {
                    return std::numeric_limits<Real>::lowest();
                }
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> MaxNorm() const
            {
                if constexpr ( n > 0 )
                {
                    Real m = Abs(v[0]);
                    for( Int i = 1; i < n; ++i )
                    {
                        m = Tools::Max(m,Abs(v[i]));
                    }
                    return m;
                }
                else
                {
                    return Scalar::Zero<Real>;
                }
            }
            
            
            
            force_inline friend Scal Dot( cref<CLASS> x, cref<CLASS> y )
            {
                Scal r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += x.v[i] * y.v[i];
                }
                return r;
            }
            
            force_inline friend Scal InnerProduct( cref<CLASS> x, cref<CLASS> y )
            {
                Scal r (0);
                
                for( Int i = 0; i < n; ++i )
                {
                    r += Conj(x.v[i]) * y.v[i];
                }
                return r;
            }
            
            
            force_inline friend Real AngleBetweenUnitVectors( cref<CLASS> u, cref<CLASS> w )
            {
                Real a = 0;
                Real b = 0;
                
                for( int i = 0; i < n; ++i )
                {
                    a += Re( Conj(u[i]-w[i]) * (u[i]-w[i]) );
                    b += Re( Conj(u[i]+w[i]) * (u[i]+w[i]) );
                }
                
                return Scalar::Two<Real> * atan( Sqrt(a/b) );
            }
            
            force_inline friend Real Angle( cref<CLASS> x, cref<CLASS> y )
            {
                CLASS u = x;
                CLASS w = y;
                
                u.Normalize();
                w.Normalize();
                
                return AngleBetweenUnitVectors(u,w);
            }

            
            force_inline friend void Plus( cref<CLASS> x, cref<CLASS> y, mref<CLASS> z )
            {
                for( Int i = 0; i < n; ++i )
                {
                    z.v[i] = x.v[i] + y.v[i];
                }
            }
   
            force_inline friend void Times( cref<Scal> scale, cref<CLASS> x, mref<CLASS> y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] = scale * x.v[i];
                }
            }
            
            force_inline friend void axpy( cref<Scal> alpha, cref<CLASS> x, mref<CLASS> y )
            {
                for( Int i = 0; i < n; ++i )
                {
                    y.v[i] += alpha * x.v[i];
                }
            }

            
            std::string ToString() const
            {
                std::stringstream sout;
                sout << "{ ";
                sout << Tools::ToString(v[0]);
                for( Int i = 1; i < n; ++i )
                {
                    sout << ", " << Tools::ToString(v[i]);
                }
                sout << " }";
                return sout.str();
            }
            
            template<class Stream_T>
            Stream_T & ToStream( mref<Stream_T> s ) const
            {
                s << "{ ";
                s << Tools::ToString(v[0]);
                for( Int i = 1; i < n; ++i )
                {
                    s << ", " << Tools::ToString(v[i]);
                }
                s << " }";
                
                return s;
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
        
        
        template<typename Scal, typename Int>
        void Cross( cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v, mref<Vector<3,Scal,Int>> w )
        {
            w[0] = u[1] * v[2] - u[2] * v[1];
            w[1] = u[2] * v[0] - u[0] * v[2];
            w[2] = u[0] * v[1] - u[1] * v[0];
        }
        
    } // namespace Tiny
    
} // namespace Tensors
