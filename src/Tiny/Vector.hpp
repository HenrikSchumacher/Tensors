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
            
            static_assert( n > 0, "Vector dimension must be postive.");

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
            constexpr CLASS( const std::initializer_list<S> w )
            {
                const Int n__ = std::min(n,static_cast<Int>(w.size()));
//                
                cptr<S> w_ = &(*w.begin());
                
                if( n__ == 1 )
                {
                    const Scal value = scalar_cast<Scal>(w_[0]);
                    
                    for( Int i = 0; i < n; ++i )
                    {
                        v[i] = value;
                    }
                }
                else
                {
                    for( Int i = 0; i < n__; ++i )
                    {
                        v[i] = scalar_cast<Scal>(w_[i]);
                    }
                    
                    for( Int i = n__; i < n; ++i )
                    {
                        v[i] = Scalar::Zero<Scal>;
                    }
                }
                
//                const Int m = int_cast<Int>(w.size());
//
//                if(m > n)
//                {
//                    eprint(TO_STD_STRING(CLASS)+": Length of initializer list must not exceed length of n");
//                }
//                else
//                {
//                    cptr<S> w_ = &(*w.begin());
//                    
//                    for( Int i = 0; i < m; ++i )
//                    {
//                        v[i] = scalar_cast<Scal>(w_[i]);
//                    }
//                    for( Int i = m; i < n; ++i )
//                    {
//                        v[i] = Scalar::Zero<Scal>;
//                    }
//                }
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
            
            constexpr void SetZero()
            {
                zerofy_buffer<n>( &v[0] );
            }
            
            constexpr void Fill( cref<Scal> init )
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
            force_inline std::enable_if_t<SameQ<Real,Dummy>,void> MinMax( mref<Real> min_, mref<Real> max_ ) const
            {
                Real min = v[0];
                Real max = v[0];
                
                for( Int i = 1; i < n; ++i )
                {
                    min = Tools::Min(min,v[i]);
                    max = Tools::Max(max,v[i]);
                }

                min_ = min;
                max_ = max;
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Min() const
            {
                Real m = v[0];
                for( Int i = 1; i < n; ++i )
                {
                    m = Tools::Min(m,v[i]);
                }
                return m;
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Int> MinPos() const
            {
                Real min = v[0];
                Int  pos = 0;
                
                for( Int i = 1; i < n; ++i )
                {
                    if( v[i] < min )
                    {
                        pos = i;
                        min = v[i];
                    }
                }
                
                return pos;
            }

            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Max() const
            {
                Real m = v[0];
                for( Int i = 1; i < n; ++i )
                {
                    m = Tools::Max(m,v[i]);
                }
                return m;
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Int> MaxPos() const
            {
                Real max = v[0];
                Int  pos = 0;
                
                for( Int i = 1; i < n; ++i )
                {
                    if( v[i] > max )
                    {
                        pos = i;
                        max = v[i];
                    }
                }
                
                return pos;
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> MaxNorm() const
            {
                Real m = Abs(v[0]);
                for( Int i = 1; i < n; ++i )
                {
                    m = Tools::Max(m,Abs(v[i]));
                }
                return m;
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
            
            force_inline friend CLASS Plus( cref<CLASS> x, cref<CLASS> y  )
            {
                CLASS z;
                
                Plus( x, y, z );
  
                return z;
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

            template<
                Scalar::Flag alpha_flag = Scalar::Flag::Generic,
                Scalar::Flag beta_flag  = Scalar::Flag::Generic
            >
            force_inline friend void LinearCombine(
                cref<Scal> alpha, cref<CLASS> x, cref<Scal> beta, cref<CLASS> y, mref<CLASS> z
            )
            {
                // Sets z = alpha * x + beta * y.
                
                for( Int i = 0; i < n; ++i )
                {
                    switch( alpha_flag )
                    {
                        case Scalar::Flag::Generic:
                        {
                            switch( beta_flag )
                            {
                                case Scalar::Flag::Generic:
                                {
                                    z.v[i] = alpha * x.v[i] + beta * y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Plus:
                                {
                                    z.v[i] = alpha * x.v[i] + y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Zero:
                                {
                                    z.v[i] = alpha * x.v[i];
                                    break;
                                }
                                case Scalar::Flag::Minus:
                                {
                                    z.v[i] = alpha * x.v[i] - y.v[i];
                                    break;
                                }
                            }
                            break;
                        }
                        case Scalar::Flag::Plus:
                        {
                            switch( beta_flag )
                            {
                                case Scalar::Flag::Generic:
                                {
                                    z.v[i] = x.v[i] + beta * y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Plus:
                                {
                                    z.v[i] = x.v[i] + y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Zero:
                                {
                                    z.v[i] = x.v[i];
                                    break;
                                }
                                case Scalar::Flag::Minus:
                                {
                                    z.v[i] = x.v[i] - y.v[i];
                                    break;
                                }
                            }
                            break;
                        }
                        case Scalar::Flag::Zero:
                        {
                            switch( beta_flag )
                            {
                                case Scalar::Flag::Generic:
                                {
                                    z.v[i] = beta * y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Plus:
                                {
                                    z.v[i] = y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Zero:
                                {
                                    z.v[i] = Scalar::Zero<Scal>;
                                    break;
                                }
                                case Scalar::Flag::Minus:
                                {
                                    z.v[i] = - y.v[i];
                                    break;
                                }
                            }
                            break;
                        }
                        case Scalar::Flag::Minus:
                        {
                            switch( beta_flag )
                            {
                                case Scalar::Flag::Generic:
                                {
                                    z.v[i] = - x.v[i] + beta * y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Plus:
                                {
                                    z.v[i] = - x.v[i] + y.v[i];
                                    break;
                                }
                                case Scalar::Flag::Zero:
                                {
                                    z.v[i] = - x.v[i];
                                    break;
                                }
                                case Scalar::Flag::Minus:
                                {
                                    z.v[i] = - (x.v[i] + y.v[i]);
                                    break;
                                }
                            }
                            break;
                        }
                    }
                    
                }
            }
            
            template<
                Scalar::Flag alpha_flag = Scalar::Flag::Generic,
                Scalar::Flag beta_flag  = Scalar::Flag::Generic
            >
            force_inline friend CLASS LinearCombine(
                cref<Scal> alpha, cref<CLASS> x, cref<Scal> beta, cref<CLASS> y
            )
            {
                // Returns alpha * x + beta * y.
                CLASS z;
                
                LinearCombine<alpha_flag,beta_flag>( alpha, x, beta, y, z );
                
                return z;
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
        force_inline void Cross( 
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v, mref<Vector<3,Scal,Int>> w )
        {
            w[0] = u[1] * v[2] - u[2] * v[1];
            w[1] = u[2] * v[0] - u[0] * v[2];
            w[2] = u[0] * v[1] - u[1] * v[0];
        }
        
        template<typename Scal, typename Int>
        force_inline Vector<3,Scal,Int> Cross( 
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v )
        {
            Vector<3,Scal,Int> w;
            Cross( u, v, w );
            return w;
        }
        
        template<typename Scal, typename Int>
        force_inline Scal Det( 
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v, mref<Vector<3,Scal,Int>> w
        )
        {
            return w[0] * ( u[1] * v[2] - u[2] * v[1] )
                +  w[1] * ( u[2] * v[0] - u[0] * v[2] )
                +  w[2] * ( u[0] * v[1] - u[1] * v[0] );
        }
        
        template<typename Scal, typename Int>
        force_inline Scal Det( cref<Vector<2,Scal,Int>> u, cref<Vector<2,Scal,Int>> v )
        {
            return u[0] * v[1] - u[1] * v[0];
        }
        
        template<int n, typename Scal, typename Int>
        force_inline Scalar::Real<Scal> Distance( cref<Vector<n,Scal,Int>> u, cref<Vector<n,Scal,Int>> v )
        {
            Scalar::Real<Scal> r2 = 0;
            
            for( Int i = 0; i < n; ++i )
            {
                r2 += (u[i] - v[i]) * (u[i] - v[i]);
            }
            return Sqrt(r2);
        }
        
    } // namespace Tiny
    
} // namespace Tensors
