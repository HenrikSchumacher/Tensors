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
            CLASS( cptr<S> vector )
            {
                Read( vector );
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
            
            template<
                typename a_T, typename x_T, typename b_T, typename y_T,
                Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = Id, Op opy = Id
            >
            force_inline void LinearCombine(
                cref<a_T> a, cptr<x_T> x, cref<b_T> b, cptr<y_T> y
            )
            {
                // Sets z = a * x + b * y.
                
                combine_buffers<a_flag, b_flag, n, Sequential, opx, opy>(
                     scalar_cast<Scal>(a), x, scalar_cast<Scal>(b), y, &v[0]
                );
            }
//            
//            template<Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = Id, Op opy = Id>
//            force_inline friend void LinearCombine(
//                cref<Scal> a, cref<CLASS> x, cref<Scal> b, cref<CLASS> y, mref<CLASS> z
//            )
//            {
//                // Sets z = a * x + b * y.
//                
//                combine_buffers<a_flag, b_flag, n, Sequential, opx, opy>(
//                    a, x.data(), b, y.data(), z.data()
//                );
//            }
            
            
//            force_inline friend void axpy( cref<Scal> a, cref<CLASS> x, mref<CLASS> y )
//            {
//                combine_buffers<F_Gen, F_Plus, n, Sequential>(
//                    a, x.data(), Scalar::One<Scal>, y.data()
//                );
//            }
            
            
            
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
                    min = std::min(min,v[i]);
                    max = std::max(max,v[i]);
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
                    m = std::min(m,v[i]);
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
                    m = std::max(m,v[i]);
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
                    m = std::max(m,Abs(v[i]));
                }
                return m;
            }
            
            
            [[nodiscard]] force_inline friend Real AngleBetweenUnitVectors( cref<CLASS> u, cref<CLASS> w )
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
            
            [[nodiscard]] force_inline friend Real Angle( cref<CLASS> x, cref<CLASS> y )
            {
                CLASS u = x;
                CLASS w = y;
                
                u.Normalize();
                w.Normalize();
                
                return AngleBetweenUnitVectors(u,w);
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
        [[nodiscard]] force_inline const Vector<3,Scal,Int> Cross(
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v )
        {
            Vector<3,Scal,Int> w;
            Cross( u, v, w );
            return w;
        }
        
        template<typename Scal, typename Int>
        [[nodiscard]] force_inline const Scal Det(
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v, mref<Vector<3,Scal,Int>> w
        )
        {
            return w[0] * ( u[1] * v[2] - u[2] * v[1] )
                +  w[1] * ( u[2] * v[0] - u[0] * v[2] )
                +  w[2] * ( u[0] * v[1] - u[1] * v[0] );
        }
        
        template<typename Scal, typename Int>
        [[nodiscard]] force_inline const Scal Det( cref<Vector<2,Scal,Int>> u, cref<Vector<2,Scal,Int>> v )
        {
            return u[0] * v[1] - u[1] * v[0];
        }
        
        template<int n, typename Scal, typename Int>
        [[nodiscard]] force_inline const Scalar::Real<Scal> Distance( cref<Vector<n,Scal,Int>> u, cref<Vector<n,Scal,Int>> v )
        {
            Scalar::Real<Scal> r2 = 0;
            
            for( Int i = 0; i < n; ++i )
            {
                r2 += (u[i] - v[i]) * (u[i] - v[i]);
            }
            return Sqrt(r2);
        }
        
        
        
        
        template<
            int n,
            typename a_T, typename x_T, typename x_Int,
            typename b_T, typename y_T, typename y_Int,
                          typename z_T, typename z_Int,
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = Id, Op opy = Id
        >
        force_inline void LinearCombine(
            cref<a_T> a, cref<Vector<n,x_T,x_Int>> x,
            cref<b_T> b, cref<Vector<n,y_T,y_Int>> y,
                         mref<Vector<n,z_T,z_Int>> z
        )
        {
            // Computes  z = a * x + b * y.
            
            combine_buffers<a_flag, b_flag, n, Sequential, opx, opy>(
                scalar_cast<z_T>(a), x.data(), scalar_cast<z_T>(b), y.data(), z.data()
            );
        }
        
        
        template<
            int n, typename Scal, typename Int,
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = Id, Op opy = Id,
            typename a_T, typename x_T, typename x_Int,
            typename b_T, typename y_T, typename y_Int
        >
        [[nodiscard]] force_inline const Vector<n,Scal,Int> MakeVector(
            cref<a_T> a, cref<Vector<n,x_T,x_Int>> x,
            cref<b_T> b, cref<Vector<n,y_T,y_Int>> y
        )
        {
            // Returns z = a * x + b * y.
            Vector<n,Scal,Int> z;
            
            LinearCombine( a, x, b, y, z);
            
            return z;
        }
        
        
        
        template<
            int n, typename Scal, typename Int,
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = Id, Op opy = Id,
            typename a_T, typename x_T, typename b_T, typename y_T
        >
        [[nodiscard]] force_inline const Vector<n,Scal,Int> MakeVector(
            cref<a_T> a, cptr<x_T> x,
            cref<b_T> b, cptr<y_T> y
        )
        {
            // Returns z = a * x + b * y.
            Vector<n,Scal,Int> z;
            
            combine_buffers<a_flag, b_flag, n, Sequential, opx, opy>(
                scalar_cast<Scal>(a), x, scalar_cast<Scal>(b), y, z.data()
            );
            
            return z;
        }
        
        
         
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int
        >
        [[nodiscard]] force_inline const 
        Vector<n,decltype(x_T(0)+y_T(0)),decltype(x_Int(0)+y_Int(0))> operator+(
            cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y
        )
        {
            // Returns z = x + y.
            
            using T = decltype(x_T  (0) + y_T  (0));
            using I = decltype(x_Int(0) + y_Int(0));
            
            return MakeVector<n,T,I,F_Plus,F_Plus>(
                Scalar::One<T>,x,Scalar::One<T>,y
            );
        }
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int
        >
        [[nodiscard]] force_inline const
        Vector<n,decltype(x_T(0)+y_T(0)),decltype(x_Int(0)+y_Int(0))> operator-(
            cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y
        )
        {
            // Returns z = x + y.
            
            using T = decltype(x_T  (0) + y_T  (0));
            using I = decltype(x_Int(0) + y_Int(0));
            
            return MakeVector<n,T,I,F_Plus,F_Minus>(
                Scalar::One<T>,x,-Scalar::One<T>,y
            );
        }
        
        
        

        template<int n, typename a_T, typename x_T, typename Int>
        [[nodiscard]] force_inline const 
        Vector<n,decltype( x_T(1) * a_T(1) ),Int> operator*( 
            cref<a_T> a, cref<Vector<n,x_T,Int>> x
        )
        {
            // Returns z = a * x.
            
            using T = decltype(x_T(1) * a_T(1));
            
            return MakeVector<n,T,Int,F_Gen,F_Zero>(
                scalar_cast<T>(a),x.data(),Scalar::Zero<T>,x.data()
            );
        }
        
        template<int n, typename x_T, typename Int, typename a_T>
        [[nodiscard]] force_inline const 
        Vector<n,decltype( x_T(1) * a_T(1) ),Int> operator*( 
            cref<Vector<n,x_T,Int>> x, cref<a_T> a
        )
        {
            // Returns z = a * x.
            
            using T = decltype(x_T(1) * a_T(1));
            
            return MakeVector<n,T,Int,F_Gen,F_Zero>(
                scalar_cast<T>(a),x.data(),Scalar::Zero<T>,x.data()
            );
        }

        
        
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const decltype( x_T(1) * y_T(1) ) Dot(
            cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y
        )
        {
            return dot_buffers<n,Sequential,Id,Id>( x.data(), y.data() );
        }
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const decltype( x_T(1) * y_T(1) ) InnerProduct(
            cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y
        )
        {
            return dot_buffers<n,Sequential,Conj,Id>( x.data(), y.data() );
        }
        
    } // namespace Tiny
    
} // namespace Tensors
