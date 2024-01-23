#pragma once

// TODO: Expression templates: https://en.wikipedia.org/wiki/Expression_templates
namespace Tensors
{
    namespace Tiny
    {
        template<int n_, typename Scal_, typename Int_, Size_T alignment> class VectorList;
        
        template<int n_, typename Scal_, typename Int_>
        class Vector
        {
            /// Very slim vector type of fixed length, with basic arithmetic operations.
            
        public:
            
            using Class_T = Vector;
            
#include "Tiny_Details.hpp"
            
        public:
            
            static constexpr Int n = n_;
            
            Vector() = default;

            ~Vector() = default;

            Vector(std::nullptr_t) = delete;

            explicit Vector( const Scal * a )
            {
                Read(a);
            }

            // Copy assignment operator
            Vector & operator=( Vector other )
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                swap(*this, other);

                return *this;
            }

            /* Move constructor */
            Vector( Vector && other ) noexcept
            {
                swap(*this, other);
            }
            

            template<typename S, Size_T alignment>
            Vector( cref<VectorList<n,S,Int,alignment>> v_list, const Int k )
            {
                Read(v_list, k);
            }
            
            template<typename S>
            Vector( cref<Tensor2<S,Int>> matrix, const Int k )
            {
                Read(matrix.data(k));
            }
            
            template<typename S>
            Vector( cptr<S> matrix, const Int k )
            {
                Read( &matrix[n * k] );
            }
            
            
            template<typename S>
            Vector( cptr<S> vector )
            {
                Read( vector );
            }
            
            template<typename S>
            constexpr Vector( const std::initializer_list<S> w )
            {
                const Int n__ = Tools::Min(n,static_cast<Int>(w.size()));
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
            
            explicit Vector( const Scal init )
            :   v {{init}}
            {}
            
            // Copy constructor
            Vector( const Vector & other )
            {
                Read( &other.v[0] );
            }

            friend void swap( Vector & A, Vector & B ) noexcept
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
            
///######################################################
///##                  Artihmethic                     ##
///######################################################
            
            template<
                typename a_T, typename x_T, typename b_T, typename y_T,
                Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = O_Id, Op opy = O_Id
            >
            force_inline Vector & LinearCombine(
                cref<a_T> a, cptr<x_T> x, cref<b_T> b, cptr<y_T> y
            )
            {
                // Sets z = a * x + b * y.
                
                combine_buffers<a_flag, b_flag, n, Sequential, opx, opy>(
                     scalar_cast<Scal>(a), x, scalar_cast<Scal>(b), y, &v[0]
                );
            }
            
            template<class T>
            force_inline mref<Vector> operator+=( cref<Tiny::Vector<n,T,Int>> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += s[i];
                }
                // TODO: Compare to
//                combine_buffers<F_Plus,F_Plus,n>(
//                     Scalar::One<Scal>, x, Scalar::One<Scal>, y, &v[0]
//                );
                
                return *this;
            }
            
            template<class T>
            force_inline mref<Vector> operator-=( cref<Tiny::Vector<n,T,Int>> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] -= s[i];
                }
  
                // TODO: Compare to
//                combine_buffers<F_Plus,F_Minus,n>(
//                     Scalar::One<Scal>, x, -Scalar::One<Scal>, y, &v[0]
//                );
                
                return *this;
            }
            
            
            // TODO: Vectorize all these.
            
            template<class T>
            force_inline mref<Vector> operator*=( cref<Tiny::Vector<n,T,Int>> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline mref<Vector> operator/=( cref<Tiny::Vector<n,T,Int>> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] /= s[i];
                }
                return *this;
            }
            
            template<class T>
            force_inline mref<Vector> operator+=( cref<T> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] += s;
                }
                return *this;
            }
            
            template<class T>
            force_inline mref<Vector> operator-=( cref<T> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] -= s;
                }
                return *this;
            }
            
            template<class T>
            force_inline mref<Vector> operator*=( cref<T> s )
            {
                for(Int i = 0; i < n; ++i )
                {
                    v[i] *= s;
                }
                return *this;
            }
            

            force_inline Real SquaredNorm() const
            {
                return norm_2_squared<n>( &v[0] );
            }
            
            force_inline Real Norm() const
            {
                return norm_2<n>( &v[0] );
            }
            
            force_inline friend Real Norm( cref<Vector> u )
            {
                return u.Norm();
            }
            
            force_inline void Normalize()
            {
                *this *= Inv(Norm());
            }
            
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,std::pair<Real,Real>> MinMax() const
            {
                return minmax_buffer<n>(&v[0]);
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Min() const
            {
                return min_buffer<n>(&v[0]);
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Int> MinPos() const
            {
                return min_pos_buffer<n>(&v[0]);
            }

            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> Max() const
            {
                return max_buffer<n>(&v[0]);
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Int> MaxPos() const
            {
                return max_pos_buffer<n>(&v[0]);
            }
            
            template <typename Dummy = Scal>
            force_inline std::enable_if_t<SameQ<Real,Dummy>,Real> MaxNorm() const
            {
                return norm_max<n>( &v[0] );
            }
            
  
            [[nodiscard]] force_inline friend Real AngleBetweenUnitVectors( cref<Vector> u, cref<Vector> w )
            {
                const Real a = (u-w).SquaredNorm();
                const Real b = (u+w).SquaredNorm();
                                
                return Scalar::Two<Real> * atan( Sqrt(a/b) );
            }
            
            [[nodiscard]] force_inline friend Real Angle( cref<Vector> x, cref<Vector> y )
            {
                Vector u = x;
                Vector w = y;
                
                u.Normalize();
                w.Normalize();
                
                return AngleBetweenUnitVectors(u,w);
            }

            
            [[nodiscard]] friend std::string ToString( cref<Vector> x )
            {
                std::stringstream sout;
                sout << "{ ";
                sout << ToString(x.v[0]);
                for( Int i = 1; i < n; ++i )
                {
                    sout << ", " << ToString(x.v[i]);
                }
                sout << " }";
                return sout.str();
            }
            
            template<class Stream_T>
            Stream_T & ToStream( mref<Stream_T> s ) const
            {
                s << "{ ";
                s << ToString(v[0]);
                for( Int i = 1; i < n; ++i )
                {
                    s << ", " << ToString(v[i]);
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
                return std::string("Tiny::Vector") + "<"+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
        };
                
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
            cref<Vector<3,Scal,Int>> u, cref<Vector<3,Scal,Int>> v, cref<Vector<3,Scal,Int>> w
        )
        {
            return w[0] * ( u[1] * v[2] - u[2] * v[1] )
                +  w[1] * ( u[2] * v[0] - u[0] * v[2] )
                +  w[2] * ( u[0] * v[1] - u[1] * v[0] );
        }
        
        template<typename Scal, typename Int>
        [[nodiscard]] force_inline const Scal Det( 
            cref<Vector<2,Scal,Int>> u, cref<Vector<2,Scal,Int>> v )
        {
            return u[0] * v[1] - u[1] * v[0];
        }

        template<int n, typename Scal, typename Int>
        [[nodiscard]] force_inline const Scalar::Real<Scal> SquaredDistance(
            cref<Vector<n,Scal,Int>> u, cref<Vector<n,Scal,Int>> v
        )
        {
            return (u-v).SquaredNorm();
        }
        
        template<int n, typename Scal, typename Int>
        [[nodiscard]] force_inline const Scalar::Real<Scal> Distance( 
            cref<Vector<n,Scal,Int>> u, cref<Vector<n,Scal,Int>> v
        )
        {
            return Sqrt(SquaredDistance(u,v));
        }
        
        
        
        
        template<
            int n,
            typename a_T, typename x_T, typename x_Int,
            typename b_T, typename y_T, typename y_Int,
                          typename z_T, typename z_Int,
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = O_Id, Op opy = O_Id
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
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = O_Id, Op opy = O_Id,
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
            
            LinearCombine( a, x, b, y, z );
            
            return z;
        }
        
        
        
        template<
            int n, typename Scal, typename Int,
            Flag a_flag = F_Gen, Flag b_flag = F_Gen, Op opx = O_Id, Op opy = O_Id,
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
        
        
         
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const 
        Vector<n,decltype(x_T(0)+y_T(0)),decltype(x_Int(0)+y_Int(0))> 
        operator+( cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y )
        {
            // Returns z = x + y.
            
            using T = decltype(x_T  (0) + y_T  (0));
            using I = decltype(x_Int(0) + y_Int(0));
            
            return MakeVector<n,T,I,F_Plus,F_Plus>(
                Scalar::One<T>,x,Scalar::One<T>,y
            );
        }
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const
        Vector<n,decltype(x_T(0)+y_T(0)),decltype(x_Int(0)+y_Int(0))> 
        operator-( cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y )
        {
            // Returns z = x + y.
            
            using T = decltype(x_T  (0) + y_T  (0));
            using I = decltype(x_Int(0) + y_Int(0));
            
            return MakeVector<n,T,I,F_Plus,F_Minus>(
                Scalar::One<T>,x,-Scalar::One<T>,y
            );
        }
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const
        Vector<n,decltype(x_T(1)*y_T(1)),decltype(x_Int(0)+y_Int(0))>
        operator*( cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y )
        {
            // Returns z = x + y.
            
            using T = decltype(x_T  (1) * y_T  (1));
            using I = decltype(x_Int(0) + y_Int(0));
            
            Vector<n,T,I> z;
            
            for( I i = 0; i < n; ++i )
            {
                z[i] = x[i] * y[i];
            }
            
            return z;
        }
        
        

        template<int n, typename a_T, typename x_T, typename Int>
        [[nodiscard]] force_inline const 
        std::enable_if_t<Scalar::ScalarQ<a_T>, Vector<n,decltype( x_T(1) * a_T(1) ),Int>>
        operator*( cref<a_T> a, cref<Vector<n,x_T,Int>> x )
        {
            // Returns z = a * x.
         
            static_assert(Scalar::ScalarQ<a_T>, "a_T must be a scalar type.");
            
            using T = decltype(x_T(1) * a_T(1));
            
            return MakeVector<n,T,Int,F_Gen,F_Zero>(
                scalar_cast<T>(a),x.data(),Scalar::Zero<T>,x.data()
            );
        }
        
        template<int n, typename x_T, typename Int, typename a_T>
        [[nodiscard]] force_inline const 
        std::enable_if_t<Scalar::ScalarQ<a_T>, Vector<n,decltype( x_T(1) * a_T(1) ),Int>>
        operator*( cref<Vector<n,x_T,Int>> x, cref<a_T> a )
        {
            // Returns a * x.
            
            using T = decltype(x_T(1) * a_T(1));
            
            return MakeVector<n,T,Int,F_Gen,F_Zero>(
                scalar_cast<T>(a),x.data(),Scalar::Zero<T>,x.data()
            );
        }
        
        template<int n, typename x_T, typename Int, typename a_T>
        [[nodiscard]] force_inline const
        std::enable_if_t<Scalar::ScalarQ<a_T>, Vector<n,decltype( x_T(1) * a_T(1) ),Int>>
        operator/( cref<Vector<n,x_T,Int>> x, cref<a_T> a )
        {
            // Returns x/a.
            
            return x * Inv<a_T>(a);
        }
        
        
        template<int n, typename a_T, typename x_T, typename z_T, typename Int>
        force_inline std::enable_if_t<Scalar::ScalarQ<a_T>, void> 
        Times( cref<a_T> a, cref<Vector<n,x_T,Int>> x, mref<Vector<n,z_T,Int>> z )
        {
            // Returns z = a * x.
            
            static_assert(Scalar::ScalarQ<a_T>, "a_T must be a scalar type.");
            
            combine_buffers<F_Gen, F_Zero, n, Sequential>(
                scalar_cast<z_T>(a), x.data(), scalar_cast<z_T>(0), z.data()
            );
        }
        
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const decltype( x_T(1) * y_T(1) ) 
        Dot( cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y )
        {
            return dot_buffers<n,Sequential,O_Id,O_Id>( x.data(), y.data() );
        }
        
        template<int n, typename x_T, typename x_Int, typename y_T, typename y_Int>
        [[nodiscard]] force_inline const decltype( x_T(1) * y_T(1) ) 
        InnerProduct( cref<Vector<n,x_T,x_Int>> x, cref<Vector<n,y_T,y_Int>> y )
        {
            return dot_buffers<n,Sequential,O_Conj,O_Id>( x.data(), y.data() );
        }
        
    } // namespace Tiny
    
} // namespace Tensors
