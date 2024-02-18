#pragma once

#include <initializer_list>

namespace Tensors
{
    namespace Tiny
    {
        template< int n_, typename Scal_, typename Int_> class SelfAdjointMatrix;
        
        template< int m_, int n_, typename Scal_, typename Int_, Size_T alignment> class MatrixList;
        
        template< int m_, int n_, typename Scal_, typename Int_>
        class Matrix
        {
        public:
            
            using Class_T = Matrix;
            
#include "Tiny_Constants.hpp"
                        
            static constexpr Int m = m_;
            static constexpr Int n = n_;
            
            
            using ColVector_T = Vector<m,Scal,Int>;
            using RowVector_T = Vector<n,Scal,Int>;
            
            using Vector_T    = Vector<n,Scal,Int>;
            
        protected:
            
            
            // TODO: Switching to std::array<RowVector_T,m> A would simplify several things...
            // TODO: But beware: Tiny::Vector has some nontrivial memory alignment!
            
            alignas(Tools::Alignment) std::array<std::array<Scal,n>,m> A;
            
            
        public:
            

            Matrix() = default;

            ~Matrix() = default;

            Matrix(std::nullptr_t) = delete;

            explicit Matrix( const Scal * a )
            {
                Read(a);
            }
            
            explicit Matrix( cref<Scal> init )
            :   A {{{init}}}
            {}
            
            template<typename S>
            constexpr Matrix( const std::initializer_list<S[n]> list )
            {
                const Int m__ = static_cast<Int>(list.size());
                
                auto iter { list.begin() };
                for( Int i = 0; i < m__; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = static_cast<Scal>((*iter)[j]);
                    }
                    ++iter;
                }
                
                for( Int i = m__; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = 0;
                    }
                }
            }
            
            // Copy constructor
            Matrix( const Matrix & other )
            {
                Read( &other.A[0][0] );
            }
            
            /* Move constructor */
            Matrix( Matrix && other ) noexcept
            {
                swap(*this, other);
            }
            
            // Copy assignment operator
            Matrix & operator=( Matrix other )
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                swap(*this, other);

                return *this;
            }
            
            template<typename S, Size_T alignment>
            Matrix( cref<MatrixList<m,n,S,Int,alignment>> m_list, const Int k )
            {
                Read(m_list, k);
            }
            
            template<typename S>
            Matrix( cref<Tensor3<S,Int>> tensor, const Int k )
            {
                Read(tensor.data(k));
            }

            
//######################################################
//##                     Access                       ##
//######################################################

#include "Tiny_Matrix_Common.hpp"
            
///######################################################
///##                     Memory                       ##
///######################################################
                
        public:

            void SetZero()
            {
                zerofy_buffer<m*n>( &A[0][0] );
            }
            
            template<typename T>
            void Fill( cref<T> init )
            {
                fill_buffer<m*n>( &A[0][0], static_cast<T>(init) );
            }


            template<typename B_T>
            void AddTo( mptr<B_T> B ) const
            {
                add_to_buffer<m*n>( &A[0][0], B );
            }
                
            
            void WriteRow( mref<RowVector_T> u, const Int i )
            {
                u.Read( &A[i] );
            }
            
            void WriteCol( mref<ColVector_T> v, const Int j )
            {
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][j];
                }
            }
            
            
///######################################################
///##             Reading from raw buffers             ##
///######################################################

            /// BLAS-like read-modify method _with stride_.
            /// Reads the full matrix.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Read(
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opB) )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        combine_buffers<beta_flag,alpha_flag,n,Sequential,op,opA>(
                            beta, &B[ldB*i], alpha, &A[i][0]
                        );
                    }

                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<beta_flag,alpha_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

            /// BLAS-like read-modify method _without stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Read(
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.

                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opB) )
                {
                    combine_buffers<beta_flag,alpha_flag,m*n,Sequential,op,opA>(
                        beta, B, alpha, &A[0][0]
                    );
                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        cptr<B_T> B_j = &B[n * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

            /// BLAS-like read-modify method _without stride_.
            template<Op opB = Op::Id, typename B_T>
            void Read( cptr<B_T> B )
            {
                /// Compute `opB(B)` and store it in `A`.
                
                Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
                    Scal(0), Scal(1), B
                );
            }

            /// BLAS-like read-modify method _with stride_.
            template<Op opB = Op::Id, typename B_T>
            void Read( cptr<B_T> B, const Int ldB )
            {
                /// Compute `opB(B)` and store it in `A`.
                
                Read<Scalar::Flag::Zero,Scalar::Flag::Plus,Op::Id,opB>(
                    Scal(0), Scal(1), B, ldB
                );
            }


            /// BLAS-like read-modify method _with stride_.
            /// Reads only the top left portion.
            /// It is meant to read from the right and bottom boundaries of a large matrix.
            template<
                bool chop_m_Q, bool chop_n_Q,
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void ReadChopped(
                const Int m_max, const Int n_max,
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
            )
            {
                /// Compute `alpha * opA(A) + beta * opB(B)` and store it in `A`.
                /// `opA` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opB) ? Op::Conj : Op::Id;
                
                const Int m_c = Min(m,m_max);
                const Int n_c = Min(n,n_max);

                if constexpr ( NotTransposedQ(opB) )
                {
                    for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                    {
                        combine_buffers<beta_flag,alpha_flag,COND(chop_n_Q,0,n),Sequential,op,opA>(
                            beta, &B[ldB*i], alpha, &A[i][0], n_c
                        );
                    }
                }
                else if constexpr ( TransposedQ(opB) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    
                    /// TODO:
                    /// I think best performance should be obtained
                    /// by making the fixed-size loop the inner loop.
                    
                    for( Int j = 0; j < COND(chop_n_Q,n_c,n); ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                        {
                            combine_scalars<beta_flag,alpha_flag,op,opA>(
                                beta, B_j[i], alpha, A[i][j]
                            );
                        }
                    }
                }
            }

        //    // Scattered read-modify method.
        //    // Might be useful in supernodal arithmetic for sparse matrices.
        //    template<Scalar::Flag a_flag, Op op, typename alpha_T, typename B_T>
        //    void Read( const alpha_T alpha, cptr<B_T> B, const Int ldB, cptr<Int> idx  )
        //    {
        //        // Reading A = alpha * op(B)
        //
        //        if constexpr ( op == Op::Id )
        //        {
        //            if constexpr ( a_flag == Scalar::Flag::Plus )
        //            {
        //                for( Int i = 0; i < m; ++i )
        //                {
        //                    copy_buffer<n>( &B[ldB*idx[i]], &A[i][0] );
        //                }
        //            }
        //        }
        //        else if constexpr ( op == Op::Conj )
        //        {
        //            for( Int i = 0; i < m; ++i )
        //            {
        //                for( Int j = 0; j < n; ++j )
        //                {
        //                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B[i][j] );
        //                }
        //            }
        //        }
        //        else
        //        {
        //            // TODO: Not sure whether it would be better to swap the two loops here...
        //            for( Int j = 0; j < n; ++j )
        //            {
        //                cptr<B_T> B_j = &B[ldB*idx[j]];
        //
        //                for( Int i = 0; i < m; ++i )
        //                {
        //                    A[i][j] = ScalarOperator<a_flag,op>( alpha, B_j[i] );
        //                }
        //            }
        //        }
        //    }

        ///######################################################
        ///##              Writing to raw buffers              ##
        ///######################################################

            /// BLAS-like write-modify method _with stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Write(
                cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        combine_buffers<alpha_flag,beta_flag,n,Sequential,op,opB>(
                            alpha, &A[i][0], beta, &B[ldB*i]
                        );
                    }
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    for( Int j = 0; j < n; ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }



            /// BLAS-like write-modify method _without stride_.
            template<
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void Write(
                cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.

                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    /// This is the only reason we use a version without stride:
                    /// If the whole matrix is stored contiguously, we can vectorize over
                    /// row-ends!
                    combine_buffers<alpha_flag,beta_flag,m*n,Sequential,op,opB>(
                        alpha, &A[0][0], beta, B
                    );
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    /// I think that no real optimizations can be made here.
                    /// Nonetheless, we let the compiler know that the stride is a
                    /// compile-time constant. But since the alignment of B is unknown to
                    /// the compiler, this might be quite useless.

                    // TODO: Compare with reverse ordering of loops.
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        mptr<B_T> B_j = &B[n * j];
                        
                        for( Int i = 0; i < m; ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }


            /// BLAS-like write-modify method _with stride_.
            template<Op opA = Op::Id, typename B_T>
            void Write( mptr<B_T> B, const Int ldB ) const
            {
                /// Compute `B = opA(A)`.

                Write<Scalar::Flag::Plus,Scalar::Flag::Zero,opA,Op::Id>( 
                    Scalar::One<B_T>, Scalar::Zero<B_T>, B, ldB
                );
            }

            /// BLAS-like write-modify method _without stride_.
            template<Op op = Op::Id, typename B_T>
            void Write( mptr<B_T> B ) const
            {
                /// B = opA(A)
                Write<Scalar::Flag::Plus,Scalar::Flag::Zero,op,Op::Id>(
                    Scalar::One<B_T>, Scalar::Zero<B_T>, B
                );
            }

        //    /// Row-scattered write-modify method.
        //    /// Might be useful in supernodal arithmetic for sparse matrices.
        //    template<
        //        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        //        typename alpha_T, typename beta_T, typename B_T
        //    >
        //    void Write( cref<alpha_T> alpha, cref<beta_T> beta, mptr<B_T> B, const Int ldB, cptr<Int> idx ) const
        //    {
        //
        //        // Writing B[idx[i]][j] = alpha * A[i][j] + beta * B[idx[i]][j]
        //        for( Int i = 0; i < m; ++i )
        //        {
        //            combine_buffers<alpha_flag,beta_flag,n>(
        //                alpha, &A[i][0], beta, &B[ldB*idx[i]]
        //            );
        //        }
        //    }

            /// BLAS-like write-modify method _with stride_.
            /// Writes only the top left portion.
            /// It is meant to write to the right and bottom boundaries of a large matrix.
            template<
                bool chop_m_Q, bool chop_n_Q,
                Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                Op opA = Op::Id, Op opB = Op::Id,
                typename alpha_T, typename beta_T, typename B_T
            >
            void WriteChopped(
                const Int m_max, const Int n_max,
                cref<alpha_T> alpha, cref<beta_T> beta, cptr<B_T> B, const Int ldB = n
            ) const
            {
                /// Computes `B = alpha * opA(A) + beta * opB(B)`.
                /// `opB` can only be `Op::Id` or `Op::Conj`.
                
                constexpr Op op = ConjugatedQ(opA) ? Op::Conj : Op::Id;
                
                const Int m_c = Min( m, m_max );
                const Int n_c = Min( n, n_max );
                
                if constexpr ( NotTransposedQ(opA) )
                {
                    for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                    {
                        combine_buffers<alpha_flag,beta_flag,COND(chop_n_Q,0,n),Sequential,op,opB>(
                            alpha, &A[i][0], beta, &B[ldB*i], n_c
                        );
                    }
                }
                else if constexpr ( TransposedQ(opA) )
                {
                    // TODO: Compare with reverse ordering of loops.
                    for( Int j = 0; j < COND(chop_n_Q,n_c,n); ++j )
                    {
                        mptr<B_T> B_j = &B[ldB * j];
                        
                        for( Int i = 0; i < COND(chop_m_Q,m_c,m); ++i )
                        {
                            combine_scalars<alpha_flag,beta_flag,op,opB>(
                                alpha, A[i][j], beta, B_j[i]
                            );
                        }
                    }
                }
            }
            
///######################################################
///##             Arithmetic with scalars              ##
///######################################################
         
        public:
            
            template<class T>
            force_inline mref<Matrix> operator+=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += lambda;
                    }
                }
                
                return *this;
            }

            template<class T>
            force_inline mref<Matrix> operator-=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= lambda;
                    }
                }
                
                return *this;
            }
            
            template<class T>
            force_inline mref<Matrix> operator*=( const T lambda_ )
            {
                const auto lambda = scalar_cast<Scal>(lambda_);
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= lambda;
                    }
                }
                
                return *this;
            }

            template<class T>
            force_inline mref<Matrix> operator/=( const T lambda )
            {
                return (*this) *= ( scalar_cast<Scal>(Inv<T>(lambda)) );
            }
            
///######################################################
///##                   Arithmetic                     ##
///######################################################

            // TODO: Make this more type flexible.
            // TODO: Also, where is Minus?
            force_inline friend void Plus( const Matrix & x, const Matrix & y, const Matrix & z )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        z.A[i][j] = x.A[i][j] + y.A[i][j];
                    }
                }
                return z;
            }
            
            // TODO: Make this more type flexible.
            // TODO: Also, where is operator-?
            [[nodiscard]] force_inline friend const Matrix operator+( const Matrix & x, const Matrix & y )
            {
                Matrix z;
                
                Plus( x, y, z);
                
                return z;
            }
            
            template<class T>
            force_inline
            mref<Matrix> operator+=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] += B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            template<class T>
            force_inline
            mref<Matrix> operator-=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] -= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            // TODO: Not sure whether I want * to represent compontenwise multiplication.
            template<class T>
            force_inline
            mref<Matrix> operator*=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] *= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
            // TODO: Not sure whether I want / to represent compontenwise division.
            template<class T>
            force_inline
            mref<Matrix> operator/=( cref<Tiny::Matrix<m,n,T,Int>> B )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] /= B.A[i][j];
                    }
                }
                
                return *this;
            }
            
        public:
            
            template<AddTo_T addto, int K_, typename X_T, typename Y_T >
            force_inline friend void
            Dot(
                cref<Tiny::Matrix<m_,K_,X_T ,Int>> X,
                cref<Tiny::Matrix<K_,n_,Y_T ,Int>> Y,
                mref<Tiny::Matrix<m_,n_,Scal,Int>> Z
            )
            {
                if constexpr( mat_enabledQ && (SameQ<X_T,double> || SameQ<X_T,float>) && (SameQ<Y_T,double> || SameQ<Y_T,float>) && (SameQ<Scal,double> || SameQ<Scal,float>) )
                {
                    fixed_dot_mm_clang_2<m_,n_,K_,addto>( &X[0][0], &Y[0][0], &Z[0][0] );
                }
                else
                {
                    fixed_dot_mm_vec<m_,n_,K_,addto>( &X[0][0], &Y[0][0], &Z[0][0] );
                }
            }
            
            template<AddTo_T addto, typename x_T, typename y_T
            >
            force_inline friend void
            Dot(
                cref<Tiny::Matrix<m_,n_,Scal,Int>> M,
                cref<Tiny::Vector<n_,   x_T, Int>> x,
                mref<Tiny::Vector<m_,   y_T, Int>> y
            )
            {
                if constexpr ( addto == Tensors::AddTo )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        y[i] += dot_buffers<n>( M[i], x.data() );
                    }
                }
                else
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        y[i]  = dot_buffers<n>( M[i], x.data() );
                    }
                }
            }
            
            template<Op op>
            void LowerFromUpper()
            {
                for( Int i = 0; i < n; ++i )
                {
                    // TODO: Unroll this loop trough templates?
                    for( Int j = 0; j < i; ++j )
                    {
                        A[i][j] = ScalarOperator<Scalar::Flag::Plus,op>(A[j][i]);
                    }
                }
            }
           
        public:

            template<bool upper_triangle_only = false>
            [[nodiscard]] force_inline std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<n,Scal,Int>,
                Matrix<n,n,Scal,Int>
            > AHA() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<n,Scal,Int>,
                    Matrix<n,n,Scal,Int>
                > B;
                
                
                // TODO: Not sure whether this is faster than using Dot + ConjugateTranspose.
                {
                    constexpr Int k = 0;
                    
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        // TODO: Unroll this loop trough templates?
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] = Conj_A_ki * A[k][j];
                        }
                    }
                    
                }
                
                for( Int k = 1; k < m; ++k )
                {
                    for( Int i = 0; i < n; ++i )
                    {
                        const Scal Conj_A_ki = Conj(A[k][i]);
                        
                        // TODO: Unroll this loop trough templates?
                        for( Int j = i; j < n; ++j )
                        {
                            B[i][j] += Conj_A_ki * A[k][j];
                        }
                    }
                }

                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }
            
            template<bool upper_triangle_only = false>
            [[nodiscard]] force_inline std::conditional_t<
                upper_triangle_only,
                SelfAdjointMatrix<m,Scal,Int>,
                Matrix<m,m,Scal,Int>
            > AAH() const
            {
                std::conditional_t<
                    upper_triangle_only,
                    SelfAdjointMatrix<m,Scal,Int>,
                    Matrix<m,m,Scal,Int>
                > B;
                
                for( Int i = 0; i < m; ++i )
                {
                    B[i][i] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[i][0] );
                    
                    // TODO: Unroll this loop trough templates?
                    for( Int j = i + 1; j < m; ++j )
                    {
                        B[i][j] = dot_buffers<n,Sequential,Op::Id,Op::Conj>( &A[i][0], &A[j][0] );
                    }
                }
                
                B.template LowerFromUpper<Op::Conj>();
                
                return B;
            }

        public:
            
            
            force_inline void Conjugate( Matrix & B ) const
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        B.A[i][j] = Conj(A[i][j]);
                    }
                }
            }

            [[nodiscard]] force_inline Matrix Conjugate() const
            {
                Matrix B;
                
                Conjugate(B);
                
                return B;
            }
            
        public:
            
            force_inline void Transpose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = A[i][j];
                    }
                }
                
                // TODO: Test whether the following is as fast; if yes, then simply.
//                Write<Op::Trans,Op::Id>( &B[0][0] );
            }
            
            [[nodiscard]] force_inline Matrix<n,m,Scal,Int> Transpose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                Transpose(B);
                
                return B;
            }

            force_inline void ConjugateTranspose( mref<Matrix<n,m,Scal,Int>> B ) const
            {
                for( Int j = 0; j < n; ++j )
                {
                    for( Int i = 0; i < m; ++i )
                    {
                        B[j][i] = Conj(A[i][j]);
                    }
                }
                
                // TODO: Test whether the following is as fast; if yes, then simply.
                
//                Write<Op::ConjTrans,Op::Id>( &B[0][0] );
            }

            [[nodiscard]] force_inline Matrix<n,m,Scal,Int> ConjugateTranspose() const
            {
                Matrix<n,m,Scal,Int> B;
                
                ConjugateTranspose(B);
                
                return B;
            }
            
            
            
            [[nodiscard]] force_inline Real MaxNorm() const
            {
                return norm_max<m*n>( &A[0][0] );
            }
            
            [[nodiscard]] force_inline Real FrobeniusNorm() const
            {
                Real AA = 0;
                
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        AA += AbsSquared(A[i][j]);
                    }
                }
                
                return Sqrt(AA);

                // TODO: Test whether the following is as fast; if yes, then simply.
//                norm_2<m*n>(&A[0][0]);
            }

            
            [[nodiscard]] friend std::string ToString( cref<Matrix> M )
            {
                std::stringstream sout;
                sout << "{\n";
                sout << "\t{ ";
                if( (m > 0) && (n > 0) )
                {
                    sout << ToString(M.A[0][0]);
                    for( Int j = 1; j < n; ++j )
                    {
                        sout << ", " << ToString(M.A[0][j]);
                    }
                    for( Int i = 1; i < m; ++i )
                    {
                        sout << " },\n\t{ ";
                        
                        sout << ToString(M.A[i][0]);
                        
                        for( Int j = 1; j < n; ++j )
                        {
                            sout << ", " << ToString(M.A[i][j]);
                        }
                    }
                }
                sout << " }\n}";
                return sout.str();
            }
        
            
            inline friend std::ostream & operator<<( std::ostream & s, cref<Matrix> M )
            {
                s << ToString(M);
                return s;
            }
            
        public:
            
            void Threshold( const Real threshold )
            {
                for( Int i = 0; i < m; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        if( Abs(A[i][j]) <= threshold )
                        {
                            A[i][j] = static_cast<Scal>(0);
                        }
                    }
                }
            }
            
        public:
            
            template<class T>
            force_inline void GivensLeft( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= 2 )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    // Assumes that AbsSquared(c) + AbsSquared(s) == one.
                    // Assumes that i != j.
                    // Multiplies matrix with the rotation
                    //
                    //    /               \
                    //    |      c     s  |
                    //    |  -conj(s)  c  |
                    //    \               /
                    //
                    // in the i-j-plane from the left.
                    
                    for( Int k = 0; k < n; ++k )
                    {
                        const Scal x = A[i][k];
                        const Scal y = A[j][k];
                        
                        A[i][k] =      c    * x + s * y;
                        A[j][k] = - Conj(s) * x + c * y;
                    }
                }
            }

            template<class T>
            force_inline void GivensRight( const T c_, const T s_, const Int i, const Int j )
            {
                if constexpr ( n >= 2 )
                {
                    const Scal c = scalar_cast<Scal>(c_);
                    const Scal s = scalar_cast<Scal>(s_);
                    
                    // Assumes that AbsSquared(c) + AbsSquared(s) == one.
                    // Assumes that i != j.
                    // Multiplies matrix with rotation
                    //
                    //    /               \
                    //    |      c     s  |
                    //    |  -conj(s)  c  |
                    //    \               /
                    //
                    // in the i-j-plane from the right.
                    
                    for( Int k = 0; k < m; ++k )
                    {
                        const Scal x = A[k][i];
                        const Scal y = A[k][j];
                        
                        A[k][i] = c * x - Conj(s) * y;
                        A[k][j] = s * x +    c    * y;
                    }
                }
            }
            
            
            
        public:
           
            constexpr force_inline void SetIdentity()
            {
                static_assert(m==n, "SetIdentity is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = KroneckerDelta<Real>(i,j);
                    }
                }
            }
            
            force_inline void MakeDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "MakeDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : static_cast<Scal>(0);
                    }
                }
            }
            
            force_inline void SetDiagonal( const Tensors::Tiny::Vector<n,Scal,Int> & v )
            {
                static_assert(m==n, "SetDiagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    A[i][i] = v[i];
                }
            }
            
            [[nodiscard]] Scal Det() const
            {
                if constexpr ( m != n )
                {
                    return 0;
                }
                
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if constexpr ( n == 3 )
                {
                    return (
                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
                    );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                if constexpr ( n > 3 )
                {
                    Scal M [n][n];
                    
                    Write( &M[0][0] );
                    
                    Scal sign (one);
                    
                    for(Int k = 0; k < n - 1; ++k )
                    {
                        //Pivot - row swap needed
                        if( M[k][k] == zero )
                        {
                            Int l = 0;
                            for( l = k + 1; l < n; ++l )
                            {
                                if( M[l][k] != zero )
                                {
                                    std::swap_ranges( &M[l][0], &M[l][n], &M[k][0] );
                                    sign = -sign;
                                    break;
                                }
                            }
                            
                            //No entries != 0 found in column k -> det = 0
                            if(l == n)
                            {
                                return zero;
                            }
                        }
                        
                        //Apply formula
                        for( Int i = k + 1; i < n; ++i )
                        {
                            for( Int j = k + 1; j < n; ++j )
                            {
                                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j];
                                if(k != 0)
                                {
                                    M[i][j] /= M[k-1][k-1];
                                }
                            }
                        }
                    }
                    
                    return sign * M[n-1][n-1];
                }
                
                return static_cast<Scal>(0);
            }
            
            
        public:
            
            force_inline void SetHouseHolderReflector( const Vector_T & u, const Int begin, const Int end )
            {
                static_assert(m==n, "SetHouseHolderReflector is only defined for square matrices.");
                
                // Write the HouseHolder reflection of u into the matrix; assumes that u is zero outside [begin,...,end[.
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                
                SetIdentity();

                for( Int i = begin; i < end; ++i )
                {
                    for( Int j = begin; j < end; ++j )
                    {
                        A[i][j] -= two * u[i] * Conj(u[j]);
                    }
                }
            }
            
            
            force_inline void SetGivensRotation( const Scal c, const Scal s, const Int i, const Int j )
            {
                static_assert(m==n, "SetGivensRotation is only defined for square matrices.");
                
                // Mostly meant for debugging purposes, thus not extremely optimized.
                // Assumes that AbsSquared(c) + AbsSquared(s) == one.
                // Write Givens rotion
                //
                //    /              \
                //    |     c     s  |
                //    | -conj(s)  c  |
                //    \              /
                //
                // in the i-j-plane into the matrix.
                
                SetIdentity();
                
                A[i][i] = c;
                A[i][j] = s;
                A[j][i] = -Conj(s);
                A[j][j] = c;
            }
            
            void Diagonal( Vector<n,Scal,Int> & v ) const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                for( Int i = 0; i < n; ++i )
                {
                    v[i] = A[i][i];
                }
            }
            
            [[nodiscard]] Vector<n,Scal,Int>  Diagonal() const
            {
                static_assert(m==n, "Diagonal is only defined for square matrices.");
                
                Vector<n,Scal,Int> v;
                Diagonal(v);
                return v;
            }
            
        public:

            static constexpr Int RowCount()
            {
                return m;
            }
            
            static constexpr Int ColCount()
            {
                return n;
            }
            
            static constexpr Int Dimension( const Int i )
            {
                if( i == 0 )
                {
                    return m;
                }
                if( i == 1 )
                {
                    return n;
                }
                return static_cast<Int>(0);
            }
            
            [[nodiscard]] static std::string ClassName()
            {
                return std::string("Tiny::Matrix") + "<"+std::to_string(m)+","+std::to_string(n)+","+TypeName<Scal>+","+TypeName<Int>+">";
            }
            
        };
                
        template<int m, int K, int n, typename X_T, typename Y_T, typename Int>
        [[nodiscard]] force_inline Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int>
        Dot(
            cref<Tiny::Matrix<m,K,X_T,Int>> X,
            cref<Tiny::Matrix<K,n,Y_T,Int>> Y
        )
        {
            Tiny::Matrix<m,n,decltype( X_T(1) * Y_T(1) ),Int> Z;
         
            Dot<Overwrite>(X,Y,Z);
            
            return Z;
        }
        
    } // namespace Tiny
    
    
    
} // namespace Tensors
