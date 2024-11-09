#pragma once

#include <umfpack.h>

namespace Tensors
{
    // TODO: Test whether the code works in the complex case.
    
    template<typename Scal_, typename Int_>
    class UMFPACK
    {
    public:
        
        using Scal = Scal_;
        using Int  = Int_;
  
        using F_T = Scalar::Flag;
        
        static_assert(
            SameQ<Scal,Real64> || SameQ<Scal,Complex64> ||
            SameQ<Scal,Real32> || SameQ<Scal,Complex32>,
            ""
        );
        
        using Real = Scalar::Real<Scal>;
        
        static_assert( SameQ<Int,Int64> || SameQ<Int,Int32>, "");
        
    private:
        
        void * symbolic = nullptr;
        void * numeric  = nullptr;
        
        Real * null     = nullptr;
        
        const Int m;
        const Int n;
        Tensor1<Int ,Int> outer;
        Tensor1<Int ,Int> inner;
        Tensor1<Scal,Int> values;
        
        Int  * Ap = nullptr;
        Int  * Ai = nullptr;
        Real * Ax = nullptr;
        
        Tensor1<Scal,Int> x_buffer;
        Tensor1<Scal,Int> b_buffer;
        
    public:
        
        template<typename ExtInt, typename ExtLInt>
        explicit UMFPACK(
            const ExtInt m_, const ExtInt n_,
            cptr<ExtInt> outer_, cptr<ExtLInt> inner_
        )
        : m        { int_cast<Int>(m_) }
        , n        { int_cast<Int>(n_) }
        , outer    { outer_,  m + 1    }
        , inner    { inner_,  outer[m] }
        , values   { outer[m]          }
        , Ap       { outer.data()      }
        , Ai       { inner.data()      }
        , Ax       { values.data()     }
        , x_buffer { n }
        , b_buffer { m }
        {
            SymbolicFactorization();
        }
        
        
        template<typename ExtScal, typename ExtInt, typename ExtLInt>
        explicit UMFPACK( cref<Sparse::MatrixCSR<ExtScal,ExtInt,ExtLInt>> A )
        : UMFPACK( A.RowCount(), A.ColCount(), A.Outer(), A.Inner(), A.Values() )
        {}
        
        ~UMFPACK()
        {
            FreeSymbolic();
            FreeNumeric ();
        }
        
    private:
        
        template<Op op>
        constexpr int SolveMode()
        {
            static_assert(
                (op == Op::Id       ) || (op == Op::Trans) ||
                (op == Op::ConjTrans) || (op == Op::Conj ), ""
            );
            
            // UMFPACK works with CSC format; we use CSR format.
            // Hence we have to make sure that things are correctly transposed.
            
            switch( op )
            {
                case Op::Trans:     return UMFPACK_A;
                
                case Op::Id:        return UMFPACK_Aat;
                
                case Op::Conj:      return UMFPACK_Aat;
                    
                // Here it becomes hacky.
                // We also have to conjugate the inputs and outputs.
                case Op::ConjTrans: return UMFPACK_A;
            }
        }
        
    public:
        
        template<
            Op op = Op::Id, F_T alpha_flag, F_T beta_flag,
            typename a_T, typename B_T, typename b_T, typename X_T
        >
        void Solve( const a_T alpha, cptr<B_T> B, const b_T beta, mptr<X_T> X )
        {
            if( numeric == nullptr )
            {
                eprint( ClassName()+"::Solve: NumericFactorization has not been called, yet." );
                return;
            }
            
            ptic( ClassName() + "::Solve"
                 + "<" + TypeName<a_T>
                 + "," + TypeName<B_T>
                 + "," + TypeName<b_T>
                 + "," + TypeName<X_T>
                 + ">"
            );
            
            int status = 0;
            
            // UMFPACK works with CSC format; we use CSR format.
            // Hence we have to make sure that things are correctly transposed.

            const int mode = SolveMode();

            // We have to conjugate x to emulate the conjugate-transpose solve.
            combine_buffers<
                F_T::Plus, F_T::Zero, VarSize, Sequential,
                Op::ConjTrans ? Op::Conj : Op::Id, Op::Id
            >(
                Scal(1), B, Scal(0), b_buffer.data(), m
            );


            Real * x_ptr = reinterpret_cast<Real *>(x_buffer.data());
            Real * b_ptr = reinterpret_cast<Real *>(b_buffer.data());
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_dl_solve(
                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
                    );
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_sl_solve(
                        mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null
                    );
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zl_solve(
                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
                    );
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_cl_solve(
                        mode,Ap,Ai,Ax,null,x_ptr,null,b_ptr,null,numeric,null,null
                    );
                }
            }
            else
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_di_solve(mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_si_solve(mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zi_solve(mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_ci_solve(mode,Ap,Ai,Ax,x_ptr,b_ptr,numeric,null,null);
                }
            }
            
            if( status != 0 )
            {
                eprint(ClassName()+"::Solve: Returned error code is "+ToString(status) + ".");
            }

            // We have to conjugate x to emulate the conjugate-transpose solve.
            combine_buffers<
                alpha_flag, beta_flag, VarSize, Sequential,
                Op::ConjTrans ? Op::Conj : Op::Id, Op::Id
            >(
                alpha, x_ptr, beta, X, n
            );
            
            
            ptoc( ClassName() + "::Solve"
                 + "<" + TypeName<a_T>
                 + "," + TypeName<B_T>
                 + "," + TypeName<b_T>
                 + "," + TypeName<X_T>
                 + ">"
            );
            
        }
        
        template<Op op = Op::Id, typename B_T, typename X_T>
        void Solve( cptr<B_T> B, mptr<X_T> X )
        {
            Solve<op,F_T::Plus,F_T::Zero>(
                Scal(1), cptr<B_T> B, Scal(0), mptr<X_T> X
            );
        }
        
        
        std::pair<Scal,Real> Determinant()
        {
            ptic( ClassName() + "::Determinant" );
            
            if( numeric == nullptr )
            {
                eprint( ClassName()+"::Determinant: NumericFactorization has not been called, yet." );
                
                ptoc( ClassName() + "::Determinant" );
                return std::pair<Scal,Real>( 0, 0 );
            }
            
            Scal Mx;
            Real Ex;
            
            int status = 0;
            
            Tiny::Vector<UMFPACK_INFO,double,Int> info;
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_dl_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_sl_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zl_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_cl_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
            }
            else if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_di_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_si_get_determinant(
                        &Mx,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zi_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_ci_get_determinant(
                        reinterpret_cast<Real *>(&Mx),null,&Ex,numeric,info.data()
                    );
                }
            }
            
            if( status != 0 )
            {
                eprint(ClassName() + "::Determinant: Returned error code is " + ToString(status) + ".");
            }
            
            ptoc( ClassName() + "::Determinant" );
            
            return std::pair<Scal,Real>( Mz, Ex );
        }
        
        
    private:
        
        void SymbolicFactorization()
        {
            if( symbolic != nullptr )
            {
                return;
            }
            
            
            ptic( ClassName() + "::SymbolicFactorization" );
            
            int status = 0;
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_dl_symbolic(n,n,Ap,Ai,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_sl_symbolic(n,n,Ap,Ai,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zl_symbolic(n,n,Ap,Ai,null,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_cl_symbolic(n,n,Ap,Ai,null,null,&symbolic,null,null);
                }
            }
            else if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_di_symbolic(n,n,Ap,Ai,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_si_symbolic(n,n,Ap,Ai,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zi_symbolic(n,n,Ap,Ai,null,null,&symbolic,null,null);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_ci_symbolic(n,n,Ap,Ai,null,null,&symbolic,null,null);
                }
            }
            
            if( status != 0 )
            {
                eprint(ClassName()+"::SymbolicFactorization: Returned error code is " + ToString(status) + ".");
            }
            
            ptoc( ClassName() + "::SymbolicFactorization" );
        }
        
    public:
        
        template<typename ExtScal>
        void NumericFactorization( cptr<ExtScal> a_ )
        {
            ptic( ClassName() + "::NumericFactorization" );
            
            values.Read(a_);
            
            int status = 0;
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_dl_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_sl_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zl_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_cl_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
            }
            else //  if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    status = umfpack_di_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    status = umfpack_si_numeric(Ap,Ai,Ax,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    status = umfpack_zi_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    status = umfpack_ci_numeric(Ap,Ai,Ax,null,symbolic,&numeric,null,null);
                }
            }
            
            if( status != 0 )
            {
                eprint(ClassName()+"::NumericFactorization: Returned error code is " + ToString(status) + ".");
            }
            
            ptoc( ClassName() + "::NumericFactorization" );
        }
        
        void FreeSymbolic()
        {
            if( symbolic == nullptr )
            {
                return;
            }
            
            ptic( ClassName() + "::FreeSymbolic" );
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_dl_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    umfpack_sl_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zl_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    umfpack_cl_free_symbolic(&symbolic);
                }
            }
            else // if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_di_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    umfpack_si_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zi_free_symbolic(&symbolic);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    umfpack_ci_free_symbolic(&symbolic);
                }
            }
                
            symbolic = nullptr;
            
            ptoc( ClassName() + "::FreeSymbolic" );
        }
        
        void FreeNumeric()
        {
            if( numeric == nullptr )
            {
                return;
            }
            
            ptic( ClassName() + "::FreeNumeric" );
            
            if constexpr( SameQ<Int,Int64> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_dl_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    umfpack_sl_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zl_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    umfpack_cl_free_numeric(&numeric);
                }
            }
            else // if constexpr( SameQ<Int,Int32> )
            {
                if constexpr( SameQ<Scal,Real64> )
                {
                    umfpack_di_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Real32> )
                {
                    umfpack_si_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Complex64> )
                {
                    umfpack_zi_free_numeric(&numeric);
                }
                else if constexpr( SameQ<Scal,Complex32> )
                {
                    umfpack_ci_free_numeric(&numeric);
                }
            }
            
            ptoc( ClassName() + "::FreeNumeric" );
        }
        
    public:
        
        static std::string ClassName()
        {
            return std::string("UMFPACK") + "<" + TypeName<Scal> + ">";
        }
        
    }; // class UMFPACK
    
} // namespace Sparse
