#pragma once

#include <cholmod.h>


namespace Tensors
{
    namespace CHOLMOD
    {
        template<typename Scal_, typename Int_, typename LInt_>
        class CholeskyDecomposition
        {
            
        public:
            
            static_assert(
                          SameQ<Scal_,float>
                          ||
                          SameQ<Scal_,double>
                          ||
                          SameQ<Scal_,std::complex<float>>
                          ||
                          SameQ<Scal_,std::complex<double>>
                          ,
                          "CHOLMOD::CholeskyDecomposition supports only single or double precisions floating point numbers (both real and complex)."
                          );
            
            using Scal = Scal_;
            static constexpr int dtype = Scalar::Prec<Scal> == 64u ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
            static constexpr int xtype = Scalar::RealQ<Scal>       ? CHOLMOD_REAL   : CHOLMOD_COMPLEX;
            
            
            static_assert(
                          SameQ<Int_,Int32> || SameQ<Int_,Int64>,
                          "CHOLMOD::CholeskyDecomposition supports only type Int from {Int32, Int64}."
                          );
            
            using  Int = Int_;
            
            static_assert(
                          SameQ<LInt_,Int32> || SameQ<LInt_,Int64>,
                          "CHOLMOD::CholeskyDecomposition supports only type LInt from {Int32, Int64}."
                          );
            
            static_assert(
                          SameQ<Int_,LInt_>,
                          "CHOLMOD::CholeskyDecomposition wants Int == LInt for the moment."
                          );
            
            static_assert(
                          SameQ<LInt_,Int64>
                          ||
                          ( SameQ<Int,Int32> && SameQ<LInt_,Int32>),
                          "CHOLMOD::CholeskyDecomposition supports only type LInt = Int32 only if Int = Int32."
                          );
            
            using LInt = LInt_;
            
            
//            static constexpr int itype = SameQ<Int,Int32>
//            ? SameQ<LInt,Int32>
//            ? CHOLMOD_INT
//            : CHOLMOD_INTLONG
//            : CHOLMOD_LONG;
            
            static constexpr int itype = SameQ<Int,Int32> ? CHOLMOD_INT : CHOLMOD_LONG;
            static constexpr bool long_version = SameQ<Int,Int64>;
            
            
        protected:
            
            cholmod_sparse * A = nullptr;
            cholmod_factor * L = nullptr;
            cholmod_common c ;
            
        public:
            
            template<typename Int1, typename LInt1, typename Int2>
            CholeskyDecomposition( cptr<LInt1> rp, cptr<Int1> ci, const Int2 n_ )
            {
                ptic(ClassName()+"()");
                if constexpr( long_version  )
                {
                    cholmod_l_start (&c);
                }
                else
                {
                    cholmod_start (&c);
                }
                
                c.dtype = dtype;
//                c.itype = itype;
                
                
                c.nmethods = 1 ;
                c.method[0].ordering = CHOLMOD_AMD;
                c.postorder = true;
                
//                ptic(ClassName()+": Allocating A");
                if constexpr( long_version  )
                {
                    A = cholmod_l_allocate_sparse(
                        static_cast<Size_T>(n_),
                        static_cast<Size_T>(n_),
                        static_cast<Size_T>(rp[n_]),
                        true, // Assume CSR-compliant sorting.
                        true, // Assume packed.
                        0,    // Assume full matrix is provided, not only a triangle.
                        xtype,
                        &c
                    );
                }
                else
                {
                    A = cholmod_allocate_sparse(
                        static_cast<Size_T>(n_),
                        static_cast<Size_T>(n_),
                        static_cast<Size_T>(rp[n_]),
                        true, // Assume CSR-compliant sorting.
                        true, // Assume packed.
                        0,    // Assume full matrix is provided, not only a triangle.
                        xtype,
                        &c
                    );
                }
                
                if( A == nullptr )
                {
                    eprint(ClassName()+" failed to allocate A.");
                    return;
                }
//                ptoc(ClassName()+": Allocating A");
                
//                ptic(ClassName()+": Copying pattern of A");
                copy_buffer( rp, reinterpret_cast<LInt*>(A->p), A->nrow + 1 );
                copy_buffer( ci, reinterpret_cast< Int*>(A->i), A->nzmax    );
//                ptoc(ClassName()+": Copying pattern of A");
                
                ptoc(ClassName()+"()");
            }
            
            ~CholeskyDecomposition()
            {
                ptic(std::string("~")+ClassName()+"()");
                     
                if constexpr ( long_version )
                {
                    if( L != nullptr )
                    {
                        cholmod_l_free_factor(&L, &c);
                        L = nullptr;
                    }
                    
                    if( A != nullptr )
                    {
                        cholmod_l_free_sparse(&A, &c);
                        A = nullptr;
                    }
                    
                    cholmod_l_finish(&c);
                }
                else
                {
                    if( L != nullptr )
                    {
                        cholmod_free_factor(&L, &c);
                        L = nullptr;
                    }
                    
                    if( A != nullptr )
                    {
                        cholmod_free_sparse(&A, &c);
                        A = nullptr;
                    }
                    
                    cholmod_finish(&c);
                }
                ptoc(std::string("~")+ClassName()+"()");
            }
            
        public:
            
            void SymbolicFactorization()
            {
                std::string tag = ClassName()+"::SymbolicFactorization";
                
                ptic(tag);
                
                if constexpr ( long_version )
                {
                    if ( L != nullptr )
                    {
                        cholmod_l_free_factor(&L, &c);
                        L = nullptr;
                    }
                    
                    L = cholmod_l_analyze(A, &c);
                }
                else
                {
                    if ( L != nullptr )
                    {
                        cholmod_free_factor(&L, &c);
                        L = nullptr;
                    }
                    
                    L = cholmod_analyze(A, &c);
                }
                
                dump(c.supernodal       );
                dump(c.fl               );
                dump(c.current          );
                dump(c.lnz              );
                dump(c.memory_usage     );
                
                ptoc(tag);
            }
            
            template<typename ExtScal>
            void NumericFactorization( cptr<ExtScal> A_val_ )
            {
                std::string tag = ClassName()+"::NumericFactorization";
                
                ptic(tag);

//                ptic(tag+": copy_buffer");
                copy_buffer( A_val_, reinterpret_cast<Scal*>(A->x), A->nzmax );
//                ptoc(tag+": copy_buffer");
                
                
                if constexpr ( long_version )
                {
//                    ptic(tag+": cholmod_l_factorize");
                    cholmod_l_factorize (A, L, &c);
//                    ptoc(tag+": cholmod_l_factorize");
                }
                else
                {
//                    ptic(tag+": cholmod_factorize");
                    cholmod_factorize (A, L, &c);
//                    ptoc(tag+": cholmod_factorize");
                }
                
                ptoc(tag);
            }
            
            template<typename ExtScal>
            void Solve( cptr<ExtScal> b, mptr<ExtScal> x )
            {
                std::string tag = ClassName()+"::Solve";
                
                ptic(tag);
                
                if constexpr ( long_version )
                {
//                    ptic(tag+": cholmod_l_allocate_dense");
                    cholmod_dense * b_ = cholmod_l_allocate_dense(
                        A->nrow, Scalar::One<Int>, A->nrow, xtype, &c
                    );
//                    ptoc(tag+": cholmod_l_allocate_dense");
                    
//                    ptic(tag+": copy_buffer (input)");
                    copy_buffer( b, reinterpret_cast<Scal*>(b_->x), A->nrow );
//                    ptoc(tag+": copy_buffer (input)");
                    
//                    ptic(tag+": cholmod_l_solve");
                    cholmod_dense * x_ = cholmod_l_solve(CHOLMOD_A, L, b_, &c);
//                    ptoc(tag+": cholmod_l_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + " failed to solve the system.");
                    }
                    else
                    {
//                        ptic(tag+": copy_buffer (output)");
                        copy_buffer( reinterpret_cast<Scal*>(x_->x), x, x_->nrow );
//                        ptoc(tag+": copy_buffer (output)");
                    }
                    
//                    ptic(tag+": cholmod_l_free_dense");
                    cholmod_l_free_dense(&b_, &c);
                    cholmod_l_free_dense(&x_, &c);
//                    ptoc(tag+": cholmod_l_free_dense");
                }
                else
                {
                    
//                    ptic(tag+": cholmod_allocate_dense");
                    cholmod_dense * b_ = cholmod_allocate_dense(
                        A->nrow, Scalar::One<Int>, A->nrow, xtype, &c
                    );
//                    ptoc(tag+": cholmod_allocate_dense");
                    
//                    ptic(tag+": copy_buffer (input)");
                    copy_buffer( b, reinterpret_cast<Scal*>(b_->x), A->nrow );
//                    ptoc(tag+": copy_buffer (input)");
                    
//                    ptic(tag+": cholmod_solve");
                    cholmod_dense * x_ = cholmod_solve(CHOLMOD_A, L, b_, &c);
//                    ptoc(tag+": cholmod_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + " failed to solve the system.");
                    }
                    else
                    {
//                        ptic(tag+": copy_buffer (output)");
                        copy_buffer( reinterpret_cast<Scal*>(x_->x), x, x_->nrow );
//                        ptoc(tag+": copy_buffer (output)");
                    }
                    
//                    ptic(tag+": cholmod_free_dense");
                    cholmod_free_dense(&b_, &c);
                    cholmod_free_dense(&x_, &c);
//                    ptoc(tag+": cholmod_free_dense");
                }
                
                ptoc(tag);
            }
            
            template<typename ExtScal>
            void Solve( cptr<ExtScal> b, mptr<ExtScal> x, const Size_T nrhs )
            {
                std::string tag = ClassName()+"::Solve ( " + ToString(nrhs) +" )";
                
                ptic(tag);
                
                if constexpr ( long_version )
                {
//                    ptic(tag+": cholmod_l_allocate_dense");
                    cholmod_dense * b_ = cholmod_l_allocate_dense(
                        A->nrow, nrhs, A->nrow, xtype, &c
                    );
//                    ptoc(tag+": cholmod_l_allocate_dense");
                    
                    const Size_T n = A->nrow;
                    
                    mptr<Scal> b_ptr = reinterpret_cast<Scal *>(b_->x);

                    ptic(tag+": transpose input");
                    // We have to transpose the inputs.
                    for( Size_T j = 0; j < nrhs; ++j )
                    {
                        for( Size_T i = 0; i < n; ++i )
                        {
                            b_ptr[n * j + i ] = static_cast<Scal>( b[ nrhs * i + j ] );
                        }
                    }
                    ptoc(tag+": transpose input");
                    
//                    ptic(tag+": cholmod_l_solve");
                    cholmod_dense * x_ = cholmod_l_solve(CHOLMOD_A, L, b_, &c);
//                    ptoc(tag+": cholmod_l_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + " failed to solve the system.");
                    }
                    else
                    {
                        // And have to transpose the outputs.
                        
                        ptic(tag+": transpose output");
                        cptr<Scal> x_ptr = reinterpret_cast<Scal *>(x_->x);
                        
                        for( Size_T i = 0; i < n; ++i )
                        {
                            for( Size_T j = 0; j < nrhs; ++j )
                            {
                                x[nrhs * i + j ] = static_cast<ExtScal>( x_ptr[ n * j + i ] );
                            }
                        }
                        ptoc(tag+": transpose output");
                    }
//                    ptic(tag+": cholmod_l_free_dense");
                    cholmod_l_free_dense(&b_, &c);
                    cholmod_l_free_dense(&x_, &c);
//                    ptoc(tag+": cholmod_l_free_dense");
                }
                else
                {
//                    ptic(tag+": cholmod_allocate_dense");
                    cholmod_dense * b_ = cholmod_allocate_dense(
                        A->nrow, nrhs, A->nrow, xtype, &c
                    );
//                    ptoc(tag+": cholmod_allocate_dense");
                    
                    const Size_T n = A->nrow;
                    
                    mptr<Scal> b_ptr = reinterpret_cast<Scal *>(b_->x);

                    ptic(tag+": transpose input");
                    // We have to transpose the inputs.
                    for( Size_T j = 0; j < nrhs; ++j )
                    {
                        for( Size_T i = 0; i < n; ++i )
                        {
                            b_ptr[n * j + i ] = static_cast<Scal>( b[ nrhs * i + j ] );
                        }
                    }
                    ptoc(tag+": transpose input");
                    
//                    ptic(tag+": cholmod_solve");
                    cholmod_dense * x_ = cholmod_solve(CHOLMOD_A, L, b_, &c);
//                    ptoc(tag+": cholmod_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + " failed to solve the system.");
                    }
                    else
                    {
                        // And have to transpose the outputs.
                        
                        ptic(tag+": transpose output");
                        cptr<Scal> x_ptr = reinterpret_cast<Scal *>(x_->x);
                        
                        for( Size_T i = 0; i < n; ++i )
                        {
                            for( Size_T j = 0; j < nrhs; ++j )
                            {
                                x[nrhs * i + j ] = static_cast<ExtScal>( x_ptr[ n * j + i ] );
                            }
                        }
                        ptoc(tag+": transpose output");
                    }
//                    ptic(tag+": cholmod_free_dense");
                    cholmod_free_dense(&b_, &c);
                    cholmod_free_dense(&x_, &c);
//                    ptoc(tag+": cholmod_free_dense");
                }
                
                ptoc(tag);
            }
            
            
        public:
            
            Int ThreadCount()
            {
                return c.nthreads_max;
            }
            
            Int RowCount()
            {
                return A->nrow;
            }
            
            Int ColCount()
            {
                return A->ncol;
            }
            
        public:
            
            std::string ClassName() const
            {
                return std::string("CHOLMOD::CholeskyDecomposition")+"<"+ TypeName<Scal> + "," + TypeName<Int> + "," +TypeName<LInt> + ">";
            }
            
            
        }; // class CholeskyDecomposition
            
    } // namespace CHOLMOD
    
} // namespace Tensors

