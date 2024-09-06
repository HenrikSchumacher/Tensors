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
            static constexpr int dtype  = Scalar::Prec<Scal> == 64u ? CHOLMOD_DOUBLE : CHOLMOD_SINGLE;
            static constexpr int xtype  = Scalar::RealQ<Scal>       ? CHOLMOD_REAL   : CHOLMOD_COMPLEX;
            static constexpr int xdtype = xtype + dtype;
            
            static_assert(
                SameQ<Int_,Int32> || SameQ<Int_,Int64>,
                "CHOLMOD::CholeskyDecomposition supports only type Int from {Int32, Int64}."
            );
            
            using Int = Int_;
            
            static_assert(
                SameQ<LInt_,long> || SameQ<LInt_,long long> || SameQ<LInt_,Int32> || SameQ<LInt_,Int64>,
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
            
            Tensor1<LInt,Int> rp;
            
        public:
            
            template<typename Int1, typename LInt1, typename Int2>
            CholeskyDecomposition( cptr<LInt1> rp_, cptr<Int1> ci_, const Int2 n_ )
            : rp {static_cast<Int>(n_+1)}
            {
                ptic(ClassName()+"()");
                       
                int ver [3];
                
                if constexpr( long_version )
                {
                    cholmod_l_start(&c);
                    
                    cholmod_l_version(ver);
                }
                else
                {
                    cholmod_start(&c);
                    
                    cholmod_version(ver) ;
                }
                
                
                CheckCommonStructure();
                
                print("CHOLMOD version = "
                    + ToString(ver[0]) + "."
                    + ToString(ver[1]) + "."
                    + ToString(ver[2]) + ".");
                
                dump(itype == CHOLMOD_LONG);
                dump(xtype == CHOLMOD_REAL);
                dump(dtype == CHOLMOD_DOUBLE);

                
                c.print = 5 ; // set level of verbosity.
                c.nmethods = 0 ;
//                c.method[0].ordering = CHOLMOD_AMD;
                c.postorder = true;
                
                dump(c.nmethods);
                dump(c.postorder);
                
                // TODO: Make this more flexible.
                // Input matrix is expected to be symmetric and pos-def,
                // but we expect that both upper and lower triangle are populated.

                const Size_T n   = static_cast<Size_T>(n_);


                rp[0] = 0;
                
                for( Int i = 0; i < n_; ++i )
                {

                    
                    const LInt k_begin = static_cast<LInt>(rp_[i  ]);
                    const LInt k_end   = static_cast<LInt>(rp_[i+1]);

                    LInt k = k_begin;
                    
                    // Find the diagonal entry.
                    while( k < k_end )
                    {
                        const Int j = ci_[k];
                        
                        if(i <= j)
                        {
                            break;
                        }
                        ++k;
                    }
                    
                    if( k < k_end )
                    {
                        const Int j = ci_[k];
                        
                        if( i != j )
                        {
                            eprint( ClassName() + ": Input matrix is missing a diagonal element in row " + ToString(i) +".");
                            dump(k);
                            dump(k_end);
                            dump(j);
                        }
                        
                        const LInt row_nz = k_end - k;
                        
                        rp[i+1] = rp[i] + row_nz;
                    }
                    else
                    {
                        rp[i+1] = rp[i];
                    }
                }
                
                dump(rp.Size());
                dump(rp.data()[n-1]);
                dump(rp.data()[n]);
                
                dump(rp.Last());
                
                const Size_T nnz = static_cast<Size_T>(rp.Last());
                

                
                ptic(ClassName()+": Allocating A");
                if constexpr( long_version  )
                {
                    A = cholmod_l_allocate_sparse(
                        n, n, nnz,
                        true, // Assume CSR-compliant sorting.
                        true, // Assume packed.
                        1,    // Matrix must be symmetric.
                        xdtype,
                        &c
                    );
                }
                else
                {
                    A = cholmod_allocate_sparse(
                        n, n, nnz,
                        true, // Assume CSR-compliant sorting.
                        true, // Assume packed.
                        1,    // Matrix must be symmetric.
                        xdtype,
                        &c
                    );
                }
                
                if( A == nullptr )
                {
                    eprint(ClassName()+": failed to allocate A.");
                    return;
                }
                ptoc(ClassName()+": Allocating A");
                
                ptic(ClassName()+": Copying pattern of A");
                
                mptr<LInt> A_p = reinterpret_cast<LInt*>(A->p);
                mptr< Int> A_i = reinterpret_cast< Int*>(A->i);
                
                rp.Write(A_p);
                rp.Read(rp_);
                
                for( Int i = 0; i < n_; ++i )
                {
                    const LInt k_begin = A_p[i  ];
                    const LInt k_end   = A_p[i+1];
                    const LInt k_count = k_end - k_begin;
                    
                    copy_buffer( &ci_[rp[i+1]-k_count], &A_i[A_p[i]], k_count );
                }
                                
                ptoc(ClassName()+": Copying pattern of A");
                
                ptoc(ClassName()+"()");
            }
            
            ~CholeskyDecomposition()
            {
                ptic(std::string("~")+ClassName()+"()");
                     
                if constexpr ( long_version )
                {
                    if( L != nullptr )
                    {
                        cholmod_l_free_factor(&L,&c);
                        L = nullptr;
                    }
                    
                    if( A != nullptr )
                    {
                        cholmod_l_free_sparse(&A,&c);
                        A = nullptr;
                    }
                    
                    cholmod_l_finish(&c);
                }
                else
                {
                    if( L != nullptr )
                    {
                        cholmod_free_factor(&L,&c);
                        L = nullptr;
                    }
                    
                    if( A != nullptr )
                    {
                        cholmod_free_sparse(&A,&c);
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
                        cholmod_l_free_factor(&L,&c);
                        L = nullptr;
                    }
                    
                    L = cholmod_l_analyze(A,&c);
                }
                else
                {
                    if ( L != nullptr )
                    {
                        cholmod_free_factor(&L,&c);
                        L = nullptr;
                    }
                    
                    L = cholmod_analyze(A,&c);
                }
                
                // DEBUGGING
                CheckFactor();
                
                dump(c.supernodal       );
                dump(c.fl               );
                dump(c.current          );
                dump(c.lnz              );
                dump(c.memory_usage     );
                
                dump(L->minor);
                
                print( ArrayToString( static_cast<LInt*>(A->p), {A->nrow+1} ) );
                      
                print( ArrayToString( static_cast<Int*>(A->i), {static_cast<LInt*>(A->p)[A->nrow]}) );
                
                ptoc(tag);
            }
            
            template<typename ExtScal>
            void NumericFactorization( cptr<ExtScal> A_val )
            {
                std::string tag = ClassName()+"::NumericFactorization";
                
                ptic(tag);

                ptic(tag+": copy_buffer (matrix entries)");
                cptr<LInt> A_p = reinterpret_cast<LInt*>(A->p);
                mptr<Scal> A_x = reinterpret_cast<Scal*>(A->x);
                
                const Int n = static_cast<Int>(A->nrow);
                
                for( Int i = 0; i < n; ++i )
                {
                    const LInt k_begin = A_p[i  ];
                    const LInt k_end   = A_p[i+1];
                    const LInt k_count = k_end - k_begin;
                    
                    copy_buffer( &A_val[rp[i+1]-k_count], &A_x[A_p[i]], k_count );
                }
                
                ptoc(tag+": copy_buffer (matrix entries)");
                
                // DEBUGGING
                CheckSparseMatrix();
                
                
                if constexpr ( long_version )
                {
                    ptic(tag+": cholmod_l_factorize");
                    cholmod_l_factorize(A,L,&c);
                    ptoc(tag+": cholmod_l_factorize");
                }
                else
                {
                    ptic(tag+": cholmod_factorize");
                    cholmod_factorize(A,L,&c);
                    ptoc(tag+": cholmod_factorize");
                }
                
                // DEBUGGING
                CheckFactor();
                
                dump(L->minor);
                
                print( ArrayToString( static_cast<LInt*>(A->p), {A->nrow+1} ) );
                      
                print( ArrayToString( static_cast< Int*>(A->i), {static_cast<LInt*>(A->p)[A->nrow]}) );
                
                print( ArrayToString( static_cast<Scal*>(A->x), {static_cast<LInt*>(A->p)[A->nrow]}) );
                
                
                cholmod_print_factor(L, "L", &c);
                ptoc(tag);
            }
            
            template<typename ExtScal>
            void Solve( cptr<ExtScal> b, mptr<ExtScal> x )
            {
                std::string tag = ClassName()+"::Solve";
                
                ptic(tag);
                
                if constexpr ( long_version )
                {
                    ptic(tag+": cholmod_l_allocate_dense");
                    cholmod_dense * b_ = cholmod_l_allocate_dense(
                        A->nrow, Int(1), A->nrow, xdtype, &c
                    );
                    ptoc(tag+": cholmod_l_allocate_dense");
                    
                    ptic(tag+": copy_buffer (input)");
                    copy_buffer( b, reinterpret_cast<Scal*>(b_->x), b_->nrow );
                    ptoc(tag+": copy_buffer (input)");
                    
                    ptic(tag+": cholmod_l_solve");
                    cholmod_dense * x_ = cholmod_l_solve(CHOLMOD_A,L,b_,&c);
                    ptoc(tag+": cholmod_l_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + ": failed to solve the system.");
                    }
                    else
                    {
                        ptic(tag+": copy_buffer (output)");
                        copy_buffer( reinterpret_cast<Scal*>(x_->x), x, x_->nrow );
                        ptoc(tag+": copy_buffer (output)");
                    }
                    
                    ptic(tag+": cholmod_l_free_dense");
                    cholmod_l_free_dense(&b_,&c);
                    cholmod_l_free_dense(&x_,&c);
                    ptoc(tag+": cholmod_l_free_dense");
                }
                else
                {
                    
                    ptic(tag+": cholmod_allocate_dense");
                    cholmod_dense * b_ = cholmod_allocate_dense(
                        A->nrow, Int(1), A->nrow, xdtype, &c
                    );
                    ptoc(tag+": cholmod_allocate_dense");
                    
                    ptic(tag+": copy_buffer (input)");
                    copy_buffer( b, reinterpret_cast<Scal*>(b_->x), A->nrow );
                    ptoc(tag+": copy_buffer (input)");
                    
                    ptic(tag+": cholmod_solve");
                    cholmod_dense * x_ = cholmod_solve(CHOLMOD_A,L,b_,&c);
                    ptoc(tag+": cholmod_solve");
                    
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + ": failed to solve the system.");
                    }
                    else
                    {
                        ptic(tag+": copy_buffer (output)");
                        copy_buffer( reinterpret_cast<Scal*>(x_->x), x, x_->nrow );
                        ptoc(tag+": copy_buffer (output)");
                    }
                    
                    ptic(tag+": cholmod_free_dense");
                    cholmod_free_dense(&b_,&c);
                    cholmod_free_dense(&x_,&c);
                    ptoc(tag+": cholmod_free_dense");
                }
                
                ptoc(tag);
            }
            
//            template<typename ExtScal>
//            void Solve( cptr<ExtScal> b, mptr<ExtScal> x )
//            {
//                Solve( b, x, Size_T(1) );
//            }
            
            template<typename ExtScal>
            void Solve( cptr<ExtScal> b, mptr<ExtScal> x, const Size_T nrhs )
            {
                std::string tag = ClassName()+"::Solve ( " + ToString(nrhs) +" )";
                
                ptic(tag);
                
                if constexpr ( long_version )
                {
                    ptic(tag+": cholmod_l_allocate_dense");
                    cholmod_dense * b_ = cholmod_l_allocate_dense(
                        A->nrow, nrhs, A->nrow, xdtype,&c
                    );
                    ptoc(tag+": cholmod_l_allocate_dense");
                    
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
                    
                    ptic(tag+": cholmod_l_solve");
                    cholmod_dense * x_ = cholmod_l_solve(CHOLMOD_A, L, b_,&c);
                    ptoc(tag+": cholmod_l_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + ": failed to solve the system.");
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
                    ptic(tag+": cholmod_l_free_dense");
                    cholmod_l_free_dense(&b_,&c);
                    cholmod_l_free_dense(&x_,&c);
                    ptoc(tag+": cholmod_l_free_dense");
                }
                else
                {
                    ptic(tag+": cholmod_allocate_dense");
                    cholmod_dense * b_ = cholmod_allocate_dense(
                        A->nrow, nrhs, A->nrow, xdtype,&c
                    );
                    ptoc(tag+": cholmod_allocate_dense");
                    
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
                    
                    ptic(tag+": cholmod_solve");
                    cholmod_dense * x_ = cholmod_solve(CHOLMOD_A, L, b_,&c);
                    ptoc(tag+": cholmod_solve");
                    
                    if( x_ == nullptr )
                    {
                        eprint(tag + ": failed to solve the system.");
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
                    ptic(tag+": cholmod_free_dense");
                    cholmod_free_dense(&b_,&c);
                    cholmod_free_dense(&x_,&c);
                    ptoc(tag+": cholmod_free_dense");
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
            
            void CheckCommonStructure()
            {
                const int stat = long_version
                               ? cholmod_l_check_common(&c)
                               : cholmod_check_common(&c);
                
                valprint( "CheckCommonStructure", stat == 1 ? "SUCCEEDED" : "FAILED");
            }
            
            void CheckSparseMatrix()
            {
                int stat = long_version
                         ? cholmod_l_check_sparse(A,&c)
                         : cholmod_check_sparse  (A,&c);
                
                valprint( "CheckSparseMatrix", stat == 1 ? "SUCCEEDED" : "FAILED");
                
//                int option = 2;
//                
//                if constexpr ( long_version )
//                {
//                    Int64 xmatched;
//                    Int64 pmatched;
//                    Int64 nzoffdiag;
//                    Int64 nzdiag;
//
//                    stat = cholmod_l_symmetry(A,option,&xmatched,&pmatched,&nzoffdiag,&nzdiag,&c);
//                    
//                    valprint( "matrix symmetry", stat );
//                    
//                    dump(xmatched);
//                    dump(pmatched);
//                    dump(nzoffdiag);
//                    dump(nzdiag);
//                }
//                else
//                {                    
//                    Int32 xmatched;
//                    Int32 pmatched;
//                    Int32 nzoffdiag;
//                    Int32 nzdiag;
//                    
//                    stat = cholmod_symmetry(A,option,&xmatched,&pmatched,&nzoffdiag,&nzdiag,&c);
//                    
//                    valprint( "matrix symmetry", stat );
//                    
//                    dump(xmatched);
//                    dump(pmatched);
//                    dump(nzoffdiag);
//                    dump(nzdiag);
//                }
                        
            }
            
            void CheckFactor()
            {
                const int stat = long_version
                               ? cholmod_l_check_factor(L,&c)
                               : cholmod_check_factor(L,&c);
                
                valprint( "CheckFactor", stat == 1 ? "SUCCEEDED" : "FAILED");
            }
            
            std::string ClassName() const
            {
                return std::string("CHOLMOD::CholeskyDecomposition")+"<"+ TypeName<Scal> + "," + TypeName<Int> + "," +TypeName<LInt> + ">";
            }
            
            
        }; // class CholeskyDecomposition
            
    } // namespace CHOLMOD
    
} // namespace Tensors

