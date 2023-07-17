#pragma once

//###########################################################################################
//####          IO routines
//###########################################################################################

public:

    template<typename ExtScal>
    void ReadRightHandSide( cptr<ExtScal> B, Int nrhs_ = ione )
    {
        nrhs = std::max( ione, nrhs_ );
        
        const std::string tag = ClassName() + "::ReadRightHandSide<" + TypeName<ExtScal> + "> (" + ToString(nrhs)+ ")";
        
        ptic(tag);
        
        if( X.Size() < static_cast<LInt>(n) * nrhs )
        {
            X         = VectorContainer_T(static_cast<LInt>(n)*nrhs);
            X_scratch = VectorContainer_T(static_cast<LInt>(max_n_1)*nrhs);
        }

        if ( nrhs == ione )
        {
            perm.Permute( B, X.data(), Inverse::False );
        }
        else
        {
            perm.Permute( B, X.data(), Inverse::False, nrhs );
        }
        
        ptoc(tag);
    }

    template<typename ExtScal>
    void WriteSolution( mptr<ExtScal> X_ )
    {
        const std::string tag = ClassName() + "::WriteSolution<" + TypeName<ExtScal> + "> (" + ToString(nrhs)+ ")";
        
        ptic(tag);
        
        if ( nrhs == ione )
        {
            perm.Permute( X.data(), X_, Inverse::True );
        }
        else
        {
            perm.Permute( X.data(), X_, Inverse::True, nrhs );
        }
        
        ptoc(tag);
    }


//###########################################################################################
//####          Get routines
//###########################################################################################

    Int ThreadCount() const
    {
        return thread_count;
    }

    Int RowCount() const
    {
        return n;
    }

    Int ColCount() const
    {
        return n;
    }

    Int RightHandSideCount() const
    {
        return nrhs;
    }

    //            const Matrix_T & GetL() const
    //            {
    //                return L;
    //            }


    const Matrix_T & GetU() const
    {
        std::string tag ( "U" );
        
        if( !InCacheQ(tag) )
        {
            // TODO: Debug this.
            
            Tensor1<LInt,Int> U_rp (n+1);
            U_rp[0] = 0;
            
            for( Int k = 0; k < SN_count; ++k )
            {
                const Int i_begin  = SN_rp[k  ];
                const Int i_end    = SN_rp[k+1];
                
                const LInt l_begin = SN_outer[k  ];
                const LInt l_end   = SN_outer[k+1];
                
    //                    const Int n_0 = i_end - i_begin;
                const Int n_1 = int_cast<Int>(l_end - l_begin);
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    U_rp[i+1] = U_rp[i] + (i_end-i) + n_1;
                }
            }
            
            Tensor1< Int,LInt> U_ci  (U_rp.Last(),0);
            Tensor1<Scal,LInt> U_val (U_rp.Last(),0);
            
            JobPointers<LInt> job_ptr ( SN_count, U_rp.data(), thread_count, false );
            
            ParallelDo(
                [&,this]( const Int s )
                {
                    const Int i_begin  = SN_rp[s  ];
                    const Int i_end    = SN_rp[s+1];
                    
                    const LInt l_begin = SN_outer[s  ];
                    const LInt l_end   = SN_outer[s+1];
                    
                    const Int n_0 = i_end - i_begin;
                    const Int n_1 = int_cast<Int>(l_end - l_begin);

                    const Int start = U_rp[i_begin];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int delta = i-i_begin;

                        U_ci[start + delta] = i;
                    }
                    
                    
                    cptr<Scal> U_0 = &SN_tri_val[SN_tri_ptr[s]];
                    cptr<Scal> U_1 = &SN_rec_val[SN_rec_ptr[s]];
                    
                    // Copy first row of supernode.
                    copy_buffer<VarSize,Sequential>( &SN_inner[l_begin], &U_ci[start+n_0], n_1 );
                    
                    // Copy triangular part.
                    copy_buffer<VarSize,Sequential>( U_0, &U_val[U_rp[i_begin]      ], n_0 );
                    // Copy rectangular part.
                    copy_buffer<VarSize,Sequential>( U_1, &U_val[U_rp[i_begin] + n_0], n_1 );
                    
                    // Copy the remaining rows of supernode.
                    for( Int i = i_begin+1; i < i_end; ++i )
                    {
                        // We are in the `i-loc`-th row of supernode s.
                        const Int i_loc = i-i_begin;
                        // Row `i_loc` of `U_0` has this many nonzero entries.
                        const Int n_0_i = i_end-i;
                        
                        copy_buffer<VarSize,Sequential>( &U_ci[start+i_loc], &U_ci[U_rp[i]], n_0_i + n_1 );
                        
                        // Copy triangular part.
                        copy_buffer<VarSize,Sequential>(
                            &U_0[n_0 * i_loc + i_loc],
                            &U_val[U_rp[i]],
                            n_0_i
                        );
                        
                        copy_buffer<VarSize,Sequential>(
                            &U_1[n_1 * i_loc],
                            &U_val[U_rp[i] + n_0_i],
                            n_1
                        );
                    }
                },
                job_ptr
            );
            
            this->SetCache( tag,
                std::make_any<Matrix_T>(
                    std::move(U_rp), std::move(U_ci), std::move(U_val),
                    n, n, thread_count
                )
            );
        }
        
        return std::any_cast<Matrix_T &>( this->GetCache(tag) );
    }

    const BinaryMatrix_T & GetA() const
    {
        return A;
    }

    const Permutation<Int> & GetPermutation() const
    {
        return perm;
    }

    const Tensor1<LInt,LInt> & GetValuePermutation() const
    {
        return A_inner_perm;
    }

    Int SN_Count() const
    {
        return SN_count;
    }

    const Tensor1<Int,Int> & SN_RowPointers() const
    {
        return SN_rp;
    }

    const Tensor1<LInt,Int> & SN_Outer() const
    {
        return SN_outer;
    }

    const Tensor1<Int,LInt> & SN_Inner() const
    {
        return SN_inner;
    }

    const Tensor1<LInt,Int> & SN_TrianglePointers() const
    {
        return SN_tri_ptr;
    }

    const Tensor1<Scal,LInt> & SN_TriangleValues() const
    {
        return SN_tri_val;
    }

    const Tensor1<LInt,Int> & SN_RectanglePointers() const
    {
        return SN_rec_ptr;
    }

    const Tensor1<Scal,LInt> & SN_RectangleValues() const
    {
        return SN_rec_val;
    }

    const Tensor1<Int,Int> & RowToSN() const
    {
        return row_to_SN;
    }

    const Tensor1<Scal,LInt> & Values() const
    {
        return A_val;
    }
