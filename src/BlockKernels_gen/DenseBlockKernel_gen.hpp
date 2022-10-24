#pragma once

#define CLASS DenseBlockKernel_gen
#define BASE  BlockKernel_gen<ROWS_,COLS_,MAX_RHS_COUNT_,Scalar_,Int_,Scalar_in_,Scalar_out_,   \
    x_RM,  y_RM,                                                                                \
    alpha_flag, beta_flag                                                                       \
>

////    x_RowMajor,y_RowMajor,
namespace Tensors
{
    template<
        int ROWS_, int COLS_, int MAX_RHS_COUNT_,
        typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
        bool a_RM, bool x_RM, bool y_RM, bool a_internal_RM,
        int method, int loop_order, int alpha_flag, int beta_flag
    >
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_out = Scalar_out_;
        using Scalar_in  = Scalar_in_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::MAX_RHS_COUNT;
        
        static constexpr Int BLOCK_NNZ = ROWS * COLS;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::z;
        using BASE::rhs_count;
        
        alignas(ALIGNMENT) Scalar a [(a_internal_RM)?ROWS:COLS][(a_internal_RM)?COLS:ROWS];
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( Scalar * restrict const A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            const Scalar     * restrict const A_,
            const Scalar     * restrict const A_diag_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_,
            const Int                         rhs_count_
        )
        :   BASE( A_, A_diag_, alpha_, X_, beta_, Y_, rhs_count_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
    public:
        
        virtual Int NonzeroCount() const override
        {
            return BLOCK_NNZ;
        }
                
        virtual void TransposeBlock( const Int from, const Int to ) const override
        {
            const Scalar * restrict const a_from = &A[ BLOCK_NNZ * from];
                  Scalar * restrict const a_to   = &A[ BLOCK_NNZ * to  ];
            
            if constexpr ( a_RM )
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        a_to[ROWS * j + i ] = a_from[COLS * i + j ];
                    }
                }
            }
            else
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        a_to[COLS * i + j] = a_from[ROWS * j + i];
                    }
                }
            }
        }
        
        virtual void ReadMatrix( const Int block_id )
        {
            // Read matrix.
            const Scalar * restrict const a_from = &A_const[BLOCK_NNZ * block_id];
            
            if constexpr ( a_RM )
            {
                if constexpr ( a_internal_RM )
                {
                    copy_buffer( a_from, &a[0][0], BLOCK_NNZ );
                }
                else
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        for( Int j = 0; j < COLS; ++j )
                        {
                            a[j][i] = a_from[COLS*i+j];
                        }
                    }
                }
            }
            else // !a_RM
            {
                if constexpr ( a_internal_RM )
                {
                    
                    for( Int j = 0; j < COLS; ++j )
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            a[i][j] = a_from[ROWS*j+i];
                        }
                    }
                }
                else
                {
                    copy_buffer( a_from, &a[0][0], BLOCK_NNZ );
                }
            }
        }
        
        virtual void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            this->ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            this->ReadA( block_id );
            
            switch( method )
            {
                case 0:
                {
                    // Do nothing.
                    break;
                }
                case 1:
                {
                    switch ( loop_order )
                    {
                        case 0:
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    for( Int k = 0; k < rhs_count; ++k )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 1:
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int k = 0; k < rhs_count; ++k )
                                {
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 2:
                        {
                            
                            for( Int j = 0; j < COLS; ++j )
                            {
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    for( Int k = 0; k < rhs_count; ++k )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 3:
                        {
                            for( Int j = 0; j < COLS; ++j )
                            {
                                for( Int k = 0; k < rhs_count; ++k )
                                {
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 4:
                        {
                            for( Int k = 0; k < rhs_count; ++k )
                            {
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        case 5:
                        {
                            for( Int k = 0; k < rhs_count; ++k )
                            {
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        if constexpr ( a_internal_RM )
                                        {
                                            z[k][i] += a[i][j] * x[k][j];
                                        }
                                        else
                                        {
                                            z[k][i] += a[j][i] * x[k][j];
                                        }
                                    }
                                }
                            }
                            break;
                        }
                    }
                    break;
                }
                default:
                {
                    if constexpr ( std::is_same_v<Scalar,double> )
                    {
                        cblas_dgemm(
                            CblasColMajor,
                            a_internal_RM ? CblasTrans : CblasNoTrans,
                            CblasNoTrans,
                            a_internal_RM ? ROWS : COLS,
                            rhs_count,
                            a_internal_RM ? COLS : ROWS,
                            1.0, &a[0][0], a_internal_RM ? COLS : ROWS,
                                 &x[0][0], COLS,
                            1.0, &z[0][0], ROWS
                        );
                    }
                    else
                    {
                        cblas_sgemm(
                            CblasColMajor,
                            a_internal_RM ? CblasTrans : CblasNoTrans,
                            CblasNoTrans,
                            a_internal_RM ? ROWS : COLS,
                            rhs_count,
                            a_internal_RM ? COLS : ROWS,
                            1.0f, &a[0][0], a_internal_RM ? COLS : ROWS,
                                  &x[0][0], COLS,
                            1.0f, &z[0][0], ROWS
                        );
                    }
                    break;
                }
            }

        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(MAX_RHS_COUNT)
            +","+TypeName<Scalar>::Get()
            +","+TypeName<Int>::Get()
            +","+TypeName<Scalar_in>::Get()
            +","+TypeName<Scalar_out>::Get()
            +","+ToString(a_RM)
            +","+ToString(x_RM)
            +","+ToString(y_RM)
            +","+ToString(method)
            +","+ToString(a_internal_RM)
            +","+ToString(loop_order)
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
