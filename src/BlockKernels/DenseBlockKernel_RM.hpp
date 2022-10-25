#pragma once

#define CLASS DenseBlockKernel_RM
#define BASE  BlockKernel_RM<                               \
    ROWS_,COLS_,                                            \
    Scalar_,Int_,Scalar_in_,Scalar_out_,                    \
    alpha_flag, beta_flag,                                  \
    x_RM, x_copy, x_prefetch,                               \
    y_RM                                                   \
>

////    x_RowMajor,y_RowMajor,
namespace Tensors
{
    // I picked the default values from benchmarks for
    // ROWS_ = 4, COLS_ = 4, RHS_COUNT_ = 3, alpha_flag = 1, beta_flag = 0, and doubles for all floating point types.
    template<
        int ROWS_, int COLS_,
        typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
        int alpha_flag, int beta_flag,
        bool a_RM  = true, bool a_intRM = false,  bool a_copy = true,
        bool x_RM  = true,                        bool x_copy = true,  bool x_prefetch = true,
        bool y_RM  = true,
        int method = 1,
        int loop   = 2
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
        
        using BASE::RowsSize;
        using BASE::ColsSize;
        using BASE::RhsCount;
        
        static constexpr Int BLOCK_NNZ = ROWS * COLS;
        
        // Dummy variable.
        static constexpr Int MAX_RHS_COUNT = 0;
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        using BASE::get_x;
        using BASE::get_y;
        
        const Scalar * restrict a_from = nullptr;
        
        
        alignas(ALIGNMENT) Scalar a [(a_intRM)?ROWS:COLS][(a_intRM)?COLS:ROWS];
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( Scalar * restrict const A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_,
            const Int                         rhs_count_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_, rhs_count_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
    public:
        
        virtual Int NonzeroCount() const override
        {
            return BLOCK_NNZ;
        }
                
        virtual force_inline void TransposeBlock( const Int from, const Int to ) const override
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
        
        force_inline void ReadA( const Int k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                const Scalar * restrict const a_from = &A_const[BLOCK_NNZ * k_global];
                
                if constexpr ( a_RM )
                {
                    if constexpr ( a_intRM )
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
                    if constexpr ( a_intRM )
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
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        force_inline Scalar get_a( const Int i, const Int j ) const
        {
            if constexpr ( a_copy )
            {
                if constexpr ( a_intRM )
                {
                    return a[i][j];
                }
                else
                {
                    return a[j][i];
                }
            }
            else
            {
                if constexpr ( a_RM )
                {
                    return a_from[COLS*i+j];
                }
                else
                {
                    return a_from[ROWS*j+i];
                }
            }
        }
        
        virtual force_inline void begin_row( const Int i_global ) override
        {}
        
        virtual force_inline void end_row( const Int j_global ) override
        {}
        
        virtual force_inline void apply_block( const Int k_global, const Int j_global ) override
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            ReadA( k_global );
            
            switch( method )
            {
                case 0:
                {
                    // Do nothing.
                    break;
                }
                case 1:
                {
                    switch ( loop )
                    {
                        case 0:
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    for( Int k = 0; k < RhsCount(); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 1:
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int k = 0; k < RhsCount(); ++k )
                                {
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
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
                                    for( Int k = 0; k < RhsCount(); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 3:
                        {
                            for( Int j = 0; j < COLS; ++j )
                            {
                                for( Int k = 0; k < RhsCount(); ++k )
                                {
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 4:
                        {
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    for( Int j = 0; j < COLS; ++j )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 5:
                        {
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    for( Int i = 0; i < ROWS; ++i )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
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
//                    if constexpr ( std::is_same_v<Scalar,double> )
//                    {
//                        cblas_dgemm(
//                            CblasColMajor,
//                            a_intRM ? CblasTrans : CblasNoTrans,
//                            CblasNoTrans,
//                            a_intRM ? ROWS : COLS,
//                            RhsCount(),
//                            a_intRM ? COLS : ROWS,
//                            1.0, &a[0][0], a_intRM ? COLS : ROWS,
//                                 &x[0][0], COLS,
//                            1.0, &y[0][0], ROWS
//                        );
//                    }
//                    else
//                    {
//                        cblas_sgemm(
//                            CblasColMajor,
//                            a_intRM ? CblasTrans : CblasNoTrans,
//                            CblasNoTrans,
//                            a_intRM ? ROWS : COLS,
//                            RhsCount(),
//                            a_intRM ? COLS : ROWS,
//                            1.0f, &a[0][0], a_intRM ? COLS : ROWS,
//                                  &x[0][0], COLS,
//                            1.0f, &y[0][0], ROWS
//                        );
//                    }
//                    break;
                    
                }
            }

        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+TypeName<Scalar>::Get()
            +","+TypeName<Int>::Get()
            +","+TypeName<Scalar_in>::Get()
            +","+TypeName<Scalar_out>::Get()
            +","+ToString(a_RM)+","+ToString(a_intRM)+","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_copy)
            +","+ToString(y_RM)
            +","+ToString(method)
            +","+ToString(loop)
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
