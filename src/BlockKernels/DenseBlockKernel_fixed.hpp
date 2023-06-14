#pragma once
#pragma once

#define CLASS DenseBlockKernel_fixed
#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,RHS_COUNT_,fixed,                           \
    Scal_,Scal_in_,Scal_out_,                         \
    Int_,LInt_,                                             \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    useFMA                                                  \
>

//template<
//    int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
//    typename Scal_, typename Scal_in_, typename Scal_out_,
//    typename Int_, typename LInt_,
//    int alpha_flag, int beta_flag,
//    bool a_RM  = true, bool a_intRM = false,  bool a_copy = true,
//    bool x_RM  = true, bool x_intRM = false,  bool x_copy = true,  bool x_prefetch = true,
//    bool y_RM  = true, bool y_intRM = false,
//    int method = 1, int loop   = 2,
//    bool useFMA = false
//>

namespace Tensors
{
    // I picked the default values from benchmarks for
    // ROWS_ = 4, COLS_ = 4, RHS_COUNT_ = 3, alpha_flag = 1, beta_flag = 0, and doubles for all floating point types.
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        typename Int_, typename LInt_,
        int alpha_flag, int beta_flag,
        bool a_RM,      bool a_intRM,   bool a_copy,
        bool x_RM,      bool x_intRM,   bool x_copy,    bool x_prefetch,
        bool y_RM,      bool y_intRM,
        int method,     int loop,
        bool useFMA
    >
    class CLASS : public BASE
    {
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;

        using Int        = Int_;
        using LInt       = LInt_;
        
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::RHS_COUNT;
        
        using BASE::RowsSize;
        using BASE::ColsSize;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::RhsCount;
        
        static constexpr LInt BLOCK_NNZ = ROWS * COLS;
        
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
        using BASE::rhs_count;
        
        const Scal * restrict a_from = nullptr;
        
        alignas(ALIGNMENT) Scal a [(a_intRM)?ROWS:COLS][(a_intRM)?COLS:ROWS];
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mut<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            ptr< Scal>     A_,
            const Scal_out alpha_, ptr<Scal_in>   X_,
            const Scal_out beta_,  mut<Scal_out>  Y_,
            const Int      rhs_count_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_, rhs_count_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        ~CLASS()  = default;
        
    public:
        
        LInt NonzeroCount() const
        {
            return BLOCK_NNZ;
        }
                
        force_inline void TransposeBlock( const LInt from, const LInt to ) const
        {
            ptr<Scal> a_from_ = &A[ BLOCK_NNZ * from];
            mut<Scal> a_to_   = &A[ BLOCK_NNZ * to  ];
            
            if constexpr ( a_RM )
            {
                LOOP_UNROLL_FULL
                for( Int j = 0; j < COLS; ++j )
                {
                    LOOP_UNROLL_FULL
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        a_to_[ROWS * j + i ] = a_from_[COLS * i + j ];
                    }
                }
            }
            else
            {
                LOOP_UNROLL_FULL
                for( Int i = 0; i < ROWS; ++i )
                {
                    LOOP_UNROLL_FULL
                    for( Int j = 0; j < COLS; ++j )
                    {
                        a_to_[COLS * i + j] = a_from_[ROWS * j + i];
                    }
                }
            }
        }
        
        force_inline void ReadA( const LInt k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
                
                if constexpr ( a_RM )
                {
                    if constexpr ( a_intRM )
                    {
                        copy_buffer<BLOCK_NNZ>( a_from, &a[0][0] );
                    }
                    else
                    {
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            LOOP_UNROLL_FULL
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
                        LOOP_UNROLL_FULL
                        for( Int j = 0; j < COLS; ++j )
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                a[i][j] = a_from[ROWS*j+i];
                            }
                        }
                    }
                    else
                    {
                        copy_buffer<BLOCK_NNZ>( a_from, &a[0][0] );
                    }
                }
            }
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        force_inline Scal get_a( const Int i, const Int j ) const
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
        
        force_inline void ApplyBlock( const LInt k_global, const Int j_global )
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
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    LOOP_UNROLL_FULL
                                    for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 1:
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                {
                                    LOOP_UNROLL_FULL
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
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                LOOP_UNROLL_FULL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    LOOP_UNROLL_FULL
                                    for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                    {
                                        get_y(i,k) += get_a(i,j) * get_x(j,k);
                                    }
                                }
                            }
                            break;
                        }
                        case 3:
                        {
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                LOOP_UNROLL_FULL
                                for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                {
                                    LOOP_UNROLL_FULL
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
                            LOOP_UNROLL_FULL
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                LOOP_UNROLL_FULL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    LOOP_UNROLL_FULL
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
                            LOOP_UNROLL_FULL
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < COLS; ++j )
                                {
                                    LOOP_UNROLL_FULL
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
//                    if constexpr ( std::is_same_v<Scal,double> )
//                    {
//                        cblas_dgemm(
//                            CblasColMajor,
//                            a_intRM ? CblasTrans : CblasNoTrans,
//                            CblasNoTrans,
//                            a_intRM ? ROWS : COLS,
//                            COND(fixed,RHS_COUNT,rhs_count),
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
//                            COND(fixed,RHS_COUNT,rhs_count),
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
        
        std::string ClassName() const  
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(RHS_COUNT)
            +","+ToString(fixed)
            +","+TypeName<Scal>
            +","+TypeName<Scal_in>
            +","+TypeName<Scal_out>
            +","+TypeName<Int>
            +","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(a_RM)+","+ToString(a_intRM)+","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)
            +","+ToString(method)
            +","+ToString(loop)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
