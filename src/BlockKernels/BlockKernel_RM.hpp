#pragma once

#define CLASS BlockKernel_RM

namespace Tensors
{
    template<
        int ROWS_, int COLS_,
        typename Scal_, typename Scal_in_, typename Scal_out_, typename Int_, typename LInt_,
        int alpha_flag, int beta_flag,
        bool x_RM, bool x_copy, bool x_prefetch,
        bool y_RM
    >
    class alignas( OBJECT_ALIGNMENT ) CLASS
    {
        ASSERT_ARITHMETIC(Scal_)
        ASSERT_ARITHMETIC(Scal_in_)
        ASSERT_ARITHMETIC(Scal_out_)
        ASSERT_INT(Int_);
        ASSERT_INT(LInt_);
    public:
        
        using Scal     = Scal_;
        using Int      = Int_;
        using LInt     = LInt_;
        using Scal_in  = Scal_in_;
        using Scal_out = Scal_out_;
        
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        
    protected:
        
        mut<Scal>      A       = nullptr;
        ptr<Scal>      A_const = nullptr;
        const Scal_out alpha   = 0;
        ptr<Scal_in>   X       = nullptr;
        const Scal_out beta    = 0;
        mut<Scal_out>  Y       = nullptr;

        
        const Scal_in  * restrict x_from = nullptr;
//              Scal_out * restrict y_to   = nullptr;

        
        const Int rhs_count = 1;
        const Int rows_size = ROWS;
        const Int cols_size = COLS;
        
        Tiny::VectorList<COLS,Scal,Int> x;
        Tiny::VectorList<ROWS,Scal,Int> y;

        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mut<Scal> A_ )
        :   A       ( A_      )
        ,   A_const ( nullptr )
        ,   alpha   ( 0       )
        ,   X       ( nullptr )
        ,   beta    ( 0       )
        ,   Y       ( nullptr )
        {}

        CLASS(
            ptr<Scal> A_,
            const Scal_out alpha_, ptr<Scal_in>  X_,
            const Scal_out beta_,  mut<Scal_out> Y_,
            const Int rhs_count_
        )
        :   A         ( nullptr          )
        ,   A_const   ( A_               )
        ,   alpha     ( alpha_           )
        ,   X         ( X_               )
        ,   beta      ( beta_            )
        ,   Y         ( Y_               )
        ,   rhs_count ( rhs_count_       )
        ,   rows_size ( ROWS * rhs_count )
        ,   cols_size ( COLS * rhs_count )
        ,   x         ( rhs_count, 0     )
        ,   y         ( rhs_count, 0     )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   A          ( other.A          )
        ,   A_const    ( other.A_const    )
        ,   alpha      ( other.alpha      )
        ,   X          ( other.X          )
        ,   beta       ( other.beta       )
        ,   Y          ( other.Y          )
        ,   rhs_count  ( other.rhs_count  )
        ,   rows_size  ( other.rows_size  )
        ,   cols_size  ( other.cols_size  )
        ,   x          ( other.x          )
        ,   y          ( other.y          )
        {}
        
        ~CLASS() = default;


    public:
        
        static constexpr Int RowCount()
        {
            return ROWS;
        }
        
        static constexpr Int ColCount()
        {
            return COLS;
        }
        
        Int RightHandSideCount() const
        {
            return RhsCount();
        }
        
        LInt NonzeroCount() const = 0;
        
        void TransposeBlock( const LInt from, const LInt to ) const = 0;
        
        
        
        force_inline Int ColsSize() const
        {
            return cols_size;
        }
    
        force_inline Int RowsSize() const
        {
            return rows_size;
        }

        force_inline Int RhsCount() const
        {
            return rhs_count;
        }
        
        
        force_inline void ReadX( const Int j_global )
        {
            if constexpr ( x_copy )
            {
                // CAUTION: Shadowing global variable here!
                x_from = &X[ColsSize() * j_global];
                
                if constexpr ( x_RM )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        copy_buffer<VarSize,Sequential>( &x_from[RhsCount()*j], x.data(j), ColsSize() );
                    }
                }
                else
                {
                    for( Int k = 0; k < RhsCount(); ++k )
                    {
                        for( Int j = 0; j < COLS; ++j )
                        {
                            x[j][k] = static_cast<Scal>( x_from[COLS*k+j] );
                        }
                    }
                }
            }
            else
            {
                x_from = &X[ColsSize() * j_global];
            }
        }
        

        force_inline void CleanseY()
        {
            // Clear the local vector chunk of the kernel.
            y.SetZero();
        }

        
        force_inline void Prefetch( const LInt k_global, const Int j_next )
        {
            if constexpr ( x_prefetch )
            {
                // X is accessed in an unpredictable way; let's help with a prefetch statement.
                prefetch_range<0,0>( &X[cols_size * j_next], cols_size );
            }
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        force_inline Scal_out get_cast_y( const Int i, const Int k ) const
        {
            return static_cast<Scal_out>(y[i][k]);
        }
        
        force_inline Scal & get_y( const Int i, const Int k )
        {
            return y[i][k];
        }
        
//        force_inline Scal & get_y_to( const Int i, const Int k ) const
//        {
//            if constexpr (y_RM )
//            {
//                return Y[ RowsSize() * i_global + RhsCount()*i+k];
//            }
//            else
//            {
//                return Y[ RowsSize() * i_global + ROWS()*k+i];
//            }
//        }
        
//        force_inline Scal get_cast_x( const Int j, const Int k )
//        {
//            if constexpr ( x_intRM )
//            {
//                return static_cast<Scal>( x_from[RhsCount()*j+k];
//            }
//            else
//            {
//                return static_cast<Scal>( x_from[COLS*k+j];
//            }
//        }
        
        force_inline Scal get_x( const Int j, const Int k )
        {
            if constexpr ( x_copy )
            {
                return x[j][k];
            }
            else
            {
                return static_cast<Scal>(x_from[RhsCount()*j+k]);
            }
        }
        
        
        force_inline void WriteY( const Int i_global ) const
        {
            mut<Scal_out> y_to = &Y[ RowsSize() * i_global];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            copy_buffer<VarSize,Sequential>( y.data(i), &y_to[RhsCount()*i], RhsCount() );
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = get_cast_y(i,k);
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[RhsCount()*i+k] += get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] += get_cast_y(i,k);
                            }
                        }
                    }
                }
                else
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[RhsCount()*i+k] = get_cast_y(i,k) + beta * y_to[RhsCount()*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = get_cast_y(i,k) + beta * y_to[ROWS*k+i];
                            }
                        }
                    }
                }
            }
            else if constexpr ( alpha_flag == 0 )
            {
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr ( fixed )
                    {
                        zerofy_buffer<ROWS_SIZE>( y_to );
                    }
                    else
                    {
                        zerofy_buffer<VarSize,Sequential>( y_to, RowsSize() );
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // do nothing;
                }
                else
                {
                    for( Int k = 0; k < RowsSize(); ++k )
                    {
                        y_to[k] *= beta;
                    }
                }
            }
            else // alpha_flag == -1
            {
                // alpha arbitrary;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                            
                                y_to[RhsCount()*i+k] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[RhsCount()*i+k] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                }
                else // beta_flag == -1
                {
                    // general alpha and general beta
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[RhsCount()*i+k] = alpha * get_cast_y(i,k) + beta * y_to[RhsCount()*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_to[ROWS*k+i] = alpha * get_cast_y(i,k) + beta * y_to[ROWS*k+i];
                            }
                        }
                    }
                }
            }
        }
        
        force_inline void WriteYZero( const Int i_global ) const
        {
            // CAUTION! We cannot use i_global here because BeginRow() has not been called in an empty row!
            mut<Scal_out> y_to = &Y[ RowsSize() * i_global ];
            
            if constexpr ( beta_flag == 0 )
            {
                if constexpr ( fixed )
                {
                    zerofy_buffer<ROWS_SIZE>( y_to );
                }
                else
                {
                    zerofy_buffer<VarSize,Sequential>( y_to, RowsSize() );
                }
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                if constexpr ( fixed )
                {
                    scale_buffer<ROWS_SIZE>( beta, y_to );
                }
                else
                {
                    scale_buffer<VarSize,Sequential>( beta, y_to, RowsSize() );
                }
            }
        }
        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+TypeName<Scal>+","+TypeName<Scal_in>+","+TypeName<Scal_out>
            +","+TypeName<Int>+","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(x_RM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)
            +">";
        }
  
    };

} // namespace Tensors

#undef get_z
#undef CLASS

