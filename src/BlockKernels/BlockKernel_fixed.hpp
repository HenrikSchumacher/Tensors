#pragma once

#define CLASS BlockKernel_fixed

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        typename Int_, typename LInt_,
        int alpha_flag, int beta_flag,
        bool x_RM, bool x_intRM, bool x_copy, bool x_prefetch,
        bool y_RM, bool y_intRM,
        bool use_fma
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
        using Scal_in  = Scal_in_;
        using Scal_out = Scal_out_;
        using Int      = Int_;
        using LInt     = LInt_;
        
        static constexpr Int RHS_COUNT = RHS_COUNT_;
        static constexpr Int MAX_RHS_COUNT = RHS_COUNT_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int ROWS_SIZE = ROWS_ * RHS_COUNT_;
        static constexpr Int COLS_SIZE = COLS_ * RHS_COUNT_;
        
    protected:
        
        mptr<Scal>     A       = nullptr;
        cptr<Scal>     A_const = nullptr;
        const Scal_out alpha   = 0;
        cptr<Scal_in>  X       = nullptr;
        const Scal_out beta    = 0;
        mptr<Scal_out> Y       = nullptr;

        
        const Scal_in  * restrict x_from = nullptr;
//              Scal_out * restrict y_to   = nullptr;
        
        
        Tiny::Matrix<x_intRM ? COLS : RHS_COUNT, x_intRM ? RHS_COUNT : COLS, Scal,Int> x;
        Tiny::Matrix<y_intRM ? ROWS : RHS_COUNT, y_intRM ? RHS_COUNT : ROWS, Scal,Int> y;

        const Int rhs_count = 1;
        const Int rows_size = ROWS;
        const Int cols_size = COLS;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mptr<Scal> A_ )
        :   A       ( A_      )
        ,   A_const ( nullptr )
        ,   alpha   ( 0       )
        ,   X       ( nullptr )
        ,   beta    ( 0       )
        ,   Y       ( nullptr )
        {}

        CLASS(
            cptr<Scal> A_,
            const Scal_out alpha_, cptr<Scal_in>  X_,
            const Scal_out beta_,  mptr<Scal_out> Y_,
            Int rhs_count_
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
        {
            if( fixed && (RHS_COUNT != rhs_count) )
            {
                eprint(ClassName()+"rhs_count != RHS_COUNT");
            }
            
            if( rhs_count > RHS_COUNT)
            {
                eprint(ClassName()+"rhs_count > RHS_COUNT");
            }
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   A          ( other.A          )
        ,   A_const    ( other.A_const    )
        ,   alpha      ( other.alpha      )
        ,   X          ( other.X          )
        ,   beta       ( other.beta       )
        ,   Y          ( other.Y          )
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
        
//        LInt NonzeroCount() const = 0;
        
//        void TransposeBlock( const LInt from, const LInt to ) const = 0;
        
        
        
        force_inline Int ColsSize() const
        {
            if constexpr ( fixed )
            {
                return COLS_SIZE;
            }
            else
            {
                return cols_size;
            }
        }
    
        force_inline Int RowsSize() const
        {
            if constexpr ( fixed )
            {
                return ROWS_SIZE;
            }
            else
            {
                return rows_size;
            }
        }

        force_inline Int RhsCount() const
        {
            if constexpr ( fixed )
            {
                return RHS_COUNT;
            }
            else
            {
                return rhs_count;
            }
        }
        
        
        force_inline void FMA( const Scal a, const Scal b, mref<Scal> c ) const
        {
            if constexpr ( use_fma )
            {
                c = std::fma(a,b,c);
            }
            else
            {
                c += a * b;
            }
        }
        
        force_inline void ReadX( const Int j_global )
        {
            if constexpr ( x_copy )
            {
                // CAUTION: Shadowing global variable here!
                x_from = &X[ColsSize() * j_global];
                
                if constexpr ( x_RM )
                {
                    if constexpr ( x_intRM )
                    {
                        if constexpr ( fixed )
                        {
                            copy_buffer<VarSize,Sequential>( x_from, &x[0][0], ColsSize() );
                        }
                        else
                        {
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                LOOP_UNROLL_FULL
                                for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                {
                                    x[j][k] = static_cast<Scal>( x_from[COND(fixed,RHS_COUNT,rhs_count)*j+k] );
                                }
                            }
                        }
                    }
                    else
                    {
                        LOOP_UNROLL_FULL
                        for( Int j = 0; j < COLS; ++j )
                        {
                            LOOP_UNROLL_FULL
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                x[k][j] = static_cast<Scal>( x_from[COND(fixed,RHS_COUNT,rhs_count)*j+k] );
                            }
                        }
                    }
                }
                else
                {
                    if constexpr ( x_intRM )
                    {
                        LOOP_UNROLL_FULL
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                        {
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                x[j][k] = static_cast<Scal>( x_from[COLS*k+j] );
                            }
                        }
                    }
                    else
                    {
                        // Here we are allowed to copy the full slice because internal x is column major.
                        copy_buffer<VarSize,Sequential>( x_from, &x[0][0], ColsSize() );
                    }
                }
            }
            else
            {
                x_from = &X[ColsSize() * j_global];
            }
        }
        
        force_inline void Prefetch( const LInt k_global, const Int j_next )
        {
            if constexpr ( x_prefetch )
            {
                // X is accessed in an unpredictable way; let's help with a prefetch statement.
                if constexpr ( fixed )
                {
                    prefetch_buffer<COLS_SIZE,0,0>( &X[COLS_SIZE * j_next] );
                }
                else
                {
                    prefetch_buffer<0,0>( &X[cols_size * j_next], cols_size );
                }
            }
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        force_inline Scal_out get_cast_y( const Int i, const Int k ) const
        {
            if constexpr ( y_intRM )
            {
                return static_cast<Scal_out>(y[i][k]);
            }
            else
            {
                return static_cast<Scal_out>(y[k][i]);
            }
        }
        
        force_inline mref<Scal> get_y( const Int i, const Int k )
        {
            if constexpr ( y_intRM )
            {
                return y[i][k];
            }
            else
            {
                return y[k][i];
            }
        }
        
//        force_inline Scal get_cast_x( const Int j, const Int k )
//        {
//            if constexpr ( x_intRM )
//            {
//                return static_cast<Scal>( x_from[COND(fixed,RHS_COUNT,rhs_count)*j+k];
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
                if constexpr ( x_intRM )
                {
                    return x[j][k];
                }
                else
                {
                    return x[k][j];
                }
            }
            else
            {
                if constexpr ( x_RM )
                {
                    return static_cast<Scal>(x_from[COND(fixed,RHS_COUNT,rhs_count)*j+k]);
                }
                else
                {
                    return static_cast<Scal>(x_from[COLS*k+j]);
                }
            }
        }
        
        
        force_inline void CleanseY()
        {
            // Clear the local vector chunk of the kernel.
//            zerofy_buffer<ROWS_SIZE>( &y[0][0] );
            y.SetZero();
        }
        
        force_inline void WriteY( const Int i_global ) const
        {
            mptr<Scal_out> y_to = &Y[ RowsSize() * i_global];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        if constexpr ( y_intRM)
                        {
                            if constexpr ( fixed )
                            {
//                                copy_buffer<ROWS_SIZE>( &y[0][0], y_to );
                                y.Write(y_to);
                            }
                            else
                            {
                                copy_buffer<VarSize,Sequential>( &y[0][0], y_to, RowsSize() );
                            }
                        }
                        else
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                                {
                                    y_to[COND(fixed,RHS_COUNT,rhs_count)*i+k] = get_cast_y(i,k);
                                }
                            }
                        }
                    }
                    else
                    {
                        // y is not row-major
                        
                        if ( y_intRM )
                        {
                            //transpose
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    y_to[ROWS*k+i] = get_cast_y(i,k);
                                }
                            }
                        }
                        else
                        {
                            if constexpr ( fixed )
                            {
                                copy_buffer<RHS_COUNT>( &y[0][0], y_to );
                            }
                            else
                            {
                                copy_buffer<VarSize,Sequential>( &y[0][0], y_to, rhs_count );
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                y_to[COND(fixed,RHS_COUNT,rhs_count)*i+k] += get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
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
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count); ++k )
                            {
                                y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k] = get_cast_y(i,k) + beta * y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
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
                        zerofy_buffer( y_to, RowsSize() );
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
            else // alpha_flag == -1
            {
                // alpha arbitrary;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
                            {
                                y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
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
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
                            {
                                y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
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
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
                            {
                                y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k] = alpha * get_cast_y(i,k) + beta * y_to[COND(fixed,RHS_COUNT,rhs_count)()*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < COND(fixed,RHS_COUNT,rhs_count)(); ++k )
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
            // CAUTION! We cannot use i_global here because BeginRow() has not been in an empty row!
            mptr<Scal_out> y_to = &Y[ RowsSize() * i_global ];
            
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
                LOOP_UNROLL_FULL
                for( Int k = 0; k < RowsSize(); ++k )
                {
                    y_to[k] *= beta;
                }
            }
        }

        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)+","+ToString(COLS)+","+ToString(RHS_COUNT)+","+ToString(fixed)
            
            +","+TypeName<Scal>+","+TypeName<Scal_in>+","+TypeName<Scal_out>
            +","+TypeName<Int>+","+TypeName<LInt>
            +","+ToString(alpha_flag)+","+ToString(beta_flag)

            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)+
            +">";
        }
  
    };

} // namespace Tensors

#undef get_z
#undef CLASS

