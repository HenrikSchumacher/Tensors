#pragma once

#define CLASS BlockKernel_fixed

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
        int alpha_flag, int beta_flag,
        bool x_RM, bool x_intRM, bool x_copy, bool x_prefetch,
        bool y_RM, bool y_intRM
    >
    class alignas( OBJECT_ALIGNMENT ) CLASS
    {
        ASSERT_ARITHMETIC(Scalar_)
        ASSERT_INT(Int_);
        ASSERT_ARITHMETIC(Scalar_in_)
        ASSERT_ARITHMETIC(Scalar_out_)
        
    public:
        
        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_in  = Scalar_in_;
        using Scalar_out = Scalar_out_;
        
        static constexpr Int RHS_COUNT = RHS_COUNT_;
        static constexpr Int MAX_RHS_COUNT = RHS_COUNT_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int ROWS_SIZE = ROWS_ * RHS_COUNT_;
        static constexpr Int COLS_SIZE = COLS_ * RHS_COUNT_;
        
    protected:
        
              Scalar     * restrict const A       = nullptr;
        const Scalar     * restrict const A_const = nullptr;
        const Scalar     * restrict const A_diag  = nullptr;
        const Scalar_out                  alpha   = 0;
        const Scalar_in  * restrict const X       = nullptr;
        const Scalar_out                  beta    = 0;
              Scalar_out * restrict const Y       = nullptr;

        
        const Scalar_in  * restrict x_from = nullptr;
//              Scalar_out * restrict y_to   = nullptr;
        
        alignas(ALIGNMENT) Scalar x [x_intRM ? COLS : RHS_COUNT][x_intRM ? RHS_COUNT : COLS] = {};
        alignas(ALIGNMENT) Scalar y [y_intRM ? ROWS : RHS_COUNT][y_intRM ? RHS_COUNT : ROWS] = {};

        const Int rhs_count = 1;
        const Int rows_size = ROWS;
        const Int cols_size = COLS;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( Scalar * restrict const A_ )
        :   A (A_)
        ,   A_const( nullptr )
        ,   A_diag ( nullptr )
        ,   alpha( 0 )
        ,   X( nullptr )
        ,   beta( 0 )
        ,   Y( nullptr )
        {}

        CLASS(
            const Scalar     * restrict const A_,
            const Scalar     * restrict const A_diag_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_,
                  Int                         rhs_count_
        )
        :   A         ( nullptr          )
        ,   A_const   ( A_               )
        ,   A_diag    ( A_diag_          )
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
        ,   A_diag     ( other.A_diag     )
        ,   alpha      ( other.alpha      )
        ,   X          ( other.X          )
        ,   beta       ( other.beta       )
        ,   Y          ( other.Y          )
        {}
        
        virtual ~CLASS() = default;


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
            return RHS_COUNT;
        }
        
        virtual Int NonzeroCount() const = 0;
        
        virtual void TransposeBlock( const Int from, const Int to ) const = 0;
        
        
        
        force_inline Int ColsSize()
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
    
        force_inline Int RowsSize()
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

        force_inline Int RhsCount()
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
        
        
        force_inline void ReadX( const Int j_global )
        {
            if constexpr ( x_copy )
            {
                // CAUTION: Shadowing global variable here!
                const Scalar_in * restrict const x_from = &X[COLS_SIZE * j_global];
                
                if constexpr ( x_RM )
                {
                    if constexpr ( x_intRM )
                    {
                        if constexpr ( fixed )
                        {
                            copy_cast_buffer( x_from, &x[0][0], COLS_SIZE );
                        }
                        else
                        {
                            for( Int j = 0; j < COLS; ++j )
                            {
                                for( Int k = 0; k < RHS_COUNT; ++k )
                                {
                                    x[k][j] = static_cast<Scalar>( x_from[RHS_COUNT*j+k] );
                                }
                            }
                        }
                    }
                    else
                    {
                        for( Int j = 0; j < COLS; ++j )
                        {
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                x[k][j] = static_cast<Scalar>( x_from[RHS_COUNT*j+k] );
                            }
                        }
                    }
                }
                else
                {
                    if constexpr ( x_intRM )
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int j = 0; j < COLS; ++j )
                            {
                                x[j][k] = static_cast<Scalar>( x_from[COLS*k+j] );
                            }
                        }
                    }
                    else
                    {
                        copy_cast_buffer( x_from, &x[0][0], COLS_SIZE );
                    }
                }
            }
            else
            {
                x_from = &X[COLS_SIZE * j_global];
            }
        }
        

        force_inline void BeginRow( const Int i_global )
        {
            // Store the row index for later use.
//            i_global = i_global_;
            
            // Clear the local vector chunk of the kernel.
            zerofy_buffer( &y[0][0], ROWS_SIZE );
            
            // Allow the descendant kernels to do their own thing at row start.
            begin_row( i_global );
        }
        
        virtual force_inline  void begin_row( const Int i_global ) = 0;
        
        
        force_inline void EndRow( const Int i_global )
        {
            // Allow the descendant kernels to do their own thing at row end.
            end_row( i_global );
            
            // Write the Y-slice according to i_global_.
            WriteY( i_global );
        }
        
        virtual force_inline void end_row( const Int i_global ) = 0;
        
        
        virtual force_inline void Prefetch( const Int k_global, const Int j_next )
        {
            if constexpr ( x_prefetch )
            {
                // X is accessed in an unpredictable way; let's help with a prefetch statement.
                prefetch_range<COLS_SIZE,0,0>( &X[COLS_SIZE * j_next] );
            }
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        
        virtual force_inline void ApplyBlock( const Int k_global, const Int j_global )
        {
            apply_block( k_global, j_global );
        }
        
        virtual force_inline void apply_block( const Int k_global, const Int j_global ) = 0;
        
        
        force_inline Scalar_out get_cast_y( const Int i, const Int k ) const
        {
            if constexpr ( y_intRM )
            {
                return static_cast<Scalar_out>(y[i][k]);
            }
            else
            {
                return static_cast<Scalar_out>(y[k][i]);
            }
        }
        
        force_inline Scalar & get_y( const Int i, const Int k )
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
        
//        force_inline Scalar get_cast_x( const Int j, const Int k )
//        {
//            if constexpr ( x_intRM )
//            {
//                return static_cast<Scalar>( x_from[RHS_COUNT*j+k];
//            }
//            else
//            {
//                return static_cast<Scalar>( x_from[COLS*k+j];
//            }
//        }
        
        force_inline Scalar get_x( const Int j, const Int k )
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
                if constexpr ( x_intRM )
                {
                    return static_cast<Scalar>(x_from[RHS_COUNT*j+k]);
                }
                else
                {
                    return static_cast<Scalar>(x_from[COLS*k+j]);
                }
            }
        }
        
        
        force_inline void WriteY( const Int i_global ) const
        {
            Scalar_out * restrict const y_to = &Y[ ROWS_SIZE * i_global];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        if constexpr ( y_intRM)
                        {
                            copy_cast_buffer( &y[0][0], y_to, ROWS_SIZE );
                        }
                        else
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                for( Int k = 0; k < RHS_COUNT; ++k )
                                {
                                    y_to[RHS_COUNT*i+k] = get_cast_y(i,k);
                                }
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y_to[RHS_COUNT*i+k] += get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y_to[RHS_COUNT*i+k] = get_cast_y(i,k) + beta * y_to[RHS_COUNT*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
                    zerofy_buffer( y_to, ROWS_SIZE );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // do nothing;
                }
                else
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
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
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y_to[RHS_COUNT*i+k] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y_to[RHS_COUNT*i+k] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y_to[RHS_COUNT*i+k] = alpha * get_cast_y(i,k) + beta * y_to[RHS_COUNT*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
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
            Scalar_out * restrict const y_to = &Y[ ROWS_SIZE * i_global ];
            
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer( y_to, ROWS_SIZE );
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                for( Int k = 0; k < ROWS_SIZE; ++k )
                {
                    y_to[k] *= beta;
                }
            }
        }
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(RHS_COUNT)
            +","+TypeName<Scalar>::Get()
            +","+TypeName<Int>::Get()
            +","+TypeName<Scalar_in>::Get()
            +","+TypeName<Scalar_out>::Get()
            +","+ToString(x_RM)
            +","+ToString(y_RM)
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +">";
        }
  
    };

} // namespace Tensors

#undef get_z
#undef CLASS

