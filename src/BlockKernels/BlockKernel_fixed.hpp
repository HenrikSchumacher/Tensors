#pragma once

#define CLASS BlockKernel_fixed

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
        int alpha_flag, int beta_flag,
        bool x_RM, bool x_intRM, bool x_copy, bool x_prefetch,
        bool y_RM, bool y_intRM,
        bool use_fma
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
        ,   alpha( 0 )
        ,   X( nullptr )
        ,   beta( 0 )
        ,   Y( nullptr )
        {}

        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_,
                  Int                         rhs_count_
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
            return RhsCount();
        }
        
        virtual Int NonzeroCount() const = 0;
        
        virtual void TransposeBlock( const Int from, const Int to ) const = 0;
        
        
        
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
        
        
        force_inline void FMA( const Scalar a, const Scalar b, Scalar & c ) const
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
        
//        force_inline void FMA( const Scalar a, const Scalar b, Scalar & c )
//        {
//            c += a * b;
//        }
        
//        force_inline Scalar FMA( const Scalar a, const Scalar b, const Scalar c ) const
//        {
//           return std::fma(a,b,c);
//        }
        
//        force_inline Scalar FMA( const Scalar a, const Scalar b, const Scalar c ) const
//        {
//            return a * b + c;
//        }
        
        force_inline void ReadX( const Int j_global )
        {
            if constexpr ( x_copy )
            {
                // CAUTION: Shadowing global variable here!
                const Scalar_in * restrict const x_from = &X[ColsSize() * j_global];
                
                if constexpr ( x_RM )
                {
                    if constexpr ( x_intRM )
                    {
                        if constexpr ( fixed )
                        {
                            copy_cast_buffer( x_from, &x[0][0], ColsSize() );
                        }
                        else
                        {
                            UNROLL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                UNROLL
                                for( Int k = 0; k < RhsCount(); ++k )
                                {
                                    x[j][k] = static_cast<Scalar>( x_from[RhsCount()*j+k] );
                                }
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int j = 0; j < COLS; ++j )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                x[k][j] = static_cast<Scalar>( x_from[RhsCount()*j+k] );
                            }
                        }
                    }
                }
                else
                {
                    if constexpr ( x_intRM )
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
                            for( Int j = 0; j < COLS; ++j )
                            {
                                x[j][k] = static_cast<Scalar>( x_from[COLS*k+j] );
                            }
                        }
                    }
                    else
                    {
                        // Here we are allowed to copy the full slice because internal x is column major.
                        copy_cast_buffer( x_from, &x[0][0], ColsSize() );
                    }
                }
            }
            else
            {
                x_from = &X[ColsSize() * j_global];
            }
        }
        

        force_inline void BeginRow( const Int i_global )
        {
            // Store the row index for later use.
//            i_global = i_global_;
            
            // Clear the local vector chunk of the kernel.
            zerofy_buffer( &y[0][0], ROWS_SIZE );           // TODO: Might be inefficient.
            
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
                if constexpr ( fixed )
                {
                    prefetch_range<COLS_SIZE,0,0>( &X[COLS_SIZE * j_next] );
                }
                else
                {
                    prefetch_range<0,0>( &X[cols_size * j_next], cols_size );
                }
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
//                return static_cast<Scalar>( x_from[RhsCount()*j+k];
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
                if constexpr ( x_RM )
                {
                    return static_cast<Scalar>(x_from[RhsCount()*j+k]);
                }
                else
                {
                    return static_cast<Scalar>(x_from[COLS*k+j]);
                }
            }
        }
        
        
        force_inline void WriteY( const Int i_global ) const
        {
            Scalar_out * restrict const y_to = &Y[ RowsSize() * i_global];
            
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
                                copy_cast_buffer( &y[0][0], y_to, RowsSize() );
                            }
                            else
                            {
                                UNROLL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    UNROLL
                                    for( Int k = 0; k < RhsCount(); ++k )
                                    {
                                        y_to[RhsCount()*i+k] = get_cast_y(i,k);
                                    }
                                }
                            }
                        }
                        else
                        {
                            UNROLL
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                UNROLL
                                for( Int k = 0; k < RhsCount(); ++k )
                                {
                                    y_to[RhsCount()*i+k] = get_cast_y(i,k);
                                }
                            }
                        }
                    }
                    else
                    {
                        if constexpr ( fixed || y_intRM )
                        {
                            copy_cast_buffer( &y[0][0], y_to, RhsCount() );
                        }
                        else
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                UNROLL
                                for( Int i = 0; i < ROWS; ++i )
                                {
                                    y_to[ROWS*k+i] = get_cast_y(i,k);
                                }
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        UNROLL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                y_to[RhsCount()*i+k] += get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
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
                        UNROLL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                y_to[RhsCount()*i+k] = get_cast_y(i,k) + beta * y_to[RhsCount()*i+k];
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
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
                    UNROLL
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
                        UNROLL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                y_to[RhsCount()*i+k] = alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
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
                        UNROLL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                y_to[RhsCount()*i+k] += alpha * get_cast_y(i,k);
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
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
                        UNROLL
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            UNROLL
                            for( Int k = 0; k < RhsCount(); ++k )
                            {
                                y_to[RhsCount()*i+k] = alpha * get_cast_y(i,k) + beta * y_to[RhsCount()*i+k];
                            }
                        }
                    }
                    else
                    {
                        UNROLL
                        for( Int k = 0; k < RhsCount(); ++k )
                        {
                            UNROLL
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
            Scalar_out * restrict const y_to = &Y[ RowsSize() * i_global ];
            
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer( y_to, RowsSize() );
            }
            else if constexpr ( beta_flag == 1 )
            {
                // do nothing;
            }
            else
            {
                UNROLL
                for( Int k = 0; k < RowsSize(); ++k )
                {
                    y_to[k] *= beta;
                }
            }
        }
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)+","+ToString(COLS)+","+ToString(RHS_COUNT)+","+ToString(fixed)
            
            +","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()
            +","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()
            
            +","+ToString(alpha_flag)+","+ToString(beta_flag)

            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)+
            +">";
        }
  
    };

} // namespace Tensors

#undef get_z
#undef CLASS

