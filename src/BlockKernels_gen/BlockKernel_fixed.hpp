#pragma once

#define CLASS BlockKernel_fixed

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_,
        typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
        bool x_RM, bool y_RM,
        int alpha_flag, int beta_flag
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

        
        alignas(ALIGNMENT) Scalar z [RHS_COUNT][ROWS] = {};
        alignas(ALIGNMENT) Scalar x [RHS_COUNT][COLS] = {};
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS(
            Scalar * restrict const A_
        )
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
                  Int                         rhs_count
        )
        :   A ( nullptr )
        ,   A_const( A_ )
        ,   alpha( alpha_ )
        ,   X( X_ )
        ,   beta( beta_ )
        ,   Y( Y_ )
        {
            assert( RHS_COUNT == rhs_count );
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   A          ( other.A          )
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
        
        force_inline void CleanseVector()
        {
            zerofy_buffer( &z[0][0], ROWS_SIZE );
        }
        
        force_inline void ReadVector( const Int j_global )
        {
            const Scalar_in * restrict const x_from = &X[COLS_SIZE * j_global];
            
            if constexpr ( x_RM )
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int k = 0; k < RHS_COUNT; ++k )
                    {
                        x[k][j] = static_cast<Scalar>( x_from[RHS_COUNT*j+k] );
                    }
                }
            }
            else
            {
                copy_cast_buffer( x_from, &x[0][0], COLS_SIZE );
            }
        }

        force_inline void WriteVector( const Int i ) const
        {
            Scalar_out * restrict const y  = &Y[ ROWS_SIZE * i];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int i = 0; i < ROWS; ++i )
                        {
                            for( Int k = 0; k < RHS_COUNT; ++k )
                            {
                                y[RHS_COUNT*i+k] = static_cast<Scalar_out>(z[k][i]);
                            }
                        }
                    }
                    else
                    {
                        copy_cast_buffer( &z[0][0], y, ROWS_SIZE );
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
                                y[RHS_COUNT*i+k] += static_cast<Scalar_out>(z[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y[ROWS*k+i] += static_cast<Scalar_out>(z[k][i]);
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
                                y[RHS_COUNT*i+k] = static_cast<Scalar_out>(z[k][i]) + beta * y[RHS_COUNT*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y[ROWS*k+i] = static_cast<Scalar_out>(z[k][i]) + beta * y[ROWS*k+i];
                            }
                        }
                    }
                }
            }
            else if constexpr ( alpha_flag == 0 )
            {
                if constexpr ( beta_flag == 0 )
                {
                    zerofy_buffer( y, ROWS_SIZE );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // do nothing;
                }
                else
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] *= beta;
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
                                y[RHS_COUNT*i+k] = alpha * static_cast<Scalar_out>(z[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y[ROWS*k+i] = alpha * static_cast<Scalar_out>(z[k][i]);
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
                                y[RHS_COUNT*i+k] += alpha * static_cast<Scalar_out>(z[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y[ROWS*k+i] += alpha * static_cast<Scalar_out>(z[k][i]);
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
                                y[RHS_COUNT*i+k] = alpha * static_cast<Scalar_out>(z[k][i]) + beta * y[RHS_COUNT*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < RHS_COUNT; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y[ROWS*k+i] = alpha * static_cast<Scalar_out>(z[k][i]) + beta * y[ROWS*k+i];
                            }
                        }
                    }
                }
            }
        }
        
        force_inline void WriteZero( const Int i ) const
        {
            Scalar_out * restrict const y  = &Y[ ROWS_SIZE * i];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    zerofy_buffer( y, ROWS_SIZE );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // Do nothing.
                }
                else
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] *= beta;
                    }
                }
            }
            else if constexpr ( alpha_flag == 0 )
            {
                if constexpr ( beta_flag == 0 )
                {
                    zerofy_buffer( y, ROWS_SIZE );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // do nothing;
                }
                else
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] *= beta;
                    }
                }
            }
            else // alpha_flag == -1
            {
                // alpha arbitrary;
                if constexpr ( beta_flag == 0 )
                {
                    zerofy_buffer( y, ROWS_SIZE );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // Do nothing.
                }
                else // beta_flag == -1
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] *= beta;
                    }
                }
            }
        }
        
        virtual void ApplyBlock( const Int k, const Int j ) = 0;
        
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

