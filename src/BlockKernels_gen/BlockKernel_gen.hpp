#pragma once

#define CLASS BlockKernel_gen

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int MAX_RHS_COUNT_,
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
        
        static constexpr Int MAX_RHS_COUNT = MAX_RHS_COUNT_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int MAX_ROWS_SIZE = ROWS_ * MAX_RHS_COUNT_;
        static constexpr Int MAX_COLS_SIZE = COLS_ * MAX_RHS_COUNT_;

    protected:
        
              Scalar     * restrict const A       = nullptr;
        const Scalar     * restrict const A_const = nullptr;
        const Scalar     * restrict const A_diag  = nullptr;
        const Scalar_out                  alpha   = 0;
        const Scalar_in  * restrict const X       = nullptr;
        const Scalar_out                  beta    = 0;
              Scalar_out * restrict const Y       = nullptr;
        
        const Int rhs_count = 0;
        const Int rows_size = 0;
        const Int cols_size = 0;
        
        alignas(ALIGNMENT) Scalar y [MAX_RHS_COUNT][ROWS] = {};
        alignas(ALIGNMENT) Scalar x [MAX_RHS_COUNT][COLS] = {};
        
        Int i_global = -1;
        Int j_global = -1;
        Int k_global = -1;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS(
            Scalar * restrict const A_
        )
        :   A         (A_)
        ,   A_const   ( nullptr )
        ,   A_diag    ( nullptr )
        ,   alpha     ( 0 )
        ,   X         ( nullptr )
        ,   beta      ( 0 )
        ,   Y         ( nullptr )
        ,   rhs_count ( 0 )
        ,   rows_size ( 0 )
        ,   cols_size ( 0 )
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
        :   A          ( nullptr )
        ,   A_const    ( A_ )
        ,   A_diag     ( A_diag_ )
        ,   alpha      ( alpha_ )
        ,   X          ( X_ )
        ,   beta       ( beta_ )
        ,   Y          ( Y_ )
        ,   rhs_count  ( rhs_count_       )
        ,   rows_size  ( rhs_count * ROWS )
        ,   cols_size  ( rhs_count * COLS )
        {
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
        ,   rhs_count  ( other.rhs_count  )
        ,   rows_size  ( other.rows_size  )
        ,   cols_size  ( other.cols_size  )
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
            return rhs_count;
        }
        
        virtual Int NonzeroCount() const = 0;
        
        virtual void TransposeBlock( const Int from, const Int to ) const = 0;
        
        void CleanseVector()
        {
            zerofy_buffer( &y[0][0], rows_size );
        }
        
        void ReadX( const Int j_global )
        {
            const Scalar_in * restrict const x_from = &X[cols_size * j_global];
            
            if constexpr ( x_RM )
            {
                for( Int k = 0; k < rhs_count; ++k )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        x[k][j] = static_cast<Scalar>( x_from[rhs_count*j+k] );
                    }
                }
            }
            else
            {
                copy_cast_buffer( x_from, &x[0][0], cols_size );
            }
        }

        void WriteY( const Int i_global_ ) const
        {
            Scalar_out * restrict const y_ = &Y[ rows_size * i_global_];
            
            if constexpr ( alpha_flag == 1 )
            {
                // alpha == 1;
                if constexpr ( beta_flag == 0 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] = static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                    else
                    {
                        copy_cast_buffer( &y[0][0], y_, rows_size );
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] += static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[ROWS*k+i] += static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                }
                else
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] = static_cast<Scalar_out>(y[k][i]) + beta * y_[rhs_count*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[ROWS*k+i] = static_cast<Scalar_out>(y[k][i]) + beta * y_[ROWS*k+i];
                            }
                        }
                    }
                }
            }
            else if constexpr ( alpha_flag == 0 )
            {
                if constexpr ( beta_flag == 0 )
                {
                    zerofy_buffer( y_, rows_size );
                }
                else if constexpr ( beta_flag == 1 )
                {
                    // do nothing;
                }
                else
                {
                    for( Int k = 0; k < rows_size; ++k )
                    {
                        y_[k] *= beta;
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
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] = alpha * static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[ROWS*k+i] = alpha * static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                }
                else if constexpr ( beta_flag == 1 )
                {
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] += alpha * static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[ROWS*k+i] += alpha * static_cast<Scalar_out>(y[k][i]);
                            }
                        }
                    }
                }
                else // beta_flag == -1
                {
                    // general alpha and general beta
                    if constexpr (y_RM)
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[rhs_count*i+k] = alpha * static_cast<Scalar_out>(y[k][i]) + beta * y_[rhs_count*i+k];
                            }
                        }
                    }
                    else
                    {
                        for( Int k = 0; k < rhs_count; ++k )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                y_[ROWS*k+i] = alpha * static_cast<Scalar_out>(y[k][i]) + beta * y_[ROWS*k+i];
                            }
                        }
                    }
                }
            }
        }
        
        void WriteYZero( const Int i_global_ ) const
        {
            Scalar_out * restrict const y_ = &Y[ rows_size * i_global_];
            
            // alpha == 1;
            if constexpr ( beta_flag == 0 )
            {
                zerofy_buffer( y_, rows_size );
            }
            else if constexpr ( beta_flag == 1 )
            {
                // Do nothing.
            }
            else
            {
                for( Int k = 0; k < rows_size; ++k )
                {
                    y_[k] *= beta;
                }
            }
        }
        
        virtual void Prefetch( const Int k_global_, const Int j_next )
        {
            // X is accessed in an unpredictable way; let's help with a prefetch statement.
            prefetch_range<0,0>( &X[cols_size * j_next], cols_size );
            
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        virtual void ApplyBlock( const Int k_global_, const Int j_global_ )
        {
            k_global = k_global_;
            j_global = j_global_;
            
            apply_block();
        }
        
        virtual void apply_block() = 0;
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(MAX_RHS_COUNT)
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
