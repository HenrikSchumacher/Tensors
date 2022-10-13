#pragma once

#define CLASS BlockKernel

namespace Tensors
{
    template<int ROWS_, int COLS_, int RHS_COUNT_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_ >
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
        
    protected:
        
        static constexpr Int ROWS      = ROWS_;
        static constexpr Int COLS      = COLS_;
        static constexpr Int RHS_COUNT = RHS_COUNT_;
        
        static constexpr Int COLS_SIZE = COLS * RHS_COUNT;
        static constexpr Int ROWS_SIZE = ROWS * RHS_COUNT;
        
        alignas(ALIGNMENT) Scalar z[ROWS][RHS_COUNT] = {};
        
              Scalar     * restrict const A       = nullptr;
        const Scalar     * restrict const A_const = nullptr;
        const Scalar_out                  alpha   = 0;
        const Scalar_in  * restrict const X       = nullptr;
        const Scalar_out                  beta    = 0;
              Scalar_out * restrict const Y       = nullptr;

        const int alpha_flag = 0;
        const int beta_flag  = 0;
        
    public:
        
        CLASS() = delete;
        
        CLASS(
            const Scalar     * restrict const A_
        )
        :   A (A_)
        ,   A_const( nullptr )
        ,   alpha( 0 )
        ,   X( nullptr )
        ,   beta( 0 )
        ,   Y( nullptr )
        ,   alpha_flag( 0 )
        ,   beta_flag ( 0 )
        {}

        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_
        )
        :   A ( nullptr )
        ,   A_const( A_ )
        ,   alpha( alpha_ )
        ,   X( X_ )
        ,   beta( beta_ )
        ,   Y( Y_ )
        ,   alpha_flag(
                (alpha == static_cast<Scalar_out>(1)) ? 1 : ((alpha == static_cast<Scalar_out>(0)) ? 0 : -1)
            )
        ,   beta_flag(
                (beta  == static_cast<Scalar_out>(1)) ? 1 : ((beta  == static_cast<Scalar_out>(0)) ? 0 : -1)
            )
        {
//            #pragma omp single
//            {
//                valprint("alpha     ",alpha);
//                valprint("alpha_flag",alpha_flag);
//                valprint("beta      ",beta);
//                valprint("beta_flag ",beta_flag);
//            }
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   A( other.A )
        ,   alpha( other.alpha )
        ,   X( other.X )
        ,   beta( other.beta )
        ,   Y( other.Y )
        ,   alpha_flag( other.alpha_flag )
        ,   beta_flag ( other.beta_flag )
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
        
        static constexpr Int RightHandSideCount()
        {
            return RHS_COUNT;
        }
        
        virtual Int NonzeroCount() const = 0;
        
        force_inline void CleanseVector()
        {
            zerofy_buffer( &z[0][0], ROWS_SIZE );
        }

        
        force_inline void WriteVector( const Int i ) const
        {
                  Scalar_out * restrict const y = &Y[ ROWS_SIZE * i];
            const Scalar     * restrict const z_ = &z[0][0];
            
            if( alpha_flag == 1 )
            {
                // alpha == 1;
                if( beta_flag == 0 )
                {
                    copy_cast_buffer( z_, y, ROWS_SIZE );
                }
                else if( beta_flag == 1 )
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] += static_cast<Scalar_out>(z_[k]);
                    }
                }
                else
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] = static_cast<Scalar_out>(z_[k]) + beta * y[k];
                    }
                }
            }
            else if( alpha_flag == 0 )
            {
                if( beta_flag == 0 )
                {
                    zerofy_buffer( y, ROWS_SIZE );
                }
                else if( beta_flag == 1 )
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
            else
            {
                // alpha arbitrary;
                if( beta_flag == 0 )
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] = alpha * static_cast<Scalar_out>(z_[k]);
                    }
                }
                else if( beta_flag == 1 )
                {
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] += alpha * static_cast<Scalar_out>(z_[k]);
                    }
                }
                else
                {
                    // general alpha and general beta
                    for( Int k = 0; k < ROWS_SIZE; ++k )
                    {
                        y[k] = alpha * static_cast<Scalar_out>(z_[k]) + beta * y[k];
                    }
                }
            }
        }
        
        virtual void ApplyBlock( const Int k, const Int j ) = 0;
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(ROWS)+","+ToString(COLS)+","+ToString(RHS_COUNT)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }
  
    };

} // namespace Tensors


#undef CLASS
