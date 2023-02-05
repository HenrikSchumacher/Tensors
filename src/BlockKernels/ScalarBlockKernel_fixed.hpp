#pragma once

#define CLASS ScalarBlockKernel_fixed

#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,RHS_COUNT_,fixed,                           \
    Scal_,Scal_in_,Scal_out_,                         \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    use_fma                                                 \
>

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
    class CLASS : public BASE
    {
        
        static_assert( ROWS_ == COLS_ );
        
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;
        using Int        = Int_;
        using LInt       = LInt_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::RHS_COUNT;
        
        using BASE::FMA;
        
        static constexpr LInt BLOCK_NNZ = 1;
        
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
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( Scal * restrict const A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            const Scal     * restrict const A_,
            const Scal_out                  alpha_,
            const Scal_in  * restrict const X_,
            const Scal_out                  beta_,
                  Scal_out * restrict const Y_,
            const Int                         rhs_count_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_, rhs_count_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        ~CLASS() = default;
        
    public:
        
        static constexpr LInt NonzeroCount()
        {
            return BLOCK_NNZ;
        }
                
        force_inline void TransposeBlock( const LInt from, const LInt to ) const
        {
            A[BLOCK_NNZ * to] = A[BLOCK_NNZ * from];
        }
        
        force_inline void ApplyBlock( const LInt k_global, const Int j_global )
        {
            ReadX( j_global );

            const Scal a = A_const[BLOCK_NNZ * k_global];
            
            LOOP_UNROLL_FULL
            for( Int j = 0; j < COLS; ++j )
            {
                LOOP_UNROLL_FULL
                for( Int k = 0; k < RHS_COUNT; ++k )
                {
                    FMA( a, get_x(j,k), get_y(j,k) );
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
            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)
            +","+ToString(use_fma)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
