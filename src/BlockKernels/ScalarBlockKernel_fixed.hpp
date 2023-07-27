#pragma once

#define CLASS ScalarBlockKernel_fixed

#define BASE  BlockKernel_fixed_2<                          \
    ROWS_,COLS_,NRHS_,                                      \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_prefetch,                                       \
    y_RM                                                    \
>

namespace Tensors
{
    template<
        int ROWS_, int COLS_, int NRHS_,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        typename Int_, typename LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool x_RM, bool x_prefetch,
        bool y_RM
    >
    class CLASS : public BASE
    {
        
        static_assert( ROWS_ == COLS_ );
        
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;
        using Int      = Int_;
        using LInt     = LInt_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::NRHS;
        
        static constexpr LInt BLOCK_NNZ = 1;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal> A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int      rhs_count_
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
            A[to] = A[from];
        }
        
        force_inline void ApplyBlock( const LInt k_global, const Int j_global )
        {
            ReadX( j_global );

            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,COLS_SIZE>(
                A_const[k_global], x.data(), Scalar::One<Scal>, y.data()
            );
        }
        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)
            +","+ToString(COLS)
            +","+ToString(NRHS)
            +","+TypeName<Scal>
            +","+TypeName<Scal_in>
            +","+TypeName<Scal_out>
            +","+TypeName<Int>
            +","+TypeName<LInt>
            +","+ToString(alpha_flag)
            +","+ToString(beta_flag)
            +","+ToString(x_RM)+","+ToString(x_prefetch)
            +","+ToString(y_RM)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
