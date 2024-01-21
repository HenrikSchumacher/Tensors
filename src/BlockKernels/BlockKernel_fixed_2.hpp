#pragma once

#define CLASS BlockKernel_fixed_2

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
    class alignas( ObjectAlignment ) CLASS
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
        
        static constexpr Int MAX_RHS_COUNT = NRHS_;
        static constexpr Int NRHS = NRHS_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int ROWS_SIZE = ROWS_ * NRHS_;
        static constexpr Int COLS_SIZE = COLS_ * NRHS_;
        
        static constexpr Scalar::Flag Zero    = Scalar::Flag::Zero;
        static constexpr Scalar::Flag Plus    = Scalar::Flag::Plus;
        static constexpr Scalar::Flag Generic = Scalar::Flag::Generic;
        
        static constexpr Op opX = x_RM ? Op::Id : Op::Trans;
        static constexpr Op opY = y_RM ? Op::Id : Op::Trans;

        
    protected:
        
        mptr<Scal>     A       = nullptr;
        cptr<Scal>     A_const = nullptr;
        const Scal_out alpha   = 0;
        cptr<Scal_in>  X       = nullptr;
        const Scal_out beta    = 0;
        mptr<Scal_out> Y       = nullptr;
        
        
        using y_T = typename std::conditional_t<
            Scalar::ComplexQ<Scal> || Scalar::ComplexQ<Scal_in>,
            typename Scalar::Complex<Scal>,
            typename Scalar::Real<Scal>
        >;
        
        using x_T = y_T;
        
        mutable Tiny::Matrix<COLS,NRHS,x_T,Int> x;
        mutable Tiny::Matrix<ROWS,NRHS,x_T,Int> y;
        
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
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int      rhs_count_
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
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   A         ( other.A           )
        ,   A_const   ( other.A_const     )
        ,   alpha     ( other.alpha       )
        ,   X         ( other.X           )
        ,   beta      ( other.beta        )
        ,   Y         ( other.Y           )
        ,   rhs_count ( other.rhs_count_  )
        ,   rows_size ( other.rows_size   )
        ,   cols_size ( other.cols_size   )
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
        
        force_inline Int ColsSize() const
        {
            return COLS_SIZE;
        }
    
        force_inline Int RowsSize() const
        {
            return ROWS_SIZE;
        }

        force_inline Int RhsCount() const
        {
            return NRHS;
        }
        
        force_inline void ReadX( const Int j_global ) const
        {
            x.template Read<opX>( &X[COLS_SIZE * j_global] );
        }
        
        force_inline void Prefetch( const LInt k_global, const Int j_next ) const
        {
            if constexpr ( x_prefetch )
            {
                // X is accessed in an unpredictable way; let's help with a prefetch statement.
                prefetch_buffer<COLS_SIZE,0,0>( &X[COLS_SIZE * j_next] );
            }
            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
        }
        
        
        force_inline void CleanseY() const
        {
            // Clear the local m x nrhs chunk of y.
            y.SetZero();
        }
        
        force_inline void WriteY( const Int i_global ) const
        {
            // Clear the local m x nrhs chunk y to destination in Y.
            y.template Write<alpha_flag,beta_flag,opY,Op::Id>(
                alpha, beta, &Y[ ROWS_SIZE * i_global ]
            );
        }
        
        force_inline void WriteYZero( const Int i_global ) const
        {
            // We don't have to transpose, thus we use Op::Id instead of opY.
            y.template Write<Zero,beta_flag,Op::Id,Op::Id>(
                Scal(0), beta, &Y[ ROWS_SIZE * i_global ]
            );
        }

        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"
                +ToString(ROWS)+","+ToString(COLS)+","+ToString(NRHS)
            +","+TypeName<Scal>+","+TypeName<Scal_in>+","+TypeName<Scal_out>
            +","+TypeName<Int>+","+TypeName<LInt>
            +","+ToString(alpha_flag)+","+ToString(beta_flag)

            +","+ToString(x_RM)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+
            +">";
        }
  
    };

} // namespace Tensors

#undef CLASS

