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
        
        static constexpr bool fixed = true;
        
        static constexpr Int MAX_RHS_COUNT = NRHS_;
        static constexpr Int NRHS = NRHS_;
        static constexpr Int ROWS = ROWS_;
        static constexpr Int COLS = COLS_;
        static constexpr Int ROWS_SIZE = ROWS_ * NRHS_;
        static constexpr Int COLS_SIZE = COLS_ * NRHS_;
        
        static constexpr Scalar::Flag Zero    = Scalar::Flag::Zero;
        static constexpr Scalar::Flag Plus    = Scalar::Flag::Plus;
        static constexpr Scalar::Flag Generic = Scalar::Flag::Generic;
        
        static constexpr bool vecQ = VectorizableQ<Scal_out>;
        
        
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
        using z_T = Scal_out;
        
        // Use vectorization for real types if possible.
        mutable std::conditional_t<vecQ, std::array<vec_T<NRHS,y_T>,COLS>, Tiny::Matrix<COLS,NRHS,x_T,Int> > x;
        mutable std::conditional_t<vecQ, std::array<vec_T<NRHS,y_T>,ROWS>, Tiny::Matrix<ROWS,NRHS,y_T,Int> > y;
        
        mutable std::conditional_t<
            vecQ && SameQ<y_T,z_T>,
            std::array<vec_T<NRHS,z_T>,ROWS>,
            Tiny::Matrix<ROWS,NRHS,z_T,Int>
        > z;
        
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
                return NRHS;
            }
            else
            {
                return rhs_count;
            }
        }
        
        force_inline void ReadX( const Int j_global ) const
        {
            cptr<Scal_in> x_from = &X[COLS_SIZE * j_global];
            
            if constexpr ( x_RM )
            {
                
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int k = 0; k < NRHS; ++k )
                    {
                        x[j][k] = static_cast<x_T>( x_from[NRHS*j+k] );
                    }
                }
            }
            else
            {
                // Transpose.
                for( Int k = 0; k < NRHS; ++k )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        x[j][k] = static_cast<x_T>( x_from[COLS*k+j] );
                    }
                }
            }
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
            // Clear the local vector chunk of the kernel.
            if constexpr ( vecQ )
            {
                for( Int j = 0; j < COLS; ++j )
                {
//                    zerofy_buffer<NRHS>( reinterpret_cast<y_T *>(&y[j]) );
                    
                    y[j] = Scalar::Zero<y_T>;
                }
            }
            else
            {
                y.SetZero();
            }
        }
        
    private:
        
        force_inline void ReadZ( const Int i_global ) const
        {
            cptr<Scal_out> y_from = &Y[ROWS_SIZE * i_global];

            if constexpr ( y_RM )
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < NRHS; ++j )
                    {
                        z[i][j] = static_cast<z_T>(y_from[NRHS * i + j]);
                    }
                }
            }
            else
            {
                for( Int j = 0; j < NRHS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        z[i][j] = static_cast<z_T>(y_from[ROWS *j + i]);
                    }
                }
            }
        }
        
        force_inline void WriteZ( const Int i_global ) const
        {
            mptr<Scal_out> y_to = &Y[ROWS_SIZE * i_global];
            
            if constexpr ( y_RM )
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < NRHS; ++j )
                    {
                        y_to[NRHS * i + j] = static_cast<Scal_out>(z[i][j]);
                    }
                }
            }
            else
            {
                for( Int j = 0; j < NRHS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        y_to[ROWS * j + i] = static_cast<Scal_out>(z[i][j]);
                    }
                }
            }
        }
        
    public:
        
        force_inline void WriteY( const Int i_global ) const
        {
            if constexpr ( vecQ )
            {
                if constexpr ( SameQ<y_T,z_T> )
                {
                    // "Vectorized" reduction at row end.
                    if constexpr ( beta_flag != Zero )
                    {
                        ReadZ( i_global );
                    }
                    
                    if constexpr ( alpha_flag == Zero )
                    {
                        if constexpr ( beta_flag == Zero )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] = Scalar::Zero<z_T>;
//                                z[i] = vec_T<NRHS,z_T>(0);
                            }
                        }
                        else if constexpr ( beta_flag == Plus )
                        {
                            // do nothing;
                        }
                        else
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] *= beta;
                            }
                        }
                    }
                    else if constexpr ( alpha_flag == Plus )
                    {
                        if constexpr ( beta_flag == Zero )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] = y[i];
                            }
                        }
                        else if constexpr ( beta_flag == Plus )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] += y[i];
                            }
                        }
                        else // beta_flag == Generic or beta_flag == Minus
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] = y[i] + beta * z[i];
                            }
                        }
                    }
                    else  // beta_flag == Generic or beta_flag == Minus
                    {
                        if constexpr ( beta_flag == Zero )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] = alpha * y[i];
                            }
                        }
                        else if constexpr ( beta_flag == Plus )
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] += alpha * y[i];
                            }
                        }
                        else // beta_flag == Generic or beta_flag == Minus
                        {
                            for( Int i = 0; i < ROWS; ++i )
                            {
                                z[i] = alpha * y[i] + beta * z[i];
                            }
                        }
                    }
                    
                    WriteZ( i_global );
                }
                else
                {
                    // Using "unvectorized" combine_buffers because that allows us to use
                    // a Real type T if both Scal_in and Scal are real, even if Scal_out is complex.
                    
                    // Since this reduction happens only at the ends of rows (and thus not that
                    // often), vectorization would not yield that much bang for the buck.
                    
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        combine_buffers<alpha_flag,beta_flag,NRHS>( alpha, reinterpret_cast<y_T *>(&y[i]), beta, &Y[ROWS_SIZE * i_global] );
                    }
                }
            }
            else
            {
                combine_buffers<alpha_flag,beta_flag,ROWS_SIZE>( alpha, y.data(), beta, &Y[ROWS_SIZE * i_global] );
            }
        }
        
        force_inline void WriteYZero( const Int i_global ) const
        {
            // CAUTION! We cannot use i_global here because BeginRow() has not been in an empty row!
            mptr<Scal_out> y_to = &Y[ ROWS_SIZE * i_global ];
            
            if constexpr ( beta_flag == Zero )
            {
                zerofy_buffer<ROWS_SIZE>( y_to );
            }
            else if constexpr ( beta_flag == Plus )
            {
                // do nothing;
            }
            else
            {
                scale_buffer<ROWS_SIZE>( y_to );
            }
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

