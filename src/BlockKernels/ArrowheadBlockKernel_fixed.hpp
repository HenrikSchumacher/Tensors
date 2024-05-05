#pragma once

#define CLASS ArrowheadBlockKernel_fixed

#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,RHS_COUNT_,fixed,                           \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    use_fma                                                 \
>

//template<
//    int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
//    typename Scal_, typename Int_, typename Scal_in_, typename Scal_out_,
//    Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
//                                              bool a_copy = true,
//    bool x_RM  = true, bool x_intRM = false,  bool x_copy = true,  bool x_prefetch = true,
//    bool y_RM  = true, bool y_intRM = false
//>
namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        typename Int_, typename LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
                                 bool a_copy,
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
        using Int      = Int_;
        using LInt     = LInt_;

        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        using BASE::RHS_COUNT;
        
        using BASE::FMA;
        
        static constexpr LInt BLOCK_NNZ = COLS + ROWS - 1;
        
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
        
        const Scal * restrict a_from = nullptr;
        
        Tiny::Vector<BLOCK_NNZ,Scal,Int> a;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal> A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int rhs_count_
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
            cptr<Scal> a_from_ = &A[BLOCK_NNZ * from];
            mptr<Scal> a_to_   = &A[BLOCK_NNZ * to  ];
            
            a_to_[0] = a_from_[0];
            
            LOOP_UNROLL_FULL
            for( Int i = 1; i < ROWS; ++i )
            {
                a_to_[       i] = a_from_[ROWS-1+i];
                a_to_[ROWS-1+i] = a_from_[       i];
            }
        }
        
        force_inline void ReadA( const LInt k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                a.Read( &A_const[BLOCK_NNZ * k_global] );
            }
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        force_inline Scal get_a( const Int l )
        {
            if constexpr ( a_copy )
            {
                return a[l];
            }
            else
            {
                return a_from[l];
            }
        }
        
        
        force_inline void ApplyBlock( const LInt k_global, const Int j_global )
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            ReadA( k_global );
            
            /*
            //    /                                                                  \
            //    |   get_a(0)          get_a(1)       get_a(2)      get_a(COLS-1)   |
            //    |                                                                  |
            //    |   get_a(COLS)          0              0              0           |
            //    |                                                                  |
            //    |   get_a(COLS+1)        0              0              0           |
            //    |                                                                  |
            //    |   get_a(ROWS+COLS-2)   0              0              0           |
            //    \                                                                  /
            */
            
            LOOP_UNROLL_FULL
            for( Int k = 0; k < RHS_COUNT; ++k )
            {
                FMA( get_a(0), get_x(0,k), get_y(0,k) );

                LOOP_UNROLL_FULL
                for( Int j = 1; j < COLS; ++j )
                {
                    FMA( get_a(j), get_x(j,k), get_y(0,k) );
                }

                LOOP_UNROLL_FULL
                for( Int i = 1; i < ROWS; ++i )
                {
                    FMA( get_a(COLS-1+i), get_x(0,k), get_y(i,k) );
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
                                                     +","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_intRM)+","+ToString(x_copy)+","+ToString(x_prefetch)
            +","+ToString(y_RM)+","+ToString(y_intRM)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS

