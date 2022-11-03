#pragma once

#define CLASS ArrowheadBlockKernel_fixed
#define BASE  BlockKernel_fixed<                            \
    ROWS_,COLS_,RHS_COUNT_,fixed,                           \
    Scalar_,Scalar_in_,Scalar_out_,                         \
    Int_, LInt_,                                            \
    alpha_flag, beta_flag,                                  \
    x_RM, x_intRM, x_copy, x_prefetch,                      \
    y_RM, y_intRM,                                          \
    use_fma                                                 \
>

//template<
//    int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
//    typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_,
//    int alpha_flag, int beta_flag,
//                                              bool a_copy = true,
//    bool x_RM  = true, bool x_intRM = false,  bool x_copy = true,  bool x_prefetch = true,
//    bool y_RM  = true, bool y_intRM = false
//>
namespace Tensors
{
    template<
        int ROWS_, int COLS_, int RHS_COUNT_, bool fixed,
        typename Scalar_, typename Scalar_in_, typename Scalar_out_,
        typename Int_, typename LInt_,
        int alpha_flag, int beta_flag,
                                 bool a_copy,
        bool x_RM, bool x_intRM, bool x_copy, bool x_prefetch,
        bool y_RM, bool y_intRM,
        bool use_fma
    >
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Scalar_out = Scalar_out_;
        using Scalar_in  = Scalar_in_;
        using Int        = Int_;
        using LInt       = LInt_;

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
        
        const Scalar * restrict a_from = nullptr;
        
        Scalar a [BLOCK_NNZ];
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( Scalar * restrict const A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_,
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
            const Scalar * restrict const a_from = &A[BLOCK_NNZ * from];
                  Scalar * restrict const a_to   = &A[BLOCK_NNZ * to  ];
            
            a_to[0] = a_from[0];
            
            #pragma unroll
            for( Int i = 1; i < ROWS; ++i )
            {
                a_to[       i] = a_from[ROWS-1+i];
                a_to[ROWS-1+i] = a_from[       i];
            }
        }
        
        force_inline void ReadA( const LInt k_global )
        {
            // Read matrix.
            if constexpr ( a_copy )
            {
                const Scalar * restrict const a_from = &A_const[BLOCK_NNZ * k_global];
                
                copy_buffer( a_from, &a[0], BLOCK_NNZ );
            }
            else
            {
                a_from = &A_const[BLOCK_NNZ * k_global];
            }
        }
        
        force_inline Scalar get_a( const Int l )
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
            
            //    /                                                                  \
            //    |   get_a(0)          get_a(1)       get_a(2)      get_a(COLS-1)   |
            //    |                                                                  |
            //    |   get_a(COLS)          0              0              0           |
            //    |                                                                  |
            //    |   get_a(COLS+1)        0              0              0           |
            //    |                                                                  |
            //    |   get_a(ROWS+COLS-2)   0              0              0           |
            //    \                                                                  /
            
            #pragma unroll
            for( Int k = 0; k < RHS_COUNT; ++k )
            {
                FMA( get_a(0), get_x(0,k), get_y(0,k) );

                #pragma unroll
                for( Int j = 1; j < COLS; ++j )
                {
                    FMA( get_a(j), get_x(j,k), get_y(0,k) );
                }

                #pragma unroll
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
            +","+TypeName<Scalar>::Get()
            +","+TypeName<Scalar_in>::Get()
            +","+TypeName<Scalar_out>::Get()
            +","+TypeName<Int>::Get()
            +","+TypeName<LInt>::Get()
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
