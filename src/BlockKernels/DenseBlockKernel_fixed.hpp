#pragma once
#pragma once

#define CLASS DenseBlockKernel_fixed
#define BASE  BlockKernel_fixed_2<                          \
    ROWS_,COLS_,NRHS_,                                      \
    Scal_,Scal_in_,Scal_out_,                               \
    Int_,LInt_,                                             \
    alpha_flag, beta_flag,                                  \
    x_RM, x_prefetch,                                       \
    y_RM                                                    \
>

//template<
//    int ROWS_, int COLS_, int NRHS_,
//    typename Scal_, typename Scal_in_, typename Scal_out_,
//    typename Int_, typename LInt_,
//    int alpha_flag, int beta_flag,
//    bool a_RM  = true, bool a_intRM = false,  bool a_copy = true,
//    bool x_RM  = true, bool x_prefetch = true,
//    bool y_RM  = true
//>

namespace Tensors
{
    // I picked the default values from benchmarks for
    // ROWS_ = 4, COLS_ = 4, NRHS_ = 3, alpha_flag = 1, beta_flag = 0, and doubles for all floating point types.
    template<
        int ROWS_, int COLS_, int NRHS_,
        typename Scal_, typename Scal_in_, typename Scal_out_,
        typename Int_, typename LInt_,
        Scalar::Flag alpha_flag, Scalar::Flag beta_flag,
        bool a_RM, bool a_intRM,   bool a_copy,
        bool x_RM, bool x_prefetch,
        bool y_RM
    >
    class CLASS : public BASE
    {
    public:

        using Scal     = Scal_;
        using Scal_out = Scal_out_;
        using Scal_in  = Scal_in_;

        using Int        = Int_;
        using LInt       = LInt_;
        
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::NRHS;
        
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        
        static constexpr LInt BLOCK_NNZ = ROWS * COLS;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::x;
        using BASE::y;
        
        using BASE::ReadX;
        
        const Scal * restrict a_from = nullptr;
        
        Tiny::Matrix<(a_intRM)?ROWS:COLS,(a_intRM)?COLS:ROWS,Scal,Int> a;
        
    public:
        
        CLASS() = delete;
        
        explicit CLASS( mptr<Scal> A_ )
        :   BASE( A_ )
        {}
        
        CLASS(
            cptr<Scal>     A_,
            cref<Scal_out> alpha_, cptr<Scal_in>  X_,
            cref<Scal_out> beta_,  mptr<Scal_out> Y_,
            const Int      rhs_count_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_, rhs_count_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        ~CLASS()  = default;
        
    public:
        
        LInt NonzeroCount() const
        {
            return BLOCK_NNZ;
        }
                
        force_inline void TransposeBlock( const LInt from, const LInt to ) const
        {
            cptr<Scal> a_from_ = &A[ BLOCK_NNZ * from];
            mptr<Scal> a_to_   = &A[ BLOCK_NNZ * to  ];
            
            if constexpr ( a_RM )
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
                        a_to_[ROWS * j + i ] = a_from_[COLS * i + j ];
                    }
                }
            }
            else
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
                        a_to_[COLS * i + j] = a_from_[ROWS * j + i];
                    }
                }
            }
        }
        
        force_inline void ReadA( const LInt k_global )
        {
            a_from = &A_const[BLOCK_NNZ * k_global];
            
            // Read matrix.
            if constexpr ( a_copy )
            {
                
                
                if constexpr ( a_RM == a_intRM )
                {
                    a.template Read<Op::Id>( a_from );
                }
                else
                {
                    a.template Read<Op::Trans>( a_from );
                }
            }
        }
        
        force_inline Scal get_a( const Int i, const Int j ) const
        {
            if constexpr ( a_copy )
            {
                if constexpr ( a_intRM )
                {
                    return a[i][j];
                }
                else
                {
                    return a[j][i];
                }
            }
            else
            {
                if constexpr ( a_RM )
                {
                    return a_from[COLS*i+j];
                }
                else
                {
                    return a_from[ROWS*j+i];
                }
            }
        }
        
        force_inline void ApplyBlock( const LInt k_global, const Int j_global )
        {
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            ReadX( j_global );
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            
            ReadA( k_global );

            if constexpr ( a_copy )
            {
                Dot<AddTo>(a,x,y);
            }
            else
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    for( Int i = 0; i < ROWS; ++i )
                    {
//                        combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus,NRHS>(
//                            get_a(i,j), &x[j][0], Scalar::One<Scal>, &y[i][0]
//                        );
                        
                        for( Int k = 0; k < NRHS; ++k )
                        {
                            y[i][k] += get_a(i,j) * x[j][k];
                        }
                    }
                }
            }

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
            +","+ToString(a_RM)+","+ToString(a_intRM)+","+ToString(a_copy)
            +","+ToString(x_RM)+","+ToString(x_prefetch)
            +","+ToString(y_RM)
            +">";
        }

    };
} // namespace Tensors

#undef BASE
#undef CLASS
