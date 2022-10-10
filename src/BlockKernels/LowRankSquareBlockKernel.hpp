#pragma once

#define CLASS LowRankSquareBlockKernel
#define BASE  SquareBlockKernel<SIZE_,RHS_COUNT_,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE_, int RANK_, int RHS_COUNT_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_out = Scalar_out_;
        using Scalar_in  = Scalar_in_;

        using BASE::SIZE;
        using BASE::RHS_COUNT;
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        
        static constexpr Int RANK = RANK_;
        static constexpr Int NONZERO_COUNT = 2 * SIZE_ * RANK;
        
    protected:
        
        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::z;
        

    public:
        
        CLASS() = delete;
        
        CLASS(
            const Scalar     * restrict const A_
        )
        :   BASE( A_ )
        {}
        
        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_,
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_,
                  Scalar_out * restrict const Y_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        virtual ~CLASS() override = default;
        
    public:
        
        virtual Int NonzeroCount() const override
        {
            return NONZERO_COUNT;
        }
        
        virtual force_inline void TransposeBlock( const Int from, const Int to ) const override
        {
            // U_in is of size RANK x COLS
            // V_in is of size ROWS x RANK
            
            // U_out is of size RANK x ROWS
            // V_out is of size COLS x RANK
            
            const Scalar * restrict const U_in  = &A[NONZERO_COUNT * from            ];
            const Scalar * restrict const V_in  = &A[NONZERO_COUNT * from + RANK*COLS];
            
                  Scalar * restrict const U_out = &A[NONZERO_COUNT * to              ];
                  Scalar * restrict const V_out = &A[NONZERO_COUNT * to   + RANK*COLS];

            for( Int i = 0; i < RANK; ++i )
            {
                for( Int j = 0; j < ROWS; ++j )
                {
                    U_out[ROWS * i + j] += V_in[RANK * j + i];
                }
            }
            
            for( Int i = 0; i < COLS; ++i )
            {
                for( Int j = 0; j < RANK; ++j )
                {
                    V_out[RANK * i + j] += U_in[COLS * j + i];
                }
            }
            
        }
        
        virtual force_inline void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            alignas(ALIGNMENT) Scalar x [ROWS][RHS_COUNT];
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            copy_cast_buffer( &X[ROWS_SIZE * j_global], &x[0][0], ROWS_SIZE );
            
            alignas(ALIGNMENT) Scalar w [RANK][RHS_COUNT] = {};
            
            alignas(ALIGNMENT) Scalar U [RANK][COLS];
            alignas(ALIGNMENT) Scalar V [ROWS][RANK];
            
            copy_buffer( &A_const[NONZERO_COUNT * block_id            ], &U[0][0], RANK*COLS );
            copy_buffer( &A_const[NONZERO_COUNT * block_id + RANK*COLS], &V[0][0], ROWS*RANK );
            
            for( Int i = 0; i < RANK; ++i )
            {
                for( Int j = 0; j < COLS; ++j )
                {
                    const Scalar U_i_j = U[i][j];
                    
                    for( Int k = 0; k < RHS_COUNT; ++k )
                    {
                        w[i][k] += U_i_j * x[j][k];
                    }
                }
            }

            
            for( Int i = 0; i < ROWS; ++i )
            {
                for( Int j = 0; j < RANK; ++j )
                {
                    const Scalar V_i_j = V[i][j];
                    
                    for( Int k = 0; k < RHS_COUNT; ++k )
                    {
                        z[i][k] += V_i_j * w[j][k];
                    }
                }
            }
        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+ToString(RANK)+","+ToString(RHS_COUNT)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Tensors

#undef BASE
#undef CLASS


