#pragma once

#define CLASS DenseSquareBlockKernel_BLAS
#define BASE  SquareBlockKernel<SIZE_,RHS_COUNT_,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE_, int RHS_COUNT_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
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
        
        
        static constexpr Int BLOCK_NNZ = ROWS * COLS;
        
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
            return BLOCK_NNZ;
        };
        
        
        virtual void TransposeBlock( const Int from, const Int to ) const override
        {
            const Scalar * restrict const a_from = &A[ BLOCK_NNZ * from];
                  Scalar * restrict const a_to   = &A[ BLOCK_NNZ * to  ];
            
            for( Int j = 0; j < COLS; ++j )
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    a_to[ROWS * j + i ] = a_from[COLS * i + j ];
                }
            }
        }
        
        virtual void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            // Caution!!! Should only work correctly if Scalar == Scalar_in == double!
            if constexpr ( RHS_COUNT == 1 )
            {
                cblas_dgemv( CblasRowMajor, CblasNoTrans,
                    ROWS, COLS,
                    1.0, &A_const[BLOCK_NNZ * block_id],     COLS,
                         &X[COLS * j_global],                1,
                    1.0, &z[0][0],                           1
                );
            }
            else
            {
                cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    ROWS, RHS_COUNT, COLS,
                    1.0, &A_const[BLOCK_NNZ * block_id],     COLS,
                         &X[COLS_SIZE * j_global],           RHS_COUNT,
                    1.0, &z[0][0],                           RHS_COUNT
                );
            }
        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+ToString(RHS_COUNT)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Tensors

#undef BASE
#undef CLASS


