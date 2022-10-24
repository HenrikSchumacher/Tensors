#pragma once

#define CLASS DenseSquareBlockKernel
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
        
        alignas(ALIGNMENT) Scalar x [COLS][RHS_COUNT];
        alignas(ALIGNMENT) Scalar a [ROWS][COLS];
        
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
        }
        
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
            
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            copy_cast_buffer( &X[COLS_SIZE * j_global], &x[0][0], COLS_SIZE );
            
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            copy_buffer( &A_const[BLOCK_NNZ * block_id], &a[0][0], BLOCK_NNZ );
  

//            for( Int k = 0; k < RHS_COUNT; ++k )
//            {
//                for( Int i = 0; i < ROWS; ++i )
//                {
//                    for( Int j = 0; j < COLS; ++j )
//                    {
////                        z[k][i] += a[i][j] * x[j][k];
//                        z[i][k] = std::fma( a[i][j], x[j][k], z[i][k] );
//                    }
//                }
//            }
//
            for( Int k = 0; k < RHS_COUNT; ++k )
            {
                for( Int i = 0; i < ROWS; ++i )
                {
                    for( Int j = 0; j < COLS; ++j )
                    {
//                        z[i][k] += a[i][j] * x[j][k]; 
                        z[i][k] = std::fma( a[i][j], x[j][k], z[i][k] );
                    }
                }
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


