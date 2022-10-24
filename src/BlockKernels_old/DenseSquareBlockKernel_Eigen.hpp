#pragma once

#define CLASS DenseSquareBlockKernel_Eigen
#define BASE  SquareBlockKernel<SIZE_,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_out = Scalar_out_;
        using Scalar_in  = Scalar_in_;
        
        static constexpr Int BLOCK_NNZ = SIZE_ * SIZE_;

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
        }
        
        virtual void TransposeBlock( const Int from, const Int to ) const override
        {
            const Scalar * restrict const a_from = &A[ BLOCK_NNZ * from];
                  Scalar * restrict const a_to   = &A[ BLOCK_NNZ * to  ];
            
            for( Int j = 0; j < SIZE; ++j )
            {
                for( Int i = 0; i < SIZE; ++i )
                {
                    a_to[SIZE * j + i ] = a_from[SIZE * i + j ];
                }
            }
        }
        
        virtual void ApplyBlock( const Int block_id, const Int j_global ) const override
        {
            // Caution!!! Should only work correctly if Scalar == Scalar_in!
            
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            Eigen::Matrix<Scalar,COLS,RHS_COUNT> x ( &X[COLS_SIZE * j_global] );
            
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            Eigen::Matrix<Scalar,ROWS,COLS,Eigen::RowMajor> a ( &A_const[BLOCK_NNZ * block_id] );
            
            Eigen::Map<Eigen::Matrix<Scalar,ROWS,COLS>> z_eigen (&z[0][0]);
            
            z_eigen += a * x;
            
//            for( Int i = 0; i < ROWS; ++i )
//            {
//                for( Int j = 0; j < RHS_COUNT; ++j )
//                {
//                    z[i][j] += z_eigen(i,j);
//                }
//            }
        }
        
    public:
        
        virtual std::string ClassName() const override
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Tensors

#undef BASE
#undef CLASS


