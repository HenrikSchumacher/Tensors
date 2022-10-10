#pragma once

#define CLASS DenseSquareBlockKernel_BLAS
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
        
    protected:

        using BASE::A;
        using BASE::A_const;
        using BASE::X;
        using BASE::Y;
        using BASE::z;
        using BASE::SIZE;

        static constexpr Int NONZERO_COUNT = SIZE * SIZE;
        
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
        };
        
        
        virtual force_inline void TransposeBlock( const Int from, const Int to ) const override
        {
            const Scalar * restrict const a_from = &A[ NONZERO_COUNT * from];
                  Scalar * restrict const a_to   = &A[ NONZERO_COUNT * to  ];
            
            for( Int j = 0; j < SIZE; ++j )
            {
                for( Int i = 0; i < SIZE; ++i )
                {
                    a_to[SIZE * j + i ] = a_from[SIZE * i + j ];
                }
            }
        }
        
        virtual force_inline void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            cblas_dgemv( CblasRowMajor, CblasNoTrans, SIZE, SIZE,
                        1.0, &A_const[NONZERO_COUNT * block_id], SIZE,
                             &X[SIZE * j_global], 1,
                        1.0, &z[0], 1
            );
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


