#pragma once

#define CLASS DenseSquareBlockKernel
#define CLASS SquareBlockKernel<SIZE,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Repulsor
{
    template<int SIZE, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
    class CLASS
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_out = Scalar_out_;
        using Scalar_in  = Scalar_in_;

        constexpr Int NONZERO_COUNT = SIZE * SIZE;
        
        CLASS() = delete;
        
        CLASS(
            const Scalar     * restrict const a_,
        )
        :   BASE(a_)
        {}
        
        CLASS(
            const Scalar     * restrict const A_,
            const Scalar_out                  alpha_
            const Scalar_in  * restrict const X_,
            const Scalar_out                  beta_
            const Scalar_out * restrict const Y_
        )
        :   BASE( A_, alpha_, X_, beta_, Y_ )
        {}
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE(other) {}
        
        virtual ~CLASS() override = default;
     
    protected:

        using BASE::A;
        using BASE::X;
        using BASE::Y;
        using BASE::z;
        
    public:
        
        virtual Int NonzeroCount() const override
        {
            return NONZERO_COUNT;
        };
        
        virtual void TransposeBlock( const Int from, const Int to ) override
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
        
        virtual void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            const Scalar * restrict const a  = &A[NONZERO_COUNT * block_id];
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only one.
            Scalar x [ SIZE ];
            
            copy_cast_buffer( &X[SIZE * j_global], &x[0], COLS );
      
            // TODO: SIMDization or offloading to a BLAS implementation.
            for( Int i = 0; i < SIZE; ++i )
            {
                for( Int j = 0; j < SIZE; ++j )
                {
                    z[i] += a[SIZE * i + j ] * x[j];
                }
            }
        }
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Repulsor

#undef BASE
#undef CLASS


