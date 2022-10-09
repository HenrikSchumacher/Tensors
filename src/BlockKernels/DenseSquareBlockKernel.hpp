#pragma once

#define CLASS DenseSquareBlockKernel
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
        
        ~CLASS() = default;
        
    public:
        
        static constexpr Int NonzeroCount()
        {
            return NONZERO_COUNT;
        }
        
        force_inline void TransposeBlock( const Int from, const Int to ) const
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
        
        force_inline void ApplyBlock( const Int block_id, const Int j_global )
        {
            alignas(ALIGNMENT) Scalar x [ SIZE ];
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            copy_cast_buffer( &X[SIZE * j_global], &x[0], SIZE );
            
            // It's a bit mysterious to me why copying to a local array makes this run a couple of percents faster.
            // Probably the copy has to be done anyways and this way the compiler has better guarantees.
            alignas(ALIGNMENT) Scalar a [SIZE][SIZE];
            
            copy_buffer( &A_const[NONZERO_COUNT * block_id], &a[0][0], SIZE*SIZE );
            
            for( Int i = 0; i < SIZE; ++i )
            {
                for( Int j = 0; j < SIZE; ++j )
                {
                    z[i] += a[i][j] * x[j];
                }
            }
        }
        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Tensors

#undef BASE
#undef CLASS


