#pragma once

#define CLASS LowRankSquareBlockKernel
#define BASE  SquareBlockKernel<SIZE_,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE_, int RANK_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
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
        
        static constexpr Int RANK = RANK_;
        static constexpr Int NONZERO_COUNT = 2 * SIZE * RANK;
        
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
            // U_in is of size RANK x SIZE
            // V_in is of size SIZE x RANK
            
            const Scalar * restrict const U_in  = &A_const[NONZERO_COUNT * from            ];
            const Scalar * restrict const V_in  = &A_const[NONZERO_COUNT * from + RANK*SIZE];
            
                  Scalar * restrict const U_out = &A_const[NONZERO_COUNT * to              ];
                  Scalar * restrict const V_out = &A_const[NONZERO_COUNT * to   + RANK*SIZE];

            #pragma unroll
            for( Int i = 0; i < RANK; ++i )
            {
                #pragma unroll
                for( Int j = 0; j < SIZE; ++j )
                {
                    U_out[SIZE * i + j] += V_in[SIZE * j + i];
                }
            }
            
            #pragma unroll
            for( Int i = 0; i < SIZE; ++i )
            {
                #pragma unroll
                for( Int j = 0; j < RANK; ++j )
                {
                    V_out[SIZE * i + j] += U_in[SIZE * j + i];
                }
            }
            
        }
        
        force_inline void ApplyBlock( const Int block_id, const Int j_global )
        {
            alignas(ALIGNMENT) Scalar x [ SIZE ];
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            copy_cast_buffer( &X[SIZE * j_global], &x[0], SIZE );
            
            alignas(ALIGNMENT) Scalar w [ RANK ] = {};
            
            alignas(ALIGNMENT) Scalar U [RANK][SIZE];
            alignas(ALIGNMENT) Scalar V [SIZE][RANK];
            
            copy_buffer( &A_const[NONZERO_COUNT * block_id            ], &U[0][0], RANK*SIZE );
            copy_buffer( &A_const[NONZERO_COUNT * block_id + RANK*SIZE], &V[0][0], SIZE*RANK );
            
            #pragma unroll
            for( Int i = 0; i < RANK; ++i )
            {
                #pragma unroll
                for( Int j = 0; j < SIZE; ++j )
                {
                    w[i] += U[i][j] * x[j];
                }
            }

            
            #pragma unroll
            for( Int i = 0; i < SIZE; ++i )
            {
                #pragma unroll
                for( Int j = 0; j < RANK; ++j )
                {
                    z[i] += V[i][j] * w[j];
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


