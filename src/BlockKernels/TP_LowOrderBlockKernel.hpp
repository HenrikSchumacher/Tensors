#pragma once

#define CLASS TP_LowOrderBlockKernel
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
        
        static constexpr Int NONZERO_COUNT = 1;
        
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
            A[to] = A[from];
        }
        
        virtual force_inline void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            alignas(ALIGNMENT) Scalar x_0 [RHS_COUNT];
            // Since we need the casted vector ROWS times, it might be a good idea to do the conversion only once.
            copy_cast_buffer( &X[ROWS_SIZE * j_global], &x_0[0], RHS_COUNTs );

            const Scalar a = A_const[block_id];
            
            for( Int k = 0; k < RHS_COUNT; ++k )
            {
                z[0][k] += a * x_0[k];
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


