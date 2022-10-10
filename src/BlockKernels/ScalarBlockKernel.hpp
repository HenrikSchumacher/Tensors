#pragma once

#define CLASS ScalarBlockKernel
#define BASE  SquareBlockKernel<SIZE_,RHS_COUNT_,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE_, int RHS_COUNT_, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_ >
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_in  = Scalar_in_;
        using Scalar_out = Scalar_out_;
        
        static constexpr Int NONZERO_COUNT = 1;
        
        using BASE::SIZE;
        using BASE::RHS_COUNT;
        using BASE::ROWS;
        using BASE::COLS;
        using BASE::ROWS_SIZE;
        using BASE::COLS_SIZE;
        
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
        };
        
        virtual force_inline void TransposeBlock( const Int from, const Int to ) const override
        {
            A[to] = A[from];
        }
        
        virtual force_inline void ApplyBlock( const Int block_id, const Int j_global ) override
        {
            const Scalar a = A_const[block_id];
            
            Scalar x[COLS][RHS_COUNT];
            
            copy_cast_buffer( &X[COLS_SIZE * j_global], &x[0][0], COLS_SIZE );
            
            for( Int i = 0; i < COLS; ++i )
            {
                for( Int j = 0; j < RHS_COUNT; ++j )
                {
                    z[i][j] += a * x[i][j];
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


