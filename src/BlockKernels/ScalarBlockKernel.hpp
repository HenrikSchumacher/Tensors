#pragma once

#define CLASS ScalarBlockKernel
#define BASE  SquareBlockKernel<SIZE,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Tensors
{
    template<int SIZE, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_ >
    class CLASS : public BASE
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_in  = Scalar_in_;
        using Scalar_out = Scalar_out_;

        using BASE::COLS;
        using BASE::ROWS;
        static constexpr Int NONZERO_COUNT = 1;
        
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
     
    protected:

        using BASE::A;
        using BASE::A_const;
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
            A[to] = A[from];
        }
        
        virtual void ApplyBlock( const Int k, const Int j ) override
        {
            const Scalar a_k = A_const[k];
            
            const Scalar_in * restrict x = &X[SIZE * j];
            
            for( Int l = 0; l < SIZE; ++l )
            {
                z[l] += a_k * static_cast<Scalar>(x[l]);
            }
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


