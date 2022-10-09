#pragma once

#define CLASS SquareBlockKernel
#define CLASS BlockKernel<ROWS,ROWS,Scalar_,Int_,Scalar_in_,Scalar_out_>

namespace Repulsor
{
    template<int SIZE, typename Scalar_, typename Int_, typename Scalar_in_, typename Scalar_out_>
    class CLASS
    {
    public:

        using Scalar     = Scalar_;
        using Int        = Int_;
        using Scalar_in  = Scalar_in_;
        using Scalar_out = Scalar_out_;

        CLASS() = delete;
        
        CLASS(
            const Scalar * restrict const a_,
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
        
        virtual Int NonzeroCount() const override = 0
        
        virtual void TransposeBlock( const Int from, const Int to ) = 0;
        
        virtual void ApplyBlock( const Int k, const Int j ) override = 0;
        
    public:
        
        virtual std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+ToString(SIZE)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+","+TypeName<Scalar_in>::Get()+","+TypeName<Scalar_out>::Get()+">";
        }

    };

} // namespace Repulsor

#undef BASE
#undef CLASS


