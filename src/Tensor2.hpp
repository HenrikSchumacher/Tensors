#pragma once

namespace Tensors
{

#define TENSOR_T Tensor2

    template <typename Scal_, typename Int_, Size_T alignment = DefaultAlignment>
    class Tensor2
    {

#include "Tensor_Common.hpp"
        
    private:
        
        std::array<Int,2> dims = {0,0};     // dimensions visible to user
        
    public:
        
        TENSOR_T( const Int d0, const Int d1)
        :   n    { d0 * d1 }
        ,   dims { d0, d1 }
        {
            allocate();
        }
        
        TENSOR_T( const Int d0, const Int d1, cref<Scal> init )
        :   TENSOR_T( d0, d1 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( cptr<S> a_, const Int d0, const Int d1 )
        :   TENSOR_T( d0, d1 )
        {
            Read(a_);
        }
        
    public:
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(2);
        }
        
        
        template<bool row_first = false, typename S>
        void WriteTransposed( mptr<S> b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( row_first )
            {
                for( Int i = 0; i < d_0; ++i )
                {
                    for(Int j = 0; j < d_1; ++j )
                    {
                        b[d_0 * j + i] = static_cast<S>( a[d_1 * i + j ] );
                    }
                }
            }
            else
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    for( Int i = 0; i < d_0; ++i )
                    {
                        b[d_0 * j + i] = static_cast<S>( a[d_1 * i + j ] );
                    }
                }
            }
        }
        
        template<bool row_first = false, typename S>
        void ReadTransposed( cptr<S> b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
              
            if constexpr ( row_first )
            {
                for( Int i = 0; i < d_0; ++i )
                {
                    for(Int j = 0; j < d_1; ++j )
                    {
                        a[d_1 * i + j ] = static_cast<Scal>( b[ d_0 * j + i ] );
                    }
                }
            }
            else
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    for( Int i = 0; i < d_0; ++i )
                    {
                        a[d_1 * i + j ] = static_cast<Scal>( b[ d_0 * j + i ] );
                    }
                }
            }
        }
        
        // row-wise Write
        template< typename S>
        void Write( const Int i, mptr<S> b ) const
        {
            copy_buffer( data(i), b, dims[1] );
        }
        
        // row-wise Read
        template< typename S>
        void Read( const Int i, cptr<S> b )
        {
            copy_buffer( b, data(i), dims[1] );
        }
        
        
        
        template<Op op = Op::Id, typename S>
        void Read( cptr<S> B, const Int ld_B, const Int thread_count = 1 ) const
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( op == Op::Id )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        copy_buffer( &B[ld_B * i], &a[d_1 * i], d_1 );
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Conj )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        for( Int j = 0; j < d_1; ++j )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(Conj(B[ld_B * i + j]));
                        }
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Trans )
            {
                ParallelDo(
                    [=]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(B[ld_B * j + i]);
                        }
                    },
                    d_1, thread_count
                );
            }
            else if constexpr ( op == Op::ConjTrans )
            {
                ParallelDo(
                    [=]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            a[d_1 * i + j] = scalar_cast<Scal>(Conj(B[ld_B * j + i]));
                        }
                    },
                    d_1, thread_count
                );
            }
            else
            {
                eprint(ClassName()+"::Write: No implementation for op available.");
            }
        }
        
//        template<typename S>
//        void Read( cptr<S> a_, const Int lda, const Int thread_count = 1 )
//        {
//            const Int d_0 = dims[0];
//            const Int d_1 = dims[1];
//            
//            ParallelDo(
//                [a_,lda,d_1,this]( const Int i )
//                {
//                    copy_buffer( &a_[lda * i], &a[d_1 * i], d_1 );
//                },
//                d_0, thread_count
//            );
//        }
//        
//        template<typename S>
//        void Write( mptr<S> a_, const Int lda, const Int thread_count = 1 )
//        {
//            const Int d_0 = dims[0];
//            const Int d_1 = dims[1];
//            
//            ParallelDo(
//                [=,this]( const Int i )
//                {
//                    copy_buffer( &a[d_1 * i], &a_[lda * i], d_1 );
//                },
//                d_0, thread_count
//            );
//        }
        
        template<Op op = Op::Id, typename S>
        void Write( mptr<S> B, const Int ld_B, const Int thread_count = 1 ) const
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            if constexpr ( op == Op::Id )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        copy_buffer( &a[d_1 * i], &B[ld_B * i], d_1 );
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Conj )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        for( Int j = 0; j < d_1; ++j )
                        {
                            B[ld_B * i + j] = scalar_cast<S>(Conj(a[d_1 * i + j]));
                        }
                    },
                    d_0, thread_count
                );
            }
            else if constexpr ( op == Op::Trans )
            {
                ParallelDo(
                    [=]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            B[ld_B * j + i] = scalar_cast<S>(a[d_1 * i + j]);
                        }
                    },
                    d_1, thread_count
                );
            }
            else if constexpr ( op == Op::ConjTrans )
            {
                ParallelDo(
                    [=]( const Int j )
                    {
                        for( Int i = 0; i < d_0; ++i )
                        {
                            B[ld_B * j + i] = scalar_cast<S>(Conj(a[d_1 * i + j]));
                        }
                    },
                    d_1, thread_count
                );
            }
            else
            {
                eprint(ClassName()+"::Write: No implementation for op available.");
            }
        }

    private:
        
        void BoundCheck( const Int i ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
#else
            (void)i;
#endif
        }
        
        void BoundCheck( const Int i, const Int j ) const
        {
#ifdef TOOLS_DEBUG
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
            
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(dims[1]) +" [.");
            }
#else
            (void)i;
            (void)j;
#endif
        }
        
    public:

        force_inline mptr<Scal> data( const Int i )
        {
            BoundCheck(i);
            
            return &a[i * dims[1]];
        }
        
        force_inline cptr<Scal> data( const Int i ) const
        {
            BoundCheck(i);
            
            return &a[i * dims[1]];
        }
        
        force_inline mref<Scal> operator()(const Int i, const Int j)
        {
            BoundCheck(i,j);
            
            return a[ i * dims[1] + j];
        }
        
        force_inline cref<Scal> operator()( const Int i, const Int j) const
        {
            BoundCheck(i,j);
            
            return a[i * dims[1] + j];
        }
        
        force_inline mptr<Scal> operator[](const Int i)
        {
            BoundCheck(i);
            
            return data(i);
        }
        
        force_inline cptr<Scal> operator[](const Int i) const
        {
            BoundCheck(i);
            
            return data(i);
        }
        
    public:
        
        void Resize( const Int d_0_, const Int d_1_, bool copy = true )
        {
//            ptic(ClassName() + "::Resize(" + ToString(d_0_) + "," + ToString(d_1_) + ")");
            
            const Int d_0 = Tools::Ramp(d_0_);
            const Int d_1 = Tools::Ramp(d_1_);
            
            TENSOR_T b ( d_0, d_1 );
            
            if( copy )
            {
                const Int min_d_0 = Tools::Min( b.Dimension(0), dims[0] );
                const Int min_d_1 = Tools::Min( b.Dimension(1), dims[1] );
                
                for( Int i = 0; i < min_d_0; ++i )
                {
                    copy_buffer( data(i), b.data(i), min_d_1);
                }
                
            }
            
            swap( *this, b );
                
            
//            ptoc(ClassName()+"::Resize(" + ToString(d_0_) + "," + ToString(d_1_) + ")");
        }
        
        void RequireSize( const Int d_0, const Int d_1, bool copy = false )
        {
            if( dims[0] < d_0 || dims[1] < d_1 )
            {
                Resize(d_0,d_1,copy);
            }
        }
        
        static std::string ClassName()
        {
            return std::string("Tensor2<")+TypeName<Scal>+","+TypeName<Int>+","+ToString(alignment)+">";
        }
        
    }; // Tensor2

    template<typename Scal, typename Int, typename S>
    Tensor2<Scal,Int> ToTensor2( cptr<S> a_, const Int d0, const Int d1, const bool transpose = false )
    {
        Tensor2<Scal,Int> result ( d0, d1 );

        if( transpose )
        {
            result.ReadTransposed(a_);
        }
        else
        {
            result.Read(a_);
        }
        
        return result;
    }
    
#ifdef LTEMPLATE_H
    
    
    template<typename Scal, typename Int>
    Tensor2<Scal,Int> from_MatrixRef( cref<mma::TensorRef<mreal>> A )
    {
        return ToTensor2<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    
    template<typename Scal, typename Int>
    Tensor2<Scal,Int> from_MatrixRef( cref<mma::TensorRef<mint>> A )
    {
        return ToTensor2<Scal,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    

    template<typename Scal, typename Int, 
        class = typename std::enable_if_t<FloatQ<Scal>>
    >
    mma::MatrixRef<mreal> to_transposed_MTensorRef( cref<Tensor2<Scal,Int>> B )
    {
        Int rows = B.Dimension(0);
        Int cols = B.Dimension(1);
        auto A = mma::makeMatrix<mreal>( cols, rows );

        cptr<double> a_out = A.data();

        for( Int i = 0; i < rows; ++i )
        {
            for( Int j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mreal>( B(i,j) );
            }
        }

        return A;
    }
    
    template<typename J, typename Int>
    mma::MatrixRef<mint> to_transposed_MTensorRef( cref<Tensor2<J,Int>> B )
    {
        static_assert(IntQ<J>,"");
        
        Int rows = B.Dimension(0);
        Int cols = B.Dimension(1);
        auto A = mma::makeMatrix<mint>( cols, rows );

        cptr<double> a_out = A.data();

        for( Int i = 0; i < rows; ++i )
        {
            for( Int j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mint>( B(i,j) );
            }
        }

        return A;
    }

    
#endif

    // Should be only a fall-back. BLAS is _much_ faster.
    template<typename Scal, typename I1, typename I2, typename I3>
    void Dot(
        cref<Tensor2<Scal,I1>> A,
        cref<Tensor1<Scal,I2>> x,
        mref<Tensor1<Scal,I3>> y
    )
    {
        // TODO: Use BLAS if available.
        
        I3 m = A.Dimension(0);
        I3 n = Min(A.Dimension(1),x.Dimension(0));

        if( y.Dimension(0) != m )
        {
            y = Tensor1<Scal,I3> ( m);
        }
        
        for( I3 i = 0; i < m; ++i )
        {
            Scal y_i = A(i,0) * x[0];
            
            for( I3 j = 1; j < n; ++j )
            {
                y_i += A(i,j) * x[j];
            }
            
            y[i] = y_i;
        }
    }
    
    // Should be only a fall-back. BLAS is _much_ faster.
    template<typename Scal, typename I1, typename I2, typename I3>
    void Dot(
        cref<Tensor1<Scal,I1>> x,
        cref<Tensor2<Scal,I2>> A,
        mref<Tensor1<Scal,I3>> y
    )
    {
        // TODO: Use BLAS if available.
        
        static_assert(IntQ<I1>,"");
        static_assert(IntQ<I2>,"");
        static_assert(IntQ<I3>,"");
        
        I3 m = Min(A.Dimension(0),x.Dimension(0));
        I3 n = A.Dimension(1);
        
        if( y.Dimension(0) != n )
        {
            y = Tensor1<Scal,I3> (n);
        }

        {
            const Scal x_0 = x[0];
            for( I3 j = 0; j < n; ++j )
            {
                y[j] = x_0 * A(0,j);
            }
        }
        
        for( I3 i = 1; i < m; ++i )
        {
            const Scal x_i = x[i];
            
            for( I3 j = 0; j < n; ++j )
            {
                y[j] += x_i * A(i,j);
            }
        }
    }


    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
    
} // namespace Tensors
