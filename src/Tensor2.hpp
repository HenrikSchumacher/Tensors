#pragma once

namespace Tensors {

#define TENSOR_T Tensor2

    template <typename Scalar, typename Int>
    class Tensor2
    {

#include "Tensor_Common.hpp"
        
    protected:
        
        std::array<Int,2> dims = {0,0};     // dimensions visible to user
        
    public:
        
        TENSOR_T( const Int d0, const Int d1)
        :   n    { d0 * d1 }
        ,   dims { d0, d1 }
        {
            allocate();
        }
        
        TENSOR_T( const Int d0, const Int d1, const Scalar init )
        :   TENSOR_T( d0, d1 )
        {
            Fill( init );
        }
        
        template<typename S>
        TENSOR_T( const S * a_, const Int d0, const Int d1 )
        :   TENSOR_T( d0, d1 )
        {
            Read(a_);
        }
        
    public:
        
        static constexpr Int Rank()
        {
            return static_cast<Int>(2);
        }
        
        template< typename S>
        void WriteTransposed( S * const restrict b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
            
            for( Int j = 0; j < d_1; ++j )
            {
                for( Int i = 0; i < d_0; ++i )
                {
                    b[d_0 * j + i] = static_cast<S>(a[d_1 * i + j ]);
                }
            }
        }
        
        template< typename S>
        void ReadTransposed( const S * const restrict b )
        {
            const Int d_0 = dims[0];
            const Int d_1 = dims[1];
                        
            for( Int i = 0; i < d_0; ++i )
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    a[d_1 * i + j ] = static_cast<Scalar>(b[ d_0 * j + i ]);
                }
            }
        }
        
        // row-wise Write
        template< typename S>
        void Write( const Int i, S * const b ) const
        {
//            ptic(ClassName()+"::Write( const Int i, S * const b )");
            
            copy_cast_buffer( data(i), b, dims[1] );
            
//            ptoc(ClassName()+"::Write( const Int i, S * const b )");
        }
        

        
        // row-wise Read
        template< typename S>
        void Read( const Int i, const S * const b )
        {
//            ptic(ClassName()+"::Read( const Int i, const S * const b )");
            
            copy_cast_buffer( b, data(i), dims[1] );
            
//            ptoc(ClassName()+"::Read( const Int i, const S * const b )");
        }

    private:
        
        void BoundCheck( const Int i ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
        }
        
        void BoundCheck( const Int i, const Int j ) const
        {
            if( (i < 0) || (i > dims[0]) )
            {
                eprint(ClassName()+": first index " + std::to_string(i) + " is out of bounds [ 0, " + std::to_string(dims[0]) +" [.");
            }
            if( (j < 0) || (j > dims[1]) )
            {
                eprint(ClassName()+": second index " + std::to_string(j) + " is out of bounds [ 0, " + std::to_string(dims[1]) +" [.");
            }
        }
        
    public:

        force_inline Scalar * restrict data( const Int i )
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i * dims[1]];
        }
        
        force_inline const Scalar * restrict data( const Int i ) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return &a[i * dims[1]];
        }
        
        force_inline Scalar & operator()(const Int i, const Int j)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return a[ i * dims[1] + j];
        }
        
        force_inline const Scalar & operator()( const Int i, const Int j) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i,j);
#endif
            return a[i * dims[1] + j];
        }
        
        force_inline Scalar * restrict operator[](const Int i)
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return data(i);
        }
        
        force_inline const Scalar * restrict operator[](const Int i) const
        {
#ifdef TENSORS_BOUND_CHECKS
            BoundCheck(i);
#endif
            return data(i);
        }
        
    public:
        
        inline friend std::ostream & operator<<( std::ostream & s, const TENSOR_T & tensor )
        {
            s << tensor.ToString();
            return s;
        }
        
        std::string ToString( const Int p = 16) const
        {
            
            std::stringstream sout;
            sout.precision(p);
            sout << "{\n";
            if( n > 0 )
            {
                const Int d_0 = dims[0];
                const Int d_1 = dims[1];
                
                if( d_1 > 0 )
                {
                    sout << "\t{ ";
                    
                    sout << this->operator()(0,0);
                }
                
                
                for( Int j = 1; j < d_1; ++j )
                {
                    sout << ", " << this->operator()(0,j);
                }
                
                for( Int i = 1; i < d_0; ++i )
                {
                    sout << " },\n\t{ ";
                    
                    sout << this->operator()(i,0);
                    
                    for( Int j = 1; j < d_1; ++j )
                    {
                        sout << ", " << this->operator()(i,j);
                    }
                }
            }
            sout << " }\n}";
            return sout.str();
        }
        
        void Resize( const Int d_0_, const Int d_1_ )
        {
            const Int d_0 = std::max( static_cast<Int>(0),d_0_);
            const Int d_1 = std::max( static_cast<Int>(0),d_1_);
            
            TENSOR_T b ( d_0, d_1 );
            
            const Int min_d_0 = std::min( b.Dimension(0), dims[0] );
            const Int min_d_1 = std::min( b.Dimension(1), dims[1] );
            
            for( Int i = 0; i < min_d_0; ++i )
            {
                copy_buffer( data(i), b.data(i), min_d_1);
            }
            
            swap( *this, b );
        }
        
        static std::string ClassName()
        {
            return "Tensor2<"+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    }; // Tensor2
    
//    template<typename Int>
//    void GEMV( const double alpha, const Tensor2<double,Int> & A, const CBLAS_TRANSPOSE transA,
//                                         const Tensor1<double,Int> & x,
//                      const double beta,        Tensor1<double,Int> & y )
//    {
//        Int m = A.Dimension(0);
//        Int n = A.Dimension(1);
//        Int k = A.Dimension(transA == CblasNoTrans);
//        Int stride_x = 1;
//        Int stride_y = 1;
//        
//        cblas_dgemv( CblasRowMajor, transA, m, n, alpha, A.data(), k, x.data(), stride_x, beta, y.data(), stride_y );
//    }
//    
//    template<typename Int>
//    void GEMM( const double alpha, const Tensor2<double,Int> & A, const CBLAS_TRANSPOSE transA,
//                                          const Tensor2<double,Int> & B, const CBLAS_TRANSPOSE transB,
//                      const double beta,        Tensor2<double,Int> & C )
//    {
//        Int m = C.Dimension(0);
//        Int n = C.Dimension(1);
//        Int k = A.Dimension(transA == CblasNoTrans);
//
//        cblas_dgemm( CblasRowMajor, transA, transB, m, n, k, alpha, A.data(), m, B.data(), n, beta, C.data(), n );
//    }
//
//    template<typename Int>
//    void Dot ( const Tensor2<double,Int> & A, const CBLAS_TRANSPOSE transA,
//                      const Tensor2<double,Int> & B, const CBLAS_TRANSPOSE transB,
//                            Tensor2<double,Int> & C, bool addTo = false )
//    {
//        double alpha = 1.;
//        Int m = C.Dimension(0);
//        Int n = C.Dimension(1);
//        Int k = A.Dimension(transA == CblasNoTrans);
//
//        cblas_dgemm( CblasRowMajor, transA, transB, m, n, k, alpha, A.data(), k, B.data(), n, addTo, C.data(), n );
//    }

//    template<typename Int>
//    int LinearSolve( const Tensor2<double,Int> & A, const Tensor1<double,Int> & b, Tensor1<double,Int> & x )
//    {
//        Int stride_x = 1;
//        Int n = A.Dimension(0);
//        if( x.Dimension(0) != n )
//        {
//            x = Tensor1<double,Int>(n);
//        }
//        x.Read(b.data());
//        Tensor1<Int,Int> ipiv ( n );
//        Tensor2<double,Int> A1 ( A.data(), n , n );
//
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, stride_x, A1.data(), n, ipiv.data(), x.data(), stride_x );
////        valprint("stat",stat);
//        return stat;
//    }
//    
//    template<typename Int>
//    int LinearSolve( Tensor2<double,Int> & A, Tensor1<double,Int> & b )
//    {
//        // in place variant
//        Int stride_x = 1;
//        Int stride_b = 1;
//        Int n = A.Dimension(0);
//        if( b.Dimension(0) != n )
//        {
//            b = Tensor1<double,Int>(n);
//        }
//        
//        Tensor1<Int,Int> ipiv ( n );
//        
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, stride_x, A.data(), n, ipiv.data(), b.data(), stride_b );
////        valprint("stat",stat);
//        return stat;
//    }
//    
//    template<typename Int>
//    int Inverse( const Tensor2<double,Int> & A, Tensor2<double,Int> & Ainv )
//    {
//        Int n = A.Dimension(0);
//        if( Ainv.Dimension(0) != n || Ainv.Dimension(1) != n )
//        {
//            Ainv = Tensor2<double,Int>(n,n);
//        }
//        Tensor1<Int,Int> ipiv ( n );
//        Tensor2<double,Int> A1 ( A.data(), n , n );
//        
//        double * b = Ainv.data();
//        #pragma omp simd collapse(2) aligned( b : ALIGNMENT )
//        for( Int i = 0; i < n; ++i )
//        {
//            for( Int j = 0; j < n; ++j )
//            {
//                b[ n * i + j ] = static_cast<double>(i==j);
//            }
//        }
//        // LAPACKE_dgesv does not like its 4-th argument to be const.
//        int stat = LAPACKE_dgesv( LAPACK_ROW_MAJOR, n, n, A1.data(), n, ipiv.data(), Ainv.data(), n );
////        valprint("stat",stat);
//        return stat;
//    }

    template<typename Scalar, typename Int, typename S>
    Tensor2<Scalar,Int> ToTensor2( const S * a_, const Int d0, const Int d1, const bool transpose = false )
    {
        Tensor2<Scalar,Int> result ( d0, d1 );

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
    
    
    template<typename Scalar, typename Int>
    Tensor2<Scalar,Int> from_MatrixRef( const mma::TensorRef<mreal> & A )
    {
        return ToTensor2<Scalar,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    
    template<typename Scalar, typename Int>
    Tensor2<Scalar,Int> from_MatrixRef( const mma::TensorRef<mint> & A )
    {
        return ToTensor2<Scalar,Int>( A.data(), A.dimensions()[0], A.dimensions()[1] );
    }
    

    template<typename Scalar, typename Int, IsFloat(Scalar)>
    mma::MatrixRef<mreal> to_transposed_MTensorRef( const Tensor2<Scalar,Int> & B )
    {
        Int rows = B.Dimension(0);
        Int cols = B.Dimension(1);
        auto A = mma::makeMatrix<mreal>( cols, rows );

        double * restrict const a_out = A.data();

        #pragma omp parallel for collapse(2)
        for( Int i = 0; i < rows; ++i )
        {
            for( Int j = 0; j < cols; ++j )
            {
                a_out[ rows * j + i] = static_cast<mreal>( B(i,j) );
            }
        }

        return A;
    }
    
    template<typename J, typename Int, IsInt(J)>
    mma::MatrixRef<mint> to_transposed_MTensorRef( const Tensor2<J,Int> & B )
    {
        Int rows = B.Dimension(0);
        Int cols = B.Dimension(1);
        auto A = mma::makeMatrix<mint>( cols, rows );

        double * restrict const a_out = A.data();

        #pragma omp parallel for collapse(2)
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
    template<typename Scalar, typename I1, typename I2, typename I3>
    void Dot(
        const Tensor2<Scalar,I1> & A,
        const Tensor1<Scalar,I2> & x,
              Tensor1<Scalar,I3> & y
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = A.Dimension(0);
        I3 n = std::min(A.Dimension(1),x.Dimension(0));

        if( y.Dimension(0) != m )
        {
            y = Tensor1<Scalar,I3> ( m);
        }
        
        for( I3 i = 0; i < m; ++i )
        {
            Scalar y_i = A(i,0) * x[0];
            
            for( I3 j = 1; j < n; ++j )
            {
                y_i += A(i,j) * x[j];
            }
            
            y[i] = y_i;
        }
    }
    
    // Should be only a fall-back. BLAS is _much_ faster.
    template<typename Scalar, typename I1, typename I2, typename I3>
    void Dot(
        const Tensor1<Scalar,I1> & x,
        const Tensor2<Scalar,I2> & A,
              Tensor1<Scalar,I3> & y
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = std::min(A.Dimension(0),x.Dimension(0));
        I3 n = A.Dimension(1);
        
        if( y.Dimension(0) != n )
        {
            y = Tensor1<Scalar,I3> (n);
        }

        {
            const Scalar x_0 = x[0];
            for( I3 j = 0; j < n; ++j )
            {
                y[j] = x_0 * A(0,j);
            }
        }
        
        for( I3 i = 1; i < m; ++i )
        {
            const Scalar x_i = x[i];
            
            for( I3 j = 0; j < n; ++j )
            {
                y[j] += x_i * A(i,j);
            }
        }
    }


    
#include "Tensor_Common_External.hpp"
    
#undef TENSOR_T
    
    
    
    
#ifdef USE_BLAS
    
    template<typename I1, typename I2, typename I3>
    void Dot(
        const Tensor2<double,I1> & A,
        const Tensor1<double,I2> & x,
              Tensor1<double,I3> & y
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = A.Dimension(0);
        I3 n = std::min(A.Dimension(1),x.Dimension(0));

        if( y.Dimension(0) != m )
        {
            y = Tensor1<double,I3> (m);
        }
        
        cblas_dgemv(CblasRowMajor,CblasNoTrans,m,n,1,A.data(),n,x.data(),1,0,y.data(),1);
    }
    
    template<typename I1, typename I2, typename I3>
    void Dot(
        const Tensor2<float,I1> & A,
        const Tensor1<float,I2> & x,
              Tensor1<float,I3> & y
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = A.Dimension(0);
        I3 n = std::min(A.Dimension(1),x.Dimension(0));

        if( y.Dimension(0) != m )
        {
            y = Tensor1<float,I3> (m);
        }
        
        cblas_sgemv(CblasRowMajor,CblasNoTrans,m,n,1,A.data(),n,x.data(),1,0,y.data(),1);
    }
    
    
    template<typename I1, typename I2, typename I3>
    void Dot(
        const Tensor2<double,I1> & A,
        const Tensor2<double,I2> & B,
              Tensor2<double,I3> & C
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = A.Dimension(0);
        I3 k = std::min(A.Dimension(1),B.Dimension(0));
        I3 n = B.Dimension(1);

        if( (C.Dimension(0) != m) || (C.Dimension(1) != n) )
        {
            C = Tensor2<double,I3> (m,n);
        }
        
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A.data(), k, B.data(), n, 0, C.data(), n);
    }
    
    template<typename I1, typename I2, typename I3>
    void Dot(
        const Tensor2<float,I1> & A,
        const Tensor2<float,I2> & B,
              Tensor2<float,I3> & C
    )
    {
        ASSERT_INT (I1);
        ASSERT_INT (I2);
        ASSERT_INT (I3);
        
        I3 m = A.Dimension(0);
        I3 k = std::min(A.Dimension(1),B.Dimension(0));
        I3 n = B.Dimension(1);

        if( (C.Dimension(0) != m) || (C.Dimension(1) != n) )
        {
            C = Tensor2<float,I3> (m,n);
        }
        
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A.data(), k, B.data(), n, 0, C.data(), n);
    }
#endif
    
} // namespace Tensors
