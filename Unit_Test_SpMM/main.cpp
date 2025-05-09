#include <iostream>

#define TOOLS_ENABLE_PROFILER

#include "Tensors.hpp"

using namespace Tools;
using namespace Tensors;

//constexpr Scalar::Flag Generic = Scalar::Flag::Generic;
//constexpr Scalar::Flag Plus    = Scalar::Flag::Plus;
//constexpr Scalar::Flag Minus   = Scalar::Flag::Minus;
//constexpr Scalar::Flag Zero    = Scalar::Flag::Zero;

int error_count  = 0;
int ineff_count  = 0;
int thread_count = 8;

template<typename a_T, typename X_T, typename b_T, typename Y_T, typename Scalar, typename Int, typename LInt>
    __attribute__((noinline)) void Dot_True(
    cptr<LInt> rp, cptr<Int> ci, cptr<Scalar> a, const Int m, const Int n,
    cref<a_T> alpha, cptr<X_T> X, const Int ldX,
    cref<b_T> beta,  mptr<Y_T> Y, const Int ldY,
    const Int   cols
)
{
    std::string tag = std::string("Dot_True<")
        +TypeName<Scalar>+","
        +TypeName<a_T>+","
        +TypeName<X_T>+","
        +TypeName<b_T>+","
        +TypeName<Y_T>+">";
        
    TOOLS_PTIC(tag);
    const Y_T alpha_ = static_cast<Y_T>(alpha);
    const Y_T beta_  = static_cast<Y_T>(beta);
        
    for( Int i = 0; i < m; ++i )
    {
        for( Int l = 0; l < cols; ++l )
        {
            Y[ldY * i + l] *= beta_;
        }
    }
    
    for( Int i = 0; i < m; ++i )
    {
        for( LInt k = rp[i]; k < rp[i+1]; ++k )
        {
            const Int j = ci[k];
            
            for( Int l = 0; l < cols; ++l )
            {
                Y[ldY * i + l] +=
                    alpha_ * (static_cast<Y_T>(a[k]) * static_cast<Y_T>(X[ldX * j + l]));
            }
        }
    }
    TOOLS_PTOC(tag);
}

template<
    typename a_T, typename X_T, typename b_T, typename Y_T,
    typename Scal, typename Int,  typename LInt
>
void test_SpMM( Sparse::MatrixCSR<Scal,Int,LInt> & A, Int cols )
{
    
    const Int  m   = A.RowCount();
    const Int  n   = A.ColCount();
    
    using Real = typename Scalar::Real<Y_T>;
    
    std::string s = std::string("\n    <")
          +TypeName<a_T>+","
          +TypeName<X_T>+","
          +TypeName<b_T>+","
          +TypeName<Y_T>+">";
    logprint(s);
    
    Tensor2<X_T,Int> X      ( n, cols );
    Tensor2<Y_T,Int> Y_0    ( m, cols );
    Tensor2<Y_T,Int> Y      ( m, cols );
    Tensor2<Y_T,Int> Y_True ( m, cols );
    Tensor2<Y_T,Int> Z      ( m, cols );
    
    X.Random( thread_count );
    Y_0.Random( thread_count );

    
    const a_T alpha = 1;
    const b_T beta  = 1;

    Y_True = Y_0;
    auto start_time_1 = Clock::now();
    Dot_True(
        A.Outer().data(), A.Inner().data(), A.Values().data(), m, n,
        alpha, X.data(),      cols,
        beta,  Y_True.data(), cols,
        cols
    );
    double time_1 = Tools::Duration( start_time_1, Clock::now() );
    
    Y = Y_0;
    A.SetThreadCount(1);
    auto start_time_2 = Clock::now();
    A.Dot( alpha, X, beta, Y );
    double time_2 = Tools::Duration( start_time_2, Clock::now() );
    
    Y = Y_0;
    A.SetThreadCount(thread_count);
    auto start_time_3 = Clock::now();
    A.Dot( alpha, X, beta, Y );
    double time_3 = Tools::Duration( start_time_3, Clock::now() );
    
    Real max   = 0;
    Real error = 0;
    
    for( Int i = 0; i < m; ++i )
    {
        for( Int l = 0; l < cols; ++l )
        {
            max    = Max( max,   std::abs(Y_True(i,l)) );
            Z(i,l) = Y(i,l) - Y_True(i,l);
            error  = Max( error, std::abs(Z(i,l)) );
        }
    }
    
    if( max > Real(0) )
    {
        error /= max;
    }
    
    const double seq_speedup = time_1/time_2;
    const double par_speedup = time_2/time_3;
    
    logvalprint("         Time 1     ", time_1 );
    logvalprint("         Time 2     ", time_2 );
    logvalprint("         Time 3     ", time_3 );
    logvalprint("         Seq_speedup", seq_speedup );
    logvalprint("         Par speedup", par_speedup );
    logvalprint("         error      ", error );
    
    if( error > std::sqrt( Scalar::eps<Scal> ) )
    {
        error_count++;
        eprint("Accuracy issue in "+s+".");
        logprint(s);
    }
    
    if( seq_speedup < float(0.95) )
    {
        ineff_count++;
        wprint("Performance issue in "+s+".");
        TOOLS_DUMP(seq_speedup);
    }
    
    logprint("");
}


template<typename Scal, typename Int, typename LInt>
void Test_SpMM( Int m, Int n, LInt nnz, Int cols )
{
    logprint(std::string("  Test_SpMM")
        + "<" + TypeName<Scal>
        + "," + TypeName<Int>
        + "," + TypeName<LInt>+
        + ">");
    TOOLS_PDUMP(m);
    TOOLS_PDUMP(n);
    TOOLS_PDUMP(nnz);
    TOOLS_PDUMP(cols);
    
    Tensor1<Int ,LInt> idx (nnz);
    Tensor1<Int ,LInt> jdx (nnz);
    Tensor1<Scal,LInt> a   (nnz);

    ParallelDo(
        [&]( const LInt thread )
        {
            const LInt i_begin = JobPointer( nnz, thread_count, thread     );
            const LInt i_end   = JobPointer( nnz, thread_count, thread + 1 );

            std::random_device r;
            std::default_random_engine engine ( r() );
            std::uniform_int_distribution<Int> unif_m(Int(0),static_cast<Int>(m-1));
            std::uniform_int_distribution<Int> unif_n(Int(0),static_cast<Int>(n-1));
            
            for( LInt i = i_begin; i < i_end; ++i )
            {
                idx[i] = unif_m(engine);
                jdx[i] = unif_n(engine);
            }
        },
        thread_count
    );
                    
    a.Random( thread_count );
    
    Sparse::MatrixCSR<Scal,Int,LInt> A( nnz, idx.data(), jdx.data(), a.data(), m, n, thread_count );

    if constexpr ( Scalar::Prec<Scal> == 32 )
    {
        if constexpr ( Scalar::RealQ<Scal> )
        {
            test_SpMM<Real32   ,Real32   ,Real32   ,Real32   >(A, cols);
        }
            test_SpMM<Real32   ,Real32   ,Real32   ,Complex32>(A, cols);
            test_SpMM<Real32   ,Complex32,Real32   ,Complex32>(A, cols);
            test_SpMM<Complex32,Complex32,Complex32,Complex32>(A, cols);
    }
    
    if constexpr ( Scalar::Prec<Scal> == 64 )
    {
        if constexpr ( Scalar::RealQ<Scal> )
        {
            test_SpMM<Real64   ,Real64   ,Real64   ,Real64   >(A, cols);
        }
            test_SpMM<Real64   ,Real64   ,Real64   ,Complex64>(A, cols);
            test_SpMM<Real64   ,Complex64,Real64   ,Complex64>(A, cols);
            test_SpMM<Complex64,Complex64,Complex64,Complex64>(A, cols);
    }
}

int main( int argc, const char * argv[] )
{
    std::filesystem::path path = HomeDirectory().string();
    
    Profiler::Clear( path );

    using Scal = Real64;
    using Int  = Int32;
    using LInt = Size_T;
    
    
    const Int  m            = 600000;
    const Int  n            = 600000;
    const LInt nnz          = int_cast<LInt>(
        (static_cast<double>(m) * static_cast<double>(m)) * 0.0001
    );

    TOOLS_DUMP(m);
    TOOLS_DUMP(n);
    TOOLS_DUMP(nnz);

    std::vector<Int> col_list {1,12};
    TOOLS_PTIC("Testing");
    for( Int cols : col_list )
    {
        logvalprint("cols",cols);

        Test_SpMM<Real32   ,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Real64   ,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Complex32,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Complex64,Int,LInt>( m, n, nnz, cols );

        logprint("");
    }
    TOOLS_PTOC("Testing");

    logvalprint("Total error count", error_count);
    logvalprint("Total ineff count", ineff_count);

    valprint("Total error count", error_count);
    valprint("Total ineff count", ineff_count);

    print("See file " + path.string() + "Tools_Log.txt for details.");
    
    

    
    return 0;
}
