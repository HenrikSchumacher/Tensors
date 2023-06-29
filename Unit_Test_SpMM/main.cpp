//
//  main.cpp
//  Unit_Test_SpMM
//
//  Created by Henrik on 04.02.23.
//

#include <iostream>
#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER

#include "Tensors.hpp"

using namespace Tools;
using namespace Tensors;

//constexpr Scalar::Flag Generic = Scalar::Flag::Generic;
//constexpr Scalar::Flag Plus    = Scalar::Flag::Plus;
//constexpr Scalar::Flag Minus   = Scalar::Flag::Minus;
//constexpr Scalar::Flag Zero    = Scalar::Flag::Zero;

int error_count = 0;
int ineff_count = 0;
int max_thread_count = 8;

template<typename R_out, typename T_in, typename S_out, typename T_out, typename Scalar, typename Int, typename LInt>
    __attribute__((noinline)) void Dot_True(
    ptr<LInt> rp, ptr<Int> ci, ptr<Scalar> a, const Int m, const Int n,
    const R_out alpha,  ptr<T_in>  X, const Int ldX,
    const S_out beta,   mut<T_out> Y, const Int ldY,
    const Int   cols
)
{
    std::string tag = std::string("Dot_True<")
        +TypeName<Scalar>+","
        +TypeName<R_out >+","
        +TypeName<T_in  >+","
        +TypeName<S_out >+","
        +TypeName<T_out >+">";
        
    ptic(tag);
    const T_out alpha_ = static_cast<T_out>(alpha);
    const T_out beta_  = static_cast<T_out>(beta);
        
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
                    alpha_ * (static_cast<T_out>(a[k]) * static_cast<T_out>(X[ldX * j + l]));
            }
        }
    }
    ptoc(tag);
}

template<
    typename R_out, typename T_in, typename S_out, typename T_out,
    typename Scal,  typename Int,  typename LInt
>
void test_SpMM( Sparse::MatrixCSR<Scal,Int,LInt> & A, Int cols )
{
    
    const Int  m   = A.RowCount();
    const Int  n   = A.ColCount();
    
    using Real = typename Scalar::Real<T_out>;
    
    std::string s = std::string("\n    <")
          +TypeName<R_out>+","
          +TypeName<T_in >+","
          +TypeName<S_out>+","
          +TypeName<T_out>+
    +">";
    logprint(s);
    
    Tensor2<T_in, Int> X      ( n, cols );
    Tensor2<T_out,Int> Y_0    ( m, cols );
    Tensor2<T_out,Int> Y      ( m, cols );
    Tensor2<T_out,Int> Y_True ( m, cols );
    Tensor2<T_out,Int> Z      ( m, cols );
    
    X.Random( max_thread_count );
    Y_0.Random( max_thread_count );

    
    const R_out alpha = 1;
    const S_out beta  = 1;

    Y_True = Y_0;
    auto start_time_1 = Clock::now();
    Dot_True(
        A.Outer().data(), A.Inner().data(), A.Values().data(), m, n,
        alpha, X.data(),      cols,
        beta,  Y_True.data(), cols,
        cols
    );
    float time_1 = Duration( start_time_1, Clock::now() );
    
    Y = Y_0;
    A.SetThreadCount(1);
    auto start_time_2 = Clock::now();
    A.Dot( alpha, X, beta, Y );
    float time_2 = Duration( start_time_2, Clock::now() );
    
    Y = Y_0;
    A.SetThreadCount(max_thread_count);
    auto start_time_3 = Clock::now();
    A.Dot( alpha, X, beta, Y );
    float time_3 = Duration( start_time_3, Clock::now() );
    
    Real max   = 0;
    Real error = 0;
    
    for( Int i = 0; i < m; ++i )
    {
        for( Int l = 0; l < cols; ++l )
        {
            max    = std::max( max,   std::abs(Y_True(i,l)) );
            Z(i,l) = Y(i,l) - Y_True(i,l);
            error  = std::max( error, std::abs(Z(i,l)) );
        }
    }
    
    if( max > 0 )
    {
        error /= max;
    }
    
    const float seq_speedup = time_1/time_2;
    const float par_speedup = time_2/time_3;
    
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
        dump(seq_speedup);
    }
    
    logprint("");
}


template<typename Scal, typename Int, typename LInt>
void Test_SpMM( Int m, Int n, LInt nnz, Int cols )
{
    logprint(std::string("  Test_SpMM<")
        +TypeName<Scal>+","
        +TypeName<Int>+","
        +TypeName<LInt>+
        +">");
    pdump(m);
    pdump(n);
    pdump(nnz);
    pdump(cols);
    
    Tensor1<Int ,LInt> idx (nnz);
    Tensor1<Int ,LInt> jdx (nnz);
    Tensor1<Scal,LInt> a   (nnz);

    ParallelDoReduce(
        [&]( const LInt thread )
        {
            const LInt i_begin = JobPointer( nnz, max_thread_count, thread     );
            const LInt i_end   = JobPointer( nnz, max_thread_count, thread + 1 );

            std::random_device r;
            std::default_random_engine engine ( r() );
            std::uniform_int_distribution<Int> unif_m(static_cast<Int>(0),static_cast<Int>(m-1));
            std::uniform_int_distribution<Int> unif_n(static_cast<Int>(0),static_cast<Int>(n-1));
            
            for( LInt i = i_begin; i < i_end; ++i )
            {
                idx[i] = unif_m(engine);
                jdx[i] = unif_n(engine);
            }
        },
        max_thread_count
    );
                    
    a.Random( max_thread_count );
    
    Sparse::MatrixCSR<Scal,Int,LInt> A( nnz, idx.data(), jdx.data(), a.data(), m, n, max_thread_count );

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
    
    if constexpr ( Scalar::Prec<Scal> == 128 )
    {
        if constexpr ( Scalar::RealQ<Scal> )
        {
            test_SpMM<Real128   ,Real128   ,Real128   ,Real128   >(A, cols);
        }
            test_SpMM<Real128   ,Real128   ,Real128   ,Complex128>(A, cols);
            test_SpMM<Real128   ,Complex128,Real128   ,Complex128>(A, cols);
            test_SpMM<Complex128,Complex128,Complex128,Complex128>(A, cols);
    }
}

int main( int argc, const char * argv[] )
{
    const char * homedir = getenv("HOME");

    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
    Profiler::Clear( path );

    using Scal = Real64;
    using Int  = Int32;
    using LInt = Size_T;
    
    
    const Int  m            = 600000;
    const Int  n            = 600000;
    const LInt nnz          = static_cast<LInt>(
        (static_cast<double>(m) * static_cast<double>(m)) * 0.0001
    );

    dump(m);
    dump(n);
    dump(nnz);
    
    std::vector<Int> col_list {1,12};

    ptic("Testing");
    for( Int cols : col_list )
    {
        logvalprint("cols",cols);
        Test_SpMM<Real32   ,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Real64   ,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Complex32,Int,LInt>( m, n, nnz, cols );
        Test_SpMM<Complex64,Int,LInt>( m, n, nnz, cols );
        logprint("");
    }
    ptoc("Testing");

    logvalprint("Total error count", error_count);
    logvalprint("Total ineff count", ineff_count);

    valprint("Total error count", error_count);
    valprint("Total ineff count", ineff_count);

    print("See file "+path+"Tools_Log.txt for details.");
    
    
    
    return 0;
}
