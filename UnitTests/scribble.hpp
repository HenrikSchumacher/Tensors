

print("AAAAAA");

Tensor1<Int ,LInt> idx (nnz);
Tensor1<Int ,LInt> jdx (nnz);
Tensor1<Scal,LInt> a   (nnz);

#pragma omp parallel for num_threads(max_thread_count)
for( LInt thread = 0; thread < max_thread_count; ++thread )
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
}
a.Random( max_thread_count );


Sparse::MatrixCSR<Real64,Int32,UInt64> A ( nnz, idx.data(), jdx.data(), a.data(), m, n, max_thread_count );

//    Sparse::PatternCSR<Int32,UInt64> A_0 ( A.Outer(), A.Inner(), m, n, max_thread_count );

constexpr size_t cols (9);

using Ker_T = ScalarBlockKernel_fixed<
    1, 1, cols, false,
    Real64, Real64, Real64,
    Int32, UInt64,
    1, 0,
    true, true, false, false,
    true, true,
    false
>;


Sparse::KernelMatrixCSR<Ker_T> B (A);

dump(B.ClassName());

Tensor2<Real64,Int32> X      ( n, cols );
Tensor2<Real64,Int32> Y_0    ( m, cols );
Tensor2<Real64,Int32> Y      ( m, cols );
Tensor2<Real64,Int32> Y_True ( m, cols );
Tensor2<Real64,Int32> Z      ( m, cols );

X.Random( max_thread_count );
Y_0.Random( max_thread_count );


const Real64 alpha = 1;
const Real64 beta  = 0;

Y_True = Y_0;
auto start_time_1 = Clock::now();
A.Dot( alpha, X, beta, Y_True );
float time_1 = Duration( start_time_1, Clock::now() );

Y = Y_0;
auto start_time_2 = Clock::now();
B.Dot( A.Values().data(), alpha, X.data(), beta, Y.data(), cols );
float time_2 = Duration( start_time_2, Clock::now() );
//
//    Y = Y_0;
//    A.SetThreadCount(max_thread_count);
//    auto start_time_3 = Clock::now();
//    A.Dot( alpha, X, beta, Y );
//    float time_3 = Duration( start_time_3, Clock::now() );

Real64 max   = 0;
Real64 error = 0;

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
//    const float par_speedup = time_2/time_3;

valprint("         Time 1     ", time_1 );
valprint("         Time 2     ", time_2 );
//    logvalprint("         Time 3     ", time_3 );
valprint("         Seq_speedup", seq_speedup );
//    logvalprint("         Par speedup", par_speedup );
valprint("         error      ", error );

if( error > std::sqrt( Scalar::eps<Scal> ) )
{
    error_count++;
}

if( seq_speedup < float(0.95) )
{
    ineff_count++;
    dump(seq_speedup);
}
