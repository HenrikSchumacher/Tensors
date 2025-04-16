//
//  main.cpp
//  UnitTests
//
//  Created by Henrik on 04.02.23.
//

#include <iostream>

#include "Tensors.hpp"

using namespace Tensors;

constexpr Scalar::Flag Generic = Scalar::Flag::Generic;
constexpr Scalar::Flag Plus    = Scalar::Flag::Plus;
constexpr Scalar::Flag Minus   = Scalar::Flag::Minus;
constexpr Scalar::Flag Zero    = Scalar::Flag::Zero;

int error_count = 0;
int ineff_count = 0;


template<typename R_0, typename S_0,typename R_1, typename S_1>
__attribute__((noinline)) void combine_buffers_true(
    cref<R_0> alpha, cptr<S_0> x,
    cref<R_1> beta,  mptr<S_1> y,
    Size_T n
)
{
    for( Size_T i = 0; i < n; ++i )
    {
        y[i] = static_cast<S_1>(alpha) * static_cast<S_1>(x[i]) + static_cast<S_1>(beta) * y[i];
    }
}

template<Scalar::Flag alpha_flag, Scalar::Flag beta_flag, typename R_0, typename S_0, typename R_1, typename S_1>
void test_combine_buffers(
    mref<Tensor1<S_0,Size_T>> x,
    mref<Tensor1<S_1,Size_T>> y,
    mref<Tensor1<S_1,Size_T>> z,
    mref<Tensor1<S_1,Size_T>> z_true
)
{
    const Size_T n = x.Size();
    
    R_0 alpha;
    
    std::string alpha_string;
    
    switch( alpha_flag )
    {
        case Plus:
        {
            alpha = R_0(1);
            alpha_string = "+";
            break;
        }
        case Minus:
        {
            alpha = R_0(-1);
            alpha_string = "-";
            break;
        }
        case Zero:
        {
            alpha = R_0(0);
            alpha_string = "0";
            break;
        }
        default:
        {
            Tensor1<R_0,Size_T> T_alpha (1);
            T_alpha.Random();
            alpha = T_alpha[0];
            alpha_string = "G";
            break;
        }
    }
    
    R_1 beta;
    std::string beta_string;
    
    switch( beta_flag )
    {
        case Plus:
        {
            beta = R_1(1);
            beta_string = "+";
            break;
        }
        case Minus:
        {
            beta = R_1(-1);
            beta_string = "-";
            break;
        }
        case Zero:
        {
            beta = R_1(0);
            beta_string = "0";
            break;
        }
        default:
        {
            Tensor1<R_1,Size_T> T_beta (1);
            T_beta.Random();
            beta = T_beta[0];
            beta_string = "G";
            break;
        }
    }
    
    print("{ "+alpha_string+", "+beta_string+" }");
    
    z_true = y;
    auto start_time_1 = Clock::now();
    combine_buffers_true( alpha, x.data(), beta, z_true.data(), n );
    auto time_1 = Tools::Duration( start_time_1, Clock::now() );
    
    z = y;
    auto start_time_2 = Clock::now();
    combine_buffers<alpha_flag,beta_flag>( alpha, x.data(), beta, z.data(), n );
    auto time_2 = Tools::Duration( start_time_2, Clock::now() );
    
    

    typename Scalar::Real<S_1> error = 0;
    typename Scalar::Real<S_1> max   = 0;
    
    for( Size_T i = 0; i < n ; ++i )
    {
        max   = Max( max, std::abs(z_true[i]) );
        error = Max(error, std::abs(z[i] - z_true[i]));
    }
    
    if( max > 0 )
    {
        error /= max;
    }
    
    const auto speedup = time_1/time_2;
    valprint("         Time  1", time_1 );
    valprint("         Time  2", time_2 );
    valprint("         Speedup", speedup );
    valprint("         error  ", error );
    
    if( error > std::sqrt(  Scalar::eps<S_1> ) )
    {
        error_count++;
        eprint("Accuracy issue!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111elf");
    }
    
    if( speedup < float( 0.95) )
    {
        ineff_count++;
        eprint("Performance issue!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111elf");
    }
}

template<typename R_0, typename S_0, typename R_1, typename S_1>
void Test_combine_buffers( Size_T n )
{
    tic(std::string("Test_combine_buffers<")+TypeName<R_0>+","+TypeName<S_0>+","+TypeName<R_1>+","+TypeName<S_1>+">");
    
    Tensor1<S_0,Size_T> x      ( n );
    Tensor1<S_1,Size_T> y      ( n );
    Tensor1<S_1,Size_T> z      ( n );
    Tensor1<S_1,Size_T> z_true ( n );

    x.Random();
    y.Random();
    
    test_combine_buffers<Generic,Generic,R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Generic,Plus,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Generic,Zero,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Generic,Minus,  R_0,S_0,R_1,S_1>(x,y,z,z_true);

    test_combine_buffers<Plus,   Generic,R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Plus,   Plus,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Plus,   Zero,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Plus,   Minus,  R_0,S_0,R_1,S_1>(x,y,z,z_true);

    test_combine_buffers<Zero,   Generic,R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Zero,   Plus,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Zero,   Zero,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Zero,   Minus,  R_0,S_0,R_1,S_1>(x,y,z,z_true);

    test_combine_buffers<Minus,  Generic,R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Minus,  Plus,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Minus,  Zero,   R_0,S_0,R_1,S_1>(x,y,z,z_true);
    test_combine_buffers<Minus,  Minus,  R_0,S_0,R_1,S_1>(x,y,z,z_true);
    
    toc(std::string("Test_combine_buffers<")+TypeName<R_0>+","+TypeName<S_0>+","+TypeName<R_1>+","+TypeName<S_1>+">");
    
    TOOLS_DUMP(error_count);
    
    if( error_count > int(0) )
    {
        eprint(
            std::string("Test_combine_buffers<")
               +TypeName<R_0>+","
               +TypeName<S_0>+","
               +TypeName<R_1>+","
               +TypeName<S_1>+"> detected "+ToString(error_count)+" errors."
        );
    }
    print("");
}

int main(int argc, const char * argv[])
{
    const Size_T n = 200000000;
    
    Test_combine_buffers<Real32    ,Real32    ,Real32    ,Real32    >(n);
    Test_combine_buffers<Real32    ,Real64    ,Real32    ,Real32    >(n);
    Test_combine_buffers<Real64    ,Real32    ,Real64    ,Real64    >(n);
    Test_combine_buffers<Real64    ,Real64    ,Real64    ,Real64    >(n);
    Test_combine_buffers<Complex32 ,Real32    ,Complex32 ,Complex32 >(n);
    Test_combine_buffers<Complex32 ,Real64    ,Complex32 ,Complex32 >(n);
    Test_combine_buffers<Complex32 ,Complex32 ,Complex32 ,Complex32 >(n);
    Test_combine_buffers<Complex32 ,Complex64 ,Complex32 ,Complex32 >(n);
    Test_combine_buffers<Complex64 ,Real32    ,Complex64 ,Complex64 >(n);
    Test_combine_buffers<Complex64 ,Real64    ,Complex64 ,Complex64 >(n);
    Test_combine_buffers<Complex64 ,Complex32 ,Complex64 ,Complex64 >(n);
    Test_combine_buffers<Complex64 ,Complex64 ,Complex64 ,Complex64 >(n);

    valprint("Total error count", error_count);
    valprint("Total ineff count", ineff_count);
    
    return 0;
}
