ASSERT_INT (Int_);

public:

using Scal = Scal_;
using Real = typename Scalar::Real<Scal_>;
using Int  = Int_;

protected:

Int n = 0;

Scal * restrict a ALIGNED = nullptr ;

public:

// The big four and a half:

TENSOR_T() = default;

// Destructor
~TENSOR_T()
{
    safe_free(a);
}

// Copy constructor
TENSOR_T( const TENSOR_T & other )
:   n    ( other.n    )
,   dims ( other.dims )
{
    logprint("Copy of "+ClassName()+" of size "+Tools::ToString(other.Size()) );
    
    allocate();
    Read(other.a);
}

// Copy-cast constructor
template<typename S, typename J, IS_INT(J)>
explicit TENSOR_T( const TENSOR_T<S,J> & other )
:   n    ( other.n    )
,   dims ( other.dims )
{
    logprint("Copy-cast of "+ClassName()+" of size "+Tools::ToString(other.Size()) );
    
    allocate();
    Read(other.a);
}

inline friend void swap(TENSOR_T & A, TENSOR_T & B) noexcept
{
//    logprint(A.ClassName()+": Swap");
    // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
    using std::swap;
    std::swap_ranges( &A.dims[0], &A.dims[Rank()], &B.dims[0] );
    swap( A.n, B.n );
    swap( A.a, B.a );
}

// Move constructor
TENSOR_T( TENSOR_T && other ) noexcept
:   TENSOR_T()
{
//    logprint(other.ClassName()+": Move");
    swap(*this, other);
}


/* Move-assignment operator */
TENSOR_T & operator=( TENSOR_T && other ) noexcept
{
//    print(other.ClassName()+": Move-assign");
    if( this == &other )
    {
        wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
    }
    swap( *this, other );
    return *this;
}

/* Copy-assignment operator */
TENSOR_T & operator=( const TENSOR_T & other )
{
    if( this != &other )
    {
//        logprint(other.ClassName()+": Copy-assignment of size "+Tools::ToString( other.n ));
        
        if( dims != other.dims )
        {
            n    = other.n;
            dims = other.dims;

//            logprint(other.ClassName()+": Reallocation of size "+Tools::ToString( n ) );
            
            safe_free(a);
            allocate();
        }
        Read( other.a );
    }
    return *this;
}




public:

Int Size() const
{
    return n;
}

template<typename S>
void Read( ptr<S> a_ )
{
    copy_buffer( a_, a, static_cast<Size_T>(n) );
}

//// Parallelized version.
template<typename S>
void ReadParallel( ptr<S> a_, const Size_T thread_count )
{
    copy_buffer( a_, a, static_cast<Size_T>(n), thread_count );
}

template<typename R>
std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
Read( ptr<R> re, ptr<R> im )
{
    for( Int i = 0; i < n; ++i )
    {
        a[i].real( re[i] );
        a[i].imag( im[i] );
    }
}

template<typename S>
void Write( mut<S> a_ ) const
{
    copy_buffer( a, a_, static_cast<Size_T>(n) );
}
template<typename S>
void WriteParallel( mut<S> a_, const Size_T thread_count ) const
{
    copy_buffer( a, a_, static_cast<Size_T>(n), thread_count );
}

template<typename R>
std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
Write( mut<R> re, mut<R> im ) const
{
    for( Int i = 0; i < n; ++i )
    {
        re[i] = real(a[i]);
        im[i] = imag(a[i]);
    }
}

void Fill( const Scal init )
{
    fill_buffer( a, static_cast<Size_T>(n), init );
}

void Fill( const Scal init, const Size_T thread_count )
{
    fill_buffer( a, static_cast<Size_T>(n), init, thread_count );
}

void SetZero()
{
    zerofy_buffer( a, n );
}

void SetZero( const Size_T thread_count )
{
    zerofy_buffer( a, n, thread_count );
}

void Random( Int thread_count = 1 )
{
    if constexpr (Scalar::RealQ<Scal> )
    {
        ParallelDo(
            [=]( const Int thread )
            {
                const Int i_begin = JobPointer( n, thread_count, thread     );
                const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                
                std::random_device r;
                std::default_random_engine engine ( r() );
                
                std::uniform_real_distribution<Real> unif(static_cast<Real>(-1),static_cast<Real>(1));
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    a[i] = unif(engine);
                }
            },
            thread_count
        );
    }
    else
    {
        ParallelDo(
            [=]( const Int thread )
            {
                const Int i_begin = JobPointer( n, thread_count, thread     );
                const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                
                std::random_device r;
                std::default_random_engine engine ( r() );
                
                std::uniform_real_distribution<Real> unif(static_cast<Real>(-1),static_cast<Real>(1));
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    a[i] = Scal( unif(engine), unif(engine) );
                }
            },
            thread_count
        );
    }
}

protected:

force_inline void allocate()
{
    safe_alloc( a, std::max( static_cast<Size_T>(0), static_cast<Size_T>(n) ) );
}

public:

force_inline Scal * begin()
{
    return a;
}

force_inline const Scal * begin() const
{
    return a;
}

force_inline Scal * end()
{
    return &a[n];
}

force_inline const Scal * end() const
{
    return &a[n];
}

//const Int * dimensions() const
//{
//    return &dims[0];
//}

force_inline const Int * Dimensions() const
{
    return &dims[0];
}

force_inline Int Dimension( const Int i ) const
{
    return ( i < Rank() ) ? dims[static_cast<Size_T>(i)] : static_cast<Int>(0);
}

public:

force_inline mut<Scal> data()
{
    return a;
}

force_inline ptr<Scal> data() const
{
    return a;
}


void AddFrom( ptr<Scal> b )
{
    add_to_buffer( b, a, n);
}

void AddTo( mut<Scal> b ) const
{
    add_to_buffer( a, b, n);
}

Int CountNan() const
{
    Int counter = 0;
     
    for( Int i = 0 ; i < n; ++i )
    {
        counter += std::isnan(a[i]);
    }
    
    return counter;
}


std::pair<Real,Real> MinMax() const
{
    return minmax_buffer( a, n );
}

Real Min() const
{
    return MinMax().first;
}

Real Max() const
{
    return MinMax().second;
}

Real MaxNorm() const
{
    return norm_max( a, n );
}

Real FrobeniusNorm() const
{
    return norm_2( a, n );
}

friend Real MaxDistance( const TENSOR_T & x, const TENSOR_T & y )
{
    ptr<Scal> x_a = x.a;
    ptr<Scal> y_a = y.a;

    const Int last = x.Size();

    Real max = 0;
    
    for( Int k = 0; k < last; ++k )
    {
        max = std::max( max, std::abs( x_a[k] - y_a[k] ) );
    }
    
    return max;
}

inline friend Real RelativeMaxError( const TENSOR_T & x, const TENSOR_T & y )
{
    return MaxDistance(x,y) / x.MaxNorm();
}


template<class T>
force_inline TENSOR_T & operator*=( const T alpha )
{
    scale_buffer( a, n );
    
    return *this;
}


friend void Subtract( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
{
    ptr<Scal> x_a = x.a;
    ptr<Scal> y_a = y.a;
    mut<Scal> z_a = z.a;

    const Int last = x.Size();

    for( Int k = 0; k < last; ++ k)
    {
        z_a[k] = x_a[k] - y_a[k];
    }
}

friend void Plus( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
{
    ptr<Scal> x_a = x.a;
    ptr<Scal> y_a = y.a;
    mut<Scal> z_a = z.a;

    const Int last = x.Size();

    for( Int k = 0; k < last; ++ k)
    {
        z_a[k] = x_a[k] + y_a[k];
    }
}

friend void Times( const Scal alpha, const TENSOR_T & x, TENSOR_T & y )
{
    ptr<Scal> x_a = x.a;
    mut<Scal> y_a = y.a;

    const Int last = x.Size();

    for( Int k = 0; k < last; ++ k)
    {
        y_a[k] = alpha * x_a[k];
    }
}

inline friend std::string to_string( const TENSOR_T & A )
{
    return A.ToString();
}


void WriteToFile( const std::string & s ) const
{
    std::ofstream file ( s );
    
    file << std::setprecision( std::numeric_limits<Scal>::digits10 + 1 );
    
    if( n > 0 )
    {
        file << a[0];
    }
    for( Int i = 1; i < n; ++i )
    {
        file <<"\t" << a[i];
    }
    file.close();
}

void ReadFromFile( const std::string & s ) const
{
    std::ifstream file ( s );

    for( Int i = 0; i < n; ++i )
    {
        file >> a[i];
    }
    file.close();
}

std::string ToString( int prec = 16 ) const
{
    return ArrayToString( a, dims.data(), Rank(), prec );
}

template<class Stream_T>
Stream_T & ToStream( Stream_T & s ) const
{
    return ArrayToStream( a, dims.data(), Rank(), s );
}

inline friend std::ostream & operator<<( std::ostream & s, const TENSOR_T & tensor )
{
    tensor.ToStream(s);
    return s;
}
