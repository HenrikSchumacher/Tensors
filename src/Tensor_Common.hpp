ASSERT_INT (Int_);

public:

using Scalar = Scalar_;
using Real   = typename ScalarTraits<Scalar_>::Real;
using Int    = Int_;

protected:

Int n = 0;

Scalar * restrict a __attribute__((aligned(ALIGNMENT))) = nullptr ;

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
    logprint("Copy-cast of "+ClassName()+" of size "+ToString(other.Size()) );
    
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
        Read( other.data() );
    }
    return *this;
}




public:

Int Size() const
{
    return n;
}

template<typename S>
void Read( const S * const a_ )
{
    copy_buffer( a_, a, static_cast<size_t>(n) );
}

// Parallelized version.
template<typename S>
void Read( const S * const a_, const Int thread_count )
{
    copy_buffer( a_, a, static_cast<size_t>(n), static_cast<size_t>(thread_count) );
}

template<typename S>
void Write( S * a_ ) const
{
    copy_buffer( a, a_, static_cast<size_t>(n) );
}
template<typename S>
void Write( S * a_, const Int thread_count ) const
{
    copy_buffer( a, a_, static_cast<size_t>(n), static_cast<size_t>(thread_count) );
}

void Fill( const Scalar init )
{
    fill_buffer( a, static_cast<size_t>(n), init );
}

void Fill( const Scalar init, const Int thread_count )
{
    fill_buffer( a, static_cast<size_t>(n), init, static_cast<size_t>(thread_count) );
}

void SetZero()
{
    zerofy_buffer( a, n );
}

void SetZero( const Int thread_count )
{
    zerofy_buffer( a, n, static_cast<size_t>(thread_count) );
}

void Random()
{
    std::random_device r;
    std::default_random_engine engine ( r() );
    
    std::uniform_real_distribution<Scalar> unif(static_cast<Scalar>(-1),static_cast<Scalar>(1));
    
    for( Int i = 0; i < n; ++i )
    {
        a[i] = unif(engine);
    }
}

protected:

force_inline void allocate()
{
    safe_alloc( a, std::max( static_cast<size_t>(0), static_cast<size_t>(n) ) );
}

public:

force_inline Scalar * begin()
{
    return a;
}

force_inline const Scalar * begin() const
{
    return a;
}

force_inline Scalar * end()
{
    return &a[n];
}

force_inline const Scalar * end() const
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
    return ( i < Rank() ) ? dims[static_cast<size_t>(i)] : static_cast<Int>(0);
}

public:

force_inline mut<Scalar> data()
{
    return a;
}

force_inline ptr<Scalar> data() const
{
    return a;
}


void AddFrom( ptr<Scalar> b )
{
    add_to_buffer( b, a, n);
}

void AddTo( mut<Scalar> b ) const
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

Real MaxNorm() const
{
    return norm_max( a, Size() );
}

Real FrobeniusNorm() const
{
    return norm_2( a, Size() );
}


template<class T>
force_inline TENSOR_T & operator*=( const T alpha )
{
    scale_buffer( a, Size() );
    
    return *this;
}


//friend void Subtract( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
//{
//    ptr<Scalar> x_a = x.a;
//    ptr<Scalar> y_a = y.a;
//    mut<Scalar> z_a = z.a;
//
//    const Int last = x.Size();
//
//    #pragma omp parallel for simd aligned( x_a, y_a, z_a : ALIGNMENT ) schedule( static )
//    for( Int k = 0; k < last; ++ k)
//    {
//        z_a[k] = x_a[k] - y_a[k];
//    }
//}
//
//friend void Plus( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
//{
//    ptr<Scalar> x_a = x.a;
//    ptr<Scalar> y_a = y.a;
//    mut<Scalar> z_a = z.a;
//
//    const Int last = x.Size();
//
//    #pragma omp parallel for simd aligned( x_a, y_a, z_a : ALIGNMENT ) schedule( static )
//    for( Int k = 0; k < last; ++ k)
//    {
//        z_a[k] = x_a[k] + y_a[k];
//    }
//}
//
//friend void Times( const Scalar alpha, const TENSOR_T & x, TENSOR_T & y )
//{
//    ptr<Scalar> x_a = x.a;
//    mut<Scalar> y_a = y.a;
//
//    const Int last = x.Size();
//
//    #pragma omp parallel for simd aligned( x_a, y_a : ALIGNMENT ) schedule( static )
//    for( Int k = 0; k < last; ++ k)
//    {
//        y_a[k] = alpha * x_a[k];
//    }
//}

inline friend std::string to_string( const TENSOR_T & A )
{
    return A.ToString();
}


void WriteToFile( const std::string & s ) const
{
    std::ofstream file ( s );
    
    file << std::setprecision( std::numeric_limits<Scalar>::digits10 + 1 );
    
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
