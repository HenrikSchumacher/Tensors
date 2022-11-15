ASSERT_INT  (I);


protected:

I n = 0;

T * restrict a = nullptr;

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
:   n(other.n)
,   dims(other.dims)
{
    print("Copy of "+ClassName());
    allocate();
    
    Read(other.a);
}

// Copy + cast constructor
template<typename S, typename J, IsInt(J)>
explicit TENSOR_T( const TENSOR_T<S,J> & other )
:   n(other.n)
,   dims(other.dims)
{
    allocate();
    
    Read(other.a);
}

inline friend void swap(TENSOR_T & A, TENSOR_T & B) noexcept
{
//    print("swap");
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
//    print("move");
    swap(*this, other);
}

/* Copy assignment operator */
TENSOR_T & operator=( const TENSOR_T & other )
{
    print("Copy+assign of "+ClassName());
    if( this != &other )
    {
        if( dims != other.dims )
        {
            n    = other.n;
            dims = other.dims;
            
            safe_free(a);
            allocate();
        }
        Read( other.data() );
    }
    return *this;
}

/* Move assignment operator */
TENSOR_T & operator=( TENSOR_T && other ) noexcept
{
//    print("move+assign");
    if( this == &other )
    {
        wprint("An object of type "+ClassName()+" has been move-assigned to itself.");
    }
    swap( *this, other );
    return *this;
}

public:

I Size() const
{
    return n;
}

template<typename S>
void Read( const S * const a_ )
{
//    ptic(ClassName()+"::Read( const S * const a_ )");
    
    copy_cast_buffer( a_, a, static_cast<size_t>(n) );
    
//    ptoc(ClassName()+"::Read( const S * const a_ )");
}

template<typename S>
void Write( S * a_ ) const
{
//    ptic(ClassName()+"::Write( S * a_ )");
    
    copy_cast_buffer( a, a_, n );

//    ptoc(ClassName()+"::Write( S * a_ )");
}

void Fill( const T init )
{
    fill_buffer( a, static_cast<size_t>(n), init );
}

void SetZero()
{
    zerofy_buffer( a, n );
}

void Random()
{
    std::random_device r;
    std::default_random_engine engine ( r() );
    
    std::uniform_real_distribution<T> unif(static_cast<T>(-1),static_cast<T>(1));
    
    for( I i = 0; i < n; ++i )
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

force_inline T * begin()
{
    return a;
}

force_inline const T * begin() const
{
    return a;
}

force_inline T * end()
{
    return &a[n];
}

force_inline const T * end() const
{
    return &a[n];
}

//const I * dimensions() const
//{
//    return &dims[0];
//}

force_inline const I * Dimensions() const
{
    return &dims[0];
}

force_inline I Dimension( const I i ) const
{
    return ( i < Rank() ) ? dims[static_cast<size_t>(i)] : static_cast<I>(0);
}

public:

force_inline T * data()
{
    return a;
}

force_inline const T * data() const
{
    return a;
}


void AddFrom( const T * restrict const b )
{
    #pragma omp parallel for simd schedule( static )
    for( I i = 0; i < n; ++i )
    {
        // cppcheck-suppress [arithOperationsOnVoidPointer]
        a[i] += b[i];
    }
}

void AddTo( T * restrict const b ) const
{
    #pragma omp parallel for simd schedule( static )
    for( I i = 0; i < n; ++i )
    {
        // cppcheck-suppress [arithOperationsOnVoidPointer]
        b[i] += a[i];
    }
}

//I CountNan() const
//{
//    I counter = 0;
//     
////    #pragma omp simd aligned( a : ALIGNMENT ) reduction( + : counter )
//    for( I i = 0 ; i < n; ++i )
//    {
//        counter += std::isnan(a[i]);
//    }
//    
//    return counter;
//}

T MaxNorm() const
{
    T result = static_cast<T>(0);

    #pragma omp simd aligned( a : ALIGNMENT ) reduction( max : result )
    for( I i = 0 ; i < n; ++i )
    {
        result = std::max( result, std::abs(a[i]));
    }
    
    return result;
}

T FrobeniusNorm() const
{
    T result = static_cast<T>(0);

    #pragma omp simd aligned( a : ALIGNMENT ) reduction( + : result )
    for( I i = 0 ; i < n; ++i )
    {
        result += a[i] * a[i];
    }
    
    return std::sqrt(result);
}


friend void Subtract( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
{
    const T * restrict const x_a = x.a;
    const T * restrict const y_a = y.a;
          T * restrict const z_a = z.a;
    
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x_a, y_a, z_a : ALIGNMENT ) schedule( static )
    for( I k = 0; k < last; ++ k)
    {
        z_a[k] = x_a[k] - y_a[k];
    }
}

friend void Plus( const TENSOR_T & x, const TENSOR_T & y, TENSOR_T & z )
{
    const T * restrict const x_a = x.a;
    const T * restrict const y_a = y.a;
          T * restrict const z_a = z.a;
   
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x_a, y_a, z_a : ALIGNMENT ) schedule( static )
    for( I k = 0; k < last; ++ k)
    {
        z_a[k] = x_a[k] + y_a[k];
    }
}

friend void Times( const T alpha, const TENSOR_T & x, TENSOR_T & y )
{
    const T * restrict const x_a = x.a;
          T * restrict const y_a = y.a;
    
    const I last = x.Size();
    
    #pragma omp parallel for simd aligned( x_a, y_a : ALIGNMENT ) schedule( static )
    for( I k = 0; k < last; ++ k)
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
    
    file << std::setprecision( std::numeric_limits<T>::digits10 + 1 );
    
    if( n > 0 )
    {
        file << a[0];
    }
    for( I i = 1; i < n; ++i )
    {
        file <<"\t" << a[i];
    }
    file.close();
}

void ReadFromFile( const std::string & s ) const
{
    std::ifstream file ( s );

    for( I i = 0; i < n; ++i )
    {
        file >> a[i];
    }
    file.close();
}
