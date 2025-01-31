    static_assert(IntQ<Int_>,"");

public:

    using Scal = Scal_;
    using Real = typename Scalar::Real<Scal_>;
    using Int  = Int_;

    static constexpr Size_T Alignment = alignment;

protected:

    Int n = 0;

    Scal * restrict a = nullptr ;

public:

    // The big four and a half:

    TENSOR_T() = default;

    // Destructor
    ~TENSOR_T()
    {
//        logprint("Destuctor of "+ClassName()+" of size "+ToString(Size()) );
        safe_free(a);
    }

    // Copy constructor
    TENSOR_T( const TENSOR_T & other )
    :   n    ( other.n    )
    ,   dims ( other.dims )
    {
//        logprint("Copy of "+ClassName()+" of size "+ToString(other.Size()) );
        
        allocate();
        Read(other.a);
    }

    // Copy-cast constructor
    template<typename S, typename J, Size_T alignment_>
    explicit TENSOR_T( const TENSOR_T<S,J,alignment_> & other )
    :   n    ( other.n    )
    ,   dims ( other.dims )
    {
        static_assert(IntQ<J>,"");
//        logprint("Copy-cast of "+ClassName()+" of size "+ToString(other.Size()) );
        
        allocate();
        Read(other.a);
    }

    inline friend void swap( TENSOR_T & A, TENSOR_T & B) noexcept
    {
//        logprint(A.ClassName()+": swap");
        // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
        using std::swap;
        
        if( &A == &B )
        {
            wprint( std::string("An object of type ") + ClassName() + " has been swapped to itself.");
        }
        else
        {
            swap( A.dims, B.dims );
            
            swap( A.n, B.n );
            
            swap( A.a, B.a );
        }
    }

    // Move constructor
    TENSOR_T( TENSOR_T && other ) noexcept
    :   TENSOR_T()
    {
//        logprint(other.ClassName()+": Move-constructor");
        swap(*this, other);
    }


    /* Move-assignment operator */
    mref<TENSOR_T> operator=( TENSOR_T && other ) noexcept
    {
//        logprint(other.ClassName()+": Move-assign");
        if( this == &other )
        {
            wprint("An object of type " + ClassName() + " has been move-assigned to itself.");
        }
        else
        {
            swap( *this, other );
        }
        return *this;
    }

    /* Copy-assignment operator */
    mref<TENSOR_T> operator=( const TENSOR_T & other )
    {
        if( this != &other )
        {
//            logprint(other.ClassName()+": Copy-assignment of size "+ToString( other.n ));
            
            if( dims != other.dims )
            {
                n    = other.n;
                dims = other.dims;
                
//                logprint(other.ClassName()+": Reallocation of size "+ToString( n ) );
                
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
    void Read( cptr<S> a_ )
    {
        copy_buffer( a_, a, static_cast<Size_T>(n) );
    }

    //// Parallelized version.
    template<typename S>
    void ReadParallel( cptr<S> a_, const Size_T thread_count )
    {
        copy_buffer<VarSize,Parallel>( a_, a, static_cast<Size_T>(n), thread_count );
    }

    template<typename R>
    std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
    Read( cptr<R> re, cptr<R> im )
    {
        for( Int i = 0; i < n; ++i )
        {
            a[i].real( re[i] );
            a[i].imag( im[i] );
        }
    }

    template<typename S>
    void Write( mptr<S> a_ ) const
    {
        copy_buffer( a, a_, static_cast<Size_T>(n) );
    }

    template<typename S>
    void WriteParallel( mptr<S> a_, const Size_T thread_count ) const
    {
        copy_buffer<VarSize,Parallel>( a, a_, static_cast<Size_T>(n), thread_count );
    }

    template<typename R>
    std::enable_if_t<Scalar::ComplexQ<Scal> && !Scalar::ComplexQ<R>,void>
    Write( mptr<R> re, mptr<R> im ) const
    {
        for( Int i = 0; i < n; ++i )
        {
            re[i] = real(a[i]);
            im[i] = imag(a[i]);
        }
    }

    void Fill( cref<Scal> init )
    {
        fill_buffer( a, init, static_cast<Size_T>(n) );
    }

    void Fill( cref<Scal> init, const Size_T thread_count )
    {
        fill_buffer<VarSize,Parallel>( a, init, int_cast<Size_T>(n), thread_count );
    }

    void SetZero()
    {
        zerofy_buffer( a, int_cast<Size_T>(n) );
    }

    void SetZero( const Size_T thread_count )
    {
        zerofy_buffer<VarSize,Parallel>( a, int_cast<Size_T>(n), thread_count );
    }

    void Random( const Int thread_count = 1 )
    {
        static_assert( Scalar::FloatQ<Scal>, "" );
        
        ptic("Random");
        // This uses std::mt19937_64.
        // Moreover, the pseudorandom number generators are initilized per call.
        // So this is not very efficient.
        
        using SD_T = std::random_device;
        using MT_T = std::mt19937_64;
        
        using SD_UInt = SD_T::result_type;
        using MT_UInt = MT_T::result_type;
        
        constexpr Size_T seed_size = (MT_T::state_size * sizeof(MT_UInt)) / sizeof(SD_UInt);
        
        std::vector<SD_UInt> seed_array ( seed_size );
        
        std::vector<MT_T> engines;
        
        ptic("Random - Initialize");
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            std::generate( seed_array.begin(), seed_array.end(), SD_T() );
        
            std::seed_seq seed ( seed_array.begin(), seed_array.end() );
        
            engines.emplace_back( seed );
        }
        ptoc("Random - Initialize");
        
        if constexpr (Scalar::RealQ<Scal> )
        {
            ParallelDo(
                [&,this]( const Int thread )
                {
                    const Int i_begin = JobPointer( n, thread_count, thread     );
                    const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                    
                    MT_T & engine = engines[thread];
                    
                    std::uniform_real_distribution<Real> unif(-Scalar::One<Real>,Scalar::One<Real>);
                   
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
               [&,this]( const Int thread )
               {
                   const Int i_begin = JobPointer( n, thread_count, thread     );
                   const Int i_end   = JobPointer( n, thread_count, thread + 1 );
                   
                   MT_T & engine = engines[thread];
                   
                   std::uniform_real_distribution<Real> unif(-Scalar::One<Real>,Scalar::One<Real>);
                   
                   for( Int i = i_begin; i < i_end; ++i )
                   {
                       a[i] = Scal( unif(engine), unif(engine) );
                   }
               },
               thread_count
           );
        }
        
        ptoc("Random");
    }

protected:

    force_inline void allocate()
    {
        safe_alloc( a, static_cast<Size_T>(Tools::Ramp(n)), Alignment );
    }

public:

    force_inline mptr<Scal> begin()
    {
        return a;
    }

    force_inline cptr<Scal> begin() const
    {
        return a;
    }

    force_inline mptr<Scal> end()
    {
        return &a[n];
    }

    force_inline cptr<Scal> end() const
    {
        return &a[n];
    }

    //const Int * dimensions() const
    //{
    //    return &dims[0];
    //}

    force_inline cptr<Int> Dimensions() const
    {
        return &dims[0];
    }

    force_inline Int Dimension( const Int i ) const
    {
        return ( i < Rank() ) ? dims[static_cast<Size_T>(i)] : static_cast<Int>(0);
    }

public:

    force_inline mptr<Scal> data()
    {
        return a;
    }

    force_inline cptr<Scal> data() const
    {
        return a;
    }


    void AddFrom( cptr<Scal> b )
    {
        add_to_buffer( b, a, n );
    }

    void AddTo( mptr<Scal> b ) const
    {
        add_to_buffer( a, b, n );
    }

    Int CountNaNs() const
    {
        Int counter = 0;
        
        for( Int i = 0 ; i < n; ++i )
        {
            counter += Int(NaNQ(a[i]));
        }
        
        return counter;
    }


    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,std::pair<Real,Real>> MinMax( Int thread_count = 1 ) const
    {
        return minmax_buffer( a, n, thread_count );
    }

    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,Real> Min( Int thread_count = 1 ) const
    {
        return min_buffer( a, n, thread_count );
    }

    template <typename Dummy = Scal>
    std::enable_if_t<SameQ<Real,Dummy>,Real> Max( Int thread_count = 1 ) const
    {
        return max_buffer( a, n, thread_count );
    }

    Real MaxNorm( Int thread_count = 1 ) const
    {
        return norm_max( a, n, thread_count );
    }

    Real FrobeniusNorm( Int thread_count = 1 ) const
    {
        return norm_2( a, n, thread_count );
    }

    Real FrobeniusNormSquared( Int thread_count = 1 ) const
    {
        return norm_2_squared( a, n, thread_count );
    }

    friend Real MaxDistance( cref<TENSOR_T> x, cref<TENSOR_T> y )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        
        const Int last = x.Size();
        
        Real max = 0;
        
        for( Int k = 0; k < last; ++k )
        {
            max = Tools::Max( max, Abs( x_a[k] - y_a[k] ) );
        }
        
        return max;
    }

    inline friend Real RelativeMaxError( cref<TENSOR_T> x, cref<TENSOR_T> y )
    {
        return MaxDistance(x,y) / x.MaxNorm();
    }


    template<typename T, typename I, Size_T align>
    force_inline mref<TENSOR_T> operator+=( cref<TENSOR_T<T,I,align>> b )
    {
        const Size_T m = Tools::Min( int_cast<Size_T>(n), int_cast<Size_T>(b.Size()) );
        
        combine_buffers<Scalar::Flag::Plus,Scalar::Flag::Plus>(
            Scalar::One<T>, b.data(), Scalar::One<T>, a, m
        );
        
        return *this;
    }

    template<typename T, typename I, Size_T align>
    force_inline mref<TENSOR_T> operator-=( cref<TENSOR_T<T,I,align>> b )
    {
        const Size_T m = Tools::Min( int_cast<Size_T>(n), int_cast<Size_T>(b.Size()) );
        
        combine_buffers<Scalar::Flag::Minus,Scalar::Flag::Plus>(
            -Scalar::One<T>, b.data(), Scalar::One<T>, a, m
        );
        
        return *this;
    }

    template<class T>
    force_inline mref<TENSOR_T> operator*=( cref<T> alpha )
    {
        scale_buffer( alpha, a, n );
        
        return *this;
    }


    friend void Subtract( cref<TENSOR_T> x, cref<TENSOR_T> y, mref<TENSOR_T> z )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        mptr<Scal> z_a = z.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            z_a[k] = x_a[k] - y_a[k];
        }
    }

    friend void Plus( cref<TENSOR_T> x, cref<TENSOR_T> y, mref<TENSOR_T> z )
    {
        cptr<Scal> x_a = x.a;
        cptr<Scal> y_a = y.a;
        mptr<Scal> z_a = z.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            z_a[k] = x_a[k] + y_a[k];
        }
    }

    friend void Times( const Scal alpha, cref<TENSOR_T> x, mref<TENSOR_T> y )
    {
        cptr<Scal> x_a = x.a;
        mptr<Scal> y_a = y.a;
        
        const Int last = x.Size();
        
        for( Int k = 0; k < last; ++ k)
        {
            y_a[k] = alpha * x_a[k];
        }
    }

    inline friend std::string to_string( cref<TENSOR_T> A, const int prec = 16 )
    {
        return ArrayToString( A.a, A.dims.data(), Rank(), prec );
    }

    inline friend std::string ToString( cref<TENSOR_T> A,
        const int prec = 16,
        std::string line_prefix = std::string("")
    )
    {
        return ArrayToString( A.a, A.dims.data(), Rank(), prec, line_prefix );
    }


    void WriteToFile( const std::filesystem::path & s ) const
    {
        std::ofstream file ( s );
        
        file << std::scientific << std::uppercase << std::setprecision( std::numeric_limits<Scalar::Real<Scal>>::digits10 + 1 );
        
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

    void ReadFromFile( const std::filesystem::path & s ) const
    {
        std::ifstream file ( s );
        
        for( Int i = 0; i < n; ++i )
        {
            file >> a[i];
        }
        file.close();
    }

//    [[nodiscard]] std::string ToString( int prec = 16 ) const
//    {
//        return ArrayToString( a, dims.data(), Rank(), prec );
//    }

    template<class Stream_T>
    Stream_T & ToStream( Stream_T & s ) const
    {
        return ArrayToStream( a, dims.data(), Rank(), s );
    }

    inline friend std::ostream & operator<<( std::ostream & s, cref<TENSOR_T> tensor )
    {
        tensor.ToStream(s);
        return s;
    }
