public:

    using Scalar = Scalar_;
    using Real   = typename ScalarTraits<Scalar_>::Real;
    using Int    = Int_;

    static constexpr Real zero        = 0;
    static constexpr Real half        = 0.5;
    static constexpr Real one         = 1;
    static constexpr Real two         = 2;
    static constexpr Real three       = 3;
    static constexpr Real four        = 4;
    static constexpr Real eps         = std::numeric_limits<Real>::min();
    static constexpr Real eps_squared = eps * eps;
    static constexpr Real infty       = std::numeric_limits<Real>::max();


    CLASS() = default;

    ~CLASS() = default;

    CLASS(std::nullptr_t) = delete;


    explicit CLASS( const Scalar init )
    {
        Fill(init);
    }

    explicit CLASS( const Scalar * a )
    {
        Read(a);
    }

    // Copy constructor
    CLASS( const CLASS & other )
    {
        Read( &other.A[0][0] );
    }

    friend void swap( CLASS & A, CLASS & B ) noexcept
    {
        // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
        using std::swap;
        
        swap( A.A, B.A );
    }

    // Copy assignment operator
    CLASS & operator=( CLASS other )
    {
        // copy-and-swap idiom
        // see https://stackoverflow.com/a/3279550/8248900 for details
        swap(*this, other);

        return *this;
    }

    /* Move constructor */
    CLASS( CLASS && other ) noexcept
    :   CLASS()
    {
        swap(*this, other);
    }

    template<class T>
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator/=( const T lambda )
    {
        return (*this) *= ( static_cast<T>(1)/lambda );
    }

    inline friend std::ostream & operator<<( std::ostream & s, const CLASS & M )
    {
        s << M.ToString();
        return s;
    }
