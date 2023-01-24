ASSERT_INT(Int_)

public:

    using Scalar    = Scalar_;
    using Real      = typename ScalarTraits<Scalar_>::Real;
    using Int       = Int_;

    static constexpr bool IsComplex   = ScalarTraits<Scalar>::IsComplex;

    static constexpr Real zero        = 0;
    static constexpr Real half        = 0.5;
    static constexpr Real one         = 1;
    static constexpr Real two         = 2;
    static constexpr Real three       = 3;
    static constexpr Real four        = 4;
    static constexpr Real eps         = std::numeric_limits<Real>::epsilon();
    static constexpr Real eps_squared = eps * eps;
    static constexpr Real eps_sqrt    = MyMath::sqrt(eps);
    static constexpr Real infty       = std::numeric_limits<Real>::max();
