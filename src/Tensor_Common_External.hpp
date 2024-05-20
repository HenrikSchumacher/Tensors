#ifdef LTEMPLATE_H
    
    template<typename Real, typename Int,
        class = typename std::enable_if_t<FloatQ<Real>>
    >
    inline mma::TensorRef<mreal> to_MTensorRef( cref<TENSOR_T<Real,Int>> A )
    {
        auto B = mma::makeTensor<mreal>( A.Rank(), A.Dimensions() );
        A.Write(B.data());
        return B;
    }

    template<typename Real, typename Int, 
        class = typename std::enable_if_t<FloatQ<Real>>
    >
    inline mma::TensorRef<std::complex<Real>> to_MTensorRef( cref<TENSOR_T<std::complex<Real>,Int>> A )
    {
        auto B = mma::makeTensor<std::complex<Real>>( A.Rank(), A.Dimensions() );
        A.Write(B.data());
        return B;
    }
    
    template<typename J, typename Int, 
        class = typename std::enable_if_t<IntQ<J>>
    >
    inline mma::TensorRef<mint> to_MTensorRef( cref<TENSOR_T<J,Int>> A )
    {
        auto B = mma::makeTensor<mint>( A.Rank(), A.Dimensions() );
        A.Write(B.data());
        return B;
    }
    
#endif
