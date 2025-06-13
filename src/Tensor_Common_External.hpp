#ifdef LTEMPLATE_H
    
    template<typename T, typename Int, Size_T alignment,
        class = typename std::enable_if_t<mma::HasTypeQ<T>>
    >
    inline mma::TensorRef<mma::Type<T>> to_MTensorRef(
       cref<TENSOR_T<T,Int,alignment>> A
    )
    {
        auto B = mma::makeTensor<mma::Type<T>>( A.Rank(), A.Dims() );
        
        A.Write(B.data());
        
        return B;
    }
    
#endif

#ifdef MMA_HPP

    template<typename Real, typename Int, Size_T alignment,
        class = typename std::enable_if_t<FloatQ<Real>>
    >
    inline mma::MTensorWrapper<mreal> to_MTensorWrapper(
        cref<TENSOR_T<Real,Int,alignment>> A
    )
    {
        mma::MTensorWrapper<mreal> B ( A.Rank(), A.Dims() );
        
        A.Write( B.data() );
        
        return B;
    }

    template<typename Real, typename Int, Size_T alignment,
        class = typename std::enable_if_t<FloatQ<Real>>
    >
    inline mma::MTensorWrapper<std::complex<mreal>> to_MTensorWrapper( cref<TENSOR_T<std::complex<Real>,Int,alignment>> A
    )
    {
        mma::MTensorWrapper<std::complex<mreal>> B ( A.Rank(), A.Dims() );
        
        A.Write( B.data() );
        
        return B;
    }

    template<typename J, typename Int, Size_T alignment,
        class = typename std::enable_if_t<IntQ<J>>
    >
    inline mma::MTensorWrapper<mint> to_MTensorWrapper(
        cref<TENSOR_T<J,Int,alignment>> A
    )
    {
        mma::MTensorWrapper<mint> B ( A.Rank(), A.Dims() );
        
        A.Write( B.data() );
        
        return B;
    }

#endif
