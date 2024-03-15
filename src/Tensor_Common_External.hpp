#ifdef LTEMPLATE_H
    
    template<typename Real, typename Int, IS_FLOAT(Real)>
    inline mma::TensorRef<mreal> to_MTensorRef( cref<TENSOR_T<Real,Int>> A )
    {
        auto B = mma::makeTensor<mreal>( A.Rank(), A.Dimensions() );
        
        A.Write(B.data());
        
        return B;
    }

    template<typename Real, typename Int, IS_FLOAT(Real)>
    inline mma::TensorRef<std::complex<Real>> to_MTensorRef( cref<TENSOR_T<std::complex<Real>,Int>> A )
    {
        auto B = mma::makeTensor<std::complex<Real>>( A.Rank(), A.Dimensions() );
        
        A.Write(B.data());
        
        return B;
    }
    
    template<typename J, typename Int, IS_INT(J)>
    inline mma::TensorRef<mint> to_MTensorRef( cref<TENSOR_T<J,Int>> A )
    {
        auto B = mma::makeTensor<mint>( A.Rank(), A.Dimensions() );
        
        A.Write(B.data());
        
        return B;
    }
    
#endif

#ifdef MMA_HPP

    template<typename Real, typename Int, IS_FLOAT(Real)>
    inline mma::MTensorWrapper<mreal> to_MTensorWrapper( cref<TENSOR_T<Real,Int>> A )
    {
        mma::MTensorWrapper<mreal> B ( A.Rank(), A.Dimensions() );
        
        A.Write( B.data() );
        
        return B;
    }

    template<typename Real, typename Int, IS_FLOAT(Real)>
    inline mma::MTensorWrapper<std::complex<mreal>> to_MTensorWrapper( cref<TENSOR_T<std::complex<Real>,Int>> A )
    {
        mma::MTensorWrapper<std::complex<mreal>> B ( A.Rank(), A.Dimensions() );
        
        A.Write( B.data() );
        
        return B;
    }

    template<typename J, typename Int, IS_INT(J)>
    inline mma::MTensorWrapper<mint> to_MTensorWrapper( cref<TENSOR_T<J,Int>> A )
    {
        mma::MTensorWrapper<mint> B ( A.Rank(), A.Dimensions() );
        
        A.Write( B.data() );
        
        return B;
    }

#endif
