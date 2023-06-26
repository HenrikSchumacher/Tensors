#ifdef LTEMPLATE_H

    
    template<typename Scal, typename Int>
    inline mma::TensorRef<mreal> to_MTensorRef( const TENSOR_T<Scal,Int> & A )
    {
        ASSERT_FLOAT(Scal);
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
    template<typename J, typename Int>
    inline mma::TensorRef<mint> to_MTensorRef( const TENSOR_T<J,Int> & A )
    {
        ASSERT_INT(J);
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
#endif
