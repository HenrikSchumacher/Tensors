#ifdef LTEMPLATE_H
    
    template<typename Scal, typename Int, IS_FLOAT(Scal)>
    inline mma::TensorRef<mreal> to_MTensorRef( cref<TENSOR_T<Scal,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
    template<typename J, typename Int, IS_INT(J)>
    inline mma::TensorRef<mint> to_MTensorRef( cref<TENSOR_T<J,Int>> A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
#endif
