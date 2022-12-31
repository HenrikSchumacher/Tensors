#ifdef LTEMPLATE_H

    
    template<typename Scalar, typename Int, IsFloat(Scalar)>
    inline mma::TensorRef<mreal> to_MTensorRef( const TENSOR_T<Scalar,Int> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
    template<typename J, typename Int, IsInt(J)>
    inline mma::TensorRef<mint> to_MTensorRef( const TENSOR_T<J,Int> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
#endif
