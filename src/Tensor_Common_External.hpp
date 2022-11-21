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




//template <typename Scalar, typename Int>
//std::string to_string(const TENSOR_T<Scalar,Int> & x)
//{
//    return x.ToString();ToString
//}


//template <typename Scalar, typename Int>
//void AXPY( const Scalar a, const TENSOR_T<Scalar,Int> & x, TENSOR_T<Scalar,Int> & y )
//{
//    mint n = x.Size();
//    mint stride_x = 1;
//    mint stride_y = 1;
//    cblas_daxpy( n, a, x.data(), stride_x, y.data(), stride_y );
//}

//template <typename Scalar, typename Int>
//void Scale( const Scalar a, TENSOR_T<Scalar,Int> & x )
//{
//    mint n = x.Size();
//    mint stride_x = 1;
//    cblas_dscal( n, a, x.data(),  stride_x );
//}






//
//template<>
//inline void TENSOR_T<double>::AddFrom( const double * const a_ )
//{
//    double alpha = static_cast<double>(1);
//    mint n = Size();
//    mint stride_a = 1;
//    mint stride_a_ = 1;
//    cblas_daxpy( n, alpha, a_, stride_a_, a, stride_a );
//}
//
//template<>
//inline void TENSOR_T<float>::AddFrom( const float * const a_ )
//{
//    float alpha = static_cast<float>(1);
//    mint n = Size();
//    mint stride_a = 1;
//    mint stride_a_ = 1;
//    cblas_saxpy( n, alpha, a_, stride_a_, a, stride_a );
//}

