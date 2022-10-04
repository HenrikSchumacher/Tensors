#ifdef LTEMPLATE_H

    
    template<typename T, typename I, IsFloat(T)>
    inline mma::TensorRef<mreal> to_MTensorRef( const TENSOR_T<T,I> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mreal>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
    template<typename J, typename I, IsInt(J)>
    inline mma::TensorRef<mint> to_MTensorRef( const TENSOR_T<J,I> & A )
    {
        const mint r = A.Rank();
        Tensor1<mint,mint> dims_ (r);
        dims_.Read(A.Dimensions());
        
        auto B = mma::makeTensor<mint>( r, dims_.data() );
        A.Write(B.data());
        return B;
    }
    
#endif




//template <typename T, typename I>
//std::string to_string(const TENSOR_T<T,I> & x)
//{
//    return x.ToString();ToString
//}


//template <typename T, typename I>
//void AXPY( const T a, const TENSOR_T<T,I> & x, TENSOR_T<T,I> & y )
//{
//    mint n = x.Size();
//    mint stride_x = 1;
//    mint stride_y = 1;
//    cblas_daxpy( n, a, x.data(), stride_x, y.data(), stride_y );
//}

//template <typename T, typename I>
//void Scale( const T a, TENSOR_T<T,I> & x )
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

