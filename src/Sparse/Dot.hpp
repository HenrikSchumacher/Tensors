#pragma once

namespace Tensors
{
    namespace Sparse
    {
        namespace Details
        {
            template<typename Scal>
            struct DotResult
            {
                Scal value;
                bool structurally_nonzeroQ;
            };
            
            template<Op opu = Op::Id, Op opv = Op::Id, 
                typename u_Scal, typename u_Int,
                typename v_Scal, typename v_Int
            >
            DotResult<decltype(u_Scal(1) * v_Scal(1))> Dot(
                cptr<u_Int> u_idx, mptr<u_Scal> u_val, const Size_T u_nnz,
                cptr<v_Int> v_idx, mptr<v_Scal> v_val, const Size_T v_nnz,
                const Size_T n
            )
            {
                static_assert(NotTransposedQ(opu), "");
                static_assert(NotTransposedQ(opv), "");
                
                using Scal = decltype(u_Scal(1) * v_Scal(1));
                using Int  = decltype(u_Int (0) + v_Int (0));
                
                DotResult<Scal> result { Scal{0}, false };
                
                if( (n <= 0) ||(u_nnz <= 0) || (v_nnz <= 0) )
                {
                    
                    return result;
                }
                
                const Int u_last = u_idx[u_nnz-1];
                const Int v_last = v_idx[v_nnz-1];
                
                Size_T u_ptr {0};
                Size_T v_ptr {0};
                
                Int u_i = u_idx[u_ptr];
                Int v_i = v_idx[v_ptr];
                
                if( (u_last < v_i) || (v_last < u_i) )
                {
                    return result;
                }
                
                while( true )
                {
                    if( u_i < v_i )
                    {
                        ++u_ptr;
                        
                        if( u_ptr >= u_nnz )
                        {
                            return result;
                        }
                        else
                        {
                            u_i = u_idx[u_ptr];
                            
                            if( v_last < u_i )
                            {
                                return result;
                            }
                        }
                    }
                    else if( u_i > v_i )
                    {
                        ++v_ptr;
                        
                        if( v_ptr >= v_nnz )
                        {
                            return result;
                        }
                        else
                        {
                            v_i = v_idx[v_ptr];
                            
                            if( u_last < v_i )
                            {
                                return result;
                            }
                        }
                    }
                    else // if( u_i == v_i )
                    {
                        result.structurally_nonzeroQ = true;
                        
                        result.value += Scalar::Op<opu>(u_val[u_ptr]) * Scalar::Op<opu>(v_val[v_ptr]);
                        
                        ++u_ptr;
                        
                        if( u_ptr >= u_nnz )
                        {
                            return result;
                        }
                        else
                        {
                            u_i = u_idx[u_ptr];
                            
                            if( v_last < u_i )
                            {
                                return result;
                            }
                        }
                        
                        ++v_ptr;
                        
                        if( v_ptr >= v_nnz )
                        {
                            return result;
                        }
                        else
                        {
                            v_i = v_idx[v_ptr];
                            
                            if( u_last < v_i )
                            {
                                return result;
                            }
                        }
                    }
                }
            }
            
        } // namespace Details
        
        template<Op opu = Op::Id, Op opv = Op::Id,
            typename u_Scal, typename u_Int,
            typename v_Scal, typename v_Int
        >
        decltype(u_Scal(1) * v_Scal(1)) Dot(
            cptr<u_Int> u_idx, mptr<u_Scal> u_val, const Size_T u_nnz,
            cptr<v_Int> v_idx, mptr<v_Scal> v_val, const Size_T v_nnz,
            const Size_T n
        )
        {
            return Sparse::Details::Dot<opu,opv>( u_idx, u_val, u_nnz, v_idx, v_val, v_nnz, n ).value;
        }
        
        
        namespace Details
        {
            template<
                Op opA, Op opB,
                typename A_Scal, typename A_Int, typename A_LInt,
                typename B_Scal, typename B_Int, typename B_LInt,
                typename Scal = decltype(A_Scal(1) * B_Scal(1)),
                typename Int  = decltype(A_Int (0) + B_Int (0)),
                typename LInt = decltype(A_LInt(0) + B_LInt(0))
            >
            MatrixCSR<Scal,Int,LInt>
            Dot_NN(
                const MatrixCSR<A_Scal,A_Int,A_LInt> & A,
                const MatrixCSR<B_Scal,B_Int,B_LInt> & B
            )
            {
                static_assert(NotTransposedQ(opA),"");
                static_assert(NotTransposedQ(opB),"");
             
                wprint( "Sparse::Details::Dot_NN: Implemented not tested, yet.");
                
                if( A.WellFormedQ() )
                {
                    auto job_ptr = A.JobPtr();
                    
                    const Int thread_count = A.ThreadCount();
                    
                    const Int m = A.RowCount();
                    
                    Tensor2<LInt,Int> counters ( thread_count, m, static_cast<LInt>(0) );
                    
                    // Expansion phase, utilizing counting sort to generate expanded row pointers and column indices.
                    // https://en.wikipedia.org/wiki/Counting_sort
                    
                    
                    cptr<A_LInt> A_outer = A.Outer().data();
                    cptr<A_Int > A_inner = A.Inner().data();
                    cptr<A_Scal> A_value = A.Value().data();
                    
                    cptr<A_LInt> B_outer = B.Outer().data();
                    cptr<A_Int > B_inner = B.Inner().data();
                    cptr<A_Scal> B_value = B.Value().data();
                    
                    ParallelDo(
                        [=,&job_ptr,&counters]( const Int thread )
                        {
                            const A_Int i_begin = job_ptr[thread  ];
                            const A_Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            for( A_Int i = i_begin; i < i_end; ++i )
                            {
                                LInt c_i = 0;
                                
                                const A_LInt jj_begin = A_outer[i  ];
                                const A_LInt jj_end   = A_outer[i+1];
                                
                                for( A_LInt jj = jj_begin; jj < jj_end; ++jj )
                                {
                                    const A_Int j = A_inner[jj];
                                    
                                    c_i += (B_outer[j+1] - B_outer[j]);
                                }
                                
                                c[i] = c_i;
                            }
                        },
                        thread_count
                    );
                    
                    AccumulateAssemblyCounters_Parallel( counters );
                    
                    const LInt nnz = counters.data(thread_count-1)[m-1];
                    
                    Sparse::MatrixCSR<Scal,Int,LInt> C ( m, B.ColCount(), nnz, thread_count );
                    
                    copy_buffer( counters.data(thread_count-1), &C.Outer().data()[1], m );

                    mptr< Int> C_inner  = C.Inner().data();
                    mptr<Scal> C_values = C.Value().data();
                    
                    ParallelDo(
                        [=,&job_ptr,&counters]( const Int thread )
                        {
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            mptr<LInt> c = counters.data(thread);
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const A_LInt jj_begin = A_outer[i  ];
                                const A_LInt jj_end   = A_outer[i+1];
                                
                                for( A_LInt jj = jj_begin; jj < jj_end; ++jj )
                                {
                                    const A_Int j = A_inner[jj];
                                    
                                    const B_LInt kk_begin = B_outer[j  ];
                                    const B_LInt kk_end   = B_outer[j+1];
                                    
                                    for( B_LInt kk = kk_end; kk --> kk_begin; )
                                    {
                                        const Int  k   = B_inner[kk];
                                        const LInt pos = --c[i];
                                        
                                        C_inner [pos] = k;
                                        C_values[pos] = Scalar::Op<opA>(A_value[jj]) 
                                                        *
                                                        Scalar::Op<opB>(B_value[kk]);
                                    }
                                }
                            }
                        },
                        thread_count
                    );
                    
                    // Finished expansion phase (counting sort).
                    
                    // Finally we row-sort inner and compressQ duplicates in inner and values.
                    C.Compress();
                    
                    return C;
                }
                else
                {
                    eprint("Sparse::Details::Dot_NN: Matrix A is not well-formed.");
                    
                    return Sparse::MatrixCSR<Scal,Int,LInt>();
                }
            }
            
        }  // namespace Details
        
        template<
            Op opA, Op opB,
            typename A_Scal, typename A_Int, typename A_LInt,
            typename B_Scal, typename B_Int, typename B_LInt,
            typename Scal = decltype(A_Scal(1) * B_Scal(1)),
            typename Int  = decltype(A_Int (0) + B_Int (0)),
            typename LInt = decltype(A_LInt(0) + B_LInt(0))
        >
        MatrixCSR<Scal,Int,LInt>
        Dot(
            const MatrixCSR<A_Scal,A_Int,A_LInt> & A,
            const MatrixCSR<B_Scal,B_Int,B_LInt> & B
        )
        {
            if constexpr( NotTransposedQ(opA) )
            {
                if constexpr( NotTransposedQ(opB) )
                {
                    return Sparse::Details::Dot_NN<opA,opB>( A, B );
                }
                else
                {
                    return Sparse::Details::Dot_NN<opA,Op::Id>(
                        A, B.template Op<opB>()
                    );
                }
            }
            else
            {
                if constexpr( NotTransposedQ(opB) )
                {
                    return Sparse::Details::Dot_NN<Op::Id,opB>(
                        A.template Op<opA>(), B
                    );
                }
                else
                {
                    return Sparse::Details::Dot_NN<Op::Id,Op::Id>( 
                        A.template Op<opA>(), B.template Op<opB>()
                    );
                }
            }
        }

        
//        template<typename Scal, typename Int>
//        Scal Dot_BinarySearch(
//            cptr<Int> u_idx, mptr<Scal> u_val, const Int u_nnz,
//            cptr<Int> v_idx, mptr<Scal> v_val, const Int v_nnz,
//            const Int n
//        )
//        {
//            if( (u_nnz == 0) || (v_nnz == 0) )
//            {
//                return Scal(0);
//            }
//
//            if( u_nnz > v_nnz )
//            {
//                return Dot_BinarySearch(
//                    v_idx, v_val, v_nnz,
//                    u_idx, u_val, u_nnz,
//                    n
//                );
//            }
//            
//            // Assuming that u_nnz <= v_nnz.
//            
//            Size_T v_ptr = 0;
//            Size_T pos   = 0;
//            
//            Scal result {0};
//            
//            for( Int i = 0; i < u_nnz; ++i)
//            {
//                const bool found = BinarySearch( &v_idx[v_ptr], v_nnz - v_ptr, u_idx[i], pos );
//                
//                v_ptr += pos;
//                
//                if( found )
//                {
//                    result += u_val[i] * v_val[v_ptr];
//                }
//            }
//            
//            return result;
//        }

        
    } // namespace Sparse
    
} // namespace Tensors
