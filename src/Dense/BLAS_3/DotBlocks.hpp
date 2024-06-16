void DotBlocks()
{
    ptic(ClassName()+"::DotBlocks");
    
    //TODO: Recursive implementation using quadtree for C.
    
//    const Int M_blk_max = M_BlockCount();
//    const Int N_blk_max = N_BlockCount();
//    const Int K_blk_max = K_BlockCount();
        
//    cptr<A_T> AP_ = AP.blocks.data();
//    cptr<B_T> BP_ = BP.blocks.data();
//    mptr<C_T> CP_ = CP.blocks.data();
//        
//    const Int ldAP = a_size * AP.N_BlockCount();
//    const Int ldBP = b_size * BP.N_BlockCount();
//    const Int ldCP = c_size * CP.N_BlockCount();
    
    ParallelDo(
        [=,this]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_BlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_BlockCount(), thread_count, thread + 1 );

            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_BlockCount(); ++N_blk )
                {
                    C_Block_T c ( C_T(0) );

                    for( Int K_blk = 0; K_blk < K_BlockCount(); ++K_blk )
                    {
//                        copy_buffer<m*k>( &AP_[ldAP * M_blk + a_size * K_blk], a.data() );
//                        copy_buffer<k*n>( &BP_[ldBP * N_blk + b_size * K_blk], b.data() );
                        
                        A_Block_T a ( AP.data(M_blk,K_blk) );
                        B_Block_T b ( BP.data(N_blk,K_blk) );

//
                        Dot<AddTo>( a, b, c );
                    }

//                    copy_buffer<m*n>( c.data(), &CP_[ldCP * M_blk + c_size * N_blk] );
                    
                    c.Write( CP.data(M_blk,N_blk) );



//                    for( Int K_blk = 0; K_blk < K_BlockCount(); ++K_blk )
//                    {
//                        if constexpr( mat_enabledQ && (SameQ<A_T,double> || SameQ<A_T,float>) && (SameQ<B_T,double> || SameQ<B_T,float>) && (SameQ<C_T,double> || SameQ<C_T,float>) )
//                        {
//                            Tiny::fixed_dot_mm_clang<m,n,k,AddTo>(
//                                AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), CP.data(M_blk,N_blk)
//                            );
//                        }
//                        else
//                        {
//                            Tiny::fixed_dot_mm_vec<m,n,k,AddTo>(
//                                AP.data(M_blk,K_blk), BP.data(N_blk,K_blk), CP.data(M_blk,N_blk)
//                            );
//                        }
//                    }
                    
                }
            }
        },
        thread_count
    );
    
    ptoc(ClassName()+"::DotBlocks");
}
