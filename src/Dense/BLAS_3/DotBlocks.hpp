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
        [=]( const Int thread )
        {
            const Int M_blk_begin = JobPointer( M_BlockCount(), thread_count, thread     );
            const Int M_blk_end   = JobPointer( M_BlockCount(), thread_count, thread + 1 );

            A_Block_T a;
            B_Block_T b;
            C_Block_T c;

            for( Int M_blk = M_blk_begin; M_blk < M_blk_end; ++M_blk )
            {
                for( Int N_blk = 0; N_blk < N_BlockCount(); ++N_blk )
                {
                    c.SetZero();
                    
                    for( Int K_blk = 0; K_blk < K_BlockCount(); ++K_blk )
                    {
                        a.Read( AP.data(M_blk,K_blk) );
                        b.Read( BP.data(N_blk,K_blk) );

//                        copy_buffer<m*k>( &AP_[ldAP * M_blk + a_size * K_blk], a.data() );
//                        copy_buffer<k*n>( &BP_[ldBP * N_blk + b_size * K_blk], b.data() );
//                        
                        Dot<AddTo>( a, b, c );
                    }
                    
                    c.Write( CP.data(M_blk,N_blk) );
                    
//                    copy_buffer<m*n>( c.data(), &CP_[ldCP * M_blk + c_size * N_blk] );
                }
            }
        },
        thread_count
    );
    
    ptoc(ClassName()+"::DotBlocks");
}
