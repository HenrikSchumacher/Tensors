protected:

template<typename ExtInt>
void FromPairs(
    const  ExtInt * const * const idx,
    const  ExtInt * const * const jdx,
    const LInt            *       entry_counts,
    const  Int list_count,
    const  Int final_thread_count,
    const bool compressQ = true,
    const int  symmetrizeQ = 0
)
{
    TOOLS_PTIC(ClassName()+"::FromPairs");
    
    // TODO: If ExtInt is wider than Int, we need to check that all entries of idx and jdx fit into an Int.
    
    Tensor2<LInt,Int> counters = AssemblyCounters<LInt,Int>(
        idx, jdx, entry_counts, list_count, m, symmetrizeQ
    );
    
    const LInt nnz = counters(list_count-1,m-1);
    
    if( nnz > LInt(0) )
    {
        inner = Tensor1<Int,LInt>( nnz );
        
        mptr<LInt> A_o = outer.data();
        mptr<Int>  A_i = inner.data();
        
        copy_buffer( counters.data(list_count-1), &A_o[1], m );
        
        // writing the j-indices into sep_column_indices
        // the counters array tells each thread where to write
        // since we have to decrement entries of counters array, we have to loop in reverse order to make the sort stable in the j-indices.

        if( symmetrizeQ != 0 )
        {
            ParallelDo(
                [A_o,A_i,&counters,&entry_counts,&idx,&jdx]( const Int thread )
                {
                    const LInt entry_count = entry_counts[thread];
                    
                    cptr<ExtInt> thread_idx = idx[thread];
                    cptr<ExtInt> thread_jdx = jdx[thread];
                    
                    mptr<LInt> c = counters.data(thread);
                    
                    for( LInt k = entry_count; k --> LInt(0); )
                    {
                        const Int i = static_cast<Int>(thread_idx[k]);
                        const Int j = static_cast<Int>(thread_jdx[k]);
                        {
                            const LInt pos = --c[i];
                            A_i[pos] = j;
                        }
                        
                        c[j] -= static_cast<LInt>(i != j);
                        
                        const LInt pos = c[j];
                        
                        A_i[pos] = i;
                    }
                },
                list_count
            );
        }
        else
        {
            ParallelDo(
                [A_o,A_i,&counters,&entry_counts,&idx,&jdx](
                    const Int thread
                )
                {
                    const LInt entry_count = entry_counts[thread];
                    
                    cptr<Int> t_idx = idx[thread];
                    cptr<Int> t_jdx = jdx[thread];
                    
                    mptr<LInt> c = counters.data(thread);
                    
                    for( LInt k = entry_count; k --> LInt(0); )
                    {
                        const Int i = t_idx[k];
                        const Int j = t_jdx[k];
                        {
                            const LInt pos = --c[i];
                            A_i[pos] = j;
                        }
                    }
                },
                list_count
            );
        }
        
        // From here on, we may use as many threads as we want.
        SetThreadCount( final_thread_count );
                            
        // We have to sort b_inner to be compatible with the CSR format.
        SortInner();
        
        if( compressQ )
        {
            Compress();
        }
        else
        {
            proven_duplicate_freeQ = false;
        }
        
    }
    else
    {
        SetThreadCount( final_thread_count );
    }
    
    TOOLS_PTOC(ClassName()+"::FromPairs");
}
