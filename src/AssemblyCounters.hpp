#pragma once

namespace Tensors {
    
    template<typename T, typename Int>
    inline void AccumulateAssemblyCounters( Tensor2<T,Int> & counters )
    {
        ptic("AccumulateAssemblyCounters");

        const Int thread_count = counters.Dimension(0);

        const Int m = counters.Dimension(1);

        for( Int thread = 1; thread < thread_count; ++thread )
        {
            counters(thread, 0) += counters(thread-1, 0);
        }
        for( Int i = 1; i < m; ++i )
        {
            counters(0, i) += counters(thread_count-1, i-1);

            for( Int thread = 1; thread < thread_count; ++thread )
            {
                counters(thread, i) += counters(thread-1, i);
            }
        }

        ptoc("AccumulateAssemblyCounters");
    }


    template<typename T, typename Int>
    inline void AccumulateAssemblyCounters_Parallel( Tensor2<T,Int> & counters )
    {
        static_assert(CACHE_LINE_WIDTH % sizeof(T) == 0, "CACHE_LINE_WIDTH is not divisible by sizeof(T)");

        constexpr Int per_line = CACHE_LINE_WIDTH / sizeof(T);

        ptic("AccumulateAssemblyCounters (parallel)");
        
        const Int thread_count = counters.Dimension(0);
        
        const Int            m = counters.Dimension(1);
        
        const Int line_count = (m * sizeof(T) + CACHE_LINE_WIDTH - 1 ) / CACHE_LINE_WIDTH;
        
    //        valprint("line_count",line_count);
        
        T * S_buffer = nullptr;
        safe_alloc(S_buffer,thread_count+1);
        T * restrict const S = S_buffer;
        S[0] = static_cast<T>(0);

        const Int step = line_count / thread_count;
        const Int corr = line_count % thread_count;
        
    //        for( Int thread = 0; thread < thread_count; ++thread )
    //        {
    //            // each thread does the accumulation on its chunk independently
    //            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
    //            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);
    //
    //            print("thread = "+ToString(thread)+", j_begin = "+ToString(j_begin)+", j_end = "+ToString(j_end));
    //        }
        
    //        tic("local acc");
        #pragma omp parallel for num_threads( thread_count )
        for( Int thread = 0; thread < thread_count; ++thread )
        {
            // each thread does the accumulation on its chunk independently
            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);
            
            if( j_end > j_begin )
            {
                for( Int i = 1; i < thread_count; ++i )
                {
                    counters[i][j_begin] += counters[i-1][j_begin];
                }
                
                for( Int j = j_begin+1; j < j_end; ++j )
                {
                    counters[0][j] += counters[thread_count-1][j-1];
                    
                    for( Int i = 1; i < thread_count; ++i )
                    {
                        counters[i][j] += counters[i-1][j];
                    }
                }
                
                S[thread+1] = counters(thread_count-1,j_end-1);
            }
            else
            {
                S[thread+1] = static_cast<T>(0);
            }
        }
    //        toc("local acc");
        
    //        for( Int i = 0; i < thread_count; ++i )
    //        {
    //            valprint("S[i]",S[i]);
    //        }
        // scan through the last results of each chunk
        {
            T s_local = static_cast<T>(0);
            for( Int i = 0; i < thread_count; ++i )
            {
                s_local += S[i+1];
                S[i+1] = s_local;
            }
        }

    //        for( Int i = 0; i < thread_count; ++i )
    //        {
    //            valprint("S[i]",S[i]);
    //        }
    //
    //        tic("correction");
        #pragma omp parallel for num_threads( thread_count )
        for( Int thread = 0; thread < thread_count; ++ thread )
        {
            // each thread adds-in its correction
            const T correction = S[thread];
            
            const Int j_begin = (step*(thread  ) + (corr*(thread  ))/thread_count) * per_line;
            const Int j_end   = std::min(m, (step*(thread+1) + (corr*(thread+1))/thread_count) * per_line);

            
            for( Int i = 0; i < thread_count; ++i )
            {
                T * restrict const c_i = counters.data(i);
                
                #pragma omp simd
                for( Int j = j_begin; j < j_end; ++j )
                {
                    c_i[j] += correction;
                }
            }
            
        }
    //        toc("correction");
        
        ptoc("AccumulateAssemblyCounters (parallel)");
    }




    template<typename T, typename Int>
    inline Tensor2<T,Int> AssemblyCounters(
        const Int * const * const idx,
        const Int * const * const jdx,
        const Int * entry_counts,
        const Int list_count,
        const Int m,
        const int symmetrize = 0
    )
    {
        ptic("AssemblyCounters");
        
        Tensor2<T,Int> counters (list_count, m, static_cast<Int>(0));

        // https://en.wikipedia.org/wiki/Counting_sort
        // using parallel count sort to sort the cluster (i,j)-pairs according to i.
        // storing counters of each i-index in thread-interleaved format
        // TODO: Improve data layout (transpose counts).
        #pragma omp parallel for num_threads( list_count )
        for( Int thread = 0; thread < list_count; ++thread )
        {
            const Int * restrict const thread_idx = idx[thread];
            const Int * restrict const thread_jdx = jdx[thread];
            
            const Int entry_count = entry_counts[thread];
            
            T * restrict const c = counters.data(thread);
            
            if( symmetrize!=0 )
            {
                for( Int k = 0; k < entry_count; ++k )
                {
                    const Int i = thread_idx[k];
                    const Int j = thread_jdx[k];
                    
                    c[i] ++;
                    c[j] += static_cast<Int>(i != j);
                }
            }
            else
            {
                for( Int k = 0; k < entry_count; ++k )
                {
                    const Int i = thread_idx[k];
                    
                    ++c[i];
                }
            }
        }
        
    //        print(counters.ToString());
    //        AccumulateAssemblyCounters(counters);
        AccumulateAssemblyCounters_Parallel<T,Int>(counters);
        
    //        print(counters.ToString());
        
        ptoc("AssemblyCounters");
        
        return counters;
    }
    
}
