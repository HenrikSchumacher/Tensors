#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Int, typename ExtInt, typename LInt>
        Sparse::BinaryMatrixCSR<LInt,LInt> CompressorCSR(
            const ExtInt  * const * const idx,
            const ExtInt  * const * const jdx,
            const LInt    * const entry_counts,
            const Int list_count,
            const Int m,
            const Int n,
            const Int final_thread_count
        )
        {
            static_assert(IntQ<ExtInt>,"");
            
            Tensor1<LInt,LInt> ran_ptr( Ramp(list_count) + Int(1) );
            ran_ptr[0] = 0;
            
            for( Int i = 0; i < list_count; ++i )
            {
                ran_ptr[i+1] = ran_ptr[i] + entry_counts[i];
            }
            
            const LInt C_n = ran_ptr.Last();
            
            Tensor1<ExtInt,LInt> ran_buffer( C_n );
            ran_buffer.iota();
            Tensor1<LInt *,Int> ran (list_count);
            
            for( Int i = 0; i < list_count; ++i )
            {
                ran[i] = &ran_buffer[ran_ptr[i]];
            }
            
            MatrixCSR<LInt,Int,LInt> A (
                idx, jdx, ran.data(),
                entry_counts, list_count, m, n, final_thread_count,
                false, false
            );
            
            auto [A_outer,A_inner,A_values,A_m,A_n] = A.Disband();
            TOOLS_DUMP( A_inner );
            TOOLS_DUMP( A_values );
            
            Aggregator<LInt,LInt> outer_agg ( A_inner.Size() +1 );
            outer_agg.Push(0);
            
            for( LInt k = 1; k < A_inner.Size(); ++k )
            {
                if( A_inner[k] != A_inner[k-1] )
                {
                    outer_agg.Push(k);
                }
            }
            outer_agg.Push(A_inner.Size());
            
            auto outer = outer_agg.Get();
            const LInt C_m = outer.Size() - LInt(1);
            
            Sparse::BinaryMatrixCSR<LInt,LInt> C (
                std::move(outer),
                std::move(A_values),
                C_m, C_n, final_thread_count
             );
            
            TOOLS_DUMP(C.Outer());
            TOOLS_DUMP(C.Inner());
            TOOLS_DUMP(C.RowCount());
            TOOLS_DUMP(C.ColCount());
            C.SortInner();
            
            return C;
        }
        
        
        template<typename ExtScal, typename Int, typename ExtInt, typename LInt>
        Sparse::BinaryMatrixCSR<Int,LInt> CompressorCSR(
            const ExtInt  * const * const idx,
            const ExtInt  * const * const jdx,
            const ExtScal * const * const val,
            const LInt   * const entry_counts,
            const Int list_count,
            const Int m,
            const Int n,
            const Int final_thread_count
        )
        {
            static_assert(IntQ<ExtInt>,"");
            
            return CompressorCSR(
                idx, jdx, entry_counts, list_count, m, n, final_thread_count
            );
        }
        
        
        template<typename ExtScal, typename ExtInt, typename Int, typename LInt>
        Sparse::BinaryMatrixCSR<Int,LInt> CompressorCSR(
              cref<TripleAggregator<ExtInt,ExtInt,ExtScal,LInt>> triples,
              const Int m,
              const Int n,
              const Int final_thread_count
        )
        {
            static_assert(IntQ<ExtInt>,"");
            
            LInt entry_counts = int_cast<LInt>(triples.Size());
            
            const ExtInt  * const i = triples.Get_0().data();
            const ExtInt  * const j = triples.Get_1().data();

            return CompressorCSR(
                &i, &j, &entry_counts, Int(1), m, n, final_thread_count
            );
            
        }
    }
    
}
