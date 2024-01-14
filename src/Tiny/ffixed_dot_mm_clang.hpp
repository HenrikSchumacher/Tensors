#pragma once

namespace Tensors
{
        
    namespace Tiny
    {
        template<Size_T M, Size_T K, Size_T N, AddTo_T addto, typename Scal>
        void fixed_dot_mm_clang( cptr<Scal> A, cptr<Scal> B, mptr<Scal> C )
        {
            static_assert( mat_enabledQ && (SameQ<Scal,double> || SameQ<Scal,float>), "Chosen scalar type is illegal for clang matrix extension." );
    
            // Caution: mat_T is column-major!
            // So we have to compute C^T = B^T * A^T.
            
            if constexpr ( addto == AddTo_T::True )
            {
                (*reinterpret_cast<mat_T<N,M,Scal>*>(C))
                +=
                (*reinterpret_cast<const mat_T<N,K,Scal>*>(B))
                *
                (*reinterpret_cast<const mat_T<K,M,Scal>*>(A));
            }
            else
            {
                (*reinterpret_cast<mat_T<N,M,Scal>*>(C))
                =
                (*reinterpret_cast<const mat_T<N,K,Scal>*>(B))
                *
                (*reinterpret_cast<const mat_T<K,M,Scal>*>(A));
            }
        }
        
        
        template<Size_T M, Size_T K, Size_T N, AddTo_T addto, typename Scal>
        void fixed_dot_mm_clang_2( cptr<Scal> A, cptr<Scal> B, mptr<Scal> C )
        {
            static_assert( mat_enabledQ && (SameQ<Scal,double> || SameQ<Scal,float>), "Chosen scalar type is illegal for clang matrix extension." );
    
            // Caution: mat_T is column-major!
            // So we have to compute C^T = B^T * A^T.
            
            mat_T<K,M,Scal> a = __builtin_matrix_column_major_load( A, K, M, K );
            mat_T<N,K,Scal> b = __builtin_matrix_column_major_load( B, N, K, N );
            
            
            if constexpr ( addto == AddTo_T::True )
            {
                mat_T<N,M,Scal> c = __builtin_matrix_column_major_load( C, N, M, N ) + b * a;
                
                __builtin_matrix_column_major_store( c, C, N );
            }
            else
            {
                mat_T<N,M,Scal> c =  b * a;
                
                __builtin_matrix_column_major_store( c, C, N );
            }
        }
        
    } // namespace Tiny
    
} // namespace Tensors

