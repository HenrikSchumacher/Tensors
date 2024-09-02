#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Scal, typename Int>
        Scal Dot(
            cptr<Int> u_idx, mptr<Scal> u_val, const Int u_nnz,
            cptr<Int> v_idx, mptr<Scal> v_val, const Int v_nnz,
            const Int n
        )
        {
            if( (u_nnz == 0) || (v_nnz == 0) )
            {
                return Scal(0);
            }

            const Int u_last = u_idx[u_nnz-1];
            const Int v_last = v_idx[v_nnz-1];
            
            Int u_ptr {0};
            Int v_ptr {0};
            
            Int u_i = u_idx[u_ptr];
            Int v_i = v_idx[v_ptr];
            
            if( (u_last < v_i) || (v_last < u_i) )
            {
                return Scal(0);
            }
            

            Scal result {0};

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
                else
                {
                    result += u_val[u_ptr] * v_val[v_ptr];

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
        
    } // namespace Sparse
    
} // namespace Tensors
