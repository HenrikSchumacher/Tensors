//###########################################################################################
//####          Elimination tree
//###########################################################################################
            
public:
    
    const Tree<Int> & EliminationTree()
    {
        // TODO: How can this be parallelized?
        
        // TODO: read Kumar, Kumar, Basu - A parallel algorithm for elimination tree computation and symbolic factorization
        
        if( ! eTree_initialized )
        {
            ptic(ClassName()+"::EliminationTree");
            
            // See Bollhöfer, Schenk, Janalik, Hamm, Gullapalli - State-of-the-Art Sparse Direct Solvers
            
            // I want to make it possible to use unsigned integer types for Int.
            // Hence using -1 as "no_element" is not an option.
            // We have to use something else instead of 0 to mark empty places.
            const Int no_element = n;

            Tensor1<Int,Int> parents ( n, no_element );
            
            // A vector for path compression.
            Tensor1<Int,Int> a ( n, no_element );
            
            ptr<LInt> A_diag  = A.Diag().data();
            ptr<LInt> A_outer = A.Outer().data();
            ptr< Int> A_inner = A.Inner().data();

            for( Int k = 1; k < n; ++k )
            {
                // We need visit all i < k with A_ik != 0.
                const LInt l_begin = A_outer[k];
                const LInt l_end   = A_diag [k];
//                        const LInt l_end   = A_diag[k+1]-1;
                
                for( LInt l = l_begin; l < l_end; ++l )
                {
                    Int i = A_inner[l];
                    
                    while( ( i != no_element ) && ( i < k ) )
                    {
                        Int j = a[i];
                        
                        a[i] = k;
                        
                        if( j == no_element )
                        {
                            parents[i] = k;
                        }
                        i = j;
                    }
                }
            }

            eTree = Tree<Int> ( std::move(parents), thread_count );

            eTree_initialized = true;
            
            ptoc(ClassName()+"::EliminationTree");
        }
        
        return eTree;
    }
