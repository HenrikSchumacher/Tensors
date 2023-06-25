#pragma once

//###########################################################################################
//####          Supernodal numeric factorization
//###########################################################################################
            
public:
    
    template<typename ExtScal>
    void NumericFactorization(
        ptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        ptic(ClassName()+"::NumericFactorization");
        
//        dump(openblas_get_num_threads());
//        openblas_set_num_threads(1);
//        dump(openblas_get_num_threads());
        
        SymbolicFactorization();

        reg = reg_;
        A_inner_perm.Permute( A_val_, A_val.data(), Inverse::False );
        
        ptic("Zerofy buffers.");
        SN_tri_val.SetZero( thread_count );
        SN_rec_val.SetZero( thread_count );
        ptoc("Zerofy buffers.");
        
        ptic("Initialize factorizers");
        std::vector<std::unique_ptr<Factorizer>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer>(*this);
            },
            thread_count
        );
        
        ptoc("Initialize factorizers");
        
        // Parallel traversal in postorder
        aTree.Traverse_DFS_Parallel( SN_list, tree_top_depth );
        
        SN_factorized = true;
        
        ptoc(ClassName()+"::NumericFactorization");
        
    }
