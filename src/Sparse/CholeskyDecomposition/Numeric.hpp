#pragma once

//###########################################################################################
//####          Supernodal numeric factorization
//###########################################################################################
            
public:
    
    template<typename ExtScal>
    void NumericFactorization(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization<" + TypeName<ExtScal> + ">";
       
        ptic(tag);
        
        SymbolicFactorization();

        reg = reg_;
//        A_inner_perm.Permute( A_val_, A_val.data(), Inverse::False );
        
        ParallelDo(
            [&]( const LInt i )
            {
                A_val[i] = static_cast<Scal>(A_val_[A_inner_perm[i]]);
            },
            A_inner_perm.Size(), static_cast<LInt>(thread_count)
        );
        
        ptic(tag + ": Zerofy buffers.");
        SN_tri_val.SetZero( thread_count );
        SN_rec_val.SetZero( thread_count );
        ptoc(tag + ": Zerofy buffers.");
        
        ptic(tag + ": Initialize factorizers");
        
        std::vector<std::unique_ptr<Factorizer>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer>(*this);
            },
            thread_count
        );
        
        ptoc(tag + ": Initialize factorizers");
        
        
        dump(thread_count);
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<Parallel>( SN_list  );
        
        SN_factorized = true;
        
        ptoc(tag);
        
    }
