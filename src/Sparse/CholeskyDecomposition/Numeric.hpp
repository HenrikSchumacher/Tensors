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
        NumericFactorization_Multifrontal( A_val_, reg_ );
    }
    
    template<typename ExtScal>
    void NumericFactorization_Multifrontal(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_Multifrontal<" + TypeName<ExtScal> + ">";
       
        ptic(tag);
        
        SymbolicFactorization();

        this->ClearCache();
        
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
        
        
        ptic(tag + ": Initialize update buffers.");
        
        // TODO: We might want to use memory stacks or pools
        // TODO: if we don't want to rely on the sytem's way to handle the memory.
        // TODO: Each thread should probably get its own memory stack.
        // TODO: This stack is a expandable container.
        // TODO: Supernodes are visited in depth-first order.
        // TODO: When visited for the first time,
        // TODO: the supernode's update block is pushed to stack.
        // TODO: After a supernode has fetched the updates from its children,
        // TODO: the childrens' memory is popped.
        // TODO: Children need to know: A pointer to their block and maybe also the thread
        // TODO: on which the block was allocated.
        // TODO: Both might be fused into a simple struct.
        
        SN_updates = std::vector<Update_T> ( SN_count );
        
        ptoc(tag + ": Initialize update buffers.");
        
        
        ptic(tag + ": Initialize factorizers");
        
        std::vector<std::unique_ptr<Factorizer_MF_T>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_MF_T>(*this);
            },
            thread_count
        );
        
        ptoc(tag + ": Initialize factorizers");
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<Parallel>( SN_list  );
        
        SN_factorized = true;
        
        ptic(tag + ": Release update buffers.");
//        SN_up_ptr = Tensor1<LInt, Int>(  Int(0) );
//        SN_up_val = Tensor1<Scal,LInt>( LInt(0) );
        
        SN_updates = std::vector<Update_T> ( LInt(0) );
        
        ptoc(tag + ": Release update buffers.");
        
        ptoc(tag);
        
    }


    template<typename ExtScal>
    void NumericFactorization_LeftLooking(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_LeftLooking<" + TypeName<ExtScal> + ">";
       
        ptic(tag);
        
        SymbolicFactorization();

        this->ClearCache();
        
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
        
        std::vector<std::unique_ptr<Factorizer_LL_T>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_LL_T>(*this);
            },
            thread_count
        );
        
        ptoc(tag + ": Initialize factorizers");
        
        // Parallel traversal in postorder
        aTree.template Traverse_Postordered<Parallel>( SN_list  );
        
        SN_factorized = true;
        
        ptoc(tag);
        
    }
