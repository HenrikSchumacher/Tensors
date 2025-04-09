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
    
        TOOLS_PTIC(tag);
        
        SymbolicFactorization();
        
        ReadNonzeroValues( A_val_, reg_ );

        ClearFactors();
        
        TOOLS_PTIC(tag + ": Initialize update buffers.");
        
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
        
    //        SN_updates = std::vector<Update_T> ( SN_count, nullptr );
        
        TOOLS_PTOC(tag + ": Initialize update buffers.");
        
        
        TOOLS_PTIC(tag + ": Initialize factorizers");
        
        std::vector<std::unique_ptr<Factorizer_MF_T>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_MF_T>(*this);
            },
            thread_count
        );
        
        Factorizer_MF_T worker(*this);
        
        TOOLS_PTOC(tag + ": Initialize factorizers");
        
        // Parallel traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_Postordered<Parallel>( SN_list );
        }
        else
        {
            aTree.template Traverse_Postordered<Sequential>( SN_list );
        }
        
        SN_factorized = true;
        
        TOOLS_PTIC(tag + ": Release update buffers.");
        
        SN_updates = std::vector<Update_T>();
        
        TOOLS_PTOC(tag + ": Release update buffers.");
        
        TOOLS_PTOC(tag);
    }


    template<typename ExtScal>
    void NumericFactorization_LeftLooking(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_LeftLooking<" + TypeName<ExtScal> + ">";
       
        TOOLS_PTIC(tag);
        
        SymbolicFactorization();

        ReadNonzeroValues( A_val_, reg_ );
        
        ClearFactors();
        
        TOOLS_PTIC(tag + ": Initialize factorizers");
        
        std::vector<std::unique_ptr<Factorizer_LL_T>> SN_list (thread_count);
        
        ParallelDo(
            [&SN_list,this]( const Int thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_LL_T>(*this);
            },
            thread_count
        );
        
        TOOLS_PTOC(tag + ": Initialize factorizers");
        
        // Parallel traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_Postordered<Parallel>( SN_list );
        }
        else
        {
            aTree.template Traverse_Postordered<Sequential>( SN_list );
        }
        
        SN_factorized = true;
        
        TOOLS_PTOC(tag);
        
    }



    template<typename ExtScal>
    void ReadNonzeroValues( cptr<ExtScal> A_val_, const ExtScal reg_ )
    {
        std::string tag = ClassName()+"::ReadNonzeroValues<" + TypeName<ExtScal> + ">";
       
        TOOLS_PTIC(tag);
        
        ParallelDo(
            [&]( const LInt i )
            {
                A_val[i] = static_cast<Scal>(A_val_[A_inner_perm[i]]);
            },
            A_inner_perm.Size(), static_cast<LInt>(thread_count)
        );
        
        reg = reg_;
        
        this->ClearCache();
        
        TOOLS_PTOC(tag);
    }

    void ClearFactors()
    {
        TOOLS_PTIC(ClassName()+"::ClearFactors");
        
        SN_tri_val.SetZero( thread_count );
        SN_rec_val.SetZero( thread_count );
        
        TOOLS_PTOC(ClassName()+"::ClearFactors");
    }



