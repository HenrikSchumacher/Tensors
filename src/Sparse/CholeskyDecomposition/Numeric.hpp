public:

    template<typename ExtScal>
    int NumericFactorization(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        return NumericFactorization_Multifrontal( A_val_, reg_ );
    }

    template<typename ExtScal>
    int NumericFactorization_Multifrontal(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_Multifrontal<" + TypeName<ExtScal> + ">";
    
        TOOLS_PTIMER(timer,tag);
        
        SymbolicFactorization();
        ReadNonzeroValues( A_val_, reg_ );
        ClearFactors();
        
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
        
        SN_updates = std::vector<Update_T> ( ToSize_T(SN_count) );

        Size_T use_threads = ToSize_T(thread_count);
        
        std::vector<std::unique_ptr<Factorizer_MF_T>> SN_list (use_threads);
        
        ParallelDo(
            [&SN_list,this]( const Size_T thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_MF_T>(*this);
            },
            use_threads
        );
        
        Factorizer_MF_T worker(*this);
        
        // Parallel traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_PostOrdered<Parallel>( SN_list );
        }
        else
        {
            aTree.template Traverse_PostOrdered<Sequential>( SN_list );
        }
        
        SN_numerically_goodQ = true;
        
        Do(
            [&SN_list,this]( const Size_T thread )
            {
                SN_numerically_goodQ
                = SN_numerically_goodQ && SN_list[thread]->GoodQ();
            },
            use_threads
        );
        
        if( !SN_numerically_goodQ )
        {
            eprint(ClassName()+"::NumericFactorization_Multifrontal: Could not complete numeric factorization. Matrix is not (sufficiently) positive-definite.");
        }
        
        // We mark this as factorized in any case to not attempt the factorization again.
        SN_factorized = true;
        
        SN_updates = std::vector<Update_T>();
        
        return (SN_numerically_goodQ ? 0 : 1);
    }


    template<typename ExtScal>
    int NumericFactorization_LeftLooking(
        cptr<ExtScal> A_val_,
        const ExtScal reg_  = 0 // Regularization parameter for the diagonal.
    )
    {
        std::string tag = ClassName()+"::NumericFactorization_LeftLooking<" + TypeName<ExtScal> + ">";
       
        TOOLS_PTIMER(timer,tag);
        
        SymbolicFactorization();

        ReadNonzeroValues( A_val_, reg_ );
        
        ClearFactors();
        
        Size_T use_threads = ToSize_T(thread_count);
        
        std::vector<std::unique_ptr<Factorizer_LL_T>> SN_list (use_threads);
        
        ParallelDo(
            [&SN_list,this]( const Size_T thread )
            {
                SN_list[thread] = std::make_unique<Factorizer_LL_T>(*this);
            },
            use_threads
        );
        
        // Parallel traversal in postorder
        if( thread_count > Int(1) )
        {
            aTree.template Traverse_PostOrdered<Parallel>( SN_list );
        }
        else
        {
            aTree.template Traverse_PostOrdered<Sequential>( SN_list );
        }
        
        SN_numerically_goodQ = true;
        
        Do(
            [&SN_list,this]( const Size_T thread )
            {
                SN_numerically_goodQ
                = SN_numerically_goodQ && SN_list[thread]->GoodQ();
            },
            use_threads
        );
        
        if( !SN_numerically_goodQ )
        {
            eprint(ClassName()+"::NumericFactorization_LeftLooking: Could not complete numeric factorization. Matrix is not (sufficiently) positive-definite.");
        }
        
        // We mark this as factorized in any case to not attempt the factorization again.
        SN_factorized = true;
        
        return (SN_numerically_goodQ ? 0 : 1);
    }



    template<typename ExtScal>
    void ReadNonzeroValues( cptr<ExtScal> A_val_, const ExtScal reg_ )
    {
        TOOLS_PTIMER(timer,ClassName()+"::ReadNonzeroValues<" + TypeName<ExtScal> + ">");
        
        ParallelDo(
            [&]( const LInt i )
            {
                A_val[i] = static_cast<Scal>(A_val_[A_inner_perm[i]]);
            },
            A_inner_perm.Size(), static_cast<LInt>(thread_count)
        );
        
        reg = reg_;
        
        this->ClearCache();
    }

    void ClearFactors()
    {
        TOOLS_PTIMER(timer,ClassName()+"::ClearFactors");
        
        SN_tri_val.SetZero( thread_count );
        SN_rec_val.SetZero( thread_count );
    }



