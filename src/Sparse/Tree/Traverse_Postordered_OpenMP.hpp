public:

    template<class Worker_T>
    void Traverse_Postordered_OpenMP( mref<std::vector<std::unique_ptr<Worker_T>>> workers )
    {
        ptic(ClassName() + "::Traverse_Postordered_OpenMP");
        
        Int root = Root();
        
        #pragma omp parallel num_threads( workers.size() )
        {
            #pragma omp single
            {
                for( Int child = ChildPointer(root); child < ChildPointer(root+1); ++child )
                {
                    #pragma omp task private(n) untied
                    {
                        Traverse_Children_Postordered_OpenMP( &workers[0], ChildIndex(child) );
                    }
                }
            }
        }
        
        #pragma omp taskwait

        ptoc(ClassName() + "::Traverse_Postordered_OpenMP");
    }

    template<class Worker_T>
    void Traverse_Children_Postordered_OpenMP( std::unique_ptr<Worker_T> * workers, const Int node )
    {
        for( Int child = ChildPointer(node); child < ChildPointer(node+1); ++child )
        {
            #pragma omp task private(n) untied
            {
                Traverse_Children_Postordered_OpenMP( &workers[0], ChildIndex(child) );
            }
        }
        
        #pragma omp taskwait

        const Int thread = static_cast<Int>(omp_get_thread_num());

        mref<Worker_T> worker = *workers[thread];
        
        worker( node );
    }
