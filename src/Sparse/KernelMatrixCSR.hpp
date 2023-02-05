#pragma once

namespace Tensors
{
    namespace Sparse
    {
        template<typename Kernel_T>
        class KernelMatrixCSR
        {
        public:
            
            using Scal     = typename Kernel_T::Scal;
            using Int      = typename Kernel_T::Int;
            using LInt     = typename Kernel_T::LInt;
            using Scal_in  = typename Kernel_T::Scal_in;
            using Scal_out = typename Kernel_T::Scal_out;
            
            using Pattern_T = Sparse::PatternCSR<Int,LInt>;
            
            KernelMatrixCSR() = delete;
            
            //        KernelMatrixCSR()
            //        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
            //        {}
            
            explicit KernelMatrixCSR( const Pattern_T & pattern_ )
            :   pattern ( pattern_ )
            ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
            {}
            
            // Copy constructor
            KernelMatrixCSR( const KernelMatrixCSR & other )
            :   pattern ( other.pattern )
            ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
            {}
            
            ~KernelMatrixCSR() = default;
            
        protected:
            
            const Pattern_T & pattern;
            Kernel_T kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT };
            
        public:
            
            Int RowCount() const
            {
                return pattern.RowCount() * Kernel_T::RowCount();
            }
            
            Int ColCount() const
            {
                return pattern.ColCount() * Kernel_T::ColCount();
            }
            
            LInt NonzeroCount() const
            {
                return pattern.NonzeroCount() * Kernel_T::BLOCK_NNZ;
            }
            
            
            //##############################################################################################
            //      Symmetrization
            //##############################################################################################
            
        public:
            
            void FillLowerTriangleFromUpperTriangle( mut<Scal> values ) const
            {
                ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
                
                if( pattern.WellFormed() && (pattern.RowCount()>= pattern.ColCount()) )
                {
                    ptr<LInt> diag   = pattern.Diag().data();
                    ptr<LInt> outer  = pattern.Outer().data();
                    ptr<Int>  inner  = pattern.Inner().data();
                    
                    const auto & job_ptr = pattern.LowerTriangularJobPtr();
                    
                    const Int thread_count = job_ptr.ThreadCount();
                    
                    // OpenMP has a considerable overhead at launching the threads...
                    if( thread_count > 1)
                    {
                        #pragma omp parallel for num_threads( thread_count )
                        for( Int thread = 0; thread < thread_count; ++thread )
                        {
                            Kernel_T ker ( values );
                            
                            const Int i_begin = job_ptr[thread  ];
                            const Int i_end   = job_ptr[thread+1];
                            
                            for( Int i = i_begin; i < i_end; ++i )
                            {
                                const LInt k_begin = outer[i];
                                const LInt k_end   =  diag[i];
                                
                                for( LInt k = k_begin; k < k_end; ++k )
                                {
                                    const Int j = inner[k];
                                    
                                    LInt L =  diag[j];
                                    LInt R = outer[j+1]-1;
                                    
                                    while( L < R )
                                    {
                                        const LInt M = R - (R-L)/static_cast<LInt>(2);
                                        const  Int col = inner[M];
                                        
                                        if( col > i )
                                        {
                                            R = M-1;
                                        }
                                        else
                                        {
                                            L = M;
                                        }
                                    }
                                    
                                    ker.TransposeBlock(L,k);
                                    
                                } // for( LInt k = k_begin; k < k_end; ++k )
                                
                            } // for( Int i = i_begin; i < i_end; ++i )
                            
                        } // #pragma omp parallel
                    }
                    else
                    {
                        Kernel_T ker ( values );
                        
                        const Int i_begin = job_ptr[0  ];
                        const Int i_end   = job_ptr[0+1];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            const LInt k_begin = outer[i];
                            const LInt k_end   =  diag[i];
                            
                            for( LInt k = k_begin; k < k_end; ++k )
                            {
                                const Int j = inner[k];
                                
                                LInt L =  diag[j];
                                LInt R = outer[j+1]-1;
                                
                                while( L < R )
                                {
                                    const LInt M = R - (R-L)/static_cast<LInt>(2);
                                    const  Int col = inner[M];
                                    
                                    if( col > i )
                                    {
                                        R = M-1;
                                    }
                                    else
                                    {
                                        L = M;
                                    }
                                }
                                
                                ker.TransposeBlock(L,k);
                                
                            } // for( Int k = k_begin; k < k_end; ++k )
                            
                        } // for( Int i = i_begin; i < i_end; ++i )
                    }
                }
                
                ptoc(ClassName()+"::FillLowerTriangleFromUpperTriangle");
            }
            
            
//##############################################################################################
//      Matrix multiplication
//##############################################################################################
            
            void Scale( mut<Scal_out> Y, const Scal_out beta, const Int rhs_count ) const
            {
                const Int size = RowCount() * rhs_count;
                
                if( beta == static_cast<Scal_out>(0) )
                {
                    zerofy_buffer( Y, size, pattern.ThreadCount() );
                }
                else
                {
                    scale_buffer(beta, Y, size, pattern.ThreadCount() );
                }
            }
            
            __attribute__((flatten)) void Dot(
                ptr<Scal> A,
                const Scal_out alpha, ptr<Scal_in>  X,
                const Scal_out beta,  mut<Scal_out> Y,
                const Int rhs_count
            ) const
            {
                ptic(ClassName()+"::Dot" );
                
                if( (alpha == static_cast<Scal_out>(0)) || (NonzeroCount() <= 0) )
                {
                    Scale( Y, beta, rhs_count );
                    
                    ptoc(ClassName()+"::Dot" );
                    
                    return;
                }
                
                const auto & job_ptr = pattern.JobPtr();
                
                const Int thread_count = job_ptr.ThreadCount();
                
                // OpenMP has a considerable overhead at launching the threads...
                if( thread_count > 1 )
                {
                    #pragma omp parallel for num_threads( thread_count )
                    for( Int thread = 0; thread < thread_count; ++thread )
                    {
                        // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                        Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                        
                        ptr<LInt> rp = pattern.Outer().data();
                        ptr<Int>  ci = pattern.Inner().data();
                        
                        // Kernel is supposed the following rows of pattern:
                        const Int i_begin = job_ptr[thread  ];
                        const Int i_end   = job_ptr[thread+1];
                        
                        for( Int i = i_begin; i < i_end; ++i )
                        {
                            // These are the corresponding nonzero blocks in i-th row.
                            const LInt k_begin = rp[i  ];
                            const LInt k_end   = rp[i+1];
                            
                            if( k_end > k_begin )
                            {
                                // Clear the local vector chunk of the kernel.
                                ker.CleanseY();
                                
                                // Perform all but the last calculation in row with prefetch.
                                for( LInt k = k_begin; k < k_end-1; ++k )
                                {
                                    const Int j = ci[k];
                                    
                                    ker.Prefetch(k,ci[k+1]);
                                    
                                    // Let the kernel apply to the k-th block to the j-th chunk of the input.
                                    // The result is stored in the kernel's local vector chunk X.
                                    ker.ApplyBlock(k,j);
                                }
                                
                                // Perform last calculation in row without prefetch.
                                {
                                    const LInt k = k_end-1;
                                    
                                    const Int j = ci[k];
                                    
                                    // Let the kernel apply to the k-th block to the j-th chunk of the input X.
                                    // The result is stored in the kernel's local vector chunk.
                                    ker.ApplyBlock(k,j);
                                }
                                
                                // Incorporate the kernel's local vector chunk into the i-th chunk if the output Y.
                                
                                ker.WriteY(i);
                            }
                            else
                            {
                                // Make sure that Y(i) is correctly overwritten in the case that there are not entries in the row.
                                ker.WriteYZero(i);
                            }
                            
                        }
                    }
                }
                else
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                    
                    ptr<LInt> rp = pattern.Outer().data();
                    ptr<Int> ci = pattern.Inner().data();
                    
                    // Kernel is supposed the following rows of pattern:
                    const Int i_begin = job_ptr[0  ];
                    const Int i_end   = job_ptr[0+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        // These are the corresponding nonzero blocks in i-th row.
                        const LInt k_begin = rp[i  ];
                        const LInt k_end   = rp[i+1];
                        
                        if( k_end > k_begin )
                        {
                            // Clear the local vector chunk of the kernel.
                            ker.CleanseY();
                            
                            // Perform all but the last calculation in row with prefetch.
                            for( LInt k = k_begin; k < k_end-1; ++k )
                            {
                                const Int j = ci[k];
                                
                                ker.Prefetch(k,ci[k+1]);
                                
                                // Let the kernel apply to the k-th block to the j-th chunk of the input.
                                // The result is stored in the kernel's local vector chunk X.
                                ker.ApplyBlock(k,j);
                            }
                            
                            // Perform last calculation in row without prefetch.
                            {
                                const LInt k = k_end-1;
                                
                                const Int j = ci[k];
                                
                                // Let the kernel apply to the k-th block to the j-th chunk of the input X.
                                // The result is stored in the kernel's local vector chunk.
                                ker.ApplyBlock(k,j);
                            }
                            
                            // Incorporate the kernel's local vector chunk into the i-th chunk if the output Y.
                            
                            ker.WriteY(i);
                        }
                        else
                        {
                            // Make sure that Y(i) is correctly overwritten in the case that there are not entries in the row.
                            ker.WriteYZero(i);
                        }
                        
                    }
                    
                }
                ptoc(ClassName()+"::Dot" );
            }
            
    //###########################################################################################
    //####          Permute
    //###########################################################################################
            
        public:
            
            Permutation<LInt> Permute(
                const Permutation<Int> & p,  // row    permutation
                const Permutation<Int> & q  // column permutation
            )
            {
                // Modifies inner, outer, and values accordingly; returns the permutation to be applied to the nonzero values.
                
                const LInt nnz = this->inner.Size();
                
                Permutation<LInt> perm;
                
                std::tie( this->outer, this->inner, perm ) = SparseMatrixPermutation(
                    this->outer.data(), this->inner.data(), p, q, nnz, true
                );

                if( !p.IsTrivial() || !q.IsTrivial() )
                {
                    ptr<Scal> u = this->new_values.data();
                    mut<Scal> v = this->values.data();
                    Tensor1<Scal,LInt> new_values ( nnz );
                    
                    #pragma omp parallel for num_threads( this->ThreadCount() ) schedule(static)
                    for( Int i = 0; i < nnz; ++i )
                    {
                        v[i] = u[perm[i]];
                    }
                    
                    swap( this->values, new_values);
                }
                
                this->inner_sorted = true;
                
                this->diag_ptr_initialized = false;
                this->job_ptr_initialized  = false;
                this->upper_triangular_job_ptr_initialized = false;
                this->lower_triangular_job_ptr_initialized = false;
                
                return perm;
            }
            
        public:
            
            std::string ClassName() const
            {
                return "Sparse::KernelMatrixCSR<"+kernel.ClassName()+">";
            }
            
        }; // class KernelMatrixCSR
        
    }; // namespace Sparse
        
} // namespace Tensors
