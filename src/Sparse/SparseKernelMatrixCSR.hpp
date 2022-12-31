#pragma once

namespace Tensors
{
    template<typename Kernel_T>
    class SparseKernelMatrixCSR
    {
    public:
        
        using Scalar     = typename Kernel_T::Scalar;
        using Int        = typename Kernel_T::Int;
        using LInt       = typename Kernel_T::LInt;
        using Scalar_in  = typename Kernel_T::Scalar_in;
        using Scalar_out = typename Kernel_T::Scalar_out;
        
        using SparsityPattern_T = SparsePatternCSR<Int,LInt>;
        
        SparseKernelMatrixCSR() = delete;
        
        //        KernelMatrixCSR()
        //        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        //        {}
        
        explicit SparseKernelMatrixCSR(
                                 const SparsityPattern_T & pattern_
                                 )
        :   pattern ( pattern_ )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        // Copy constructor
        SparseKernelMatrixCSR( const SparseKernelMatrixCSR & other )
        :   pattern ( other.pattern )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        ~SparseKernelMatrixCSR() = default;
        
    protected:
        
        const SparsityPattern_T & pattern;
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
        
        void FillLowerTriangleFromUpperTriangle( Scalar * restrict const values ) const
        {
            ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
            
            if( pattern.WellFormed() && (pattern.RowCount()>= pattern.ColCount()) )
            {
                const LInt * restrict const diag   = pattern.Diag().data();
                const LInt * restrict const outer  = pattern.Outer().data();
                const  Int * restrict const inner  = pattern.Inner().data();
                
                const auto & job_ptr = pattern.LowerTriangularJobPtr();
                
                const Int thread_count = job_ptr.Size()-1;
                
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
        
        void Scale( Scalar_out * restrict const Y, const Scalar_out beta, const Int rhs_count ) const
        {
            const Int size = RowCount() * rhs_count;
            
            if( beta == static_cast<Scalar_out>(0) )
            {
                zerofy_buffer( Y, size, pattern.ThreadCount() );
            }
            else
            {
                scale_buffer(beta, Y, size, pattern.ThreadCount() );
            }
        }
        
        __attribute__((flatten)) void Dot(
            const Scalar     * restrict const A,
            const Scalar_out                  alpha,
            const Scalar_in  * restrict const X,
            const Scalar_out                  beta,
                  Scalar_out * restrict const Y,
            const Int                         rhs_count
        ) const
        {
            ptic(ClassName()+"::Dot" );
            
            if( (alpha == static_cast<Scalar_out>(0)) || (NonzeroCount() <= 0) )
            {
                Scale( Y, beta, rhs_count );
                
                ptoc(ClassName()+"::Dot" );
                
                return;
            }
            
            const auto & job_ptr = pattern.JobPtr();
            
            const Int thread_count = job_ptr.Size()-1;
            
            // OpenMP has a considerable overhead at launching the threads...
            if( thread_count > 1 )
            {
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                    
                    const LInt * restrict const rp = pattern.Outer().data();
                    const  Int * restrict const ci = pattern.Inner().data();
                    
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
                
                const LInt * restrict const rp = pattern.Outer().data();
                const  Int * restrict const ci = pattern.Inner().data();
                
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
        
    public:
        
        std::string ClassName() const
        {
            return "SparseKernelMatrixCSR<"+kernel.ClassName()+">";
        }
        
    }; // class KernelMatrixCSR
        
} // namespace Tensors
