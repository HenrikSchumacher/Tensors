#pragma once

#define CLASS SparseKernelMatrixCSR_gen

namespace Tensors
{
    template<typename Kernel_T>
    class CLASS
    {
    public:
        
        using Scalar     = typename Kernel_T::Scalar;
        using Int        = typename Kernel_T::Int;
        using Scalar_in  = typename Kernel_T::Scalar_in;
        using Scalar_out = typename Kernel_T::Scalar_out;
        
        using SparsityPattern_T = SparsityPatternCSR<Int>;
        
        CLASS()
        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        explicit CLASS(
            const SparsityPattern_T & pattern_
        )
        :   pattern ( pattern_ )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {
//            print("Creating class "+TO_STD_STRING(CLASS));
//            print(kernel.ClassName());
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   pattern ( other.pattern )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}

        ~CLASS() = default;
        
    protected:
        
        const SparsityPattern_T   & pattern;
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
        
        Int NonzeroCount() const
        {
            return pattern.NonzeroCount() * Kernel_T::NONZERO_COUNT;
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
                const Int * restrict const diag   = pattern.Diag().data();
                const Int * restrict const outer  = pattern.Outer().data();
                const Int * restrict const inner  = pattern.Inner().data();
                
                const auto & job_ptr = pattern.LowerTriangularJobPtr();
                
                const Int thread_count = job_ptr.Size()-1;
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    Kernel_T ker ( values );
                    
                    const Int i_begin = job_ptr[thread];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const Int k_begin = outer[i];
                        const Int k_end   =  diag[i];
                        
                        for( Int k = k_begin; k < k_end; ++k )
                        {
                            const Int j = inner[k];
                            
                            Int L =  diag[j];
                            Int R = outer[j+1]-1;
                            
                            while( L < R )
                            {
                                const Int M   = R - (R-L)/static_cast<Int>(2);
                                const Int col = inner[M];

                                if( col > i )
                                {
                                    R = M-1;
                                }
                                else
                                {
                                    L = M;
                                }
                            }
                            
                            ker.TransposeBlock(L, k);
                            
                        } // for( Int k = k_begin; k < k_end; ++k )

                    } // for( Int i = i_begin; i < i_end; ++i )
                    
                } // #pragma omp parallel
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
                zerofy_buffer(Y, size);
            }
            else
            {
                #pragma omp parallel for simd num_threads( pattern.ThreadCount() ) schedule( static )
                for( Int i = 0; i < size; ++i )
                {
                    Y[i] *= beta;
                }
            }
        }
        
        void Dot(
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

//            const Int rows_size = Kernel_T::RowCount() * rhs_count;
            const Int cols_size = Kernel_T::ColCount() * rhs_count;

            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );

                
                const Int * restrict const rp = pattern.Outer().data();
                const Int * restrict const ci = pattern.Inner().data();
                
                // Kernel is supposed the following rows of pattern:
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    // These are the corresponding nonzero blocks in i-th row.
                    const Int k_begin = rp[i  ];
                    const Int k_end   = rp[i+1];
                    
                    if( k_end > k_begin )
                    {
                        // Clear the local vector chunk of the kernel.
                        ker.CleanseVector();
                        
                        // Perform all but the last calculation in row with prefetch.
                        for( Int k = k_begin; k < k_end-1; ++k )
                        {
                            const Int j = ci[k];
                            
                            // X is accessed in an unpredictable way; let's help with a prefetch statement.
                            prefetch_range<0,0>( &X[cols_size * ci[k+1]], cols_size );
                            
                            // The buffer A is accessed in-order; thus we can rely on the CPU's prefetcher.
                            
                            // Let the kernel apply to the k-th block to the j-th chunk of the input.
                            // The result is stored in the kernel's local vector chunk X.
                            ker.ApplyBlock(k,j);
                        }
                        
                        // Perform last calculation in row without prefetch.
                        {
                            const Int k = k_end-1;
                            
                            const Int j = ci[k];
                            
//                            prefetch_range<1,0>( &Y[rows_size * i], rows_size );
                            
                            // Let the kernel apply to the k-th block to the j-th chunk of the input X.
                            // The result is stored in the kernel's local vector chunk.
                            ker.ApplyBlock(k,j);
                        }
                        
                        // Incorporate the kernel's local vector chunk into the i-th chunk if the output Y.
                        
                        ker.WriteVector(i);
                    }
                    else
                    {
                        ker.WriteZero(i);
                    }
                    
                }
            }
            ptoc(ClassName()+"::Dot" );
        }
        
    public:
        
        std::string ClassName() const
        {
            return TO_STD_STRING(CLASS)+"<"+kernel.ClassName()+">";
        }
        
    };
    
}// namespace Tensors

#undef CLASS
