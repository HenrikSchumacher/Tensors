#pragma once

namespace Tensors
{
    template<typename Kernel_T>
    class DiagonalKernelMatrix
    {
    public:
        
        using Scalar     = typename Kernel_T::Scalar;
        using Int        = typename Kernel_T::Int;
        using LInt       = typename Kernel_T::LInt;
        using Scalar_in  = typename Kernel_T::Scalar_in;
        using Scalar_out = typename Kernel_T::Scalar_out;
        
        DiagonalKernelMatrix()
        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        DiagonalKernelMatrix(
              const Int n_,
              const Int thread_count_
              )
        :   n ( n_ )
        ,   thread_count(thread_count_)
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        // Copy constructor
        DiagonalKernelMatrix( const DiagonalKernelMatrix & other )
        :   n ( other.n )
        ,   thread_count( other.thread_count )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        ~DiagonalKernelMatrix() = default;
        
    protected:
        
        const Int n            = 0;
        const Int thread_count = 1;
        
        Kernel_T kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT };
        
    public:
        
        Int RowCount() const
        {
            return n * Kernel_T::RowCount();
        }
        
        Int ColCount() const
        {
            return n * Kernel_T::ColCount();
        }
        
        Int NonzeroCount() const
        {
            return n * Kernel_T::BLOCK_NNZ;
        }
        
        
        //##############################################################################################
        //      Symmetrization
        //##############################################################################################
        
    public:
        
        
//##############################################################################################
//      Matrix multiplication
//##############################################################################################
        
        void Scale( mut<Scalar_out> Y, const Scalar_out beta, const Int rhs_count ) const
        {
            const Int size = RowCount() * rhs_count;
            
            if( beta == static_cast<Scalar_out>(0) )
            {
                zerofy_buffer( Y, size, thread_count );
            }
            else
            {
                scale_buffer( beta, Y, size, thread_count );
            }
        }
        
        
        __attribute__((flatten)) void Dot(
            ptr<Scalar> A,
            const Scalar_out alpha, ptr<Scalar_in>  X,
            const Scalar_out beta,  mut<Scalar_out> Y,
            const Int rhs_count
        ) const
        {
            ptic(ClassName()+"::Dot" );
            
            if( (alpha == static_cast<Scalar_out>(0)) || (NonzeroCount() <= 0) )
            {
                Scale( Y, beta, rhs_count );
                
                ptoc(ClassName()+"::Dot" );
                
                return;
            }
            
            const auto & job_ptr = JobPointers<Int>(n,thread_count);
            
            // OpenMP has a considerable overhead at launching the threads...
            if( thread_count > 1)
            {
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        ker.CleanseY();
                        ker.ApplyBlock(i,i);
                        ker.WriteY(i);
                    }
                }
            }
            else
            {
                // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                
                const Int i_begin = job_ptr[0  ];
                const Int i_end   = job_ptr[0+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    ker.CleanseY();
                    ker.ApplyBlock(i,i);
                    ker.WriteY(i);
                }
            }
            
            ptoc(ClassName()+"::Dot" );
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Sparse::DiagonalKernelMatrix<"+kernel.ClassName()+">";
        }
        
    };
    
} // namespace Tensors
