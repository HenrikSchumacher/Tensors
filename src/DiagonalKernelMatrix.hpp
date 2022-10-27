#pragma once

#define CLASS DiagonalKernelMatrix

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
        
        CLASS()
        :   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        explicit CLASS(
            const Int n_,
            const Int thread_count
        )
        ,   n ( n_ )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   pattern ( other.pattern )
        ,   kernel { nullptr, 0, nullptr, 0, nullptr, Kernel_T::MAX_RHS_COUNT }
        {}

        ~CLASS() = default;
        
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
        
//        void FillLowerTriangleFromUpperTriangle( Scalar * restrict const values ) const
//        {
//            ptic(ClassName()+"::FillLowerTriangleFromUpperTriangle");
//
//            if( pattern.WellFormed() && (pattern.RowCount()>= pattern.ColCount()) )
//            {
//                const Int * restrict const diag   = pattern.Diag().data();
//                const Int * restrict const outer  = pattern.Outer().data();
//                const Int * restrict const inner  = pattern.Inner().data();
//
//                const auto & job_ptr = pattern.LowerTriangularJobPtr();
//
//                const Int thread_count = job_ptr.Size()-1;
//
//                #pragma omp parallel for num_threads( thread_count )
//                for( Int thread = 0; thread < thread_count; ++thread )
//                {
//                    Kernel_T ker ( values );
//
//                    const Int i_begin = job_ptr[thread  ];
//                    const Int i_end   = job_ptr[thread+1];
//
//                    for( Int i = i_begin; i < i_end; ++i )
//                    {
//                        const Int k_begin = outer[i];
//                        const Int k_end   =  diag[i];
//
//                        for( Int k = k_begin; k < k_end; ++k )
//                        {
//                            const Int j = inner[k];
//
//                            Int L =  diag[j];
//                            Int R = outer[j+1]-1;
//
//                            while( L < R )
//                            {
//                                const Int M = R - (R-L)/static_cast<Int>(2);
//                                const Int j = inner[M];
//
//                                if( j > i )
//                                {
//                                    R = M-1;
//                                }
//                                else
//                                {
//                                    L = M;
//                                }
//                            }
//
//                            ker.TransposeBlock(L,k);
//
//                        } // for( Int k = k_begin; k < k_end; ++k )
//
//                    } // for( Int i = i_begin; i < i_end; ++i )
//
//                } // #pragma omp parallel
//            }
//
//            ptoc(ClassName()+"::FillLowerTriangleFromUpperTriangle");
//        }
        
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
                if( thread_count > 1)
                {
                    // OpenMP has a considerable overhead at launching the threads...
                    #pragma omp parallel for simd num_threads( thread_count ) schedule( static )
                    for( Int i = 0; i < size; ++i )
                    {
                        Y[i] *= beta;
                    }
                }
                else
                {
                    for( Int i = 0; i < size; ++i )
                    {
                        Y[i] *= beta;
                    }
                }
            }
        }
        
        
        void force_inline Dot(
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
            
            const auto & job_ptr = JobPointers<Int>(n,n);
            
            const Int thread_count = job_ptr.Size()-1;

            if( thread_count > 1)
            {
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int i = k;
                        const Int j = k;
                        
                        ker.BeginRow(i);
                        ker.ApplyBlock(k,j);
                        ker.EndRow(i);
                    }
                }
            }
            else
            {
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    // Initialize local kernel and feed it all the information that is going to be constant along its life time.
                    Kernel_T ker ( A, alpha, X, beta, Y, rhs_count );
                    
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                    for( Int k = k_begin; k < k_end; ++k )
                    {
                        const Int i = k;
                        const Int j = k;
                        
                        ker.BeginRow(i);
                        ker.ApplyBlock(k,j);
                        ker.EndRow(i);
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
