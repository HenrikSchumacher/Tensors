#pragma once

namespace Tensors
{

#define CLASS SparseBinaryMatrixCSR
#define BASE  SparsityPatternCSR<Int,LInt>
    
    template<typename Int, typename LInt>
    class CLASS : public BASE
    {
        ASSERT_INT  (Int);
        ASSERT_INT  (LInt);
        
    protected:
        
        using BASE::m;
        using BASE::n;
        using BASE::outer;
        using BASE::inner;
        using BASE::thread_count;
        using BASE::job_ptr;
        using BASE::upper_triangular_job_ptr;
        using BASE::lower_triangular_job_ptr;
        using BASE::Dot_;
        
    public:
        
        using BASE::RowCount;
        using BASE::ColCount;
        using BASE::NonzeroCount;
        using BASE::ThreadCount;
        using BASE::SetThreadCount;
        using BASE::Outer;
        using BASE::Inner;
        using BASE::JobPtr;
        using BASE::Diag;
        using BASE::RequireJobPtr;
        using BASE::RequireUpperTriangularJobPtr;
        using BASE::RequireLowerTriangularJobPtr;
        using BASE::UpperTriangularJobPtr;
        using BASE::LowerTriangularJobPtr;
        using BASE::CreateTransposeCounters;
        using BASE::WellFormed;
        
        CLASS() : BASE() {};

        CLASS(
            const long long m_,
            const long long n_,
            const long long thread_count_
        ) : BASE( m_, n_, thread_count_ ) {}
        
        template<typename I_0, typename I_1, typename I_2, typename I_3>
        CLASS(
            const I_0 m_,
            const I_1 n_,
            const I_2 nnz_,
            const I_3 thread_count_
        ) : BASE( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<LInt>(nnz_), static_cast<Int>(thread_count_) ) {}
        
        
        template<typename J0, typename J1, typename I_0, typename I_1, typename I_3>
        CLASS(
            const J0 * const outer_,
            const J1 * const inner_,
            const I_0 m_,
            const I_1 n_,
            const I_3 thread_count_
        ) : BASE(outer_, inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) ) {};
        
        // Copy constructor
        CLASS( const CLASS & other ) : BASE( other )
//        :   outer       ( other.outer         )
//        ,   inner       ( other.inner         )
//        ,   m           ( other.m             )
//        ,   n           ( other.n             )
//        ,   thread_count( other.thread_count  )
        {}
        
        friend void swap (CLASS &A, CLASS &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.outer,        B.outer        );
            swap( A.inner,        B.inner        );
            swap( A.m,            B.m            );
            swap( A.n,            B.n            );
            swap( A.thread_count, B.thread_count );
        }
        
        friend void swap (CLASS &A, BASE &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.outer,        B.outer        );
            swap( A.inner,        B.inner        );
            swap( A.m,            B.m            );
            swap( A.n,            B.n            );
            swap( A.thread_count, B.thread_count );
        }
        
        // Copy assignment operator
        CLASS & operator=(CLASS other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        CLASS( CLASS && other ) noexcept : CLASS()
        {
            swap(*this, other);
        }
        
        
        CLASS(
            std::vector<Int> & idx,
            std::vector<Int> & jdx,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE( idx, jdx, m_, n_, final_thread_count, compress, symmetrize ) {};
        
        CLASS(
            const std::vector<std::vector<Int>> & idx,
            const std::vector<std::vector<Int>> & jdx,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE( idx, jdx, m_, n_, final_thread_count, compress, symmetrize ) {};
        
        CLASS(
            const std::vector<PairAggregator<Int, Int, LInt>> & idx,
            const Int m_,
            const Int n_,
            const Int final_thread_count,
            const bool compress   = true,
            const int  symmetrize = 0
        )
        :   BASE( idx, m_, n_, final_thread_count, compress, symmetrize ) {};
    
        virtual ~CLASS() = default;
        
        
        CLASS Transpose() const
        {
            ptic(ClassName()+"::Transpose");
            
            RequireJobPtr();
            
            Tensor2<LInt,Int> counters = CreateTransposeCounters();
            
            CLASS<Int,LInt> B ( n, m, outer[m], thread_count );

            copy_buffer( counters.data(thread_count-1), &B.Outer().data()[1], n );
            
            if( WellFormed() )
            {
                #pragma omp parallel for num_threads( thread_count )
                for( Int thread = 0; thread < thread_count; ++thread )
                {
                    const Int i_begin = job_ptr[thread  ];
                    const Int i_end   = job_ptr[thread+1];
                    
                          LInt * restrict const c = counters.data(thread);

                           Int * restrict const B_inner  = B.Inner().data();

                    const LInt * restrict const A_outer  = Outer().data();
                    const  Int * restrict const A_inner  = Inner().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt k_begin = A_outer[i  ];
                        const LInt k_end   = A_outer[i+1];
                        
                        for( LInt k = k_end-1; k > k_begin-1; --k )
                        {
                            const Int j = A_inner[k];
                            const LInt pos = --c[ j ];
                            B_inner [pos] = i;
                        }
                    }
                }
            }
            
            // Finished counting sort.

            // We only have to care about the correct ordering of inner indices and values.
            B.SortInner();

            ptoc(ClassName()+"::Transpose");
            
            return B;
        }
        
        
//##############################################################################################
//####          Matrix Multiplication
//##############################################################################################
    
    public:
        
        CLASS Dot( const CLASS & B ) const
        {
            CLASS<Int,LInt> result;
            
            BASE C = this->DotBinary_(B);
            
            swap(result,C);
            
            return result;
        }
        
        CLASS DotBinary( const CLASS & B ) const
        {
            CLASS result;
            
            BASE C = this->DotBinary_(B);
            
            swap(result,C);
            
            return result;
        }
        
        
//##############################################################################################
//####          Matrix Multiplication
//##############################################################################################
        
        
        // Assume all nonzeros are equal to 1.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const T_in  * X,
            const T_out beta,
                  T_out * Y,
            const Int cols = static_cast<Int>(1)
        ) const
        {
            Dot_( alpha, X, beta, Y, cols );
        }

        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor1<T_in, Int> & X,
            const T_out beta,
                  Tensor1<T_out,Int> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( alpha, X.data(), beta, Y.data(), static_cast<Int>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha,
            const Tensor2<T_in, Int> & X,
            const T_out beta,
                  Tensor2<T_out,Int> & Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
            {
                Dot_( alpha, X.data(), beta, Y.data(), X.Dimension(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        
        
        // Supply an external list of values.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext * ext_values,
            const T_ext   alpha,
            const T_in  * X,
            const T_out   beta,
                  T_out * Y,
            const Int     cols = static_cast<Int>(1)
        ) const
        {
            Dot_( ext_values, alpha, X, beta, Y, cols );
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const Tensor1<T_ext, LInt> & ext_values,
            const T_ext                  alpha,
            const Tensor1<T_in, Int> &   X,
            const T_out                  beta,
                  Tensor1<T_out,Int> &   Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m )
            {
                Dot_( ext_values.data(), alpha, X.data(), beta, Y.data(), static_cast<Int>(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const Tensor1<T_ext, LInt> & ext_values,
            const                        T_ext alpha,
            const Tensor2<T_in, Int> &   X,
            const T_out                  beta,
                  Tensor2<T_out,Int> &   Y
        ) const
        {
            if( X.Dimension(0) == n && Y.Dimension(0) == m && (X.Dimension(1) == Y.Dimension(1)) )
            {
                Dot_( ext_values.data(), alpha, X.data(), beta, Y.data(), X.Dimension(1) );
            }
            else
            {
                eprint(ClassName()+"::Dot: shapes of matrix, input, and output do not match.");
            }
        }
        
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
        }
        
    }; // CLASS

    
} // namespace Tensors


#undef CLASS
#undef BASE
