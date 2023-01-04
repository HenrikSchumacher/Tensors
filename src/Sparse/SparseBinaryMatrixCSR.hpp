#pragma once

namespace Tensors
{
        
#define CLASS SparseBinaryMatrixCSR
    template<typename Int_, typename LInt_>
    class CLASS : public SparsePatternCSR<Int_,LInt_>
    {        
    private:
        
        using Base_T = SparsePatternCSR<Int_,LInt_>;
        
    public:
        
        using Int    = Int_;
        using LInt   = LInt_;
        
    protected:
        
        using Base_T::m;
        using Base_T::n;
        using Base_T::outer;
        using Base_T::inner;
        using Base_T::thread_count;
        using Base_T::job_ptr;
        using Base_T::upper_triangular_job_ptr;
        using Base_T::lower_triangular_job_ptr;
        using Base_T::Dot_;
        
    public:
        
        using Base_T::RowCount;
        using Base_T::ColCount;
        using Base_T::NonzeroCount;
        using Base_T::ThreadCount;
        using Base_T::SetThreadCount;
        using Base_T::Outer;
        using Base_T::Inner;
        using Base_T::JobPtr;
        using Base_T::Diag;
        using Base_T::SortInner;
        using Base_T::RequireDiag;
        using Base_T::RequireJobPtr;
        using Base_T::RequireUpperTriangularJobPtr;
        using Base_T::RequireLowerTriangularJobPtr;
        using Base_T::UpperTriangularJobPtr;
        using Base_T::LowerTriangularJobPtr;
        using Base_T::CreateTransposeCounters;
        using Base_T::WellFormed;
        
        CLASS()
        :   Base_T()
        {}
        
        template<typename I_0, typename I_1, typename I_3>
        CLASS(
              const I_0 m_,
              const I_1 n_,
              const I_3 thread_count_
              )
        :   Base_T( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        template<typename I_0, typename I_1, typename I_2, typename I_3>
        CLASS(
              const I_0 m_,
              const I_1 n_,
              const I_2 nnz_,
              const I_3 thread_count_
              )
        :   Base_T( static_cast<Int>(m_), static_cast<Int>(n_), static_cast<LInt>(nnz_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_2);
            ASSERT_INT(I_3);
        }
        
        
        template<typename J_0, typename J_1, typename I_0, typename I_1, typename I_3>
        CLASS(
              const J_0 * const outer_,
              const J_1 * const inner_,
              const I_0 m_,
              const I_1 n_,
              const I_3 thread_count_
              )
        :   Base_T( outer_, inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(J_0);
            ASSERT_INT(J_1);
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        template<typename I_0, typename I_1, typename I_3>
        CLASS(
              const Tensor1<LInt, Int> & outer_,
              const Tensor1< Int,LInt> & inner_,
              const I_0 m_,
              const I_1 n_,
              const I_3 thread_count_
              )
        :   Base_T( outer_, inner_, static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        template<typename I_0, typename I_1, typename I_3>
        CLASS(
              Tensor1<LInt, Int> && outer_,
              Tensor1< Int,LInt> && inner_,
              const I_0 m_,
              const I_1 n_,
              const I_3 thread_count_
              )
        :   Base_T( std::move(outer_), std::move(inner_), static_cast<Int>(m_), static_cast<Int>(n_), static_cast<Int>(thread_count_) )
        {
            ASSERT_INT(I_0);
            ASSERT_INT(I_1);
            ASSERT_INT(I_3);
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        :   Base_T( other )
        {
            logprint("Copy of "+ClassName()+" of size {"+ToString(m)+", "+ToString(n)+"}, nn z = "+ToString(NonzeroCount()));
        }
        
        friend void swap( CLASS &A, CLASS & B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;
            
            swap( A.outer,        B.outer        );
            swap( A.inner,        B.inner        );
            swap( A.m,            B.m            );
            swap( A.n,            B.n            );
            swap( A.thread_count, B.thread_count );
        }
        
        // (Copy-)assignment operator
        CLASS & operator=( CLASS other ) // Pass by value is okay, because we use copy-swap idiom and copy elision.
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
        
        // We do not need a move-assignment operator, because we use the copy-swap idiom!
        
        
        CLASS(
              const Int    * const * const idx,
              const Int    * const * const jdx,
              const LInt   *         const entry_counts,
              const Int list_count,
              const Int m_,
              const Int n_,
              const Int final_thread_count,
              const bool compress   = true,
              const int  symmetrize = 0
              )
        :   Base_T ( idx, jdx, entry_counts, list_count, m_, n_, final_thread_count, compress, symmetrize )
        {}
        
        CLASS(
              std::vector<Int> & idx,
              std::vector<Int> & jdx,
              const Int m_,
              const Int n_,
              const Int final_thread_count,
              const bool compress   = true,
              const int  symmetrize = 0
              )
        :   Base_T( idx, jdx, m_, n_, final_thread_count, compress, symmetrize )
        {}
        
        CLASS(
              const std::vector<std::vector<Int>> & idx,
              const std::vector<std::vector<Int>> & jdx,
              const Int m_,
              const Int n_,
              const Int final_thread_count,
              const bool compress   = true,
              const int  symmetrize = 0
              )
        :   Base_T( idx, jdx, m_, n_, final_thread_count, compress, symmetrize )
        {}
        
        CLASS(
              const std::vector<PairAggregator<Int, Int, LInt>> & idx,
              const Int m_,
              const Int n_,
              const Int final_thread_count,
              const bool compress   = true,
              const int  symmetrize = 0
              )
        :   Base_T( idx, m_, n_, final_thread_count, compress, symmetrize )
        {}
        
        virtual ~CLASS() = default;
        
        
        CLASS Transpose() const
        {
            ptic(ClassName()+"::Transpose");
            
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
                    
                    mut<LInt> c        = counters.data(thread);
                    mut<Int>  B_inner  = B.Inner().data();
                    ptr<LInt> A_outer  = Outer().data();
                    ptr< Int> A_inner  = Inner().data();
                    
                    for( Int i = i_begin; i < i_end; ++i )
                    {
                        const LInt k_begin = A_outer[i  ];
                        const LInt k_end   = A_outer[i+1];
                        
                        for( LInt k = k_end; k --> k_begin; )
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
        
        
//###########################################################################################
//####          Permute
//###########################################################################################
        
    public:
        
        CLASS Permute(
            const Permutation<Int> & p,  // row    permutation
            const Permutation<Int> & q,  // column permutation
        bool sort = true        // Whether to restore row-wise ordering (as in demanded by CSR).
        )
        {
            if( (!p.IsTrivial()) && (p.Size() != m) )
            {
                eprint(ClassName()+"::Permute: Length of first permutation does not coincide with RowCount().");
                return CLASS();
            }
            
            if( (!q.IsTrivial()) && (q.Size() != n) )
            {
                eprint(ClassName()+"::Permute: Length of second permutation does not coincide with ColCount().");
                return CLASS();
            }
            
            CLASS B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
            
            if( !p.IsTrivial() )
            {
                if( !q.IsTrivial() )
                {
                    return permute<true,true>(p,q);
                }
                else
                {
                    return permute<true,false>(p,q);
                }
            }
            else // if( p.IsTrivial() )
            {
                if( !q.IsTrivial() )
                {
                    return permute<false,true>(p,q);
                }
                else
                {
                    return B.inner = inner;
                }
            }
        }
        
    protected:
        
        template<bool PermuteRows, bool PermuteCols>
        CLASS permute( Permutation<Int> & P, Permutation<Int> & Q, bool sort = true )
        {
            CLASS B( RowCount(), ColCount(), NonzeroCount(), ThreadCount() );
            
            ptr<Int> p     =  COND( PermuteRows, Q.GetPermutation().data(),        nullptr );
            ptr<Int> q_inv =  COND( PermuteCols, Q.GetInversePermutation().data(), nullptr );
            
            if constexpr ( PermuteRows )
            {
                B.outer[0] = 0;
                
                #pragma omp parallel for num_threads( ThreadCount() ) schedule( static )
                for( Int i = 0; i < m; ++i )
                {
                    const Int p_i = p[i];
                    
                    B.outer[i+1] = outer[p_i+1] - outer[p_i];
                }
                
                B.outer.Accumulate();
            }
            else
            {
                B.outer = outer;
            }
            
            auto & B_job_ptr = B.JobPtr();
            
            const Int thread_count = B_job_ptr.ThreadCount();
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                const Int i_begin = B_job_ptr[thread  ];
                const Int i_end   = B_job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt A_begin = outer[COND( PermuteRows, p[i], i)];
                    
                    const LInt B_begin = B.outer[i  ];
                    const LInt B_end   = B.outer[i+1];
                    
                    const LInt k_max = B_end - B_begin;
                    
                    for( LInt k = 0; k < k_max; ++k )
                    {
                        if constexpr ( PermuteCols )
                        {
                            B.inner[B_begin+k] = q_inv[inner[A_begin+k]];
                        }
                        else
                        {
                            B.inner[B_begin+k] = inner[A_begin+k];
                        }
                    }
                    
                    if( sort )
                    {
                        std::sort( &B.inner[B_begin], &B.inner[B_end] );
                    }
                }
            }
            
            return B;
        }
        
//#########################################################################################
//####          Matrix Multiplication
//#########################################################################################
        
    public:
        
        CLASS Dot( const CLASS & B ) const
        {
            CLASS<Int,LInt> result;
            
            Base_T C = this->DotBinary_(B);
            
            swap(result,C);
            
            return result;
        }
        
        CLASS DotBinary( const CLASS & B ) const
        {
            CLASS result;
            
            Base_T C = this->DotBinary_(B);
            
            swap(result,C);
            
            return result;
        }
        
//###########################################################################################
//####          Matrix Multiplication
//###########################################################################################
        
        
        // Assume all nonzeros are equal to 1.
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha, ptr<T_in>  X,
            const T_out beta,  mut<T_out> Y,
            const Int cols = 1
        ) const
        {
            Dot_( alpha, X, beta, Y, cols );
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
            const T_ext alpha, const Tensor1<T_in, Int> & X,
            const T_out beta,        Tensor1<T_out,Int> & Y
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
             const T_ext alpha, const Tensor2<T_in, Int> & X,
             const T_out beta,        Tensor2<T_out,Int> & Y
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
            ptr<T_ext>  ext_values,
            const T_ext alpha, ptr<T_in>  X,
            const T_out beta,  mut<T_out> Y,
            const Int cols = 1
        ) const
        {
            Dot_( ext_values, alpha, X, beta, Y, cols );
        }
        
        template<typename T_ext, typename T_in, typename T_out>
        void Dot(
             const Tensor1<T_ext,LInt> & ext_values,
             const T_ext alpha, const Tensor1<T_in, Int> & X,
             const T_out beta,        Tensor1<T_out,Int> & Y
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
            const Tensor1<T_ext,LInt> & ext_values,
            const T_ext alpha, const Tensor2< T_in,Int> & X,
            const T_out beta,        Tensor2<T_out,Int> & Y
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
        
//###########################################################################################
//####          Triangular solve
//###########################################################################################
        
    public:
        
        template< Int RHS_COUNT, typename Scalar, bool unitDiag = false>
        void SolveUpperTriangular_Sequential_0( ptr<Scalar> values, ptr<Scalar> b, mut<Scalar> x )
        {
            this->template SolveUpperTriangular_Sequential_0_<RHS_COUNT,unitDiag>(values,b,x);
        }
        
    public:
        
        static std::string ClassName()
        {
            return TO_STD_STRING(CLASS)+"<"+TypeName<Int>::Get()+","+TypeName<LInt>::Get()+">";
        }
        
    }; // CLASS
    
} // namespace Tensors

#undef CLASS
