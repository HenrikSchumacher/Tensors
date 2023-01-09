#pragma once

namespace Tensors
{
    
    enum class Compose : bool
    {
        Post = true,
        Pre  = false
    };
    
#ifdef True
    #undef True
#endif
    
#ifdef False
    #undef False
#endif
    
    enum class Inverse : bool
    {
        True  = true,
        False = false
    };
    
    template<typename Int>
    class Permutation
    {
        mutable Tensor1<Int,Int> p;
        mutable Tensor1<Int,Int> p_inv;
        
        mutable Tensor1<Int,Int> scratch;
        
        Int thread_count = 1;
        
        mutable bool is_trivial     = true;
        mutable bool p_computed     = true;
        mutable bool p_inv_computed = true;
        
    public:
        
        Permutation()
        :   p              ( 0     )
        ,   p_inv          ( 0     )
        ,   scratch        ( 0     )
        ,   thread_count   ( 1     )
        ,   is_trivial     ( true  )
        ,   p_computed     ( true  )
        ,   p_inv_computed ( true  )
        {}
        
        explicit Permutation( const Int n, const Int thread_count_ = 1 )
        :   p              ( iota<Int,Int>(n) )
        ,   p_inv          ( iota<Int,Int>(n) )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        ,   is_trivial     ( true             )
        ,   p_computed     ( true             )
        ,   p_inv_computed ( true             )
        {}
        
        template<typename J, IS_INT(J)>
        Permutation( ptr<J> p_, const Int n, const Inverse inverse, const Int thread_count_ = 1 )
        :   p              ( n                )
        ,   p_inv          ( n                )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        {
            if( inverse == Inverse::True )
            {
                SetInversePermutation(p_);
            }
            else
            {
                SetPermutation(p_);
            }
        }
        
        ~Permutation() = default;
        
        
        // Copy constructor
        Permutation( const Permutation & other )
        :   p                 ( other.p                 )
        ,   p_inv             ( other.p_inv             )
        ,   scratch           ( other.scratch           )
        ,   thread_count      ( other.thread_count      )
        ,   is_trivial        ( other.is_trivial        )
        ,   p_computed        ( other.p_computed        )
        ,   p_inv_computed    ( other.p_inv_computed    )
        {}
        
        // We could also simply use the implicitly created copy constructor.
        
        friend void swap (Permutation &A, Permutation &B ) noexcept
        {
            // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
            using std::swap;

            swap( A.p,                 B.p                 );
            swap( A.p_inv,             B.p_inv             );
            swap( A.scratch,           B.scratch           );
            swap( A.thread_count,      B.thread_count      );
            swap( A.is_trivial,        B.is_trivial        );
            swap( A.p_computed,        B.p_computed        );
            swap( A.p_inv_computed,    B.p_inv_computed    );
        }
        
        // Copy assignment operator
        Permutation & operator=(Permutation other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        // Move constructor
        Permutation( Permutation && other ) noexcept : Permutation()
        {
            swap(*this, other);
        }
        
        
        
    public:
        
        bool IsTrivial() const
        {
            return is_trivial;
        }

        void SetTrivial()
        {
            p.iota();
            p_inv.iota();
            is_trivial     = true;
            p_computed     = true;
            p_inv_computed = true;
        }
        
        Int Size() const
        {
            return p.Size();
        }
        
        Int ThreadCount() const
        {
            return thread_count;
        }
        
        const Int & operator()( Int i )
        {
            p[i];
        }
        
        template<typename J>
        void SetPermutation( ptr<J> p_ )
        {
            // TODO: check p_ for triviality during copy.
            is_trivial = true;
            
            const Int n = Size();

            if( thread_count > 1 )
            {
                #pragma omp parallel for num_threads( thread_count ) reduction( && : is_trivial )
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_i = static_cast<Int>(p_[i]);
                    
                    is_trivial = is_trivial && (p_i == i );
                    
                    p[i] = p_i;
                    p_inv[p_i] = i;
                }
            }
            else
            {
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_i = static_cast<Int>(p_[i]);

                    is_trivial = is_trivial && (p_i == i );
                    
                    p[i] = p_i;
                    p_inv[p_i] = i;
                }
            }
            
            p_computed     = true;
            p_inv_computed = true;
        }
        
        const Tensor1<Int,Int> & GetPermutation() const
        {
//            RequirePermutation();
            return p;
        }
        
        template<typename J>
        void SetInversePermutation( ptr<J> p_inv_ )
        {
            print("SetInversePermutation");
            // TODO: check p_inv_ for triviality during copy.
//            p_inv.Read(p_inv_);

            is_trivial = true;
            
            const Int n = Size();
            
            if( thread_count > 1 )
            {
                #pragma omp parallel for num_threads( thread_count ) reduction( && : is_trivial)
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_inv_i = static_cast<Int>(p_inv_[i]);
                    
                    is_trivial = is_trivial && (p_inv_i == i );
                    
                    p_inv[i] = p_inv_i;
                    p[p_inv_i] = i;
                }
            }
            else
            {
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_inv_i = static_cast<Int>(p_inv_[i]);
                    
                    is_trivial = is_trivial && (p_inv_i == i );
                    
                    p_inv[i] = p_inv_i;
                    p[p_inv_i] = i;
                }
            }
            
            p_computed     = true;
            p_inv_computed = true;
        }
        
        const Tensor1<Int,Int> & GetInversePermutation() const
        {
//            RequireInversePermutation();
            return p_inv;
        }
        
        void Invert( const Inverse inverse )
        {
            if( inverse == Inverse::True )
            {
                using std::swap;
                
                swap( p, p_inv );
                swap( p_computed, p_inv_computed );
            }
        }
        
        void RequirePermutation()
        {
            if( !p_computed )
            {
                const Int n = Size();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < n; ++i )
                {
                    p[p_inv[i]] = i;
                }
            }
        }
        
        void RequireInversePermutation()
        {
            if( !p_inv_computed )
            {
                const Int n = Size();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < n; ++i )
                {
                    p_inv[p[i]] = i;
                }
            }
        }
        
        
        void Compose( const Permutation & q, const Compose prepost = Compose::Post )
        {
            if( is_trivial )
            {
                if( q.is_trivial )
                {
                    is_trivial = true;
                }
                else
                {
                    // TODO: Check for triviality.
                    p.Read( q.GetPermutation().data() );
                    p_inv.Read( q.GetInversePermutation().data() );

                    is_trivial     = false;
                    p_computed     = true;
                    p_inv_computed = true;
                }
            }
            else
            {
                if( q.is_trivial )
                {
                    // Do nothing.
                }
                else
                {
                    //Careful! This invalidates pointers!
                    
                    is_trivial     = false;
                    p_computed     = true;
                    p_inv_computed = true;
                    
//                    ptr<Int> q_    = q.GetPermutation().data();
//                    ptr<Int> q_inv = q.GetInversePermutation().data();
                    
                    const Int * restrict a     = nullptr;
                    const Int * restrict b     = nullptr;
                    const Int * restrict a_inv = nullptr;
                    const Int * restrict b_inv = nullptr;
                    
                    if( prepost == Compose::Post )
                    {
                        // i.e. do p[i] <- p[q[i]];
                        
                        a =   GetPermutation().data();
                        b = q.GetPermutation().data();
                        
                        a_inv =   GetInversePermutation().data();
                        b_inv = q.GetInversePermutation().data();
                    }
                    else
                    {
                        // i.e. do p[i] <- q[p[i]];
                        
                        a = q.GetPermutation().data();
                        b =   GetPermutation().data();
                        
                        a_inv = q.GetInversePermutation().data();
                        b_inv =   GetInversePermutation().data();
                    }
                    
                    #pragma omp parallel for num_threads( thread_count )
                    for( Int i = 0; i < Size(); ++i )
                    {
                        scratch[i] = a[b[i]];
                    }
                        
                    swap(p,scratch);
                    
                    //post
                    #pragma omp parallel for num_threads( thread_count )
                    for( Int i = 0; i < Size(); ++i )
                    {
                        scratch[i] = b_inv[a_inv[i]];
                    }
                    
                    swap(p_inv,scratch);
                }
            }
        }

//        void Permute( ptr<Int> a, const Inverse inverse )
//        {
//            // "In-place" permutation using scratch space
//
//            scratch.Read(a);
//
//            if( !is_trivial )
//            {
//                Invert( inverse );
//
//                ptr<Int> r = GetPermutation().data();
//
//                #pragma omp parallel for num_threads( thread_count )
//                for( Int i = 0; i < Size(); ++i )
//                {
//                    const Int j = r[i];
//
//                    if( i != j )
//                    {
//                        a[i] = scratch[j];
//                    }
//                }
//
//                Invert( inverse );
//            }
//        }
        
        template<typename T>
        void Permute( ptr<T> a, mut<T> b, const Inverse inverse = Inverse::False )
        {
            // Permute a into b, i.e., b[i] <- a[p[i]];
            ptic(ClassName()+"::Permute");
            
            if( !is_trivial )
            {
                Invert( inverse );
                
                ptr<Int> r = GetPermutation().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < Size(); ++i )
                {
                    b[i] = a[r[i]];
                }
                
                Invert( inverse );
            }
            else
            {
                copy_buffer(a, b, Size(), thread_count );
            }
            
            ptoc(ClassName()+"::Permute");
        }
        
        template<typename T>
        void PermuteRowPointers( ptr<T> a, mut<T> b, const Inverse inverse = Inverse::False )
        {
            // Permute a into b, i.e., b[i] <- a[p[i]];
            ptic(ClassName()+"::Permute");
            
            if( !is_trivial )
            {
                Invert( inverse );
                
                ptr<Int> r = GetPermutation().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < Size(); ++i )
                {
                    b[i] = a[r[i]];
                }
                
                Invert( inverse );
            }
            else
            {
                copy_buffer(a, b, Size(), thread_count );
            }
            
            ptoc(ClassName()+"::Permute");
        }
        
        template<typename T>
        void Permute( ptr<T> a, mut<T> b, const size_t chunk, const Inverse inverse )
        {
            // Permute a chunkwise into b, i.e., b[size*i+k] <- a[size*p[i]+k];
            ptic(ClassName()+"::Permute ("+ToString(chunk)+")");
            if( !is_trivial )
            {
                Invert( inverse );
                
                ptr<Int> r = GetPermutation().data();

                const Int n = Size();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < n; ++i )
                {
                    // a[r[i]] -> b[i].
                    copy_buffer( &a[chunk * r[i]], &b[chunk * i], chunk );
                }
                
                Invert( inverse );
            }
            else
            {
                copy_buffer(a, b, Size()*chunk, thread_count );
            }
            
            ptoc(ClassName()+"::Permute ("+ToString(chunk)+")");
        }
        
        // Somewhat dangereous. Use this only if you know what you are doing!
        Tensor1<Int,Int> & Scratch()
        {
            return scratch;
        }
        
        // Somewhat dangerrous. Use this only if you know what you are doing!
        void SwapScratch( const Inverse inverse = Inverse::False )
        {
            const Int n = p.Size();
            
            is_trivial = true;
            
            if( inverse == Inverse::False )
            {
                swap( p, scratch );
                
                #pragma omp parallel for num_threads( thread_count ) reduction( && : is_trivial )
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_i = p[i];
                    
                    is_trivial = is_trivial && (p_i == i );
                    
                    p_inv[p_i] = i;
                }
            }
            else
            {
                swap( p_inv, scratch );
                
                #pragma omp parallel for num_threads( thread_count ) reduction( && : is_trivial )
                for( Int i = 0; i < n; ++i )
                {
                    const Int p_inv_i = p_inv[i];
                    
                    is_trivial = is_trivial && (p_inv_i == i );
                    
                    p[p_inv_i] = i;
                }
            }

            p_computed     = true;
            p_inv_computed = true;
        }
        
        bool IsValidPermutation() const
        {
            const Int n = Size();
            
            if( (n == 0) )
            {
                return true;
            }
            
            mut<Int> s = scratch.data();

            {
                scratch.SetZero();
                
                ptr<Int> r = GetPermutation().data();
                
                const Int n = Size();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < n; ++i )
                {
                    s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                }
                
                auto m = minmax_buffer( scratch.data(), n );
                
                if( !( (m.first == Int(1)) && (m.second == Int(1)) ) )
                {
                    eprint(ClassName()+"::IsValidPermutation: field p is not a permutation!");
                    return false;
                }
            }

            {
                scratch.SetZero();
                
                ptr<Int> r = GetInversePermutation().data();
                
                const Int n = Size();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < n; ++i )
                {
                    s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                }
                
                std::pair<Int,Int> m = minmax_buffer( scratch.data(), n );
                
                if( !( (m.first == Int(1)) && (m.second == Int(1)) ) )
                {
                    eprint(ClassName()+"::IsValidPermutation: field p_inv is not a permutation!");
                    return false;
                }
            }

            // We may run this final test only if the other two are passed because they guarantee
            // that we won't get a segfault.
            {
                Int fails = 0;
                ptr<Int> p_     = GetPermutation().data();
                ptr<Int> p_inv_ = GetInversePermutation().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < Size(); ++i )
                {
                    fails += static_cast<Int>(i != p_inv_[p_[i]]);
                }
                
                if( fails > 0 )
                {
                    eprint(ClassName()+"::Check: field p_inv is not the inverse of p!");
                    
                    return false;
                }
            }

            return true;
        }
    
        
                       
    public:
        
        std::string ClassName() const
        {
            return "Sparse::CholeskyFactorizer<"+TypeName<Int>::Get()+">";
        }

    }; // class Permutation
    
    
    
    template<bool P_Trivial, bool Q_Trivial, bool Sort, typename LInt, typename Int, typename ExtLInt, typename ExtInt>
    std::tuple<Tensor1<LInt,Int>,Tensor1<Int,LInt>,Permutation<LInt>>
    sparseMatrixPermutation(
        ptr<ExtLInt>     & outer,
        ptr<ExtInt>      & inner,
        const Permutation<Int> & P,  // row    permutation
        const Permutation<Int> & Q,  // column permutation
        const LInt nnz,
        bool sort = true
    )
    {
        const Int m = P.Size();
        
        const Int thread_count = P.ThreadCount();
        
        ptr<Int> p     = P.GetPermutation().data();
        ptr<Int> q_inv = Q.GetInversePermutation().data();

        Tensor1<LInt, Int> new_outer ( m+1 );
        Tensor1< Int,LInt> new_inner ( nnz );
        Permutation<LInt>  perm      ( nnz );
        
        if constexpr ( P_Trivial )
        {
            copy_buffer( outer, new_outer.data(), m+1, thread_count );
            
            if constexpr ( Q_Trivial )
            {
                copy_buffer( inner, new_inner.data(), nnz, thread_count );
            }
        }
        else
        {
            new_outer[0] = 0;
            
            #pragma omp parallel for num_threads( thread_count ) schedule( static )
            for( Int i = 0; i < m; ++i )
            {
                const Int p_i = p[i];
                
                new_outer[i+1] = static_cast<LInt>(outer[p_i+1] - outer[p_i]);
            }
            
            parallel_accumulate( new_outer.data(), m+1, thread_count );
        }

        
//        if constexpr ( !P_Trivial || !Q_Trivial )
//        {
            JobPointers<Int> job_ptr ( m, new_outer.data(),  thread_count );
            
            
            mut<LInt> scratch = perm.Scratch().data();
            
            #pragma omp parallel for num_threads( thread_count )
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                TwoArrayQuickSort<Int,LInt,Int> quick_sort;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt begin = outer[ COND( P_Trivial, i, p[i] ) ];
                    
                    const LInt new_begin = new_outer[i  ];
                    const LInt new_end   = new_outer[i+1];
                    
                    const LInt k_max = new_end - new_begin;
                    
                    if constexpr ( Q_Trivial )
                    {
                        copy_buffer( &inner[begin], &new_inner[new_begin], k_max );
                        
                        for( LInt k = 0; k < k_max; ++k )
                        {
                            scratch[new_begin+k] = begin+k;
                        }
                    }
                    else
                    {
                        for( LInt k = 0; k < k_max; ++k )
                        {
                            new_inner[new_begin+k] = q_inv[inner[begin+k]];
                            scratch  [new_begin+k] = begin+k;
                        }
                        
                        if constexpr ( Sort )
                        {
                            quick_sort( &new_inner[new_begin], &scratch [new_begin], static_cast<Int>(k_max) );
                        }
                    }
                }
            }
            
            perm.SwapScratch();

        return std::make_tuple( new_outer, new_inner, perm );
    }
    
    template<typename Int, typename LInt, typename ExtLInt, typename ExtLInt2, typename ExtInt>
    std::tuple<Tensor1<LInt,Int>,Tensor1<Int,LInt>,Permutation<LInt>>
    SparseMatrixPermutation(
        ptr<ExtLInt>     & outer,
        ptr<ExtInt>      & inner,
        const Permutation<Int> & P,  // row    permutation
        const Permutation<Int> & Q,  // column permutation
        const ExtLInt2 nnz,
        bool sort = true        // Whether to restore row-wise ordering (as in demanded by CSR).
    )
    {
        // returns
        // i  ) The permuted array "outer".
        // ii ) The permuted array "inner".
        // iii) The permutation that has to be applied to the nonzero values.
        
//        if( P.IsTrivial() )
//        {
//            if( Q.IsTrivial() )
//            {
//                if( sort )
//                {
//                    return sparseMatrixPermutation<true,true,true,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//                else
//                {
//                    return sparseMatrixPermutation<true,true,false,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//            }
//            else
//            {
//                if( sort )
//                {
//                    return sparseMatrixPermutation<true,false,true,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//                else
//                {
//                    return sparseMatrixPermutation<true,false,false,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//            }
//        }
//        else // if( !p.IsTrivial() )
//        {
//            if( Q.IsTrivial() )
//            {
//                if( sort )
//                {
//                    return sparseMatrixPermutation<false,true,true,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//                else
//                {
//                    return sparseMatrixPermutation<false,true,false,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//            }
//            else
//            {
//                if( sort )
//                {
//                    return sparseMatrixPermutation<false,false,true,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//                else
//                {
//                    return sparseMatrixPermutation<false,false,false,LInt,Int>(outer,inner,P,Q,nnz);
//                }
//            }
//        }
        
        if( sort )
        {
            return sparseMatrixPermutation<false,false,true,LInt,Int>(outer,inner,P,Q,nnz);
        }
        else
        {
            return sparseMatrixPermutation<false,false,false,LInt,Int>(outer,inner,P,Q,nnz);
        }
    }
    
    
}
