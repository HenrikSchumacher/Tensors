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
        Int n;
        
        mutable Tensor1<Int,Int> p;
        mutable Tensor1<Int,Int> p_inv;
        
        mutable Tensor1<Int,Int> scratch;
        
        Int thread_count = 1;
        
        mutable bool is_trivial     = true;
        mutable bool p_computed     = true;
        mutable bool p_inv_computed = true;
        
    public:
        
        Permutation()
        :   n              ( 0     )
        ,   p              ( n     )
        ,   p_inv          ( n     )
        ,   scratch        ( n     )
        ,   thread_count   ( 1     )
        ,   is_trivial     ( true  )
        ,   p_computed     ( true  )
        ,   p_inv_computed ( true  )
        {}
        
        explicit Permutation( const Int n_, const Int thread_count_ )
        :   n              ( n_               )
        ,   p              ( iota<Int,Int>(n) )
        ,   p_inv          ( iota<Int,Int>(n) )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        ,   is_trivial     ( true             )
        ,   p_computed     ( true             )
        ,   p_inv_computed ( true             )
        {}
        
        template<typename J, IS_INT(J)>
        Permutation( ptr<J> p_, const Int n_, const Inverse inverseQ, const Int thread_count_ )
        :   n              ( n_               )
        ,   p              ( n                )
        ,   p_inv          ( n                )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        {
            if( inverseQ == Inverse::True )
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
        :   n                 ( other.n                 )
        ,   p                 ( other.p                 )
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

            swap( A.n,                 B.n                 );
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
            return n;
        }
        
        Int ThreadCount() const
        {
            return thread_count;
        }
        
        const Int & operator()( const Int i )
        {
            return p[i];
        }
        
        template<typename J>
        void SetPermutation( ptr<J> p_ )
        {
            // TODO: check p_ for triviality during copy.
            
            is_trivial = ParallelDoReduce(
                [=]( const Int i ) -> bool
                {
                    const Int p_i = static_cast<Int>(p_[i]);
                    
                    p[i] = p_i;
                    p_inv[p_i] = i;
                    
                    return p_i == i;
                },
                AndReducer(),
                true,
                n,
                thread_count
            );
            
            p_computed     = true;
            p_inv_computed = true;
        }
        
        const Tensor1<Int,Int> & GetPermutation() const
        {
            return p;
        }
        
        template<typename J>
        void SetInversePermutation( ptr<J> p_inv_ )
        {
            print("SetInversePermutation");
            // TODO: check p_inv_ for triviality during copy.
//            p_inv.Read(p_inv_);

            ParallelDoReduce(
                [=]( const Int i ) -> bool
                {
                    const Int p_inv_i = static_cast<Int>(p_inv_[i]);
                    
                    p_inv[i] = p_inv_i;
                    p[p_inv_i] = i;
                    
                    return p_inv_i == i;
                },
                AndReducer(),
                true,
                n,
                thread_count
            );
            
            p_computed     = true;
            p_inv_computed = true;
        }
        
        const Tensor1<Int,Int> & GetInversePermutation() const
        {
            return p_inv;
        }
        
        void Invert( const Inverse inverseQ )
        {
            ptic(ClassName()+"::Invert");
            if( inverseQ == Inverse::True )
            {
                using std::swap;
                
                swap( p, p_inv );
                swap( p_computed, p_inv_computed );
            }
            ptoc(ClassName()+"::Invert");
        }
        
        void RequirePermutation()
        {
            if( !p_computed )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        p[p_inv[i]] = i;
                    },
                    n,
                    thread_count
                );
            }
        }
        
        void RequireInversePermutation()
        {
            if( !p_inv_computed )
            {
                ParallelDo(
                    [=]( const Int i )
                    {
                        p_inv[p[i]] = i;
                    },
                    n,
                    thread_count
                );
            }
        }
        
        
        void Compose( const Permutation & q, const Compose prepost )
        {
            ptic(ClassName()+"::Compose");
            
            if( is_trivial )
            {
                if( q.is_trivial )
                {
                    // Do nothing.
                }
                else
                {
                    p.Read( q.GetPermutation().data() );
                    p_inv.Read( q.GetInversePermutation().data() );

                    is_trivial     = q.IsTrivial();
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
                    
                    is_trivial = ParallelDoReduce(
                        [=]( const Int i ) -> bool
                        {
                            scratch[i] = a[b[i]];
                            
                            return scratch[i] == i;
                        },
                        AndReducer(),
                        true,
                        n,
                        thread_count
                    );
                        
                    swap(p,scratch);
                    
                    //post
                    ParallelDo(
                        [=]( const Int i )
                        {
                            scratch[i] = b_inv[a_inv[i]];
                        },
                        n,
                        thread_count
                    );
                    
                    swap(p_inv,scratch);
                }
            }
            ptoc(ClassName()+"::Compose");
        }
        
        template<typename S, typename T>
        void Permute( ptr<S> a, mut<T> b, const Inverse inverseQ )
        {
            // Permute a into b, i.e., b[i] <- a[p[i]];
            ptic(ClassName()+"::Permute");
            
            if( !is_trivial )
            {
                Invert( inverseQ );
                
                ptr<Int> r = GetPermutation().data();
                
                ParallelDo(
                    [=]( const Int i )
                    {
                        b[i] = static_cast<T>(a[r[i]]);
                    },
                    n,
                    thread_count
                );

                
                Invert( inverseQ );
            }
            else
            {
                copy_buffer(a, b, n, thread_count );
            }
            
            ptoc(ClassName()+"::Permute");
        }
        
        template<typename S, typename T>
        void Permute( const Tensor1<S,Int> & a, Tensor1<T,Int> & b, const Inverse inverseQ )
        {
            if( a.Size() != n )
            {
                eprint(ClassName()+"::Permute: First input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( b.Size() != n )
            {
                eprint(ClassName()+"::Permute: Second input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            Permute( a.data(), b.data(), inverseQ );
        }
        
        template<typename S, typename T>
        void Permute( ptr<S> a, mut<T> b, const Inverse inverseQ, size_t chunk )
        {
            // Permute a chunkwise into b, i.e., b[size*i+k] <- a[size*p[i]+k];
            ptic(ClassName()+"::Permute ("+ToString(chunk)+")");
            if( !is_trivial )
            {
                Invert( inverseQ );
                
                ptr<Int> r = GetPermutation().data();

                ParallelDo(
                    [=]( const Int i )
                    {
                        copy_buffer( &a[chunk * r[i]], &b[chunk * i], chunk );
                    },
                    n,
                    thread_count
                );
                
                Invert( inverseQ );
            }
            else
            {
                copy_buffer(a, b, n*chunk, thread_count );
            }
            
            ptoc(ClassName()+"::Permute ("+ToString(chunk)+")");
        }
        
        template<typename S, typename T>
        void Permute( const Tensor2<S,Int> & a, Tensor2<T,Int> & b, const Inverse inverseQ )
        {
            if( a.Size() != n )
            {
                eprint(ClassName()+"::Permute: First input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( b.Size() != n )
            {
                eprint(ClassName()+"::Permute: Second input array has incorrect dimension. Doing nothing.");
                return;
            }
            
            if( a.Dimension(1) != b.Dimension(1) )
            {
                eprint(ClassName()+"::Permute: Number of columns of input arrays do not coincide. Doing nothing.");
                return;
            }
            
            Permute( a.data(), b.data(), inverseQ, a.Dimension(1) );
        }
        
        
        // Somewhat dangereous. Use this only if you know what you are doing!
        Tensor1<Int,Int> & Scratch()
        {
            return scratch;
        }
        
        // Somewhat dangerrous. Use this only if you know what you are doing!
        void SwapScratch( Inverse inverseQ )
        {
            is_trivial = true;
            
            if( inverseQ == Inverse::False )
            {
                swap( p, scratch );
                
                is_trivial = ParallelDoReduce(
                    [=]( const Int i ) -> bool
                    {
                        const Int p_i = p[i];
                        
                        p_inv[p_i] = i;
                        
                        return p_i == i;
                    },
                    AndReducer(),
                    true,
                    n,
                    thread_count
                );
            }
            else
            {
                swap( p_inv, scratch );
                
                is_trivial = ParallelDoReduce(
                    [=]( const Int i ) -> bool
                    {
                        const Int p_inv_i = p_inv[i];
                        
                        p[p_inv_i] = i;
                        
                        return p_inv_i == i;
                    },
                    AndReducer(),
                    true,
                    n,
                    thread_count
                );
            }

            p_computed     = true;
            p_inv_computed = true;
        }
        
        bool IsValidPermutation() const
        {
            if( (n == 0) )
            {
                return true;
            }
            
            mut<Int> s = scratch.data();

            {
                scratch.SetZero();
                
                ptr<Int> r = GetPermutation().data();
                
                ParallelDo(
                    [=]( const Int i )
                    {
                        s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                    },
                    n,
                    thread_count
                );
                
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
                
                ParallelDo(
                    [=]( const Int i )
                    {
                        s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                    },
                    n,
                    thread_count
                );
                
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
                
                ParallelDo(
                    [=]( const Int i )
                    {
                        fails += static_cast<Int>(i != p_inv_[p_[i]]);
                    },
                    n,
                    thread_count
                );
                
                if( fails > 0 )
                {
                    eprint(ClassName()+"::Check: field p_inv is not the inverseQ of p!");
                    
                    return false;
                }
            }

            return true;
        }
    
        
                       
    public:
        
        std::string ClassName() const
        {
            return "Permutation<"+TypeName<Int>+">";
        }

    }; // class Permutation
    
    
    
    template<bool P_Trivial, bool Q_Trivial, bool Sort, typename LInt, typename Int>
    Permutation<LInt> permutePatternCSR(
        Tensor1<LInt,Int> & outer,
        Tensor1<Int,LInt> & inner,
        const Permutation<Int> & P,  // row    permutation
        const Permutation<Int> & Q,  // column permutation
        const LInt nnz,
        bool sort = true
    )
    {
        ptic("PermutePatternCSR");
        
        const Int m = P.Size();

        const Int thread_count = P.ThreadCount();

        dump(thread_count);
        
        ptr<Int> p     = P.GetPermutation().data();
        ptr<Int> q_inv = Q.GetInversePermutation().data();

        Tensor1<LInt, Int> new_outer ( m+1 );
        Tensor1< Int,LInt> new_inner ( nnz );
        Permutation<LInt>  perm      ( nnz, thread_count );

        if constexpr ( P_Trivial && Q_Trivial )
        {
            ptoc("PermutePatternCSR");
            return perm;
        }
            
        if constexpr ( P_Trivial )
        {
            swap( outer, new_outer );
        }
        else
        {
            new_outer[0] = 0;
            
            ParallelDo(
                [&]( const Int i )
                {
                    const Int p_i = p[i];

                    new_outer[i+1] = static_cast<LInt>(outer[p_i+1] - outer[p_i]);
                },
                m,
                thread_count
            );

            parallel_accumulate( new_outer.data(), m+1, thread_count );
        }

        mut<LInt> scratch = perm.Scratch().data();

        JobPointers<Int> job_ptr ( m, new_outer.data(), thread_count );
        
        ParallelDo(
            [&,scratch]( const Int thread )
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
            },
            thread_count
        );
        
        swap( outer, new_outer );
        swap( inner, new_inner );
        
        perm.SwapScratch( Inverse::False );
        
        ptoc("PermutePatternCSR");

        return perm;
    }
    
    template<typename Int, typename LInt>
    Permutation<LInt> PermutePatternCSR(
        Tensor1<LInt,Int> & outer,
        Tensor1<Int,LInt> & inner,
        const Permutation<Int> & P,  // row    permutation
        const Permutation<Int> & Q,  // column permutation
        const LInt nnz,
        bool sort = true        // Whether to restore row-wise ordering (as in demanded by CSR).
    )
    {
        // returns
        // i  ) The permuted array "outer".
        // ii ) The permuted array "inner".
        // iii) The permutation that has to be applied to the nonzero values.
        
        if( sort )
        {
            return permutePatternCSR<false,false,true,LInt,Int>(outer,inner,P,Q,nnz);
        }
        else
        {
            return permutePatternCSR<false,false,false,LInt,Int>(outer,inner,P,Q,nnz);
        }
    }
    
    
} // namespace Tensors
