#pragma once

namespace Tensors
{
    
    enum class Compose : bool
    {
        Post = true,
        Pre  = false
    };
    
    enum class Inverse : bool
    {
        True  = true,
        False = false
    };
    
    template<typename Int_>
    class Permutation
    {
        ASSERT_INT(Int_);
     
    public:
        
        using Int = Int_;
        
    protected:
        
        Int n;
        
        mutable Tensor1<Int,Int> p;
        mutable Tensor1<Int,Int> p_inv;
        
        mutable Tensor1<Int,Int> scratch;
        
        Int thread_count = 1;
        
        mutable bool is_trivial     = true;
        mutable bool p_computed     = true;
        mutable bool p_inv_computed = true;
        mutable bool is_valid       = false;
        
        static constexpr Int zero = Scalar::Zero<Int>;
        static constexpr Int one  = Scalar::One<Int>;
        
    public:
        
        Permutation()
        :   n              ( zero  )
        ,   p              ( n     )
        ,   p_inv          ( n     )
        ,   scratch        ( n     )
        ,   thread_count   ( one   )
        ,   is_trivial     ( true  )
        ,   p_computed     ( true  )
        ,   p_inv_computed ( true  )
        ,   is_valid       ( true  )    // Yeah, we say that an empty permutation is valid! ^^
        {}
        
        Permutation( const Int n_, const Int thread_count_ )
        :   n              ( n_               )
        ,   p              ( iota<Int,Int>(n) )
        ,   p_inv          ( iota<Int,Int>(n) )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        ,   is_trivial     ( true             )
        ,   p_computed     ( true             )
        ,   p_inv_computed ( true             )
        ,   is_valid       ( n_ > zero        )
        {}
        
        
        Permutation( Tensor1<Int,Int> && p_, const Inverse inverseQ, const Int thread_count_ )
        :   n              ( p_.Size()        )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        ,   is_trivial     ( false            )
        ,   p_computed     ( false            )
        ,   p_inv_computed ( false            )
        {
            if( inverseQ == Inverse::True )
            {
                if( !PermutationQ(p_.data()) )
                {
                    eprint(ClassName()+"() input is not a permutation.");
                    is_valid = false;
                    return;
                }
                p_inv = std::move(p_);
                p_inv_computed = true;
                
                RequirePermutation();
                is_valid = true;
            }
            else
            {
                if( !PermutationQ(p_.data()) )
                {
                    eprint(ClassName()+"() input is not a permutation.");
                    is_valid = false;
                    return;
                }
                p = std::move(p_);
                p_computed = true;
                RequireInversePermutation();
                is_valid = true;
            }
        }
        
        template<typename J>
        Permutation( cptr<J> p_, const Int n_, const Inverse inverseQ, const Int thread_count_ )
        :   n              ( n_               )
        ,   p              ( n                )
        ,   p_inv          ( n                )
        ,   scratch        ( n                )
        ,   thread_count   ( thread_count_    )
        {
            ASSERT_INT(J)
            if( !PermutationQ(p_) )
            {
                eprint(ClassName()+"() input is not a permutation.");
                is_valid = false;
                return;
            }
            is_valid = true;
            
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
        
        
        /* Copy constructor */
        Permutation( const Permutation & other )
        :   n                 ( other.n                 )
        ,   p                 ( other.p                 )
        ,   p_inv             ( other.p_inv             )
        ,   scratch           ( other.scratch           )
        ,   thread_count      ( other.thread_count      )
        ,   is_trivial        ( other.is_trivial        )
        ,   p_computed        ( other.p_computed        )
        ,   p_inv_computed    ( other.p_inv_computed    )
        ,   is_valid          ( other.is_valid          )
        {}
        
        // We could also simply use the implicitly created copy constructor.
        
        /* Swap function */
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
            swap( A.is_valid,          B.is_valid          );
        }
        
        /* Copy assignment operator */
        Permutation & operator=(Permutation other)
        {
            // copy-and-swap idiom
            // see https://stackoverflow.com/a/3279550/8248900 for details

            swap(*this, other);

            return *this;
        }

        /* Move constructor */
        Permutation( Permutation && other ) noexcept : Permutation()
        {
            swap(*this, other);
        }
        
        
        
    public:

        bool IsValid() const
        {
            return is_valid;
        }
        
        bool TrivialQ() const
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
        void SetPermutation( cptr<J> p_ )
        {
            // TODO: check p_ for triviality during copy.
            
            is_trivial = ParallelDoReduce(
                [=,this]( const Int i ) -> bool
                {
                    const Int p_i = int_cast<Int>(p_[i]);
                    
                    p[i] = p_i;
                    p_inv[p_i] = i;
                    
                    return (p_i == i);
                },
                AndReducer(), true,
                zero, n, thread_count
            );
            
            p_computed     = true;
            p_inv_computed = true;
        }

        
        // There is no nonconstant GetPermutation() because we want the error-free execution of the constructors to be a certificate.
        cref<Tensor1<Int,Int>> GetPermutation() const
        {
            return p;
        }
        
        template<typename J>
        void SetInversePermutation( cptr<J> p_inv_ )
        {
//            print("SetInversePermutation");
            // TODO: check p_inv_ for triviality during copy.
//            p_inv.Read(p_inv_);

            ParallelDoReduce(
                [=,this]( const Int i ) -> bool
                {
                    const Int p_inv_i = static_cast<Int>(p_inv_[i]);
                    
                    p_inv[i] = p_inv_i;
                    p[p_inv_i] = i;
                    
                    return (p_inv_i == i);
                },
                AndReducer(), true,
                zero, n, thread_count
            );
            
            p_computed     = true;
            p_inv_computed = true;
        }
        

        // There is no nonconstant GetPermutation() because we want the error-free execution of the constructors to be a certificate.
        cref<Tensor1<Int,Int>> GetInversePermutation() const
        {
            return p_inv;
        }

        
        void Invert( const Inverse inverseQ )
        {
//            ptic(ClassName()+"::Invert");
            if( inverseQ == Inverse::True )
            {
                using std::swap;
                
                swap( p, p_inv );
                swap( p_computed, p_inv_computed );
            }
//            ptoc(ClassName()+"::Invert");
        }
        
        void RequirePermutation()
        {
            if( !p_computed )
            {
                if ( p.Size() != n )
                {
                    p = Tensor1<Int,Int>(n);
                }
                
                ParallelDo(
                    [=,this]( const Int i )
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
            if( !p_inv_computed  )
            {
                if ( p_inv.Size() != n )
                {
                    p_inv = Tensor1<Int,Int>(n);
                }
                ParallelDo(
                    [=,this]( const Int i )
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

                    is_trivial     = q.TrivialQ();
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
                    
//                    cptr<Int> q_    = q.GetPermutation().data();
//                    cptr<Int> q_inv = q.GetInversePermutation().data();
                    
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
                        [=,this]( const Int i ) -> bool
                        {
                            scratch[i] = a[b[i]];
                            
                            return (scratch[i] == i);
                        },
                        AndReducer(), true,
                        zero, n, thread_count
                    );
                        
                    swap(p,scratch);
                    
                    //post
                    ParallelDo(
                        [=,this]( const Int i )
                        {
                            scratch[i] = b_inv[a_inv[i]];
                        },
                        n, thread_count
                    );
                    
                    swap(p_inv,scratch);
                }
            }
            ptoc(ClassName()+"::Compose");
        }
        
        // TODO: We could add a leading dimension argument...
        template<typename S, typename T>
        void Permute( cptr<S> a, mptr<T> b, const Inverse inverseQ, Size_T chunk = Scalar::One<Size_T> )
        {
            // Permute a chunkwise into b, i.e., b[size*i+k] <- a[size*p[i]+k];
            
            std::string tag = ClassName()+"::Permute ( " + (inverseQ == Inverse::True ? "inv, " : "id, " ) + ToString(chunk) + " )";
            
            ptic(tag);
            
            if( !is_trivial )
            {
                Invert( inverseQ );
                
                cptr<Int> r = GetPermutation().data();

                if( chunk == Scalar::One<Size_T> )
                {
                    ParallelDo(
                        [=]( const Int i )
                        {
                            b[i] = static_cast<T>(a[r[i]]);
                        },
                        n, thread_count
                    );
                }
                else
                {
                    ParallelDo(
                        [=]( const Int i )
                        {
                            copy_buffer( &a[chunk * r[i]], &b[chunk * i], chunk );
                        },
                        n, thread_count
                    );
                }
                
                Invert( inverseQ );
            }
            else
            {
                copy_buffer<VarSize,Parallel>( a, b, n*chunk, thread_count );
            }
            
            ptoc(tag);
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
        
        
        // Somewhat dangerous. Use this only if you know what you are doing!
        Tensor1<Int,Int> & Scratch()
        {
            return scratch;
        }
        
        // Somewhat dangerous. Use this only if you know what you are doing!
        void SwapScratch( Inverse inverseQ )
        {
            is_trivial = true;
            
            if( inverseQ == Inverse::False )
            {
                swap( p, scratch );
                
                is_trivial = ParallelDoReduce(
                    [=,this]( const Int i ) -> bool
                    {
                        const Int p_i = p[i];
                        
                        p_inv[p_i] = i;
                        
                        return (p_i == i);
                    },
                    AndReducer(), true,
                    zero, n, thread_count
                );
            }
            else
            {
                swap( p_inv, scratch );
                
                is_trivial = ParallelDoReduce(
                    [=,this]( const Int i ) -> bool
                    {
                        const Int p_inv_i = p_inv[i];
                        
                        p[p_inv_i] = i;
                        
                        return (p_inv_i == i);
                    },
                    AndReducer(), true,
                    zero, n, thread_count
                );
            }

            p_computed     = true;
            p_inv_computed = true;
        }
        
        
        template<typename J>
        bool PermutationQ( cptr<J> p_ ) const
        {
            std::string tag = ClassName() + "::PermutationQ";
            
            ptic(tag);
            if( (n == zero) || (n == one ) )
            {
                ptoc(tag);
                return true;
            }
            
            scratch.SetZero();
            
            for( Int i = 0; i < n; ++i )
            {
                const Int p_i = int_cast<Int>(p_[i]);
                
                if( (p_i < zero) || (p_i >= n) )
                {
                    wprint(tag + ": Input list p has value p["+ToString(i)+"] = "+ToString(p_i)+" out of range [0,"+ToString(n)+"[!");
                    ptoc(tag);
                    return false;
                }
                else
                {
                    ++scratch[p_i];
                }
            }

            const auto [m_0, m_1] = minmax_buffer( scratch.data(), n );

            if( m_0 != one )
            {
                eprint(tag + ": Input does not attain all values in range!");
            }
            
            if( m_1 != one )
            {
                eprint(tag + ": Input has duplicates!");
            }
            
            ptoc(tag);
            
            return ( m_0 == one ) && ( m_1 );
        }
        
        bool PermutationQ() const
        {
            if( PermutationQ(p.data()) )
            {
                if( PermutationQ(p_inv.data()) )
                {
                    return true;
                }
                else
                {
                    eprint(ClassName()+"::PermutationQ: field p_inv is not a permutation!");
                    return false;
                }
            }
            else
            {
                eprint(ClassName()+"::PermutationQ: field p is not a permutation!");
                return false;
            }
        }
                       
    public:
        
        std::string ClassName() const
        {
            return std::string("Permutation")+"<"+TypeName<Int>+">";
        }

    }; // class Permutation
    
    
    
    template<bool P_TrivialQ, bool Q_TrivialQ, bool SortQ, typename LInt, typename Int>
    Tensor1<LInt,LInt> permutePatternCSR(
        mref<Tensor1<LInt,Int>> outer,
        mref<Tensor1<Int,LInt>> inner,
        cref<Permutation<Int>>  P,  // row    permutation
        cref<Permutation<Int>>  Q,  // column permutation
        const LInt nnz,
        bool  sort = true
    )
    {
        std::string tag = std::string("PermutePatternCSR<") + TypeName<LInt> + "," + TypeName<Int> + ">";
        
        ptic(tag);
        
        const Int m = P.Size();

        const Int thread_count = P.ThreadCount();
        
        cptr<Int> p     = P.GetPermutation().data();
        cptr<Int> q_inv = Q.GetInversePermutation().data();

        Tensor1<LInt, Int> new_outer ( m+1 );
        Tensor1< Int,LInt> new_inner ( nnz );
        
        Tensor1<LInt,LInt> perm ( nnz );

        if constexpr ( P_TrivialQ && Q_TrivialQ )
        {
            ptoc(tag);
            perm.iota();
            return perm;
        }
            
        if constexpr ( P_TrivialQ )
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
                m, thread_count
            );

            parallel_accumulate( new_outer.data(), m+1, thread_count );
        }
        
        mptr<LInt> scratch = perm.data();

        JobPointers<Int> job_ptr ( m, new_outer.data(), thread_count );
        
        ParallelDo(
            [&,scratch]( const Int thread )
            {
                TwoArraySort<Int,LInt,Int> S;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    const LInt begin = outer[ COND( P_TrivialQ, i, p[i] ) ];
                    
                    const LInt new_begin = new_outer[i  ];
                    const LInt new_end   = new_outer[i+1];
                    
                    const LInt k_max = new_end - new_begin;
                    
                    if constexpr ( Q_TrivialQ )
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
                        
                        if constexpr ( SortQ )
                        {
                            S( &new_inner[new_begin], &scratch [new_begin], static_cast<Int>(k_max) );
                        }
                    }
                }
            },
            thread_count
        );
        
        swap( outer, new_outer );
        swap( inner, new_inner );
        
        ptoc(tag);

        return perm;
    }
    
    template<typename Int, typename LInt>
    Tensor1<LInt,LInt> PermutePatternCSR(
        mref<Tensor1<LInt,Int>> outer,
        mref<Tensor1<Int,LInt>> inner,
        cref<Permutation<Int>> P,  // row    permutation
        cref<Permutation<Int>> Q,  // column permutation
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
