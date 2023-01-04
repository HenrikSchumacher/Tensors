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
        Tensor1<Int,Int> p;
        Tensor1<Int,Int> p_inv;
        
        mutable Tensor1<Int,Int> scratch;
        
        Int thread_count = 1;
        
        bool is_trivial     = true;
        bool p_computed     = true;
        bool p_inv_computed = true;
        
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
        
        Permutation( ptr<Int> p_, const Int n, const Inverse inverse, const Int thread_count_ = 1 )
        :   p              ( n                )
        ,   p_inv          ( n                )
        ,   scratch        ( n                )
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
        
        
        bool IsTrivial() const
        {
            is_trivial;
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
        
        const Int & operator()( Int i )
        {
            p[i];
        }
        
        void SetPermutation( ptr<Int> p_ )
        {
            // TODO: check p_ for triviality during copy.
            p.Read(p_);
            is_trivial     = false;
            p_computed     = true;
            p_inv_computed = false;
        }
        
        const Tensor1<Int,Int> & GetPermutation()
        {
            RequirePermutation();
            return p;
        }
        
        void SetInversePermutation( ptr<Int> p_inv_ )
        {
            // TODO: check p_inv_ for triviality during copy.
            p_inv.Read(p_inv);
            is_trivial     = false;
            p_computed     = false;
            p_inv_computed = true;
        }
        
        const Tensor1<Int,Int> & GetInversePermutation()
        {
            RequireInversePermutation();
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
                    
                    ptr<Int> q_    = q.GetPermutation().data();
                    ptr<Int> q_inv = q.GetInversePermutation().data();
                    
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
        void Permute( ptr<T> a, mut<T> b, const Inverse inverse )
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
            if( inverse == Inverse::False )
            {
                swap( p, scratch );
            }
            else
            {
                swap( p_inv, scratch );
            }
            
            p_computed     = false;
            p_inv_computed = false;
            is_trivial     = false;
        }
        
        bool IsValidPermutation()
        {
            const Int n = Size();
            
            if( (n == 0) || is_trivial )
            {
                return true;
            }
            
            mut<Int> s = scratch.data();
            
            {
                scratch.SetZero();
                
                ptr<Int> r = GetPermutation().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < Size(); ++i )
                {
                    s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                }
                
                std::pair<Int,Int> m = std::minmax( scratch.begin(), scratch.end() );
                
                if( !( (m.first == Int(1)) && (m.second == Int(1)) ) )
                {
                    eprint(ClassName()+"::IsValidPermutation: field p is not a permutation!");
                    return false;
                }
            }
            
            {
                scratch.SetZero();
                
                ptr<Int> r = GetInversePermutation().data();
                
                #pragma omp parallel for num_threads( thread_count )
                for( Int i = 0; i < Size(); ++i )
                {
                    s[r[i]] += static_cast<Int>((Int(0) <= r[i]) && (r[i] < n));
                }
                
                std::pair<Int,Int> m = std::minmax( scratch.begin(), scratch.end() );
                
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
                }
            }
        }
                
        
                       
    public:
        
        std::string ClassName() const
        {
            return "Sparse::CholeskyFactorizer<"+TypeName<Int>::Get()+">";
        }

    }; // class Permutation
    
}
