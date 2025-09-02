#pragma once

namespace Tensors
{
    namespace Sparse
    {
        // TODO: This is a very incomplete class that can only hold sorted indices and values. More work has to be done on this.
        template<typename Scal_, typename Int_>
        class Vector final
        {
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            
        protected:
            
            Int n = 0;
            Int thread_count = 1;
            Tensor1<Int ,Int> idx;  // Needs to be sorted!
            Tensor1<Scal,Int> val;
            
        public:
            
            template<typename I_0, typename I_1, typename I_2>
            Vector(
                   const I_0 n_,
                   const I_1 nnz_,
                   const I_2 thread_count_
                   )
            :   n            { int_cast<Int>(n_)            }
            ,   idx          { int_cast<Int>(nnz_)          }
            ,   val          { int_cast<Int>(nnz_)          }
            ,   thread_count { int_cast<Int>(thread_count_) }
            {
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_2>,"");
            }
            
            template<typename S, typename J_0, typename I_0, typename I_1, typename I_2>
            Vector(
                   const J_0 * const idx_,  // Needs t be sorted!
                   const S   * const val_,
                   const I_0 n_,
                   const I_1 nnz_,
                   const I_2 thread_count_
                   )
            :   n    { int_cast<Int>(n_)         }
            ,   idx  { idx_, int_cast<Int>(nnz_) }
            ,   val  { val_, int_cast<Int>(nnz_) }
            ,   thread_count { int_cast<Int>(thread_count_) }
            {
                static_assert(ArithmeticQ<S>,"");
                static_assert(IntQ<I_0>,"");
                static_assert(IntQ<I_1>,"");
                static_assert(IntQ<I_2>,"");
            }
            
            // Default constructor
            Vector() = default;
            // Destructor
            ~Vector() = default;
            // Copy constructor
            Vector( const Vector & other ) = default;
            // Copy assignment operator
            Vector & operator=( const Vector & other ) = default;
            // Move constructor
            Vector( Vector && other ) = default;
            // Move assignment operator
            Vector & operator=( Vector && other ) = default;
            
            
            friend void swap (Vector & A, Vector & B ) noexcept
            {
                using std::swap;
                
                swap( A.n,            B.n            );
                swap( A.thread_count, B.thread_count );
                swap( A.idx,          B.idx          );
                swap( A.val,          B.val          );
            }
            
        public:

            
            Int Size() const
            {
                return n;
            }
            
            Int NonzeroCount() const
            {
                return idx.Size();
            }
            
            mref<Tensor1<Scal,Int>> Indices()
            {
                return idx;
            }
            
            cref<Tensor1<Scal,Int>> Indices() const
            {
                return idx;
            }
            
            mref<Tensor1<Scal,Int>> Values()
            {
                return val;
            }
            
            cref<Tensor1<Scal,Int>> Values() const
            {
                return val;
            }
            
            mref<Tensor1<Scal,Int>> Value()
            {
                return val;
            }
            
            cref<Tensor1<Scal,Int>> Value() const
            {
                return val;
            }
            
            mref<Scal> Value( const Int k )
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < Int(0) || k >= val.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return val[k];
            }
            
            cref<Scal> Value( const Int k ) const
            {
#ifdef TENSORS_BOUND_CHECKS
                if( k < Int(0) || k >= val.Size() )
                {
                    eprint(this->ClassName()+"::Value(" + ToString(k) + "): Access out of bounds.");
                }
#endif
                return val[k];
            }
            
        public:
            
            static std::string MethodName( const std::string & tag )
            {
                return ClassName() + "::" + tag;
            }
            
            static std::string ClassName()
            {
                return std::string("Sparse::Vector")
                    + "<" + TypeName<Scal>
                    + "," + TypeName<Int>
                    + ">";
            }
            
        }; // Vector
        
        /*!@brief This effectively reads odd the portions of the sparse vectors `u` and `v` given by `{u_i_begin,...,u_i_end-1}` and `{ v_i_begin,...,v_i_end-1}` and computes their dot product without doing any actual copy.
         * Binary search (`FindRange`) is used to find the range in `u_idx` and `v_idx` that correspond to the indices `{u_i_begin,...,u_i_end-1}` and `{v_i_begin,...,v_i_end-1}`, so `u_idx` and `v_idx` are required to be sorted.
         */
        
        template<
            typename u_Scal, typename u_Int,
            typename v_Scal, typename v_Int,
            typename Scal = decltype( u_Scal(0) *  v_Scal(0))
        >
        static Scal VectorPartialDot(
            cptr<u_Int>  u_idx,
            cptr<u_Scal> u_val,
            const u_Int  u_n,
            const u_Int  u_nnz,
            const u_Int  u_i_begin,
            const u_Int  u_i_end,
            cptr<v_Int>  v_idx,
            cptr<v_Scal> v_val,
            const v_Int  v_n,
            const v_Int  v_nnz,
            const v_Int  v_i_begin,
            const v_Int  v_i_end
        )
        {
            Scal sum = 0;
            
            if( u_i_begin > u_i_end ) { return sum; }
            if( v_i_begin > v_i_end ) { return sum; }
            
            if( u_i_begin < u_Int(0) )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": u_i_begin < 0.");
                return sum;
            }
            if( u_i_end > u_n )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": u_i_end > u_n.");
                return sum;
            }
            
            if( v_i_begin < v_Int(0) )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": v_i_begin  < 0");
                return sum;
            }
            if( v_i_end > v_n )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": v_i_end > v_n.");
                return sum;
            }
            
            if( std::cmp_not_equal( u_i_end - u_i_begin, v_i_end - v_i_begin) )
            {
                eprint(std::string("Sparse::VectorPartialDot")+": u_i_end - u_i_begin != v_i_end - v_i_begin.");
                return sum;
            }
            
            // TODO: Check cost here if u_i_begin == 0 and u_i_end == u_n.
            auto [u_p_begin,u_p_end] = FindRange(u_idx,u_nnz,u_i_begin,u_i_end);
            
            // TODO: Check cost here if v_i_begin == 0 and v_i_end == v_n.
            auto [v_p_begin,v_p_end] = FindRange(v_idx,v_nnz,v_i_begin,v_i_end);
            
            // One would think that one could use Details::Dot from here on, but we have to apply a shift of the indices in order to test for equality.
            
            // Pointers into u_idx and v_idx.
            u_Int u_p = u_p_begin;
            v_Int v_p = v_p_begin;
            
            if ( u_p >= u_p_end ) { return sum; }
            // This automatically corrects non-0-basedness!
            u_Int u_i = u_idx[u_p] - u_i_begin;
            
            if ( v_p >= v_p_end ) { return sum; }
            // This automatically corrects non-0-basedness!
            v_Int v_i = v_idx[v_p] - v_i_begin;
            
            
            auto move_u_forward = [&u_p,&u_i,u_idx,u_i_begin,u_p_end]() -> bool
            {
                if ( ++u_p >= u_p_end ) { return false; }
                // This automatically corrects non-0-basedness!
                u_i = u_idx[u_p] - u_i_begin;
                return true;
            };
            
            auto move_v_forward = [&v_p,&v_i,v_idx,v_i_begin,v_p_end]() -> bool
            {
                if ( ++v_p >= v_p_end ) { return false; }
                // This automatically corrects non-0-basedness!
                v_i = v_idx[v_p] - v_i_begin;
                return true;
            };
            
            while( true )
            {
                while ( u_i < v_i )
                {
                    if( !move_u_forward() ) { return sum; }
                }
                // If we have not terminated, yet, then we have u_i >= v_i now.
                
                while ( v_i < u_i )
                {
                    if( !move_v_forward() ) { return sum; }
                }
                // If we have not terminated, yet, then we have v_i >= u_i now.
                
                if ( u_i == v_i )
                {
                    sum += static_cast<Scal>(u_val[u_p]) * static_cast<Scal>(v_val[v_p]);
                    
                    if( !move_u_forward() ) { return sum; }
                    if( !move_v_forward() ) { return sum; }
                }
                else
                {
                    // We actually can reach this point, i.e., if
                    // u_idx = {0,2,4} and v_idx = {1,2,5}.
                    
                    if ( u_i < v_i )
                    {
                        if( !move_u_forward() ) { return sum; }
                    }
                    else // if ( u_i > v_i )
                    {
                        if( !move_v_forward() ) { return sum; }
                    }
                }
            }
            
            return sum;
        }
        
    } // namespace Sparse
    
    
    /*!@brief This effectively reads odd the portions of the sparse vectors `u` and `v` given by `{u_i_begin,...,u_i_end-1}` and `{ v_i_begin,...,v_i_end-1}` and computes their dot product without doing any actual copy.
     */
    
    template<
        typename u_Scal, typename u_Int,
        typename v_Scal, typename v_Int,
        typename Scal = decltype( u_Scal(0) * v_Scal(0))
    >
    Scal PartialDot(
        cref<Sparse::Vector<u_Scal,u_Int>> u,
        const u_Int  u_i_begin,
        const u_Int  u_i_end,
        cref<Sparse::Vector<v_Scal,v_Int>> v,
        const v_Int  v_i_begin,
        const v_Int  v_i_end
    )
    {
        return Sparse::VectorPartialDot(
            u.Indices().data(),
            u.Values().data(),
            u.Size(), u.NonzeroCount(), u_i_begin, u_i_end,
            v.Indices().data(),
            v.Values().data(),
            v.Size(), v.NonzeroCount(), v_i_begin, v_i_end
        );
    }
    
    template<
        typename u_Scal, typename u_Int,
        typename v_Scal, typename v_Int,
        typename Scal = decltype( u_Scal(0) * v_Scal(0) )
    >
    Scal PartialDot(
        cref<Sparse::Vector<u_Scal,u_Int>> u,
        cref<Sparse::Vector<v_Scal,v_Int>> v
    )
    {
        return Sparse::template VectorPartialDot<u_Scal,u_Int,v_Scal,v_Int,Scal>(
            u.Indices().data(),
            u.Values().data(),
            u.Size(), u.NonzeroCount(), u_Int(0), u.Size(),
            v.Indices().data(),
            v.Values().data(),
            v.Size(), v.NonzeroCount(), v_Int(0), v.Size()
        );
    }
    
} // namespace Tensors
