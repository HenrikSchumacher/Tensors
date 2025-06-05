#pragma once

namespace Tensors
{
    template<typename T_, typename Int_>
    class RaggedList
    {
        static_assert(IntQ<Int_>,"");
    public:
        
        using T   = T_;
        using Int = Int_;
        
        RaggedList()
        : ptrs(Int(1))
        {
            ptrs.Push(Int(0));
        }
        
        RaggedList(
            const Int expected_sublist_count,
            const Int expected_element_t_count
        )
        : ptrs(expected_sublist_count + Int(1))
        , vals(expected_element_t_count)
        {
            ptrs.Push(Int(0));
        }
        
        ~RaggedList() = default;
        
        class Sublist_T
        {
            
        private:
            
            T * sublist_begin = nullptr;
            T * sublist_end   = nullptr;
            
        public:
            
            mptr<T> begin()
            {
                return sublist_begin;
            }
            
            cptr<T> begin() const
            {
                return sublist_begin;
            }
            
            mptr<T> end()
            {
                return sublist_end;
            }
            
            cptr<T> end() const
            {
                return sublist_end;
            }
            
            Int Size() const
            {
                return std::distance( sublist_begin, sublist_end );
            }
            
            template<typename S>
            void Write( mptr<S> target ) const
            {
                if constexpr ( SameQ<T,S> )
                {
                    std::copy( sublist_begin, sublist_end, target );
                }
                else
                {
                    std::transform( sublist_begin, sublist_end, target, static_caster<T,S>() );
                }
            }
            
            template<typename S>
            void Read( mptr<S> target )
            {
                if constexpr ( SameQ<T,S> )
                {
                    std::copy( target, &target[Size()], sublist_begin );
                }
                else
                {
                    std::transform( target, &target[Size()], sublist_begin, static_caster<S,T>() );
                }
            }
        };
        
    private:
        
        Aggregator<Int,Int> ptrs;
        Aggregator<T,Int>   vals;
        
    public:
        
//        mref<Aggregator<Int,Int>> Pointers()
//        {
//            return ptrs;
//        }
        
        cref<Aggregator<Int,Int>> Pointers() const
        {
            return ptrs;
        }
        
//        mref<Aggregator<T,Int>> Elements()
//        {
//            return vals;
//        }
        
        cref<Aggregator<T,Int>> Elements() const
        {
            return vals;
        }
        
        void FinishSublist()
        {
            ptrs.Push(vals.Size());
        }
        
        void Push( const T x )
        {
            vals.Push(x);
        }
        
        Int SublistCount() const
        {
            return ptrs.Size();
        }
        
        Int ElementCount() const
        {
            return vals.Size();
        }
        
        Int SublistSize( const Int i ) const
        {
            if( (Int(0) <= i) && (i < SublistCount() ) )
            {
                return ptrs[i+1] - ptrs[i];
            }
            else if ( i == SublistCount() )
            {
                return vals.Size() - ptrs[i];
            }
            else
            {
                return Int(0);
            }
        }
        
        Sublist_T Sublist( const Int i )
        {
            if( (Int(0) <= i) && (i < SublistCount() ) )
            {
                return Sublist_T( { &vals[ptrs[i]], &vals[ptrs[i+1]] } );
            }
            else if ( i == SublistCount() )
            {
                return Sublist_T( { &vals[ptrs[i]], &vals[vals.Size()] } );
            }
            else
            {
                return Sublist_T( { &vals[0], &vals[0] } );
            }
        }
        
    public:
        
        std::string ClassName() const
        {
            return std::string("RaggedList")
                + "<" + TypeName<T>
                + "," + TypeName<Int>
                + ">";
        }
        
    }; // class RaggedList
    
    
#ifdef LTEMPLATE_H
        
    template<
        typename T, typename Int,
        class = typename std::enable_if_t<mma::HasTypeQ<T>>
    >
    inline mma::TensorRef<mma::Type<T>> to_MTensorRef(
        RaggedList<T,Int>::Sublist & sublist
    )
    {
        mint d = sublist.Size();
        auto A = mma::makeTensor<mma::Type<T>>( mint(1), &d );
        sublist.Write( A.data() );
        
        return A;
    }
    
#endif
    
} // namespace Tensors
