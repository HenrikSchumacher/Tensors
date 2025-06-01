#pragma once

namespace Tensors
{
    namespace Tiny
    {
        // This is basically a Tensor2 whose last dimension is a compile-time constant. This way we can help the compiler to speed up the indexing operations a little. (The compiler has the discretion to use fused shift-load operations.)
        
        template<
            int n_, typename Scal_, typename Int_,
            Size_T alignment = DefaultAlignment
        >
        class VectorList_AoS : public Tensor2<Scal_,Int_,alignment>
        {
            
        public:
            
            using Base_T = Tensor2<Scal_,Int_,alignment>;
            using Scal   = Base_T::Scal;
            using Int    = Base_T::Int;
            
            static constexpr Int n = static_cast<Int>(n_);
            
            static constexpr Size_T Alignment = Base_T::Alignment;
            
            
        public:
            
            VectorList_AoS()
            :   Base_T()
            {}
            
            explicit VectorList_AoS( const Int m_ )
            :   Base_T(m_,n)
            {}
            
            VectorList_AoS( const Int m_, const Scal init )
            :   Base_T(m_,n,init)
            {}
            
            VectorList_AoS( cptr<Scal> a, const Int m_ )
            :   Base_T(a,m_,n)
            {}
            
            ~VectorList_AoS() = default;
            
            // Copy constructor
            VectorList_AoS(const VectorList_AoS & other ) = default;
            
            
            friend void swap(VectorList_AoS & A, VectorList_AoS & B ) noexcept
            {
                using std::swap;
                
                swap(static_cast<Base_T &>(A), static_cast<Base_T &>(B));
            }
            
            // Copy assignment
            VectorList_AoS & operator=( VectorList_AoS other) noexcept
            {
                swap(*this, other);
                return *this;
            }
            
            // Move constructor
            VectorList_AoS( VectorList_AoS && other) noexcept
            :   VectorList_AoS()
            {
                swap(*this, other);
            }
            
//            // Move assignment
//            VectorList_AoS & operator=(VectorList_AoS && other) noexcept
//            {
//                swap(*this, other);
//                return *this;
//            }
            
        protected:
            
            using Base_T::a;
            
        public:
            
            TOOLS_FORCE_INLINE mptr<Scal> data()
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE cptr<Scal> data() const
            {
                return a;
            }
            
            template<typename I>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i )
            {
                static_assert(IntQ<I>,"");
                return &a[n * i];
            }
            
            template<typename I>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const
            {
                static_assert(IntQ<I>,"");
                return &a[n * i];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j )
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE cptr<Scal> data( const Int i, const J j ) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE mref<Scal> operator()( const I i, const J j)
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return a[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE cref<Scal> operator()( const I i, const J j) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return a[n * i + j];
            }
            
            
        public:
            
            static std::string ClassName()
            {
                return ct_string("VectorList_AoS")
                    + "<" + to_ct_string(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + to_ct_string(Alignment) + ">";
            }
            
        }; // class VectorList_AoS
    
    } // namespace Tiny
        
} // namespace Tensors
