#pragma once

namespace Tensors
{
    namespace Tiny
    {
        // This is basically a Tensor3 whose last two dimension are a compile-time constants. This way we can help the compiler to speed up the indexing operations a little. (The compiler has the discretion to use fused shift-load operations.)
        
        template<
            int m_, int n_, typename Scal_, typename Int_,
            Size_T alignment = DefaultAlignment
        >
        class MatrixList_AoS : public Tensor3<Scal_,Int_,alignment>
        {
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            
            static constexpr Int n  = static_cast<Int>(n_);
            static constexpr Int m  = static_cast<Int>(m_);
            static constexpr Int mn = m * n;
            

            static constexpr Size_T Alignment = alignment;
            
            using Base_T = Tensor3<Scal,Int,Alignment>;
            

            
        public:
            
            MatrixList_AoS()
            :   Base_T()
            {}
            
            explicit MatrixList_AoS( const Scal k_ )
            :   Base_T(k_,m,n)
            {}
            
            MatrixList_AoS( const Int k_, const Scal init )
            :   Base_T(k_,m,n,init)
            {}
            
            ~MatrixList_AoS() = default;
            
            // Copy constructor
            MatrixList_AoS(const MatrixList_AoS & other ) = default;
            
            // Copy assignment
            MatrixList_AoS & operator=(MatrixList_AoS other) noexcept
            {
                swap(*this, other);
                return *this;
            }
            
            // Move constructor
            MatrixList_AoS( MatrixList_AoS && other) noexcept
            :   MatrixList_AoS()
            {
                swap(*this, other);
            }
            
            // Move assignment
            MatrixList_AoS & operator=(MatrixList_AoS && other) noexcept
            {
                swap(*this, other);
                return *this;
            }
            
            friend void swap(MatrixList_AoS & A, MatrixList_AoS & B ) noexcept
            {
                using std::swap;
                
                swap(static_cast<Base_T & >(A), static_cast<Base_T &>(B));
            }
            
        protected:
            
            using Base_T::a;
            
        public:
            
            TOOLS_FORCE_INLINE mptr<Int> data()
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE cptr<Int> data() const
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE mptr<Int> data( const Int i )
            {
                return &a[mn * i];
            }
            
            TOOLS_FORCE_INLINE cptr<Int> data( const Int i ) const
            {
                return &a[mn * i];
            }

            TOOLS_FORCE_INLINE mptr<Int> data( const Int i, const bool j )
            {
                return &a[mn * i + n * j];
            }
            
            TOOLS_FORCE_INLINE cptr<Int> data( const Int i, const bool j ) const
            {
                return &a[mn * i + m * j];
            }
            
            TOOLS_FORCE_INLINE mptr<Int> data( const Int i, const bool j, const bool k)
            {
                return &a[mn * i + n * j + k];
            }
            
            TOOLS_FORCE_INLINE mptr<Int> data( const Int i, const bool j, const bool k) const
            {
                return &a[mn * i + n * j + k];
            }
            
            TOOLS_FORCE_INLINE mref<Int> operator()( const Int i, const bool j, const bool k)
            {
                return a[mn * i + n * j + k];
            }
                
            TOOLS_FORCE_INLINE cref<Int> operator()( const Int i, const bool j, const bool k) const
            {
                return a[mn * i + m * j + k];
            }
            
            
        public:
            
            static std::string ClassName()
            {
                return ct_string("MatrixList_AoS")
                    + "<" + to_ct_string(m)
                    + "m" + to_ct_string(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + to_ct_string(Alignment) + ">";
            }
            
        }; // class MatrixList_AoS
    
    } // namespace Tiny
        
} // namespace Tensors
