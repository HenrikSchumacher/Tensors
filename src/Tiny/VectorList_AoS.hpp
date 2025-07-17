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
        class VectorList_AoS final
        {
            
        public:
            
            using Scal   = Scal_;
            using Int    = Int_;
            
            static constexpr Int n = static_cast<Int>(n_);
            
            static constexpr Size_T Alignment = alignment;
            
            using Tensor_T = Tensor2<Scal,Int,Alignment>;
            
        public:

            
            explicit VectorList_AoS( const Int m_ )
            :   a(m_,n)
            {}
            
            VectorList_AoS( const Int m_, const Scal init )
            :   a(m_,n,init)
            {}
            
            template<typename S>
            VectorList_AoS( cptr<S> a_ptr, const Int m_ )
            :   a(a_ptr,m_,n)
            {}
    
            // Default constructor
            VectorList_AoS() = default;
            // Destructor
            ~VectorList_AoS() = default;
            // Copy constructor
            VectorList_AoS( const VectorList_AoS & other ) = default;
            // Copy assignment operator
            VectorList_AoS & operator=( const VectorList_AoS & other ) = default;
            // Move constructor
            VectorList_AoS( VectorList_AoS && other ) = default;
            // Move assignment operator
            VectorList_AoS & operator=( VectorList_AoS && other ) = default;
            
            // Copy-cast constructor
            template<typename T, typename I, Size_T align>
            VectorList_AoS( const VectorList_AoS<n,T,I,align> & other )
            :   a( other.a )
            {}
            
            friend void swap(VectorList_AoS & A, VectorList_AoS & B ) noexcept
            {
                using std::swap;
                swap(A.a, B.a);
            }
            
        protected:
            
            Tensor_T a;
            
        public:
            
            TOOLS_FORCE_INLINE mptr<Tensor_T> Tensor()
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE cptr<Tensor_T> Tensor() const
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE mptr<Scal> data()
            {
                return a.data();
            }
            
            TOOLS_FORCE_INLINE cptr<Scal> data() const
            {
                return a.data();
            }
            
            template<typename I>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i )
            {
                static_assert(IntQ<I>,"");
                return &a.data()[n * i];
            }
            
            template<typename I>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const
            {
                static_assert(IntQ<I>,"");
                return &a.data()[n * i];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j )
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a.data()[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE cptr<Scal> data( const Int i, const J j ) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a.data()[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE mref<Scal> operator()( const I i, const J j)
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return a.data()[n * i + j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE cref<Scal> operator()( const I i, const J j) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return a.data()[n * i + j];
            }
            
//            template< bool copyQ>
//            void Resize( const Int d_0_, const Int d_1_, const Int thread_count = 1 ) = delete;
//            
//            template< bool copyQ>
//            void RequireSize( const Int d_0, const Int d_1, const Int thread_count = 1 ) = delete;

            template< typename S>
            void Write( mptr<S> b ) const
            {
                a.Write(b);
            }
            
            template< typename S>
            void Read( cptr<S> b )
            {
                a.Read(b);
            }
            
            // row-wise Write
            template< typename S>
            void Write( const Int i, mptr<S> b ) const
            {
                a.Write(i,b);
            }
            
            // row-wise Read
            template< typename S>
            void Read( const Int i, cptr<S> b )
            {
                a.Read(i,b);
            }
            
            void SetZero()
            {
                a.SetZero();
            }
            
            void Fill( cref<Scal> init )
            {
                a.Fill(init);
            }
            
            TOOLS_FORCE_INLINE Int Dim( const Int i ) const
            {
                if( i == Int(0) )
                {
                    return a.Dim(0);
                }
                else if( i == Int(1) )
                {
                    return n;
                }
                else
                {
                    return Int(0);
                }
            }
            
            template<bool copy>
            void Resize( const Int new_size_, const Int thread_count = 1 )
            {
                const Int new_size = Ramp(new_size_);
                
                VectorList_AoS b (new_size);
                
                if constexpr ( copy )
                {
                    if( new_size <= b.Dim(0) )
                    {
                        b.a.ReadParallel(a.data(),thread_count);
                    }
                    else
                    {
                        a.WriteParallel(b.a.data(),thread_count);
                    }
                }
                
                swap( *this, b );
            }
            
            Size_T AllocatedByteCount() const
            {
                return a.AllocatedByteCount();
            }
            
            inline friend std::string ToString(
                cref<VectorList_AoS> A, std::string line_prefix = ""
            )
            {
                return ToString(A.a,line_prefix);
            }

            template<typename F>
            inline friend std::string ToString(
                cref<VectorList_AoS> A, F && fun, std::string line_prefix = ""
            )
            {
                return ToString(A.a,fun,line_prefix);
            }
            
#ifdef LTEMPLATE_H
            template<class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
                cref<VectorList_AoS> A
            )
            {
                return to_MTensorRef(A.a);
            }
#endif

#ifdef MMA_HPP
            template<class = typename std::enable_if_t<FloatQ<Real>>>
            inline mma::MTensorWrapper<mma::Type<Scal>> to_MTensorWrapper(
                cref<VectorList_AoS> A
            )
            {
                return to_MTensorWrapper(A.a);
            }
#endif
            
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
