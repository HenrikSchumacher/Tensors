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
        class MatrixList_AoS final
        {
            
        public:
            
            using Scal = Scal_;
            using Int  = Int_;
            
            static constexpr Int n  = static_cast<Int>(n_);
            static constexpr Int m  = static_cast<Int>(m_);
            static constexpr Int mn = m * n;

            static constexpr Size_T Alignment = alignment;
            
            using Tensor_T = Tensor3<Scal,Int,Alignment>;
            
        public:
            
            explicit MatrixList_AoS( const Int k_ )
            :   a(k_,m,n)
            {}
            
            MatrixList_AoS( const Int k_, const Scal init )
            :   a(k_,m,n,init)
            {}
            
            MatrixList_AoS( cptr<Scal> a_, const Int k_ )
            :   a(a_,k_,m,n)
            {}
            
            // Default constructor
            MatrixList_AoS() = default;
            // Destructor
            ~MatrixList_AoS() = default;
            // Copy constructor
            MatrixList_AoS( const MatrixList_AoS & other ) = default;
            // Copy assignment operator
            MatrixList_AoS & operator=( const MatrixList_AoS & other ) = default;
            // Move constructor
            MatrixList_AoS( MatrixList_AoS && other ) = default;
            // Move assignment operator
            MatrixList_AoS & operator=( MatrixList_AoS && other ) = default;

            // Copy-cast constructor
            template<typename T, typename I, Size_T align>
            MatrixList_AoS( const MatrixList_AoS<m,n,T,I,align> & other )
            :   a( other.a )
            {}
            
            
            friend void swap(MatrixList_AoS & A, MatrixList_AoS & B ) noexcept
            {
                using std::swap;
                
                swap(A.a,B.a);
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
                return &a.data()[mn * i];
            }
            
            template<typename I>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const
            {
                static_assert(IntQ<I>,"");
                return &a.data()[mn * i];
            }

            template<typename I, typename J>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j )
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a.data()[mn * i + n * j];
            }
            
            template<typename I, typename J>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i, const J j ) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                return &a.data()[mn * i + m * j];
            }
            
            template<typename I, typename J, typename K>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j, const K k)
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                static_assert(IntQ<K>,"");
                return &a.data()[mn * i + n * j + k];
            }
            
            template<typename I, typename J, typename K>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j, const K k) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                static_assert(IntQ<K>,"");
                return &a.data()[mn * i + n * j + k];
            }
            
            template<typename I, typename J, typename K>
            TOOLS_FORCE_INLINE mref<Scal> operator()( const I i, const J j, const K k)
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                static_assert(IntQ<K>,"");
                return a.data()[mn * i + n * j + k];
            }
            
            template<typename I, typename J, typename K>
            TOOLS_FORCE_INLINE cref<Scal> operator()( const I i, const J j, const K k) const
            {
                static_assert(IntQ<I>,"");
                static_assert(IntQ<J>,"");
                static_assert(IntQ<K>,"");
                return a.data()[mn * i + m * j + k];
            }
            
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
            
            template< typename S>
            void Write( const Int i, mptr<S> b ) const
            {
                a.Write(i,b);
            }
            
            template< typename S>
            void Read( const Int i, cptr<S> b )
            {
                a.Read(i,b);
            }

            TOOLS_FORCE_INLINE Int Dim( const Int i ) const
            {
                if( i == Int(0) )
                {
                    return a.Dim(0);
                }
                else if( i == Int(1) )
                {
                    return m;
                }
                else if( i == Int(2) )
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
                
                MatrixList_AoS b (new_size);
                
                if constexpr ( copy )
                {
                    if( new_size <= b.Dim(0) )
                    {
                        b.a.ReadParallel(a,thread_count);
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
                cref<MatrixList_AoS> A, std::string line_prefix = ""
            )
            {
                return ToString(A.a,line_prefix);
            }

            template<typename F>
            inline friend std::string ToString(
                cref<MatrixList_AoS> A, F && fun, std::string line_prefix = ""
            )
            {
                return ToString(A.a,fun,line_prefix);
            }
            
#ifdef LTEMPLATE_H
            template<class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
                cref<MatrixList_AoS> A
            )
            {
                return to_MTensorRef(A.a);
            }
#endif

#ifdef MMA_HPP
            template<class = typename std::enable_if_t<FloatQ<Real>>>
            inline mma::MTensorWrapper<mma::Type<Scal>> to_MTensorWrapper(
                cref<MatrixList_AoS> A
            )
            {
                return to_MTensorWrapper(A.a);
            }
#endif
            
            
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
