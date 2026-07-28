#pragma once

namespace Tensors
{
    namespace Tiny
    {
        /*!@brief Container for storing a list if fixed-size vectors. Designed for interoperability with `Tiny::Vector`.
         *
         * This is basically a `Tensor2` whose last dimension is a compile-time constant. Hence, the entries of the vectors are stored contiguously in the rows of the matrix.
         *
         * By making the last dimension compile-time constants, we can help the compiler to speed up the indexing operations a little. (The compiler has the discretion to use fused shift-load operations if the last dimension is a powers of 2.)
         *
         * The underlying container is a `Tensor1`.
         */
        
        template<
            int n_, typename Scal_, IntQ Int_,
            Size_T alignment = DefaultAlignment
        >
        class VectorList_AoS final
        {
            
        public:
            
            /*!@brief The type used for the entries.*/
            using Scal   = Scal_;
            /*!@brief An integral type used for indexing.*/
            using Int    = Int_;
            /*!@brief The type used for the real part of entries.*/
            using Real = typename Scalar::Real<Scal_>;
            
            static constexpr Int n    = static_cast<Int>(n_);
            static constexpr Int rank = 2;
            
            static constexpr Size_T Alignment = alignment;

            using Tensor_T = Tensor1<Scal,Int,Alignment>;
            
        public:

            
            /*!@brief Construct a container with number of vectors equal to `vector_count_`. Beware: The values in this container are not initialized!*/
            explicit VectorList_AoS( const Int vector_count_ )
//            :   a(m_,n)
            :   a            { vector_count_ * n }
            ,   vector_count { vector_count_     }
            {}
            
            /*!@brief Construct a container with number of vectors equal to `vector_count_`, then initialize all values by `init`.*/
            VectorList_AoS( const Int vector_count_, const Scal init )
//            :   a(m_,n,init)
            :   a            { vector_count_ * n, init }
            ,   vector_count { vector_count_           }
            {}
            
            /*!@brief Construct a container with number of vectors equal to `vector_count_` and read data from the buffer `a_ptr`.*/
            template<typename S>
            VectorList_AoS( cptr<S> a_ptr, const Int vector_count_ )
//            :   a(a_ptr,m_,n)
            :   a            { a_ptr, vector_count_ * n }
            ,   vector_count { vector_count_            }
            {}
    
            // Default constructor
            VectorList_AoS() = default;
            // Destructor
            ~VectorList_AoS() noexcept = default;
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
            ,   vector_count( other.vector_count )
            {}
            
            friend void swap(VectorList_AoS & A, VectorList_AoS & B ) noexcept
            {
                using std::swap;
                swap(A.a, B.a);
                swap(A.vector_count, B.vector_count);
            }
            
        protected:
            
            Tensor_T a;
            
            Int vector_count = 0;
            
        public:
            
            TOOLS_FORCE_INLINE mptr<Tensor_T> Tensor() noexcept
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE cptr<Tensor_T> Tensor() const noexcept
            {
                return a;
            }
            
            /*!@brief Return mutable pointer to first element of the first vector.*/
            TOOLS_FORCE_INLINE mptr<Scal> data() noexcept
            {
                return a.data();
            }
            
            /*!@brief Return immutable pointer to first element of the first vector.*/
            TOOLS_FORCE_INLINE cptr<Scal> data() const noexcept
            {
                return a.data();
            }
            
            /*!@brief Return mutable pointer to first element of the `i`-th vector.*/
            template<IntQ I>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i ) noexcept
            {
                return &a.data()[n * i];
            }
            /*!@brief Return immutable pointer to first element of the `i`-th vector.*/
            template<IntQ I>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const noexcept
            {
                return &a.data()[n * i];
            }
            
            /*!@brief Return mutable pointer to `j`-th element of the `i`-th vector.*/
            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j ) noexcept
            {
                return &a.data()[n * i + j];
            }
            
            /*!@brief Return immutable pointer to `j`-th element of the `i`-th vector.*/
            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i, const J j ) const noexcept
            {
                return &a.data()[n * i + j];
            }
            /*!@brief Access `j`-th element of the `i`-th vector.*/
            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE mref<Scal> operator()( const I i, const J j) noexcept
            {
                return a.data()[n * i + j];
            }
            
            /*!@brief Access `j`-th element of the `i`-th vector, read only.*/
            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE cref<Scal> operator()( const I i, const J j) const noexcept
            {
                return a.data()[n * i + j];
            }
            

            auto WriteAccess()
            {
                return [this]( const Int i, const Int j ) -> Scal&
                {
                    return a.data()[n * i + j];
                };
            }
            
            auto ReadAccess() const
            {
                return [this]( const Int i, const Int j ) -> Scal
                {
                    return a.data()[n * i + j];
                };
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
//                a.Write(i,b);
                copy_buffer<n>(this->data(i),b);
            }
            
            // row-wise Read
            template< typename S>
            void Read( const Int i, cptr<S> b )
            {
//                a.Read(i,b);
                copy_buffer<n>(b,this->data(i));
            }
            
            void SetZero()
            {
                a.SetZero();
            }
            
            void Fill( cref<Scal> init )
            {
                a.Fill(init);
            }
            
            /*!@brief Return size in dimension `i`.*/
            TOOLS_FORCE_INLINE Int Dim( const Int i ) const noexcept
            {
                if( i == Int(0) )
                {
//                    return a.Dim(0);
                    return vector_count;
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
            
            Int Size() const noexcept
            {
                return a.Size();
            }
            
            static constexpr Int Rank() noexcept
            {
                return static_cast<Int>(rank);
            }
            
//            inline friend std::ostream & operator<<( std::ostream & s, cref<VectorList_AoS> A )
//            {
//                return s << A.a;
//            }
//            
//            inline friend std::string ToString( cref<VectorList_AoS> A )
//            {
//                return ToString(A.a);
//            }
//            
//            inline friend std::string ToString( cref<VectorList_AoS> A, cref<std::string> line_prefix )
//            {
//                return ToString(A.a, line_prefix);
//            }
            
            
            
            
            inline friend std::ostream & operator<<( std::ostream & s, cref<VectorList_AoS> list )
            {
                return s << OutString::FromMatrix(list.ReadAccess(), list.Dim(0), n);
            }

            inline friend std::string ToString( cref<VectorList_AoS> list )
            {
                return OutString::FromMatrix(list.ReadAccess(), list.Dim(0), n);
            }
            
            inline friend std::string ToString( cref<VectorList_AoS> list, cref<std::string> line_prefix )
            {
                return OutString::FromArray(
                    list.ReadAccess(),
                    list.Dim(0), line_prefix + "{\n", ",\n", "\n" + line_prefix + "}",
                    n,           line_prefix + " { ", ", ", " }"
                );
            }
            
            
            Size_T AllocatedByteCount() const
            {
                return a.AllocatedByteCount();
            }
            
#ifdef LTEMPLATE_H
//            template<class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
//            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
//                cref<VectorList_AoS> A
//            )
//            {
//                return to_MTensorRef(A.a);
//            }
            
            template<bool replace_inftyQ = false, class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
                cref<VectorList_AoS> A
            )
            {
                using T = mma::Type<Scal>;
                
                mint dims [2] = {static_cast<mint>(A.vector_count),static_cast<mint>(n)};
                
                auto B = mma::makeTensor<T>( A.Rank(), &dims[0] );
                
                if constexpr ( SameQ<T,double> && replace_inftyQ )
                {
                    copy_buffer_replace_infty(A.data(),B.data(),A.Size());
                }
                else
                {
                    A.Write(B.data());
                }
                
                return B;
            }
#endif

#ifdef TENSORS_MMA_HPP
//            template<class = typename std::enable_if_t<FloatQ<Real>>>
//            inline mma::MTensorWrapper<mma::Type<Scal>> to_MTensorWrapper(
//                cref<VectorList_AoS> A
//            )
//            {
//                return to_MTensorWrapper(A.a);
//            }
            
            template<bool replace_inftyQ = false, class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::MTensorWrapper<mma::Type<Scal>> to_MTensorWrapper(
                cref<VectorList_AoS> A
            )
            {
                // TODO: Change this.
//                return to_MTensorWrapper(A.a);
                
                using T = mma::Type<Scal>;
                
                mint dims [2] = {static_cast<mint>(A.vector_count),static_cast<mint>(n)};
                
                mma::MTensorWrapper<T> B ( A.Rank(), &dims[0] );
                
                if constexpr ( SameQ<T,double> && replace_inftyQ )
                {
                    copy_buffer_replace_infty(A.data(),B.data(),A.Size());
                }
                else
                {
                    A.Write(B.data());
                }
                
                return B;
            }
#endif
            
        public:
            
            static constexpr std::string ClassName() noexcept
            {
                return std::string("VectorList_AoS")
                    + "<" + ToString(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + ToString(Alignment) + ">";
            }
            
        }; // class VectorList_AoS
        
    } // namespace Tiny
        
} // namespace Tensors
