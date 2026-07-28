#pragma once

namespace Tensors
{
    namespace Tiny
    {
        /*!@brief Container for storing a list if fixed-size matrices. Designed for interoperability with `Tiny::Matrix`.
         *
         * This is basically a `Tensor3` whose last two dimension are compile-time constants. The matrices are stored consecutively in row-major format.
         *
         * By making the last dimension compile-time constants, way we can help the compiler to speed up the indexing operations a little. (The compiler has the discretion to use fused shift-load operations if the last two dimensions are powers of 2.)
         *
         * The underlying container is a `Tensor1`.
         */
        
        template<
            int m_, int n_, typename Scal_, IntQ Int_,
            Size_T alignment = DefaultAlignment
        >
        class MatrixList_AoS final
        {
            
        public:
            
            /*!@brief The type used for the entries.*/
            using Scal   = Scal_;
            /*!@brief The integral type used for indexing.*/
            using Int    = Int_;
            /*!@brief The type used for the real part of entries.*/
            using Real = typename Scalar::Real<Scal_>;
            
            static constexpr Int n    = static_cast<Int>(n_);
            static constexpr Int m    = static_cast<Int>(m_);
            static constexpr Int mn   = m * n;
            static constexpr Int rank = 3;
            
            static constexpr Size_T Alignment = alignment;
            
//            using Tensor_T = Tensor3<Scal,Int,Alignment>;
            using Tensor_T = Tensor1<Scal,Int,Alignment>;
            
        public:
            
            /*!@brief Construct a container with number of matrices equal to `matrix_count_`. Beware: The values in this container are not initialized!*/
            explicit MatrixList_AoS( const Int matrix_count_ )
//            :   a(matrix_count_,m,n)
            :   a(matrix_count_ * m * n)
            ,   matrix_count { matrix_count_ }
            {}
            
            /*!@brief Construct a container with number of matrices equal to `matrix_count_`, then initialize all values by `init`.*/
            MatrixList_AoS( const Int matrix_count_, const Scal init )
            :   a(matrix_count_ * m * n,init)
            ,   matrix_count { matrix_count_ }
            {}
            
            /*!@brief Construct a container with number of matrices equal to `matrix_count_` and read data from the buffer `a_ptr`.*/
            MatrixList_AoS( cptr<Scal> a_ptr, const Int matrix_count_ )
            :   a(a_ptr, matrix_count_ * m * n)
            ,   matrix_count { matrix_count_ }
            {}
            
            // Default constructor
            MatrixList_AoS() = default;
            // Destructor
            ~MatrixList_AoS() noexcept = default;
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
            ,   matrix_count( other.matrix_count )
            {}
            
            
            friend void swap(MatrixList_AoS & A, MatrixList_AoS & B ) noexcept
            {
                using std::swap;
                
                swap(A.a,B.a);
                swap(A.matrix_count,B.matrix_count);
            }
            
        protected:
            
            Tensor_T a;
            
            Int matrix_count = 0;
            
        public:
            
            TOOLS_FORCE_INLINE mptr<Tensor_T> Tensor()
            {
                return a;
            }
            
            TOOLS_FORCE_INLINE cptr<Tensor_T> Tensor() const
            {
                return a;
            }
            
            
            TOOLS_FORCE_INLINE mptr<Scal> data() noexcept
            {
                return a.data();
            }

            TOOLS_FORCE_INLINE cptr<Scal> data() const noexcept
            {
                return a.data();
            }
            
            template<IntQ I>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i ) noexcept
            {
                return &a.data()[mn * i];
            }
            
            template<IntQ I>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i ) const noexcept
            {
                return &a.data()[mn * i];
            }

            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j ) noexcept
            {
                return &a.data()[mn * i + n * j];
            }
            
            template<IntQ I, IntQ J>
            TOOLS_FORCE_INLINE cptr<Scal> data( const I i, const J j ) const noexcept
            {
                return &a.data()[mn * i + m * j];
            }
            
            template<IntQ I, IntQ J, IntQ K>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j, const K k) noexcept
            {
                return &a.data()[mn * i + n * j + k];
            }
            
            template<IntQ I, IntQ J, IntQ K>
            TOOLS_FORCE_INLINE mptr<Scal> data( const I i, const J j, const K k) const noexcept
            {
                return &a.data()[mn * i + n * j + k];
            }
            
            template<IntQ I, IntQ J, IntQ K>
            TOOLS_FORCE_INLINE mref<Scal> operator()( const I i, const J j, const K k ) noexcept
            {
                return a.data()[mn * i + n * j + k];
            }
            
            template<IntQ I, IntQ J, IntQ K>
            TOOLS_FORCE_INLINE cref<Scal> operator()( const I i, const J j, const K k ) const noexcept
            {
                return a.data()[mn * i + m * j + k];
            }
            
            auto WriteAccess()
            {
                return [this]( const Int i, const Int j, const Int k ) -> Scal&
                {
                    return a.data()[mn * i + m * j + k];
                };
            }
            
            auto ReadAccess() const
            {
                return [this]( const Int i, const Int j, const Int k ) -> Scal
                {
                    return a.data()[mn * i + m * j + k];
                };
            }
            
            template<typename S>
            void Write( mptr<S> b ) const
            {
                a.Write(b);
            }
            
            template<typename S>
            void Read( cptr<S> b )
            {
                a.Read(b);
            }
            
            template<typename S>
            void Write( const Int i, mptr<S> b ) const
            {
//                a.Write(i,b);
                
                copy_buffer<mn>(this->data(i),b);
            }
            
            template<typename S>
            void Read( const Int i, cptr<S> b )
            {
//                a.Read(i,b);
                copy_buffer<mn>(b,this->data(i));
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
                    return matrix_count;
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

            Int Size() const noexcept
            {
                return a.Size();
            }
            
            static constexpr Int Rank() noexcept
            {
                return static_cast<Int>(rank);
            }

//            inline friend std::ostream & operator<<( std::ostream & s, cref<MatrixList_AoS> A )
//            {
//                return s << A.a;
//            }
//            
//            inline friend std::string ToString( cref<MatrixList_AoS> A )
//            {
//                return ToString( A.a );
//            }
//            
//            inline friend std::string ToString( cref<MatrixList_AoS> A, cref<std::string> line_prefix )
//            {
//                return ToString( A.a, line_prefix );
//            }
            
            
            inline friend std::ostream & operator<<( std::ostream & s, cref<MatrixList_AoS> list )
            {
                return s << OutString::FromCube(
                    list.ReadAccess(), list.Dim(0), m, n
                );
            }

            inline friend std::string ToString( cref<MatrixList_AoS> list )
            {
                return OutString::FromCube(list.ReadAccess(), list.Dim(0), m, n);
            }
            
            inline friend std::string ToString( cref<MatrixList_AoS> list, cref<std::string> line_prefix )
            {
                return OutString::FromArray(
                    list.ReadAccess(),
                    list.Dim(0), line_prefix + "{\n", ",\n", "\n" + line_prefix + "}",
                    m,           line_prefix + " { ", ", ", " }",
                    n,                          "{ ", ", ", " }"
                );
            }
            
            
            Size_T AllocatedByteCount() const
            {
                return a.AllocatedByteCount();
            }
            
#ifdef LTEMPLATE_H
//            template<class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
//            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
//                cref<MatrixList_AoS> A
//            )
//            {
//                // TODO: Change this.
//                return to_MTensorRef(A.a);
//            }
            
            template<bool replace_inftyQ = false, class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::TensorRef<mma::Type<Scal>> to_MTensorRef(
                cref<MatrixList_AoS> A
            )
            {
                using T = mma::Type<Scal>;
                
                mint dims [3] = {static_cast<mint>(A.matrix_count),static_cast<mint>(m),static_cast<mint>(n)};
                
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
            template<bool replace_inftyQ = false, class = typename std::enable_if_t<mma::HasTypeQ<Scal>>>
            friend mma::MTensorWrapper<mma::Type<Scal>> to_MTensorWrapper(
                cref<MatrixList_AoS> A
            )
            {
                // TODO: Change this.
//                return to_MTensorWrapper(A.a);
                
                using T = mma::Type<Scal>;
                
                mint dims [3] = {static_cast<mint>(A.matrix_count),static_cast<mint>(m),static_cast<mint>(n)};
                
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
                return std::string("MatrixList_AoS")
                    + "<" + ToString(m)
                    + "m" + ToString(n)
                    + "," + TypeName<Scal>
                    + "," + TypeName<Int>
                    + "," + ToString(Alignment) + ">";
            }
            
        }; // class MatrixList_AoS
    
    } // namespace Tiny
        
} // namespace Tensors
