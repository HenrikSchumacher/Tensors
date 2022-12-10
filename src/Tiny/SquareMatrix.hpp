#pragma once

namespace Tensors
{
    namespace Tiny
    {
        template< int n_, typename Scalar_, typename Int_>
        struct SquareMatrix : public Tensors::Tiny::Matrix<n_,n_,Scalar_,Int_>
        {
        private:
            
            using Base_T = Tensors::Tiny::Matrix<n_,n_,Scalar_,Int_>;
            
        public:
            
            using Scalar   = Scalar_;
            using Real     = typename ScalarTraits<Scalar_>::RealType;
            using Int      = Int_;
            
            using Base_T::n;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
            using Base_T::zero;
            using Base_T::half;
            using Base_T::one;
            using Base_T::two;
            using Base_T::three;
            using Base_T::four;
            using Base_T::eps;
            using Base_T::infty;
            using Base_T::Write;
            using Base_T::Read;
            using Base_T::operator+=;
            using Base_T::operator-=;
            using Base_T::operator*=;
            using Base_T::operator/=;
            
            
            SquareMatrix() = default;
            
            ~SquareMatrix() = default;
            
            SquareMatrix(std::nullptr_t) = delete;
            
            explicit SquareMatrix( const Scalar init )
            :   Base_T(init)
            {}
            
            explicit SquareMatrix( const Scalar * a )
            {
                Read(a);
            }
            
            // Copy constructor
            explicit SquareMatrix( const SquareMatrix & other )
            :   Base_T(other)
            {}
            
            friend void swap( SquareMatrix & A, SquareMatrix & B ) noexcept
            {
                // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
                using std::swap;
                
                swap(A.A, B.A);
            }
            
            // Copy assignment operator
            SquareMatrix & operator=( SquareMatrix other )
            {
                // copy-and-swap idiom
                // see https://stackoverflow.com/a/3279550/8248900 for details
                swap(*this, other);

                return *this;
            }

            /* Move constructor */
            SquareMatrix( SquareMatrix && other ) noexcept
            :   SquareMatrix()
            {
                swap(*this, other);
            }
            
        protected:
            
            using Base_T::A;
            
        public:
           
            void SetIdentity()
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? static_cast<Scalar>(1) : static_cast<Scalar>(0);
                    }
                }
            }
            
            void SetDiagonal( const Tensors::Tiny::Vector<n,Scalar,Int> & v )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : static_cast<Scalar>(0);
                    }
                }
            }
            
            Scalar Det() const
            {
                if constexpr ( n == 1 )
                {
                    return A[0][0];
                }
                
                if constexpr ( n == 2 )
                {
                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
                }
                
                if constexpr ( n == 3 )
                {
                    return (
                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
                    );
                }
                
                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
                
                if constexpr ( n > 3 )
                {
                    Scalar M [n][n];
                    
                    Write( &M[0][0] );
                    
                    Scalar sign (one);
                    
                    for(Int k = 0; k < n - 1; ++k )
                    {
                        //Pivot - row swap needed
                        if( M[k][k] == zero )
                        {
                            Int m = 0;
                            for( m = k + 1; m < n; ++m )
                            {
                                if( M[m][k] != zero )
                                {
                                    std::swap_ranges( &M[m][0], &M[m][n], &M[k][0] );
                                    sign = -sign;
                                    break;
                                }
                            }
                            
                            //No entries != 0 found in column k -> det = 0
                            if(m == n)
                            {
                                return zero;
                            }
                        }
                        
                        //Apply formula
                        for( Int i = k + 1; i < n; ++i )
                        {
                            for( Int j = k + 1; j < n; ++j )
                            {
                                M[i][j] = M[k][k] * M[i][j] - M[i][k] * M[k][j];
                                if(k != 0)
                                {
                                    M[i][j] /= M[k-1][k-1];
                                }
                            }
                        }
                    }
                    
                    return sign * M[n-1][n-1];
                }
                
                return static_cast<Scalar>(0);
            }
            
            
        public:
            
            static constexpr Int AmbientDimension()
            {
                return n;
            }
            
            static std::string ClassName()
            {
                return "SquareMatrix<"+std::to_string(n)+","+TypeName<Scalar>::Get()+","+TypeName<Int>::Get()+">";
            }
            
        };
        
    } // namespace Tiny
        
} // namespace Tensors
