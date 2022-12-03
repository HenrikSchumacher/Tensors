#pragma once

namespace Tensors
{
    namespace Small
    {
        template< int n_, typename Scalar_, typename Int_>
        struct SquareMatrix : public Tensors::Small::Matrix<n_,n_,Scalar_,Int_>
        {
        public:
            
            using Matrix_T = Tensors::Small::Matrix<n_,n_,Scalar_,Int_>;
            using Scalar   = Scalar_;
            using Real     = typename ScalarTraits<Scalar_>::RealType;
            using Int      = Int_;
            
            using Matrix_T::n;
            
            using Vector_T = Vector<n,Scalar,Int>;
            
            using Matrix_T::zero;
            using Matrix_T::half;
            using Matrix_T::one;
            using Matrix_T::two;
            using Matrix_T::three;
            using Matrix_T::four;
            using Matrix_T::eps;
            using Matrix_T::infty;
            
            // Uses only upper triangle.
            
            SquareMatrix() = default;
            
            ~SquareMatrix() = default;
            
            explicit SquareMatrix( const Scalar init )
            :   Matrix_T(init)
            {}
            
            SquareMatrix( const SquareMatrix & other )
            :   Matrix_T(other)
            {}
            
        protected:
            
            using Matrix_T::A;
            
        public:
           
            void SetIdentity()
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? one : zero;
                    }
                }
            }
            
            void SetDiagonal( const Tensors::Small::Vector<n,Scalar,Int> & v )
            {
                for( Int i = 0; i < n; ++i )
                {
                    for( Int j = 0; j < n; ++j )
                    {
                        A[i][j] = (i==j) ? v[i] : zero;
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
                    
                    Scalar sign = one;
                    
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
                
                return zero;
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
        
    } // namespace Small
        
} // namespace Tensors
