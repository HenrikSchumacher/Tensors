#pragma once


#define CLASS SmallSymmetricMatrix

namespace CycleSampler {
        
    template< int AmbDim, typename Real, typename Int>
    struct CLASS
    {
    public:
        
        using Vector_T = SmallVector<AmbDim,Real,Int>;
        
        static constexpr Real zero              = 0;
        static constexpr Real half              = 0.5;
        static constexpr Real one               = 1;
        static constexpr Real two               = 2;
        static constexpr Real three             = 3;
        static constexpr Real four              = 4;
        static constexpr Real eps               = std::numeric_limits<Real>::min();
        static constexpr Real infty             = std::numeric_limits<Real>::max();
        
        // Uses only upper triangle.
        
        Real A [AmbDim][AmbDim] = {};
        
         CLASS() = default;
       
        ~CLASS() = default;
        
        explicit CLASS( const Real init )
        {
            Fill(init);
        }
        
        // Copy constructor
        CLASS( const CLASS & other )
        {
            Read( &other.A[0][0] );
        }
        
        force_inline Real * data()
        {
            return &A[0][0];
        }
        
        force_inline const Real * data() const
        {
            return &A[0][0];
        }
        
        force_inline void SetZero()
        {
            zerofy_buffer( &A[0][0], AmbDim * AmbDim );
        }
        
        force_inline void Fill( const Real init )
        {
            fill_buffer( &A[0][0], init, AmbDim * AmbDim );
        }
        
        force_inline Real & operator()( Int i, Int j )
        {
            return A[i][j];
        }
        
        force_inline const Real & operator()( const Int i, const Int j ) const
        {
            return A[i][j];
        }
        
        friend CLASS operator+( const CLASS & x, const CLASS & y )
        {
            CLASS z;
            for( Int i = 0; i < AmbDim; ++i )
            {
                for( Int j = i; j < AmbDim; ++j )
                {
                    z.A[i][j] = x.A[i][j] + y.A[i][j];
                }
            }
            return z;
        }
        
        void operator+=( const CLASS & B )
        {
            for( Int i = 0; i < AmbDim; ++i )
            {
                for( Int j = i; j < AmbDim; ++j )
                {
                    A[i][j] += B.A[i][j];
                }
            }
        }
        
        void operator*=( const CLASS & B )
        {
            for( Int i = 0; i < AmbDim; ++i )
            {
                for( Int j = i; j < AmbDim; ++j )
                {
                    A[i][j] *= B.A[i][j];
                }
            }
        }
        
        CLASS & operator=( const CLASS & B )
        {
            Read(&B.A[0][0]);
            
            return *this;
        }
        
        void Dot( const Vector_T & x, Vector_T & y ) const
        {
            for( Int i = 0; i < AmbDim; ++i )
            {
                Real y_i = 0;
                for( Int j = 0; j < i; ++j )
                {
                    y_i += A[j][i] * x[j];
                }
                for( Int j = i; j < AmbDim; ++j )
                {
                    y_i += A[i][j] * x[j];
                }
                
                y[i] = y_i;
            }
        }
        
        Real InnerProduct( const Vector_T & x, const Vector_T & y ) const
        {
            Real result = 0;
            for( Int i = 0; i < AmbDim; ++i )
            {
                Real z_i = 0;
                for( Int j = 0; j < i; ++j )
                {
                    z_i += A[j][i] * x[j];
                }
                for( Int j = i; j < AmbDim; ++j )
                {
                    z_i += A[i][j] * x[j];
                }
                
                result += y[i] * z_i;
                
            }
            
            return result;
        }
        
        void Cholesky()
        {
            for( Int k = 0; k < AmbDim; ++k )
            {
                const Real a = A[k][k] = sqrt(A[k][k]);
                const Real ainv = one/a;

                for( Int j = k+1; j < AmbDim; ++j )
                {
                    A[k][j] *= ainv;
                }

                for( Int i = k+1; i < AmbDim; ++i )
                {
                    for( Int j = i; j < AmbDim; ++j )
                    {
                        A[i][j] -= A[k][i] * A[k][j];
                    }
                }
            }
        }
        
        
        void CholeskySolve( const Vector_T & b, Vector_T & x ) const
        {
            x = b;
            CholeskySolve(x);
        }
        
        void CholeskySolve( Vector_T & x ) const
        {
            //In-place solve.
            
            // Lower triangular back substitution
            for( Int i = 0; i < AmbDim; ++i )
            {
                for( Int j = 0; j < i; ++j )
                {
                    x[i] -= A[j][i] * x[j];
                }
                x[i] /= A[i][i];
            }
            
            // Upper triangular back substitution
            for( Int i = AmbDim-1; i > -1; --i )
            {
                for( Int j = i+1; j < AmbDim; ++j )
                {
                    x[i] -= A[i][j] * x[j];
                }
                x[i] /= A[i][i];
            }
        }
        
        Real SmallestEigenvalue() const
        {
            if( AmbDim == 2)
            {
                Real lambda_min = half * (
                    A[0][0] + A[1][1]
                    - std::sqrt(
                        std::abs(
                            (A[0][0]-A[1][1])*(A[0][0]-A[1][1]) + four * A[0][1]*A[0][1]
                        )
                    )
                );
                
                return lambda_min;
            }
                    
            if( AmbDim == 3)
            {
                Real lambda_min;
                
                const Real p1 = A[0][1]*A[0][1] + A[0][2]*A[0][2] + A[1][2]*A[1][2];
                
                if( std::sqrt(p1) < eps * std::sqrt( A[0][0]*A[0][0] + A[1][1]*A[1][1] + A[2][2]*A[2][2]) )
                {
                    // A is diagonal
                    lambda_min = std::min( A[0][0], std::min(A[1][1],A[2][2]) );
                }
                else
                {
                    const Real q         = ( A[0][0] + A[1][1] + A[2][2] ) / three;
                    const Real delta [3] = { A[0][0]-q, A[1][1]-q, A[2][2]-q } ;
                    const Real p2   = delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2] + two*p1;
                    const Real p    = std::sqrt( p2 / static_cast<Real>(6) );
                    const Real pinv = one/p;
                    const Real b11  = delta[0] * pinv;
                    const Real b22  = delta[1] * pinv;
                    const Real b33  = delta[2] * pinv;
                    const Real b12  = A[0][1] * pinv;
                    const Real b13  = A[0][2] * pinv;
                    const Real b23  = A[1][2] * pinv;
                    
                    const Real r = half * (two * b12 * b23 * b13 - b11 * b23 * b23 - b12 *b12 * b33 + b11 * b22 * b33 - b13 *b13 * b22);
                    
                    
                    const Real phi = ( r <= -one )
                        ? ( static_cast<Real>(M_PI) / three )
                        : ( ( r >= one ) ? zero : acos(r) / three );
                    
                    // The eigenvalues are ordered this way: eig2 <= eig1 <= eig0.

//                    Real eig0 = q + two * p * cos( phi );
//                    Real eig2 = q + two * p * cos( phi + two * M_PI/ three );
//                    Real eig1 = three * q - eig0 - eig2;
                       
                    lambda_min = q + two * p * cos( phi + two * M_PI/ three );
                }
        
                return lambda_min;
            }
                    
            using Matrix_T = Eigen::Matrix<Real,AmbDim,AmbDim>;

            Matrix_T Sigma (&A[0][0]);
            
            Eigen::SelfAdjointEigenSolver<Matrix_T> eigs;
            
            eigs.compute(Sigma);

            return eigs.eigenvalues()[0];
        }
        
        force_inline void Write( Real * target ) const
        {
            copy_buffer( &A[0][0], target, AmbDim * AmbDim );
        }
        
        force_inline void Read( Real const * const source )
        {
            copy_buffer(source, &A[0][0], AmbDim * AmbDim);
        }
        
        std::string ToString( const Int n = 16) const
        {
            std::stringstream sout;
            sout.precision(n);
            sout << "{\n";
            sout << "\t{ ";
            
            sout << A[0][0];
            for( Int j = 1; j < AmbDim; ++j )
            {
                sout << ", " << A[0][j];
            }
            
            for( Int i = 1; i < AmbDim; ++i )
            {
                sout << " },\n\t{ ";
                
                sout << A[i][0];
                
                for( Int j = 1; j < AmbDim; ++j )
                {
                    sout << ", " << A[i][j];
                }
            }
            sout << " }\n}";
            return sout.str();
        }
        
//            Real Det() const
//            {
//                if( AmbDim == 2 )
//                {
//                    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
//                }
//                
//                if( AmbDim == 3 )
//                {
//                    return (
//                          A[0][0]*A[1][1]*A[2][2] + A[0][1]*A[1][2]*A[2][0] + A[0][2]*A[1][0]*A[2][1]
//                        - A[0][0]*A[1][2]*A[2][1] - A[0][1]*A[1][0]*A[2][2] - A[0][2]*A[1][1]*A[2][0]
//                    );
//                }
//                
//                // Bareiss algorithm copied and adapted from https://cs.stackexchange.com/q/124759/146040
//                
//                SmallSquareMatrix<AmbDim,Real,Int> M;
//                
//                M.Read(&A[0][0]);
//                
//                Real sign = one;
//
//                for(Int k = 0; k < AmbDim - 1; ++k )
//                {
//                    //Pivot - row swap needed
//                    if( M(k,k) == zero )
//                    {
//                        Int m = 0;
//                        for( m = k + 1; m < AmbDim; ++m )
//                        {
//                            if( M(m,k) != zero )
//                            {
//                                std::swap_ranges( &M(m,0), &M(m,AmbDim), &M(k,0) );
//                                sign = -sign;
//                                break;
//                            }
//                        }
//
//                        //No entries != 0 found in column k -> det = 0
//                        if(m == AmbDim) {
//                            return zero;
//                        }
//                    }
//
//                    //Apply formula
//                    for( Int i = k + 1; i < AmbDim; ++i )
//                    {
//                        for( Int j = k + 1; j < AmbDim; ++j )
//                        {
//                            M(i,j) = M(k,k) * M(i,j) - M(i,k) * M(k,j);
//                            if(k != 0)
//                            {
//                                M(i,j) /= M(k-1,k-1);
//                            }
//                        }
//                    }
//                }
//
//                return sign * M(AmbDim-1,AmbDim-1);
//            }
        
//            #pragma omp declare reduction( + : CLASS : omp_out+=omp_in )
        
    public:
        
        static constexpr Int AmbientDimension()
        {
            return AmbDim;
        }
        
        static std::string ClassName()
        {
            return "CLASS<"+std::to_string(AmbDim)+","+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    };
        
} // namespace CycleSampler

#undef CLASS
