//######################################################
//##                     Memory                       ##
//######################################################
            
        public:
            
            explicit CLASS( const Scalar init )
            :   A {{{init}}}
            {}
            
            void SetZero()
            {
                zerofy_buffer<m*n>( &A[0][0] );
            }
            
            void Fill( const Scalar init )
            {
                fill_buffer<m*n>( &A[0][0], init );
            }
            
            template<typename T>
            void Write( T * const target ) const
            {
                copy_buffer<m*n>( &A[0][0], target );
            }
            
            template<typename T>
            void Read( T const * const source )
            {
                copy_buffer<m*n>( source, &A[0][0] );
            }


//######################################################
//##                  Arithmetic                      ##
//######################################################
         
public:
    
    friend CLASS operator+( const CLASS & x, const CLASS & y )
    {
        CLASS z;
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                z.A[i][j] = x.A[i][j] + y.A[i][j];
            }
        }
        return z;
    }

    
    template<class T>
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator+=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] += lambda;
            }
        }
        
        return *this;
    }
    

    template<class T>
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator-=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] -= lambda;
            }
        }
        
        return *this;
    }
    
    template<class T>
    std::enable_if_t<
        std::is_same_v<T,Scalar> || (ScalarTraits<Scalar>::IsComplex && std::is_same_v<T,Real>),
        CLASS &
    >
    operator*=( const T lambda )
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                A[i][j] *= lambda;
            }
        }
        
        return *this;
    }
    
    
    void Transpose( CLASS & B ) const
    {
        for( Int j = 0; j < n; ++j )
        {
            for( Int i = 0; i < m; ++i )
            {
                B.A[j][i] = A[i][j];
            }
        }
    }
    
    void ConjugateTranspose( CLASS & B ) const
    {
        for( Int j = 0; j < n; ++j )
        {
            for( Int i = 0; i < m; ++i )
            {
                B.A[j][i] = conj(A[i][j]);
            }
        }
    }
    
    void Conjugate( CLASS & B ) const
    {
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                B.A[i][j] = conj(A[i][j]);
            }
        }
    }
