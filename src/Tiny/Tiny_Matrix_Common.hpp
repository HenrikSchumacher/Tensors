public:

    /*!@brief Overload of `swap`.*/
    friend void swap( Class_T & X, Class_T & Y ) noexcept
    {
        // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
//        using std::swap;
//        
//        swap( X.A, Y.A );
        
//        Scal buffer [Class_T::RowCount()][Class_T::ColCount()];
//        
//        X.Write( &buffer[0][0] );
//        X.Read ( &Y[0][0]      );
//        Y.Read ( &buffer[0][0] );
//        
        const Int mn = Class_T::RowCount() * Class_T::ColCount();
        
        std::swap_ranges( &X[0][0], &X[0][0] + mn, &Y[0][0] );
    }

            
public:
    
    /*!@brief Return mutable pointer to internal raw buffer.*/
    mptr<Scal> data()
    {
        return &A[0][0];
    }
    
    /*!@brief Return immutable pointer to internal raw buffer.*/
    cptr<Scal> data() const
    {
        return &A[0][0];
    }

    /*!@brief Return mutable pointer to first element in row `i`.*/
    mptr<Scal> data( const Int i )
    {
        return &A[i][0];
    }

    /*!@brief Return immutable pointer to first element in row `i`.*/
    cptr<Scal> data( const Int i ) const
    {
        return &A[i][0];
    }
    
    /*!@brief Access entry at position `{i,j}`.*/
    mref<Scal> operator()( const Int i, const Int j )
    {
        return A[i][j];
    }
    
    /*!@brief Access entry at position `{i,j}`, read only.*/
    cref<Scal> operator()( const Int i, const Int j ) const
    {
        return A[i][j];
    }
    
    /*!@brief Return mutable pointer to first element in row `i`. This way, syntax like `A[i][j]` can be used.*/
    mptr<Scal> operator[]( const Int i )
    {
        return &A[i][0];
    }
    
    /*!@brief Return immutable pointer to first element in row `i`. This way, syntax like `A[i][j]` can be used.*/
    cptr<Scal> operator[]( const Int i ) const
    {
        return &A[i][0];
    }
