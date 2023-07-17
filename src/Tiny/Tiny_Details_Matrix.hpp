public:

    // Copy constructor
    CLASS( const CLASS & other )
    {
        Read( &other.A[0][0] );
    }

    friend void swap( CLASS & X, CLASS & Y ) noexcept
    {
        // see https://stackoverflow.com/questions/5695548/public-friend-swap-member-function for details
        using std::swap;
        
        swap( X.A, Y.A );
    }


//######################################################
//##                     Access                       ##
//######################################################
            
public:
    
    mptr<Scal> data()
    {
        return &A[0][0];
    }
    
    cptr<Scal> data() const
    {
        return &A[0][0];
    }
    
    mref<Scal> operator()( const Int i, const Int j )
    {
        return A[i][j];
    }
    
    cref<Scal> operator()( const Int i, const Int j ) const
    {
        return A[i][j];
    }
    
    mptr<Scal> operator[]( const Int i )
    {
        return &A[i][0];
    }
    
    cptr<Scal> operator[]( const Int i ) const
    {
        return &A[i][0];
    }
