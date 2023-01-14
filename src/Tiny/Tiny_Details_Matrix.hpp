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
    
    mut<Scalar> data()
    {
        return &A[0][0];
    }
    
    ptr<Scalar> data() const
    {
        return &A[0][0];
    }
    
    Scalar & operator()( const Int i, const Int j )
    {
        return A[i][j];
    }
    
    const Scalar & operator()( const Int i, const Int j ) const
    {
        return A[i][j];
    }
    
    mut<Scalar> operator[]( const Int i )
    {
        return &A[i][0];
    }
    
    ptr<Scalar> operator[]( const Int i ) const
    {
        return &A[i][0];
    }
