//######################################################
//##                     Access                       ##
//######################################################
            
public:
    
    Scalar * restrict data()
    {
        return &A[0][0];
    }
    
    const Scalar * restrict data() const
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
    
    Scalar * operator[]( const Int i )
    {
        return &A[i][0];
    }
    
    const Scalar * operator[]( const Int i ) const
    {
        return &A[i][0];
    }
