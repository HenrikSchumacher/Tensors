#pragma once


namespace Tensors
{
    class BitField
    {
    public:
        
        using UInt = UInt64;
        
    private:
        
        UInt n;
        Tensor1<UInt,UInt> a;
        
        static constexpr UInt size = 64;
        static constexpr UInt mask = size - 1;
        
    public:
        
        BitField()
        :   n { 0 }
        ,   a { 0 }
        {}
        
        BitField( UInt n_ )
        :   n { n_ }
        ,   a { (n + size - 1) / size }
        {}
               
        BitField( UInt n_, bool init )
        :   n { n_ }
        ,   a { (n + size - 1) / size, init ? ~UInt(0) : UInt(0) }
        {}
               
        
        bool GetBit( const UInt i ) const
        {
            const UInt j = i / size;
            const UInt k = i % size;
            
            return get_bit(a[j],k);
        }

        bool operator[]( const UInt i ) const
        {
            return GetBit(i);
        }
            
        void SetBit( const UInt i, bool bit )
        {
            const UInt j = i / size;
            const UInt k = i % size;
            
            return set_bit(a[j],k,bit);
        }
        
        void ActivateBit( const UInt i )
        {
            const UInt j = i / size;
            const UInt k = i % size;
            
            activate_bit(a[j],k);
        }
        
        void DeactivateBit( const UInt i )
        {
            const UInt j = i / size;
            const UInt k = i % size;
            
            deactivate_bit(a[j],k);
        }
            
        void Fill( const bool init )
        {
            a.Fill( init ? ~UInt(0) : UInt(0) );
        }
            
               
        
    };
}
