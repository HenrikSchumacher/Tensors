#pragma once

#define MATHEMATICA

#include "mathlink.h"

#if defined( P )
    #undef P
#endif

#include <string>
#include <cstdint>
#include <ostream>
#include <sstream>
#include <complex>



namespace mma
{
    WolframLibraryData libData;
    
    inline void print(const char *msg)
    {
        if (libData->AbortQ())
        {
            return; // trying to use the MathLink connection during an abort appears to break it
        }

        MLINK link = libData->getMathLink(libData);
        
        MLPutFunction(link, "EvaluatePacket", 1);
        
        MLPutFunction(link, "Print", 1);
        
        MLPutString(link, msg);
        
        libData->processMathLink(link);
        
        int pkt = MLNextPacket(link);
        
        if (pkt == RETURNPKT)
        {
            MLNewPacket(link);
        }
    }

    /// Call _Mathematica_'s `Print[]`, `std::string` argument version.
    inline void print(const std::string &msg)
    {
        print(msg.c_str());
    }
    
    
    template<typename T> inline T & get( MArgument marg );
    
    template<> inline mint & get<mint>( MArgument marg )
    {
        return *((marg).integer);
    }
    
    template<> inline mreal & get<mreal>( MArgument marg )
    {
        return *((marg).real);
    }
    
    template<> inline mcomplex & get<mcomplex>( MArgument marg )
    {
        return *((marg).cmplex);
    }
    
    template<> inline std::complex<mreal> & get<std::complex<mreal>>( MArgument marg )
    {
        return *(reinterpret_cast<std::complex<mreal>*>((marg).cmplex));
    }
    
    template<> inline MTensor & get<MTensor>( MArgument marg )
    {
        return *((marg).tensor);
    }
    
    template<> inline MSparseArray & get<MSparseArray>( MArgument marg )
    {
        return *((marg).sparse);
    }
    
    template<> inline mbool & get<mbool>( MArgument marg )
    {
        return *((marg).boolean);
    }
    
    template<> inline char* & get<char*>( MArgument marg )
    {
        return *((marg).utf8string);
    }
    
    
    template<typename T> inline T * data( MTensor & M );
    
    template<> inline mreal * data<mreal>( MTensor & M )
    {
        return libData->MTensor_getRealData(M);
    }
    
    template<> inline mint * data<mint>( MTensor & M )
    {
        return libData->MTensor_getIntegerData(M);
    }
    
    template<> inline mcomplex * data<mcomplex>( MTensor & M )
    {
        return libData->MTensor_getComplexData(M);
    }
    
    template<> inline std::complex<mreal> * data<std::complex<mreal>>( MTensor & M )
    {
        return reinterpret_cast<std::complex<mreal>*>( libData->MTensor_getComplexData(M) );
    }
    
    
    
    inline const mint * dimensions( MTensor & M )
    {
        return libData->MTensor_getDimensions(M);
    }
    
    inline mint rank( MTensor & M )
    {
        return libData->MTensor_getRank(M);
    }
    
    
    template<typename T> inline T * data( MArgument marg )
    {
        return data<T>(get<MTensor>(marg));
    }
    
    
    template<typename T>
    inline MTensor make_MTensor( const mint rank, const mint * dims );
    
    template<> inline MTensor make_MTensor<mint>( const mint rank, const mint * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Integer, rank, dims, &M );
        
        return M;
    }
    
    template<> inline MTensor make_MTensor<mreal>( const mint rank, const mint * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Real, rank, dims, &M );
        
        return M;
    }
    
    
    template<> inline MTensor make_MTensor<mcomplex>( const mint rank, const mint * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Complex, rank, dims, &M );
        
        return M;
    }
    
    template<> inline MTensor make_MTensor<std::complex<mreal>>( const mint rank, const mint * dims )
    {
        MTensor M;
        
        libData->MTensor_new( MType_Complex, rank, dims, &M );
        
        return M;
    }
    
    
    template<typename T>
    inline MTensor make_MTensor( const std::initializer_list<mint> dims )
    {
        return make_MTensor<T>( static_cast<mint>(dims.size()), &dims.begin()[0] );
    }
    
    
    inline void disown( MTensor & M )
    {
        libData->MTensor_disown(M);
    }
}


extern "C" DLLEXPORT mint WolframLibrary_getVersion()
{
    return WolframLibraryVersion;
}

extern "C" DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData)
{
    mma::libData = libData;
    return LIBRARY_NO_ERROR;
}

extern "C" DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData)
{
    return;
}
