
#pragma once 

// date 2022-01-12
//      tiffWriter.h
//      简单写入tiff（tile-base）图
//      
// #include "typedef.h"

#include <string>
#include <vector>
using namespace std;

struct tiff;
typedef struct tiff TIFF;

typedef unsigned long long ull;
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ul;
typedef long long ll;
typedef unsigned short ushort;


// 1.  openFile
// 2.  writeImageInformation
// 3.  writeBaseParts
// 4.  finishImage

class tiffWriter {
protected:

    //! Reference to the file to be written
    TIFF *_tiff;

public:

    //! Tile size
    uint _tileSize;
    //! JPEG compression quality
    float _quality;
    //! Min and max values of the image that is written to disk
    uint _cDepth = 3;
    
    tiffWriter();
    tiffWriter(int tileSize);
    virtual ~tiffWriter();

    // Opens the file for writing and keeps handle
    int openFile(const std::string& fileName);
    int openFile(const std::string& fileName, const char* mode);


    //! Writes the image information like image size, tile size, color and data types
    int writeImageInformation(const ull& sizeX, const ull& sizeY);

    int writeH(const ull& Height);

    void setSpacing(const double &sx, const double &sy);
    void appendLevelDirTags();

    //! Write image functions for different data types. This function provides functionality to write parts
    //! of the input image, so it does not have to be loaded in memory fully, can be useful for testing or
    //! large processing pipelines.

    void writeBaseImagePartToTIFFTile(uchar *data, uint pos);

    void writeBaseImagePartToTIFFTile_compress(uchar* data, uint pos, long dataLen);

    //! Will close the base image and finish writing the image pyramid and optionally the thumbnail image.
    //! Subsequently the image will be closed.
    int finishImage();

  
    void setTileSize(const uint& tileSize) {
        _tileSize = tileSize;
    }

    void setTileDepth(const uint& Depth) {
        _cDepth = Depth;
    }

    //! Set JPEG quality (default value = 30)
    int setJPEGQuality(const float& quality) 
    {
        if (quality > 0 && quality <= 100) {
            _quality = quality; 

        } else if(quality<=0){
            _quality = 1; 
            
        }else{
            _quality = 100; 
        } 

        return 0;
    }


};

