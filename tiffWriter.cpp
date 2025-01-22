
// date 2022-01-12
//      tiffWriter.cpp
//      这里直接采用基于tile的单层tiff的图片；
//      完成tiff的写入

#include "tiffWriter.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <chrono>

// extern "C" {
// #include "tiffio.h"

// };

#include <tiffio.h>

using namespace std;

tiffWriter::tiffWriter() : _tiff(nullptr),
     _quality(85), _tileSize(256)
{
    TIFFSetWarningHandler(nullptr);

}

tiffWriter::tiffWriter(int tileSize) : _tiff(nullptr),
     _quality(85), _tileSize(tileSize)
{
    TIFFSetWarningHandler(nullptr);

}

void tiffWriter::setSpacing(const double &sx, const double &sy)
{
    if (_tiff)
    {
        TIFFSetField(_tiff, TIFFTAG_RESOLUTIONUNIT, RESUNIT_CENTIMETER);
        TIFFSetField(_tiff, TIFFTAG_XRESOLUTION, sx);
        TIFFSetField(_tiff, TIFFTAG_YRESOLUTION, sy);
    
    }
}

void tiffWriter::appendLevelDirTags()
{
    TIFFWriteDirectory(_tiff);
}


tiffWriter::~tiffWriter() {
    if (_tiff) {
        TIFFClose(_tiff);
        _tiff = nullptr;
    }
}

//! 1. openfile
int tiffWriter::openFile(const std::string& fileName) {
    _tiff = TIFFOpen(fileName.c_str(), "w8");  
    if (!_tiff) {
        cerr << "Failed to open TIFF file for writing" << endl;
        return -1;
    }

    return 0;
}

//! 1. openfile
int tiffWriter::openFile(const std::string& fileName, const char* mode) {
    _tiff = TIFFOpen(fileName.c_str(), mode);  
    if (!_tiff) {
        cerr << "Failed to open TIFF file for read/append/write" << endl;
        return -1;
    }

    return 0;
}



// ! 3. write image information
int tiffWriter::writeImageInformation(const ull& sizeX, const ull& sizeY) {
    if (_tiff) {
        TIFFSetField(_tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(_tiff, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(_tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
        // cout << "Depth is " << _cDepth << endl;
        TIFFSetField(_tiff, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)_cDepth);
        if(_cDepth==4){
            uint16 out[1];
            out[0] = EXTRASAMPLE_ASSOCALPHA;
            TIFFSetField(_tiff, TIFFTAG_EXTRASAMPLES, 1, &out);
        }
        TIFFSetField(_tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(_tiff, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

        // set pyramid tags （jpeg）
        TIFFSetField(_tiff, TIFFTAG_COMPRESSION, COMPRESSION_JPEG);
        TIFFSetField(_tiff, TIFFTAG_JPEGQUALITY, (uint)_quality);

        TIFFSetField(_tiff, TIFFTAG_TILEWIDTH, _tileSize);
        TIFFSetField(_tiff, TIFFTAG_TILELENGTH, _tileSize);

        TIFFSetField(_tiff, TIFFTAG_IMAGEWIDTH, sizeX);
        TIFFSetField(_tiff, TIFFTAG_IMAGELENGTH, sizeY); 

        TIFFSetField(_tiff, TIFFTAG_MAKE, "Lensee Intelligent Technology Co.,Ltd");
        TIFFSetField(_tiff, TIFFTAG_IMAGEDESCRIPTION, "TCT screening");

        TIFFSetField(_tiff, TIFFTAG_SUBFILETYPE, FILETYPE_REDUCEDIMAGE);
        return 0;
    } else {
        return -1;
    }
}

int tiffWriter::writeH(const ull& Height) {
    if (_tiff) {
        TIFFSetField(_tiff, TIFFTAG_IMAGELENGTH, Height); 
        return 0;
    } else {
        return -1;
    }
}



void tiffWriter::writeBaseImagePartToTIFFTile(uchar* data, uint pos) 
{
    
    uint npixels_size = _tileSize * _tileSize * _cDepth *  sizeof(uchar);

    TIFFWriteEncodedTile(_tiff, pos, data, npixels_size);
        
}

void tiffWriter::writeBaseImagePartToTIFFTile_compress(uchar* data, uint pos, long dataLen) 
{
    
    TIFFWriteRawTile(_tiff, pos, data, dataLen); 
}


int tiffWriter::finishImage() {

    TIFFClose(_tiff);
    _tiff = nullptr;

    return 0;
}


