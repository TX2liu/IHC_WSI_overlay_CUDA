// date 2022-01-17
//          实现对pyr-mstiff的读取
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

#include <tiffio.h>

using namespace std;
using namespace cv;

struct tiff;
typedef struct tiff TIFF;

class tiffReader
{
protected:
    vector<TIFF *> _tiff_list;
    string _img_path_base;
    vector<vector<int>> _HW;
    int _level_count;
    vector<int> _tile_per_line;
    vector<int> _tile_per_column;
    vector<vector<int>> _tileHW;
    vector<toff_t *> _byteCounts_list;

    tsample_t _cDepth;

public:
    tiffReader(const string &img_path_base);
    virtual ~tiffReader();
    tiffReader();
    void Init(const string &img_path_base);

    int getWholeImageHeight(const int &level);
    int getWholeImageWidth(const int &level);
    int getTileH(const int &level);
    int getTileW(const int &level);
    int getTileNumPerRow(const int &level);
    int getTileNumPerCol(const int &level);
    int getLevelCnt();

    cv::Mat getTile(const int &level, const int &row_idx, const int &col_idx);
    cv::Mat getLevel0TileFast(const int &row_idx, const int &col_idx);
    cv::Mat getLevel0TileFastGray(const int &row_idx, const int &col_idx);
    int32_t getTileUncompress(const int &level, const int &row_idx, const int &col_idx, std::vector<uchar> &udata);
    int32_t getTileOriUncompress(const int &level, const int &row_idx, const int &col_idx, std::vector<uchar> &udata);
    cv::Mat getLevelImage(const int &level);

    int32_t getChnNum();

    int getDepth();
};