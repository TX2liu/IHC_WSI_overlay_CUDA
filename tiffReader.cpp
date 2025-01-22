// date 2022-01-17
//      tiffReader.cpp
//      完成pyr-mstiff的读取
//      只针对pyr-mstiff的图层读取
//

#include "tiffReader.h"

/**
 * @brief 
 * 
 * @param val 
 * @param minV :闭空间
 * @param maxV :开空间
 * @return int 
 */
int minMaxCrop(const int &val, const int &minV, const int &maxV)
{
    int rval;
    if (val < minV)
    {
        rval = minV;
    }
    else if (val > maxV-1)
    {
        rval = maxV - 1;
    }
    else
    {
        rval = val;
    }

    return rval;
}
tiffReader::tiffReader()
{
}

tiffReader::tiffReader(const string &img_path_base) : _img_path_base(img_path_base)
{
    vector<cv::String> tiff_path_vec;
    cv::glob(_img_path_base + "/*.tiff", tiff_path_vec, false);

    _level_count = tiff_path_vec.size();

    if (_level_count == 0)
    {
        cerr << "读取路径出错，该路径下没有tiff图" << endl;
        _tiff_list = {nullptr};
        _byteCounts_list = {nullptr};
        // _byteOffset_list = {nullptr};
        _HW = {{0, 0}};
        _tile_per_line = {0};
        _tile_per_column = {0};
        _tileHW = {{0, 0}};
        _cDepth = 3;
    }
    else
    {
        _tiff_list.resize(_level_count, nullptr);
        _byteCounts_list.resize(_level_count, nullptr);
        // _byteOffset_list.resize(_level_count, nullptr);
        _HW.resize(_level_count, {0, 0});
        _tile_per_line.resize(_level_count, 0);
        _tile_per_column.resize(_level_count, 0);
        _tileHW.resize(_level_count, {0, 0});

        for (size_t level = 0; level < _level_count; level++)
        {
            string img_level_path = _img_path_base + "/" + to_string(level) + ".tiff";
            _tiff_list[level] = TIFFOpen(img_level_path.c_str(), "r");
            if (!_tiff_list[level])
            {
                cerr << "图层读取失败 : " << img_level_path << endl;
                continue;
            }

            int img_whole_width = 0;
            int img_whole_height = 0;
            int tileH = 0;
            int tileW = 0;

            TIFFGetField(_tiff_list[level], TIFFTAG_IMAGEWIDTH, &img_whole_width);
            TIFFGetField(_tiff_list[level], TIFFTAG_IMAGELENGTH, &img_whole_height);
            TIFFGetField(_tiff_list[level], TIFFTAG_TILEWIDTH, &tileW);
            TIFFGetField(_tiff_list[level], TIFFTAG_TILELENGTH, &tileH);

            TIFFGetField(_tiff_list[level], TIFFTAG_TILEBYTECOUNTS, &_byteCounts_list[level]);

            _HW[level] = {img_whole_height, img_whole_width};
            _tileHW[level] = {tileH, tileW};

            _tile_per_line[level] = (img_whole_width + tileW - 1) / tileW;
            _tile_per_column[level] = (img_whole_height + tileH - 1) / tileH;
        }

        TIFFGetField(_tiff_list[0], TIFFTAG_SAMPLESPERPIXEL, &_cDepth);
    }
}

void tiffReader::Init(const string &img_path_base)
{
    _img_path_base = img_path_base;
    vector<cv::String> tiff_path_vec;
    cv::glob(_img_path_base + "/*.tiff", tiff_path_vec, false);

    _level_count = tiff_path_vec.size();

    if (_level_count == 0)
    {
        cerr << "读取路径出错，该路径下没有tiff图" << endl;
        _tiff_list = {nullptr};
        _byteCounts_list = {nullptr};
        // _byteOffset_list = {nullptr};
        _HW = {{0, 0}};
        _tile_per_line = {0};
        _tile_per_column = {0};
        _tileHW = {{0, 0}};
        _cDepth = 3;
    }
    else
    {
        _tiff_list.clear();
        _byteCounts_list.clear();
        _HW.clear();
        _tile_per_line.clear();
        _tile_per_column.clear();
        _tileHW.clear();

        _tiff_list.resize(_level_count, nullptr);
        _byteCounts_list.resize(_level_count, nullptr);
        // _byteOffset_list.resize(_level_count, nullptr);
        _HW.resize(_level_count, {0, 0});
        _tile_per_line.resize(_level_count, 0);
        _tile_per_column.resize(_level_count, 0);
        _tileHW.resize(_level_count, {0, 0});

        for (size_t level = 0; level < _level_count; level++)
        {
            string img_level_path = _img_path_base + "/" + to_string(level) + ".tiff";
            _tiff_list[level] = TIFFOpen(img_level_path.c_str(), "r");
            if (!_tiff_list[level])
            {
                cerr << "图层读取失败 : " << img_level_path << endl;
                continue;
            }

            int img_whole_width = 0;
            int img_whole_height = 0;
            int tileH = 0;
            int tileW = 0;

            TIFFGetField(_tiff_list[level], TIFFTAG_IMAGEWIDTH, &img_whole_width);
            TIFFGetField(_tiff_list[level], TIFFTAG_IMAGELENGTH, &img_whole_height);
            TIFFGetField(_tiff_list[level], TIFFTAG_TILEWIDTH, &tileW);
            TIFFGetField(_tiff_list[level], TIFFTAG_TILELENGTH, &tileH);

            TIFFGetField(_tiff_list[level], TIFFTAG_TILEBYTECOUNTS, &_byteCounts_list[level]);
            // TIFFGetField(_tiff_list[level], TIFFTAG_TILEOFFSETS, &_byteOffset_list[level]);

            _HW[level] = {img_whole_height, img_whole_width};
            _tileHW[level] = {tileH, tileW};

            _tile_per_line[level] = (img_whole_width + tileW - 1) / tileW;
            _tile_per_column[level] = (img_whole_height + tileH - 1) / tileH;
        }

        TIFFGetField(_tiff_list[0], TIFFTAG_SAMPLESPERPIXEL, &_cDepth);
    }
}

tiffReader::~tiffReader()
{
    for (auto &_tiff : _tiff_list)
    {
        if (_tiff)
        {
            TIFFClose(_tiff);
            _tiff = nullptr;
        }
    }
}

int tiffReader::getWholeImageHeight(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _HW[level][0];
}

int tiffReader::getWholeImageWidth(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _HW[level][1];
}

int tiffReader::getTileH(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _tileHW[level][0];
}

int tiffReader::getTileW(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _tileHW[level][1];
}

int tiffReader::getTileNumPerRow(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _tile_per_line[level];
}

int tiffReader::getTileNumPerCol(const int &level)
{
    assert(level >= 0 && level < _level_count);

    return _tile_per_column[level];
}

int tiffReader::getLevelCnt()
{
    return _level_count;
}

int32_t tiffReader::getChnNum()
{
    return _cDepth;
}
int tiffReader::getDepth()
{
    return _cDepth;
}

cv::Mat tiffReader::getTile(const int &level, const int &row_idx, const int &col_idx)
{
    bool is_level_meet = (level >= 0 && level < _level_count);
    bool is_row_meet = (row_idx >= 0 && row_idx < _tile_per_column[level]);
    bool is_col_meet = (col_idx >= 0 && col_idx < _tile_per_line[level]);

    cv::Mat tile;

    if ((is_level_meet) && (is_row_meet) && (is_col_meet))
    {
    
        int pos = row_idx * _tile_per_line[level] + col_idx;
        // std::cout << "level, row, col, tile_per_line : " << level << ", " << row_idx << ", " << col_idx << ", " << _tile_per_line[level] << std::endl;

        vector<uchar> udata(_byteCounts_list[level][pos]);
        // std::cout << "udata size is " << _byteCounts_list[level][pos] << std::endl;
        // TIFFReadRawTile(_tiff_list[level], pos, udata.data(), _byteCounts_list[level][pos]);
        auto retNum = TIFFReadRawTile(_tiff_list[level], pos, udata.data(), _byteCounts_list[level][pos]);
        if (retNum > 0)
        { // 读取正确
            tile = cv::imdecode(udata, IMREAD_UNCHANGED);
        }
        else
        {
            // 读取失败, 返回上一层的几个组成当前一个tile
            if (level > 1)
            {
                int pos11 = row_idx * 2 * _tile_per_line[level - 1] + col_idx * 2;
                int pos12 = row_idx * 2 * _tile_per_line[level - 1] + (col_idx * 2 + 1);
                int pos21 = (row_idx * 2 + 1) * _tile_per_line[level - 1] + col_idx * 2;
                int pos22 = (row_idx * 2 + 1) * _tile_per_line[level - 1] + (col_idx * 2 + 1);
                vector<uchar> udata11(_byteCounts_list[level - 1][pos11]);
                vector<uchar> udata12(_byteCounts_list[level - 1][pos12]);
                vector<uchar> udata21(_byteCounts_list[level - 1][pos21]);
                vector<uchar> udata22(_byteCounts_list[level - 1][pos22]);
                auto retNum11 = TIFFReadRawTile(_tiff_list[level - 1], pos11, udata11.data(), _byteCounts_list[level - 1][pos11]);
                auto retNum12 = TIFFReadRawTile(_tiff_list[level - 1], pos12, udata12.data(), _byteCounts_list[level - 1][pos12]);
                auto retNum21 = TIFFReadRawTile(_tiff_list[level - 1], pos21, udata21.data(), _byteCounts_list[level - 1][pos21]);
                auto retNum22 = TIFFReadRawTile(_tiff_list[level - 1], pos22, udata22.data(), _byteCounts_list[level - 1][pos22]);
                if (retNum11 < 0 || retNum12 < 0 || retNum21 < 0 || retNum22 < 0)
                {
                    // tile = cv::Mat::ones(_tileHW[level][0], _tileHW[level][1], CV_MAKETYPE(CV_8U, _cDepth));
                    tile = cv::Mat(_tileHW[level][0], _tileHW[level][1], CV_MAKETYPE(CV_8U, _cDepth), cv::Scalar::all(255));
                }
                else
                {
                    cv::Mat tile11 = cv::imdecode(udata11, IMREAD_UNCHANGED);
                    cv::Mat tile12 = cv::imdecode(udata12, IMREAD_UNCHANGED);
                    cv::Mat tile21 = cv::imdecode(udata21, IMREAD_UNCHANGED);
                    cv::Mat tile22 = cv::imdecode(udata22, IMREAD_UNCHANGED);
                    cv::Mat tile2x(_tileHW[level - 1][0] * 2, _tileHW[level - 1][1] * 2, CV_MAKETYPE(CV_8U, _cDepth));
                    tile11.copyTo(tile2x(cv::Rect(0, 0, _tileHW[level - 1][1], _tileHW[level - 1][0])));
                    tile12.copyTo(tile2x(cv::Rect(_tileHW[level - 1][1], 0, _tileHW[level - 1][1], _tileHW[level - 1][0])));
                    tile21.copyTo(tile2x(cv::Rect(0, _tileHW[level - 1][0], _tileHW[level - 1][1], _tileHW[level - 1][0])));
                    tile22.copyTo(tile2x(cv::Rect(_tileHW[level - 1][1], _tileHW[level - 1][0], _tileHW[level - 1][1], _tileHW[level - 1][0])));
                    cv::resize(tile2x, tile, cv::Size(_tileHW[level][1], _tileHW[level][0]));
                }
            }
            else
            {
                tile = cv::Mat(_tileHW[level][0], _tileHW[level][1], CV_MAKETYPE(CV_8U, _cDepth), cv::Scalar::all(255));
            }
        }

        if (_cDepth == 3)
        {
            cvtColor(tile, tile, COLOR_RGB2BGR);
        }
        else
        {
            cvtColor(tile, tile, COLOR_RGBA2BGRA);
        }
    }else{
        tile = cv::Mat(_tileHW[level][0], _tileHW[level][1], CV_MAKETYPE(CV_8U, _cDepth), cv::Scalar::all(255));
    }

    return tile;
}

int tiffReader::getTileUncompress(const int &level, const int &row_idx, const int &col_idx, std::vector<uchar> &udata)
{
    bool is_level_meet = (level >= 0 && level < _level_count);
    bool is_row_meet = (row_idx >= 0 && row_idx < _tile_per_column[level]);
    bool is_col_meet = (col_idx >= 0 && col_idx < _tile_per_line[level]);

    if((!is_level_meet) || (!is_row_meet) || (!is_col_meet))
    {
        return -1;
    }

    int pos = row_idx * _tile_per_line[level] + col_idx;

    udata.clear();
    udata.resize(_byteCounts_list[level][pos]);
    auto ret = TIFFReadRawTile(_tiff_list[level], pos, udata.data(), _byteCounts_list[level][pos]);
    if(ret>0){
        // 往上的层级，边缘处的tile，存的不一定都是组织，需规定在整图宽高之内
        if (level > 2)
        {
            int edge_tile_width = _HW[level][1] - col_idx * _tileHW[level][1];
            int edge_tile_height = _HW[level][0] - row_idx * _tileHW[level][0];
            bool is_edge = (edge_tile_width < _tileHW[level][1]) || (edge_tile_height < _tileHW[level][0]);
            if (is_edge)
            {
                udata.clear();
                cv::Mat tile = cv::imdecode(udata, IMREAD_UNCHANGED);
                tile = tile(Range(0, _HW[level][0] - row_idx * _tileHW[level][0]), Range(0, _HW[level][1] - col_idx * _tileHW[level][1]));
                if (_cDepth == 3)
                {
                    cvtColor(tile, tile, COLOR_RGB2BGR);
                    vector<int> compression_params;
                    compression_params.push_back(IMWRITE_JPEG_QUALITY);
                    compression_params.push_back(70);
                    cv::imencode(".jpeg", tile, udata, compression_params);
                }
                else
                {
                    cvtColor(tile, tile, COLOR_RGBA2BGRA);
                    cv::imencode(".png", tile, udata);
                }
            }
        }

        return ret;
    }else{
        return -1;
    }
}

int32_t tiffReader::getTileOriUncompress(const int &level, const int &row_idx, const int &col_idx, std::vector<uchar> &udata)
{

    bool is_level_meet = (level >= 0 && level < _level_count);
    bool is_row_meet = (row_idx >= 0 && row_idx < _tile_per_column[level]);
    bool is_col_meet = (col_idx >= 0 && col_idx < _tile_per_line[level]);

    if ((!is_level_meet) || (!is_row_meet) || (!is_col_meet))
    {
        return -1;
    }

    int pos = row_idx * _tile_per_line[level] + col_idx;

    udata.clear();
    udata.resize(_byteCounts_list[level][pos]);
    int ret = TIFFReadRawTile(_tiff_list[level], pos, udata.data(), _byteCounts_list[level][pos]);
    if(ret>0){
        return pos;
    }else{
        return -1;
    }

}

cv::Mat tiffReader::getLevel0TileFast(const int &row_idx, const int &col_idx)
{
    cv::Mat tile;

    int row_idx_new = minMaxCrop(row_idx, 0, _tile_per_column[0]);
    int col_idx_new = minMaxCrop(col_idx, 0, _tile_per_line[0]);

    int pos = row_idx_new * _tile_per_line[0] + col_idx_new;

    vector<uchar> udata(_byteCounts_list[0][pos]);
    int ret = TIFFReadRawTile(_tiff_list[0], pos, udata.data(), _byteCounts_list[0][pos]);

    if(ret>0){
        tile = cv::imdecode(udata, IMREAD_UNCHANGED);

        // cvtColor(tile, tile, COLOR_RGB2BGR);
        if (_cDepth == 3)
        {
            cvtColor(tile, tile, COLOR_RGB2BGR);
        }
        else
        {
            cvtColor(tile, tile, COLOR_RGBA2BGRA);
        }
    }else{
        tile = cv::Mat(_tileHW[0][1], _tileHW[0][0], CV_MAKETYPE(CV_8U, _cDepth), cv::Scalar::all(255));
    }

    return tile;
}

cv::Mat tiffReader::getLevel0TileFastGray(const int &row_idx, const int &col_idx)
{
    cv::Mat tile;

    int row_idx_new = minMaxCrop(row_idx, 0, _tile_per_column[0]);
    int col_idx_new = minMaxCrop(col_idx, 0, _tile_per_line[0]);

    int pos = row_idx_new * _tile_per_line[0] + col_idx_new;

    vector<uchar> udata(_byteCounts_list[0][pos]);
    int ret = TIFFReadRawTile(_tiff_list[0], pos, udata.data(), _byteCounts_list[0][pos]);

    if(ret>0){
        tile = cv::imdecode(udata, IMREAD_UNCHANGED);

        // cvtColor(tile, tile, COLOR_RGB2GRAY);
        if (_cDepth == 3)
        {
            cvtColor(tile, tile, COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(tile, tile, COLOR_RGBA2GRAY);
        }
    }else{
        tile = cv::Mat(_tileHW[0][1], _tileHW[0][0], CV_8U, cv::Scalar::all(255));
    }
    

    return tile;
}

cv::Mat tiffReader::getLevelImage(const int &level)
{
    // cv::Mat levelImg(_HW[level][0], _HW[level][1], CV_8UC3);
    cv::Mat levelImg(_HW[level][0], _HW[level][1], CV_MAKETYPE(CV_8U, _cDepth));

    int tileW = _tileHW[level][1];
    int tileH = _tileHW[level][0];
    for (int row = 0; row < _tile_per_column[level]; row++)
    {
        for (int col = 0; col < _tile_per_line[level]; col++)
        {
            int xstart = col * tileW;
            int xend = xstart + tileW < _HW[level][1] ? xstart + tileW : _HW[level][1];
            int ystart = row * tileH;
            int yend = ystart + tileH < _HW[level][0] ? ystart + tileH : _HW[level][0];

            // int s_width = xend - xstart;
            // int s_height = yend - ystart;

            cv::Mat tile = getTile(level, row, col);

            tile(cv::Rect(0, 0, xend - xstart, yend - ystart)).copyTo(levelImg(Range(ystart, yend), Range(xstart, xend)));
        }
    }

    return levelImg;
}

