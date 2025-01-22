#ifndef __IHC_POSTIVE_CAL_H_
#define __IHC_POSTIVE_CAL_H_

#include "cuda_runtime.h"

#include <vector>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

/**
 * th1, th2, rep 都是用于通道2解卷积的
 * */
cv::Mat ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2, const float& rep);

/**
 * 实践发现，其实不需要采用watershed进行分割，采用简单的阈值分割即可，将几个阈值简化，简化为两个通道的阈值即可；
 * ori_img:     原始图像
 * th1 :        用于DAB通道解卷积之后，通道0的阈值
 * th2 ：       用于DAB通道解卷积之后， 通道2的阈值
 * */
cv::Mat ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2);

void ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2, uchar* dab_h);

#endif
