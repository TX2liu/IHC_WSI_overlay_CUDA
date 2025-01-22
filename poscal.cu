
#include<vector>
#include "utility.hpp"
#include "poscal.h"


#define BLKSIZE 1024
#define inf 0x7f800000 
#define WARP_SIZE 32

using namespace ppl::cv::cuda;
__device__
inline float3 log_float3(float3 input)
{
    float3 out;
    out.x = log(input.x);
    out.y = log(input.y);
    out.z = log(input.z);

    return out;
}


// channel must be 3
__global__
void ihc_pos_cal_kernel(const uchar* src, const int src_rows, const int src_cols, float* split_ch0, float* split_ch2)
{
    int element_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int element_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (element_y >= src_rows || element_x >= src_cols) {
        return;
    }

    // int stride = 3*src_cols;

    float3 log10_rev = make_float3(-0.434294481903252, -0.434294481903252, -0.434294481903252);
    float3 rep_tmp = make_float3(0.00392156862745098, 0.00392156862745098, 0.00392156862745098);
    float3 scalar_val = make_float3(2.0, 2.0, 2.0);
    float3 rgb_hrd_col0 = make_float3(1.477869,0.38395065,-0.78720218);
    float3 rgb_hrd_col1 = make_float3(-1.4279728,1.5426698,-0.61642468);
    float3 rgb_hrd_col2 = make_float3(0.35112682,-1.1020564,1.9520504);

    uchar3 input = ((uchar3*)src)[element_y*src_cols+element_x];
    float3 inVal = make_float3(input.z, input.y, input.x);

    float3 log_out = log_float3( (scalar_val + inVal*rep_tmp ))*log10_rev;


    float* ch0 = (float*)(split_ch0 + element_y*src_cols);
    float* ch2 = (float*)(split_ch2 + element_y*src_cols);
    ch0[element_x] = log_out.x*rgb_hrd_col0.x + log_out.y*rgb_hrd_col0.y + log_out.z*rgb_hrd_col0.z;  
    ch2[element_x] = log_out.x*rgb_hrd_col2.x + log_out.y*rgb_hrd_col2.y + log_out.z*rgb_hrd_col2.z;  

}


__global__ 
void minMax_kernel(float *data_in, float* minValue, float* maxValue, int data_num)
{
    //Holds intermediates in shared memory reduction
    __shared__ float sdata_max[BLKSIZE];
    __shared__ float sdata_min[BLKSIZE];

    float maxV = -inf;
    float minV = inf;
    int offset = threadIdx.x;
    while(offset < data_num){
        
        if(maxV < data_in[offset]){
            maxV = data_in[offset];
        }
        if(minV>data_in[offset]){
            minV = data_in[offset];
        }

        offset += blockDim.x;
    }

    sdata_max[threadIdx.x] = maxV;
    sdata_min[threadIdx.x] = minV;
    __syncthreads();
    
    //shared memory reduction sweep
    #pragma unrolling
    for (int i = BLKSIZE>>1; i > 0; i>>=1){
        if (threadIdx.x < i){ 
            if(sdata_max[threadIdx.x] < sdata_max[threadIdx.x+i]){
                sdata_max[threadIdx.x] = sdata_max[threadIdx.x+i];
            }
            if(sdata_min[threadIdx.x] > sdata_min[threadIdx.x+i]){
                sdata_min[threadIdx.x] = sdata_min[threadIdx.x+i];
            }
            
            __syncthreads();
        }
        
        
    }

    minValue[blockIdx.x] = sdata_min[0];
    maxValue[blockIdx.x] = sdata_max[0];


}

__global__ 
void thresh_bin(float* Y, uchar* mask, float thresh, float* minV, float* maxV, int numRows, int numCols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= numCols || ty >= numRows ){
        return;
    }
    
    size_t ind = tx + ty * numCols;

    float th =  (thresh*(maxV[0]-minV[0])+minV[0]);
    mask[ind] = Y[ind] > th ? 255 : 0;
}

__global__
void dab_cal(int* dab, float* hrd_dab, float th1, float th2, float* minV, float* maxV, int numRows, int numCols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= numCols || ty >= numRows ){
        return;
    }

    float TH1 = th1*(maxV[0]-minV[0]) + minV[0];
    float TH2 = th2*(maxV[0]-minV[0]) + minV[0];

    size_t ind = tx + ty * numCols;

    if(hrd_dab[ind]<TH1){
        dab[ind] = 1;
    }else if(hrd_dab[ind]>TH2){
        dab[ind] = 2;
    }else{
        dab[ind] = 0;
    }
    
}

__global__
void dab_cal(uchar* dab, uchar* hematoxylin_thresh, float* hrd_dab, float* minV, float* maxV, const float th, const int numRows, const int numCols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= numCols || ty >= numRows ){
        return;
    }

    float TH1 = th*(maxV[0]-minV[0]) + minV[0];


    size_t ind = tx + ty * numCols;

    if(hematoxylin_thresh[ind]>0){
        if(hrd_dab[ind]>TH1){
            dab[ind] = 255;
        }else{
            dab[ind] = 127;
        }
    }else{
        dab[ind] = 0;
    }
    
    
}

__global__
void dab_cal(uchar* dab, float* ch0, float* hrd_dab, float* minMaxV, const float th1, const float th2, const int numRows, const int numCols)
{
    int tx = threadIdx.x + blockIdx.x * blockDim.x;
    int ty = threadIdx.y + blockIdx.y * blockDim.y;

    if (tx >= numCols || ty >= numRows ){
        return;
    }

    float TH1 = th1*(minMaxV[1]-minMaxV[0]) + minMaxV[0];
    float TH2 = th2*(minMaxV[3]-minMaxV[2]) + minMaxV[2];


    size_t ind = tx + ty * numCols;

    if(ch0[ind]>TH1){
        if(hrd_dab[ind]>TH2){
            dab[ind] = 255;
        }else{
            dab[ind] = 127;
        }
    }else{
        dab[ind] = 0;
    }
    
    
}

cv::Mat ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2, const float& rep)
{
    
    
    int numRows = ori_img.rows;
    int numCols = ori_img.cols;
    uint64_t numPixels = numRows*numCols;
    uint64_t totalSize = numPixels*ori_img.channels();

    uchar* img_d;
    cudaMalloc(&img_d, totalSize* sizeof(uchar));
    cudaMemcpy(img_d, (uchar*)ori_img.data, totalSize* sizeof(uchar), cudaMemcpyHostToDevice);

    float* ch0_d;
    float* ch2_d;
    cudaMalloc(&ch0_d, numPixels* sizeof(float));
    cudaMalloc(&ch2_d, numPixels* sizeof(float));

    // min and max
    float* maxV_d;
	float* minV_d;
	cudaMalloc(&maxV_d, sizeof(float));
	cudaMalloc(&minV_d, sizeof(float));

    uchar* hematoxylin_thresh_d;    
    cudaMalloc(&hematoxylin_thresh_d, numPixels*sizeof(uchar));


    uchar* dab_d;
    cudaMalloc(&dab_d, numPixels*sizeof(uchar));

    dim3 block1, grid1;
    block1.x = 32;
    block1.y = 32;
    grid1.x  = (numCols+block1.x-1)/block1.x;
    grid1.y  = (numRows+block1.y-1)/block1.y;

    ihc_pos_cal_kernel<<<grid1, block1>>>(img_d, numRows, numCols, ch0_d, ch2_d);

    minMax_kernel<<<1, BLKSIZE>>>(ch0_d, minV_d, maxV_d, numPixels);

    float th = 77.0/255.0;
    thresh_bin<<<grid1, block1>>>(ch0_d, hematoxylin_thresh_d, th, minV_d, maxV_d, numRows, numCols);

    minMax_kernel<<<1, BLKSIZE>>>(ch2_d, minV_d, maxV_d, numPixels);

    float TH1 = th1/rep;
    // float TH2 = th2/rep;

    cudaDeviceSynchronize();

    // dab_cal<<< grid1, block1 >>>(dab_d, ch2_d, TH1, TH2, minV_d, maxV_d, numRows, numCols);
    dab_cal<<< grid1, block1 >>>(dab_d, hematoxylin_thresh_d, ch2_d, minV_d, maxV_d, TH1, numRows, numCols);
    cv::Mat dab(numRows, numCols, CV_8U);
    

    // cv::Mat hematoxylin_thresh(numRows, numCols, CV_8U);
    // cv::Mat dab(numRows, numCols, CV_32S);
    // cudaMemcpy((uchar*)hematoxylin_thresh.data, hematoxylin_thresh_d, numPixels*sizeof(uchar), cudaMemcpyDeviceToHost);
    cudaMemcpy((uchar*)dab.data, dab_d, numPixels*sizeof(uchar), cudaMemcpyDeviceToHost);

    // cv::watershed(ori_img, dab);
    // dab.convertTo(dab, CV_8U, 127);
    // cv::threshold(dab, dab, 127, 255, THRESH_BINARY);

    // cv::Mat neg_reg;
    // cv::bitwise_not(dab, neg_reg);
    // cv::bitwise_and(neg_reg, hematoxylin_thresh, neg_reg);

    // bitwise_and(dab, hematoxylin_thresh, dab);
    // dab.setTo(127, neg_reg);

    cudaFree(img_d);
    cudaFree(ch0_d);
    cudaFree(ch2_d);
    cudaFree(maxV_d);
    cudaFree(minV_d);
    cudaFree(hematoxylin_thresh_d);
    cudaFree(dab_d);



    return dab;

}



cv::Mat ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2)
{
    
    int numRows = ori_img.rows;
    int numCols = ori_img.cols;
    uint64_t numPixels = numRows*numCols;
    uint64_t totalSize = numPixels*ori_img.channels();

    uchar* img_d;
    cudaMalloc(&img_d, totalSize* sizeof(uchar));
    cudaMemcpy(img_d, (uchar*)ori_img.data, totalSize* sizeof(uchar), cudaMemcpyHostToDevice);

    float* ch0_d;
    float* ch2_d;
    cudaMalloc(&ch0_d, numPixels* sizeof(float));
    cudaMalloc(&ch2_d, numPixels* sizeof(float));

    float* minMaxVal_d;
    cudaMalloc(&minMaxVal_d, 4*sizeof(float));


    uchar* dab_d;
    cudaMalloc(&dab_d, numPixels*sizeof(uchar));

    dim3 block1, grid1;
    block1.x = 32;
    block1.y = 32;
    grid1.x  = (numCols+block1.x-1)/block1.x;
    grid1.y  = (numRows+block1.y-1)/block1.y;

    ihc_pos_cal_kernel<<<grid1, block1>>>(img_d, numRows, numCols, ch0_d, ch2_d);
    minMax_kernel<<<1, BLKSIZE>>>(ch0_d, minMaxVal_d+0, minMaxVal_d+1, numPixels);
    minMax_kernel<<<1, BLKSIZE>>>(ch2_d, minMaxVal_d+2, minMaxVal_d+3, numPixels);

    // // th1=77/255.0;
    // // th2=(0.4~0.45)/0.95
    dab_cal<<< grid1, block1 >>>(dab_d, ch0_d, ch2_d, minMaxVal_d, th1, th2, numRows, numCols);

    cudaDeviceSynchronize();

    cv::Mat dab(numRows, numCols, CV_8U);
    
    cudaMemcpy((uchar*)dab.data, dab_d, numPixels*sizeof(uchar), cudaMemcpyDeviceToHost);


    cudaFree(img_d);
    cudaFree(ch0_d);
    cudaFree(ch2_d);
    cudaFree(minMaxVal_d);
    cudaFree(dab_d);



    return dab;

}


void ihc_cell_cal_cuda(const cv::Mat& ori_img, const float& th1, const float& th2, uchar* dab_h)
{
    
    int numRows = ori_img.rows;
    int numCols = ori_img.cols;
    uint64_t numPixels = numRows*numCols;
    uint64_t totalSize = numPixels*ori_img.channels();

    uchar* img_d;
    cudaMalloc(&img_d, totalSize* sizeof(uchar));
    cudaMemcpy(img_d, (uchar*)ori_img.data, totalSize* sizeof(uchar), cudaMemcpyHostToDevice);

    float* ch0_d;
    float* ch2_d;
    cudaMalloc(&ch0_d, numPixels* sizeof(float));
    cudaMalloc(&ch2_d, numPixels* sizeof(float));

    float* minMaxVal_d;
    cudaMalloc(&minMaxVal_d, 4*sizeof(float));


    uchar* dab_d;
    cudaMalloc(&dab_d, numPixels*sizeof(uchar));

    dim3 block1, grid1;
    block1.x = 32;
    block1.y = 32;
    grid1.x  = (numCols+block1.x-1)/block1.x;
    grid1.y  = (numRows+block1.y-1)/block1.y;

    ihc_pos_cal_kernel<<<grid1, block1>>>(img_d, numRows, numCols, ch0_d, ch2_d);
    minMax_kernel<<<1, BLKSIZE>>>(ch0_d, minMaxVal_d+0, minMaxVal_d+1, numPixels);
    minMax_kernel<<<1, BLKSIZE>>>(ch2_d, minMaxVal_d+2, minMaxVal_d+3, numPixels);

    // // th1=77/255.0;
    // // th2=(0.4~0.45)/0.95
    dab_cal<<< grid1, block1 >>>(dab_d, ch0_d, ch2_d, minMaxVal_d, th1, th2, numRows, numCols);

    cudaDeviceSynchronize();

    // cv::Mat dab(numRows, numCols, CV_8U);
    
    cudaMemcpy(dab_h, dab_d, numPixels*sizeof(uchar), cudaMemcpyDeviceToHost);


    cudaFree(img_d);
    cudaFree(ch0_d);
    cudaFree(ch2_d);
    cudaFree(minMaxVal_d);
    cudaFree(dab_d);


}