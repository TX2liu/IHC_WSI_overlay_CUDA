// date : 2021-09-01
//      ihc_wsi_overlay_cuda.cpp
//      完成整体WSI的两张阳性率区域叠加，并求解阳性率区域，染色，最后生成新的金字塔图的过程；
//      两张大图输入是两张已经经过配准，并且大小一致的金字塔图
// 
// date 2021-09-02
//      调试，先构建CPU版本，进行调试，去除bug部分
//  
// date 2021-09-03
//      进行参数化，调试程序结构，形成Tx 部署版本
//      加入CPU版调试的参数，加入阳性率计算


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <iostream>
#include <algorithm>
#include <functional>
#include <memory>
#include <iterator>
#include <fstream>
#include <numeric>
#include <utility>

#include <vector>
#include <string>
#include <iomanip>
#include <cmath>

// 用于创建文件树
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h> // get dirname and filename
#include <errno.h>  // disp error information

#include "sutil.h"

#include "poscal.h"

#include <chrono>

using namespace cv;
using namespace std;
// using namespace ppl::cv::cuda;

cv::Mat ihc_cell_cal(const Mat& ori_img, const Mat& img_hrd, const float& th1, const float& th2, const float& rep);


void split(const string& s, vector<string>& tokens, char delim = ' ') {
    tokens.clear();
    auto string_find_first_not = [s, delim](size_t pos = 0) -> size_t {
        for (size_t i = pos; i < s.size(); i++){
            if (s[i] != delim) return i;
        }
        return string::npos;
    };   
    size_t lastPos = string_find_first_not(0);
    size_t pos = s.find(delim, lastPos);
    while (lastPos != string::npos) {
        tokens.emplace_back(s.substr(lastPos, pos - lastPos));
        lastPos = string_find_first_not(pos);
        pos = s.find(delim, lastPos);
    }
}

int main(int argc,char *argv[])
{
	if (argc !=7){//输入参数个数必须是6
		cout << "********params error, only 4！********" << endl;
        cout << "now argc is " << argc << endl;
		cout << "execute order" << endl;
		cout << "argv 0: exe file " << endl; 
		cout << "argv 1:（左边 IHC)" << endl;
		cout << "argv 2: (右边 IHC)" << endl;
        cout << "argv 3: save new pyramid path " << endl;
        cout << "argv 4: positive threshold 1 : 默认参数 77 ，一般在60~90之间调整， 不超过255"  << endl;
        cout << "argv 5: positive threshold 2 : 默认参数 0.425,一般在0.3~0.5调整 "  << endl;
        cout << "argv 6: 金字塔是20x（20）还是10x（10）";
        // cout << "argv 7: 细胞面积，默认为314 " << endl;
		cout << "example: " << endl;
		cout << "************************************************************" << endl;
		return -1;
	}

    // 四种颜色
    cv::Scalar ref_color(255, 0, 0);
    cv::Scalar flt_color(0, 255, 0);
    cv::Scalar overlay_pos(0, 0,255);
    cv::Scalar overlay_neg(0,255,255);
    uchar alpha_val = 255;


    //获取两个金字塔文件路径
    std::string referenc_img_tile_folder(argv[1]);
    std::string template_img_tile_folder(argv[2]);

    // 获取金字塔所有子文件
    vector<string> v_referenc_folders = getFiles(referenc_img_tile_folder);
    vector<string> v_template_folders = getFiles(template_img_tile_folder);

    string pyrDown_base = string(argv[3]);

    // 后续作为参数传输进来
    float pos_th1 = std::stof(argv[4])/255.0;
    float pos_th2 = std::stof(argv[5]);
    int num_level = std::stoi(argv[6]);
    // float cell_area = std::stof(argv[7]);
    if((num_level !=20) || (num_level !=10)){
        cerr << "倍率选择出错， 必须要是 20 或者 10"  << endl;
        return -1;
    }
    double num_Rate = num_level==20 ? 1. : 2.;

    const int tilesize = 256;
    const int tile_shift = 8;

    int tilesNumCols_ref = 0;
    int tilesNumRows_ref = 0;
    int tilesNumCols_flt = 0;
    int tilesNumRows_flt = 0;

    const int regsit_level_disp = 0;
    string ref_path_base = referenc_img_tile_folder+"/"+v_referenc_folders[regsit_level_disp];
    string flt_path_base = template_img_tile_folder+"/"+v_template_folders[regsit_level_disp];

    // 获取图像路径
    string path_in = "/*.jpeg";
    string path_glob_ref = ref_path_base + path_in;
    string path_glob_flt = flt_path_base + path_in;
    vector<cv::String> v_img_path_ref;
    vector<cv::String> v_img_path_flt;
    cv::glob(path_glob_ref, v_img_path_ref, false); 
    cv::glob(path_glob_flt, v_img_path_flt, false); 
    
    if( v_img_path_ref.empty() || v_img_path_flt.empty()){
        cerr << " 路径出错导致tile数为0 或者 没有jpeg图像" << endl;
        return -1;
    }

    get_col_row_num(v_img_path_ref, tilesNumCols_ref, tilesNumRows_ref);
    get_col_row_num(v_img_path_flt, tilesNumCols_flt, tilesNumRows_flt);

    int ref_whole_width = tilesNumCols_ref * tilesize;
    int ref_whole_height = tilesNumRows_ref * tilesize;
    int flt_whole_width = tilesNumCols_flt * tilesize;
    int flt_whole_height = tilesNumRows_flt * tilesize;

    if((ref_whole_height != flt_whole_height) || (ref_whole_width != flt_whole_width)){
        std::cerr << "图像大小不一致 " << std::endl;
        return -1;
    }

    cout << "ref and flt width, height : " << ref_whole_width << ", " << ref_whole_height << endl;

    
    const int pos_thresh = 240;
    const int neg_thresh1 = 120;
    const int neg_thresh2 = 140;
    const int ori_g_thresh = 50;


    // 构建存储器 , ref_whole_height, ref_whole_width 第0级的都是1024的整数倍
    const int fix_row_len = 1024;
    const int fix_row_num = ref_whole_height/fix_row_len;
    const int tile_row_num_0 = 4; // 1024 是 256 的 4 倍
    const int tile_col_num_0 = (ref_whole_width)/tilesize;
    const int tile_overlap = 1;

    // 用于每个level的变量
    int pyrsize_rows;  // pyrdown 之后的子图行数
    int pyrsize_cols;  // pyrdown 之后的子图列数
    int tile_row_num_level; // 计算该level下的行切块个数
    int tile_col_num_level; // 计算该level下的列切块个数

    const int max_level_count = 10;
    Mat tmp_vcon_row_vec[max_level_count+1]{};   // 用于记录leve1到level10的暂存拼接量
    Mat tmp_last_row[2]{};  // 用于记录level1 和level2, 当前row_count对应图的最后一列

    int level_tile_count[max_level_count+1]={0};


    // 用于创建文件夹folder
    struct stat st = {0};
    // 创建主文件夹folder，在该文件夹里边的分放各级（ level ）的切块图
    
    auto tss = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
	ss << std::put_time(std::localtime(&tss), "%Y_%m%d_%H%M%S");

    std::string pyDown_path = pyrDown_base + "/" + ss.str();
    cout << "pyDown_path is " << pyDown_path << endl;

    if (stat(pyDown_path.c_str(), &st) == -1){ 
        mkdir(pyDown_path.c_str(), 0700);
    }

    for(int level_i=0; level_i<=max_level_count; level_i++){
        // 创建该level的文件夹
        if (stat((pyDown_path+"/"+to_string(level_i)).c_str(), &st) == -1){
            mkdir((pyDown_path+"/"+to_string(level_i)).c_str(), 0700);
        }
    }

    // for image with compression (Jpeg)
    const char format_name[] = ".png";
    // vector<int> compression_params;
    // compression_params.push_back(IMWRITE_JPEG_QUALITY);
    // compression_params.push_back(75);

    cv::Mat ref_big_image(fix_row_len, ref_whole_width, CV_8UC3);
    cv::Mat flt_big_image(fix_row_len, ref_whole_width, CV_8UC3);
    
    cv::Mat pos_region_ref(fix_row_len, ref_whole_width, CV_8U);
    cv::Mat pos_region_flt(fix_row_len, ref_whole_width, CV_8U);

    cv::Mat overlay_ihc(fix_row_len+1, ref_whole_width, CV_8UC4);
    
    std::vector<uint64_t> lpos_rpos(fix_row_len, 0);
    std::vector<uint64_t> lpos_rneg(fix_row_len, 0);
    std::vector<uint64_t> lneg_rpos(fix_row_len, 0);
    std::vector<uint64_t> lneg_rneg(fix_row_len, 0);


    for(int row_ind=0; row_ind<fix_row_num; row_ind++){

        //#pragma omp parallel for collapse(2)
        for(int col_ind=0; col_ind<tile_col_num_0; col_ind++){
            for(int k_ind=0; k_ind<4; k_ind++){
                int row_Idx = row_ind*4+k_ind;

                string flt_path = flt_path_base + "/" + to_string(col_ind) + "_" + to_string(row_Idx) + ".jpeg";
                string ref_path = ref_path_base + "/" + to_string(col_ind) + "_" + to_string(row_Idx) + ".jpeg";
                cv::Mat flt_img = cv::imread(flt_path);
                cv::Mat ref_img = cv::imread(ref_path);

                int tile_col_start = ref_img.cols> tilesize ? 1 : 0;
                int tile_row_start = ref_img.rows> tilesize ? 1 : 0;
                // int row_start = row_Idx*tilesize;
                int row_start = k_ind*tilesize;
                int col_start = col_ind*tilesize; 

                ref_img(cv::Rect(tile_col_start,tile_row_start, tilesize, tilesize)).copyTo(ref_big_image(cv::Rect(col_start, row_start, tilesize, tilesize)));
                flt_img(cv::Rect(tile_col_start,tile_row_start, tilesize, tilesize)).copyTo(flt_big_image(cv::Rect(col_start, row_start, tilesize, tilesize)));
                
            }
        }

        int row_stride = row_ind==0 ? 0 : 1;

        uchar* pos_region_flt_ptr = (uchar*)pos_region_flt.data;
        uchar* pos_region_ref_ptr = (uchar*)pos_region_ref.data;

        ihc_cell_cal_cuda(ref_big_image, pos_th1, pos_th2, pos_region_ref_ptr);
        ihc_cell_cal_cuda(flt_big_image, pos_th1, pos_th2, pos_region_flt_ptr);


        // 变换之后，经过插值，可能不是原先的255或者127， 可能存在一定的范围
       // #pragma omp parallel for 
        for(int row=0; row<fix_row_len; row++){
            
            // bgr(只针对g通道)
            uchar* ref_ori_ptr = ref_big_image.ptr<uchar>(row);
            uchar* flt_reg_ptr = flt_big_image.ptr<uchar>(row); 
            // one channel
            uchar* flt_ptr = pos_region_flt.ptr<uchar>(row);
            uchar* ref_ptr = pos_region_ref.ptr<uchar>(row);
            // bgra
            uchar* over_ptr = overlay_ihc.ptr<uchar>(row+row_stride);

            for(int col=0; col<ref_whole_width; col++){

                bool is_flt_ori_greater = flt_reg_ptr[3*col+1]>ori_g_thresh;
                bool is_flt_pos_greater = flt_ptr[col]>=pos_thresh;
                bool is_flt_neg = flt_ptr[col]>neg_thresh1 && flt_ptr[col]<neg_thresh2;

                bool is_ref_ori_greater = ref_ori_ptr[3*col+1]>ori_g_thresh;
                bool is_ref_pos_greater = ref_ptr[col]>pos_thresh;
                bool is_ref_neg  = ref_ptr[col]>neg_thresh1 && ref_ptr[col]<neg_thresh2;

                if((is_flt_ori_greater && is_flt_pos_greater) && (is_ref_ori_greater && is_ref_pos_greater) ){  // 双阳
                    over_ptr[4*col+0] = overlay_pos[0];
                    over_ptr[4*col+1] = overlay_pos[1];
                    over_ptr[4*col+2] = overlay_pos[2];
                    over_ptr[4*col+3] = alpha_val;

                    lpos_rpos[row]++;

                }else if((is_flt_ori_greater && is_flt_pos_greater) && is_ref_neg){  // 右阳左阴
                    over_ptr[4*col+0] = flt_color[0];
                    over_ptr[4*col+1] = flt_color[1];
                    over_ptr[4*col+2] = flt_color[2];
                    over_ptr[4*col+3] = alpha_val;

                    lneg_rpos[row]++;

                }else if(is_flt_neg && (is_ref_ori_greater && is_ref_pos_greater) ){  // 左阳右阴
                    over_ptr[4*col+0] = ref_color[0];
                    over_ptr[4*col+1] = ref_color[1];
                    over_ptr[4*col+2] = ref_color[2];
                    over_ptr[4*col+3] = alpha_val;

                    lpos_rneg[row]++;

                }else if( is_flt_neg && is_ref_neg){  // 双阴
                    over_ptr[4*col+0] = overlay_neg[0];
                    over_ptr[4*col+1] = overlay_neg[1];
                    over_ptr[4*col+2] = overlay_neg[2];
                    over_ptr[4*col+3] = alpha_val;

                    lneg_rneg[row]++;

                }else{ // 其他
                    over_ptr[4*col+0] = 0;
                    over_ptr[4*col+1] = 0;
                    over_ptr[4*col+2] = 0;
                    over_ptr[4*col+3] = 0;
                }

            }
        }

        cv::Mat vcon_row;
        if(row_ind>0){
            vcon_row = overlay_ihc;
        }else{
            vcon_row = overlay_ihc.rowRange(0, fix_row_len);
        }


        if(row_ind==0){
            // 第0级截取/存储
            // #pragma omp parallel for num_threads(4)
           // #pragma omp parallel for
            for(int tile_i=0; tile_i<tile_row_num_0; tile_i++){ // 1024是tile_size的4倍
                for(int tile_j=0; tile_j<tile_col_num_0; tile_j++){ //
                    int offset_x = (tile_j==0) ? 0 : tile_overlap;
                    int offset_y = (tile_i==0) ? 0 : tile_overlap;
                    int x = (tile_j<<tile_shift) - offset_x;
                    int x1 = ((tile_j+1)<<tile_shift);
                    int y = (tile_i<<tile_shift) - offset_y;
                    int y1 = ((tile_i+1)<<tile_shift);

                    Mat tile_img = vcon_row(Range(y, y1), Range(x, x1));
            
                    imwrite(pyDown_path+"/0/"+to_string(tile_j)+"_"+ to_string(tile_i + row_ind*tile_row_num_0)+format_name, tile_img);
                }
            }
        }else{
            // #pragma omp parallel for num_threads(4)
           // #pragma omp parallel for
            for(int tile_i=0; tile_i<tile_row_num_0; tile_i++){
                for(int tile_j=0; tile_j<tile_col_num_0; tile_j++){
            
                    int offset_x = (tile_j==0) ? 0 : tile_overlap;
                    int x = (tile_j<<tile_shift) - offset_x;
                    int x1 = (tile_j+1)<<tile_shift;
                    int y = tile_i<<tile_shift;
                    int y1 = ((tile_i+1)<<tile_shift)+tile_overlap;
                    Mat tile_img = vcon_row(Range(y, y1), Range(x, x1));
            
                    imwrite(pyDown_path+"/0/"+to_string(tile_j)+"_"+to_string(tile_i+row_ind*
                        tile_row_num_0)+format_name, tile_img);
                }
            }
        }


         // 其他级
        for(int level_i=1; level_i<=10; level_i++){
            
            pyrsize_rows = fix_row_len>>level_i;  // 每次只右移一位
            pyrsize_cols = ref_whole_width>>level_i;
            tile_row_num_level = (pyrsize_rows + tilesize-1)>>tile_shift;
            tile_col_num_level = (pyrsize_cols + tilesize-1)>>tile_shift;

            cv::resize(vcon_row, vcon_row, Size(pyrsize_cols, pyrsize_rows));
  
            if(level_i<=2){ // 不能小于256行， 此时不高于2级，因此pyrsize_cols依然能被256整除

                // #pragma omp parallel for collapse(2) num_threads(4)
               // #pragma omp parallel for collapse(2)
                for(int tile_i=0; tile_i<tile_row_num_level; tile_i++){
                    for(int tile_j=0; tile_j<tile_col_num_level; tile_j++){
                
                        int x = (tile_j<<tile_shift);
                        // int x1 = (tile_j+1)<<tile_shift;
                        int x1 = x+tilesize<=pyrsize_cols ? x+tilesize : pyrsize_cols;
                        int y = (tile_i<<tile_shift);
                        int y1 = (tile_i+1)<<tile_shift;
                        if(x1>x){
                            Mat tile_img = vcon_row(Range(y, y1), Range(x, x1));
                            imwrite(pyDown_path+"/"+to_string(level_i)+"/"+to_string(tile_j)+"_"+
                                to_string(tile_i+level_tile_count[level_i]*tile_row_num_level)+format_name, tile_img);
                        }
                        
                    }
                }

                level_tile_count[level_i]++;

            }else{ // 小于256的，存起来，与其他行的同一级的图像再进行拼接
                // tmp_vcon_row_vec[level_i] = vcon_row.clone();

                if(tmp_vcon_row_vec[level_i].rows>0){
                    vconcat(tmp_vcon_row_vec[level_i], vcon_row, tmp_vcon_row_vec[level_i]); // 先进行拼接
                    
                    if(tmp_vcon_row_vec[level_i].rows > tilesize){//判断是否超过tile_size，超过tile_size,才能进行切图
                        // #pragma omp parallel for num_threads(4)
                        //#pragma omp parallel for
                        for(int tile_j=0; tile_j<tile_col_num_level; tile_j++){ // 这时，无法判定pyrsize_cols是否被256整除
                            int x = (tile_j<<tile_shift);
                            int x1 = x+tilesize<=pyrsize_cols ? x+tilesize : pyrsize_cols;
                            Mat tile_img = tmp_vcon_row_vec[level_i](Range(0, tilesize), Range(x, x1));
                    
                            imwrite(pyDown_path+"/"+to_string(level_i)+"/"+to_string(tile_j)+"_"+
                                to_string(level_tile_count[level_i])+format_name, tile_img);
                        }
                        
                        // 分割出剩下的一部分等待跟下一次得到的vcon_row 进行拼接
                        tmp_vcon_row_vec[level_i] = tmp_vcon_row_vec[level_i].rowRange(tilesize, tmp_vcon_row_vec[level_i].rows);

                        level_tile_count[level_i]++; // 对每一行进行递增，记录该level下的行数
                    }
                }else{
                    tmp_vcon_row_vec[level_i] = vcon_row.clone();
                }

            }
        }
        
        if(row_ind==0){
            overlay_ihc.row(fix_row_len-1).copyTo(overlay_ihc.row(0));
        }else{
            overlay_ihc.row(fix_row_len).copyTo(overlay_ihc.row(0));
        }


        // 这个可以用来表示进度
        // cout << "finish at " << row_ind << " / " << fix_row_num << endl;
        cout << "finish:" << (row_ind * 100) / (float)fix_row_num << endl;
    }
    cout << "finish:" << 100.0 << endl;


    for(int level_i=3; level_i<=10; level_i++){
        
        if(tmp_vcon_row_vec[level_i].cols>tilesize){//判断列是否超过256, 因为行必定小于256了(如果超过256,在前边就已经截取了)
            int tile_col_num = (tmp_vcon_row_vec[level_i].cols + tilesize-1)>>8;
            // #pragma omp parallel for num_threads(4)
            //#pragma omp parallel for
            for(int tile_j=0; tile_j<tile_col_num; tile_j++){
                int x = tile_j<<tile_shift;
                int xend = x+tilesize < tmp_vcon_row_vec[level_i].cols ? x+tilesize : tmp_vcon_row_vec[level_i].cols;
                if(xend>x){
                    Mat tile_img = tmp_vcon_row_vec[level_i].colRange(x, xend);
                    imwrite(pyDown_path+"/"+to_string(level_i)+"/"+to_string(tile_j)+"_"+
                        to_string(level_tile_count[level_i])+format_name, tile_img);    
                }
                   
            }       
        }else{
            imwrite(pyDown_path+"/"+to_string(level_i)+"/"+"0_"+to_string(level_tile_count[level_i])+
                format_name, tmp_vcon_row_vec[level_i]);
        }
    }

    // 最后一部分，对行列都小于256的部分进行在缩小，直至行列都必须为1（deepzoom插件的要求）
    int count_last_level = 11;
    while((tmp_vcon_row_vec[10].rows)/2>=1 || (tmp_vcon_row_vec[10].cols)/2>=1){
        // 创建该level的文件夹
        if (stat((pyDown_path+"/"+to_string(count_last_level)).c_str(), &st) == -1){
            mkdir((pyDown_path+"/"+to_string(count_last_level)).c_str(), 0700);
        }

        cv::resize(tmp_vcon_row_vec[10], tmp_vcon_row_vec[10],  Size((tmp_vcon_row_vec[10].cols+1)/2, (tmp_vcon_row_vec[10].rows+1)/2));
        imwrite(pyDown_path+"/"+to_string(count_last_level)+"/0_0"+format_name, tmp_vcon_row_vec[10]);

        count_last_level++;
    }
    
    // =========================依据deepzoom的要求，重新修改已经生成好的子文件夹名称 ===========================
    swap_folder_name(pyDown_path);


    // 计算一下各种阳性率
    uint64_t lpos_rpos_sum = 0;
    uint64_t lpos_rneg_sum = 0;
    uint64_t lneg_rpos_sum = 0;
    uint64_t lneg_rneg_sum = 0;
    for(int i=0; i<fix_row_len; i++){
        lpos_rpos_sum += lpos_rpos[i];
        lpos_rneg_sum += lpos_rneg[i];
        lneg_rpos_sum += lneg_rpos[i];
        lneg_rneg_sum += lneg_rneg[i];
    }

    double lrpos_to_lprn = (double)lpos_rpos_sum/(double)(lpos_rpos_sum + lpos_rneg_sum);
    double lrpos_to_lnrp = (double)lpos_rpos_sum/(double)(lpos_rpos_sum + lneg_rpos_sum);
    double lrpos_to_lrneg = (double)lpos_rpos_sum/(double)(lpos_rpos_sum + lneg_rneg_sum+lpos_rneg_sum+lneg_rpos_sum);

    // 20x的倍率下, 细胞核直径约为6um， 对应像素面积大约为314
    // 10x的倍率下，对应面积大约为78.5;
    double nuclei_area = 314.0/(num_Rate*num_Rate);
    uint64_t lrpos_cell_nums = (double)lpos_rpos_sum/nuclei_area;
    uint64_t lpos_cell_nums = (double)(lpos_rpos_sum + lpos_rneg_sum)/nuclei_area;
    uint64_t rpos_cell_nums = (double)(lpos_rpos_sum + lneg_rpos_sum)/nuclei_area;
    
    // cout << "双阳/左阳右阴 ： " << lrpos_to_lprn << endl;
    // cout << "双阳/右阳左阴 ： " << lrpos_to_lnrp << endl;
    // cout << "双阳/全部 ： " << lrpos_to_lrneg << endl;
    cout << "leftRight:" << lrpos_to_lprn << endl;
    cout << "rightLeft:" << lrpos_to_lnrp << endl;
    cout << "all:" << lrpos_to_lrneg << endl;
    cout << "doublePos:" << lrpos_cell_nums << endl;  // 双阳细胞总数
    cout << "leftPos:" << lpos_cell_nums << endl;    // 左边细胞总数
    cout << "rightPos:" << rpos_cell_nums << endl;   // 右边细胞总数


	return 0;
}


