
#ifndef _SUTIL_H_
#define _SUTIL_H_

#include <numeric>  //atoi
#include <iterator>
#include <vector>

// 用于创建文件树
#include <sstream>
#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h>  // get dirname and filename
#include <string>
#include <set>

#include <memory>

#include <omp.h>

using namespace std;

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

void get_col_row_num(const vector<cv::String>& path, int& numCols, int& numRows)
{
    set<int> col_ind;
    set<int> row_ind;
    for(auto& v : path){
        vector<string> str_v1 = split(v, '/');
        vector<string> str_v2 = split(str_v1[str_v1.size()-1], '.');
        vector<string> str_v3 = split(str_v2[0], '_');

        col_ind.insert(std::stoi(str_v3[0]));
        row_ind.insert(std::stoi(str_v3[1]));
        
    }
    
    numCols = (*col_ind.end());
    numRows = (*row_ind.end());
}


vector<int> sort_indexes(const vector<int> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),[&v](int i1, int i2) {return v[i1] > v[i2];});

  return idx;
}



// 针对dzi中的某一层进行拼接
cv::Mat tile_concat(const string path_base)
{
    // 获取图像路径
    string path_in = "/*.jpeg";
    string path_glob = path_base + path_in;
    vector<cv::String> v_img_path;
    cv::glob(path_glob, v_img_path, false); // recurse

    set<int> col_ind;
    set<int> row_ind;
    for(auto& v : v_img_path){

        cv::Mat img = cv::imread(v);
        if(img.empty()){
            continue;
        }
        vector<string> str_v1 = split(v, '/');
        vector<string> str_v2 = split(str_v1[str_v1.size()-1], '.');
        vector<string> str_v3 = split(str_v2[0], '_');

        col_ind.insert(std::stoi(str_v3[0]));
        row_ind.insert(std::stoi(str_v3[1]));
        
    }

    const int row = (*row_ind.end());
    const int col = (*col_ind.end());

    // cout << "row, col " << row << ", " << col << endl;
    cv::Mat merge_all;
    cv::Mat merge_one_col;
    cv::Mat img;
    const int tilesize = 256;

    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){

            
            std::string img_path = path_base + "/" + to_string(j) + "_" + to_string(i) + ".jpeg";
            img = cv::imread(img_path);
            if(img.empty()){
                continue;
            }

            int cols_ind = img.cols>= tilesize ? tilesize : img.cols;
            int rows_ind = img.rows>=tilesize ? tilesize : img.rows;
            if(j==0){
                merge_one_col = img(cv::Rect(0,0, cols_ind, rows_ind));
            }else{
                int rows_ = merge_one_col.rows;
                hconcat(merge_one_col, img(cv::Rect(0,0, cols_ind, rows_)), merge_one_col);
            }
        }


        if(i==0){
            merge_all = merge_one_col;
        }else{
            vconcat(merge_all, merge_one_col, merge_all);
        }
    }

    return merge_all;
}


// 针对dzi中的某一层进行拼接
cv::Mat tile_merge(const string path_base)
{
    // 获取图像路径
    string path_in = "/*.jpeg";
    string path_glob = path_base + path_in;
    vector<cv::String> v_img_path;
    cv::glob(path_glob, v_img_path, true); // recurse

    set<int> col_ind;
    set<int> row_ind;
    for(auto& v : v_img_path){

        cv::Mat img = cv::imread(v);
        if(img.empty()){
            continue;
        }
        
        vector<string> str_v1 = split(v, '/');
        vector<string> str_v2 = split(str_v1[str_v1.size()-1], '.');
        vector<string> str_v3 = split(str_v2[0], '_');

        col_ind.insert(std::stoi(str_v3[0]));
        row_ind.insert(std::stoi(str_v3[1]));
        
    }

    const int row = (*row_ind.end());
    const int col = (*col_ind.end());

    const int tilesize = 256;

    string path_last = path_base + "/" + to_string(col-1) + "_" + to_string(row-1) + ".jpeg";
    cv::Mat img_last = cv::imread(path_last);

    uint img_width = (col-1) * tilesize + img_last.cols;
    uint img_height = (row-1) * tilesize + img_last.rows;


    cv::Mat merge_all(img_height, img_width, CV_8UC3);

    #pragma omp parallel for collapse(2)
    for(int i=0; i<row; i++){
        for(int j=0; j<col; j++){
            
            std::string img_path = path_base + "/" + to_string(j) + "_" + to_string(i) + ".jpeg";
            cv::Mat img = cv::imread(img_path);

            int cols_w = img.cols>= tilesize ? tilesize : img.cols;
            int rows_h = img.rows>=tilesize ? tilesize : img.rows;

            int row_start = i*tilesize;
            int col_start = j*tilesize; 

            img(cv::Rect(0,0, cols_w, rows_h)).copyTo(merge_all(cv::Rect(col_start, row_start, cols_w, rows_h))); 

        }

    }

    return merge_all;
}


vector<int> sort_indexes_greater(const vector<float> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),[&v](int i1, int i2) {return v[i1] > v[i2];});

  return idx;
}


vector<string> getFiles(const std::string& dir){
    
    std::vector<std::string> files;
    std::vector<int> inds;

    std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        if( strcmp( dirent_ptr->d_name , "." ) == 0 || strcmp( dirent_ptr->d_name , "..") == 0)
            continue;
        files.push_back(std::string(dirent_ptr->d_name));
        inds.push_back(std::stoi(std::string(dirent_ptr->d_name)));
    }

    vector<int> indx = sort_indexes(inds);

    vector<string> files_out;
    for(size_t i=0; i<indx.size(); i++){
        files_out.push_back(files[indx[i]]);
    }

    return files_out;

}

vector<int> sort_indexes_less(const vector<int> &v) {

  // initialize original index locations
  vector<int> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),[&v](int i1, int i2) {return v[i1] < v[i2];});

  return idx;
}

void swap_folder_name(const std::string& dir) {
    
    std::vector<std::string> files;
    std::vector<int> inds;

    int max_tmp = 0; 
    int re_ind;
    std::string name_tmp = "name_tmp";

    std::shared_ptr<DIR> directory_ptr(opendir(dir.c_str()), [](DIR* dir){ dir && closedir(dir); });
    struct dirent *dirent_ptr;

    while ((dirent_ptr = readdir(directory_ptr.get())) != nullptr) {
        if( strcmp( dirent_ptr->d_name , "." ) == 0 || strcmp( dirent_ptr->d_name , "..") == 0)
            continue;
        files.emplace_back(dirent_ptr->d_name);
        //inds.push_back(std::stoi(std::string(dirent_ptr->d_name)));
        inds.push_back(atoi(dirent_ptr->d_name));
    }

    vector<int> indx = sort_indexes_less(inds);

    for(int ind : inds){
        if(ind>max_tmp){
            max_tmp = ind;
        }
    }

    
    for(size_t i=0; i<indx.size()/2; i++){

        re_ind = max_tmp - inds[indx[i]];

        if (rename((dir+"/"+files[indx[i]]).c_str(), (dir+"/"+name_tmp).c_str())){
	        std::cout << "Error: " << strerror(errno) << std::endl;
	    }
        
        if (rename((dir+"/"+std::to_string(re_ind)).c_str(), (dir+"/"+std::to_string(inds[indx[i]])).c_str())){
	        std::cout << "Error: " << strerror(errno) << std::endl;
	    }

        if (rename((dir+"/"+name_tmp).c_str(), (dir+"/"+std::to_string(re_ind)).c_str())){
	        std::cout << "Error: " << strerror(errno) << std::endl;
	    }
    }

}


#endif