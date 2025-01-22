

# add_definitions(-O3 -pthread)


cmake_minimum_required(VERSION 3.9)


set(PROJECT overlap_cuda_WSI)
set(PROJECT1 tiff_overlap_cuda_WSI)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE "Release")

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-12/include)
link_directories(/usr/local/cuda-12/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -fopenmp -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins ${PROJECT_SOURCE_DIR}/poscal.cu)
target_link_libraries(myplugins nvinfer cudart)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(TIFF REQUIRED)


set(SOURCES
ihc_wsi_overlay_cuda.cpp
)

set(SOURCES1
pyrmstiff_ihc_wsi_overlay_cuda.cpp
tiffReader.cpp
tiffWriter.cpp
)


include_directories(
${OpenCV_INCLUDES}
)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH TRUE)


add_executable(${PROJECT} ${SOURCES})
add_executable(${PROJECT1} ${SOURCES1})


target_link_libraries(${PROJECT}
	${OpenCV_LIBS} 
	${OpenMP_CXX_FLAGS}
    nvinfer
    cudart
    myplugins
)

target_link_libraries(${PROJECT1}
	${OpenCV_LIBS} 
	${OpenMP_CXX_FLAGS}
    ${TIFF_LIBRARIES}
    nvinfer
    cudart
    myplugins
)


add_definitions(-O3 -pthread)

