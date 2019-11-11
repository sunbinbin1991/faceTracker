cmake_minimum_required(VERSION 2.8)

project(tracker)
  
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_CXX_STANDARD 14)
if(CMAKE_COMPILER_IS_GNUCXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wextra -pedantic -Wno-long-long")
endif()

#SET(CMAKE_BUILD_TYPE DEBUG)  
#SET(CMAKE_BUILD_TYPE RELEASE)  

#find opencv lib
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/x64/vc14/lib)
set(OpenCV_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/opencv/include) 
find_package(OpenCV REQUIRED)

#use omp to speed up
if (OPENMP_FOUND)
    message("OpenMP found!")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# add 3rdparty
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/ncnn/debug)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/concurrentqueue)

#add common ref
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#add example
add_subdirectory(example)

# add src 
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/detect)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)




