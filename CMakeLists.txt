cmake_minimum_required(VERSION 3.4)

project(camera_calibration)

set( CMAKE_BUILD_TYPE Release )

# 设置编译选项
# 允许c++11标准、O3优化、多线程。match选项可避免一些cpu上的问题
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

find_package(OpenCV 3 REQUIRED)
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

add_executable(main 
    src/main.cpp
    src/util.cpp
)
target_link_libraries(main  
    ${OpenCV_LIBS}
)
