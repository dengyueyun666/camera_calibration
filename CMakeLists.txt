cmake_minimum_required(VERSION 3.4)

project(camera_calibration)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -O3 -pthread" )

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

add_executable(main 
    src/main.cpp
    src/util.cpp
)
target_link_libraries(main  
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
)
