cmake_minimum_required(VERSION 2.8)
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

project(freespace_detect)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories("/usr/local/include/eigen3")

add_executable(freespace_detect main.cpp FreeSpaceDetector.cpp)
target_link_libraries(freespace_detect ${PCL_LIBRARIES} ${OpenCV_LIBS})