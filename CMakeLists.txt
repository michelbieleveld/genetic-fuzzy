cmake_minimum_required (VERSION 2.6)
project (Thesis)
set(CMAKE_BUILD_TYPE Release)

find_package(GDAL)
include_directories(${GDAL_INCLUDE_DIRS})

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
include_directories(${MPI_INCLUDE_PATH})

# The version number.
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
include_directories( BEFORE tclap-1.2.1/include)
include_directories( BEFORE cereal-1.1.2/include)
include_directories( BEFORE ${CMAKE_CURRENT_BINARY_DIR})
 
configure_file("include/thesis.h.in" "${PROJECT_BINARY_DIR}/thesis.h")
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
include_directories(include)

file(GLOB SOURCES "src/*.cpp")
 
# add the executable
add_executable(Thesis ${SOURCES})
set_property(TARGET Thesis PROPERTY CXX_STANDARD 14)
TARGET_LINK_LIBRARIES( Thesis ${GDAL_LIBRARIES} ${MPI_LIBRARIES} -lstdc++fs)

