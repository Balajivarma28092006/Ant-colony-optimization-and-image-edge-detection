cmake_minimum_required(VERSION 3.15)

project(ACOEdgeProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    aco_edge.cpp
    tigr.c
)

# Create executable
add_executable(aco_edge ${SOURCES})

# Window link libraries for TIGR
if(WIN32)
    target_link_libraries(aco_edge PRIVATE d3d9)
elseif(APPLE)
    target_link_libraries(aco_edge PRIVATE "-framework OpenGL -framework Cocoa")
else()
    target_link_libraries(aco_edge PRIVATE GL X11)
endif()

target_link_libraries(aco_edge PRIVATE OpenMP::OpenMP_CXX)
