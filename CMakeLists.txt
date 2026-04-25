cmake_minimum_required(VERSION 3.15)

project(ACOEdgeProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD 99)

find_package(OpenMP REQUIRED)

# Common source files (shared across executables)
set(COMMON_SOURCES
    tigr.c
)

# ACO Edge executable
add_executable(aco_edge aco_edge.cpp ${COMMON_SOURCES})

# Sobel executable
add_executable(sobel sobel.cpp ${COMMON_SOURCES})

# No-edge executable
add_executable(noedge noedge.cpp ${COMMON_SOURCES})

# Platform-specific libraries
if(WIN32)
    set(PLATFORM_LIBS d3d9)
elseif(APPLE)
    set(PLATFORM_LIBS "-framework OpenGL -framework Cocoa")
else()
    set(PLATFORM_LIBS GL X11)
endif()

# Link everything properly
foreach(target aco_edge sobel noedge)
    target_link_libraries(${target} PRIVATE ${PLATFORM_LIBS} OpenMP::OpenMP_CXX)
endforeach()