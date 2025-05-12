option(TRITON_SHARED_BUILD_CPU_BACKEND "Build triton-shared CPU backend" ON)

set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools/triton-shared-opt)

if (TRITON_SHARED_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonShared ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
endif()
