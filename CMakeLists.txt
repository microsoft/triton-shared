include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
add_triton_plugin(TritonShared ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
