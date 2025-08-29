option(TRITON_SHARED_BUILD_CPU_BACKEND "Build triton-shared CPU backend" ON)
option(TRITON_SHARED_BUILD_TRITON_SAN "Build the triton-san sanitizer suite" ON)

set(TRITON_SHARED_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_SHARED_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files
include_directories(${Python3_INCLUDE_DIR})
include_directories(${pybind11_INCLUDE_DIR})

function(add_symlink source_file link_name)
    if (EXISTS ${source_file})
        file(CREATE_LINK ${source_file} ${link_name} SYMBOLIC)
    else()
        message(WARNING "${source_file} not found. Skipping symlink creation for ${link_name}.")
    endif()
endfunction()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools/triton-shared-opt)

if (TRITON_SHARED_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonShared ${CMAKE_CURRENT_SOURCE_DIR}/triton_shared.cc LINK_LIBS TritonSharedAnalysis TritonToLinalg TritonTilingExtIR)
    target_link_libraries(TritonShared PRIVATE Python3::Module pybind11::headers)
endif()

# Add symlinks to selected pytest files and the clang-format setting in triton. The tests are imported into triton-shared’s test folder to 
# run under triton-shared's conftest configuration, and the clang-format link ensures consistent code style enforcement across both repositories.
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "python" "examples" "test_core.py" OUTPUT_VARIABLE TRITON_SHARED_TEST_CORE)
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "python" "examples" "test_annotations.py" OUTPUT_VARIABLE TRITON_SHARED_TEST_ANNOTATIONS)
cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR ".clang-format" OUTPUT_VARIABLE TRITON_SHARED_CLANG_FORMAT_SETTING)
cmake_path(APPEND CMAKE_SOURCE_DIR "python" "test" "unit" "language" "test_core.py" OUTPUT_VARIABLE TRITON_TEST_CORE)
cmake_path(APPEND CMAKE_SOURCE_DIR "python" "test" "unit" "language" "test_annotations.py" OUTPUT_VARIABLE TRITON_TEST_ANNOTATIONS)
cmake_path(APPEND CMAKE_SOURCE_DIR ".clang-format" OUTPUT_VARIABLE TRITON_CLANG_FORMAT_SETTING)

add_symlink(${TRITON_TEST_CORE} ${TRITON_SHARED_TEST_CORE})
add_symlink(${TRITON_TEST_ANNOTATIONS} ${TRITON_SHARED_TEST_ANNOTATIONS})
add_symlink(${TRITON_CLANG_FORMAT_SETTING} ${TRITON_SHARED_CLANG_FORMAT_SETTING})