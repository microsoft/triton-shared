cmake_minimum_required(VERSION 3.20)
project(sanitizer-attributes)

#===============================================================================
# 1. LOAD LLVM CONFIGURATION
#===============================================================================
# Set this to a valid LLVM installation dir
set(LT_LLVM_INSTALL_DIR "/workspace/triton_shared/triton/llvm-project/install" CACHE PATH "LLVM installation directory")
set(LLVM_ENABLE_ABI_BREAKING_CHECKS ON)

# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

find_package(LLVM REQUIRED CONFIG)
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 19)
 message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 19 or above")
endif()

# HelloWorld includes headers from LLVM - update the include paths accordingly
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})

#===============================================================================
# 2. LLVM-TUTOR BUILD CONFIGURATION
#===============================================================================
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# LLVM is normally built without RTTI. Be consistent with that.
# if(NOT LLVM_ENABLE_RTTI)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
# endif()

#===============================================================================
# 3. ADD THE TARGET
#===============================================================================

# set(CMAKE_CXX_VISIBILITY_PRESET hidden)
# set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)
# add_definitions(-fvisibility=default)

add_library(SanitizerAttributes SHARED SanitizerAttributes.cpp)

# Allow undefined symbols in shared objects on Darwin (this is the default
# behaviour on Linux)
target_link_libraries(SanitizerAttributes "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")

# This is set to -fvisibility=hidden in the top level CMake file
# which causes the llvmGetPassPluginInfo symbol to be hidden and
# an "entry point not found" error. Reset it just for this target
if(NOT MSVC)
  target_compile_options(SanitizerAttributes PRIVATE -fvisibility=default -fno-rtti)
endif()