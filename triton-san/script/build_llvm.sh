# Build a custom llvm with sanitizer and openmp supports.
# This script uses the same version of LLVM that is recorded in llvm-hash.txt
# and installs the required LLVM projects for triton-san (compiler-rt, openmp).

#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <path to Triton source directory> <desired path for LLVM installation>"
  exit 1
fi

PARENT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_PATH="$(realpath "$1")"
LLVM_PATH="$(realpath "$2")"

# include utility functions
source "${PARENT_FOLDER}/utility.inc"

print_info "Installing LLVM to path: ${LLVM_PATH}."
cd $LLVM_PATH

LLVM_HASH_FILE="${TRITON_PATH}/cmake/llvm-hash.txt"
if [ ! -e "${LLVM_HASH_FILE}" ]; then
  print_error_and_exit "${LLVM_HASH_FILE} does not exist."
fi

LLVM_BUILD_DIR="${LLVM_PATH}/llvm-build"
LLVM_INSTALL_DIR="${LLVM_PATH}/llvm-install"
LLVM_SOURCE_DIR="${LLVM_PATH}/llvm-project"
LLVM_SOURCE="${LLVM_SOURCE_DIR}/llvm"

# compiler-rt and clang are the sanitizer-specific LLVM projects
# openmp is used for parallelizing the triton grid for ThreadSanitizer (TSan)
LLVM_PROJECTS="clang;compiler-rt;openmp;mlir;lld"

# these are the targets supported by the Triton language
# Triton's build script for LLVM uses these exact targets
# see https://github.com/triton-lang/triton/blob/main/scripts/build-llvm-project.sh
LLVM_TARGETS="Native;NVPTX;AMDGPU"

mkdir -p "${LLVM_BUILD_DIR}"
mkdir -p "${LLVM_INSTALL_DIR}"

git clone https://github.com/llvm/llvm-project "${LLVM_SOURCE_DIR}"

LLVM_HASH=$(cat "${LLVM_HASH_FILE}")

cd "${LLVM_SOURCE_DIR}"
git checkout ${LLVM_HASH}

if ! git merge-base --is-ancestor 62ff9ac HEAD; then
  echo "cherry pick commit 62ff9ac to avoid OpenMP build failure"
  git cherry-pick 62ff9ac
fi

export CXXFLAGS="-Wno-unused-command-line-argument $CXXFLAGS" 
export CFLAGS="-Wno-unused-command-line-argument $CFLAGS" 

cd "$LLVM_BUILD_DIR"
cmake -GNinja -DCMAKE_BUILD_TYPE=Release        \
  -DCMAKE_C_COMPILER=clang                      \
  -DCMAKE_CXX_COMPILER=clang++                  \
  -DCMAKE_INSTALL_PREFIX=$LLVM_INSTALL_DIR      \
  -DLLVM_ENABLE_PROJECTS=$LLVM_PROJECTS         \
  -DLLVM_TARGETS_TO_BUILD=$LLVM_TARGETS         \
  $LLVM_SOURCE

ninja -j$(nproc --all) -l$(nproc --all) || exit -1
ninja -j$(nproc --all) -l$(nproc --all) install

print_info "LLVM installation succeeded."
