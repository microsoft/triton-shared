#!/bin/bash
set -e
set -x

# current directory if argument not provided
input_path="${1:-.}"

CURR=$(realpath "$input_path")

# Check if the path exists
if [ ! -e "$CURR" ]; then
  echo "Error: Path '$CURR' does not exist."
  exit 1
fi

echo "Using path: $CURR"
cd $CURR

# set up venv
python3 -m venv ${CURR}/venv
source ${CURR}/venv/bin/activate

# clone triton-shared
# needs to be cloned before building llvm for the hash
export TRITON_PLUGIN_DIRS="${CURR}/triton_shared"
git clone --recurse-submodules https://github.com/timthlu/triton-shared.git triton_shared

cd triton_shared
git checkout san_clean

cd triton

python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist pybind11 setuptools torch==2.7 # need torch 2.7 for triton 3.3.0
# sudo apt-get update -y
sudo apt-get install -y ccache clang lld
# pip3 install ninja cmake wheel pytest pybind11 setuptools

cd "${CURR}"

# build llvm
TRITON_SHARED_DIR="${CURR}/triton_shared"
LLVM_HASH_FILE="${TRITON_SHARED_DIR}/triton/cmake/llvm-hash.txt"
if [ ! -e "${LLVM_HASH_FILE}" ]; then
  print_error "${LLVM_HASH_FILE} does not exist"
fi

BUILD_DIR="${CURR}/llvm-build"
INSTALL_DIR="${CURR}/llvm-install"
SOURCE_DIR="${CURR}/llvm-project"
LLVM_SOURCE="${SOURCE_DIR}/llvm"
PROJECTS="clang;compiler-rt;mlir;openmp"

if [ -d "${BUILD_DIR}" ]; then
  rm -rf "${BUILD_DIR}"
fi

if [ -d "${INSTALL_DIR}" ]; then
  rm -rf "${INSTALL_DIR}"
fi

mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

if [ ! -e "${SOURCE_DIR}/.git" ]; then
  git clone https://github.com/llvm/llvm-project "${SOURCE_DIR}"
fi
cd "${SOURCE_DIR}"

LLVM_HASH=$(cat "${LLVM_HASH_FILE}")

# CURRENT_BRANCH=$(git branch | awk '{if($1 == "*"){print $2}}')
# if [ "$CURRENT_BRANCH" = "$BRANCH" ]; then
#  echo "Already in ${BRANCH}, skip 'git checkout'"
# else
#  git checkout $BRANCH
# fi

git checkout ${LLVM_HASH}

export CXXFLAGS="-Wno-unused-command-line-argument $CXXFLAGS" 
export CFLAGS="-Wno-unused-command-line-argument $CFLAGS" 

cd $BUILD_DIR
cmake -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang               \
  -DCMAKE_CXX_COMPILER=clang++           \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR    \
  -DLLVM_ENABLE_PROJECTS=$PROJECTS       \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  $LLVM_SOURCE

echo "Installing to: $INSTALL_DIR"
ninja -j$(nproc --all) -l$(nproc --all) || exit -1
ninja -j$(nproc --all) -l$(nproc --all) install
echo "Install succeeded\n"

# prepare for triton_shared build
export PATH="${INSTALL_DIR}/bin:${PATH}"
which clang
rm -rf ~/.triton
rm -rf "${TRITON_SHARED_DIR}/triton/build"

# build triton-shared with the custom LLVM
cd "${TRITON_SHARED_DIR}/triton"

export LLVM_BUILD_DIR="${BUILD_DIR}"
LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib LLVM_SYSPATH=$LLVM_BUILD_DIR TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=false python3 -m pip install --no-build-isolation -vvv '.[tests]'

echo "Build completed"