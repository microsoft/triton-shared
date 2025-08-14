# using a custom llvm install, builds triton_shared

#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <path to llvm-build directory> <path to Python venv>"
  exit 0
fi

TRITON_SHARED_PATH="$(realpath "$(dirname "$0")/..")"
LLVM_BUILD_PATH="$(realpath "$1")"
VENV_PATH="$(realpath "$2")"

# check if the path exists
if [ ! -d "${LLVM_BUILD_PATH}" ]; then
  echo "Error: Path \"${LLVM_BUILD_PATH}\" is not a valid directory for LLVM build."
  exit 1
fi

if [ ! -d "${VENV_PATH}" ]; then
  echo "Error: Path \"${VENV_PATH}\" is not a valid directory for Python virtual environment."
  exit 1
fi


if [ ! -e "${VENV_PATH}/bin/activate" ]; then
  echo "Error: \"${VENV_PATH}/bin/activate\" does not exist"
  exit 1
fi

# activate Python virtual environment
. ${VENV_PATH}/bin/activate

# use '~/.triton-san' as the cache folder
export TRITON_HOME="$(realpath "~/.triton-san")"
echo "Use \"${TRITON_HOME}\" as triton-san's cache"
if [ -e "${TRITON_HOME}" ]; then
  rm -rf "${TRITON_HOME}"
fi

# build triton-shared with the custom LLVM
cd "${TRITON_SHARED_PATH}/triton"
export TRITON_PLUGIN_DIRS="${TRITON_SHARED_PATH}"
export LLVM_BUILD_DIR="${LLVM_BUILD_PATH}"
LLVM_INCLUDE_DIRS="${LLVM_BUILD_DIR}/include" LLVM_LIBRARY_DIR="${LLVM_BUILD_DIR}/lib" LLVM_SYSPATH="${LLVM_BUILD_DIR}" TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=false python3 -m pip install --no-build-isolation -vvv '.[tests]'

echo "triton-shared build completed"
