# using a custom llvm install, builds triton_shared

#!/bin/bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <path to llvm-build directory> <path to Python venv>"
  exit 1
fi

PARENT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_SHARED_PATH="$(realpath "${PARENT_FOLDER}/../..")"
LLVM_BUILD_PATH="$(realpath "$1")"
VENV_PATH="$(realpath "$2")"

# include utility functions
source "${PARENT_FOLDER}/utility.inc"

# check if the path exists
if [ ! -d "${LLVM_BUILD_PATH}" ]; then
  print_error_and_exit "Path \"${LLVM_BUILD_PATH}\" is not a valid directory for LLVM build."
fi

if [ ! -d "${VENV_PATH}" ]; then
  print_error_and_exit "Path \"${VENV_PATH}\" is not a valid directory for Python virtual environment."
fi


if [ ! -e "${VENV_PATH}/bin/activate" ]; then
  print_error_and_exit "\"${VENV_PATH}/bin/activate\" does not exist"
fi

# activate Python virtual environment
. ${VENV_PATH}/bin/activate

# use '~/.triton-san/.triton' as the cache folder
export TRITON_HOME="$(readlink -f "$HOME/.triton-san")"
print_info "Use \"${TRITON_HOME}\" as triton-san's cache."
if [ -e "${TRITON_HOME}" ]; then
  warning_msg=("The path ${TRITON_HOME} already exists and will be overwritten."
               "Please ensure that only one TritonSan installation is present.")
  print_warning "${warning_msg[@]}"
  rm -rf "${TRITON_HOME}"
fi
mkdir -p "${TRITON_HOME}"

# build triton-shared with the custom LLVM
cd "${TRITON_SHARED_PATH}/triton"
export TRITON_PLUGIN_DIRS="${TRITON_SHARED_PATH}"
export LLVM_BUILD_DIR="${LLVM_BUILD_PATH}"
LLVM_INCLUDE_DIRS="${LLVM_BUILD_DIR}/include" LLVM_LIBRARY_DIR="${LLVM_BUILD_DIR}/lib" LLVM_SYSPATH="${LLVM_BUILD_DIR}" TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_WITH_CCACHE=false python3 -m pip install --no-build-isolation -vvv '.[tests]'

print_info "triton-shared and triton build completed."