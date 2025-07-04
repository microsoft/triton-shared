# users need to call source on this file before running triton programs using run_triton
# this is a one time setup of required environment variables

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <existing path to venv> <existing path to llvm install dir> <existing path to triton shared>"
  exit 1
fi

VENV_PATH="$(realpath "$1")"
LLVM_INSTALL_PATH="$(realpath "$2")"
TRITON_SHARED_PATH="$(realpath "$3")"

# check if the path exists
if [ ! -e "$VENV_PATH" ]; then
  echo "Error: Path '$VENV_PATH' does not exist."
  exit 1
fi

if [ ! -e "$LLVM_INSTALL_PATH" ]; then
  echo "Error: Path '$LLVM_INSTALL_PATH' does not exist."
  exit 1
fi

if [ ! -e "$TRITON_SHARED_PATH" ]; then
  echo "Error: Path '$TRITON_SHARED_PATH' does not exist."
  exit 1
fi

source ${VENV_PATH}/bin/activate

export PATH="${LLVM_INSTALL_PATH}/bin:${VENV_PATH}/bin:${PATH}"
export LLVM_BINARY_DIR="${LLVM_INSTALL_PATH}/bin"

export TRITON_SHARED_OPT_PATH="$(find "$TRITON_SHARED_PATH" -type f -name "triton-shared-opt" | head -n 1)"
# export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_DIR}/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
