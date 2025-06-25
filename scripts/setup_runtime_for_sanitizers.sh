# users need to call source setup.sh before running triton
# contains required environment variables

# current directory if argument not provided
input_path="${1:-.}"

CURR=$(realpath "$input_path")

# Check if the path exists
if [ ! -e "$CURR" ]; then
  echo "Error: Path '$CURR' does not exist."
  exit 1
fi

echo "Using path: $CURR"

source ${CURR}/venv/bin/activate

TRITON_SHARED_DIR="${CURR}/triton_shared"
INSTALL_DIR="${CURR}/llvm-install"

export PATH="${INSTALL_DIR}/bin:${CURR}/venv/bin:${PATH}"
export LLVM_BINARY_DIR="${INSTALL_DIR}/bin"
export TRITON_SHARED_OPT_PATH="${TRITON_SHARED_DIR}/triton/build/cmake.linux-x86_64-cpython-3.12/third_party/triton_shared/tools/triton-shared-opt/triton-shared-opt"
# export LD_LIBRARY_PATH="${INSTALL_DIR}/lib/x86_64-unknown-linux-gnu" # for openmp
