# executes a triton program with sanitizers enabled/disabled
# USAGE: run_triton.sh <sanitizer type> python triton_program.py
# <sanitizer type> can be "asan" or omitted
# assume that source setup_runtime_for_sanitizers was run before this
# use LLVM_BINARY_DIR to obtain the locations of the .so files used for LD_PRELOAD

sanitizer_type=$1
env_args=""

# echo "$*"
llvm_install_dir="$(dirname "${LLVM_BINARY_DIR}")"

if [ "${sanitizer_type}" = "asan" ]; then
    asan_dir="$(find "$llvm_install_dir" -type f -name "libclang_rt.asan.so" | head -n 1)"

    env_args="LD_PRELOAD=\"$asan_dir\" \
TRITON_ALWAYS_COMPILE=1 \
TRITON_SHARED_SANITIZER_TYPE=\"asan\" \
ASAN_OPTIONS=\"detect_leaks=0\""
    
    # shift command line arguments to the left by 1 to account for "asan"
    shift 1
fi

# invoke python function
echo "Running command: ${env_args} $*"
eval "${env_args} $*"