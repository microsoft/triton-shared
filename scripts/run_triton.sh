# assume that setup was run before this
# use LLVM_BINARY_DIR to obtain the locations of the .so files used for LD_PRELOAD

sanitizer_type=$1
env_args=""

# echo "$*"

if [ "${sanitizer_type}" = "asan" ]; then
    env_args="LD_PRELOAD=\"$(dirname "${LLVM_BINARY_DIR}")/lib/clang/21/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so\" \
        TRITON_ALWAYS_COMPILE=1 \
        SANITIZER_TYPE=\"asan\" \
        ASAN_OPTIONS=\"detect_leaks=0\""
    
    # shift command line arguments to the left by 1 to account for "asan"
    shift 1
fi

# invoke python function
echo "Running command: ${env_args} $*"
eval "${env_args} $*"