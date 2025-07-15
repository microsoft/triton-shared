# executes a triton program with sanitizers enabled/disabled
# assume that source setup_runtime_for_sanitizers was run before this
# use LLVM_BINARY_DIR to obtain the locations of the .so files used for LD_PRELOAD

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <sanitizer type> python program.py. <sanitizer type> can be \"asan\""
    exit 1
fi

sanitizer_type=$1

if [ "$sanitizer_type" != "asan" ]; then
    echo "Error: Unsupported <sanitizer type> $sanitizer_type. Usage: $0 <sanitizer type> python program.py. <sanitizer type> can be \"asan\""
    exit 1
fi

env_args=""

llvm_install_dir="$(dirname "${LLVM_BINARY_DIR}")"

if [ "${sanitizer_type}" = "asan" ]; then
    # find path to asan shared library
    asan_dir="$(find "$llvm_install_dir" -type f -name "libclang_rt.asan.so")"

    if [ -z "$asan_dir" ]; then
        echo "Error: unable to find libclang_rt.asan.so in $llvm_install_dir"
        exit 1
    fi

    count=$(echo "$asan_dir" | wc -l)

    if [ "$count" -gt 1 ]; then
        echo "Error: multiple libclang_rt.asan.so found in $llvm_install_dir"
        echo "$asan_dir"
        exit 1
    fi

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