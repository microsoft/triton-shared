# Install triton-san in the designated directory

#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <desired path for triton-san installation> <path to venv> <path to LLVM install dir>"
  exit 1
fi

PARENT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_SAN_PATH="$(realpath "${PARENT_FOLDER}/..")"
TRITON_SHARED_PATH="$(realpath "${PARENT_FOLDER}/../..")"
TRITON_SAN_INSTALL_DIR="$(realpath "$1")"
VENV_PATH="$(realpath "$2")"
LLVM_INSTALL_DIR="$(realpath "$3")"

# include utility functions
source "${PARENT_FOLDER}/utility.inc"

if [ ! -d "${TRITON_SAN_INSTALL_DIR}" ]; then
  print_error_and_exit "Path \"${TRITON_SAN_INSTALL_DIR}\" is not a valid directory for installing triton-san."
fi

TRITON_SAN_INSTALL_ROOT="${TRITON_SAN_INSTALL_DIR}/triton-san"

if [ -e "${TRITON_SAN_INSTALL_ROOT}" ]; then
  rm -rf "${TRITON_SAN_INSTALL_ROOT}"
fi

mkdir -p "${TRITON_SAN_INSTALL_ROOT}"

# Locate required executables and objects
locate_file "triton-shared-opt" "TRITON_SHARED_OPT_PATH" "${TRITON_SHARED_PATH}"
locate_file "libclang_rt.asan.so" "ASAN_OBJ_PATH" "${LLVM_INSTALL_DIR}"
locate_file "libclang_rt.tsan.so" "TSAN_OBJ_PATH" "${LLVM_INSTALL_DIR}"
locate_file "libarcher.so" "ARCHER_OBJ_PATH" "${LLVM_INSTALL_DIR}"

# Generate the suppression file
cp "${TRITON_SAN_PATH}/template/tsan_suppression.in" "${TRITON_SAN_INSTALL_ROOT}/tsan_suppression.txt"

# Generate triton-san helper script
sed -e "s#@ROOT@#${TRITON_SAN_INSTALL_ROOT}#"                    \
    -e "s#@TRITON_SHARED_OPT_PATH@#${TRITON_SHARED_OPT_PATH}#"   \
    -e "s#@ASAN_OBJ_PATH@#${ASAN_OBJ_PATH}#"                     \
    -e "s#@TSAN_OBJ_PATH@#${TSAN_OBJ_PATH}#"                     \
    -e "s#@ARCHER_OBJ_PATH@#${ARCHER_OBJ_PATH}#"                 \
    -e "s#@VENV_PATH@#${VENV_PATH}#"                             \
    -e "s#@LLVM_INSTALL_DIR@#${LLVM_INSTALL_DIR}#"               \
    "${TRITON_SAN_PATH}/template/triton-san.in" > "${TRITON_SAN_INSTALL_ROOT}/triton-san"

chmod +x "${TRITON_SAN_INSTALL_ROOT}/triton-san"

# Copy examples
cp -r "${TRITON_SAN_PATH}/examples" "${TRITON_SAN_INSTALL_ROOT}/examples"

info_message=("triton-san has been installed into ${TRITON_SAN_INSTALL_ROOT} successfully."
              "To optionally enable global access, add triton-san to your PATH environment variable using:"
              'export PATH="${PATH}:'"${TRITON_SAN_INSTALL_ROOT}\"")
print_info "${info_message[@]}"