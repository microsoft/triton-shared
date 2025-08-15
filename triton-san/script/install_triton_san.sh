# Install triton-san in the designated directory

#!/bin/bash
set -e

function locate_file {
  FILE_NAME="$1"
  FILE_KEY="$2"
  SEARCH_PATH="$(realpath "$3")"
  if [ ! -d "${SEARCH_PATH}" ]; then
    echo "Error: ${SEARCH_PATH} is not a valid directory"
    return 1
  fi

  LOCATED_FILE_PATH="$(find "${SEARCH_PATH}" -type f -name "${FILE_NAME}")"

  if [ -z "${LOCATED_FILE_PATH}" ]; then
    echo "Error: unable to find ${FILE_NAME} in ${SEARCH_PATH}"
    return 1
  fi

  COUNT=$(echo "${LOCATED_FILE_PATH}" | wc -l)

  if [ ${COUNT} -gt 1 ]; then
    echo "Error: multiple ${FILE_PATH} found in ${SEARCH_PATH}"
    echo "${LOCATED_FILE_PATH}"
    return 1
  fi

  export ${FILE_KEY}_PATH="${LOCATED_FILE_PATH}"
  return 0
}

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <desired path for triton-san installation> <path to venv> <path to LLVM install dir>"
  exit 1
fi

TRITON_SAN_INSTALL_DIR="$(realpath "$1")"
if [ ! -d "${TRITON_SAN_INSTALL_DIR}" ]; then
  echo "Error: Path \"${LLVM_BUILD_PATH}\" is not a valid directory for installing triton-san."
  exit 1
fi

TRITON_SAN_INSTALL_ROOT="${TRITON_SAN_INSTALL_DIR}/triton-san"

if [ -e "${TRITON_SAN_INSTALL_ROOT}" ]; then
  rm -rf "${TRITON_SAN_INSTALL_ROOT}"
fi

mkdir -p "${TRITON_SAN_INSTALL_ROOT}"

SCRIPT_FOLDER="$(realpath "$(dirname "$0")/..")"
TRITON_SHARED_PATH="$(realpath "$(dirname "$0")/../..")"
VENV_PATH="$(realpath "$2")"
LLVM_INSTALL_DIR="$(realpath "$3")"

# Locate required executables and objects
if ! locate_file "triton-shared-opt" "TRITON_SHARED_OPT" "${TRITON_SHARED_PATH}"; then
  exit 1
fi

if ! locate_file "libclang_rt.asan.so" "ASAN_OBJ" "${LLVM_INSTALL_DIR}"; then
  exit 1
fi

if ! locate_file "libclang_rt.tsan.so" "TSAN_OBJ" "${LLVM_INSTALL_DIR}"; then
  exit 1
fi

if ! locate_file "libarcher.so" "ARCHER_OBJ" "${LLVM_INSTALL_DIR}"; then
  exit 1
fi


# Generate the suppression file
cp "${SCRIPT_FOLDER}/template/tsan_suppression.in" "${TRITON_SAN_INSTALL_ROOT}/tsan_suppression.txt"

# Generate triton-san helper script
sed -e "s#@ROOT@#${TRITON_SAN_INSTALL_ROOT}#"                    \
    -e "s#@TRITON_SHARED_OPT_PATH@#${TRITON_SHARED_OPT_PATH}#"   \
    -e "s#@ASAN_OBJ_PATH@#${ASAN_OBJ_PATH}#"                     \
    -e "s#@TSAN_OBJ_PATH@#${TSAN_OBJ_PATH}#"                     \
    -e "s#@ARCHER_OBJ_PATH@#${ARCHER_OBJ_PATH}#"                 \
    -e "s#@VENV_PATH@#${VENV_PATH}#"                             \
    -e "s#@LLVM_INSTALL_DIR@#${LLVM_INSTALL_DIR}#"               \
    "${SCRIPT_FOLDER}/template/triton-san.in" > "${TRITON_SAN_INSTALL_ROOT}/triton-san"

chmod +x "${TRITON_SAN_INSTALL_ROOT}/triton-san"

# Copy examples
cp -r "${SCRIPT_FOLDER}/examples" "${TRITON_SAN_INSTALL_ROOT}/examples"

echo "triton-san has been installed into ${TRITON_SAN_INSTALL_ROOT} successfully"
echo "To optionally enable global access, add triton-san to your PATH environment variable using:"
echo ' export PATH="${PATH}'":${TRITON_SAN_INSTALL_ROOT}\""
