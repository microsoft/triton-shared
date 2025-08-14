# Install triton-san in the designated directory

#!/bin/bash
set -e

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <desired path for triton-san installation> <path to venv> <path to LLVM install dir>"
  exit 0
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

SCRIPT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_SHARED_PATH="$(realpath "$(dirname "$0")/..")"
VENV_PATH="$(realpath "$2")"
LLVM_INSTALL_DIR="$(realpath "$3")"

# Locate triton-shared-opt
TRITON_SHARED_OPT_PATH="$(find "${TRITON_SHARED_PATH}" -type f -executable -name "triton-shared-opt")"

if [ -z "${TRITON_SHARED_OPT_PATH}" ]; then
  echo "Error: unable to find triton-shared-opt in ${TRITON_SHARED_PATH}"
  exit 1
fi

count=$(echo "${TRITON_SHARED_OPT_PATH}" | wc -l)

if [ "$count" -gt 1 ]; then
  echo "Error: multiple triton-shared-opt found in ${TRITON_SHARED_PATH}"
  echo "${TRITON_SHARED_OPT_PATH}"
  exit 1
fi

# Generate the suppression file
cat <<TSAN_SUPPRESSION > "${TRITON_SAN_INSTALL_ROOT}/tsan_suppression.txt"
called_from_lib:libomp.so
called_from_lib:libtorch_python.so
called_from_lib:libtorch_cpu.so
called_from_lib:libtorch_cuda.so"
TSAN_SUPPRESSION

# Generate triton-san helper script
sed -e "s#@ROOT@#${TRITON_SAN_INSTALL_ROOT}#"                    \
    -e "s#@TRITON_SHARED_OPT_PATH@#${TRITON_SHARED_OPT_PATH}#"   \
    -e "s#@VENV_PATH@#${VENV_PATH}#"                             \
    -e "s#@LLVM_INSTALL_DIR@#${LLVM_INSTALL_DIR}#"               \
    "${SCRIPT_FOLDER}/template/triton-san.in" > "${TRITON_SAN_INSTALL_ROOT}/triton-san"

chmod 775 "${TRITON_SAN_INSTALL_ROOT}/triton-san"

# Copy examples
cp -r "${SCRIPT_FOLDER}/example" "${TRITON_SAN_INSTALL_ROOT}/example"

echo "triton-san has been installed into ${TRITON_SAN_INSTALL_ROOT} successfully"
echo "To optionally enable global access, add triton-san to your PATH environment variable using:"
echo ' export PATH="${PATH}'":${TRITON_SAN_INSTALL_ROOT}\""
