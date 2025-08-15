#!/bin/bash
set -e

ROOT="$(realpath "$(dirname "$0")/../..")"
SCRIPT_FOLDER="$(realpath "$(dirname "$0")/script")"
VENV_PATH="${ROOT}/venv"
LLVM_PATH="${ROOT}/llvm"
LLVM_BUILD_PATH="${LLVM_PATH}/llvm-build"
LLVM_INSTALL_DIR="${LLVM_PATH}/llvm-install"
TRITON_SAN_INSTALL_DIR="${ROOT}"

echo "================= Setup Python Virtual Environment ================"
if [ ! -e "${VENV_PATH}" ]; then
  "${SCRIPT_FOLDER}/setup_prerequisite.sh" "${VENV_PATH}"
else 
  echo "Reuse Python virtual environment at ${VENV_PATH}"
fi

# start the virtual environment to ensure prerequisites like cmake and ninja are available in the current session
. "${VENV_PATH}/bin/activate"
echo -e "\n\n\n"

echo "=========================== Build LLVM ============================"
if [ ! -e "${LLVM_PATH}" ]; then
  mkdir -p "${LLVM_PATH}"
  "${SCRIPT_FOLDER}/build_llvm.sh" "${LLVM_PATH}"
else
  echo "Reuse LLVM binary at ${LLVM_PATH}"
fi
echo -e "\n\n\n"

echo "================== Build trion-shared and triton =================="
"${SCRIPT_FOLDER}/build_triton_shared_with_triton_san.sh" "${LLVM_BUILD_PATH}" "${VENV_PATH}"
echo -e "\n\n\n"

echo "======================== Install triton-san ======================="
"${SCRIPT_FOLDER}/install_triton_san.sh" "${TRITON_SAN_INSTALL_DIR}" "${VENV_PATH}" "${LLVM_INSTALL_DIR}"
