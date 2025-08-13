#!/bin/bash
set -e
set -x

ROOT="$(realpath "$(dirname "$0")/../..")"
SCRIPT_FOLDER="$(realpath "$(dirname "$0")")"
VENV_PATH="${ROOT}/venv"
LLVM_PATH="${ROOT}/llvm"
LLVM_BUILD_PATH="${LLVM_PATH}/llvm-build"

if [ ! -e "${VENV_PATH}" ]; then
  . "${SCRIPT_FOLDER}/setup_prerequisite.sh" "${VENV_PATH}"
fi

if [ ! -e "${LLVM_PATH}" ]; then
  mkdir -p "${LLVM_PATH}"
  . "${SCRIPT_FOLDER}/build_llvm.sh" "${LLVM_PATH}"
fi

. "${SCRIPT_FOLDER}/build_triton_shared_with_triton_san.sh" "${LLVM_BUILD_PATH}" "${VENV_PATH}"
