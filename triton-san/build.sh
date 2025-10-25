#!/bin/bash
set -e

function check_required_files() {
  for f in "$@"; do
    if [ ! -e "$f" ]; then
      return 1
    fi
  done
  return 0
}

PARENT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_SHARED_PATH="$(realpath "${PARENT_FOLDER}/..")"
ROOT="$(realpath "${PARENT_FOLDER}/../..")"
SCRIPT_FOLDER="$(realpath "${PARENT_FOLDER}/script")"
VENV_PATH="${ROOT}/venv"
LLVM_PATH="${ROOT}/llvm"
TRITON_PATH="${ROOT}/triton"
LLVM_BUILD_PATH="${LLVM_PATH}/llvm-build"
LLVM_INSTALL_DIR="${LLVM_PATH}/llvm-install"
TRITON_SAN_INSTALL_DIR="${ROOT}"

# include utility functions
source "${SCRIPT_FOLDER}/utility.inc"

echo "================= Check out triton ================"
TRITON_HASH_FILE="${TRITON_SHARED_PATH}/triton-hash.txt"
if [ ! -e "${TRITON_HASH_FILE}" ]; then
  print_error_and_exit "${TRITON_HASH_FILE} does not exist."
fi

EXPECT_HEAD_COMMIT=$(cat "${TRITON_HASH_FILE}")
if [ -e "${TRITON_PATH}" ]; then
  warning_msg=("The path ${TRITON_PATH} already exists.")
  print_warning "${warning_msg[@]}"
  pushd "${TRITON_PATH}" > /dev/null
  HEAD_COMMIT=$(git rev-parse HEAD)
  if [ "${HEAD_COMMIT}" != "${EXPECT_HEAD_COMMIT}" ]; then
    error_msg=("The head commit of the existing Triton repository does not match the expected commit hash."
               "HEAD: ${HEAD_COMMIT}, EXPECTED: ${EXPECT_HEAD_COMMIT}")
    print_error_and_exit "${error_msg[@]}"
  fi
  EXIST_TRITON_BUILD="${TRITON_PATH}/build"
  if [ -e "${EXIST_TRITON_BUILD}" ]; then
    rm -rf "${TRITON_PATH}/build"
    echo "Remove ${TRITON_PATH}/build"
  fi
  popd > /dev/null
else
  mkdir -p "${TRITON_PATH}"
  git clone https://github.com/triton-lang/triton.git "${TRITON_PATH}"
  pushd "${TRITON_PATH}" > /dev/null
  cd "${TRITON_PATH}" && git checkout ${EXPECT_HEAD_COMMIT}
  popd > /dev/null
fi

echo -e "\n\n\n"

echo "================= Setup Python virtual environment ================"
if [ ! -e "${VENV_PATH}" ]; then
  "${SCRIPT_FOLDER}/setup_prerequisite.sh" "${TRITON_PATH}" "${VENV_PATH}" 
else 
  required_file=(
    "${VENV_PATH}/bin/activate"
  )
  if ! check_required_files "${required_file[@]}"; then
    error_msg=(
      "${VENV_PATH} exists but is not a valid Python virtual environment."
      "Please delete the directory ${VENV_PATH} and rerun this build script."
    )
    print_error_and_exit "${error_msg[@]}"
  fi
  print_info "Reuse Python virtual environment at ${VENV_PATH}."
fi

# start the virtual environment to ensure prerequisites like cmake and ninja are available in the current session
. "${VENV_PATH}/bin/activate"
echo -e "\n\n\n"

echo "=========================== Build LLVM ============================"
if [ ! -e "${LLVM_PATH}" ]; then
  mkdir -p "${LLVM_PATH}"
  "${SCRIPT_FOLDER}/build_llvm.sh" "${TRITON_PATH}" "${LLVM_PATH}"
else
  required_file=(
    "${LLVM_PATH}/llvm-install/bin"
    "${LLVM_PATH}/llvm-install/include"
    "${LLVM_PATH}/llvm-install/lib"
  )
  if ! check_required_files "${required_file[@]}"; then
    error_msg=("${LLVM_PATH} exists but has no valid LLVM binary."
               "Please delete the directory ${LLVM_PATH} and rerun this build script.")
    print_error_and_exit "${error_msg[@]}"
  fi
  print_info "Reuse LLVM binary at ${LLVM_PATH}."
fi
echo -e "\n\n\n"

echo "================== Build trion-shared and triton =================="
export PATH="${LLVM_PATH}/llvm-install/bin:${PATH}"
"${SCRIPT_FOLDER}/build_triton_shared_with_triton_san.sh" "${LLVM_BUILD_PATH}" "${VENV_PATH}" "${TRITON_PATH}" "${TRITON_SHARED_PATH}"
echo -e "\n\n\n"

echo "======================== Install triton-san ======================="
"${SCRIPT_FOLDER}/install_triton_san.sh" "${LLVM_INSTALL_DIR}" "${VENV_PATH}" "${TRITON_PATH}" "${TRITON_SAN_INSTALL_DIR}"
