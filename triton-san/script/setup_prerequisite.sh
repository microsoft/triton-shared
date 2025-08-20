# Sets up a Python virtual environment and install prerequisites to prepare for future installations of LLVM, triton-shared, and Triton

#!/bin/bash
set -e

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <desired path to venv>"
  exit 1
fi

PARENT_FOLDER="$(realpath "$(dirname "$0")")"
TRITON_SHARED_PATH="$(realpath "${PARENT_FOLDER}/../..")"
VENV_PATH="$(realpath "$1")"

# include utility functions
source "${PARENT_FOLDER}/utility.inc"

# check if the path exists
if [ ! -e "${VENV_PATH}" ]; then
  mkdir -p "${VENV_PATH}"
fi

# set up venv
python3 -m venv ${VENV_PATH} --prompt triton-san
source ${VENV_PATH}/bin/activate

python3 -m pip install --upgrade pip
python3 -m pip install cmake==3.24 ninja pytest-xdist pybind11 setuptools torch

# echo to user to start virtual environment
info_message=("Please start the virtual environment: source \"${VENV_PATH}/bin/activate\""
              "This export is recommended for subsequent triton-shared builds: export TRITON_PLUGIN_DIRS=\"${TRITON_SHARED_PATH}/triton_shared\"")
print_info "${info_message[@]}"