#!/bin/bash

ERROR_COLOR_CODE='\033[0;31m'     # red
INFO_COLOR_CODE='\033[0;32m'      # green
WARNING_COLOR_CODE='\033[0;33m'   # brown/orange
NO_COLOR_CODE='\033[0m'

function print_error_and_exit() {
  echo -e "${ERROR_COLOR_CODE}Error: $1${NO_COLOR_CODE}"
  shift
  for m in "$@"; do
    echo -e "${ERROR_COLOR_CODE}$m${NO_COLOR_CODE}"
  done
  exit 1
}

function print_info() {
  echo -e "${INFO_COLOR_CODE}Status: $1${NO_COLOR_CODE}" 
  shift
  for m in "$@"; do
    echo -e ${INFO_COLOR_CODE}$m${NO_COLOR_CODE}
  done
}

function print_warning() {
  echo -e "${WARNING_COLOR_CODE}Warning: $1${NO_COLOR_CODE}" 
  shift
  for m in "$@"; do
    echo -e ${WARNING_COLOR_CODE}$m${NO_COLOR_CODE}
  done
}

function locate_file {
  FILE_NAME="$1"
  FILE_KEY="$2"
  SEARCH_PATH="$(realpath "$3")"
  if [ ! -d "${SEARCH_PATH}" ]; then
    print_error_and_exit "${SEARCH_PATH} is not a valid directory."
  fi

  LOCATED_FILE_PATH="$(find "${SEARCH_PATH}" -type f -name "${FILE_NAME}")"

  if [ -z "${LOCATED_FILE_PATH}" ]; then
    print_error_and_exit "Unable to find ${FILE_NAME} in ${SEARCH_PATH}."
  fi

  COUNT=$(echo "${LOCATED_FILE_PATH}" | wc -l)

  if [ ${COUNT} -gt 1 ]; then
    error_msg=("Multiple ${FILE_PATH} found in ${SEARCH_PATH}, including:"
               "${LOCATED_FILE_PATH}")
    print_error_and_exit "${error_msg[@]}"
  fi

  export ${FILE_KEY}="${LOCATED_FILE_PATH}"
  return 0
}