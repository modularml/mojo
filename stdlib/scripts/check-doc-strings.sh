#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="${SCRIPT_DIR}/../.."

check_doc_string() {
  local pkg=$1
  echo "Checking API doc string conformance for package ${pkg}"

  local warnings_file="${BUILD_DIR}/${pkg}_warnings.txt"
  rm -f "${warnings_file}"
  mojo doc -warn-missing-doc-strings -o /dev/null "${REPO_ROOT}/${pkg}" > "${warnings_file}" 2>&1
  python3 "${SCRIPT_DIR}"/check-file-is-empty.py "${warnings_file}"
}

BUILD_DIR="${REPO_ROOT}"/build
mkdir -p "${BUILD_DIR}"

check_doc_string stdlib
check_doc_string test_utils

