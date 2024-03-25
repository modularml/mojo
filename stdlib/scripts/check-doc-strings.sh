#!/usr/bin/env bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2024, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
