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
REPO_ROOT=$(realpath "${SCRIPT_DIR}/../..")
BUILD_DIR="${REPO_ROOT}"/build
mkdir -p "${BUILD_DIR}"

ACTUAL_COMPILER_VERSION=$(mojo --version | tr " " "\n" | sed -n 2p)
EXPECTED_COMPILER_VERSION=$(<"${REPO_ROOT}"/stdlib/COMPATIBLE_COMPILER_VERSION)

if [ -z "${MOJO_OVERRIDE_COMPILER_VERSION_CHECK:-}" ]; then
  if [ "${EXPECTED_COMPILER_VERSION}" != "${ACTUAL_COMPILER_VERSION}" ]; then
    echo "Mismatch in compiler versions! Cannot build the standard library."
    echo "Expected compiler version: ${EXPECTED_COMPILER_VERSION}"
    echo "Current installed compiler version: ${ACTUAL_COMPILER_VERSION}"
    echo "Please run modular update nightly/mojo to get the latest compiler."
    exit 1
  fi
fi

STDLIB_PATH="${REPO_ROOT}/stdlib/src"

echo "Packaging up the Standard Library."
STDLIB_PACKAGE_NAME="stdlib.mojopkg"
FULL_STDLIB_PACKAGE_PATH="${BUILD_DIR}"/"${STDLIB_PACKAGE_NAME}"
mojo package "${STDLIB_PATH}" -o "${FULL_STDLIB_PACKAGE_PATH}"

echo Successfully created "${FULL_STDLIB_PACKAGE_PATH}"
