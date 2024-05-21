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
REPO_ROOT="${SCRIPT_DIR}"/../..
BUILD_DIR="${REPO_ROOT}"/build

echo "Creating build directory for building the stdlib running the tests in."
mkdir -p "${BUILD_DIR}"

source "${SCRIPT_DIR}"/build-stdlib.sh

echo "Packaging up the test_utils."
TEST_UTILS_PATH="${REPO_ROOT}/stdlib/test/test_utils"
mojo package "${TEST_UTILS_PATH}" -o "${BUILD_DIR}/test_utils.mojopkg"

TEST_PATH="${REPO_ROOT}/stdlib/test"
if [[ $# -gt 0 ]]; then
  # If an argument is provided, use it as the specific test file or directory
  TEST_PATH=$1
fi

# Run the tests
lit -sv "${TEST_PATH}"
