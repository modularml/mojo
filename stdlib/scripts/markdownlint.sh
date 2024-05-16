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

# This runs markdownlint on any files passed to it, using a custom config.
# It's really just a pass-through to `markdownlint` to enforce the config.

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LINT_CONFIG="${SCRIPT_DIR}/.markdownlint.yaml"

# Check for markdownlint.
if ! command -v markdownlint &>/dev/null; then
    echo "Error: markdownlint is not installed."
    echo "       Please install via npm install markdownlint-cli"
    exit 1
fi

# Check for arguments.
if [ $# -eq 0 ]; then
    echo "ERROR: You must pass files/directories/globs for MD files to lint."
    exit 1
fi

markdownlint --config "${LINT_CONFIG}" "$@"
