#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

[[ "$CHECK" == "False" || "$CHECK" == "0" ]] && prefix="format" || prefix="lint"

[[ "$FAST" == "False" || "$FAST" == "0" ]] && suffix="" || suffix="-fast"


if [[ $# -gt 0 ]]; then
    echo "error: //:$prefix$suffix does not accept arguments." >&2
    if [[ "$suffix" == "" ]]; then
        echo "Run './bazelw run //:$prefix$suffix' without arguments to $prefix all files." >&2
    else
        echo "Run './bazelw run //:$prefix$suffix' without arguments to $prefix changed files." >&2
    fi
    exit 1
fi

# Set RUNFILES_DIR so the multirun script can find its dependencies
export RUNFILES_DIR="${0}.runfiles"
exec "${RUNFILES_DIR}/_main/bazel/lint/multirun.bash"
