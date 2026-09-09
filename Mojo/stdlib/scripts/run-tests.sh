#!/usr/bin/env bash
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

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_ROOT="${SCRIPT_DIR}"/../../..


FILTER=""
if [[ $# -gt 0 ]]; then
  FILTER="${1#./}" # remove leading relative file path if it has one
  if [[ -f ${FILTER} ]]; then # specific file
    FILTER="//mojo/$(dirname "$FILTER"):$(basename "$FILTER").test"
  else
    # specific test directory
    FILTER="${FILTER%/}" # remove trailing / if it has one
    FILTER="//mojo/${FILTER}/..."
  fi
else
  FILTER="//Mojo/stdlib/test/..."
fi

exec "$REPO_ROOT"/bazelw test "$FILTER"
