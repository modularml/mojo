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

# shellcheck disable=SC2034  # Variables are used when sourced

use_max_private=1
batch_size=64
max_length=131072

extra_pipelines_args=(
  --ep-size 4
  --data-parallel-degree 1
  --device-graph-capture
  --trust-remote-code
  --enable-structured-output
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=minimax-m3
scenarios=
k2vv_mode=
circuit_breaker=0

# Skip the known MiniMax-M3 failures (the official MiniMax API fails them too).
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../minimax-m3-known-failures.sh"
