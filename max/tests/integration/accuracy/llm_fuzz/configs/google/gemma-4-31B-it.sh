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

batch_size=64
max_length=131072

extra_pipelines_args=(
  --device-memory-utilization=0.75
  --max-batch-input-tokens 32768
  --enable-prefix-caching
  --kv-cache-format float8_e4m3fn
  --enable-structured-output
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=gemma4
scenarios=
k2vv_mode=
circuit_breaker=0
