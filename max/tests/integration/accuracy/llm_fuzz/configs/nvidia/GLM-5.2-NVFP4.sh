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

batch_size=8
max_length=163840

extra_pipelines_args=(
  --data-parallel-degree 1
  --ep-size 8
  --kv-cache-format float8_e4m3fn
  --device-memory-utilization 0.7
  --speculative-method mtp
  --num-speculative-tokens 5
  --max-batch-input-tokens 1024
  --enable-structured-output
  --tool-parser glm45
  --reasoning-parser glm45
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=glm-5.1
scenarios=
k2vv_mode=
circuit_breaker=0
