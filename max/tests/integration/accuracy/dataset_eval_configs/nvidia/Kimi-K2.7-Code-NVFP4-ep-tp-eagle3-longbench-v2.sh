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
pipeline=nvidia/Kimi-K2.7-Code-NVFP4

batch_size=64
max_length=262144

extra_pipelines_args=(
  --device-memory-utilization=0.75
  --ep-size 8
  --ep-use-allreduce
  --data-parallel-degree 1
  --max-batch-input-tokens 4096
  --enable-prefix-caching
  --kv-cache-format float8_e4m3fn
  --trust-remote-code
  --speculative-method eagle
  --draft-model-path nvidia/Kimi-K2.6-Eagle3
  # No Eagle3 draft is trained for K2.7-Code; the K2.6 draft is reused in
  # production. draft.sliding_window override mirrors the production
  # deployment; it is not declared in the K2.6 draft's HF config.
  --model-override=draft.sliding_window=12288
  --num-speculative-tokens 3
  --reasoning-parser kimik2_5
  --tool-parser kimik2_5
  --enable-structured-output
  --enable-penalties
)
extra_longbench_v2_args=(
  --max_new_tokens 8192
  --max_context_length=100000 # https://github.com/THUDM/LongBench/issues/134
  --client_timeout=5000
)

evaluator=longbench-v2
