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

# DFlash variant of the K2.7-Code ep-tp config. The flags below mirror the
# Kimi-K2.7-Code deployment's engine args verbatim; the only deviations are
# the two the fuzz runner forces:
#   * disk_offload_dir / disk_offload_max_gb -- the deployment's
#     /cache/max-cache mount does not exist on the fuzz runner (same
#     substitution the eagle config makes).
#   * max_length -- the runner always passes --max-length, which the
#     deployment omits; 262144 is the checkpoint's max_position_embeddings,
#     i.e. the value the deployment gets by default.
# Deliberately absent because the deployment does not set them:
# --max-num-steps, --enable-prefix-caching, --device-memory-utilization,
# and any draft quantization override (the draft checkpoint is bf16).

batch_size=64
max_length=262144

extra_pipelines_args=(
  --ep-size=8
  --ep-use-allreduce
  --max-batch-input-tokens=4096
  --data-parallel-degree=1
  --draft-model-path=nvidia/Kimi-K2.7-Code-DFlash
  --speculative-method=dflash
  --num-speculative-tokens=7
  --kv-cache-format=float8_e4m3fn
  --kv-connector-config '{"type":"tiered","host_offload_max_gb":512,"disk_offload_dir":"/tmp/max_kv_tiered","disk_offload_max_gb":1024}'
  --reasoning-parser=kimik2_5
  --tool-parser=kimik2_5
  --enable-structured-output
  --enable-penalties
  --trust-remote-code
  --served-model-name=nvidia/Kimi-K2.7-Code-NVFP4
)

# llm-fuzz knobs. Empty scenarios runs the tool's full default suite.
model_profile=kimi-k2.5
scenarios=
k2vv_mode=full
circuit_breaker=0
