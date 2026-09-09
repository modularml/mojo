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

# Detects number and memory amounts of local GPUs.
# Outputs relevant bazelrc lines.

set -euo pipefail

if [[ "$(uname)" == "Linux" ]]; then
    if command -v nvidia-smi > /dev/null; then
        # One line per GPU, in MiB.
        output=$(nvidia-smi --query-gpu=memory.total --format=noheader,nounits)
        # We assume all GPUs are the same, so just grab the first
        mem_mib="$(echo "$output" | head -n 1)"
        # We specify GiB for local memory
        mem_gib=$(( mem_mib / 1024 ))

        gpu_count=$(echo "$output" | wc -l)
    elif command -v amd-smi > /dev/null; then
        if ! command -v jq > /dev/null; then
            echo "error: jq not found, necessary for gpu memory detection with amd-smi. please install jq."
            exit 1
        fi
        output=$(amd-smi static --vram --json)
        # For simplicity, we're assuming that this is output in MB
        unit=$(echo "$output" | jq -r '.gpu_data[0].vram.size.unit')
        if [[ "$unit" != "MB" ]]; then
            echo "error: assumed ami-smi output outputs MB, got $unit"
            exit 1
        fi
        # We assume all GPUs are the same, so just grab the first
        mem_mb=$(echo "$output" | jq -r '.gpu_data[0].vram.size.value')
        mem_gib=$(( mem_mb * 1000000 / 1073741824 ))

        gpu_count=$(echo "$output" | jq '.gpu_data | length')
    elif command -v rocm-smi > /dev/null; then
        # This is the older alternative to amd-smi.
        # The output is more confusing to parse, it doesn't seem worth it to support.
        echo "error: local resource detection for AMD GPUs only supported with amd-smi, which was not found."
        exit 1
    else
        # No GPUs
        echo "build --local_resources=gpu-memory=0"
        exit 0
    fi
else # We assume Mac is the only other case
    if ! command -v jq > /dev/null; then
        echo "error: jq not found, necessary for gpu memory detection on macos. please install jq."
        exit 1
    fi
    mem_gb=$(system_profiler SPHardwareDataType -json | jq --raw-output '.SPHardwareDataType[0].physical_memory' | cut -d ' ' -f 1)
    mem_gib=$(( mem_gb * 1000000000 / 1073741824 ))

    gpu_count=1
fi

echo "build --local_resources=gpu-memory=$mem_gib"
# At this point, we assume we have at least 1 GPU
echo "build --local_resources=gpu-1=1"
if (( gpu_count >= 2 )); then
    echo "build --local_resources=gpu-2=1"
fi
if (( gpu_count >= 4 )); then
    echo "build --local_resources=gpu-4=1"
fi
if (( gpu_count >= 8 )); then
    echo "build --local_resources=gpu-8=1"
fi
