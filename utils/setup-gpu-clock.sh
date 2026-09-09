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

if ! [ "$(id -u)" = 0 ]; then
  echo "The script should be run as root." >&2
  exit 1
fi

# Configures Nvidia GPU settings. These settings should persist
# across reboots because we're setting `persistence-mode=1`.
set_nvidia_gpu_config() {
  echo "set_nvidia_gpu_config: Setting NVidia GPU config".

  nvidia-smi --persistence-mode=1
  nvidia-smi --auto-boost-default=0
  nvidia-smi -acp 0
  for i in $(seq 0 $(( $(nvidia-smi -L | wc -l) - 1 ))); do
    # `-ac` is deprecated on newer drivers, which REFUSE it and still exit 0 --
    # so the exit status cannot be trusted and the output has to be inspected.
    # Silence here previously meant "clocks pinned" to every caller while the
    # clock was in fact free to move.
    ac_out=$(nvidia-smi -ac "$(nvidia-smi --query-gpu=clocks.max.memory,clocks.max.sm --format=csv,noheader,nounits -i "$i" | sed 's/\ //')" -i "$i" 2>&1)
    echo "$ac_out"
    if grep -qi 'deprecated' <<<"$ac_out"; then
      echo "WARNING: gpu$i: 'nvidia-smi -ac' is deprecated on this driver; application clocks were NOT set." >&2
      echo "WARNING: gpu$i: to pin the clock, use 'nvidia-smi -lgc <mhz>,<mhz>' BELOW the board's" >&2
      echo "WARNING: gpu$i: power-limited clock, and release it with '-rgc'. See docs/internal/GpuClockPinning.md." >&2
    fi

    max_power_limit=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits -i "$i")
    nvidia-smi -pl "$max_power_limit" -i "$i"

    # `--vboost max` trades power budget for voltage. On a board that already
    # sits at its software power cap there is no budget to trade, so it buys
    # voltage instead of frequency and costs throughput -- measured at 6.8% on
    # a B200 under a decode workload. Only raise it when the board has headroom.
    power_cap=$(nvidia-smi --query-gpu=clocks_event_reasons.sw_power_cap --format=csv,noheader,nounits -i "$i")
    if [[ "$(echo "$power_cap" | tr -d '[:space:]')" == "Active" ]]; then
      echo "NOTE: gpu$i: at its software power cap; skipping '--vboost' (it costs throughput there)." >&2
    else
      max_vboost=$(nvidia-smi boost-slider --list -i "$i" | awk '/vboost/{print $4; exit}')
      nvidia-smi boost-slider --vboost "$max_vboost" -i "$i"
    fi
  done
}


# Configures AMD GPUs. These settings are important for performance testing.
#
# Performance determinism caps GFXCLK to trade peak clock for a repeatable one.
# The 1900 MHz cap was tuned for MI300 (gfx942, ~2100 MHz peak), so apply it
# only there rather than everywhere but the parts known to suffer: on gfx950
# (MI350/MI355, 2400 MHz peak) it costs 13% of serving throughput and AMD
# documents the mode as unsupported there, and newer parts should not silently
# inherit an MI300 value either. Match on the GFX version because it is a
# stable machine identifier; the marketing name varies by SKU.
set_amd_gpu_config() {
  echo "set_amd_gpu_config: Setting AMD GPU config".

  local version
  version=$(rocm-smi --showproductname 2>&1 |
    awk '/GFX Version/ {print $NF; exit}')

  # Unpinning a machine we failed to identify would silently change what its
  # benchmarks measure, so do nothing until someone looks. rocm-smi reports
  # N/A rather than failing when it cannot read the version.
  if [ -z "$version" ] || [ "$version" = "N/A" ]; then
    echo "Warning: cannot read GFX version; leaving GPU clocks alone." >&2
    return
  fi

  if [ "$version" != "gfx942" ]; then
    # The cap is applied at runtime and does not survive a reboot, so a machine
    # pinned by an earlier version of this script stays pinned until cleared.
    rocm-smi --resetperfdeterminism
    echo "set_amd_gpu_config: $version is not gfx942; perf determinism cleared."
    return
  fi

  if ! rocm-smi --setperfdeterminism 1900 2>&1; then
    echo "Warning: rocm-smi command failed from setup-gpu-clock.sh" >&2
  fi
}


# Auto-detect GPU type
if command -v nvidia-smi >/dev/null 2>&1; then
  set_nvidia_gpu_config
elif command -v rocm-smi >/dev/null 2>&1; then
  set_amd_gpu_config
else
  echo "No NVIDIA or AMD GPU attached"
fi
