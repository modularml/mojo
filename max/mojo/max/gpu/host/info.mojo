# ===----------------------------------------------------------------------=== #
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
# ===----------------------------------------------------------------------=== #
"""Contains information about GPU architectures and their capabilities.

This module provides detailed specifications for various GPU models including
NVIDIA and AMD GPUs. It includes information about compute capabilities,
memory specifications, thread organization, and performance characteristics.
"""


@__doc_inline
from std._gpu.host.info import (
    A10,
    A100,
    AMDCDNA2Family,
    AMDCDNA3Family,
    AMDCDNA4Family,
    AMDRDNAFamily,
    AcceleratorArchitectureFamily,
    AppleMetalFamily,
    B100,
    B200,
    B300,
    DGXSpark,
    GPUInfo,
    GTX1060,
    GTX1080Ti,
    GTX970,
    H100,
    JetsonThor,
    L4,
    MI250X,
    MI300A,
    MI300X,
    MI355X,
    MetalM1,
    MetalM1Metal4,
    MetalM2,
    MetalM2Metal4,
    MetalM3,
    MetalM3Metal4,
    MetalM4,
    MetalM4Metal4,
    MetalM5,
    MetalM5Metal4,
    NoGPU,
    NvidiaAdaFamily,
    NvidiaAmpereDatacenterFamily,
    NvidiaAmpereEmbeddedFamily,
    NvidiaAmpereWorkstationFamily,
    NvidiaBlackwellConsumerFamily,
    NvidiaBlackwellFamily,
    NvidiaHopperFamily,
    NvidiaMaxwellFamily,
    NvidiaPascalFamily,
    NvidiaTuringFamily,
    OrinNano,
    RTX2060,
    RTX3090,
    RTX4090,
    RTX4090m,
    RTX5090,
    Radeon6900,
    Radeon7600,
    Radeon7800,
    Radeon780m,
    Radeon7900,
    Radeon8060s,
    Radeon860m,
    Radeon880m,
    Radeon9060,
    Radeon9070,
    SteamDeck,
    TeslaP100,
    get_gpu_target,
    is_accelerator,
    is_cpu,
    is_gpu,
    is_valid_target,
)

from std._gpu.host.info import (
    _all_targets,
    _get_a100_target,
    _get_empty_target,
    _get_h100_target,
    _get_metal_m1_target,
    _get_metal_m2_target,
    _get_mi300x_target,
    _get_mi355x_target,
    _is_sm10x_gpu,
    _is_sm12x_gpu,
)
