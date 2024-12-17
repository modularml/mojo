# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Implements the sys package."""

from ._io import stderr, stdout
from .arg import argv
from .debug import breakpointhook
from .ffi import DEFAULT_RTLD, RTLD, DLHandle, external_call
from .info import (
    alignof,
    bitwidthof,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_avx,
    has_avx2,
    has_avx512f,
    has_fma,
    has_intel_amx,
    has_neon,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    has_nvidia_gpu_accelerator,
    has_sse4,
    has_vnni,
    is_amd_gpu,
    is_apple_m1,
    is_apple_m2,
    is_apple_m3,
    is_apple_silicon,
    is_big_endian,
    is_gpu,
    is_little_endian,
    is_neoverse_n1,
    is_nvidia_gpu,
    is_x86,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    os_is_linux,
    os_is_macos,
    os_is_windows,
    simdbitwidth,
    simdbytewidth,
    simdwidthof,
    sizeof,
)
from .intrinsics import (
    PrefetchCache,
    PrefetchLocality,
    PrefetchOptions,
    PrefetchRW,
    _RegisterPackType,
    compressed_store,
    gather,
    llvm_intrinsic,
    masked_load,
    masked_store,
    prefetch,
    scatter,
    strided_load,
    strided_store,
)
from .param_env import env_get_bool, env_get_int, env_get_string, is_defined
from .terminate import exit
