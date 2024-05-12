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

from .arg import argv
from .debug import breakpointhook
from ._io import stderr, stdout
from .ffi import RTLD, DEFAULT_RTLD, DLHandle, external_call
from .info import (
    is_x86,
    has_sse4,
    has_avx,
    has_avx2,
    has_avx512f,
    has_vnni,
    has_neon,
    has_neon_int8_dotprod,
    has_neon_int8_matmul,
    is_apple_m1,
    is_apple_m2,
    is_apple_m3,
    is_apple_silicon,
    is_neoverse_n1,
    has_intel_amx,
    os_is_macos,
    os_is_linux,
    os_is_windows,
    triple_is_nvidia_cuda,
    is_little_endian,
    is_big_endian,
    simdbitwidth,
    simdbytewidth,
    sizeof,
    alignof,
    bitwidthof,
    simdwidthof,
    num_physical_cores,
    num_logical_cores,
    num_performance_cores,
)
from .intrinsics import (
    llvm_intrinsic,
    gather,
    scatter,
    PrefetchLocality,
    PrefetchRW,
    PrefetchCache,
    PrefetchOptions,
    prefetch,
    masked_load,
    masked_store,
    compressed_store,
    strided_load,
    strided_store,
    _RegisterPackType,
)
from .param_env import (
    is_defined,
    env_get_int,
    env_get_string,
)
from .terminate import exit
