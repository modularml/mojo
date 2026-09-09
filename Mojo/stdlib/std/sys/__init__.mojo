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
"""System runtime: I/O, hardware info, intrinsics, compile-time utils.

The `sys` package provides low-level access to system functionality and runtime
information. It includes tools for interacting with the operating system,
querying hardware capabilities, and accessing compiler intrinsics. This package
bridges Mojo code and the underlying system environment.

Use this package for system-level programming, hardware-specific optimizations,
or when you need direct access to platform capabilities and compiler features.
For foreign function interface (FFI) functionality, use the `ffi` module.
"""

from ._assembly import inlined_assembly
from ._io import stderr, stdin, stdout
from .arg import argv
from .compile import codegen_unreachable
from .debug import breakpointhook
from .info import (
    CompilationTarget,
    Vendor,
    align_of,
    bit_width_of,
    has_accelerator,
    has_amd_gpu_accelerator,
    has_amd_rdna_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_apple_gpu,
    is_big_endian,
    is_gpu,
    is_little_endian,
    is_nvidia_gpu,
    num_logical_cores,
    num_performance_cores,
    num_physical_cores,
    simd_bit_width,
    simd_byte_width,
    simd_width_of,
    size_of,
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
from .defines import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
    is_defined,
)
from .terminate import exit
