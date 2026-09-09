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
"""Provides low-level GPU intrinsic operations and memory access primitives.

Implements hardware-specific intrinsics that map directly to GPU assembly
instructions, focusing on NVIDIA GPU architectures. Includes:

- Global memory load/store operations with cache control
- Warp-level primitives and synchronization
- Memory fence and barrier operations
- Atomic operations and memory ordering primitives

These low-level primitives should be used carefully as they correspond
directly to hardware instructions and require understanding of the
underlying GPU architecture.
"""


# `std._gpu.intrinsics` makes this visible by importing it, and call sites import
# it from there rather than from `std.sys._assembly`. Keep that path working.
@__doc_inline
from std.sys._assembly import inlined_assembly


@__doc_inline
from std._gpu.intrinsics import (
    AMDBufferResource,
    CacheOperation,
    Scope,
    byte_permute,
    cvt_pk_fp8_f32_raw,
    ds_read_tr8_b64,
    ds_read_tr16_b64,
    get_ib_sts,
    ldg,
    lop,
    mulhi,
    mulwide,
    permlane_shuffle,
    permlane_swap,
    threadfence,
    warpgroup_reg_alloc,
    warpgroup_reg_dealloc,
)

from std._gpu.intrinsics import _get_nvtx_register_constraint
