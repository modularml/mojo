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
"""Low-level memory management: pointers, allocations, address spaces.

The `memory` package provides primitives for direct memory manipulation and
pointer operations. It offers multiple pointer types with varying safety
guarantees, from reference-counted smart pointers to raw unsafe pointers, along
with functions for memory operations and allocation. This package enables
systems programming and interfacing with external code requiring explicit
memory control.

Use this package for performance-critical code requiring manual memory control,
interfacing with C libraries, implementing custom data structures, or accessing
specialized memory. Most code should prefer higher-level collections and
automatic memory management.
"""

from .alloc import Allocation, ThinAllocation, alloc, dealloc, Layout
from .arc_pointer import ArcPointer
from .memory import (
    unsafe_memcmp,
    unsafe_memcpy,
    # TODO(MSTDL-2918): Remove this export once the `memmove` deprecation is
    # dropped; callers should use `unsafe_memmove`.
    memmove,
    unsafe_memmove,
    unsafe_memset,
    unsafe_memset_zero,
    unsafe_destroy_n,
    is_trivially_copyable,
    is_trivially_deletable,
    is_trivially_movable,
    unsafe_uninit_copy_n,
    unsafe_uninit_move_n,
    forget_deinit,
)
from .stack_allocation import stack_allocation, unsafe_stack_allocation
from .owned_pointer import OwnedPointer
from .address_space import AddressSpace
from .pointer import (
    ImmOpaquePointer,
    ImmPointer,
    MutOpaquePointer,
    MutPointer,
    OpaquePointer,
    OptionalPointer,
    Pointer,
)
from .unsafe import bitcast, pack_bits
from .unsafe_pointer import UnsafePointer
from .maybe_uninit import MaybeUninit
