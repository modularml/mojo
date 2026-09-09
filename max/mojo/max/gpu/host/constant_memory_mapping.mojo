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
"""This module provides functionality for mapping constant memory between host and device.

The module includes the `ConstantMemoryMapping` struct which represents a mapping of
constant memory that can be used for efficient data transfer between host and GPU device.
"""


@fieldwise_init
struct ConstantMemoryMapping(TrivialRegisterPassable):
    """Represents a mapping of constant memory between host and device.

    This struct encapsulates the information needed to manage constant memory
    that can be accessed by GPU kernels. Constant memory provides a fast, read-only
    cache accessible by all threads on the GPU device.

    Attributes:
        name: A string identifier for the constant memory mapping.
        ptr: Pointer to the memory location.
        byte_count: Size of the memory mapping in bytes.
    """

    var name: StaticString
    """A string identifier for the constant memory mapping.

    This name is used to uniquely identify the constant memory region in the GPU
    programming model, allowing the runtime to properly associate the memory with
    kernel references to constant memory symbols.
    """

    var ptr: OpaquePointer[MutUntrackedOrigin]
    """Pointer to the host memory location that will be mapped to device constant memory.

    This raw pointer represents the starting address of the memory region that will be
    accessible as constant memory on the GPU. The memory should remain valid for the
    lifetime of any kernels that access it.
    """

    var byte_count: Int
    """Size of the memory mapping in bytes.

    Specifies the total size of the constant memory region. This value is used by the
    runtime to determine how much data to transfer between host and device. The size
    must be sufficient to hold all data needed by GPU kernels.
    """
