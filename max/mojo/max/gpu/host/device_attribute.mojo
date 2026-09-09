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
"""This module defines GPU device attributes that can be queried from CUDA-compatible devices.

The module provides the `DeviceAttribute` struct which encapsulates the various device
properties and capabilities that can be queried through the CUDA driver API. Each attribute
is represented as a constant with a corresponding integer value that maps to the CUDA
driver's attribute enumeration.

These attributes allow applications to query specific hardware capabilities and limitations
of GPU devices, such as maximum thread counts, memory sizes, compute capabilities, and
supported features.

:::note
See the [`DeviceContext`](/api/mojo/max/gpu/host/device_context/DeviceContext/) page
for examples that retrieve `DeviceAttribute` values.
:::
"""


@fieldwise_init("implicit")
struct DeviceAttribute(TrivialRegisterPassable):
    """
    Represents CUDA device attributes that can be queried from a GPU device.

    This struct encapsulates the various device properties and capabilities that can be
    queried through the CUDA driver API. Each attribute is represented as a constant
    with a corresponding integer value that maps to the CUDA driver's attribute enum.
    """

    var _value: Int32
    """The integer value representing the specific device attribute."""

    comptime MAX_THREADS_PER_BLOCK = Self(1)
    """Maximum number of threads per block.
    """

    comptime MAX_BLOCK_DIM_X = Self(2)
    """Maximum block dimension X.
    """

    comptime MAX_BLOCK_DIM_Y = Self(3)
    """Maximum block dimension Y.
    """

    comptime MAX_BLOCK_DIM_Z = Self(4)
    """Maximum block dimension Z.
    """

    comptime MAX_GRID_DIM_X = Self(5)
    """Maximum grid dimension X.
    """

    comptime MAX_GRID_DIM_Y = Self(6)
    """Maximum grid dimension Y.
    """

    comptime MAX_GRID_DIM_Z = Self(7)
    """Maximum grid dimension Z.
    """

    comptime MAX_SHARED_MEMORY_PER_BLOCK = Self(8)
    """Maximum shared memory available per block in bytes.
    """

    comptime WARP_SIZE = Self(10)
    """Warp size in threads.
    """

    comptime MAX_REGISTERS_PER_BLOCK = Self(12)
    """Maximum number of 32-bit registers available per block.
    """

    comptime CLOCK_RATE = Self(13)
    """Typical clock frequency in kilohertz.
    """

    comptime MULTIPROCESSOR_COUNT = Self(16)
    """Number of multiprocessors on device.
    """

    comptime MAX_THREADS_PER_MULTIPROCESSOR = Self(39)
    """Maximum resident threads per multiprocessor.
    """

    comptime COMPUTE_CAPABILITY_MAJOR = Self(75)
    """Major compute capability version number.
    """

    comptime COMPUTE_CAPABILITY_MINOR = Self(76)
    """Minor compute capability version number.
    """
    comptime MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = Self(81)
    """Maximum shared memory available per multiprocessor in bytes.
    """
    comptime MAX_REGISTERS_PER_MULTIPROCESSOR = Self(82)
    """Maximum number of 32-bit registers available per multiprocessor.
    """
    comptime COOPERATIVE_LAUNCH = Self(95)
    """Device supports launching cooperative kernels.
    """
    comptime MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = Self(97)
    """Maximum shared memory per block usable via `cudaFuncSetAttribute`.
    """
    comptime MAX_BLOCKS_PER_MULTIPROCESSOR = Self(106)
    """Maximum resident blocks per multiprocessor.
    """
    comptime MAX_ACCESS_POLICY_WINDOW_SIZE = Self(109)
    """CUDA-only: Maximum value of CUaccessPolicyWindow::num_bytes.
    """

    # Context-level attributes (values >= 1000).
    comptime PARALLELISM_LEVEL = Self(1000)
    """Number of parallel worker threads for CPU work dispatch.
    """
