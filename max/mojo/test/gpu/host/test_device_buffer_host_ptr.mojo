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
"""Tests `DeviceBuffer.unsafe_host_ptr()` on a unified-memory device.

Apple GPUs share their allocations with the CPU, so the host can read what a
kernel wrote without an `enqueue_copy` round trip. Devices with separate memory
must raise instead, which the last test checks so non-Metal CI covers the other
half of the contract.
"""

from max.gpu.host import DeviceContext
from std.testing import assert_equal, assert_raises, TestSuite


# Launched single-threaded (grid=1/block=1), so the kernels write
# unconditionally and need no `global_idx`.
def _fill_kernel(dst: Pointer[Int32, MutAnyOrigin]):
    dst[] = 7
    dst[unsafe_offset=1] = 8
    dst[unsafe_offset=2] = 9


def _double_kernel(dst: Pointer[Int32, MutAnyOrigin]):
    dst[] = dst[] * 2


def test_host_ptr_reads_kernel_writes() raises:
    with DeviceContext() as ctx:
        if ctx.api() != "metal":
            return
        var dev = ctx.enqueue_create_buffer[.int32](3)
        ctx.enqueue_function[_fill_kernel](
            dev.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        ctx.synchronize()
        var host = dev.unsafe_host_ptr()
        assert_equal(Int(host[]), 7)
        assert_equal(Int(host[unsafe_offset=1]), 8)
        assert_equal(Int(host[unsafe_offset=2]), 9)


def test_kernel_reads_host_writes() raises:
    with DeviceContext() as ctx:
        if ctx.api() != "metal":
            return
        var dev = ctx.enqueue_create_buffer[.int32](1)
        dev.unsafe_host_ptr()[] = 21
        ctx.enqueue_function[_double_kernel](
            dev.unsafe_ptr(), grid_dim=1, block_dim=1
        )
        ctx.synchronize()
        assert_equal(Int(dev.unsafe_host_ptr()[]), 42)


def test_host_ptr_carries_sub_buffer_offset() raises:
    with DeviceContext() as ctx:
        if ctx.api() != "metal":
            return
        var dev = ctx.enqueue_create_buffer[.int32](8)
        var sub = dev.create_sub_buffer[.int32](2, 4)
        dev.unsafe_host_ptr()[unsafe_offset=2] = 55
        assert_equal(Int(sub.unsafe_host_ptr()[]), 55)


def test_host_ptr_raises_without_unified_memory() raises:
    with DeviceContext() as ctx:
        if ctx.api() == "metal":
            return
        var dev = ctx.enqueue_create_buffer[.int32](4)
        with assert_raises():
            _ = dev.unsafe_host_ptr()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
