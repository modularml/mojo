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

"""Smoke tests for kernel launches that bind pointer arguments via
`DevicePointer`.

Two cases are exercised:

1. Every pointer arg bound from a `DevicePointer` (the new path under
   development).
2. A mix of `DeviceBuffer` and `DevicePointer` args in a single launch.

Both must produce identical results, which guards against regressions in the
existing `DeviceBuffer` arg-translation path while `DevicePointer` is being
developed.
"""

from asyncrt_test_utils import create_test_device_context
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import TestSuite, assert_equal


def vec_add(
    in0: Pointer[Float32, MutAnyOrigin],
    in1: Pointer[Float32, MutAnyOrigin],
    output: Pointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    # `Int` is not device-passable; widen the fixed-width arg.
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[unsafe_offset=tid] = in0[unsafe_offset=tid] + in1[unsafe_offset=tid]


comptime _LENGTH = 1024
comptime _BLOCK_DIM = 32


def test_kernel_with_device_pointers() raises:
    """Bind every pointer arg from a `DevicePointer`."""
    var ctx = create_test_device_context()

    var in0 = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var in1 = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var out = ctx.enqueue_create_buffer[.float32](_LENGTH)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(_LENGTH):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(2 * i + 1)

    var kernel = ctx.compile_function[vec_add]()
    ctx.enqueue_function(
        kernel,
        in0.device_ptr(),
        in1.device_ptr(),
        out.device_ptr(),
        Int32(_LENGTH),
        grid_dim=(_LENGTH // _BLOCK_DIM),
        block_dim=_BLOCK_DIM,
    )
    ctx.synchronize()
    # `DevicePointer` is a non-owning view, so each buffer it borrows must be
    # kept alive until the enqueued kernel has completed. `out` is kept alive
    # by its `map_to_host()` use below; keep the inputs alive explicitly.
    _ = in0^
    _ = in1^

    with out.map_to_host() as out_host:
        for i in range(_LENGTH):
            assert_equal(
                out_host[i],
                Float32(i) + Float32(2 * i + 1),
                String("at index ", i, " the value is ", out_host[i]),
            )


def test_kernel_mixed_buffer_and_device_pointer() raises:
    """Mix `DeviceBuffer` and `DevicePointer` args in a single launch."""
    var ctx = create_test_device_context()

    var in0 = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var in1 = ctx.enqueue_create_buffer[.float32](_LENGTH)
    var out = ctx.enqueue_create_buffer[.float32](_LENGTH)

    with in0.map_to_host() as in0_host, in1.map_to_host() as in1_host:
        for i in range(_LENGTH):
            in0_host[i] = Float32(i)
            in1_host[i] = Float32(2 * i + 1)

    var kernel = ctx.compile_function[vec_add]()
    ctx.enqueue_function(
        kernel,
        in0,  # DeviceBuffer
        in1.device_ptr(),  # DevicePointer
        out,  # DeviceBuffer
        Int32(_LENGTH),
        grid_dim=(_LENGTH // _BLOCK_DIM),
        block_dim=_BLOCK_DIM,
    )
    ctx.synchronize()
    # `in1` is borrowed by a non-owning `DevicePointer`, so it must be kept
    # alive until the enqueued kernel has completed.
    _ = in1^

    with out.map_to_host() as out_host:
        for i in range(_LENGTH):
            assert_equal(
                out_host[i],
                Float32(i) + Float32(2 * i + 1),
                String("at index ", i, " the value is ", out_host[i]),
            )


def main() raises:
    # TODO(MOCO-2556): Use automatic discovery when it can handle global_idx.
    # TestSuite.discover_tests[__functions_in_module()]().run()
    var suite = TestSuite()

    suite.test[test_kernel_with_device_pointers]()
    suite.test[test_kernel_mixed_buffer_and_device_pointer]()

    suite^.run()
