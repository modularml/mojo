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
"""Tests alias-scoped raw-buffer DMA into LDS."""

from max.gpu import lane_id, WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.memory import CacheOperation
from max.gpu.sync import s_waitcnt
from max.gpu.intrinsics import AMDBufferResource
from std.memory import AddressSpace, unsafe_stack_allocation
from std.testing import assert_equal

from structured_kernels.amd_tile_io import _load_from_lds, _load_to_lds


def _copy_through_lds[
    dtype: DType,
    width: Int,
](
    src: MutPointer[Scalar[dtype], MutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    valid_elements: Int32,
):
    var smem = unsafe_stack_allocation[
        WARP_SIZE * width,
        Scalar[dtype],
        alignment=16,
        address_space=.SHARED,
    ]()
    var lane = Int(lane_id())
    var offset = lane * width
    var resource = AMDBufferResource(src, Int(valid_elements))
    _load_to_lds[width=width, cache_policy=CacheOperation.ALWAYS](
        resource,
        Int32(offset),
        smem,
    )
    s_waitcnt[vmcnt=0]()
    var values = _load_from_lds[width=width](smem + offset)
    comptime for i in range(width):
        dst[offset + i] = values[i]


def _value[dtype: DType](index: Int) -> Scalar[dtype]:
    return Scalar[dtype](index % 8)


def _run_case[dtype: DType, width: Int](ctx: DeviceContext) raises:
    comptime count = WARP_SIZE * width
    var host_in = ctx.enqueue_create_host_buffer[dtype](count)
    var host_out = ctx.enqueue_create_host_buffer[dtype](count)
    for i in range(count):
        host_in[i] = _value[dtype](i)

    var dev_in = ctx.enqueue_create_buffer[dtype](count)
    var dev_out = ctx.enqueue_create_buffer[dtype](count)
    ctx.enqueue_copy(dev_in, host_in)

    ctx.enqueue_function[_copy_through_lds[dtype, width]](
        dev_in,
        dev_out,
        Int32(count),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()
    for i in range(count):
        assert_equal(
            host_out[i].cast[.float32](),
            _value[dtype](i).cast[.float32](),
        )

    comptime valid_elements = count // 2
    dev_out.enqueue_fill(Scalar[dtype](7))
    ctx.enqueue_function[_copy_through_lds[dtype, width]](
        dev_in,
        dev_out,
        Int32(valid_elements),
        grid_dim=1,
        block_dim=WARP_SIZE,
    )
    ctx.enqueue_copy(host_out, dev_out)
    ctx.synchronize()
    for i in range(count):
        var expected = _value[dtype](i) if i < valid_elements else Scalar[
            dtype
        ](0)
        assert_equal(
            host_out[i].cast[.float32](),
            expected.cast[.float32](),
        )

    _ = dev_in^
    _ = dev_out^


def main() raises:
    with DeviceContext() as ctx:
        _run_case[.bfloat16, 8](ctx)
        _run_case[.float8_e4m3fn, 16](ctx)
