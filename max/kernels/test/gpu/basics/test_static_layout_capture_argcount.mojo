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
#
# Regression test: GPU kernel launches must pack captured arguments correctly
# when a captured value has a zero-sized type -- for example a `TileTensor`'s
# fully-static layout struct (storeSize == 0).
#
# Such a capture carries no runtime data and the device kernel elides it, but the
# host launch path used to still emit a slot for it in the packed argument array.
# Because CUDA reads kernel arguments positionally, a zero-sized slot preceding a
# real argument shifted every following argument by one: two captured
# static-layout tensors crashed with CUDA_ERROR_ILLEGAL_ADDRESS, and Metal
# rejected the zero-sized argument outright. The fix drops zero-sized captures
# when packing the launch arguments (see `DeviceFunction._call_with_pack` and
# `_call_with_pack_checked`).

from std.sys import align_of
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal
from layout import Coord, TileTensor, row_major

comptime dtype = DType.float32
comptime N = 256
comptime block_dim = 64


# V1: a single fully-static-layout TileTensor captured DIRECTLY into the
# launched kernel (the minimal `abl_D` repro). The zero-sized layout struct is
# the ONLY capture, so its over-count slot is trailing. Kernel writes 1.0 to
# every element; we assert every element reads back as 1.0.
def run_v1_single_static_capture(ctx: DeviceContext) raises:
    var out_host = ctx.enqueue_create_host_buffer[dtype](N)
    var out_dev = ctx.enqueue_create_buffer[dtype](N)
    out_dev.enqueue_fill(-1.0)
    var out_tt = TileTensor(out_dev, row_major[N]())

    @__parameter
    @__copy_capture(out_tt)
    def kernel():
        var tid = Int(global_idx.x)
        if tid < N:
            out_tt.store[width=1, alignment=align_of[dtype]()](
                Coord(tid), SIMD[dtype, 1](1.0)
            )

    ctx.enqueue_function[kernel](
        grid_dim=(N + block_dim - 1) // block_dim,
        block_dim=block_dim,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(out_host[i], Scalar[dtype](1.0))


# V2: a fully-static-layout TileTensor captured together with a REAL captured
# scalar that the kernel reads. If the zero-sized layout slot displaces the
# real scalar in the packed argument buffer, the kernel reads a wrong scalar and
# the stored values are wrong. Kernel writes `scale` to every element.
def run_v2_static_capture_plus_scalar(ctx: DeviceContext) raises:
    var out_host = ctx.enqueue_create_host_buffer[dtype](N)
    var out_dev = ctx.enqueue_create_buffer[dtype](N)
    out_dev.enqueue_fill(-1.0)
    var out_tt = TileTensor(out_dev, row_major[N]())
    var scale = Scalar[dtype](7.0)

    @__parameter
    @__copy_capture(out_tt, scale)
    def kernel():
        var tid = Int(global_idx.x)
        if tid < N:
            out_tt.store[width=1, alignment=align_of[dtype]()](
                Coord(tid), SIMD[dtype, 1](scale)
            )

    ctx.enqueue_function[kernel](
        grid_dim=(N + block_dim - 1) // block_dim,
        block_dim=block_dim,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(out_host[i], Scalar[dtype](7.0))


# V3: TWO fully-static-layout TileTensors captured. Two zero-sized layout slots
# are in play; if either tensor's storage pointer is displaced by a zero-sized
# slot, writes land in the wrong buffer. Kernel writes 2.0 to A and 3.0 to B.
def run_v3_two_static_captures(ctx: DeviceContext) raises:
    var a_host = ctx.enqueue_create_host_buffer[dtype](N)
    var b_host = ctx.enqueue_create_host_buffer[dtype](N)
    var a_dev = ctx.enqueue_create_buffer[dtype](N)
    var b_dev = ctx.enqueue_create_buffer[dtype](N)
    a_dev.enqueue_fill(-1.0)
    b_dev.enqueue_fill(-1.0)
    var a_tt = TileTensor(a_dev, row_major[N]())
    var b_tt = TileTensor(b_dev, row_major[N]())

    @__parameter
    @__copy_capture(a_tt, b_tt)
    def kernel():
        var tid = Int(global_idx.x)
        if tid < N:
            a_tt.store[width=1, alignment=align_of[dtype]()](
                Coord(tid), SIMD[dtype, 1](2.0)
            )
            b_tt.store[width=1, alignment=align_of[dtype]()](
                Coord(tid), SIMD[dtype, 1](3.0)
            )

    ctx.enqueue_function[kernel](
        grid_dim=(N + block_dim - 1) // block_dim,
        block_dim=block_dim,
    )
    ctx.enqueue_copy(a_host, a_dev)
    ctx.enqueue_copy(b_host, b_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(a_host[i], Scalar[dtype](2.0))
        assert_equal(b_host[i], Scalar[dtype](3.0))


# V4: a real captured scalar FIRST, then a fully-static-layout TileTensor. Tests
# whether the zero-sized layout slot corrupts when it is NOT the first capture.
# Kernel writes `scale` to every element of the tensor.
def run_v4_scalar_first_then_static(ctx: DeviceContext) raises:
    var out_host = ctx.enqueue_create_host_buffer[dtype](N)
    var out_dev = ctx.enqueue_create_buffer[dtype](N)
    out_dev.enqueue_fill(-1.0)
    var scale = Scalar[dtype](5.0)
    var out_tt = TileTensor(out_dev, row_major[N]())

    @__parameter
    @__copy_capture(scale, out_tt)
    def kernel():
        var tid = Int(global_idx.x)
        if tid < N:
            out_tt.store[width=1, alignment=align_of[dtype]()](
                Coord(tid), SIMD[dtype, 1](scale)
            )

    ctx.enqueue_function[kernel](
        grid_dim=(N + block_dim - 1) // block_dim,
        block_dim=block_dim,
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(N):
        assert_equal(out_host[i], Scalar[dtype](5.0))


def main() raises:
    with DeviceContext() as ctx:
        run_v1_single_static_capture(ctx)
        run_v2_static_capture_plus_scalar(ctx)
        run_v3_two_static_captures(ctx)
        run_v4_scalar_first_then_static(ctx)
