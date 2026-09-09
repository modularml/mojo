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
# Trace driver for the fused GEMV + partial RMS norm kernel.
#
# Runs the kernel ONCE with `enable_trace=True`, copies the per-block
# timestamp buffer back to host, and dumps a CSV. Each row is one
# block; columns are the 7 trace roles defined in gemv_partial_norm.mojo
# plus the block index:
#
#   TRACE_CSV_BEGIN
#   block,T_ENTER,T_K_START,T_K_END,T_ATOMIC_OUT,
#         T_APPLY_START,T_APPLY_END,T_EXIT
#   0,<ts>,<ts>,...
#   ...
#   TRACE_CSV_END
#
# Build:
#   mojo build -O3 trace_gemv_partial_norm.mojo -o trace_driver
# ===----------------------------------------------------------------------=== #

from std.memory import alloc

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major

from nn.gemv_partial_norm import (
    GEMV_TRACE_EVENTS_PER_BLOCK,
    GmemTrace,
    gemv_and_partial_norm_with_scratch,
)


def main() raises:
    comptime a_type = DType.bfloat16
    comptime c_type = DType.bfloat16

    # Primary shape (Kimi M=1 path).
    comptime tile_n = 4
    comptime num_threads = 256
    var M = 1
    var N = 2112
    var K = 7168
    var N_NORMED = 1536
    var N_UNNORMED = N - N_NORMED

    comptime a_shape = row_major(Coord(Idx[1], Idx[7168]))
    comptime b_shape = row_major(Coord(Idx[2112], Idx[7168]))
    comptime normed_shape = row_major(Coord(Idx[1], Idx[1536]))
    var unnormed_shape = row_major(Coord(Idx[1], N_UNNORMED))
    comptime gamma_shape = row_major(Idx[1536])

    var num_blocks = (N + tile_n - 1) // tile_n

    with DeviceContext() as ctx:
        var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
        var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
        var gamma_dev = ctx.enqueue_create_buffer[a_type](N_NORMED)
        var normed_dev = ctx.enqueue_create_buffer[c_type](M * N_NORMED)
        var unnormed_dev = ctx.enqueue_create_buffer[c_type](M * N_UNNORMED)

        ctx.enqueue_memset(a_dev, Scalar[a_type](0.01))
        ctx.enqueue_memset(b_dev, Scalar[a_type](0.01))
        ctx.enqueue_memset(gamma_dev, Scalar[a_type](1.0))
        ctx.enqueue_memset(normed_dev, Scalar[c_type](0.0))
        ctx.enqueue_memset(unnormed_dev, Scalar[c_type](0.0))

        var a_tensor = TileTensor(a_dev, a_shape)
        var b_tensor = TileTensor(b_dev, b_shape)
        var gamma_tensor = TileTensor(gamma_dev, gamma_shape)
        var normed_tensor = TileTensor(normed_dev, normed_shape)
        var unnormed_tensor = TileTensor(unnormed_dev, unnormed_shape)

        var eps = Float32(0.001)

        var counter_buf = ctx.enqueue_create_buffer[.int32](1)
        ctx.enqueue_memset(counter_buf, Int32(0))

        # Trace buffer: num_blocks * GEMV_TRACE_EVENTS_PER_BLOCK uint64.
        var trace_buf = ctx.enqueue_create_buffer[.uint64](
            num_blocks * GEMV_TRACE_EVENTS_PER_BLOCK
        )
        ctx.enqueue_memset(trace_buf, UInt64(0))

        ctx.synchronize()

        # Warmup (untraced): warms the L2/icache.
        gemv_and_partial_norm_with_scratch[
            transpose_b=True,
            tile_n=tile_n,
            num_threads=num_threads,
        ](
            normed_tensor,
            unnormed_tensor,
            a_tensor,
            b_tensor,
            gamma_tensor,
            eps,
            counter_buf.unsafe_ptr().as_unsafe_any_origin(),
            ctx,
        )
        ctx.synchronize()

        # Traced run. `enable_trace=True` compiles in the record sites;
        # `trace_buf=GmemTrace(buf)` carries the device pointer.
        gemv_and_partial_norm_with_scratch[
            transpose_b=True,
            tile_n=tile_n,
            num_threads=num_threads,
            enable_trace=True,
        ](
            normed_tensor,
            unnormed_tensor,
            a_tensor,
            b_tensor,
            gamma_tensor,
            eps,
            counter_buf.unsafe_ptr().as_unsafe_any_origin(),
            ctx,
            trace_buf=GmemTrace(
                trace_buf.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
            ),
        )
        ctx.synchronize()

        var trace_host = List(
            length=num_blocks * GEMV_TRACE_EVENTS_PER_BLOCK,
            fill=UInt64(0),
        )
        ctx.enqueue_copy(trace_host, trace_buf)
        ctx.synchronize()

        print("TRACE_CSV_BEGIN")
        print(
            "block,T_ENTER,T_K_START,T_K_END,T_ATOMIC_IN,T_ATOMIC_OUT,"
            "T_APPLY_START,T_LOADS_DONE,T_REDUCE_DONE,T_APPLY_END,T_EXIT"
        )
        for b in range(num_blocks):
            var base = b * GEMV_TRACE_EVENTS_PER_BLOCK
            print(
                t"{b},"
                t"{Int(trace_host[base + 0])},"
                t"{Int(trace_host[base + 1])},"
                t"{Int(trace_host[base + 2])},"
                t"{Int(trace_host[base + 3])},"
                t"{Int(trace_host[base + 4])},"
                t"{Int(trace_host[base + 5])},"
                t"{Int(trace_host[base + 6])},"
                t"{Int(trace_host[base + 7])},"
                t"{Int(trace_host[base + 8])},"
                t"{Int(trace_host[base + 9])}"
            )
        print("TRACE_CSV_END")
        _ = trace_host^
