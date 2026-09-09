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

from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import row_mean_of_squares_qk
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_row_mean_of_squares_qk_gpu[
    in_dtype: DType
](
    ctx: DeviceContext,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-3,
) raises:
    print(
        "== run_row_mean_of_squares_qk_gpu rows=",
        rows,
        " q_cols=",
        q_cols,
        " k_cols=",
        k_cols,
    )

    comptime out_dtype = DType.float32

    var q_h = ctx.enqueue_create_host_buffer[in_dtype](rows * q_cols)
    var k_h = ctx.enqueue_create_host_buffer[in_dtype](rows * k_cols)
    var out_h = ctx.enqueue_create_host_buffer[out_dtype](rows * 2)

    rand[in_dtype](q_h.as_span())
    rand[in_dtype](k_h.as_span())

    var q_d = ctx.enqueue_create_buffer[in_dtype](rows * q_cols)
    var k_d = ctx.enqueue_create_buffer[in_dtype](rows * k_cols)
    var out_d = ctx.enqueue_create_buffer[out_dtype](rows * 2)

    var q_buf = TileTensor(q_d, row_major(Coord(Index(rows, q_cols))))
    var k_buf = TileTensor(k_d, row_major(Coord(Index(rows, k_cols))))
    var out_buf = TileTensor(out_d, row_major(Coord(Index(rows, 2))))

    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)

    row_mean_of_squares_qk[target="gpu"](
        out_buf, q_buf, k_buf, rows, q_cols, k_cols, ctx
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Float64 reference oracle for both operands.
    for r in range(rows):
        var accq = Float64(0)
        for c in range(q_cols):
            var v = q_h[r * q_cols + c].cast[.float64]()
            accq += v * v
        assert_almost_equal(
            Float64(out_h[r * 2 + 0]),
            accq / Float64(q_cols),
            rtol=rtol,
            atol=atol,
        )

        var acck = Float64(0)
        for c in range(k_cols):
            var v = k_h[r * k_cols + c].cast[.float64]()
            acck += v * v
        assert_almost_equal(
            Float64(out_h[r * 2 + 1]),
            acck / Float64(k_cols),
            rtol=rtol,
            atol=atol,
        )

    _ = q_d
    _ = k_d
    _ = out_d


def main() raises:
    with DeviceContext() as ctx:
        # bfloat16 (primary): q/k with different widths (the TP case where
        # q_dim = n_heads*head_dim, k_dim = n_kv_heads*head_dim), decode +
        # prefill row counts, plus an odd k tail and a wide grid-stride case.
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 16, 1536, 256)
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 512, 1536, 256)
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 2048, 4096, 512)
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 16, 1536, 257)
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 4, 8192, 8192)
        # Equal widths must also work.
        run_row_mean_of_squares_qk_gpu[.bfloat16](ctx, 16, 512, 512)

        # float32 must also be accepted (tighter tolerance).
        run_row_mean_of_squares_qk_gpu[.float32](
            ctx, 16, 1536, 256, rtol=1e-6, atol=1e-6
        )
        run_row_mean_of_squares_qk_gpu[.float32](
            ctx, 16, 1537, 257, rtol=1e-6, atol=1e-6
        )
        # Few rows + a row size past `_SPLITK_MIN_ROW` (32768) at float32's
        # SIMD width (clears `_SPLITK_MAX_SIMD`): exercises the split-K
        # kernel for both the q and k `row_mean_of_squares` calls, writing
        # back through the stride-2 `q_out`/`k_out` closures.
        run_row_mean_of_squares_qk_gpu[.float32](
            ctx, 4, 65536, 40000, rtol=1e-6, atol=1e-6
        )
