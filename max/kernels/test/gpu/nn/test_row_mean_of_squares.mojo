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
from nn.normalization import row_mean_of_squares
from std.testing import assert_almost_equal

from std.utils.index import Index


def run_row_mean_of_squares_gpu[
    in_dtype: DType, out_dtype: DType = DType.float32
](
    ctx: DeviceContext,
    rows: Int,
    cols: Int,
    rtol: Float64 = 1e-3,
    atol: Float64 = 1e-3,
) raises:
    print("== run_row_mean_of_squares_gpu rows=", rows, " cols=", cols)

    var data_h = ctx.enqueue_create_host_buffer[in_dtype](rows * cols)
    var out_h = ctx.enqueue_create_host_buffer[out_dtype](rows)

    rand[in_dtype](data_h.as_span())

    var data_d = ctx.enqueue_create_buffer[in_dtype](rows * cols)
    var out_d = ctx.enqueue_create_buffer[out_dtype](rows)

    var shape = Index(rows, cols)
    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(Index(rows, 1))))

    ctx.enqueue_copy(data_d, data_h)

    @always_inline
    def input_fn[
        width: Int
    ](coords: Coord) {var data_buf} -> SIMD[in_dtype, width]:
        var idx = data_buf.layout(coords)
        return data_buf.raw_load[width=width](idx)

    @always_inline
    def output_fn[
        width: SIMDLength
    ](coords: Coord, val: SIMD[out_dtype, width]) {var out_buf} -> None:
        var idx = out_buf.layout(Coord(Index(Int(coords[0].value()), 0)))
        out_buf.raw_store[width=width](idx, val)

    row_mean_of_squares[in_dtype, out_dtype, 2, target="gpu"](
        input_fn, output_fn, Coord(shape), ctx
    )
    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    # Float64 reference oracle.
    for r in range(rows):
        var acc = Float64(0)
        for c in range(cols):
            var v = data_h[r * cols + c].cast[.float64]()
            acc += v * v
        var expected = acc / Float64(cols)
        assert_almost_equal(Float64(out_h[r]), expected, rtol=rtol, atol=atol)

    _ = data_d
    _ = out_d


def main() raises:
    with DeviceContext() as ctx:
        # bfloat16 (primary): decode + prefill shapes, plus odd-N tail.
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 16, 1536)
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 16, 256)
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 512, 1536)
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 2048, 256)
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 16, 1537)
        # A column count larger than one block can cover (grid-stride loop).
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 4, 8192)
        # Few rows + a row size past `_SPLITK_MIN_ROW` (32768): exercises the
        # inner-axis split-K tier now that `supports_tiled` no longer blocks
        # it here. float32's narrower SIMD width also clears `_SPLITK_MAX_SIMD`
        # (bf16's wider natural width does not, so it stays on block/warp).
        run_row_mean_of_squares_gpu[.bfloat16](ctx, 4, 65536)
        run_row_mean_of_squares_gpu[.float32](
            ctx, 4, 65536, rtol=1e-6, atol=1e-6
        )

        # float32 must also be accepted (tighter tolerance).
        run_row_mean_of_squares_gpu[.float32](
            ctx, 16, 1536, rtol=1e-6, atol=1e-6
        )
        run_row_mean_of_squares_gpu[.float32](
            ctx, 16, 256, rtol=1e-6, atol=1e-6
        )
        run_row_mean_of_squares_gpu[.float32](
            ctx, 16, 1537, rtol=1e-6, atol=1e-6
        )
