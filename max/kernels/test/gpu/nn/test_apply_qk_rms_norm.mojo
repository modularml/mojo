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

from std.math import rsqrt
from std.random import rand

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from nn.normalization import apply_qk_rms_norm
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_apply_qk_rms_norm_gpu[
    in_dtype: DType
](
    ctx: DeviceContext,
    rows: Int,
    q_cols: Int,
    k_cols: Int,
    epsilon: Float32 = 1e-6,
    rtol: Float64 = 1e-2,
    atol: Float64 = 1e-2,
) raises:
    print(
        "== run_apply_qk_rms_norm_gpu rows=",
        rows,
        " q_cols=",
        q_cols,
        " k_cols=",
        k_cols,
        " in_dtype=",
        in_dtype,
    )

    comptime out_dtype = in_dtype

    var q_h = ctx.enqueue_create_host_buffer[in_dtype](rows * q_cols)
    var k_h = ctx.enqueue_create_host_buffer[in_dtype](rows * k_cols)
    var gq_h = ctx.enqueue_create_host_buffer[.float32](q_cols)
    var gk_h = ctx.enqueue_create_host_buffer[.float32](k_cols)
    var var_h = ctx.enqueue_create_host_buffer[.float32](rows * 2)
    var qo_h = ctx.enqueue_create_host_buffer[out_dtype](rows * q_cols)
    var ko_h = ctx.enqueue_create_host_buffer[out_dtype](rows * k_cols)

    rand[in_dtype](q_h.as_span())
    rand[in_dtype](k_h.as_span())
    rand[.float32](gq_h.as_span())
    rand[.float32](gk_h.as_span())
    # qk_var holds mean-of-squares stats (positive). Use a positive range.
    rand[.float32](var_h.as_span())
    for i in range(rows * 2):
        var_h[i] = var_h[i] * Float32(0.5) + Float32(0.25)

    var q_d = ctx.enqueue_create_buffer[in_dtype](rows * q_cols)
    var k_d = ctx.enqueue_create_buffer[in_dtype](rows * k_cols)
    var gq_d = ctx.enqueue_create_buffer[.float32](q_cols)
    var gk_d = ctx.enqueue_create_buffer[.float32](k_cols)
    var var_d = ctx.enqueue_create_buffer[.float32](rows * 2)
    var qo_d = ctx.enqueue_create_buffer[out_dtype](rows * q_cols)
    var ko_d = ctx.enqueue_create_buffer[out_dtype](rows * k_cols)

    var q_buf = TileTensor(q_d, row_major(Coord(Index(rows, q_cols))))
    var k_buf = TileTensor(k_d, row_major(Coord(Index(rows, k_cols))))
    var gq_buf = TileTensor(gq_d, row_major(Coord(Index(q_cols))))
    var gk_buf = TileTensor(gk_d, row_major(Coord(Index(k_cols))))
    var var_buf = TileTensor(var_d, row_major(Coord(Index(rows, 2))))
    var qo_buf = TileTensor(qo_d, row_major(Coord(Index(rows, q_cols))))
    var ko_buf = TileTensor(ko_d, row_major(Coord(Index(rows, k_cols))))

    ctx.enqueue_copy(q_d, q_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(gq_d, gq_h)
    ctx.enqueue_copy(gk_d, gk_h)
    ctx.enqueue_copy(var_d, var_h)

    apply_qk_rms_norm[target="gpu",](
        qo_buf,
        ko_buf,
        gq_buf,
        gk_buf,
        var_buf,
        q_buf,
        k_buf,
        epsilon,
        rows,
        q_cols,
        k_cols,
        ctx,
    )

    ctx.enqueue_copy(qo_h, qo_d)
    ctx.enqueue_copy(ko_h, ko_d)
    ctx.synchronize()

    # Reference oracle computed in float32 with the SAME grouping the kernel
    # uses: ((x_f32 * rs) * gamma) then cast. rsqrt computed in float32.
    for r in range(rows):
        var rs_q = rsqrt(var_h[r * 2 + 0] + epsilon)
        for c in range(q_cols):
            var xf = q_h[r * q_cols + c].cast[.float32]()
            var g = gq_h[c]
            var expect = ((xf * rs_q) * g).cast[out_dtype]()
            assert_almost_equal(
                qo_h[r * q_cols + c],
                expect,
                rtol=rtol,
                atol=atol,
            )

        var rs_k = rsqrt(var_h[r * 2 + 1] + epsilon)
        for c in range(k_cols):
            var xf = k_h[r * k_cols + c].cast[.float32]()
            var g = gk_h[c]
            var expect = ((xf * rs_k) * g).cast[out_dtype]()
            assert_almost_equal(
                ko_h[r * k_cols + c],
                expect,
                rtol=rtol,
                atol=atol,
            )

    _ = q_d
    _ = k_d
    _ = gq_d
    _ = gk_d
    _ = var_d
    _ = qo_d
    _ = ko_d


def main() raises:
    with DeviceContext() as ctx:
        # bfloat16 (primary). MiniMax-M2.7 TP4 decode dims: per-rank Q slice
        # Nq=1536 (6144/4), K slice Nk=256 (1024/4). Decode (small M) +
        # prefill row counts, an odd k tail (scalar path), and a wide
        # grid-stride case. bf16 cast loses precision -> looser tolerance.
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 1, 1536, 256)
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 16, 1536, 256)
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 17, 1536, 256)
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 512, 1536, 256)
        # Non-simd-multiple width exercises the scalar fallback path.
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 16, 1536, 257)
        # Wide grid-stride case.
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 4, 8192, 8192)
        # Equal widths must also work.
        run_apply_qk_rms_norm_gpu[.bfloat16](ctx, 16, 512, 512)

        # float32 must also be accepted (tight tolerance: same f32 grouping).
        run_apply_qk_rms_norm_gpu[.float32](
            ctx, 16, 1536, 256, rtol=1e-6, atol=1e-6
        )
        run_apply_qk_rms_norm_gpu[.float32](
            ctx, 17, 1537, 257, rtol=1e-6, atol=1e-6
        )
