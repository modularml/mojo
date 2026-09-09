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
from std.sys import align_of

from max.gpu.host import DeviceContext
from layout import Coord, TileTensor, row_major
from layout.math import mean, variance
from nn.normalization import *
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_layer_norm_gpu[
    dtype: DType, rank: Int
](ctx: DeviceContext, shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    print("== run_layer_norm_gpu")

    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var data_h = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var res = ctx.enqueue_create_host_buffer[dtype](rows * cols)
    var gamma_h = ctx.enqueue_create_host_buffer[dtype](cols)
    var beta_h = ctx.enqueue_create_host_buffer[dtype](cols)

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        data_h[i] = val

    for i in range(cols):
        gamma_h[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_h[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var data_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    # Distinct output buffer: input_fn (reads) and output_fn (writes) are
    # separate value-closure args, so they must reference distinct buffer
    # origins (writing in place into `data_d` would alias the read).
    var out_d = ctx.enqueue_create_buffer[dtype](rows * cols)
    var gamma_d = ctx.enqueue_create_buffer[dtype](cols)
    var beta_d = ctx.enqueue_create_buffer[dtype](cols)

    var param_shape = Index(cols)

    var data_buf = TileTensor(data_d, row_major(Coord(shape)))
    var out_buf = TileTensor(out_d, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_d, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_d, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype]()

    ctx.enqueue_copy(data_d, data_h)
    ctx.enqueue_copy(gamma_d, gamma_h)
    ctx.enqueue_copy(beta_d, beta_h)

    @always_inline
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var data_buf} -> SIMD[dtype, width]:
        var idx = data_buf.layout(coords)

        return data_buf.raw_load[
            width=width, alignment=alignment * align_of[dtype]()
        ](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var out_buf}:
        var idx = out_buf.layout(coords)
        out_buf.raw_store[width=width, alignment=alignment * align_of[dtype]()](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm[dtype, rank, target="gpu"](
        input_fn,
        output_fn,
        Coord(shape),
        Int(cols),
        gamma,
        beta,
        epsilon,
        ctx,
    )
    ctx.enqueue_copy(res, out_d)
    ctx.synchronize()

    for r in range(rows):
        var vec = TileTensor(
            data_h.unsafe_ptr() + r * cols,
            row_major(cols),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
        for c in range(cols):
            var idx = r * cols + c
            var val = ((data_h[idx] - mean_ref) * norm_factor_ref) * gamma_h[
                c
            ] + beta_h[c]
            assert_almost_equal(val, res[idx], rtol=rtol)

    _ = data_d
    _ = gamma_d
    _ = beta_d


def main() raises:
    with DeviceContext() as ctx:
        # End-to-end layer_norm across shapes. The blocked/warp-tiled kernel
        # selection is a scaffolder implementation detail, so only the op's
        # end-to-end behavior is tested here.
        run_layer_norm_gpu[.float32](ctx, Index(3, 5))
        run_layer_norm_gpu[.float32](ctx, Index(3, 8))
        run_layer_norm_gpu[.float32](ctx, Index(7, 33))
        run_layer_norm_gpu[.float32](ctx, Index(1, 1024))
        run_layer_norm_gpu[.float32](ctx, Index(1, 8192), rtol=0.1)
        run_layer_norm_gpu[.float32](ctx, Index(10, 4096))
        # variable rank
        run_layer_norm_gpu[.float32](ctx, Index(5))
        run_layer_norm_gpu[.float32](ctx, Index(3, 4, 10, 20, 8))
        run_layer_norm_gpu[.float32](ctx, Index(1, 5, 6, 10, 128))
