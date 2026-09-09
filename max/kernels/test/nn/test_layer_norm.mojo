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

from std.itertools import product
from layout import Coord, Idx, TileTensor, row_major
from layout.math import mean, variance
from nn.normalization import *
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def run_layer_norm_cpu[
    dtype: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.01) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = List(length=rows * cols, fill=Scalar[dtype](0))
    var output_ptr = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma_ptr = List(length=cols, fill=Scalar[dtype](0))
    var beta_ptr = List(length=cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        var val = Scalar[dtype](i)
        input_ptr[i] = val

    for i in range(cols):
        gamma_ptr[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()
        beta_ptr[i] = (Float64(i) / Float64(cols)).cast[dtype]()

    var param_shape = IndexList[1](cols)

    var input_buf = TileTensor(input_ptr, row_major(Coord(shape)))
    var output_buf = TileTensor(output_ptr, row_major(Coord(shape)))
    var gamma = TileTensor(gamma_ptr, row_major(Coord(param_shape)))
    var beta = TileTensor(beta_ptr, row_major(Coord(param_shape)))
    var epsilon = Scalar[dtype](0.0001)

    @always_inline
    def input_fn[
        width: Int,
        alignment: Int,
    ](coords: Coord) {var input_buf} -> SIMD[dtype, width]:
        var idx = input_buf.layout(coords)
        return input_buf.raw_load[width=width, alignment=alignment](idx)

    @always_inline
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) {var output_buf}:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[width=width, alignment=alignment](
            idx, rebind[SIMD[dtype, width]](val)
        )

    layer_norm[dtype, rank, target="cpu"](
        input_fn,
        output_fn,
        Coord(shape),
        Int(cols),
        gamma,
        beta,
        epsilon,
    )

    var input_ptr_ptr: MutPointer[
        input_ptr.T, origin_of(input_ptr)
    ] = input_ptr.unsafe_ptr()
    for r, c in product(range(rows), range(cols)):
        var vec = TileTensor(
            input_ptr_ptr + r * cols,
            row_major(cols),
        )
        var mean_ref = mean(vec)
        var var_ref = variance(vec, correction=0)
        var norm_factor_ref = rsqrt(var_ref + epsilon)
        var idx = r * cols + c
        var val = ((input_ptr[idx] - mean_ref) * norm_factor_ref) * gamma_ptr[
            c
        ] + beta_ptr[c]
        assert_almost_equal(val, output_ptr[idx], rtol=rtol)


def main() raises:
    print("0")
    run_layer_norm_cpu[.float32](Index(3, 5))
    print("1")
    run_layer_norm_cpu[.float32](Index(3, 8))
    print("2")
    run_layer_norm_cpu[.float32](Index(7, 33))
    print("3")
    run_layer_norm_cpu[.float32](Index(1, 1024))
    print("4")
    run_layer_norm_cpu[.float32](Index(1, 8192))

    # variable rank
    print("5")
    run_layer_norm_cpu[.float32](Index(0))
    print("6")
    run_layer_norm_cpu[.float32](Index(5))
    print("7")
    run_layer_norm_cpu[.float32](Index(3, 4, 10, 20, 8))
    print("8")
    run_layer_norm_cpu[.float32](Index(1, 5, 6, 10, 128))

    # float64 regression for KERN-3270: simd_width is 4 on AVX2, so
    # num_cols=5 hits the vector loop plus a scalar tail, the shape that
    # segfaulted.
    print("9")
    run_layer_norm_cpu[.float64](Index(4, 5))
    print("10")
    run_layer_norm_cpu[.float64](Index(3, 5))
    print("11")
    run_layer_norm_cpu[.float64](Index(7, 33))
