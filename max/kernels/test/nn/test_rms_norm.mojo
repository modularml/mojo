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

from std.math import sqrt
from std.sys.info import CompilationTarget

from std.itertools import product
from layout import Coord, Idx, TileTensor, row_major
from nn.normalization import *
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def compute_rms[
    dtype: DType
](data: TileTensor[dtype, ...], size: Int, eps: Float32) -> Scalar[.float32]:
    comptime assert data.rank == 1, "data.rank must be 1"
    var sum_of_squares = Float32()
    for i in range(size):
        var d = data.raw_load(i).cast[.float32]()
        sum_of_squares += d * d
    return sqrt((sum_of_squares / Float32(data.num_elements())) + eps)


def run_rms_norm_cpu[
    dtype: DType, rank: Int
](shape: IndexList[rank], rtol: Float64 = 0.001) raises:
    var cols = shape[rank - 1]
    var rows = shape.flattened_length() // cols

    var input_ptr = List(length=rows * cols, fill=Scalar[dtype](0))
    var output_ptr = List(length=rows * cols, fill=Scalar[dtype](0))
    var gamma_ptr = List(length=cols, fill=Scalar[dtype](0))

    for i in range(rows * cols):
        input_ptr[i] = Scalar[dtype](i)

    for i in range(cols):
        gamma_ptr[i] = (Float64(i + cols) / Float64(cols)).cast[dtype]()

    var param_shape = Index(cols)

    var input_buf = TileTensor(input_ptr, row_major(Coord(shape)))
    var output_buf = TileTensor(output_ptr, row_major(Coord(shape)))
    var gamma = TileTensor(
        gamma_ptr,
        row_major(Coord(param_shape)),
    )
    var epsilon = Float32(0.0001)
    var weight_offset = Scalar[dtype](0.0)

    @__copy_capture(input_buf)
    @always_inline
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[dtype, width]:
        var idx = input_buf.layout(coords)
        return input_buf.raw_load[width=width](idx)

    @always_inline
    @__copy_capture(output_buf)
    @__parameter
    def identity_output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[dtype, width]) -> None:
        var idx = output_buf.layout(coords)
        output_buf.raw_store[width=width, alignment=alignment](idx, val)

    rms_norm_cpu[input_fn, identity_output_fn, multiply_before_cast=True](
        Coord(shape),
        gamma,
        epsilon,
        weight_offset,
    )

    var input_ptr_ptr: MutPointer[
        input_ptr.T, origin_of(input_ptr)
    ] = input_ptr.unsafe_ptr()
    for r, c in product(range(rows), range(cols)):
        var vec = TileTensor(
            input_ptr_ptr + r * cols,
            row_major(cols),
        )
        var rms_ref = compute_rms(vec, cols, epsilon)
        var idx = r * cols + c
        # PyTorch converts the input to float32 before computing the RMS norm
        # https://github.com/meta-llama/llama/blob/689c7f261b9c5514636ecc3c5fefefcbb3e6eed7/llama/model.py#L76
        var val = (input_ptr[idx].cast[.float32]() / rms_ref).cast[dtype]() * (
            gamma_ptr[c] + weight_offset
        )
        assert_almost_equal(val, output_ptr[idx], rtol=rtol)


def run_rms_norm_tests[dtype: DType](rtol: Float64 = 0.001) raises:
    run_rms_norm_cpu[dtype](Index(15, 11), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 5), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 55), rtol)
    # run_rms_norm_cpu[dtype](Index(7, 557), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 8191), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 8192), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 16384), rtol)
    # run_rms_norm_cpu[dtype](Index(2, 16385), rtol)

    # # variable rank
    # run_rms_norm_cpu[dtype](Index(0), rtol)
    run_rms_norm_cpu[dtype](Index(5), rtol)
    run_rms_norm_cpu[dtype](Index(3, 4, 10, 20, 8), rtol)
    # run_rms_norm_cpu[dtype](Index(1, 5, 6, 10, 128), rtol)


def main() raises:
    run_rms_norm_tests[.float32]()

    comptime if not CompilationTarget.has_neon():
        run_rms_norm_tests[.bfloat16](rtol=1e-2)
