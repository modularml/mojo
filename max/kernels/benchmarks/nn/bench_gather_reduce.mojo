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

from std.random import random_si64
from std.sys import simd_width_of, size_of

from std.benchmark import Bench, Bencher, BenchId
from layout import Coord, TileTensor, row_major
from nn.gather_scatter import gather_reduce

from std.utils import IndexList


@always_inline
def add(x: SIMD, y: type_of(x)) -> type_of(x):
    return x + y


def bench_gather_reduce(mut b: Bencher):
    comptime type = DType.float32
    var num_rows = 500000
    var embedding_dim = 32
    var multi_hot_dim = 100
    comptime l3_size = 32  # mb
    comptime clear_size = l3_size * 2 * 1_000_000
    var num_indices = clear_size // (
        size_of[type]() * embedding_dim * multi_hot_dim
    )
    var input_shape = IndexList[2](num_rows, embedding_dim)
    var output_shape = IndexList[2](num_indices, embedding_dim)
    var indices_shape = IndexList[2](num_indices, multi_hot_dim)
    var input_storage = List(
        length=input_shape.flattened_length(), fill=Scalar[type](1)
    )
    var output_storage = List(
        length=output_shape.flattened_length(), fill=Scalar[type](0)
    )
    var indices_storage = List(
        length=indices_shape.flattened_length(), fill=Int32(0)
    )
    var input = TileTensor(input_storage, row_major(Coord(input_shape)))
    var output = TileTensor(output_storage, row_major(Coord(output_shape)))
    var indices = TileTensor(indices_storage, row_major(Coord(indices_shape)))
    for i in range(Int(indices.dim[0]())):
        for j in range(Int(indices.dim[1]())):
            indices[i, j] = random_si64(0, num_rows).cast[.int32]()

    def to_bench() {imm}:
        gather_reduce[type, 0, 1, simd_width_of[type](), add](
            output,
            input,
            indices,
            0,
        )

    b.iter(to_bench)

    print(output[0, 0])


def main() raises:
    var m = Bench()
    m.bench_function(
        bench_gather_reduce, BenchId("gather_reduce_dlrm1_multihot")
    )
    m.dump_report()
