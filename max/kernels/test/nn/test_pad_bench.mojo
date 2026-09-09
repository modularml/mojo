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

from std.os import abort

import std.benchmark
from std.benchmark import Unit, keep
from layout import Coord, TileTensor, row_major
from nn.pad import pad_constant, pad_reflect
from std.python import Python

from std.utils import IndexList, product


def pretty_print(
    name: String,
    size: Int,
    rounds: Int,
    time: Float64,
    msg: String,
) raises:
    var py = Python.import_module("builtins")
    _ = py.print(
        py.str("[{:<20}|rounds={}|size={:<4}] time: {:>8.3f} (ms) {}").format(
            name,
            String(rounds),
            String(size),
            time,
            msg,
        )
    )


def bench[
    func: def[rank: Int, size: Int, verify: Bool = False]() thin raises -> None,
    rank: Int,
    size: Int,
    name: String,
]() raises:
    comptime N = 100

    def runner():
        try:
            for _ in range(N):
                var result = func[rank, size]()
                keep(result)
        except e:
            abort(String(e))

    var ms = std.benchmark.run(runner, 1, 10)

    pretty_print(
        name,
        size,
        N,
        ms.mean(Unit.ms),
        "",
    )


def test_pad_constant_nd[rank: Int, n: Int, verify: Bool = False]() raises:
    comptime d_pre = 3
    comptime d_post = 7
    comptime d = d_pre + d_post

    @always_inline
    def get_in_out_shapes[rank: Int = 1]() -> Array[IndexList[rank], 2]:
        var in_shape = IndexList[rank]()
        var out_shape = IndexList[rank]()

        comptime if rank == 1:
            in_shape = [n]
            out_shape = [n + d]
        elif rank == 2:
            in_shape = [n, n]
            out_shape = [n + d, n + d]
        elif rank == 3:
            in_shape = [n, n, n]
            out_shape = [n + d, n + d, n + d]
        elif rank == 4:
            in_shape = [n, n, n, n]
            out_shape = [n + d, n + d, n + d, n + d]
        return [in_shape, out_shape]

    comptime in_out_shape = get_in_out_shapes[rank]()
    comptime in_shape = in_out_shape[0]
    comptime out_shape = in_out_shape[1]

    comptime in_size = product(in_shape)
    comptime out_size = product(out_shape)

    # create a big input matrix and fill it with 1
    var input_ptr = List(length=in_size, fill=Int(0))
    var input = TileTensor(
        input_ptr,
        row_major(Coord(in_shape)),
    ).fill(1)

    # Create a padding array
    var paddings_stack = Array[Int, 2 * rank](fill={})
    var paddings = TileTensor(paddings_stack, row_major[2 * rank]())

    comptime for i in range(rank):
        paddings[2 * i] = d_pre
        paddings[2 * i + 1] = d_post

    # Create an output matrix and fill with 0
    var output_ptr = List(length=out_size, fill=Int(0))
    var output = TileTensor(
        output_ptr,
        row_major(Coord(out_shape)),
    )

    # constant padding value = 7
    var constant = Int(7)

    # pad
    pad_constant(output, input, paddings._storage, constant)

    if verify:
        # Simple verification: check that the padding values are correct
        # and the center contains the input values
        for i in range(out_size):
            # Just verify no crash occurs and values are set
            _ = output.raw_load(i)
    _ = output_ptr^
    _ = input_ptr^


def test_pad_reflect_nd[rank: Int, n: Int, verify: Bool = False]() raises:
    comptime d_pre = 3
    comptime d_post = 7
    comptime d = d_pre + d_post

    @always_inline
    def get_in_out_shapes[rank: Int = 1]() -> Array[IndexList[rank], 2]:
        var in_shape = IndexList[rank]()
        var out_shape = IndexList[rank]()

        comptime if rank == 1:
            in_shape = [n]
            out_shape = [n + d]
        elif rank == 2:
            in_shape = [n, n]
            out_shape = [n + d, n + d]
        elif rank == 3:
            in_shape = [n, n, n]
            out_shape = [n + d, n + d, n + d]
        elif rank == 4:
            in_shape = [n, n, n, n]
            out_shape = [n + d, n + d, n + d, n + d]
        return [in_shape, out_shape]

    comptime in_out_shape = get_in_out_shapes[rank]()
    comptime in_shape = in_out_shape[0]
    comptime out_shape = in_out_shape[1]

    comptime in_size = product(in_shape)
    comptime out_size = product(out_shape)

    # create a big input matrix and fill it with 1
    var input_ptr = List(length=in_size, fill=Int(0))
    var input = TileTensor(
        input_ptr,
        row_major(Coord(in_shape)),
    ).fill(1)

    # Create a padding array
    var paddings_stack = Array[Int, 2 * rank](fill={})
    var paddings = TileTensor(paddings_stack, row_major[2 * rank]())

    comptime for i in range(rank):
        paddings[2 * i] = d_pre
        paddings[2 * i + 1] = d_post

    # Create an output matrix and fill with 0
    var output_ptr = List(length=out_size, fill=Int(0))
    var output = TileTensor(
        output_ptr,
        row_major(Coord(out_shape)),
    )

    # pad
    pad_reflect(output, input, paddings._storage)

    if verify:
        # Simple verification: check that values are set
        for i in range(out_size):
            _ = output.raw_load(i)
    _ = output_ptr^
    _ = input_ptr^


# CHECK-LABEL: test_pad_iterative
def main() raises:
    print("== test_pad_iterative")

    def _all[N: Int]() raises:
        bench[test_pad_constant_nd, 1, N, "test_pad_constant_1d"]()
        bench[test_pad_constant_nd, 2, N, "test_pad_constant_2d"]()
        bench[test_pad_constant_nd, 3, N, "test_pad_constant_3d"]()
        # bench[test_pad_constant_nd, 4, N, "test_pad_constant_4d"]()

        bench[test_pad_reflect_nd, 1, N, "test_pad_reflect_1d"]()
        bench[test_pad_reflect_nd, 2, N, "test_pad_reflect_2d"]()
        bench[test_pad_reflect_nd, 3, N, "test_pad_reflect_3d"]()
        # bench[test_pad_reflect_nd, 4, N, "test_pad_reflect_4d"]()

    # _all[64]()
    # _all[128]()
    # _all[256]()
    # _all[512]()
    # _all[1024]()

    test_pad_constant_nd[1, 64, True]()
    test_pad_constant_nd[2, 64, True]()
    test_pad_constant_nd[3, 64, True]()

    test_pad_reflect_nd[1, 64, True]()
    test_pad_reflect_nd[2, 64, True]()
    test_pad_reflect_nd[3, 64, True]()
