# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# ===----------------------------------------------------------------------=== #
#
# Benchmarking performance of vectorize over basic operations
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo %s -t | FileCheck %s
# CHECK: Benchmark results

from random import rand
from algorithm import vectorize

from benchmark import Unit, run
from benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)

from memory.unsafe import DTypePointer
from memory import memcmp
from buffer import Buffer


@value
struct Op(Stringable):
    var op_code: Int
    alias add = 0
    alias sub = 1
    alias mul = 2
    alias div = 3
    alias fma = 4
    alias ld = 5
    alias st = 6

    @always_inline("nodebug")
    fn __eq__(self, other: Op) -> Bool:
        return self.op_code == other.op_code

    @always_inline("nodebug")
    fn __str__(self) -> String:
        alias op_list = List[String](
            "add", "sub", "mul", "div", "fma", "ld", "st"
        )
        return "op." + op_list[self.op_code]


fn test_vectorize[
    N: Int,
    simd_width: Int = 1,
    op: Op = Op.add,
    const_operand: Bool = True,
    dtype: DType = DType.float32,
    unroll_factor: Int = 1,
](inout m: Bench) raises:
    constrained[(N % simd_width) == 0]()
    # Create a mem of size N
    alias buffer_align = 64
    var vector = DTypePointer[dtype].alloc(N, alignment=buffer_align)
    var result = DTypePointer[dtype].alloc(N, alignment=buffer_align)

    @always_inline
    @parameter
    fn ld_vector[simd_width: Int](idx: Int):
        SIMD[size=simd_width].store(
            vector, idx + 1, SIMD[vector.type, simd_width](idx)
        )

    @always_inline
    @parameter
    fn st_vector[simd_width: Int](idx: Int):
        SIMD[size=simd_width].store(
            result, idx, SIMD[size=simd_width].load(vector, idx)
        )

    @__copy_capture(vector)
    @always_inline
    @parameter
    fn arithmetic_const[simd_width: Int](idx: Int):
        alias x: Scalar[dtype] = 2
        alias y: Scalar[dtype] = 3

        @parameter
        if op == Op.add:
            SIMD[size=simd_width].store(
                vector, idx, SIMD[size=simd_width].load(vector, idx) + x
            )
        elif op == Op.sub:
            SIMD[size=simd_width].store(
                vector, idx, SIMD[size=simd_width].load(vector, idx) - x
            )
        elif op == Op.mul:
            SIMD[size=simd_width].store(
                vector, idx, SIMD[size=simd_width].load(vector, idx) * x
            )
        elif op == Op.div:
            SIMD[size=simd_width].store(
                vector, idx, SIMD[size=simd_width].load(vector, idx) / x
            )
        elif op == Op.fma:
            SIMD[size=simd_width].store(
                vector, idx, SIMD[size=simd_width].load(vector, idx) * x + y
            )

    @__copy_capture(vector)
    @always_inline
    @parameter
    fn arithmetic_vector[simd_width: Int](idx: Int):
        @parameter
        if op == Op.add:
            SIMD[size=simd_width].store(
                vector,
                idx,
                SIMD[size=simd_width].load(vector, idx)
                + SIMD[size=simd_width].load(vector, idx),
            )
        elif op == Op.sub:
            SIMD[size=simd_width].store(
                vector,
                idx,
                SIMD[size=simd_width].load(vector, idx)
                - SIMD[size=simd_width].load(vector, idx),
            )
        elif op == Op.mul:
            SIMD[size=simd_width].store(
                vector,
                idx,
                SIMD[size=simd_width].load(vector, idx)
                * SIMD[size=simd_width].load(vector, idx),
            )
        elif op == Op.div:
            SIMD[size=simd_width].store(
                vector,
                idx,
                SIMD[size=simd_width].load(vector, idx)
                / SIMD[size=simd_width].load(vector, idx),
            )
        elif op == Op.fma:
            SIMD[size=simd_width].store(
                vector,
                idx,
                SIMD[size=simd_width].load(vector, idx)
                * SIMD[size=simd_width].load(vector, idx)
                + SIMD[size=simd_width].load(vector, idx),
            )

    @always_inline
    @parameter
    fn bench_(inout b: Bencher):
        @always_inline
        @parameter
        fn call_fn():
            @parameter
            if op == Op.ld:
                vectorize[ld_vector, simd_width, unroll_factor=unroll_factor](N)
            elif op == Op.st:
                vectorize[st_vector, simd_width, unroll_factor=unroll_factor](N)
            else:

                @parameter
                if const_operand:
                    vectorize[
                        arithmetic_const,
                        simd_width,
                        unroll_factor=unroll_factor,
                    ](N)
                else:
                    vectorize[
                        arithmetic_vector,
                        simd_width,
                        unroll_factor=unroll_factor,
                    ](N)
            keep(vector)

        b.iter[call_fn]()

    var bench_id = BenchId(
        str(op)
        + ", N="
        + str(N)
        + ", simd="
        + str(simd_width)
        + ", const_operand="
        + str(const_operand)
        + ", dtype="
        + str(dtype)
        + ", unroll_factor="
        + str(unroll_factor)
    )

    m.bench_function[bench_](
        bench_id, ThroughputMeasure(BenchMetric.elements, N)
    )
    vector.free()
    result.free()


# TODO: move this function to a common module for benchmarking.
@always_inline
fn unroll_nested_call[
    func: fn[List[Int]] () raises capturing -> None,
    count: List[Int],  # TODO: a better datatype to use? e.g., Dim?
    loop_idx: Int = 0,
    index_prev: List[Int] = List[Int](),
]() raises:
    """Fully unroll a nested loop of depth `depth` and call function `func`
    at the innermost loop with a List of indices from all levels as arguments.

    for loop_idx0 in range(count[0]):
        for loop_idx1 in range(count[1]):
            for loop_idx2 in range(count[2]):
                func(List[loop_idx0, loop_idx1, loop_idx2])

    Params:
    - func: function to call at the innermost loop.
    - count: List[Int] contains the total count of iterations for each loop,
    outmost loop is at index=0, inner-most loop at index=depth-1
    - loop_idx: index of the current loop
    - index_prev: List[Int] of all indices from outer loops.
    """
    alias depth = len(count)

    @always_inline
    @parameter
    fn append_index(x: List[Int], y: Int) -> List[Int]:
        var z = x
        z.append(y)
        return z

    @parameter
    for i in range(count[loop_idx]):
        alias index = append_index(index_prev, i)

        @parameter
        if loop_idx < depth - 1:
            unroll_nested_call[func, count, loop_idx + 1, index]()
        else:
            func[index]()


fn bench_compare():
    alias type = DType.uint8
    alias width = simdwidthof[type]()
    alias unit = Unit.ns
    # increasing will reduce the benefit of passing the size as a paramater
    alias multiplier = 2
    # Add .5 of the elements that fit into a simd register
    alias size: Int = int(multiplier * width + (width * 0.5))
    alias unroll_factor = 2
    alias its = 1000

    var p1 = DTypePointer[type].alloc(size)
    var p2 = DTypePointer[type].alloc(size)
    print("Benchmark results")
    rand(p1, size)

    @parameter
    fn arg_size():
        @parameter
        fn closure[width: Int](i: Int):
            SIMD.store(
                p2,
                i,
                SIMD[size=width].load(p1, i) + SIMD[size=width].load(p2, i),
            )

        for i in range(its):
            vectorize[closure, width](size)

    @parameter
    fn param_size():
        @parameter
        fn closure[width: Int](i: Int):
            SIMD.store(
                p2,
                i,
                SIMD[size=width].load(p1, i) + SIMD[size=width].load(p2, i),
            )

        for i in range(its):
            vectorize[closure, width, size=size]()

    @parameter
    fn arg_size_unroll():
        @parameter
        fn closure[width: Int](i: Int):
            SIMD.store(
                p2,
                i,
                SIMD[size=width].load(p1, i) + SIMD[size=width].load(p2, i),
            )

        for i in range(its):
            vectorize[closure, width, unroll_factor=unroll_factor](size)

    @parameter
    fn param_size_unroll():
        @parameter
        fn closure[width: Int](i: Int):
            SIMD.store(
                p2,
                i,
                SIMD[size=width].load(p1, i) + SIMD[size=width].load(p2, i),
            )

        for i in range(its):
            vectorize[closure, width, size=size, unroll_factor=unroll_factor]()

    var arg = run[arg_size](max_runtime_secs=0.5).mean(unit)
    print(SIMD[size=size].load(p2))
    memset_zero(p2, size)

    var param = run[param_size](max_runtime_secs=0.5).mean(unit)
    print(SIMD[size=size].load(p2))
    memset_zero(p2, size)

    var arg_unroll = run[arg_size_unroll](max_runtime_secs=0.5).mean(unit)
    print(SIMD[size=size].load(p2))
    memset_zero(p2, size)

    var param_unroll = run[param_size_unroll](max_runtime_secs=0.5).mean(unit)
    print(SIMD[size=size].load(p2))

    print(
        "calculating",
        size,
        "elements,",
        width,
        "elements fit into the SIMD register\n",
    )

    print(" size as argument:", arg, unit)
    print("         unrolled:", arg_unroll, unit)
    print()
    print("size as parameter:", param, unit)
    print("         unrolled:", param_unroll, unit)
    print(
        "\nPassing size as a parameter and unrolling is",
        arg_unroll / param_unroll,
        "x faster",
    )
    p1.free()
    p2.free()


fn main() raises:
    alias vec_size_list = List[Int](512, 2048)
    alias simd_width_list = List[Int](1, 2, 4, 8)
    alias op_list = List[Op](Op.add, Op.mul, Op.fma, Op.ld)
    alias const_operand_list = List[Bool](True)
    alias dtype_list = List[DType](DType.float32)
    alias unroll_factor_list = List[Int](1, 2, 4)

    var m = Bench()

    @always_inline
    @parameter
    fn callback[index: List[Int]]() raises:
        @parameter
        if len(index) == 6:
            alias vec_size = vec_size_list[index[0]]
            alias simd_width = simd_width_list[index[1]]
            alias op_code = op_list[index[2]]
            alias const_op = const_operand_list[index[3]]
            alias dtype = dtype_list[index[4]]
            alias unroll_factor = unroll_factor_list[index[5]]
            test_vectorize[
                vec_size, simd_width, op_code, const_op, dtype, unroll_factor
            ](m)

    unroll_nested_call[
        callback,
        List[Int](
            len(vec_size_list),
            len(simd_width_list),
            len(op_list),
            len(const_operand_list),
            len(dtype_list),
            len(unroll_factor_list),
        ),
    ]()

    m.dump_report()
