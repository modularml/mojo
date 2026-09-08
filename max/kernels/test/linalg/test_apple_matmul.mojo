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
# Checks Apple cblas_sgemm matmul C = A*B and apple_gemv, when called from
# Matmul.mojo functions
#
# ===----------------------------------------------------------------------=== #

from std.collections import Optional

import std.benchmark
from layout import Coord, Idx, TileTensor, row_major
from layout.tensor_engine import DefaultEngine
from std.memory import alloc
from linalg.bmm import batched_matmul
from linalg.matmul import matmul
from linalg.matmul.cpu import matmul as _matmul_cpu
from linalg.packing import (
    _pack_b_ndbuffer_impl,
    _pack_matmul_b_shape_func_impl,
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
    pack_transposed_b_ndbuffer,
)
from linalg.utils import elementwise_epilogue_type
from std.testing import assert_almost_equal, assert_true

from std.utils.index import IndexList

comptime alignment = 64
comptime some_constant = 20
comptime do_benchmarking = False


def bench_run(
    func: Some[ImplicitlyCopyable & (def() raises)],
) raises -> std.benchmark.Report:
    return std.benchmark.run(func, 2, 1_000_000, 1, 3)


def gemm_naive[
    transpose_b: Bool
](
    a: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    b: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    c: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert a.flat_rank == 2
    comptime assert b.flat_rank == 2
    comptime assert c.flat_rank == 2
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.dtype]()
                var b_val = (b[j, p] if transpose_b else b[p, j]).cast[
                    c.dtype
                ]()
                c[i, j] += a_val * b_val


def gemm_naive_elementwise[
    transpose_b: Bool
](
    a: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    b: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    c: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    m: Int,
    n: Int,
    k: Int,
    val: Int,
):
    comptime assert a.flat_rank == 2
    comptime assert b.flat_rank == 2
    comptime assert c.flat_rank == 2
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a[i, p].cast[c.dtype]()
                var b_val = (b[j, p] if transpose_b else b[p, j]).cast[
                    c.dtype
                ]()
                c[i, j] += a_val * b_val

    for i in range(m):
        for j in range(n):
            c[i, j] += Scalar[c.dtype](val)


def test_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    b_packed: Bool,
    epilogue_fn: Optional[elementwise_epilogue_type],
](
    c: TileTensor[
        mut=True,
        c_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    a: TileTensor[
        mut=False,
        a_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    b: TileTensor[
        mut=False,
        b_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    bp: TileTensor[
        mut=True,
        b_type,
        Engine=DefaultEngine[element_width=1],
        address_space=.GENERIC,
        ...,
    ],
    m: Int,
    n: Int,
    k: Int,
    kernel_type_m: Int,
) raises -> Int:
    var c1_ptr = alloc[Scalar[c_type]](m * n, alignment=alignment)
    var golden_shape = row_major(Coord(m, n))
    var golden = TileTensor[Engine=DefaultEngine[element_width=1]](
        c1_ptr, golden_shape
    )
    for i in range(m):
        for j in range(n):
            golden[i, j] = 0

    if b_packed:
        if not transpose_b:
            if kernel_type_m != 0:
                _pack_b_ndbuffer_impl[a_type, c_type, transpose_b](
                    b, bp, kernel_type_m
                )
            else:
                pack_b_ndbuffer[a_type, c_type](b, bp)
        else:
            if kernel_type_m != 0:
                _pack_b_ndbuffer_impl[
                    a_type,
                    c_type,
                    transpose_b,
                ](b, bp, kernel_type_m)
            else:
                pack_transposed_b_ndbuffer[a_type, c_type](b, bp)

    @always_inline
    def bench_fn_matmul() raises {var}:
        if kernel_type_m != 0:
            _matmul_cpu[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=epilogue_fn,
            ](
                c,
                a,
                bp,
                kernel_type_m,
            )
        else:
            matmul[
                transpose_b=transpose_b,
                b_packed=b_packed,
                elementwise_lambda_fn=epilogue_fn,
            ](c, a, bp)

    bench_fn_matmul()

    comptime if do_benchmarking:
        var matmul_perf = bench_run(bench_fn_matmul)
        std.benchmark.keep(c[Coord(Idx[0], Idx[0])])
        print(
            "Apple Matmul GFLOP/s for (M, N, K) = (",
            m,
            n,
            k,
            "): ",
            1e-9 * (Float64((2 * m * k * n)) / matmul_perf.mean()),
        )

    comptime if epilogue_fn:
        gemm_naive_elementwise[transpose_b](
            a, b, golden, m, n, k, some_constant
        )
    else:
        gemm_naive[transpose_b](a, b, golden, m, n, k)

    var errors: Int = 0
    comptime assert c.flat_rank == 2
    for i in range(m):
        for j in range(n):
            if c[i, j] != golden[i, j]:
                assert_almost_equal(
                    c[i, j],
                    golden[i, j],
                    msg=String(
                        "values do not agree for ",
                        m,
                        "x",
                        n,
                        "x",
                        k,
                        " using the dtype=",
                        a_type,
                        ",",
                        b_type,
                        ",",
                        c_type,
                    ),
                )

    c1_ptr.free()
    return errors


def test_matmul[
    has_epilogue_fusion: Bool,
    *,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
    transpose_b: Bool,
](m: Int, n: Int, k: Int) raises:
    print("== test_matmul")
    var errors: Int
    var kernel_type_m = m if mixed_kernels else 0

    var a_ptr = alloc[Scalar[a_type]](m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type]](k * n, alignment=alignment)
    var b = TileTensor(
        b_ptr,
        row_major(Coord(n, k) if transpose_b else Coord(k, n)),
    )

    var padded_n_k: IndexList[2]
    if kernel_type_m != 0:
        padded_n_k = _pack_matmul_b_shape_func_impl[
            a_type,
            c_type,
            transpose_b,
        ](b, kernel_type_m)
    else:
        padded_n_k = pack_matmul_b_shape_func[a_type, c_type, transpose_b](
            TileTensor(b)
        )

    var padded_n = (
        padded_n_k[1] if b_packed or (not b_packed and transpose_b) else n
    )
    var padded_k = (
        padded_n_k[0] if b_packed or (not b_packed and transpose_b) else k
    )

    var c0_ptr = alloc[Scalar[c_type]](m * n, alignment=alignment)

    var bp_ptr = alloc[Scalar[b_type]](padded_k * padded_n, alignment=alignment)

    var bp = TileTensor(bp_ptr, row_major(padded_k, padded_n))
    var a = TileTensor(a_ptr, row_major(m, k))
    var c = TileTensor(c0_ptr, row_major(m, n))

    for i in range(m):
        for p in range(k):
            a[i, p] = Scalar[a_type](0.001) * Scalar[a_type](i)

    for p in range(n if transpose_b else k):
        for j in range(k if transpose_b else n):
            b[p, j] = Scalar[b_type](0.002) * Scalar[b_type](p)
            if b_packed and not transpose_b:
                bp[j, p] = b[p, j]
            else:
                bp[p, j] = b[p, j]

    for i in range(m):
        for j in range(n):
            c[i, j] = 0

    @__parameter
    @always_inline
    @__copy_capture(c)
    def epilogue_fn[
        _type: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[_type, width]) -> None:
        c.store(Coord(idx), rebind[SIMD[c_type, width]](val + some_constant))

    comptime if has_epilogue_fusion:
        errors = test_matmul[
            a_type,
            b_type,
            c_type,
            transpose_b,  # transpose_b
            b_packed,  # b_packed
            epilogue_fn,
        ](
            c,
            a,
            b,
            bp,
            m,
            n,
            k,
            m if mixed_kernels else 0,
        )
    else:
        errors = test_matmul[
            a_type,
            b_type,
            c_type,
            transpose_b,  # transpose_b
            b_packed,  # b_packed
            None,
        ](
            c,
            a,
            b,
            bp,
            m,
            n,
            k,
            m if mixed_kernels else 0,
        )
    if errors > 0:
        return
    print("Success")

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()


def test_shapes[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    mixed_kernels: Bool,
]() raises:
    @__parameter
    def test_shapes_helper[
        transpose_b: Bool = False
    ](m: Int, n: Int, k: Int) raises:
        # Test without output fusion.
        test_matmul[
            False,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
            transpose_b=transpose_b,
        ](m, n, k)
        # Test with output fusion.
        test_matmul[
            True,
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            mixed_kernels=mixed_kernels,
            transpose_b=transpose_b,
        ](m, n, k)

    # Test various matmul and gemv shapes with and without transpose_b.

    # Test with transpose_b = False
    test_shapes_helper(256, 1024, 4096)
    test_shapes_helper(4, 5, 6)
    test_shapes_helper(15, 16, 17)
    test_shapes_helper(24, 32, 64)
    test_shapes_helper(61, 73, 79)
    test_shapes_helper(123, 456, 321)
    test_shapes_helper(256, 256, 256)
    test_shapes_helper(2, 65, 1200)

    # Test with transpose_b = True
    test_shapes_helper[True](256, 1024, 4096)
    test_shapes_helper[True](4, 5, 6)
    test_shapes_helper[True](15, 16, 17)
    test_shapes_helper[True](24, 32, 64)
    test_shapes_helper[True](61, 73, 79)
    test_shapes_helper[True](123, 456, 321)
    test_shapes_helper[True](256, 256, 256)
    test_shapes_helper[True](2, 65, 1200)

    # Test with transpose_b = False
    test_shapes_helper(1, 5120, 3072)
    test_shapes_helper(1, 3072, 3072)
    test_shapes_helper(1, 12288, 3072)
    test_shapes_helper(1, 3072, 12288)
    test_shapes_helper(1, 32768, 3072)

    # Test with transpose_b = True
    test_shapes_helper[True](1, 5120, 3072)
    test_shapes_helper[True](1, 3072, 3072)
    test_shapes_helper[True](1, 12288, 3072)
    test_shapes_helper[True](1, 3072, 12288)
    test_shapes_helper[True](1, 32768, 3072)


def test_types[b_packed: Bool, mixed_kernels: Bool]() raises:
    test_shapes[
        DType.float32,
        DType.float32,
        DType.float32,
        b_packed,
        mixed_kernels,
    ]()


def bmm_naive(
    c: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    a: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    b: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
    val: Int = 0,
    transpose_b: Bool = False,
):
    comptime assert a.flat_rank == 3
    comptime assert b.flat_rank == 3
    comptime assert c.flat_rank == 3
    for batch in range(batches):
        for i in range(m):
            for p in range(k):
                for j in range(n):
                    var a_val = a[batch, i, p].cast[c.dtype]()
                    var b_val: Scalar[c.dtype]
                    if transpose_b:
                        b_val = b[batch, j, p].cast[c.dtype]()
                    else:
                        b_val = b[batch, p, j].cast[c.dtype]()
                    c[batch, i, j] += a_val * b_val

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c[batch, i, j] += Scalar[c.dtype](val)


def test_batched_matmul[
    has_lambda: Bool
](
    c: TileTensor[
        mut=True,
        address_space=.GENERIC,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    a: TileTensor[
        mut=True,
        address_space=.GENERIC,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    b: TileTensor[
        mut=True,
        address_space=.GENERIC,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    comptime assert a.flat_rank == 3
    comptime assert b.flat_rank == 3
    comptime assert c.flat_rank == 3
    var golden_ptr = alloc[Scalar[c.dtype]](
        batches * m * n, alignment=alignment
    )
    var golden_shape = row_major(Coord(batches, m, n))
    var golden = TileTensor(golden_ptr, golden_shape)

    for batch in range(batches):
        for i in range(m):
            for j in range(k):
                a[batch, i, j] = Scalar[a.dtype](i + j) * Scalar[a.dtype](0.001)

    for batch in range(batches):
        for i in range(k):
            for j in range(n):
                b[batch, i, j] = Scalar[b.dtype](i + k) * Scalar[b.dtype](0.001)

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c[batch, i, j] = 0
                golden[batch, i, j] = 0

    @__parameter
    @always_inline
    @__copy_capture(c)
    def epilogue_fn[
        _type: DType,
        width: SIMDLength,
        rank: Int,
        *,
        alignment: Int = 1,
    ](coords: IndexList[rank], val: SIMD[_type, width],) -> None:
        c.store_linear(
            rebind[IndexList[3]](coords),
            rebind[SIMD[c.dtype, width]](val + some_constant),
        )

    @always_inline
    def bench_fn_batched_matmul() raises {var}:
        comptime if has_lambda:
            batched_matmul[
                transpose_a=False,
                transpose_b=False,
                elementwise_epilogue_fn=epilogue_fn,
            ](c, a, b)
        else:
            batched_matmul[
                transpose_a=False,
                transpose_b=False,
            ](c, a, b)

    bench_fn_batched_matmul()

    comptime if do_benchmarking:
        var batched_matmul_perf = bench_run(bench_fn_batched_matmul)
        std.benchmark.keep(c[Coord(Idx[0], Idx[0], Idx[0])])
        print(
            "Apple Batched Matmul GFLOP/s for (BATCHES, M, N, K) = (",
            batches,
            m,
            n,
            k,
            "): ",
            1e-9
            * (Float64((2 * batches * m * k * n)) / batched_matmul_perf.mean()),
        )

    comptime if has_lambda:
        bmm_naive(golden, a, b, batches, m, n, k, some_constant)
    else:
        bmm_naive(golden, a, b, batches, m, n, k)

    var errors: Int = 0
    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                if c[batch, i, j] != golden[batch, i, j]:
                    if errors < 10:
                        print(
                            c[batch, i, j],
                            golden[batch, i, j],
                            c[batch, i, j] - golden[batch, i, j],
                            "at",
                            batch,
                            i,
                            j,
                        )
                    errors += 1

    assert_true(
        errors == 0,
        String(
            "num of errors must be 0, but got ",
            errors,
            " for dimensions Batch=",
            batches,
            " M=",
            m,
            ", N=",
            n,
            ", K=",
            k,
        ),
    )

    golden_ptr.free()


def test_batched_matmul(batch: Int, m: Int, n: Int, k: Int) raises:
    comptime c_type = DType.float32
    comptime a_type = DType.float32
    comptime b_type = DType.float32

    var c_ptr = alloc[Scalar[c_type]](batch * m * n, alignment=alignment)
    var a_ptr = alloc[Scalar[a_type]](batch * m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type]](batch * k * n, alignment=alignment)

    var c_shape = row_major(Coord(batch, m, n))
    var a_shape = row_major(Coord(batch, m, k))
    var b_shape = row_major(Coord(batch, k, n))
    var c = TileTensor(c_ptr, c_shape)
    var a = TileTensor(a_ptr, a_shape)
    var b = TileTensor(b_ptr, b_shape)

    test_batched_matmul[False](c, a, b, batch, m, n, k)
    test_batched_matmul[True](c, a, b, batch, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_batched_matmul() raises:
    for batch in [1, 2, 4, 9, 12]:
        test_batched_matmul(batch, 256, 1024, 4096)
        test_batched_matmul(batch, 4, 5, 6)
        test_batched_matmul(batch, 15, 16, 17)
        test_batched_matmul(batch, 24, 32, 64)
        test_batched_matmul(batch, 61, 73, 79)
        test_batched_matmul(batch, 123, 456, 321)
        test_batched_matmul(batch, 256, 256, 256)
        test_batched_matmul(batch, 2, 65, 1200)


def main() raises:
    test_types[b_packed=True, mixed_kernels=False]()
    test_types[b_packed=True, mixed_kernels=True]()
    test_types[b_packed=False, mixed_kernels=False]()
    test_types[b_packed=False, mixed_kernels=True]()

    test_batched_matmul()
