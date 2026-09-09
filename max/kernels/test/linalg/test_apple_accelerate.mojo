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

from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.cpu.apple_accelerate import (
    apple_batched_matmul,
    apple_matmul,
)
from std.testing import *

from std.utils.index import Index

comptime alignment = 64


comptime a_type = DType.float32
comptime b_type = DType.float32
comptime c_type = DType.float32


def gemm_naive(
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert c.flat_rank >= 2
    comptime assert a.flat_rank >= 2
    comptime assert b.flat_rank >= 2
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var a_val = a.load(Coord(i, p)).cast[c.dtype]()[0]
                var b_val = b.load(Coord(p, j)).cast[c.dtype]()[0]
                var cur = c.load(Coord(i, j))[0]
                c.store(Coord(i, j), cur + a_val * b_val)


def test_matmul(
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=True, ...],
    b: TileTensor[mut=True, ...],
    m: Int,
    n: Int,
    k: Int,
) raises:
    comptime assert c.flat_rank >= 2
    comptime assert a.flat_rank >= 2
    comptime assert b.flat_rank >= 2
    var golden_ptr = alloc[Scalar[c.dtype]](m * n, alignment=alignment)
    var golden = TileTensor(golden_ptr, row_major(Coord(m, n)))

    for i in range(m):
        for j in range(k):
            a.store(
                Coord(i, j),
                Scalar[a.dtype](i + j) * Scalar[a.dtype](0.001),
            )

    for i in range(k):
        for j in range(n):
            b.store(
                Coord(i, j),
                Scalar[b.dtype](i + k) * Scalar[b.dtype](0.001),
            )

    for i in range(m):
        for j in range(n):
            c.store(Coord(i, j), Scalar[c.dtype](0))
            golden.store(Coord(i, j), Scalar[golden.dtype](0))

    apple_matmul(c, a, b)
    gemm_naive(golden, a, b, m, n, k)

    var errors: Int = 0
    for i in range(m):
        for j in range(n):
            if c.load(Coord(i, j)) != golden.load(Coord(i, j)):
                if errors < 10:
                    print(c.load(Coord(i, j)) - golden.load(Coord(i, j)))
                errors += 1

    assert_true(
        errors == 0,
        String(
            "num of errors must be 0, but got ",
            errors,
            " for dimensions M=",
            m,
            ", N=",
            n,
            ", K=",
            k,
        ),
    )

    golden_ptr.free()


def test_matmul(m: Int, n: Int, k: Int) raises:
    var c_ptr = alloc[Scalar[c_type]](m * n, alignment=alignment)
    var a_ptr = alloc[Scalar[a_type]](m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type]](k * n, alignment=alignment)

    var c = TileTensor(c_ptr, row_major(Coord(m, n)))
    var a = TileTensor(a_ptr, row_major(Coord(m, k)))
    var b = TileTensor(b_ptr, row_major(Coord(k, n)))

    test_matmul(c, a, b, m, n, k)

    c_ptr.free()
    b_ptr.free()
    a_ptr.free()


def test_matmul() raises:
    test_matmul(256, 1024, 4096)
    test_matmul(4, 5, 6)
    test_matmul(15, 16, 17)
    test_matmul(24, 32, 64)
    test_matmul(61, 73, 79)
    test_matmul(123, 456, 321)
    test_matmul(256, 256, 256)
    test_matmul(2, 65, 1200)


def bmm_naive(
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b: TileTensor,
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert c.flat_rank >= 3
    comptime assert a.flat_rank >= 3
    comptime assert b.flat_rank >= 3
    for batch in range(batches):
        for i in range(m):
            for p in range(k):
                for j in range(n):
                    var a_val = a.load(Coord(batch, i, p)).cast[c.dtype]()[0]
                    var b_val = b.load(Coord(batch, p, j)).cast[c.dtype]()[0]
                    var cur = c.load(Coord(batch, i, j))[0]
                    c.store(Coord(batch, i, j), cur + a_val * b_val)


def test_batched_matmul(
    c: TileTensor[mut=True, ...],
    a: TileTensor[mut=True, ...],
    b: TileTensor[mut=True, ...],
    batches: Int,
    m: Int,
    n: Int,
    k: Int,
) raises:
    comptime assert c.flat_rank >= 3
    comptime assert a.flat_rank >= 3
    comptime assert b.flat_rank >= 3
    var golden_ptr = alloc[Scalar[c.dtype]](
        batches * m * n, alignment=alignment
    )
    var golden = TileTensor(golden_ptr, row_major(Coord(batches, m, n)))

    for batch in range(batches):
        for i in range(m):
            for j in range(k):
                a.store(
                    Coord(batch, i, j),
                    Scalar[a.dtype](i + j) * Scalar[a.dtype](0.001),
                )

    for batch in range(batches):
        for i in range(k):
            for j in range(n):
                b.store(
                    Coord(batch, i, j),
                    Scalar[b.dtype](i + k) * Scalar[b.dtype](0.001),
                )

    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                c.store(Coord(batch, i, j), Scalar[c.dtype](0))
                golden.store(Coord(batch, i, j), Scalar[golden.dtype](0))

    var c_shape = Index(batches, m, n)
    apple_batched_matmul(c, a, b, c_shape)
    bmm_naive(golden, a, b, batches, m, n, k)

    var errors: Int = 0
    for batch in range(batches):
        for i in range(m):
            for j in range(n):
                if c.load(Coord(batch, i, j)) != golden.load(
                    Coord(batch, i, j)
                ):
                    if errors < 10:
                        print(
                            c.load(Coord(batch, i, j))
                            - golden.load(Coord(batch, i, j)),
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
    var c_ptr = alloc[Scalar[c_type]](batch * m * n, alignment=alignment)
    var a_ptr = alloc[Scalar[a_type]](batch * m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type]](batch * k * n, alignment=alignment)

    var c = TileTensor(c_ptr, row_major(Coord(batch, m, n)))
    var a = TileTensor(a_ptr, row_major(Coord(batch, m, k)))
    var b = TileTensor(b_ptr, row_major(Coord(batch, k, n)))

    test_batched_matmul(c, a, b, batch, m, n, k)

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
    test_matmul()
    test_batched_matmul()
