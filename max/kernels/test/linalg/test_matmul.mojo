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
# Checks x86 int8 matmul C = A*B with prepacked B
#
# ===----------------------------------------------------------------------=== #

from std.sys.info import CompilationTarget

from layout import Coord, Idx, TileTensor
from layout.tile_layout import row_major
from linalg.matmul import matmul
from linalg.matmul.cpu import matmul as _matmul_cpu
from linalg.packing import (
    pack_b_ndbuffer,
    pack_matmul_b_shape_func,
)
from std.testing import assert_almost_equal, assert_equal


comptime alignment = 64


def gemm_naive[
    a_type: DType, b_type: DType, c_type: DType
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert a.rank == 2 and a.flat_rank == 2
    comptime assert b.rank == 2 and b.flat_rank == 2
    comptime assert c.rank == 2 and c.flat_rank == 2
    for i in range(m):
        for p in range(k):
            for j in range(n):
                var cur = rebind[Scalar[c_type]](c[i, j])
                var av = rebind[Scalar[c_type]](a[i, p].cast[c_type]())
                var bv = rebind[Scalar[c_type]](b[p, j].cast[c_type]())
                c[i, j] = cur + av * bv


def test_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    b_packed: Bool,
    saturated: Bool,
](m: Int, n: Int, k: Int, kernel_type_m: Int) raises:
    var a_ptr = alloc[Scalar[a_type],](m * k, alignment=alignment)
    var b_ptr = alloc[Scalar[b_type],](k * n, alignment=alignment)
    var b = TileTensor(b_ptr.as_unsafe_any_origin(), row_major(Coord(k, n)))

    var padded_n_k = pack_matmul_b_shape_func[
        a_type,
        c_type,
        transpose_b,
    ](b, kernel_type_m)

    var padded_n = padded_n_k[1] if b_packed else n
    var padded_k = padded_n_k[0] if b_packed else k

    var bp_ptr = alloc[Scalar[b_type],](
        padded_k * padded_n, alignment=alignment
    )
    var c0_ptr = alloc[Scalar[c_type],](m * n, alignment=alignment)
    var c1_ptr = alloc[Scalar[c_type],](m * n, alignment=alignment)

    var a = TileTensor(a_ptr.as_unsafe_any_origin(), row_major(Coord(m, k)))
    var bp = TileTensor(
        bp_ptr.as_unsafe_any_origin(), row_major(Coord(padded_k, padded_n))
    )
    var c = TileTensor(c0_ptr.as_unsafe_any_origin(), row_major(Coord(m, n)))
    var golden = TileTensor(
        c1_ptr.as_unsafe_any_origin(), row_major(Coord(m, n))
    )

    # saturated VNNI only has a range [0,127] for the input a
    var vnni_range: Int = 128 if saturated else 256
    var cnt: Int = 0
    for i in range(m):
        for p in range(k):
            # uint8 but limited to [0,127]
            a[i, p] = Scalar[a_type](cnt % vnni_range)
            cnt += 1

    cnt = 0
    for p in range(k):
        for j in range(n):
            # int8 [-128, 127]
            b[p, j] = Scalar[b_type](cnt % 256 - 128)
            bp[p, j] = b[p, j]
            cnt += 1

    for i in range(m):
        for j in range(n):
            c[i, j] = 0
            golden[i, j] = c[i, j]

    comptime if b_packed:
        pack_b_ndbuffer[a_type, c_type](b, bp, kernel_type_m)

    if kernel_type_m != 0:
        _matmul_cpu[
            transpose_b=transpose_b,
            b_packed=b_packed,
            saturated_vnni=saturated,
        ](c, a, bp, kernel_type_m)
    else:
        matmul[
            transpose_b=transpose_b,
            b_packed=b_packed,
            saturated_vnni=saturated,
        ](c, a, bp)

    gemm_naive(a, b, golden, m, n, k)

    for i in range(m):
        for j in range(n):
            var msg = String(
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
            )

            comptime if c_type.is_floating_point():
                assert_almost_equal(c[i, j], golden[i, j], msg)
            else:
                assert_equal(c[i, j], golden[i, j], msg)

    a_ptr.free()
    b_ptr.free()
    bp_ptr.free()
    c0_ptr.free()
    c1_ptr.free()


def test_matmul[
    *,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    saturated: Bool,
    mixed_kernels: Bool,
](m: Int, n: Int, k: Int) raises:
    test_matmul[
        a_type,
        b_type,
        c_type,
        False,  # transpose_b
        b_packed,
        saturated=saturated,
    ](m, n, k, m if mixed_kernels else 0)


def test_shapes[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    b_packed: Bool,
    saturated: Bool,
    mixed_kernels: Bool,
]() raises:
    @__parameter
    def test_shapes_helper(m: Int, n: Int, k: Int) raises:
        test_matmul[
            a_type=a_type,
            b_type=b_type,
            c_type=c_type,
            b_packed=b_packed,
            saturated=saturated,
            mixed_kernels=mixed_kernels,
        ](m, n, k)

    test_shapes_helper(256, 1024, 4096)
    test_shapes_helper(4, 5, 6)
    test_shapes_helper(15, 16, 17)
    test_shapes_helper(24, 32, 64)
    test_shapes_helper(61, 73, 79)
    test_shapes_helper(123, 456, 321)
    test_shapes_helper(256, 256, 256)
    test_shapes_helper(2, 65, 1200)
    test_shapes_helper(1, 5, 10)
    test_shapes_helper(1, 768, 384)
    test_shapes_helper(1, 1, 1)
    test_shapes_helper(768, 1, 384)
    test_shapes_helper(1, 384, 1)
    test_shapes_helper(384, 768, 1)


def test_types[b_packed: Bool, saturated: Bool, mixed_kernels: Bool]() raises:
    comptime if b_packed and CompilationTarget.is_macos():
        return

    test_shapes[
        DType.uint8,
        DType.uint8,
        DType.int32,
        b_packed,
        saturated,
        mixed_kernels,
    ]()
    test_shapes[
        DType.uint8, DType.int8, DType.int32, b_packed, saturated, mixed_kernels
    ]()
    test_shapes[
        DType.int8, DType.int8, DType.int32, b_packed, saturated, mixed_kernels
    ]()
    if not saturated:
        test_shapes[
            DType.float32,
            DType.float32,
            DType.float32,
            b_packed,
            saturated,
            mixed_kernels,
        ]()


def main() raises:
    test_types[b_packed=False, saturated=False, mixed_kernels=False]()
    test_types[b_packed=True, saturated=False, mixed_kernels=False]()
    test_types[b_packed=False, saturated=True, mixed_kernels=False]()
    test_types[b_packed=True, saturated=True, mixed_kernels=False]()
    test_types[b_packed=False, saturated=False, mixed_kernels=True]()
    test_types[b_packed=True, saturated=False, mixed_kernels=True]()
    test_types[b_packed=False, saturated=True, mixed_kernels=True]()
    test_types[b_packed=True, saturated=True, mixed_kernels=True]()
