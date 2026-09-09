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

from std.math import align_up
from std.sys import align_of

from layout import Idx, TileTensor
from layout.tile_layout import row_major
from linalg.matmul.cpu.default import Inner_matmul_default
from linalg.matmul.cpu.i8mm import Inner_matmul_i8mm
from linalg.matmul.cpu.neon import Inner_matmul_neon
from linalg.matmul.cpu.vnni import Inner_matmul_vnni
from linalg.utils import (
    GemmShape,
    InnerKernelID,
    KernelConfig,
    get_kernel_config,
    get_matmul_arch_factor,
    select_inner_kernel,
    use_i8mm_fn,
    use_vnni_fn,
)
from std.testing import assert_equal

from std.utils import IndexList
from std.utils.index import Index

comptime M: Int = 64
comptime N: Int = 64
comptime K: Int = 256


def _matmul_inner_loop[
    kernel_rows: Int,
    kernel_cols: Int,
    simd_size: Int,
    saturated_vnni: Bool,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b_packed: TileTensor,
    global_offset: GemmShape,
    global_bound: GemmShape,
    tile_n_k: IndexList[2],
    skip_boundary_check: Bool,
):
    comptime kernel_id = select_inner_kernel[a.dtype, b_packed.dtype, c.dtype]()

    comptime if kernel_id == InnerKernelID.DEFAULT:
        Inner_matmul_default().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.VNNI:
        Inner_matmul_vnni[saturated_vnni]().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.NEON:
        Inner_matmul_neon().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    elif kernel_id == InnerKernelID.I8MM:
        Inner_matmul_i8mm().__inner_matmul__[
            kernel_rows, kernel_cols, simd_size
        ](
            c,
            a,
            b_packed,
            global_offset,
            global_bound,
            tile_n_k,
            skip_boundary_check,
        )
    else:
        comptime assert False, "no _run_inner_loop implementation"


def matmul_inner_loop[
    config: KernelConfig,
](
    c: TileTensor[mut=True, ...],
    a: TileTensor,
    b_packed: TileTensor,
    m: Int,
    n: Int,
    k: Int,
):
    _matmul_inner_loop[
        config.kernel_rows,
        config.kernel_cols,
        config.simd_size,
        False,  # saturated_vnni
    ](
        c,
        a,
        b_packed,
        # Below are configurations for outer loops, just
        #  use the trivial numbers for now.
        GemmShape(0, 0, 0),  # Tile offset.
        GemmShape(m, n, k),  # Global tile dimension.
        Index(n, k),  # Local tile dimension.
        True,  # skip_boundary_check
    )


def test_micro_kernel[
    a_type: DType, b_type: DType, c_type: DType, saturated_vnni: Bool = False
](m: Int, n: Int, k: Int) raises:
    print("== test_micro_kernel")

    comptime config = get_kernel_config[a_type, b_type, c_type]()
    comptime use_vnni = use_vnni_fn[a_type, b_type, c_type]()
    comptime use_i8mm = use_i8mm_fn[a_type, b_type, c_type]()
    comptime factor = get_matmul_arch_factor[use_vnni, use_i8mm]()
    var np = align_up(n, config.kernel_cols)
    var kh = align_up(k, factor)

    comptime alignment = align_of[SIMD[c_type, config.simd_size]]()

    var a_ptr = alloc[Scalar[a_type],](m * k, alignment=alignment)
    var b_packed_ptr = alloc[Scalar[b_type]](
        (np // config.kernel_cols)
        * (kh // factor)
        * (factor * config.kernel_cols),
        alignment=alignment,
    )
    var c_ptr = alloc[Scalar[c_type],](m * n, alignment=alignment)
    var a = TileTensor(a_ptr, row_major(m, k))
    var b_packed = TileTensor(
        b_packed_ptr,
        row_major(
            np // config.kernel_cols,
            kh // factor,
            factor * config.kernel_cols,
        ),
    )
    var c = TileTensor(c_ptr, row_major(m, n))

    _ = a.fill(1)
    _ = b_packed.fill(1)
    _ = c.fill(0)

    matmul_inner_loop[config](c, a, b_packed, m, n, k)

    assert_equal(Int(c_ptr[0]), k)
    a_ptr.free()
    b_packed_ptr.free()
    c_ptr.free()


@export
def kernel_export_dynamic(m: Int, n: Int, k: Int) abi("Mojo") raises:
    test_micro_kernel[.float32, DType.float32, DType.float32](m, n, k)


def main() raises:
    test_micro_kernel[.float32, DType.float32, DType.float32](M, N, K)
    test_micro_kernel[.uint8, DType.int8, DType.int32](M, N, K)
    test_micro_kernel[
        DType.uint8, DType.int8, DType.int32, saturated_vnni=True
    ](M, N, K)

    test_micro_kernel[.bfloat16, DType.bfloat16, DType.bfloat16](M, N, K)
    test_micro_kernel[.bfloat16, DType.bfloat16, DType.float32](M, N, K)

    # Test int8 x int8 -> int8 to ensure it doesn't dispatch to i8mm (which
    # requires 32-bit output). Use smaller k to fit in int8 range.
    test_micro_kernel[.int8, DType.int8, DType.int8](M, N, 100)
