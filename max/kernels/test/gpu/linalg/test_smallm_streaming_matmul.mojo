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
"""Correctness test for the MI355X small-M streaming matmul pair.

Runs ``smallm_preshuffle_b`` on a random row-major weight, feeds the shuffled
buffer to ``smallm_streaming_matmul``, and checks against vendor BLAS on the
ORIGINAL weight — so a bug in either the permutation or the fragment reader
fails the pair. Sweeps M across every row-tile instantiation, including
non-multiples of 16 (row clamping) and the register-resident-A band.

  mojo -D N=512 -D K=2048 test_smallm_streaming_matmul.mojo
"""

from std.math import isnan
from std.sys import get_defined_int

from layout import TileTensor, row_major
from max.gpu.host import DeviceContext
import linalg.matmul.vendor.blas as vendor_blas
from std.random import random_float64, seed
from linalg.matmul.gpu.amd.smallm_streaming_matmul import (
    smallm_preshuffle_b,
    smallm_streaming_matmul,
)

comptime TEST_N = get_defined_int["N", 512]()
comptime TEST_K = get_defined_int["K", 2048]()


def test_smallm_streaming[N: Int, K: Int](m: Int, ctx: DeviceContext) raises:
    comptime dtype = DType.bfloat16
    comptime max_m = 128

    seed(20260814 + m)
    var device_a = ctx.enqueue_create_buffer[dtype](max_m * K)
    var device_b = ctx.enqueue_create_buffer[dtype](N * K)
    var device_b_shuf = ctx.enqueue_create_buffer[dtype](N * K)
    var device_c = ctx.enqueue_create_buffer[.float32](max_m * N)
    var device_scratch = ctx.enqueue_create_buffer[dtype](max_m * K)
    var device_c_ref = ctx.enqueue_create_buffer[.float32](max_m * N)

    with device_a.map_to_host() as host_a, device_b.map_to_host() as host_b:
        for i in range(max_m * K):
            host_a[i] = random_float64(-0.5, 0.5).cast[dtype]()
        for i in range(N * K):
            host_b[i] = random_float64(-0.5, 0.5).cast[dtype]()

    var a_tt = TileTensor(device_a, row_major[max_m, K]())
    var b_tt = TileTensor(device_b, row_major[N, K]())
    var c_tt = TileTensor(device_c, row_major[max_m, N]())
    var c_ref_tt = TileTensor(device_c_ref, row_major[max_m, N]())

    ctx.enqueue_memset(device_c_ref, 0)
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    # The production entry points take graph-marshaled AnyOrigin tensors;
    # rebind the test's concretely-originated buffers to match.
    var b_shuf_dst = MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(device_b_shuf.unsafe_ptr())
    )
    var b_src = ImmPointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=Int(device_b.unsafe_ptr())
    )
    smallm_preshuffle_b[dtype, k_static=K](b_shuf_dst, b_src, N, ctx)

    var a_ptr = ImmPointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=Int(device_a.unsafe_ptr())
    )
    comptime a_layout = row_major[max_m, K]()
    var a_any = TileTensor[dtype, type_of(a_layout), ImmutAnyOrigin](
        a_ptr, a_layout
    )
    var c_ptr = MutPointer[Float32, MutAnyOrigin](
        unsafe_from_address=Int(device_c.unsafe_ptr())
    )
    comptime c_layout = row_major[max_m, N]()
    var c_any = TileTensor[.float32, type_of(c_layout), MutAnyOrigin](
        c_ptr, c_layout
    )
    var b_shuf_src = ImmPointer[Scalar[dtype], ImmutAnyOrigin](
        unsafe_from_address=Int(device_b_shuf.unsafe_ptr())
    )
    var scratch_ptr = MutPointer[Scalar[dtype], MutAnyOrigin](
        unsafe_from_address=Int(device_scratch.unsafe_ptr())
    )

    ctx.enqueue_memset(device_c, 0)
    # A small grid cap forces the grid-strided column loop to iterate even at
    # the test's small N (production caps at 512 blocks).
    smallm_streaming_matmul[k_static=K, max_grid_blocks=8](
        c_any,
        a_any,
        b_shuf_src,
        scratch_ptr,
        m,
        N,
        ctx,
    )

    with device_c.map_to_host() as host_c, device_c_ref.map_to_host() as host_c_ref:
        # Tolerance formula from `test_4wave_matmul.mojo` (pytorch-like).
        comptime rel_tol = Float32(1.6e-2)
        comptime abs_tol = Float32(1e-5)
        var errors = 0
        var printed = 0
        # Only the first `m` rows are defined output; vendor computed all
        # max_m rows, the kernel only wrote m of them.
        for i in range(m * N):
            var actual = host_c[i]
            var expected = host_c_ref[i]
            var diff = abs(actual - expected)
            var threshold = abs_tol + rel_tol * abs(expected)
            # NaN fails every ordered comparison, so it would otherwise
            # slip past the threshold check and print PASS.
            if diff > threshold or isnan(actual):
                if printed < 10:
                    var row, col = divmod(i, N)
                    print(
                        "Mismatch at (",
                        row,
                        ",",
                        col,
                        "): actual=",
                        actual,
                        " expected=",
                        expected,
                        " diff=",
                        diff,
                    )
                    printed += 1
                errors += 1
        if errors != 0:
            raise Error(
                t"smallm_streaming m={m}: {errors} mismatches of {m * N}"
            )
    print(t"PASS m={m} (N={N}, K={K})")


def main() raises:
    with DeviceContext() as ctx:
        # Every row-tile instantiation plus clamp edges: 1..16 -> 1 tile
        # (register-A at this K), 17..32 -> 2, 33..64 -> 4, 65..128 -> 8.
        for m in [1, 2, 5, 16, 17, 32, 33, 64, 65, 128]:
            test_smallm_streaming[TEST_N, TEST_K](m, ctx)
        # An odd column-tile count exercises the col_tiles tail block, whose
        # reads clamp to the last real tile.
        test_smallm_streaming[TEST_N - 16, TEST_K](32, ctx)
