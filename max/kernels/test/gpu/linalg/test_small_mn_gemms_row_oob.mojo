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
# SM100 (B200) regression test: row out-of-bounds in small-MN matmul configs.
#
# `small_MN_gemms` launches with a grid covering ceildiv(m, tile_m) * tile_m
# rows, so tile_m > 1 configs overshoot the [M, N] output whenever the
# runtime m % tile_m != 0; the tail rows must be neither read nor written.
#
# Cases are derived at compile time from the small-MN tuning table (every
# tile_m > 1 entry, first non-divisible M in its range) and driven through
# the public `matmul_dispatch_sm100` entry. The C buffer carries sentinel
# guard rows past row M: the valid region must match a cuBLAS reference and
# the sentinels must survive. Only OOB *writes* are detectable here; A's
# zero-filled guard rows keep OOB reads from faulting.

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from std.memory import alloc
from internal_utils import assert_almost_equal, assert_equal
from std.random import rand
from layout import Coord, TileTensor, row_major, Idx
from linalg.gemv import GEMVAlgorithm
from linalg.matmul.gpu.sm100_structured.default.dispatch import (
    matmul_dispatch_sm100,
)
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    _get_tuning_list_small_MN_gemms_bf16,
)


def _first_non_divisible_m(m_start: Int, m_end: Int, tile_m: Int) -> Int:
    """First M in [m_start, m_end) with M % tile_m != 0, or -1 if none.

    Only such M values make the grid overshoot the final rows; for divisible
    M the grid covers the output exactly and there is nothing to detect.
    """
    for m in range(m_start, m_end):
        if m % tile_m != 0:
            return m
    return -1


def _num_row_oob_cases() -> Int:
    """Number of tuning entries whose grid can overshoot the final rows."""
    var configs = _get_tuning_list_small_MN_gemms_bf16()
    var count = 0
    for config in configs:
        if (
            config.tile_m > 1
            and _first_non_divisible_m(config.M, config.M_end, config.tile_m)
            > 0
        ):
            count += 1
    return count


def run_case[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    N: Int,
    K: Int,
    tile_m: Int,
](ctx: DeviceContext, M: Int) raises:
    """Run one (N, K, M) case through `matmul_dispatch_sm100` and check OOB.

    `tile_m` is the tile_m of the tuning config that runtime `M` selects; it
    sizes the guard region (the OOB block touches at most tile_m - 1 rows
    past M).
    """
    print(
        "[case] MNK=(",
        M,
        N,
        K,
        ") tile_m=",
        tile_m,
        " M % tile_m=",
        M % tile_m,
        " guard_rows=",
        tile_m,
    )

    # The matmul sees an [M, N] output, but the C/A buffers carry `tile_m`
    # extra guard rows so a worst-case OOB block lands in allocated sentinel
    # memory rather than off the end of the allocation (which would be a hard
    # fault instead of a detectable corruption).
    var guard_rows = tile_m
    var m_alloc = M + guard_rows

    var a_size = m_alloc * K
    var b_size = N * K  # transpose_b: B is [N, K]
    var c_size = m_alloc * N
    var c_logical = M * N

    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_ref_host_ptr = alloc[Scalar[c_type]](c_logical)
    # Sentinel reference for the guard region: what guard rows must still be.
    var sentinel_ref_ptr = alloc[Scalar[c_type]](guard_rows * N)

    # Random activations/weights for the logical [M, K] x [N, K] problem.
    rand(a_host_ptr, M * K, min=-1.0, max=1.0)
    rand(b_host_ptr, b_size, min=-1.0, max=1.0)
    # Zero the A guard rows: an OOB activation read from the last block lands
    # here (in-bounds of the allocation) instead of faulting.
    for i in range(M * K, a_size):
        a_host_ptr[i] = Scalar[a_type](0)

    # Sentinel fill the entire C buffer (incl. guard rows). A correct kernel
    # overwrites exactly the first M rows; guard rows must keep the sentinel.
    var sentinel = Scalar[c_type](-987654.0)
    for i in range(c_size):
        c_host_ptr[i] = sentinel
    for i in range(guard_rows * N):
        sentinel_ref_ptr[i] = sentinel

    var a_dev = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[c_type](c_size)
    # Separate logical-[M, N] device buffer for the cuBLAS reference (no guard
    # rows): the reference must NOT touch the kernel's `c_dev`, so the sentinel
    # guard check below sees only what the kernel-under-test wrote.
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](c_logical)

    ctx.enqueue_copy(a_dev, a_host_ptr)
    ctx.enqueue_copy(b_dev, b_host_ptr)
    ctx.enqueue_copy(c_dev, c_host_ptr)

    # Static N and K (matched at comptime by small_MN_gemms_rule), dynamic M
    # (so the dispatcher uses runtime `m` -> `ceildiv(m, tile_m)` grid). The A
    # and C tensors report M rows even though their buffers hold M + guard_rows.
    var a_tt = TileTensor(a_dev, row_major(Coord(Int64(M), Idx[K])))
    var b_tt = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c_tt = TileTensor(c_dev, row_major(Coord(Int64(M), Idx[N])))

    matmul_dispatch_sm100[transpose_b=True](
        c_tt, a_tt.as_immut(), b_tt.as_immut(), ctx
    )

    # (a) reference: cuBLAS on-device for the logical [M, N] region. cuBLAS is
    # told the correct M rows, so it never writes past row M and runs in
    # microseconds (a host-side naive loop over M*N*K MACs in interpreted
    # Mojo timed out the test). Note this verifies whatever path the
    # dispatcher selected: routing to small_MN_gemms is assumed (the tuning
    # table is consulted first under default build flags), not observed
    # directly.
    var c_ref_tt = TileTensor(c_ref_dev, row_major(Coord(Int64(M), Idx[N])))
    vendor_blas.matmul(
        ctx,
        c_ref_tt.to_layout_tensor(),
        a_tt.to_layout_tensor(),
        b_tt.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_host_ptr, c_dev)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev)
    ctx.synchronize()

    # Long-K bf16 reduction -> loose-ish tol (cuBLAS vs Mojo kernel drift a
    # few bf16 ulps).
    assert_almost_equal(
        c_host_ptr,
        c_ref_host_ptr,
        c_logical,
        "valid [M, N] region must match cuBLAS reference",
        atol=1e-1,
        rtol=1e-2,
    )
    print("    valid region matches reference")

    # (b) THE OOB GUARD: the `tile_m` guard rows past row M must be untouched.
    assert_equal(
        c_host_ptr + c_logical,
        sentinel_ref_ptr,
        guard_rows * N,
        (
            "OOB WRITE: guard rows past M were clobbered -- the kernel wrote"
            " output rows without a `row < m` bound"
        ),
    )
    print("    guard rows intact (no OOB row write)")
    print("    PASS")

    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_ref_host_ptr.free()
    sentinel_ref_ptr.free()
    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = c_ref_dev^


def main() raises:
    # If retuning ever removes all tile_m > 1 entries (or leaves only ranges
    # with no non-divisible M), this test no longer covers anything; fail the
    # build loudly so it gets updated or removed rather than passing vacuously.
    comptime assert _num_row_oob_cases() > 0, (
        "no tile_m > 1 entry with a non-divisible M in"
        " _get_tuning_list_small_MN_gemms_bf16(); update or remove this test"
    )

    with DeviceContext() as ctx:
        comptime bf16 = DType.bfloat16

        print("=" * 64)
        print("small-MN matmul row-OOB regression (SM100 / B200)")
        print("=" * 64)

        comptime configs = _get_tuning_list_small_MN_gemms_bf16()
        comptime for config in configs:
            # run_case sizes the guard region from the table's tile_m, but the
            # cpasync kernel's row tile is hard-wired to 16 (asserted in
            # gemm_mma_cpasync_kernel). A smaller table value would undersize
            # the guard, so a regressed kernel could fault past it instead of
            # tripping the sentinel check.
            comptime if config.kernel_kind == GEMVAlgorithm.GEMM_MMA_CPASYNC:
                comptime assert config.tile_m == 16, (
                    "GEMM_MMA_CPASYNC entry must have tile_m=16 (the kernel's"
                    " hard-wired row tile); this test sizes its guard region"
                    " from the table's tile_m"
                )
            comptime if config.tile_m > 1:
                comptime M_test = _first_non_divisible_m(
                    config.M, config.M_end, config.tile_m
                )
                comptime if M_test > 0:
                    run_case[
                        bf16,
                        bf16,
                        bf16,
                        N=config.N,
                        K=config.K,
                        tile_m=config.tile_m,
                    ](ctx, M_test)

        print("\n=== ALL CASES PASSED (no OOB row writes) ===\n")
