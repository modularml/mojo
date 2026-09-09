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
"""Empirically validate that block-tile SRD OOB reads in ping-pong-style
`TileLoaderLDS` callers do NOT contaminate matmul output on non-aligned
shapes — both M and N axes.

`TileLoaderLDS` (`structured_kernels/amd_tile_io.mojo`) builds its
`AMDBufferResource` from a `.tile[BM, K]()` / `.tile[BN, K]()` sub-tile
of the relevant operand, sizing `num_records` from the block base. On
non-aligned `M` (rows) or `Cout` (cols), the SRD allows MUBUF reads
past the actual allocation — formally UB by the HIP memory model, and
a possible source of "garbage" returned in OOB lanes.

This test asks the empirical question per axis: does that garbage
actually contaminate the in-bounds region of the matmul output?

Methodology (one shared backing buffer per axis):

  - **N axis** (`Cout_aligned=512`, `Cout_unaligned=500`):
    Single B buffer of `Cout_aligned * K`. Run `structured_4wave_matmul`
    twice with two `TileTensor` views — `[M, Cout_aligned]` and
    `[M, Cout_unaligned]`. Same A. Compare cols 0..Cout_unaligned. If
    OOB cols 500..511 corrupt the in-bounds cols (last N-block), the
    outputs differ.

  - **M axis** (`M_aligned=2048`, `M_unaligned=1900`):
    Single A buffer of `M_aligned * K`. Run twice with `[M_aligned, K]`
    and `[M_unaligned, K]` views. Same B. Compare rows 0..M_unaligned.
    If OOB rows 1900..2047 corrupt the in-bounds rows (last M-block),
    the outputs differ.

The test reuses the SAME backing buffer per axis so the OOB region is
deterministic (whatever lives at the aligned-but-not-semantic bytes)
rather than depending on allocator residue. That's the "worst-case
non-zero garbage" scenario.

Empirically (gfx950 / MI355X, FP8): per-(m, n) MMA independence + the
output store's `if m < M:` / `if n < N:` guard masks the SRD OOB reads
on both axes. UB is formal only, not a correctness regression.
"""

from max.gpu.host import DeviceContext
from std.random import rand

from layout import TileTensor, row_major

from linalg.matmul.gpu.amd.amd_4wave_matmul import structured_4wave_matmul


def test_nonaligned_n(ctx: DeviceContext) raises:
    """N-axis SRD-OOB non-contamination check."""

    comptime a_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16

    comptime M = 4096  # aligned to BM=128
    comptime K = 2048  # multiple of 2*BK=256
    comptime Cout_aligned = 512  # backing buffer width
    comptime Cout_unaligned = 500  # semantic width for the "unaligned" run

    var a_dev = ctx.enqueue_create_buffer[a_type](M * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](Cout_aligned * K)
    var c_aligned_dev = ctx.enqueue_create_buffer[c_type](M * Cout_aligned)
    var c_unaligned_dev = ctx.enqueue_create_buffer[c_type](M * Cout_unaligned)

    var a_host = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](Cout_aligned * K)
    rand(a_host.unsafe_ptr(), M * K)
    rand(b_host.unsafe_ptr(), Cout_aligned * K)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime a_layout = row_major[M, K]()
    comptime b_aligned_layout = row_major[Cout_aligned, K]()
    comptime b_unaligned_layout = row_major[Cout_unaligned, K]()
    comptime c_aligned_layout = row_major[M, Cout_aligned]()
    comptime c_unaligned_layout = row_major[M, Cout_unaligned]()

    var a_t = TileTensor(a_dev, a_layout)

    # --- Run 1: aligned (Cout=512) ---
    var b_aligned = TileTensor(b_dev, b_aligned_layout)
    var c_aligned = TileTensor(c_aligned_dev, c_aligned_layout)
    structured_4wave_matmul(a_t, b_aligned, c_aligned, ctx)

    # --- Run 2: unaligned (Cout=500), same B buffer (cols 500..511
    #     contain the same bytes as run 1 — deterministic "garbage") ---
    var b_unaligned = TileTensor(b_dev, b_unaligned_layout)
    var c_unaligned = TileTensor(c_unaligned_dev, c_unaligned_layout)
    structured_4wave_matmul(a_t, b_unaligned, c_unaligned, ctx)

    var c_aligned_host = ctx.enqueue_create_host_buffer[c_type](
        M * Cout_aligned
    )
    var c_unaligned_host = ctx.enqueue_create_host_buffer[c_type](
        M * Cout_unaligned
    )
    ctx.enqueue_copy(c_aligned_host, c_aligned_dev)
    ctx.enqueue_copy(c_unaligned_host, c_unaligned_dev)
    ctx.synchronize()

    # Split into "interior" cols (untouched by the unaligned last block)
    # and "last-block-tail" cols (where OOB reads would contaminate IF
    # cross-column leakage existed).
    comptime LAST_BLOCK_START = 384  # 3 * 128 = first col of the 4th N-block
    var interior_mismatches = 0
    var tail_mismatches = 0
    for m in range(M):
        for n in range(Cout_unaligned):
            var aligned_v = c_aligned_host[m * Cout_aligned + n]
            var unaligned_v = c_unaligned_host[m * Cout_unaligned + n]
            if aligned_v != unaligned_v:
                if n < LAST_BLOCK_START:
                    interior_mismatches += 1
                else:
                    tail_mismatches += 1

    print(
        "[N] Shape: M=",
        M,
        " K=",
        K,
        " Cout_aligned=",
        Cout_aligned,
        " Cout_unaligned=",
        Cout_unaligned,
    )
    print(
        "  Interior cols (0..",
        LAST_BLOCK_START,
        "): mismatches=",
        interior_mismatches,
    )
    print(
        "  Last-block-tail cols (",
        LAST_BLOCK_START,
        "..",
        Cout_unaligned,
        "): mismatches=",
        tail_mismatches,
    )
    if interior_mismatches == 0 and tail_mismatches == 0:
        print(
            "  RESULT [N]: outputs are byte-identical in the in-bounds region."
        )
    else:
        raise Error("non-aligned-N matmul output contamination detected")


def test_nonaligned_m(ctx: DeviceContext) raises:
    """M-axis SRD-OOB non-contamination check."""

    comptime a_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16

    # M_aligned is a multiple of BM=128. M_unaligned is NOT aligned.
    # Both runs use M > 512 so the launcher picks BM=128.
    comptime M_aligned = 2048
    # Last block: rows 1792..1919 (only 1792..1899 valid).
    comptime M_unaligned = 1900
    comptime K = 2048
    comptime N = 512  # aligned to BN=128

    var a_dev = ctx.enqueue_create_buffer[a_type](M_aligned * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var c_aligned_dev = ctx.enqueue_create_buffer[c_type](M_aligned * N)
    var c_unaligned_dev = ctx.enqueue_create_buffer[c_type](M_unaligned * N)

    var a_host = ctx.enqueue_create_host_buffer[a_type](M_aligned * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](N * K)
    rand(a_host.unsafe_ptr(), M_aligned * K)
    rand(b_host.unsafe_ptr(), N * K)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)

    comptime a_aligned_layout = row_major[M_aligned, K]()
    comptime a_unaligned_layout = row_major[M_unaligned, K]()
    comptime b_layout = row_major[N, K]()
    comptime c_aligned_layout = row_major[M_aligned, N]()
    comptime c_unaligned_layout = row_major[M_unaligned, N]()

    var b_t = TileTensor(b_dev, b_layout)

    # --- Aligned run (M=2048) ---
    var a_aligned = TileTensor(a_dev, a_aligned_layout)
    var c_aligned = TileTensor(c_aligned_dev, c_aligned_layout)
    structured_4wave_matmul(a_aligned, b_t, c_aligned, ctx)

    # --- Unaligned run (M=1900) — same A backing buffer, so rows
    #     1900..2047 contain the same bytes as the aligned run ---
    var a_unaligned = TileTensor(a_dev, a_unaligned_layout)
    var c_unaligned = TileTensor(c_unaligned_dev, c_unaligned_layout)
    structured_4wave_matmul(a_unaligned, b_t, c_unaligned, ctx)

    var c_aligned_host = ctx.enqueue_create_host_buffer[c_type](M_aligned * N)
    var c_unaligned_host = ctx.enqueue_create_host_buffer[c_type](
        M_unaligned * N
    )
    ctx.enqueue_copy(c_aligned_host, c_aligned_dev)
    ctx.enqueue_copy(c_unaligned_host, c_unaligned_dev)
    ctx.synchronize()

    comptime LAST_BLOCK_START = 1792  # 14*128 = first row of the 15th M-block
    var interior_mismatches = 0
    var tail_mismatches = 0
    for m in range(M_unaligned):
        for n in range(N):
            if c_aligned_host[m * N + n] != c_unaligned_host[m * N + n]:
                if m < LAST_BLOCK_START:
                    interior_mismatches += 1
                else:
                    tail_mismatches += 1

    print(
        "[M] Shape: M_aligned=",
        M_aligned,
        " M_unaligned=",
        M_unaligned,
        " K=",
        K,
        " N=",
        N,
    )
    print(
        "  Interior rows (0..",
        LAST_BLOCK_START,
        "): mismatches=",
        interior_mismatches,
    )
    print(
        "  Last-block-tail rows (",
        LAST_BLOCK_START,
        "..",
        M_unaligned,
        "): mismatches=",
        tail_mismatches,
    )
    if interior_mismatches == 0 and tail_mismatches == 0:
        print(
            "  RESULT [M]: outputs are byte-identical in the in-bounds region."
        )
    else:
        raise Error("non-aligned-M matmul output contamination detected")


def main() raises:
    with DeviceContext() as ctx:
        test_nonaligned_n(ctx)
        test_nonaligned_m(ctx)
    print("ALL non-aligned axis tests PASSED")
