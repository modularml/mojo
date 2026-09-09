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
"""Apple M5 weight-only NVFP4 (W4A16) matmul: `out = activation @ weight^T`.

Apple silicon GPU (Metal 4, `compute_capability == 5`). The FP4 weight stays
packed in DRAM (`uint8 [N, K//2]` lo-nibble + `float8_e4m3fn [N, K//16]` block
scales); each `(BN, BK)` weight sub-tile is COOPERATIVELY dequantized to bf16
once per K-strip into threadgroup memory (SMEM), then the dense bf16 simdgroup
MMA reads B from SMEM. No FP4/FP8 MMA hardware is used -- the dequant target is
bf16 and the MMA is the dense bf16 path.

A-vs-B decision: in `out = x @ W^T` the FP4 weight is the B operand
(`transpose_b=True`, W is `[N, K]`). The SMEM B sub-tile is stored row-major
`(BN, BK)` -- the same lane->element order the dense `transpose_b=True` path
reads -- so feeding it to `mma()` reproduces the dense `hw_transpose_b` exactly.
The activation A is the dense bf16 operand, loaded from DRAM each K-strip.

WHY cooperative-SMEM (not inline-per-fragment dequant): the cost of the W4A16
path over the dense bf16 path is entirely the FP4 dequant (the MMA + epilogue
are identical to dense). The earlier inline kernel dequantized each B fragment
in the MMA loop; this kernel decodes the whole `(BN, BK)` sub-tile ONCE per
strip, cooperatively across the threadgroup, so one decode phase feeds `BK//16`
K-steps of MMA. At `BK=32` (one cooperative decode feeds two 16x16 K-steps) this
beats the inline kernel at every measured shape, bit-exact -- see
`benchmarks/gpu/linalg/bench_apple_fp4_smem_bk.mojo` and the kernel docstring.
The dequant phase is a barrier-fenced SCALAR-ALU compute phase that the M5
cannot overlap with the MMA via double-buffering (measured: double-buffer ==
single-buffer for FP4); the win is amortizing the un-overlapped decode over more
MMA work, NOT hiding it. The double buffer is kept because it is free here (the
next strip's decode is issued before the current strip's MMA) and matches the
dense pipelining idiom.

This kernel mirrors `AppleM5MatMul._run_gemm_body` for everything but the B side
(Morton scheduler, per-simdgroup 32x32 / 2x2 tiling, edge-tile bounds, K-strip +
tail loop, cast/epilogue store). The dense fp16/bf16/fp32 path is untouched.
"""

from std.collections import Optional
from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.memory import unsafe_stack_allocation
from std.sys import align_of
from std.utils import IndexList

from layout import TensorEngine, TileTensor, Idx
from layout.tile_layout import Layout, TensorLayout, row_major
from layout.coord import Coord

from linalg.arch.apple.mma import MmaOpApple
from linalg.fp4_utils import (
    decode_e2m1_to_bf16,
    decode_e2m1_to_f32,
    NVFP4_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.apple.fp4_gemv import enqueue_apple_fp4_gemv
from linalg.matmul.gpu.apple.matmul_8x8 import gemm_kernel_apple_8x8
from linalg.matmul.gpu.apple.matmul2d_fp4 import enqueue_matmul2d_fp4_smem
from linalg.matmul.gpu.apple.matmul_kernel import (
    AppleM5MatMul,
    enqueue_apple_matmul,
)
from linalg.utils import elementwise_epilogue_type


struct AppleM5Fp4MatMul[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    *,
    BM: Int = 128,
    BK: Int = 64,
    coalesce_scales: Bool = True,
]:
    """W4A16 GEMM: bf16 activation x packed-FP4 weight, cooperative-SMEM dequant.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue; receives
            `SIMD[c_type, width]` at absolute `(row, col)` (AMD's contract).
        BM: Threadgroup M-tile height (also the Morton grid_m divisor). The
            simdgroup tile is a square `32x32` (2x2 `MmaOpApple`), so `BM=128`
            stacks 4 M-simdgroups (8 simdgroups/threadgroup) and `BM=64` stacks
            2 (4 simdgroups). `BM=128` amortizes the cooperative decode over more
            MMA work and wins at large M; `BM=64` keeps occupancy up at small M.
            The `enqueue_*` path selects between them by M.
        BK: K-strip depth (cols of the weight decoded per cooperative phase).
            One cooperative decode feeds `BK//16` 16x16 MMA K-steps, so a deeper
            `BK` halves the barrier + decode-phase count per unit of MMA work.
            `BK=64` measured +34-45% over `BK=32` at large M on M5 Max (deeper
            amortization of the un-overlappable decode), bit-exact. The decode
            keeps each thread's run to `COLS_PER_THREAD <= 16` bf16 cols (one FP8
            scale block) -- a hard M5 constraint: the Metal backend crashes on
            SIMD arithmetic over 16-bit-element vectors wider than 16 lanes
            (>= 48 bytes), so `BK`/`BM` must satisfy `COLS_PER_THREAD <= 16`.
            `BK=64` at `BM=128` gives `COLS_PER_THREAD=16` (the sweet spot);
            `BK=128` regresses (SMEM pressure) and `BK=64` at `BM=64` would force
            `COLS_PER_THREAD=32` (the crash), so the `enqueue_*` path pairs
            `BK=64` with `BM=128` and `BK=32` with `BM=64`.
        coalesce_scales: On the interior fast path, stage the strip's FP8 block
            scales into SMEM with one coalesced cooperative load (reused by all
            threads), instead of each thread issuing a scattered 1-byte scale
            load. Measured +7-12% at large M (the scale load is uncoalesced-
            throughput-bound, the dominant *reducible* decode cost); but it adds
            a per-strip barrier that regresses small M, so the `enqueue_*` path
            enables it only for `BM=128`. No effect on the bounded edge/tail
            path (which keeps the per-thread scale load). Bit-exact either way.

    The whole threadgroup cooperatively decodes each `(BN, BK)` packed-FP4
    weight sub-tile (+ its per-16-col FP8 block scales) into a row-major
    `(BN, BK)` bf16 SMEM tile, then runs the simdgroup MMA (`BK//16` K-steps per
    call) reading B from SMEM; A loads from DRAM. Double-buffered. Deeper `BK`
    feeds more MMA K-steps per cooperative decode, halving the barrier +
    decode-phase count per unit of MMA work; `BK=64` (at `BM=128`) measured
    +34-45% over `BK=32` at large M, bit-exact, and is the production default
    for the `BM=128` path. The hard ceiling is the M5 constraint that each
    thread's decode run stay `<= 16` bf16 cols (16-bit SIMD arithmetic wider than
    16 lanes crashes the Metal backend) -- `BK=64`/`BM=128` gives exactly 16, and
    `BK=128` regresses on SMEM pressure -- so `BK=64` is the sweet spot. `BN=64`
    is the Morton grid_n divisor (shared with dense).
    """

    comptime in_type = DType.bfloat16  # activation dtype (and dequant target)
    comptime BN = 64
    comptime SG_M = 32
    comptime SG_N = 32
    comptime SF = NVFP4_SF_VECTOR_SIZE
    comptime NBLK_PER_STRIP = Self.BK // Self.SF  # FP8 scale blocks per strip
    comptime NUM_MMA_M = Self.SG_M // 16
    comptime NUM_MMA_N = Self.SG_N // 16
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE

    comptime Mma = MmaOpApple[
        DType.float32,
        Self.in_type,
        num_m_mmas=Self.NUM_MMA_M,
        num_n_mmas=Self.NUM_MMA_N,
        transpose_b=True,
    ]

    # Reuse the dense kernel's Morton scheduler (parameter-independent statics).
    comptime _Sched = AppleM5MatMul[Self.in_type]

    # Cooperative DRAM->SMEM decode work split (comptime, interior-shaped). Each
    # N-row of this strip has `BK//2` packed bytes; split them row-contiguously
    # across `THREADS_PER_ROW` threads, each owning `BYTES_PER_THREAD` contiguous
    # bytes = `COLS_PER_THREAD` bf16 cols. With `BK=32` and the square geometry,
    # `COLS_PER_THREAD` divides 16 (one FP8 scale block per thread's run), so the
    # decode is a single hoisted-scalar * vector with no per-col branch on the
    # interior.
    comptime PACKED_COLS = Self.BK // 2
    comptime THREADS_PER_ROW = Self.THREADS_PER_BLOCK // Self.BN
    comptime BYTES_PER_THREAD = Self.PACKED_COLS // Self.THREADS_PER_ROW
    comptime COLS_PER_THREAD = Self.BYTES_PER_THREAD * 2

    @__name(t"apple_fp4_matmul_run_{Self.c_type}_bm{Self.BM}_bk{Self.BK}")
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        packed_layout: TensorLayout,
        scale_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        packed_engine: TensorEngine,
        scale_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        packed: TileTensor[
            .uint8, packed_layout, ImmutAnyOrigin, Engine=packed_engine
        ],
        scales: TileTensor[
            .float8_e4m3fn, scale_layout, ImmutAnyOrigin, Engine=scale_engine
        ],
        log2_grid_m: UInt32,
        log2_grid_n: UInt32,
    ):
        """W4A16 GEMM kernel entry; `M`/`N`/`K` derive from C, A, and packed.

        Parameters:
            c_layout: Compile-time `TensorLayout` of the output tile `c`.
            a_layout: Compile-time `TensorLayout` of the activation tile `a`.
            packed_layout: Compile-time `TensorLayout` of the packed FP4
                weight `packed`.
            scale_layout: Compile-time `TensorLayout` of the FP8 block
                scales `scales`.
            c_engine: `TensorEngine` of the output tile `c`.
            a_engine: `TensorEngine` of the activation tile `a`.
            packed_engine: `TensorEngine` of the packed FP4 weight `packed`.
            scale_engine: `TensorEngine` of the FP8 block scales `scales`.

        Args:
            c: Output tile `(M, N)` of dtype `c_type`, row-major; receives
                the matmul result.
            a: Activation tile `(M, K)` of `bfloat16`, row-major; the A
                operand of `out = a @ W^T`.
            packed: Packed FP4 weight `(N, K//2)` of `uint8`, lo-nibble
                first; the transposed B operand.
            scales: FP8-E4M3 block scales `(N, ceil(K/16))`; one scale per
                16 K columns.
            log2_grid_m: Base-2 log of the power-of-two M-side grid length
                for the rectangular Morton scheduler.
            log2_grid_n: Base-2 log of the power-of-two N-side grid length
                for the rectangular Morton scheduler.

        C is `(M, N)` row-major, A is `(M, K)` row-major (bf16 activation),
        `packed` is the FP4 weight `(N, K//2)`, `scales` `(N, ceil(K/16))`. Grid
        is `(1<<log2_grid_m) * (1<<log2_grid_n)` threadgroups; OOB threadgroups
        early-return after Morton decode.

        Edge/tail bounds (bit-exact vs materialize -> dense bf16 matmul): the
        cooperative stage zero-fills SMEM entries whose `n_abs >= N` (OOB N-row)
        or whose K column runs past `k_valid` (the partial last strip), so the
        MMA reads exact dequant for valid `(n, k)` and zero elsewhere -- a zero
        B contributes nothing, matching the dense reference's sum over `k < K`.
        Ragged M is handled on the A side via `mma[bounded=True]` (`a_valid_rows`
        zero-fills A rows past M); B from SMEM is treated fully valid because the
        stage already zeroed the OOB/tail entries.
        """
        comptime assert c.flat_rank == 2, "C must be 2D"
        comptime assert a.flat_rank == 2, "A must be 2D"

        var c_ptr = c.ptr
        var a_ptr = a.ptr
        var m = Int(c.dim[0]())
        var n = Int(c.dim[1]())
        var k = Int(a.dim[1]())

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N
        comptime Mma = Self.Mma

        comptime BM_i32: Int32 = Int32(BM)
        comptime BN_i32: Int32 = Int32(BN)
        comptime BK_i32: Int32 = Int32(BK)
        comptime SG_M_i32: Int32 = Int32(SG_M)
        comptime SG_N_i32: Int32 = Int32(SG_N)
        comptime NUM_SG_N_i32: Int32 = Int32(Self.NUM_SG_N)

        var m_i32 = Int32(m)
        var n_i32 = Int32(n)
        var k_i32 = Int32(k)

        var grid_m = ceildiv(m_i32, BM_i32)
        var grid_n = ceildiv(n_i32, BN_i32)

        var tid = Int32(thread_idx.x)
        var sg_id = tid // Int32(WARP_SIZE)
        var sg_m_idx = sg_id // NUM_SG_N_i32
        var sg_n_idx = sg_id % NUM_SG_N_i32

        var tile_mn = Self._Sched.morton_decode_2d_rect(
            UInt32(block_idx.x), log2_grid_m, log2_grid_n
        )
        var tile_m = Int32(tile_mn[0])
        var tile_n = Int32(tile_mn[1])
        if tile_m >= grid_m or tile_n >= grid_n:
            return

        var tg_row = tile_m * BM_i32
        var tg_col = tile_n * BN_i32

        # Double-buffered (2, BN, BK) bf16 SMEM weight sub-tile. Wrapped in a
        # TileTensor view so both buffers are addressed by `(buf, nrow, col)`
        # indexing (in-bounds by construction) -- no raw SMEM pointer arithmetic.
        var b_smem = unsafe_stack_allocation[
            2 * BN * BK,
            Scalar[Self.in_type],
            address_space=.SHARED,
        ]()
        var b_smem_view = TileTensor(
            b_smem,
            Layout(Coord(2, BN, BK), Coord(BN * BK, BK, Idx[1])),
        )
        # Interior coalesced-scale SMEM (lever 1c): (2, BN, NBLK) f32 block
        # scales, staged once per strip. DOUBLE-buffered like `b_smem` above --
        # both `_stage_scales` (write) and the interior `_stage_dequant` (read)
        # index this at buffer `buf = ks % 2 in {0, 1}`, so it must hold both
        # buffers; sizing it single-buffered ran `buf=1` `BN*NBLK` f32 past the
        # end (an SMEM OOB read+write on the interior coalesced path,
        # K/BK >= 2 strips). Sized 1 when disabled (no SMEM); the view is only
        # read on the `coalesce_scales` path.
        comptime NBLK = Self.NBLK_PER_STRIP
        comptime SCALE_SMEM = (2 * BN * NBLK) if Self.coalesce_scales else 1
        var s_smem = unsafe_stack_allocation[
            SCALE_SMEM,
            Float32,
            address_space=.SHARED,
        ]()
        var s_smem_view = TileTensor(
            s_smem,
            Layout(Coord(2, BN, NBLK), Coord(BN * NBLK, NBLK, Idx[1])),
        )

        var mma_op = Mma()
        var accum = Mma.zero_accum()

        var row_base = tg_row + sg_m_idx * SG_M_i32
        var col_base = tg_col + sg_n_idx * SG_N_i32
        # The threadgroup must keep ALL threads live through the cooperative
        # stage + barriers (early-return would deadlock the barrier), so unlike
        # the inline kernel there is no per-simdgroup early return here. The
        # epilogue is bounded so out-of-range simdgroups simply write nothing.
        var sg_row_idx = row_base // SG_M_i32
        var sg_col_idx = col_base // SG_N_i32

        var a_mat = TileTensor(a_ptr, Layout(Coord(m, k), Coord(k, Idx[1])))
        var a_slab = a_mat.tile(Coord(Idx[SG_M], k), Coord(Int(sg_row_idx), 0))

        var k_full_strips = k_i32 // BK_i32
        var has_k_tail = (k_i32 % BK_i32) != 0
        var tail_count: Int32 = 1 if has_k_tail else 0
        var num_strips = k_full_strips + tail_count

        comptime PACKED_COLS = Self.PACKED_COLS
        comptime THREADS_PER_ROW = Self.THREADS_PER_ROW
        comptime BYTES_PER_THREAD = Self.BYTES_PER_THREAD
        comptime COLS_PER_THREAD = Self.COLS_PER_THREAD
        comptime SF = NVFP4_SF_VECTOR_SIZE
        # Two invariants in one: (1) each thread's decode run is exactly one FP8
        # scale block (one hoisted scalar, no per-col branch), AND (2) the
        # decode's 16-bit SIMD width (= COLS_PER_THREAD) stays <= 16 -- a hard M5
        # limit: the Metal backend crashes on >16-lane 16-bit-element SIMD
        # arithmetic. `BK=64`/`BM=128` -> COLS_PER_THREAD=16 (the max); pairing
        # `BK=64` with `BM=64` (-> 32) or `BK>=128`/`BM=128` (-> >=32) trips this.
        comptime assert COLS_PER_THREAD <= SF and (SF % COLS_PER_THREAD) == 0, (
            "cooperative decode needs COLS_PER_THREAD <= 16 (one scale block +"
            " M5's 16-lane 16-bit SIMD limit); check the BM/BK pairing"
        )

        # Decode this thread's `COLS_PER_THREAD` in-bounds cols: one wide packed
        # load, batched nibble expand, branch-free bit-arith decode, scaled by
        # the one block scale the run shares (`COLS_PER_THREAD <= 16`). Shared by
        # the interior path and the bounded path's in-bounds run.
        @always_inline
        @__parameter
        def _decode_run(
            n_abs: Int,
            k0: Int,
            col0: Int,
            col_in_row: Int,
            scale_abs: Float32,
        ) -> SIMD[Self.in_type, COLS_PER_THREAD]:
            var byte0 = col_in_row * BYTES_PER_THREAD
            # This thread's contiguous packed byte-run: one TileTensor width-load
            # (coalesced K; adjacent threads read adjacent bytes) -- no raw ptr.
            var bytes = packed.load[width=BYTES_PER_THREAD, alignment=1](
                Coord(n_abs, (k0 // 2) + byte0)
            )
            var nib = SIMD[.uint16, COLS_PER_THREAD](0)

            comptime for j in range(BYTES_PER_THREAD):
                var bj = UInt16(bytes[j])
                nib[2 * j] = bj & UInt16(0xF)
                nib[2 * j + 1] = (bj >> UInt16(4)) & UInt16(0xF)
            # f32-native decode (no bf16->f32 widen); bit-identical to
            # `E2M1_TO_FLOAT32[nibble]`, so `* scale_abs` matches the materialize
            # oracle exactly. The scale multiply + bf16 cast stay in f32.
            return (decode_e2m1_to_f32(nib) * scale_abs).cast[Self.in_type]()

        # Cooperative DRAM->SMEM dequant of one (BN, BK) weight sub-tile rooted
        # at column `k0`. `bounded` is a comptime fast/slow switch keyed once per
        # tile (like the dense kernel's `is_edge_tile`): the INTERIOR path is
        # branch-free (all N-rows in bounds, a full BK strip) -- a wide packed
        # load, batched nibble decode, one hoisted scale; the BOUNDED path
        # zero-fills OOB N-rows (`n_abs >= N`) and K cols past `k_valid` (the
        # partial last strip), so the MMA reads exact dequant for valid `(n, k)`
        # and a clean zero elsewhere. `k_valid` is the in-bounds K width of this
        # strip (used only on the bounded path).
        @always_inline
        @__parameter
        def _stage_dequant[bounded: Bool](k0: Int32, k_valid: Int32, buf: Int):
            var t = Int(tid)
            var nrow = t // THREADS_PER_ROW
            var col_in_row = t % THREADS_PER_ROW
            var col0 = col_in_row * COLS_PER_THREAD  # first bf16 col in strip
            var n_abs = Int(tg_col) + nrow

            comptime if bounded:
                var kv = Int(k_valid)
                # OOB N-row, or this thread's whole col-run is past the K tail:
                # zero-fill (the MMA must read a clean zero, not stale SMEM).
                if n_abs >= n or col0 >= kv:
                    b_smem_view.store[width=COLS_PER_THREAD](
                        Coord(buf, nrow, col0),
                        SIMD[Self.in_type, COLS_PER_THREAD](0),
                    )
                    return

                var k_abs0 = Int(k0) + col0
                var scale_abs = abs(
                    scales[n_abs, k_abs0 // SF][0].cast[.float32]()
                )
                if col0 + COLS_PER_THREAD <= kv:
                    # Whole run in-bounds: the interior fast path (below).
                    b_smem_view.store[width=COLS_PER_THREAD](
                        Coord(buf, nrow, col0),
                        _decode_run(
                            n_abs, Int(k0), col0, col_in_row, scale_abs
                        ),
                    )
                else:
                    # K tail straddles this thread's run: decode per-col, zero
                    # past `k_valid`. Rare (only the partial last strip's
                    # boundary thread).
                    var vals = SIMD[Self.in_type, COLS_PER_THREAD](0)

                    comptime for i in range(COLS_PER_THREAD):
                        var col = col0 + i
                        if col < kv:
                            var k_abs = Int(k0) + col
                            var byte = packed[n_abs, k_abs >> 1][0]
                            var shift = UInt8(4) if (k_abs & 1) == 1 else UInt8(
                                0
                            )
                            var nibble = UInt16((byte >> shift) & UInt8(0xF))
                            var dec = decode_e2m1_to_bf16(UInt16(nibble))
                            vals[i] = (dec.cast[.float32]() * scale_abs).cast[
                                Self.in_type
                            ]()
                    b_smem_view.store[width=COLS_PER_THREAD](
                        Coord(buf, nrow, col0), vals
                    )
            else:
                # Interior: no bounds branch. One wide contiguous packed load ->
                # nibble expand -> decode -> one hoisted scale. The scale comes
                # from the coalesced SMEM stage (lever 1c) when enabled, else a
                # per-thread scattered load.
                comptime if Self.coalesce_scales:
                    var scale_abs = s_smem_view[buf, nrow, col0 // SF][0]
                    b_smem_view.store[width=COLS_PER_THREAD](
                        Coord(buf, nrow, col0),
                        _decode_run(
                            n_abs, Int(k0), col0, col_in_row, scale_abs
                        ),
                    )
                else:
                    var scale_abs = abs(
                        scales[n_abs, (Int(k0) + col0) // SF][0].cast[
                            DType.float32
                        ]()
                    )
                    b_smem_view.store[width=COLS_PER_THREAD](
                        Coord(buf, nrow, col0),
                        _decode_run(
                            n_abs, Int(k0), col0, col_in_row, scale_abs
                        ),
                    )

        # Coalesced cooperative scale stage (lever 1c): the first `BN*NBLK`
        # threads each load one FP8 block scale -> f32 (abs) into `s_smem[buf]`,
        # so the interior decode reads its scale from SMEM instead of issuing a
        # scattered 1-byte DRAM load. The caller barriers after this before the
        # decode reads it. Only emitted when `coalesce_scales`.
        @always_inline
        @__parameter
        def _stage_scales(k0: Int32, buf: Int):
            comptime NSCALE = BN * NBLK
            var t = Int(tid)
            if t < NSCALE:
                var srow = t // NBLK
                var sblk = t % NBLK
                s_smem_view.store[width=1](
                    Coord(buf, srow, sblk),
                    Float32(
                        abs(
                            scales[Int(tg_col) + srow, Int(k0) // SF + sblk][
                                0
                            ].cast[.float32]()
                        )
                    ),
                )

        @always_inline
        @__parameter
        def _mma_from_smem[
            bounded: Bool
        ](buf: Int, k_strip_local: Int32, valid_rows: Int32):
            # Select this buffer's (BN, BK) plane from the double-buffered view,
            # then descend to the simdgroup's (SG_N, BK) B sub-tile -- all
            # TileTensor tile addressing (the legitimate structured-tiling win;
            # KB exceptions/apple-mma-fragment-is-not-distribute-expressible),
            # no raw SMEM pointer arithmetic.
            var b_plane = b_smem_view.tile[1, BN, BK](buf, 0, 0)
            var b_tile = b_plane.reshape(
                Layout(Coord(BN, BK), Coord(BK, Idx[1]))
            )
            var b_sub = b_tile.tile[SG_N, BK](Int(sg_n_idx), 0)
            var a_sub = a_slab.tile[SG_M, BK](0, Int(k_strip_local))
            # B from SMEM is fully valid (the stage zeroed OOB/tail). Only A's
            # rows can run past M (ragged), so on the bounded path predicate on
            # `a_valid_rows`; B `valid_cols`/`k_valid` cover the full SMEM tile.
            comptime if bounded:
                mma_op.mma[bounded=True](
                    accum,
                    a_sub,
                    b_sub,
                    a_valid_rows=Int(valid_rows),
                    b_valid_cols=SG_N,
                    k_valid=BK,
                )
            else:
                mma_op.mma(accum, a_sub, b_sub)

        # Branch once per threadgroup tile (the dense kernel's idiom): an edge
        # threadgroup (some N-row past N or some M-row past M) OR a partial-K
        # tail forces the bounded stage + MMA; the all-interior, all-full-K case
        # (the production large shapes) is fully branch-free.
        var tg_is_edge = (tg_row + BM_i32 > m_i32) or (tg_col + BN_i32 > n_i32)
        var valid_rows = max(Int32(0), min(SG_M_i32, m_i32 - row_base))

        @always_inline
        @__parameter
        def _run_strips[bounded: Bool]():
            comptime use_coalesced = Self.coalesce_scales and not bounded
            comptime if use_coalesced:
                # Lever 1c (interior only): per strip, coalesced scale-stage ->
                # barrier -> packed-decode (reads SMEM scales) -> barrier -> MMA
                # -> barrier. The extra barrier is cheaper on M5 than the
                # scattered per-thread scale load it removes (measured +7-12% at
                # large M); a software pipeline to hide it loses (M5 pipelining
                # rebuilds the scheduler's job at higher pressure -- PASS notes).
                for ks in range(num_strips):
                    var cur = Int(ks % 2)
                    var k0 = Int32(ks) * BK_i32
                    _stage_scales(k0, cur)
                    barrier()
                    _stage_dequant[False](k0, BK_i32, cur)
                    barrier()
                    _mma_from_smem[False](cur, Int32(ks), valid_rows)
                    barrier()
            else:
                var k0_first = Int32(0)
                var kv_first = min(BK_i32, k_i32)
                _stage_dequant[bounded](k0_first, kv_first, 0)
                barrier()
                for ks in range(num_strips):
                    var cur = Int(ks % 2)
                    var nxt = Int((ks + 1) % 2)
                    if ks + 1 < num_strips:
                        var k0_next = (ks + 1) * BK_i32
                        var kv_next = min(BK_i32, k_i32 - k0_next)
                        _stage_dequant[bounded](k0_next, kv_next, nxt)
                    _mma_from_smem[bounded](cur, ks, valid_rows)
                    barrier()

        # Interior path requires BOTH a non-edge tile AND no K tail (a partial
        # strip always needs the bounded stage for the OOB-K zero-fill).
        if tg_is_edge or has_k_tail:
            _run_strips[True]()
        else:
            _run_strips[False]()

        # ---- Epilogue (mirrors `AppleM5MatMul._run_gemm_body`): bounded store
        # for fp32 fast path; cast/lambda path for non-fp32 out or fused lambda.
        comptime use_epilogue_path = (
            Self.c_type != .float32 or Self.elementwise_lambda_fn
        )
        var valid_cols = max(Int32(0), min(SG_N_i32, n_i32 - col_base))

        comptime if use_epilogue_path:
            var c_mat = TileTensor(c_ptr, row_major(m, n))
            var c_sub = c_mat.tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            comptime elem_align = align_of[Scalar[Self.c_type]]()
            var c_vec = c_sub.vectorize[1, 4]()
            var tile_row_base = Int(row_base)
            var tile_col_base = Int(col_base)

            @always_inline
            @__parameter
            def _write4(
                lrow: Int,
                lcol: Int,
                arow: Int,
                acol: Int,
                v_fp32: SIMD[.float32, 4],
            ):
                var y = v_fp32.cast[Self.c_type]()
                comptime if Self.elementwise_lambda_fn:
                    comptime epilogue = Self.elementwise_lambda_fn.value()
                    if acol + 3 < n:
                        epilogue[Self.c_type, 4, alignment=elem_align](
                            IndexList[2](arow, acol), y
                        )
                    else:
                        for e in range(min(4, n - acol)):
                            epilogue[Self.c_type, 1](
                                IndexList[2](arow, acol + e),
                                SIMD[Self.c_type, 1](y[e]),
                            )
                else:
                    if acol + 3 < n:
                        c_vec.store[alignment=elem_align](
                            Coord(lrow, lcol // 4), y
                        )
                    else:
                        for e in range(min(4, n - acol)):
                            c_sub.store[width=1, alignment=elem_align](
                                Coord(lrow, lcol + e),
                                SIMD[Self.c_type, 1](y[e]),
                            )

            comptime for mi in range(Mma.num_m_mmas):
                comptime for ni in range(Mma.num_n_mmas):
                    var frag = accum[mi * Mma.num_n_mmas + ni]
                    var lcol = ni * 16 + Int(mma_op.cb)
                    var lrow = mi * 16 + Int(mma_op.rb)
                    var acol = tile_col_base + lcol
                    var arow = tile_row_base + lrow
                    if arow < m and acol < n:
                        _write4(
                            lrow, lcol, arow, acol, frag.slice[4, offset=0]()
                        )
                    if arow + 8 < m and acol < n:
                        _write4(
                            lrow + 8,
                            lcol,
                            arow + 8,
                            acol,
                            frag.slice[4, offset=4](),
                        )
        else:
            var c_ptr_fp32 = rebind[UnsafePointer[Float32, MutAnyOrigin]](c_ptr)
            var c_mat_fp32 = TileTensor(c_ptr_fp32, row_major(m, n))
            var c_sub_fp32 = c_mat_fp32.tile[SG_M, SG_N](
                Int(sg_row_idx), Int(sg_col_idx)
            )
            mma_op.store_bounded(
                accum, c_sub_fp32, Int(valid_rows), Int(valid_cols)
            )


# M >= this uses the wider BM=128 tile (deep dequant amortization across 8
# co-resident simdgroups); below it the BM=64 tile keeps occupancy up when the
# 128-tall tile would waste rows. Crossover measured on M5 Max
# (bench_apple_fp4_smem_bk): at large M BM=128 wins ~+12-16%, at M<=64 BM=64
# wins ~+30-50%. Threshold set conservatively at the first M where BM=128 fills
# both 64-row halves of a 128-row tile. (Used only by the small-M FUSED path
# below; large M takes the materialize -> dense path, which has no BM choice.)
comptime _FP4_TALL_M_THRESHOLD = 256

# M >= this materializes the FP4 weight to a TRANSIENT bf16 buffer and runs the
# dense bf16 MMA, instead of the cooperative-SMEM fused dequant. WHY: the fused
# kernel's per-element FP4 decode sits on the MMA's critical path (the M5 cannot
# co-issue the scalar-ALU decode with the matrix MMA -- they serialize through a
# shared per-core issue path; measured, see the kernel docstring + the agent KB
# "apple-m5-quantized-matmul-ceiling"). Materializing pays a one-shot dense
# decode pass (memory-bound, near roofline) + a dense bf16 GEMM (also near
# roofline, dequant fully off its critical path), so at LARGE M it beats the
# fused path on M5 Max (24-28 vs 18-22 TF/s at M=2048, ~1.25-1.34x).
#
# The transient buffer is one weight's worth of bf16 (N*K), allocated at
# EXECUTION time inside the launcher -- it does NOT become a persistent graph
# constant, so the packed weight stays 4-bit in DRAM at rest (no 4x blow-up).
#
# THRESHOLD = 1536 (NOT 256), and this is the load-bearing measured result:
# the per-call `enqueue_create_buffer(N*K bf16)` + free is NOT free -- it costs
# ~0.4 ms (square N=K=3072) to ~2.2 ms (wide N=12288) per call, which ERASES the
# materialize win at moderate M. Measured crossover (fused BM=128 vs
# materialize-WITH-real-alloc, M5 Max, median of 3, bench_apple_fp4_crossover):
#   square N=K=3072:   M=1024 ~tie, M>=1152 materialize wins 1.09-1.18x,
#                      M=512-1024 FUSED wins 0.85-0.98x.
#   wide-N N=12288:    M=1024 fused 0.89x, M=1536 materialize 1.10x.
#   wide-K K=12288:    M=1536 materialize 1.24x.
# So 1536 is the lowest M past the crossover for EVERY measured shape (square /
# wide-N / wide-K) with >=1.10x margin; below it the fused path is faster or
# tied, so routing M<1536 to fused is never a regression. (The brief's "M~512"
# crossover was from a bench that PRE-ALLOCATED the bf16 buffer outside the hot
# loop; the real per-call alloc the launcher pays roughly doubles the crossover.
# This is the "per-call alloc changes the picture, report it with numbers"
# case.) Re-derive: bench_apple_fp4_crossover.mojo.
comptime _FP4_MATERIALIZE_M_THRESHOLD = 1536

# The `matmul2d` W4A16 kernel (`enqueue_matmul2d_fp4_smem`, the wider 16x32x16
# native `matmul2d`-tile MMA) is the DEFAULT only in the DEEP-K niche (K >=
# 18432, M >= 1024; see `_M2D_DEEPK_*`), and an OPT-IN alternative elsewhere.
# Off the deep-K niche the crossover (M5 Max, median of 3, FLUX-representative
# MID-K shapes) found it dominated at every M by whichever incumbent band the
# default selects (this is why it is NOT the general default):
#
#   | M    | m2d_smem | FUSED(BK64) | MAT->DENSE | winner       |
#   | 64   | 1.8-2.1  | 6.6-10.6    | 2.6-3.0    | FUSED (m2d 0.20-0.29x)
#   | 256  | 7.1-7.7  | 19.0-24.8   | 9.9-11.9   | FUSED (m2d 0.29-0.40x)
#   | 512  | 13.0-15.6| 23.3-26.4   | 18.6-21.0  | FUSED (m2d 0.49-0.66x)
#   | 1024 | 33.9     | 24.9-27.7   | 37.0      | MAT   (m2d 0.92x, mid-K)
#   | 2048 | 41.1     | 25.7-27.7   | 44.4      | MAT   (m2d 0.93x, mid-K)
#   | 4096 | 40.1     | 27.4-28.7   | 49.0      | MAT   (m2d 0.82x, mid-K)
#
# (The M=1024-4096 rows are the PASS-23 mid-K square 3072^2 at smem_bk=256; MAT
# wins every mid-K cell M >= 1024 because at mid-K the materialized weight is
# small enough that MAT->DENSE stays ~roofline -- no DRAM wall. At small M
# m2d_smem starves: its tuned tile is TG_M=1024 (16 SG x tm=4), so at M=64 only
# ~6% of the tile does useful work.) The wall only INVERTS the result at deep-K
# (K >= 18432, >= 226 MB bf16 weight), which is the `_M2D_DEEPK_*` default route.
# `use_matmul2d=True` forces m2d on ALL aligned shapes (for A/B and future
# geometry that could reach MLX's 40-53 MIDPOINT), gated on the interior-tile
# alignment it requires (N % TG_N == 0, K % smem_bk == 0) with a fall-back to the
# incumbent when unaligned.
#
# The `matmul2d` kernel's interior-only geometry (production default 16/1/4/1/256
# -> TG_N=32, smem_bk=256): N must be a multiple of TG_N=32 and K a multiple of
# smem_bk=256 (the committed fused/materialize paths handle ragged N/K, so the
# opt-in must fall back to them off the aligned interior). smem_bk=256 gives
# COLS_PER_THREAD=16 (the M5 SIMD-width max at 512 threads), the tuned optimum,
# so the default kernel geometry (and hence this alignment gate) is 256.
comptime _M2D_TG_N = 32
comptime _M2D_SMEM_BK = 256

# DEEP-K niche: at deep K the MAT->DENSE path must READ a large materialized bf16
# weight (6144*18432*2 = 226 MB at the real FLUX.2-dev FFN-down) which hits the M5
# DRAM wall, while `matmul2d_fp4` stays 4-bit-in-DRAM (decode->SMEM->register B
# frag, MMA register-fed) and is DRAM-wall-IMMUNE (holds ~41-42 TF/s flat). So at
# deep-K + large-M, m2d BEATS the walled incumbent -- the one regime where it wins.
# THERMAL-FAIR interleaved A/B (shared warmup, best-of-8), N=6144 K=18432, MAT vs
# m2d(BK256):
#   M=512 : MAT 21.5 / m2d 21.2  (MAT wins 0.99x -- m2d's TG_M=1024 tile starves)
#   M=1024: MAT 37.4 / m2d 42.0  (m2d wins 1.12x)
#   M=2048: MAT 41.6 / m2d 42.5  (m2d wins 1.02x -- narrow but consistent)
#   M=4096: MAT 30.3 / m2d 39.6  (m2d wins 1.31x -- MAT walls hard here)
# So at K >= 18432 the deep-K crossover is M >= 1024 (m2d never below MAT there,
# 1.02-1.31x). At MID-K (K <= 12288) the wall does NOT bite (MAT stays ~roofline),
# so MAT beats m2d at every M >= 2048 even at BK256 (m2d 0.83-0.98x MAT, same
# harness) -- the deeper strip does NOT widen the niche into mid-K. Hence the
# predicate is a DEEP-K gate (K >= 18432), NOT a general large-M gate. The lone
# extra win, wide-N 12288x3072 M=1024 (m2d 1.03x MAT), is a mid-K FUSED->MAT band
# seam and is NOT captured (too narrow / not deep-K) -- the incumbent handles it.
# Re-derive: bench_apple_fp4_reconcile.mojo. Neither packed kernel reaches MLX-q4
# here (MLX ~43, also 4-bit-in-DRAM, ~0.95-0.98x its own dense); m2d closes the
# MAT gap (up to 1.31x), not the MLX gap (~0.95-0.98x MLX-q4 at M>=1024).
comptime _M2D_DEEPK_K_THRESHOLD = 18432
comptime _M2D_DEEPK_M_THRESHOLD = 1024


@always_inline
def _enqueue_apple_fp4_materialize_dense[
    c_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Materialize the FP4 weight to a transient bf16 buffer, then dense GEMM.

    The large-M path of `enqueue_apple_fp4_matmul`. Allocates a transient
    `[N, K]` bf16 weight buffer at execution time, dequantizes the packed FP4
    weight + FP8 block scales into it (`enqueue_fp4_materialize`, the same
    per-element `E2M1 * |block_scale|` decode the fused kernel and the test
    oracle use -- so this path is bit-exact vs the fused path), then runs the
    stock dense bf16 MMA (`enqueue_apple_matmul`, `transpose_b=True`) reading the
    materialized weight. The optional fused epilogue is threaded straight through
    to the dense matmul (which applies it on the dense path). The NVFP4
    `weight_scale_2` per-tensor scalar is applied OUTSIDE the kernel by the graph
    lowering (a post-matmul multiply), identically for both paths.

    Buffer lifetime (the correctness hinge): the dense GEMM reads `wdense` after
    this function returns -- `enqueue_*` is async. `DeviceBuffer.__deinit__`
    schedules a STREAM-ORDERED free (`AsyncRT_DeviceBuffer_release`: "the actual
    deallocation may occur asynchronously after all operations using this buffer
    have completed", `device_context.mojo`), so the free cannot race the GEMM
    even on one in-order stream. The `_ = wdense_dev^` at the end pins the handle
    alive until AFTER both `enqueue_*` calls have been issued onto the stream
    (otherwise the handle would drop at its last textual use, before the GEMM is
    enqueued). Same pattern as `mxfp4_dequant_matmul_amd` (the AMD W4A16 sibling
    custom-op launcher), which allocates its FP8 scratch the same way and ends
    with `_ = b_fp8_buf^`.
    """
    var wdense_dev = ctx.enqueue_create_buffer[.bfloat16](n * k)
    var wdense_tt = TileTensor(wdense_dev.unsafe_ptr(), row_major(n, k))

    enqueue_fp4_materialize[.bfloat16](wdense_tt, packed, scales, ctx)

    if ctx.compute_capability() == 5:
        enqueue_apple_matmul[
            in_type=DType.bfloat16,
            c_type=c_type,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ](c, a, wdense_tt.as_immut(), ctx)
    else:
        comptime apple_kernel = gemm_kernel_apple_8x8[
            c_type,
            DType.bfloat16,
            DType.bfloat16,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(wdense_tt).LayoutType,
            type_of(c).Engine,
            type_of(a).Engine,
            type_of(wdense_tt).Engine,
            transpose_b=True,
            elementwise_lambda_fn=elementwise_lambda_fn,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=16,
            NUM_SIMDGROUPS=4,
        ]
        ctx.enqueue_function[apple_kernel](
            c,
            a,
            wdense_tt.as_immut(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(n, 64), ceildiv(m, 64)),
            block_dim=(4 * WARP_SIZE,),
        )

    # Keep the transient weight alive through the async materialize + GEMM
    # enqueue (see the buffer-lifetime note in the docstring).
    _ = wdense_dev^


@always_inline
def _launch_apple_fp4_matmul[
    c_type: DType,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type],
    *,
    BM: Int,
    BK: Int,
    coalesce_scales: Bool,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    m: Int,
    n: Int,
    ctx: DeviceContext,
) raises:
    """Build the rectangular-Morton launch for one (BM, BK) and enqueue it.

    Factored out of `enqueue_apple_fp4_matmul` so the M-adaptive dispatch can
    select the tile geometry `BM`/`BK` (and whether to coalesce the scale load)
    and launch the matching kernel instantiation. `grid_m`/`grid_n` use the
    geometry's `BM`/`BN`, so the Morton decode is consistent with the geometry
    the kernel reads from its own `Self.BM`.
    """
    comptime MM = AppleM5Fp4MatMul[
        c_type,
        elementwise_lambda_fn,
        BM=BM,
        BK=BK,
        coalesce_scales=coalesce_scales,
    ]

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    var side_m = 1
    var log2_m: UInt32 = 0
    while side_m < grid_m:
        side_m *= 2
        log2_m += 1
    var side_n = 1
    var log2_n: UInt32 = 0
    while side_n < grid_n:
        side_n *= 2
        log2_n += 1

    var grid_dim = side_m * side_n

    comptime kernel = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(packed).LayoutType,
        type_of(scales).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(packed).Engine,
        type_of(scales).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        packed.as_immut(),
        scales.as_immut(),
        log2_m,
        log2_n,
        grid_dim=(grid_dim),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


@always_inline
def enqueue_apple_fp4_matmul[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    *,
    use_matmul2d: Bool = False,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    packed: TileTensor[.uint8, ...],
    scales: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the W4A16 matmul: `out = a @ dequant(packed, scales)^T`.

    `a` is the bf16 activation `(M, K)`, `packed` the FP4 weight `(N, K//2)`
    (lo-nibble first), `scales` the FP8-E4M3 block scales `(N, ceil(K/16))`. C is
    `(M, N)`.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract), threaded to whichever path runs.
        use_matmul2d: OPT-IN OVERRIDE. When `True`, route EVERY aligned interior
            (N % `_M2D_TG_N=32` == 0, K % `_M2D_SMEM_BK=256` == 0) to the native
            `matmul2d`-tile W4A16 kernel
            (`enqueue_matmul2d_fp4_smem`), falling back to the incumbent only off
            that aligned interior. DEFAULT `False` -- off the deep-K niche the P1
            crossover measured `matmul2d` dominated at every production M by the
            incumbent bands (FUSED at M<=512, MAT->DENSE at mid-K M>=1024), so it
            is not forced in general; the flag keeps it reachable for A/B and
            future geometry. NOTE this override is ORTHOGONAL to the deep-K niche
            below, which routes to `matmul2d` BY DEFAULT (`use_matmul2d=False`)
            where it actually wins. See the `_M2D_*` threshold comments.

    Args:
        c: Output tile `(M, N)` of dtype `c_type`; receives the matmul
            result.
        a: Activation tile `(M, K)` of `bfloat16`; the A operand.
        packed: Packed FP4 weight `(N, K//2)` of `uint8`, lo-nibble first;
            the transposed B operand.
        scales: FP8-E4M3 block scales `(N, ceil(K/16))`; one scale per 16 K
            columns.
        ctx: Device context used to enqueue the kernel(s).

    The default strategy (`use_matmul2d=False`) is chosen by (K, M) -- all paths
    produce bit-identical dequant (the E2M1 * |scale| arithmetic is the same on
    every path; the GEMV differs only by its fp32 reduction order vs the MMA):

    - Batch-1 decode (`M == 1`): the register-resident W4A16 GEMV
      (`enqueue_apple_fp4_gemv`), no MMA to feed. See `fp4_gemv.mojo` for the
      rationale (bandwidth win, ~1.53x at the Llama-8B down-proj shape).
    - Deep-K niche (`K >= 18432` AND `M >= 1024`, aligned interior): the
      `matmul2d` W4A16 kernel (`enqueue_matmul2d_fp4_smem`). At deep K the
      MAT->DENSE path below must read a large (>=226 MB) materialized bf16 weight
      and hits the M5 DRAM wall, while `matmul2d` stays 4-bit-in-DRAM and holds
      ~42 TF/s -- measured 1.02-1.31x over the walled MAT at M in {1024,2048,4096}
      (thermal-fair A/B). This is the real FLUX.2-dev FFN-down (N=6144, K=18432)
      at prefill. Only when there is no fused epilogue (the `matmul2d` interior
      store does not yet apply the lambda).
    - Large M (`M >= 1536`): MATERIALIZE the weight to a transient `(N, K)` bf16
      buffer (`enqueue_fp4_materialize`), then run the stock dense bf16 MMA
      (`enqueue_apple_matmul`). The FP4 decode is off the MMA critical path, so
      this beats the fused path ~1.25-1.34x on M5 (which serializes the
      scalar-ALU decode against the matrix MMA). The transient buffer is
      execution-time only -- it is NOT a persistent graph constant, so the
      packed weight stays 4-bit in DRAM at rest. The 1536 crossover (vs a naive
      ~512) accounts for the per-call buffer alloc/free the launcher pays.
    - Mid M (`256 <= M < 1536`): the FUSED cooperative-SMEM path with `BM=128`
      (deep dequant amortization across 8 co-resident simdgroups). Faster than
      materialize here once the per-call alloc is paid.
    - Small M (`M < 256`): the FUSED path with `BM=64` (keeps occupancy up when
      a 128-tall tile would waste rows).

    In the fused bands each `(BN, BK)` weight sub-tile is cooperatively
    dequantized to bf16 in threadgroup memory once per K-strip, then the dense
    bf16 MMA reads B from SMEM; weights stay packed in DRAM the whole time.

    On pre-M5 Apple silicon GPUs the M5-only fused /
    matmul2d paths are skipped: the call routes unconditionally to
    materialize->dense (FP4 decode to a transient bf16 buffer, then the pre-M5
    dense bf16 `gemm_kernel_apple_8x8`).
    """
    var cc = ctx.compute_capability()

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_apple_fp4_matmul: c_type must be one of {fp16, bf16, fp32}"

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(packed.dim[0]()) == n, "packed must be (N, K//2)")
    debug_assert(Int(packed.dim[1]()) == k // 2, "packed must be (N, K//2)")
    debug_assert(Int(scales.dim[0]()) == n, "scales must be (N, ceil(K/16))")
    # MmaOpApple narrows the A row stride to UInt16; A-slab stride = K.
    debug_assert(
        k <= 65535, "Apple FP4 matmul: K must fit in UInt16; got K=", k
    )

    # Pre-M5 (M1-M4): route unconditionally to materialize->dense, which
    # decodes the FP4 weight to a transient bf16 buffer (hardware-neutral) and
    # runs the pre-M5 dense bf16 `gemm_kernel_apple_8x8`.
    if cc != 5:
        # NVFP4's block size is 16, so K is always a multiple of 16 (hence 8)
        # for well-formed weights, however to be safe we assert on a malformed
        # K instead of producing silently bad results.
        debug_assert(
            k % 16 == 0,
            (
                "Apple pre-M5 FP4 matmul: K must be a multiple of 16 (NVFP4"
                " block size); the 8x8 GEMM truncates ragged K. Got K="
            ),
            k,
        )
        _enqueue_apple_fp4_materialize_dense[c_type, elementwise_lambda_fn](
            c, a, packed, scales, m, n, k, ctx
        )
        return

    # `matmul2d` W4A16 path (`enqueue_matmul2d_fp4_smem`). Its interior kernel is
    # tile-aligned (N % TG_N == 0, K % smem_bk == 0), so every route to it is
    # gated on that alignment with a fall-through to the incumbent (which handles
    # ragged N/K). `elementwise_lambda_fn` is not yet wired on the `matmul2d`
    # interior store, so only route the no-epilogue case.
    comptime _m2d_no_epi = not elementwise_lambda_fn
    var m2d_aligned = (n % _M2D_TG_N) == 0 and (k % _M2D_SMEM_BK) == 0

    # (1) DEEP-K niche = the DEFAULT route where m2d actually wins: at K >= 18432
    # the MAT->DENSE incumbent hits the M5 DRAM wall reading its 226 MB+
    # materialized bf16 weight, while m2d stays 4-bit-in-DRAM and holds ~42 TF/s.
    # Measured (thermal-fair A/B): m2d beats MAT 1.02-1.31x at M >= 1024,
    # K >= 18432 (loses at M=512; at MID-K the wall doesn't bite so MAT wins --
    # hence a DEEP-K gate, not a general large-M gate). This is the real FLUX.2-dev
    # FFN-down (N=6144, K=18432) at prefill M ~ 4096. See the `_M2D_DEEPK_*`
    # threshold comment for the full crossover.
    comptime if _m2d_no_epi:
        if (
            m2d_aligned
            and k >= _M2D_DEEPK_K_THRESHOLD
            and m >= _M2D_DEEPK_M_THRESHOLD
        ):
            enqueue_matmul2d_fp4_smem[c_type=c_type](c, a, packed, scales, ctx)
            return

    # (2) OPT-IN force to m2d (A/B + future geometry). Off the deep-K niche m2d is
    # dominated (MAT wins mid-K large-M, FUSED wins small-M; see the `_M2D_*`
    # table), so this is NOT taken by default -- only when the caller passes
    # `use_matmul2d=True`, and still only on the aligned interior.
    comptime if use_matmul2d and _m2d_no_epi:
        if m2d_aligned:
            enqueue_matmul2d_fp4_smem[c_type=c_type](c, a, packed, scales, ctx)
            return

    # (3) BATCH-1 DECODE: M == 1 -> register-resident W4A16 GEMV, no MMA; see
    # fp4_gemv.mojo for the full rationale.
    if m == 1:
        enqueue_apple_fp4_gemv[c_type, elementwise_lambda_fn](
            c, a, packed, scales, n, k, ctx
        )
        return

    # Three-way M-adaptive dispatch (two independent crossovers, both measured):
    if m >= _FP4_MATERIALIZE_M_THRESHOLD:
        # Large M (>= 1536): materialize the FP4 weight to a transient bf16
        # buffer and run the dense bf16 MMA. The fused dequant is OFF the MMA
        # critical path (one dense decode pass + a dense GEMM, both ~roofline),
        # beating the fused cooperative-SMEM path ~1.25-1.34x on M5 (which
        # serializes the scalar-ALU decode against the matrix MMA). The
        # transient buffer is execution-time only (not a graph constant) -- no
        # persistent 4x weight blow-up. (The per-call alloc/free is what pushes
        # this crossover to 1536 rather than ~512; see the threshold comment.)
        _enqueue_apple_fp4_materialize_dense[c_type, elementwise_lambda_fn](
            c, a, packed, scales, m, n, k, ctx
        )
    elif m >= _FP4_TALL_M_THRESHOLD:
        # Mid M ([256, 1536)): fused cooperative-SMEM, BM=128, BK=64 -- deep
        # dequant amortization across 8 co-resident simdgroups, AND the deep K
        # strip (one cooperative decode feeds 4 MMA K-steps) that measured
        # +34-45% over BK=32 at large M, bit-exact. Faster than materialize once
        # the per-call alloc is paid (the crossover sweep shows fused winning or
        # tied for M < 1536). Coalesce the scale load into SMEM (+7-12%; the
        # extra barrier is hidden by the wide tile's high occupancy).
        _launch_apple_fp4_matmul[
            c_type, elementwise_lambda_fn, BM=128, BK=64, coalesce_scales=True
        ](c, a, packed, scales, m, n, ctx)
    else:
        # Small M (< 256): fused cooperative-SMEM, BM=64 -- keeps occupancy up
        # (no wasted rows in a 128-tall tile). Keep the scattered per-thread
        # scale load: the coalesce barrier regresses at this low occupancy.
        # BK=32 here (NOT 64): at BM=64 a BK=64 strip forces COLS_PER_THREAD=32,
        # which crashes the M5 Metal backend (>16-lane 16-bit SIMD arithmetic);
        # BK=64's win needs the BM=128 thread count to keep COLS_PER_THREAD=16.
        _launch_apple_fp4_matmul[
            c_type, elementwise_lambda_fn, BM=64, BK=32, coalesce_scales=False
        ](c, a, packed, scales, m, n, ctx)
