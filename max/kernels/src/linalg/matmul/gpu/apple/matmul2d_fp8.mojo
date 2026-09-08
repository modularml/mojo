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
"""Apple M5 weight-only FP8 (W8A16) tiled matmul: bf16 A x fp8 B -> fp32.

Target: Apple M5 (`compute_capability == 5`) only.

The DENSE W8A16 path (`enqueue_matmul2d_fp8`) now runs on the shared
`AppleM5MatMul` simdgroup-tiled GEMM body (`matmul_kernel.mojo`) specialized on
`b_type=float8_e4m3fn` -- the weight-loader seam swap -- so it inherits the dense
body's Morton schedule and epilogue with only the B operand widened. What remains
here is the GROUPED (MoE) `Matmul2dFp8.run_grouped` + `_gemm_body`, pending its
own migration onto the shared body's group-indexed wrapper.

`out = activation @ weight^T` where the FP8 weight stays `float8_e4m3fn [N, K]`
in DRAM (one byte per element -- NOT packed, NO block scales). This is the
tiled-MMA decode kernel that beats the warp-per-column GEMV's structural ceiling
at large-N / small-K (Mamba in_proj `[17504, 3136]`, MLP up `[12544, 3136]`),
where the GEMV is capped ~250 GB/s while a tiled matmul amortizes the per-output
cost across a threadgroup tile.

## Why this structure (NOT the FP4 `matmul2d` transpose_right / SMEM path)

FP4 (`matmul2d_fp4.mojo`) decodes through SMEM because its dequant (nibble unpack
+ E2M1 LUT + per-16 block scale) is expensive and its per-lane B gather is
N-scattered -- coalescing it through SMEM took it 10 -> 38 TF/s (KB
`apple-m5-gpu-perf-model`). FP8 has NO such decode: the widen is native. Crucially,
the Apple simdgroup MMA (`_mma_apple_transposable`, `mma_apple.mojo:213`) accepts
an FP8 B operand DIRECTLY (`float8_e4m3fn` is in its valid-float input set; KGEN
lowers the fp8 fragment to AIR's `<8 x i8>`), so a bf16 A x fp8 B -> fp32 MMA is a
single native instruction -- no manual widen, no SMEM staging.

So this kernel mirrors the WINNING dense bf16 structure (`AppleM5MatMul`), which
sustains ~400 GB/s at exactly this M=1 large-N/small-K decode regime: a
simdgroup-tiled GEMM with `MmaOpApple`, the B operand loaded DIRECTLY from DRAM
each K-strip (Apple has no async copy and SMEM staging DEGRADES matmul -- KB
`apple-m5-gpu-performance-considerations`), coalesced via the col-major `(K, N)`
weight view (K-contiguous fast axis). The ONLY difference from the dense bf16
path is `MmaOpApple`'s `b_type = float8_e4m3fn` (1 byte/weight vs 2) -- the
"weight-load seam" swap. `MmaOpApple` already carries a separate `b_type` and
loads A / B fragments independently, so this needs NO change to the shared MMA op
or the bf16 matmul.

The per-tensor scalar `weight_scale` folds in OUTSIDE the kernel (a post-matmul
multiply by the graph lowering), identically to the GEMV: a scalar factors out of
the fp32 sum, so the fold is EXACT.

Ragged M/N/K are handled by `MmaOpApple`'s bounded load (zero-fill OOB) + the
`< M`/`< N` store guard -- no tile-alignment requirement (unlike the FP4
interior-only `matmul2d`). The 2x2 (SG=32) simdgroup subtile is the M5
register-cliff optimum (KB `apple-m5-gpu-perf-model`); do NOT exceed 4
accumulators.
"""

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import align_of

from layout import Idx, TensorEngine, TileTensor
from layout.coord import Coord
from layout.tile_layout import Layout, TensorLayout

from linalg.arch.apple.mma import MmaOpApple
from linalg.matmul.gpu.apple.matmul2d_fp4 import _require_apple_m5
from linalg.matmul.gpu.apple.matmul_kernel import AppleM5MatMul
from linalg.utils import elementwise_epilogue_type


struct Matmul2dFp8[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
]:
    """W8A16 simdgroup-tiled GEMM: bf16 A x fp8 B -> fp32, direct DRAM B feed.

    Mirrors `AppleM5MatMul` (the ~400 GB/s dense bf16 decode path) with the sole
    change that the B (weight) operand is `float8_e4m3fn` via `MmaOpApple`'s
    `b_type`. A is bf16, C is `c_type`, accumulation fp32. No SMEM, no barrier:
    each K-strip's B fragments load straight from DRAM (coalesced K-contiguous via
    the col-major `(K, N)` view) and feed the native fp8-capable MMA.

    Parameters:
        c_type: Output element type (fp16, bf16, fp32). Accumulation is fp32.
        elementwise_lambda_fn: Optional fused epilogue (AMD's `(row, col)`
            contract). Currently unused on the interior store; the launcher routes
            fused-epilogue shapes elsewhere.
        block_m: Threadgroup block rows `BM` (multiple of `SG_M`).
        block_n: Threadgroup block cols `BN` (multiple of `SG_N`).
        block_k: K-strip depth `BK` per accumulate step (multiple of 16).
        sg_m: Simdgroup subtile rows `SG_M` (multiple of 16).
        sg_n: Simdgroup subtile cols `SG_N` (multiple of 16). `sg_m/16 * sg_n/16`
            is the accumulator count; `2x2 == 4` is the M5 optimum (do NOT exceed).
    """

    comptime in_type = DType.bfloat16  # activation dtype (A + MMA input)
    comptime b_type = DType.float8_e4m3fn  # weight dtype (B, native fp8 MMA)
    comptime BM = Self.block_m
    comptime BN = Self.block_n
    comptime BK = Self.block_k
    comptime SG_M = Self.sg_m
    comptime SG_N = Self.sg_n
    comptime NUM_MMA_M = Self.SG_M // 16
    comptime NUM_MMA_N = Self.SG_N // 16
    comptime NUM_SG_M = Self.BM // Self.SG_M
    comptime NUM_SG_N = Self.BN // Self.SG_N
    comptime NUM_SG = Self.NUM_SG_M * Self.NUM_SG_N
    comptime THREADS_PER_BLOCK = Self.NUM_SG * WARP_SIZE
    # bf16 A x fp8 B -> fp32. `b_type` is the ONLY deviation from the dense bf16
    # `AppleM5MatMul.Mma`; transpose is supplied by the col-major (K, N) B view
    # (hw_transpose_b = b_col_major XOR transpose_b), matching the dense path.
    comptime Mma = MmaOpApple[
        DType.float32,
        Self.in_type,
        Self.NUM_MMA_M,
        Self.NUM_MMA_N,
        b_type=Self.b_type,
    ]

    @always_inline
    @staticmethod
    def _gemm_body[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        w_layout: TensorLayout,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=_],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=_],
        weight: TileTensor[Self.b_type, w_layout, ImmutAnyOrigin, Engine=_],
        M: Int,
        N: Int,
        K: Int,
        active: Bool = True,
    ):
        """Shared per-threadgroup-tile GEMM body: C `(M, N)` = A `(M, K)` @ W
        `(N, K)`^T for ONE contiguous `(M, K)` A slab / `(N, K)` weight slab.

        The grouped `run_grouped` (one expert group per `block_idx.z`) drives this
        body with per-slab base views. (The dense W8A16 path now runs on the
        shared `AppleM5MatMul` body via `b_type=fp8`; this grouped body is the
        remaining standalone piece, pending its own migration onto the shared
        body's group-indexed wrapper.)

        `active` gates the K-loop only: when False (an `expert_ids[z] == -1`
        inactive LoRA group) the zero accumulator is stored unchanged, writing
        zeros for the group's rows -- matching `naive_grouped_matmul_kernel`.
        The tiling is driven by `block_idx.y`/`block_idx.x`; `block_idx.z` (the
        group) is resolved by the caller into the base views.
        """
        # The tiled store does NOT apply a fused epilogue -- dispatch routes
        # fused-epilogue shapes off this path (the grouped launcher hardcodes
        # None). Fail loud so a future caller wiring a lambda can't have it
        # silently dropped.
        comptime assert not Self.elementwise_lambda_fn, (
            "matmul2d_fp8: elementwise_lambda_fn is not applied by the tiled"
            " store"
        )

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime SG_M = Self.SG_M
        comptime SG_N = Self.SG_N

        var tile_m = Int(block_idx.y)
        var tile_n = Int(block_idx.x)

        var sg_id = Int(thread_idx.x) // WARP_SIZE
        var sg_m_idx = sg_id // Self.NUM_SG_N
        var sg_n_idx = sg_id % Self.NUM_SG_N

        var row_base = tile_m * BM + sg_m_idx * SG_M
        var col_base = tile_n * BN + sg_n_idx * SG_N

        # Fully-OOB simdgroups (incl. every group with M == 0): no later
        # threadgroup-uniform op (no SMEM/barrier), so the early return is safe
        # (matches `AppleM5MatMul._run_gemm_body`).
        if row_base >= M or col_base >= N:
            return

        var sg_row_idx = row_base // SG_M
        var sg_col_idx = col_base // SG_N

        # A `(M, K)` row-major; this simdgroup's `(SG_M, K)` slab (K hoisted out
        # of the K-loop, matching `DenseALoader`). B `(N, K)` fp8 viewed col-major
        # `(K, N)` (stride (1, K)) -> the K axis is the contiguous fast axis of the
        # B-fragment load (coalesced), and hw_transpose_b = True feeds it as the
        # right operand. This simdgroup's `(K, SG_N)` col slab.
        var a_ptr = a.ptr.unsafe_origin_cast[ImmUntrackedOrigin]()
        var a_mat = TileTensor(a_ptr, Layout(Coord(M, K), Coord(K, Idx[1])))
        var a_slab = a_mat.tile(Coord(Idx[SG_M], K), Coord(Int(sg_row_idx), 0))

        var w_ptr = weight.ptr.unsafe_origin_cast[ImmUntrackedOrigin]()
        var b_mat = TileTensor(w_ptr, Layout(Coord(K, N), Coord(Idx[1], K)))
        var b_slab = b_mat.tile(Coord(K, Idx[SG_N]), Coord(0, Int(sg_col_idx)))

        var mma_op = Self.Mma()
        var accum = Self.Mma.zero_accum()

        var sg_m_end = row_base + SG_M
        var sg_n_end = col_base + SG_N
        var is_edge_tile = (sg_m_end > M) or (sg_n_end > N)

        var k_full_strips = K // BK
        var has_k_tail = (K % BK) != 0

        # `active == False` (expert_ids[z] == -1): skip the matmul, store zeros.
        if active:
            if is_edge_tile:
                var valid_rows = max(1, min(SG_M, M - row_base))
                var valid_cols = max(1, min(SG_N, N - col_base))
                var k_total = k_full_strips + (1 if has_k_tail else 0)
                for ks in range(k_total):
                    var k_valid = min(BK, K - ks * BK)
                    var a_sub = a_slab.tile[SG_M, BK](0, ks)
                    var b_sub = b_slab.tile[BK, SG_N](ks, 0)
                    mma_op.mma[bounded=True](
                        accum,
                        a_sub,
                        b_sub,
                        a_valid_rows=valid_rows,
                        b_valid_cols=valid_cols,
                        k_valid=k_valid,
                    )
            else:
                for ks in range(k_full_strips):
                    var a_sub = a_slab.tile[SG_M, BK](0, ks)
                    var b_sub = b_slab.tile[BK, SG_N](ks, 0)
                    mma_op.mma[bounded=False](
                        accum,
                        a_sub,
                        b_sub,
                        a_valid_rows=SG_M,
                        b_valid_cols=SG_N,
                        k_valid=BK,
                    )
                if has_k_tail:
                    var k_tail = K - k_full_strips * BK
                    var a_sub = a_slab.tile[SG_M, BK](0, k_full_strips)
                    var b_sub = b_slab.tile[BK, SG_N](k_full_strips, 0)
                    mma_op.mma[bounded=True](
                        accum,
                        a_sub,
                        b_sub,
                        a_valid_rows=SG_M,
                        b_valid_cols=SG_N,
                        k_valid=k_tail,
                    )

        # Store this simdgroup's accumulators via the vectorized cast-store
        # `_write4` -- 2 width-4 vector stores per 16x16 fragment, NOT the old
        # hand-rolled 8x per-element scatter. Each fp32 fragment's lanes 0..3
        # map to row `rb` and lanes 4..7 to row `rb+8`, cols `cb..cb+3` -- the
        # `_apple_frag_layout` that `MmaOpApple._store_fragment` writes (KB
        # `exceptions/apple-mma-fragment-is-not-distribute-expressible`),
        # reused here via `frag.slice[4]` halves so the intra-fragment scatter
        # is NOT re-derived. Mirrors the dense cousin's cast epilogue
        # (`AppleM5MatMul._run_gemm_body` `_write4`). fp32 out casts to itself
        # (identity), so this is byte-identical for every `c_type`.
        #
        # fp32 does NOT route through `mma_op.store`/`store_bounded` (which
        # write fp32 directly): calling them from this `c_type`-generic body
        # needs a base-pointer rebind to fp32 (as the dense `_fast_path_store`
        # does), which the TileTensor-idioms-only rule forbids here. `_write4`
        # reaches the same 2-vector-store result.
        @always_inline
        @__parameter
        def _apply_epilogue[bounded: Bool]():
            var c_sub = c.tile[SG_M, SG_N](Int(sg_row_idx), Int(sg_col_idx))
            # 4 contiguous out cols = one SIMD unit; element alignment only (the
            # row stride N is odd for odd N -- a full-vector alignment faults).
            comptime elem_align = align_of[Scalar[Self.c_type]]()
            var c_vec = c_sub.vectorize[1, 4]()

            @always_inline
            @__parameter
            def _write4(lrow: Int, lcol: Int, acol: Int, v: SIMD[.float32, 4]):
                var y = v.cast[Self.c_type]()
                comptime if bounded:
                    if acol + 3 < N:
                        c_vec.store[alignment=elem_align](
                            Coord(lrow, lcol // 4), y
                        )
                    else:
                        for e in range(min(4, N - acol)):
                            c_sub.store[width=1, alignment=elem_align](
                                Coord(lrow, lcol + e),
                                SIMD[Self.c_type, 1](y[e]),
                            )
                else:
                    c_vec.store[alignment=elem_align](Coord(lrow, lcol // 4), y)

            comptime for mi in range(Self.NUM_MMA_M):
                comptime for ni in range(Self.NUM_MMA_N):
                    var frag = accum[mi * Self.NUM_MMA_N + ni]
                    var lrow = mi * 16 + Int(mma_op.rb)
                    var lcol = ni * 16 + Int(mma_op.cb)
                    var acol = col_base + lcol
                    comptime if bounded:
                        var arow = row_base + lrow
                        if arow < M:
                            _write4(lrow, lcol, acol, frag.slice[4, offset=0]())
                        if arow + 8 < M:
                            _write4(
                                lrow + 8, lcol, acol, frag.slice[4, offset=4]()
                            )
                    else:
                        _write4(lrow, lcol, acol, frag.slice[4, offset=0]())
                        _write4(lrow + 8, lcol, acol, frag.slice[4, offset=4]())

        if is_edge_tile:
            _apply_epilogue[bounded=True]()
        else:
            _apply_epilogue[bounded=False]()

    @__name(t"apple_grouped_matmul2d_fp8_run_{Self.c_type}")
    @staticmethod
    def run_grouped[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        ao_layout: TensorLayout,
        ei_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
        ao_engine: TensorEngine,
        ei_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.in_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        a_offsets: TileTensor[
            mut=False, .uint32, ao_layout, MutAnyOrigin, Engine=ao_engine
        ],
        expert_ids: TileTensor[
            mut=False, .int32, ei_layout, MutAnyOrigin, Engine=ei_engine
        ],
        N_arg: Int32,
        K_arg: Int32,
    ):
        """Grouped (MoE) W8A16 kernel entry: one expert group per `block_idx.z`.

        Computes, for group `z = block_idx.z`,
        `C[a_offsets[z]:a_offsets[z+1], :] =
             A[a_offsets[z]:a_offsets[z+1], :] @ b[expert_ids[z], :, :]^T`
        i.e. the FP8 analog of `naive_grouped_matmul_kernel` but on the tiled
        simdgroup MMA. `c` is the ragged output `(total_M, N)`, `a` the ragged
        bf16 activation `(total_M, K)`, `b` the FP8 weight stack
        `(num_experts, N, K)`. Produces the RAW (unscaled) matmul; the per-expert
        `weight_scale` is folded post-matmul by the graph lowering, exactly as on
        the dense Apple FP8 Linear (a per-tensor scalar factors out of the sum).

        Grid `(ceil(N/BN), ceil(max_tokens_per_expert/BM), num_active_experts)`.
        Per-group ragged M is `a_offsets[z+1] - a_offsets[z]`; empty groups
        (M == 0) and fully-OOB M-tiles early-return in `_gemm_body`. An inactive
        LoRA group (`expert_ids[z] == -1`) writes zeros for its rows.

        The per-group base views use a base-pointer offset (`a.ptr + a_start*K`,
        `b.ptr + expert*N*K`) -- the ragged-group addressing idiom shared by every
        grouped matmul kernel in the tree (`naive_grouped_matmul_kernel`,
        `grouped_matmul_amd_kernel_launcher`); TileTensor has no runtime-offset
        ragged-slice primitive that keeps the inner `(., K)` dims static. All
        element access inside `_gemm_body` is TileTensor-indexed (MMA fragment
        loaders + guarded `c.store`).
        """
        var N = Int(N_arg)
        var K = Int(K_arg)
        var z = Int(block_idx.z)
        var a_start = Int(a_offsets[z])
        var M = Int(a_offsets[z + 1]) - a_start
        var expert = Int(expert_ids[z])
        var active = expert != -1
        # Clamp the weight-slab index so the (never-dereferenced when inactive)
        # base view is not built from a negative offset.
        var w_expert = max(expert, 0)

        var c_group = TileTensor(
            c.ptr + Int64(a_start) * Int64(N),
            Layout(Coord(M, N), Coord(N, Idx[1])),
        )
        var a_group = TileTensor(
            a.ptr + Int64(a_start) * Int64(K),
            Layout(Coord(M, K), Coord(K, Idx[1])),
        )
        var w_group = TileTensor(
            b.ptr + Int64(w_expert) * Int64(N) * Int64(K),
            Layout(Coord(N, K), Coord(K, Idx[1])),
        )
        Self._gemm_body(c_group, a_group, w_group, M, N, K, active)


@always_inline
def enqueue_matmul2d_fp8[
    c_type: DType = .float32,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    weight: TileTensor[.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Enqueue the tiled W8A16 GEMM: `out = a @ W_fp8^T` (raw, unscaled).

    `a` is the bf16 activation `(M, K)`, `weight` the FP8-E4M3 weight `(N, K)`
    (`transpose_b`). C is `(M, N)`. bf16 A x fp8 B -> fp32 on the native Apple MMA,
    direct DRAM B feed (no SMEM). Any M/N/K (bounded MMA + guarded store). The
    per-tensor scalar `weight_scale` folds post-matmul (graph lowering).

    This is the shared `AppleM5MatMul` simdgroup-tiled GEMM body specialized on
    `b_type=float8_e4m3fn` (the weight-loader seam swap) -- the SAME body the
    dense bf16 path uses, so FP8 W8A16 inherits its Morton rect tile schedule and
    epilogue with only the B operand widened. Numerically identical per output to
    the standalone kernel it replaces (Morton reorders WHICH threadgroup computes
    a tile, not the per-tile K accumulation). Single-pass launch: fp8 is not
    wired to split-K (its partial kernels are the bf16 family).

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_matmul2d_fp8: c_type must be one of {fp16, bf16, fp32}"

    comptime MM = AppleM5MatMul[
        DType.bfloat16,
        c_type,
        transpose_b=True,
        elementwise_lambda_fn=elementwise_lambda_fn,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        sg_m=sg_m,
        sg_n=sg_n,
        b_type=DType.float8_e4m3fn,
    ]

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    var k = Int(a.dim[1]())

    debug_assert(Int(a.dim[0]()) == m, "A shape (M, K) must match C's M")
    debug_assert(Int(weight.dim[0]()) == n, "weight must be (N, K)")
    debug_assert(Int(weight.dim[1]()) == k, "weight must be (N, K)")
    # MmaOpApple narrows the B row stride to UInt16; for transpose_b=True the
    # B-slab stride is K (col-major (K, N) view), so K must fit.
    debug_assert(
        k <= 65535, "Apple FP8 matmul: K must fit in UInt16; got K=", k
    )

    var grid_m = ceildiv(m, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    # Per-axis next-pow2 grid for the rectangular Z-order (mirrors
    # `enqueue_apple_matmul`; the standalone fp8 kernel used a raw `(grid_n,
    # grid_m)` grid with no Morton -- the unification restores the schedule).
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

    comptime kernel = MM.run[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(weight).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(weight).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        weight.as_immut(),
        log2_m,
        log2_n,
        grid_dim=(side_m * side_n),
        block_dim=(MM.THREADS_PER_BLOCK),
    )


@always_inline
def enqueue_grouped_matmul2d_fp8[
    c_type: DType = .float32,
    block_m: Int = 64,
    block_n: Int = 64,
    block_k: Int = 16,
    sg_m: Int = 32,
    sg_n: Int = 32,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[.bfloat16, ...],
    b: TileTensor[.float8_e4m3fn, ...],
    a_offsets: TileTensor[mut=False, .uint32, ...],
    expert_ids: TileTensor[mut=False, .int32, ...],
    max_num_tokens_per_expert: Int,
    num_active_experts: Int,
    ctx: DeviceContext,
) raises:
    """Enqueue the tiled grouped (MoE) W8A16 GEMM (`transpose_b`), raw/unscaled.

    The FP8 analog of `naive_grouped_matmul` on the tiled simdgroup MMA: ONE
    kernel dispatch handles all expert groups (grid `block_idx.z` selects the
    group), so there is no host-side per-expert launch loop. `a` is the ragged
    bf16 activation `(total_M, K)`, `b` the FP8-E4M3 weight stack
    `(num_experts, N, K)`, `c` the ragged output `(total_M, N)`. Each group `z`
    reads its expert weight slab `b[expert_ids[z]]` and its token rows
    `[a_offsets[z], a_offsets[z+1])`. bf16 A x fp8 B -> fp32 on the native Apple
    MMA, direct DRAM B feed (no SMEM). The per-expert `weight_scale` folds
    post-matmul (graph lowering), matching the dense Apple FP8 Linear.

    Raises:
        If the attached GPU is not Apple M5 (`compute_capability == 5`).
    """
    _require_apple_m5(ctx)

    comptime assert (
        c_type == .float16 or c_type == .bfloat16 or c_type == .float32
    ), "enqueue_grouped_matmul2d_fp8: c_type must be one of {fp16, bf16, fp32}"

    comptime MM = Matmul2dFp8[
        c_type,
        None,
        block_m,
        block_n,
        block_k,
        sg_m,
        sg_n,
    ]

    var n = Int(b.dim[1]())
    var k = Int(b.dim[2]())

    debug_assert(Int(a.dim[1]()) == k, "A (total_M, K) must match weight K")
    debug_assert(Int(c.dim[1]()) == n, "C (total_M, N) must match weight N")

    if num_active_experts == 0 or max_num_tokens_per_expert == 0:
        return

    var grid_m = ceildiv(max_num_tokens_per_expert, MM.BM)
    var grid_n = ceildiv(n, MM.BN)

    comptime kernel = MM.run_grouped[
        type_of(c).LayoutType,
        type_of(a).LayoutType,
        type_of(b).LayoutType,
        type_of(a_offsets).LayoutType,
        type_of(expert_ids).LayoutType,
        type_of(c).Engine,
        type_of(a).Engine,
        type_of(b).Engine,
        type_of(a_offsets).Engine,
        type_of(expert_ids).Engine,
    ]
    ctx.enqueue_function[kernel](
        c,
        a.as_immut(),
        b.as_immut(),
        a_offsets,
        expert_ids,
        Int32(n),
        Int32(k),
        grid_dim=(grid_n, grid_m, num_active_experts),
        block_dim=(MM.THREADS_PER_BLOCK),
    )
