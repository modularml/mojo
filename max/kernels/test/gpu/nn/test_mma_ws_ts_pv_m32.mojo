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

"""TS-batched P@V at MMA_M=32 (m_pack=4) with NATURAL mn-major V.

Purpose (de-risks Steps 3/5/6 of the SM100 FA4 `.ws` MMA_M=32 warp-split design;
see docs/plans/sm100-fa4-ws-mma32-warp-splitk.md)
=============================================================================
Prove the load-bearing mechanism of that design -- the warp-specialized
*TS-batched* `tcgen05.mma.ws.cta_group::1` at `MMA_M=32` (`m_pack = 128//32 = 4`)
as a 4-way intra-CTA KEY split of `P@V` -- but fed with V in the **real kernel's
natural, mn-major, depth-column-staged** layout instead of a pre-shuffled k-major
`transpose_b=True` tile. This matches how `fa4_mma` builds its V B-descriptor
(mma_warp.mojo:242-249: `is_k_major=False`, `transpose_b=False`), so the layout
proven here is the one the kernel will actually use.

Physical realization (the crux):
  * Each of the 4 warps of the (single) warpgroup independently writes its OWN
    `P_g` matrix (32 rows x 64 keys) into its datapath subpartition via
    `tcgen05_st` (all 4 warps store to the SAME TMEM column address; the
    hardware routes warp `g`'s store to subpartition `g` -- no per-warp column
    offset). These 4 independently-written matrices ARE the 4 key-splits.
  * V lives in global memory in the NATURAL `[BN_KEYS=256 keys, depth]` layout
    (keys x depth). Per depth-tile `t`, the four `[64 keys, 64 depth]` natural
    blocks `V[64g:64g+64, t*64:t*64+64]` are TMA'd into the four mn-major N-bands
    of ONE SMEM V region: band `g` (SMEM cols `[g*64, g*64+64)`) holds
    `V[64g+k, t*64+d]` at `(k, d)`. No pre-shuffle: the banding is done purely by
    the per-quarter `async_copy` global coords.
  * `SM100TensorAccumulator[..., transpose_b=False, a_tmem=True]` issues the
    batched TS MMA (A = the packed P in TMEM, B = the mn-major V region,
    MMA_M=32, MMA_N = m_pack*DEPTH_TILE = 256), contracting each `P_g` against
    band `g`'s V block -> an independent partial `O_g` in output-column band `g`.
    `C[m, g*64+d] = sum_k P_g[m,k] * V[64g+k, t*64+d] = O_g[m, t*64+d]`.
  * At `num_stages=2` (depth=128, tile 0) the accumulator's `use_3_then_1_split`
    fires -- the P contraction is staged 3/4 (k-blocks [0,3), 48 keys/quarter)
    then 1/4 (k-block [3,4), 16 keys/quarter). This is the first exercise of
    `use_3_then_1_split` at `m_pack=4`.

Two V-load arms (SAME [256,64] descriptor + MMA + readback):
  * BANDED (control, `natural_load=False`): the per-quarter load described above --
    four [64,64] TMAs banded into each [256,64] region.
  * NATURAL (`natural_load=True`): ONE whole `[256 keys, 64 depth]` TMA per depth-tile
    into an `mn_major[depth=64, keys=256]` region -- all 256 keys land on the k axis
    with NO per-quarter re-banding. This is the kernel's real V layout
    (`kv_desc_v` BMN=depth/BK=keys, mn_major). Both arms are byte-identical (the whole
    `[256,64]` region == the 4 banded quarters), proven at the layout-algebra level by
    `test_ws_v_layout_probe.mojo` (host, no GPU). Green here confirms on B200 that the
    kernel's natural V load feeds the packed WS P@V directly -- so warp-specialized
    BM=32 needs only a descriptor BMN/BK change, NOT a load re-banding.

Verification:
  * band g == P_g @ V_g   (per-partition partial -- proves batched semantics +
                           the mn-major V banding)
  * sum_g band g == P @ V (the full attention output -- the key-split sums)

Shapes: MMA_M=32, BN=256 keys (64/partition), bf16, MMA_K=16, num_k_mmas =
64/16 = 4. A batched MMA has MMA_N = M_PACK*DEPTH_TILE, and F16 caps MMA_N at
256, so DEPTH_TILE = 256/M_PACK = 64 is the max depth per MMA. Both cases run:
  * depth=64  -> 1 depth-tile, single-shot MMA (num_stages=1 regression).
  * depth=128 -> 2 depth-tiles into two C-TMEM regions from ONE packed P:
                 tile 0 staged (num_stages=2, `use_3_then_1_split`), tile 1
                 single-shot -- the BN=256/depth=128 target regime.
"""

from std.memory import bitcast
from std.random import randn, seed
from std.sys import size_of

from max.gpu import thread_idx, warp_id as get_warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.info import _is_sm10x_gpu
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import external_memory
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAKind,
    mma_arrive,
)
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_alloc,
    tcgen05_dealloc,
    tcgen05_fence_after,
    tcgen05_fence_before,
    tcgen05_ld,
    tcgen05_load_wait,
    tcgen05_release_allocation_lock,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.tensor_core_async import tile_layout_mn_major
from layout.tma_async import (
    SharedMemBarrier,
    TMATensorTile,
    create_tensor_tile,
)
from linalg.arch.sm100.mma import smem_descriptor
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    elect,
)
from std.testing import assert_true
from std.utils.index import Index, IndexList

# ---------------------------------------------------------------------------
# Compile-time constants
# ---------------------------------------------------------------------------
comptime OP_TYPE = DType.bfloat16
comptime ACC_TYPE = DType.float32

comptime MMA_M = 32
comptime M_PACK = 128 // MMA_M  # 4
comptime BN_KEYS = 256  # total keys across the 4 partitions
comptime PART_KEYS = BN_KEYS // M_PACK  # 64 keys per partition (= A operand BK)
# DEPTH_TILE = max output depth per batched MMA: MMA_N = M_PACK*DEPTH_TILE must
# be <= 256 (F16 hardware max N), so DEPTH_TILE = 256//M_PACK = 64. A full
# `depth` output is emitted as `depth // DEPTH_TILE` depth-tiled MMAs (see the
# per-case `test_pv_ts_batched[depth]`: depth=64 -> 1 tile, depth=128 -> 2).
comptime DEPTH_TILE = 64
comptime MMA_N = M_PACK * DEPTH_TILE  # 256 (== F16 hardware max N), per tile
comptime MMA_K = 16  # bf16 hardware MMA_K
comptime NUM_K_MMAS = PART_KEYS // MMA_K  # 4

comptime SWIZZLE = TensorMapSwizzle.SWIZZLE_128B
comptime NUM_THREADS = 128  # one warpgroup = 4 warps
comptime MAX_TMEM_COLS: UInt32 = 512

# Packed store: each thread packs its 64 bf16 keys into 32 u32 words.
comptime P_FRAG_U32 = PART_KEYS // 2  # 32
# Packed readback: warp g reads its per-tile band = DEPTH_TILE f32 columns.
comptime COLS_PER_WARP = MMA_N // M_PACK  # 64 == DEPTH_TILE

# TMEM layout: A (P) at column 0 (P_FRAG_U32=32 cols); C (O) tiles start at
# column 128, tile t at C_TMEM_OFFSET + t*COLS_PER_WARP (2 tiles -> [128,256)).
comptime A_TMEM_OFFSET: UInt32 = 0
comptime C_TMEM_OFFSET: UInt32 = 128

# One depth-tile's V region in SMEM is the mn-major B-tile [BMN=MMA_N=256,
# BK=PART_KEYS=64] (`tile_layout_mn_major[bf16, 256, 64, SWIZZLE_128B]`), i.e.
# 256*64 elements. Its four N-bands (each `tile_layout_mn_major[bf16, 64, 64]`)
# are byte-identical, standalone 64x64 mn-major tiles laid contiguously at
# element offset `g * DEPTH_TILE * PART_KEYS` (band g). Verified by the layout
# algebra: `tile_layout_mn_major[bf16,256,64](64g, 0) == 4096*g` and the band
# sub-block matches `tile_layout_mn_major[bf16,64,64]` bit-for-bit -- so each
# quarter's natural `[64,64]` block TMAs cleanly into its band.
comptime V_REGION_ELEMS = MMA_N * PART_KEYS  # 16384 (one depth-tile region)
comptime BAND_ELEMS = DEPTH_TILE * PART_KEYS  # 4096 (one quarter band)
comptime V_TILE_BYTES = V_REGION_ELEMS * size_of[OP_TYPE]()  # 32 KiB per tile
comptime META_BYTES = 128  # tmem_addr + per-tile v_mbar + mma_mbar, padded

comptime ATOL: Float32 = 1.0
comptime RTOL: Float32 = 0.02


# ---------------------------------------------------------------------------
# GPU kernel
# ---------------------------------------------------------------------------
@__llvm_arg_metadata(v_tma_op, `nvvm.grid_constant`)
def pv_ts_batched_kernel[
    depth: Int,
    natural_load: Bool,
    v_tile_rank: Int,
    v_tile_shape: IndexList[v_tile_rank],
    v_desc_shape: IndexList[v_tile_rank],
](
    v_tma_op: TMATensorTile[OP_TYPE, v_tile_rank, v_tile_shape, v_desc_shape],
    p_input: LayoutTensor[
        OP_TYPE, Layout.row_major(MMA_M, BN_KEYS), MutAnyOrigin
    ],
    o_output: LayoutTensor[
        ACC_TYPE, Layout.row_major(MMA_M, M_PACK * depth), MutAnyOrigin
    ],
):
    """4 warps write 4 distinct P_g into TMEM; `depth//DEPTH_TILE` depth-tiled TS
    .ws MMAs (shared packed P) over mn-major V -> 4 O_g bands.

    Two V-load arms, SAME `[256,64]` descriptor + MMA + readback:
      * `natural_load=False` (control): per depth-tile, 4 per-quarter `[64,64]` TMAs
        banded into a `[256,64]` region (band g at g*BAND_ELEMS).
      * `natural_load=True`: per depth-tile, ONE whole `[256 keys, 64 depth]` TMA into
        an mn_major[depth=64, keys=256] region -- all 256 keys land on the k axis with
        NO per-quarter re-banding. Proves the kernel's natural V load feeds the packed
        `[256,64]` batched read (host probe `test_ws_v_layout_probe.mojo`)."""

    comptime num_d_tiles = depth // DEPTH_TILE
    comptime v_bytes = num_d_tiles * V_TILE_BYTES

    # SMEM V region t is the mn-major B-tile; band g of it is a standalone
    # 64x64 mn-major tile at element offset g * BAND_ELEMS.
    comptime v_band_layout = tile_layout_mn_major[
        OP_TYPE, DEPTH_TILE, PART_KEYS, swizzle_mode=SWIZZLE
    ]()
    # Whole-region natural chunk: one [256 keys, 64 depth] block as
    # mn_major[mn=depth=64, k=keys=256] (the kernel's kv_desc_v orientation).
    comptime nat_chunk_layout = tile_layout_mn_major[
        OP_TYPE, DEPTH_TILE, BN_KEYS, swizzle_mode=SWIZZLE
    ]()

    # P@V accumulators (mn-major V, transpose_b=False -- mirrors UMMA1Type in
    # mma_warp.mojo). `use_ws` fires (cta_group=1, MMA_M<=64). PVAcc2 has
    # num_stages=2 so `use_3_then_1_split` fires (a_tmem, num_stages==2,
    # num_k_blocks=4 % 4 == 0) -- the 3/4+1/4 P-contraction staging.
    comptime PVAcc2 = SM100TensorAccumulator[
        OP_TYPE,
        ACC_TYPE,
        MMA_M,
        MMA_N,
        PART_KEYS,
        a_tmem=True,
        num_stages=2,
        transpose_b=False,
        swizzle_b=SWIZZLE,
        mma_kind=UMMAKind.KIND_F16,
    ]
    comptime PVAcc1 = SM100TensorAccumulator[
        OP_TYPE,
        ACC_TYPE,
        MMA_M,
        MMA_N,
        PART_KEYS,
        a_tmem=True,
        num_stages=1,
        transpose_b=False,
        swizzle_b=SWIZZLE,
        mma_kind=UMMAKind.KIND_F16,
    ]

    # ---- Dynamic SMEM: `num_d_tiles` mn-major V regions + metadata ----
    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var v_smem_ptr = smem_base.bitcast[Scalar[OP_TYPE]]()

    var metadata_ptr = (smem_base + v_bytes).bitcast[UInt32]()
    var ptr_tmem_addr = metadata_ptr
    # Per-tile TMA-done barriers (v_mbar[t]) and per-tile MMA-done barriers
    # (mma_mbar[t]). Fixed slots for up to 2 depth-tiles (8 B / barrier).
    var v_mbar = (metadata_ptr + 2).bitcast[SharedMemBarrier]()
    var mma_mbar = (metadata_ptr + 6).bitcast[SharedMemBarrier]()

    var tid = thread_idx.x
    var wid = get_warp_id()
    var row = Int(tid & 31)  # 0..31 : output/query row
    var g = Int(tid >> 5)  # 0..3  : datapath quarter == key partition
    var elect_one_thread = tid == 0

    if elect_one_thread:
        for t in range(num_d_tiles):
            v_mbar[t].init()
            mma_mbar[t].init()

    if wid == 0:
        tcgen05_alloc[1](ptr_tmem_addr, MAX_TMEM_COLS)
    barrier()

    var tmem_addr = ptr_tmem_addr[0]
    var a_tmem = tmem_addr + A_TMEM_OFFSET
    var c_tmem = tmem_addr + C_TMEM_OFFSET

    # ---- TMA load: for each depth-tile, place the four natural [64,64] V blocks
    # into the four mn-major N-bands of region t. Both V loads are issued up
    # front so tile 1's load overlaps tile 0's compute (the design's early load).
    # global V is [BN_KEYS=256 keys (rows), depth (cols)] row-major; the box is
    # [PART_KEYS keys, DEPTH_TILE depth]. TMA coords are (col, row) = fast-dim
    # first (Kernels KB known-limitations/sm100-tma-coord-order-col-row), so the
    # depth (col) offset t*DEPTH_TILE goes in coords[0] and the key (row) offset
    # g*PART_KEYS in coords[1]. ----
    if elect_one_thread:
        for t in range(num_d_tiles):
            v_mbar[t].expect_bytes(Int32(V_TILE_BYTES))
            comptime if natural_load:
                # ONE whole [256 keys, 64 depth] TMA -> mn_major[depth=64, keys=256]
                # region t. All 256 keys land on the k axis (no per-quarter banding);
                # the [256,64] descriptor reads them banded via the byte coincidence.
                var chunk_tile = LayoutTensor[
                    OP_TYPE,
                    nat_chunk_layout,
                    address_space=.SHARED,
                    alignment=128,
                ](v_smem_ptr + t * V_REGION_ELEMS)
                v_tma_op.async_copy(
                    chunk_tile,
                    v_mbar[t],
                    (t * DEPTH_TILE, 0),
                )
            else:
                for gq in range(M_PACK):
                    var band_tile = LayoutTensor[
                        OP_TYPE,
                        v_band_layout,
                        address_space=.SHARED,
                        alignment=128,
                    ](v_smem_ptr + t * V_REGION_ELEMS + gq * BAND_ELEMS)
                    v_tma_op.async_copy(
                        band_tile,
                        v_mbar[t],
                        (t * DEPTH_TILE, gq * PART_KEYS),
                    )
    barrier()

    # ---- Write this warp's P_g into its TMEM subpartition ----
    # Warp g owns keys-chunk g: P_g[row, :] = p_input[row, g*PART_KEYS : +64].
    # All 4 warps store to the SAME address a_tmem (datapaths=32); the hardware
    # subpartition routes warp g's store to quarter g (no per-warp offset). This
    # single packed P feeds ALL depth-tile MMAs (P is depth-independent).
    def frag_at(j: Int) {imm} -> UInt32:
        var pair = SIMD[OP_TYPE, 2]()
        pair[0] = p_input[row, g * PART_KEYS + 2 * j][0]
        pair[1] = p_input[row, g * PART_KEYS + 2 * j + 1][0]
        return bitcast[.uint32, 1](pair)

    var frag = Array[_, P_FRAG_U32](fill_with=frag_at)

    tcgen05_st[datapaths=32, bits=32, repeat=P_FRAG_U32, pack=False](
        a_tmem, frag
    )
    tcgen05_store_wait()
    tcgen05_fence_before()
    barrier()

    # ---- Issue one TS .ws MMA per depth-tile from the elected lane of warp 0 ---
    # Tile t: B = mn-major V region t (base descriptor; the accumulator adds the
    # per-stage k_offset internally), C = c_tmem + t*COLS_PER_WARP (its own TMEM
    # region so all tiles coexist for readback), completion on mma_mbar[t].
    var e: Int32 = 0
    if wid == 0:
        e = elect()

    comptime if num_d_tiles == 1:
        # depth=64 regression: single depth-tile, single-shot full contraction.
        var b0 = smem_descriptor[
            BMN=MMA_N,
            BK=PART_KEYS,
            swizzle_mode=SWIZZLE,
            is_k_major=False,
        ](v_smem_ptr)
        v_mbar[0].wait()
        PVAcc1.mma[stage_idx=0](a_tmem, b0, c_tmem, elect=e, c_scale=UInt32(0))
        if elect_one_thread:
            mma_arrive(mma_mbar + 0)
    else:
        # depth=128 target regime. Tile 0: staged 3/4 + 1/4 contraction
        # (`use_3_then_1_split` at m_pack=4). Tile 1: single-shot. Both from the
        # SAME packed P; only the V region base and C-TMEM region differ.
        var b0 = smem_descriptor[
            BMN=MMA_N,
            BK=PART_KEYS,
            swizzle_mode=SWIZZLE,
            is_k_major=False,
        ](v_smem_ptr)
        v_mbar[0].wait()
        # stage 0 overwrites (c_scale=0): k-blocks [0,3) = first 3/4 (48 keys/qtr).
        PVAcc2.mma[stage_idx=0](a_tmem, b0, c_tmem, elect=e, c_scale=UInt32(0))
        # stage 1 accumulates (scale forced to 1 internally): k-block [3,4) = 1/4.
        PVAcc2.mma[stage_idx=1](a_tmem, b0, c_tmem, elect=e, c_scale=UInt32(1))
        if elect_one_thread:
            mma_arrive(mma_mbar + 0)

        var b1 = smem_descriptor[
            BMN=MMA_N,
            BK=PART_KEYS,
            swizzle_mode=SWIZZLE,
            is_k_major=False,
        ](v_smem_ptr + V_REGION_ELEMS)
        v_mbar[1].wait()
        PVAcc1.mma[stage_idx=0](
            a_tmem,
            b1,
            c_tmem + UInt32(COLS_PER_WARP),
            elect=e,
            c_scale=UInt32(0),
        )
        if elect_one_thread:
            mma_arrive(mma_mbar + 1)

    for t in range(num_d_tiles):
        mma_mbar[t].wait(0)
    tcgen05_fence_after()

    # ---- Read back each tile's band g (warp g owns per-tile cols
    # [g*DEPTH_TILE, +DEPTH_TILE); tile t -> output depth [t*DEPTH_TILE, ..)) ----
    for t in range(num_d_tiles):
        var c_frag = tcgen05_ld[
            datapaths=32,
            bits=32,
            repeat=COLS_PER_WARP,
            dtype=ACC_TYPE,
            pack=False,
            width=COLS_PER_WARP,
        ](c_tmem + UInt32(t) * UInt32(COLS_PER_WARP))
        tcgen05_load_wait()
        var col_base = g * depth + t * DEPTH_TILE
        for j in range(COLS_PER_WARP):
            o_output[row, col_base + j] = c_frag[j]

    if wid == 0:
        tcgen05_release_allocation_lock[1]()
        tcgen05_dealloc[1](tmem_addr, MAX_TMEM_COLS)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_pv_ts_batched[
    depth: Int, natural_load: Bool = False
](ctx: DeviceContext) raises:
    comptime num_d_tiles = depth // DEPTH_TILE
    comptime out_cols = M_PACK * depth  # full output width (band g == depth wide)
    comptime v_bytes = num_d_tiles * V_TILE_BYTES
    comptime total_smem = v_bytes + META_BYTES

    print("=" * 70)
    print(
        "test_pv_ts_batched: tcgen05.mma.ws.cta_group::1.kind::f16 TS-batched"
        " M=32 (m_pack=4) BN=256 keys mn-major V ["
        + (
            "natural whole-tile load" if natural_load else "banded per-quarter load"
        )
        + "] depth="
        + String(depth)
        + " ("
        + String(num_d_tiles)
        + " depth-tile(s), MMA_N=256 each)"
    )
    print("=" * 70)

    seed(42)

    # ---- Inputs (host) ----
    # P: [32, 256] bf16. Partition g == columns [g*64, g*64+64).
    var p_inp = ManagedLayoutTensor[OP_TYPE, Layout.row_major(MMA_M, BN_KEYS)](
        ctx
    )
    # V: NATURAL [BN_KEYS=256 keys, depth] bf16 (keys x depth), NOT pre-shuffled.
    # Quarter g reads keys [g*PART_KEYS, +PART_KEYS); depth d is read directly.
    var v_inp = ManagedLayoutTensor[OP_TYPE, Layout.row_major(BN_KEYS, depth)](
        ctx
    )

    var p_host = p_inp.tensor[update=False]()
    var v_host = v_inp.tensor[update=False]()
    randn[OP_TYPE](p_host.ptr, MMA_M * BN_KEYS)
    randn[OP_TYPE](v_host.ptr, BN_KEYS * depth)

    var o_out_buf = ManagedLayoutTensor[
        ACC_TYPE, Layout.row_major(MMA_M, out_cols)
    ](ctx)

    # ---- V TMA descriptor. Banded: box = one quarter block [PART_KEYS keys,
    # DEPTH_TILE depth] (kernel lands each (quarter g, depth-tile t) sub-block in
    # band g). Natural: box = whole [BN_KEYS keys, DEPTH_TILE depth] (kernel lands
    # each depth-tile region in one TMA, all keys on the k axis). ----
    comptime v_box = Index(BN_KEYS, DEPTH_TILE) if natural_load else Index(
        PART_KEYS, DEPTH_TILE
    )
    var v_tma_op = create_tensor_tile[
        v_box,
        swizzle_mode=SWIZZLE,
    ](ctx, v_inp.device_tensor())

    # ---- Launch ----
    comptime kernel = pv_ts_batched_kernel[
        depth,
        natural_load,
        type_of(v_tma_op).rank,
        type_of(v_tma_op).tile_shape,
        type_of(v_tma_op).desc_shape,
    ]
    ctx.enqueue_function[kernel](
        v_tma_op,
        p_inp.device_tensor(),
        o_out_buf.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(NUM_THREADS),
        shared_mem_bytes=total_smem,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(total_smem)
        ),
    )
    ctx.synchronize()

    # ---- Host reference ----
    # band g (output cols [g*depth, +depth)) should equal O_g:
    #   O_g[m, dd] = sum_{k<PART_KEYS} P[m, g*PART_KEYS + k] * V[g*PART_KEYS+k, dd]
    # read straight from the natural V tensor; sum_g O_g == full attention P @ V.
    var o_out_host = o_out_buf.tensor()
    var o_out_ptr = o_out_host.ptr

    print(
        "  O_gpu[0,0]="
        + String(o_out_ptr[0])
        + "  O_gpu[31,"
        + String(out_cols - 1)
        + "]="
        + String(o_out_ptr[31 * out_cols + (out_cols - 1)])
    )

    var max_abs_err: Float32 = 0.0
    var max_rel_err: Float32 = 0.0
    var num_failures: Int = 0
    var worst_g: Int = -1
    var worst_m: Int = -1
    var worst_d: Int = -1

    for gg in range(M_PACK):
        for m in range(MMA_M):
            for dd in range(depth):
                var acc: Float32 = 0.0
                for k in range(PART_KEYS):
                    var pv = p_host[m, gg * PART_KEYS + k][0].cast[ACC_TYPE]()
                    var vv = v_host[gg * PART_KEYS + k, dd][0].cast[ACC_TYPE]()
                    acc += pv * vv
                var gpu_val = o_out_ptr[m * out_cols + (gg * depth + dd)]
                var abs_err = abs(gpu_val - acc)
                var rel_err = abs_err / max(abs(acc), Float32(1.0))
                if abs_err > max_abs_err:
                    max_abs_err = abs_err
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                if abs_err > ATOL and rel_err > RTOL:
                    num_failures += 1
                    if worst_g < 0:
                        worst_g = gg
                        worst_m = m
                        worst_d = dd

    print("  [per-band] max abs err: " + String(max_abs_err))
    print("  [per-band] max rel err: " + String(max_rel_err))
    print(
        "  [per-band] failures (atol="
        + String(ATOL)
        + " rtol="
        + String(RTOL)
        + "): "
        + String(num_failures)
        + " / "
        + String(M_PACK * MMA_M * depth)
    )
    if num_failures > 0:
        print(
            "  first failure at (g="
            + String(worst_g)
            + ", m="
            + String(worst_m)
            + ", d="
            + String(worst_d)
            + ")"
        )

    # ---- Full-output (sum over partitions) sanity ----
    var sum_max_rel: Float32 = 0.0
    for m in range(MMA_M):
        for dd in range(depth):
            var ref_sum: Float32 = 0.0
            var gpu_sum: Float32 = 0.0
            for gg in range(M_PACK):
                gpu_sum += o_out_ptr[m * out_cols + (gg * depth + dd)]
                for k in range(PART_KEYS):
                    ref_sum += (
                        p_host[m, gg * PART_KEYS + k][0].cast[ACC_TYPE]()
                        * v_host[gg * PART_KEYS + k, dd][0].cast[ACC_TYPE]()
                    )
            var rel = abs(gpu_sum - ref_sum) / max(abs(ref_sum), Float32(1.0))
            if rel > sum_max_rel:
                sum_max_rel = rel
    print("  [sum P@V] max rel err: " + String(sum_max_rel))

    assert_true(
        num_failures == 0,
        msg=String(
            "TS-batched P@V FAILED (depth=",
            depth,
            "): ",
            num_failures,
            " band elements exceed tolerance (max abs=",
            max_abs_err,
            ", max rel=",
            max_rel_err,
            ")",
        ),
    )
    print("  TS-BATCHED P@V PASSED (band g == P_g @ V_g for all g)")

    _ = p_inp^
    _ = v_inp^
    _ = o_out_buf^


def main() raises:
    with DeviceContext() as ctx:
        comptime if not _is_sm10x_gpu(ctx.default_device_info):
            print("Skipping: this test requires B200 (SM100)")
            return
        # ---- Banded per-quarter load (control) ----
        # depth=64: single batched MMA (num_stages=1 regression of the proven case).
        test_pv_ts_batched[64](ctx)
        # depth=128: two depth-tiled MMAs (MMA_N=256 each) into two C-TMEM
        # regions from one packed P -- tile 0 staged (use_3_then_1_split), tile 1
        # single-shot -- the actual BN=256/depth=128 target regime.
        test_pv_ts_batched[128](ctx)

        # ---- Natural whole-tile load (the kernel's mn_major[depth,keys] layout) ----
        # Same [256,64] descriptor + MMA, but V is loaded WITHOUT per-quarter
        # re-banding: one [256 keys, 64 depth] TMA per depth-tile, all keys on the k
        # axis. Green here == the kernel's natural V load feeds the packed WS P@V
        # (host layout proof: test_ws_v_layout_probe.mojo). So WS BM=32 needs only a
        # descriptor BMN/BK change, not a load re-banding.
        test_pv_ts_batched[64, natural_load=True](ctx)
        test_pv_ts_batched[128, natural_load=True](ctx)
        print("\nTS-batched P@V (MMA_M=32, m_pack=4, mn-major V) test PASSED.")
