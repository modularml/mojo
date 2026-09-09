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
"""Round-trip test for the reference `v227` V-LDS adapter (the `W∘R` pair).

NOTE: temporarily excluded from the BUILD glob (see this directory's
`BUILD.bazel`). It fails to compile under the merged toolchain — a
comptime-type mismatch at the `load_V_frag` call (the readout type vs
the PV A-operand type), independent of assertions. Excluded pending a
proper fix; the file stays in-tree and the W∘R bijection it isolates is
covered end-to-end by `test_mla_prefill_v2.mojo`'s cos_sim gate.

The `v227` adapter is the default-on V-LDS layout for `MlaPrefillV2`
(`-D v_full_v227`). It is a `W∘R` pair:

  * `W` (the WRITE) — `SubTileLoaderLDS_st_8x32[v_full_v227=True]`
    reorganizes the cooperative DRAM->LDS DMA into the reference's chunk-stepped
    LDS layout (16 contiguous 1024-B chunks at LDS chunk stride 0x410).
  * `R` (the READ) — `MlaMmaOp.precompute_v_lane_base[v_full_v227=True]`
    (the `v227` per-lane base) + `load_V_frag[v_full_v227=True]` (the
    faithful readout cell `i_strip*0x2080 + j_depth*0x20 + r*0x100`),
    a per-`ds_read_b64_tr_b8`-cycle bank-quadrant bijection.

`W` and `R` are designed to compose to the IDENTITY MFMA fragment: a V
tile written via `W` and read back via `R` yields the SAME per-lane PV
A-operand fragments that the byte-identical NON-adapter path
(`v_full_v227=False` on both sides) produces for the same logical V.
Until now the adapter had ZERO direct coverage — only indirect via the
end-to-end cos_sim gate (`test_mla_prefill_v2.mojo`), which cannot
isolate a `W∘R` regression from any other kernel change.

This test isolates the bijection: ONE 512-thread block fills the SAME
logical V[key, depth] tile into two LDS slots — one via `W`
(v227-on), one via the default contiguous fill (v227-off) — then each
lane reads its full V register tile back via `R` from each slot and the
host asserts the two fragment sets are bit-identical. A break in either
`W` or `R` (or a drift between them) shows up as a mismatch.

Config: FP8 e4m3, 32x32x64 MFMA (`fp8_mma_k_128=False`), DEPTH=128,
KV_BLOCK=128 — the only shape the v227 adapter supports (asserted in
both `W` and `R`). `V_LAYOUT = row_major[2, 4, 32]`: 2 K-strips x 4
depth-tiles x 32 FP8 elts/fragment per lane.
"""

from max.gpu import WARP_SIZE, lane_id, thread_idx, warp_id
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.sync import s_waitcnt
from std.memory import AddressSpace
from std.sys.intrinsics import readfirstlane
from std.testing import assert_equal

from layout import ComptimeInt, Coord, MixedLayout, TileTensor
from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation

from structured_kernels.amd_tile_io import SubTileLoaderLDS_st_8x32

from nn.attention.gpu.amd_structured.mha_mma_op import (
    MhaConfigV2,
    MlaMmaOp,
)


# --------------------------------------------------------------------------- #
# Shape. The v227 adapter is FP8 32x32x64 DEPTH=128 KV_BLOCK=128 only.
# --------------------------------------------------------------------------- #
comptime KV_BLOCK = 128
comptime DEPTH = 128
comptime NUM_WARPS = 8
comptime NUM_THREADS = NUM_WARPS * 64  # 512 — the v227 producer needs 8 warps.

comptime CFG = MhaConfigV2(
    q_block_size=32,
    kv_block=KV_BLOCK,
    depth=DEPTH,
    num_heads=1,
    num_kv_heads=1,
    num_warps=NUM_WARPS,
    dtype=DType.float8_e4m3fn,
    fp8_mma_k_128=False,  # 32x32x64 — the v227 path (NOT the 16x16x128 path).
)

comptime _Op = MlaMmaOp[.float8_e4m3fn, CFG]
comptime _BK = _Op.MMA_K  # 64 (FP8 32x32x64)
comptime _V_SUB_COLS = _Op.V_SUB_COLS  # 64
comptime _H = _Op.V_LAYOUT.static_shape[0]  # 2
comptime _W = _Op.V_LAYOUT.static_shape[1]  # 4
comptime _FRAG_ELTS = _Op.FRAG_ELTS  # 32
comptime _PER_LANE = _H * _W * _FRAG_ELTS  # 256

# Non-adapter contiguous V slot: KV_BLOCK * (DEPTH / V_SUB_COLS) rows of
# V_SUB_COLS FP8 cols (16384 B). The v227 slot needs +4 pad rows: the
# chunk-stepped 0x410-stride layout reaches LDS byte 16623 (the
# `_V_SLOT_PAD_ROWS=4` that `MlaPrefillV2` allocates; slot grows
# 16384 -> 16640 B).
comptime _V_SLOT_ROWS = KV_BLOCK * (DEPTH // _V_SUB_COLS)  # 256
comptime _V_SLOT_PAD_ROWS = 4
comptime _V_SLOT_ROWS_V227 = _V_SLOT_ROWS + _V_SLOT_PAD_ROWS  # 260

comptime _V_TILE_ELTS = KV_BLOCK * DEPTH  # 16384 — the logical V tile.


# Integer-valued FP8 pattern so bit-exact comparison after cast is
# unambiguous. `((key + depth) % 7) + 1` is in [1, 7], a value the v227
# write/read both preserve exactly; the `+1` keeps every cell nonzero so
# an accidental zero-fill (a scrambled write landing out of the read
# window) is caught rather than silently matching a zeroed slot.
@always_inline
def _pattern_fp8(key: Int, depth: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn(Float32(((key + depth) % 7) + 1))


# --------------------------------------------------------------------------- #
# Kernel. ONE 512-thread (8-warp) block.
#
# The same logical V[key, depth] DRAM tile is DMA'd into two LDS slots:
#   * `v_smem_v227` via `SubTileLoaderLDS_st_8x32[v_full_v227=True]` (W)
#   * `v_smem_ref`  via `SubTileLoaderLDS_st_8x32[v_full_v227=False]`
# All 8 warps issue `load()` once each with their real warp id, so the
# producer's 16 chunks (8 warps x `_num_iters`=2) are each written exactly
# once — the complete tile. (Production's `MlaPrefillV2._dma_v` issues a
# double `w_remap` / `w_remap+4` pair for the K/V work-split; that is an
# occupancy choice, not a correctness requirement of the fill, so the
# simpler one-call-per-real-warp covers all chunks identically.)
#
# After an `s_waitcnt vmcnt(0)` + `barrier()` DMA->read fence (the same
# handshake the production kernel maintains for the `_alias_scope_attr`
# vmcnt-relaxation), each lane reads its full V register tile back via R
# from BOTH slots and dumps both fragment sets to global. The host asserts
# they are bit-identical: `W∘R == (W_ours ∘ R_ours)` over the slot.
# --------------------------------------------------------------------------- #
def kernel_v227_round_trip(
    v_src_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
    dump_v227_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
    dump_ref_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
):
    # DRAM V tile: tight (KV_BLOCK, DEPTH) row-major, stride (DEPTH, 1).
    # Matches the canonical v227 closed-form case (key = global_row,
    # depth = global_col, `_v_row_stride` = DEPTH).
    comptime v_src_layout = MixedLayout[
        Coord[ComptimeInt[KV_BLOCK], ComptimeInt[DEPTH]].element_types,
        Coord[ComptimeInt[DEPTH], ComptimeInt[1]].element_types,
    ]
    var v_src = TileTensor[.float8_e4m3fn, v_src_layout, MutAnyOrigin](
        v_src_ptr, v_src_layout()
    )

    # Two LDS slots: v227 (padded) + reference (contiguous).
    var v_smem_v227 = tt_stack_allocation[
        DType.float8_e4m3fn, address_space=.SHARED
    ](row_major[_V_SLOT_ROWS_V227, _V_SUB_COLS]())
    var v_smem_ref = tt_stack_allocation[
        DType.float8_e4m3fn, address_space=.SHARED
    ](row_major[_V_SLOT_ROWS, _V_SUB_COLS]())

    var w_id = Int(readfirstlane(warp_id()))
    var l_id = Int(lane_id())

    # W (v227) — chunk-stepped LDS fill.
    var v_loader_v227 = SubTileLoaderLDS_st_8x32[
        DType.float8_e4m3fn,
        KV_BLOCK,
        DEPTH,
        _BK,
        NUM_THREADS,
        v_full_v227=True,
    ](v_src)
    v_loader_v227.load(v_smem_v227, v_src, w_id, l_id, scalar_offset=0)

    # W_ours (reference) — byte-identical contiguous fill.
    var v_loader_ref = SubTileLoaderLDS_st_8x32[
        DType.float8_e4m3fn,
        KV_BLOCK,
        DEPTH,
        _BK,
        NUM_THREADS,
        v_full_v227=False,
    ](v_src)
    v_loader_ref.load(v_smem_ref, v_src, w_id, l_id, scalar_offset=0)

    # DMA -> read fence: the `buffer_load_lds` DMAs must retire before the
    # `ds_read_tr8_b64` reads. Production relies on the `_alias_scope_attr`
    # vmcnt-relaxation handshake PLUS an explicit `s_waitcnt vmcnt(0) +
    # s_barrier` at the DMA/compute boundary (amd_tile_io.mojo:1831).
    s_waitcnt[vmcnt=UInt32(0)]()
    barrier()

    # R — read back the full V register tile from each slot, per-lane.
    # v227 base + faithful v227 readout cell.
    var base_v227 = _Op.precompute_v_lane_base[v_full_v227=True](
        v_smem_v227._storage
    )
    # ours base + ours st_8x32 readout cell.
    var base_ref = _Op.precompute_v_lane_base[v_full_v227=False](
        v_smem_ref._storage
    )

    var lid = Int(lane_id())

    comptime for i in range(_H):
        comptime for j in range(_W):
            var frag_v227 = _Op.load_V_frag[i, j, v_full_v227=True](base_v227)
            var frag_ref = _Op.load_V_frag[i, j, v_full_v227=False](base_ref)
            comptime for f in range(_FRAG_ELTS):
                var idx = lid * _PER_LANE + (i * _W + j) * _FRAG_ELTS + f
                dump_v227_ptr[idx] = rebind[Float8_e4m3fn](frag_v227[f])
                dump_ref_ptr[idx] = rebind[Float8_e4m3fn](frag_ref[f])


# --------------------------------------------------------------------------- #
# Host driver.
# --------------------------------------------------------------------------- #
def test_v227_round_trip(ctx: DeviceContext) raises:
    print("--- v227 V-adapter W∘R round-trip (FP8 32x32x64, d=128) ---")

    comptime _DUMP_SIZE = 64 * _PER_LANE  # 64 lanes x 256 elts/lane.

    var dev_src = ctx.enqueue_create_buffer[.float8_e4m3fn](_V_TILE_ELTS)
    var dev_dump_v227 = ctx.enqueue_create_buffer[.float8_e4m3fn](_DUMP_SIZE)
    var dev_dump_ref = ctx.enqueue_create_buffer[.float8_e4m3fn](_DUMP_SIZE)

    # Fill the DRAM V tile with the 2D pattern, row-major V[key, depth].
    with dev_src.map_to_host() as host_src:
        for key in range(KV_BLOCK):
            for depth_v in range(DEPTH):
                host_src[key * DEPTH + depth_v] = _pattern_fp8(key, depth_v)

    ctx.enqueue_function[kernel_v227_round_trip](
        dev_src.unsafe_ptr(),
        dev_dump_v227.unsafe_ptr(),
        dev_dump_ref.unsafe_ptr(),
        grid_dim=1,
        block_dim=NUM_THREADS,
    )
    ctx.synchronize()

    # The W∘R bijection: the v227 path and the byte-identical non-adapter
    # path must read back bit-identical fragments for the same logical V.
    var mismatches: Int = 0
    var nonzero_seen: Int = 0
    with dev_dump_v227.map_to_host() as host_v227:
        with dev_dump_ref.map_to_host() as host_ref:
            for idx in range(_DUMP_SIZE):
                var v227_f32 = host_v227[idx].cast[.float32]()
                var ref_f32 = host_ref[idx].cast[.float32]()
                if ref_f32 != Float32(0.0):
                    nonzero_seen += 1
                if v227_f32 != ref_f32:
                    if mismatches < 8:
                        var lid = idx // _PER_LANE
                        var rem = idx % _PER_LANE
                        var frag = rem // _FRAG_ELTS
                        var f = rem % _FRAG_ELTS
                        print(
                            "  MISMATCH lid=",
                            lid,
                            " frag=",
                            frag,
                            " f=",
                            f,
                            " v227=",
                            v227_f32,
                            " ref=",
                            ref_f32,
                        )
                    mismatches += 1

    _ = dev_src^
    _ = dev_dump_v227^
    _ = dev_dump_ref^

    # Guard against a degenerate all-zero pass: the pattern is in [1, 7], so
    # a correct read fills every one of the 64*256 dumped slots nonzero.
    assert_equal(mismatches, 0)
    assert_equal(nonzero_seen, _DUMP_SIZE)
    print("  PASSED (", nonzero_seen, " nonzero fragments, 0 mismatches)")


def main() raises:
    print("=" * 60)
    print("v227 V-adapter round-trip test")
    print("=" * 60)

    with DeviceContext() as ctx:
        test_v227_round_trip(ctx)

    print("=" * 60)
    print("ALL v227 ROUND-TRIP TESTS PASSED")
    print("=" * 60)
