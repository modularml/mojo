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
"""Component-isolation tests for the FP8 e4m3 paths in `MhaMmaOp`.

Validates the FP8 K/V SMEM->register loaders BEFORE FP8 is wired
end-to-end through `MhaPrefillV2`. The tests are kept
tightly scoped to `MhaMmaOp` itself so a regression in the loader
math is caught without running the full kernel.

Coverage:

1. Comptime constants — at d=128, KV_BLOCK=64:
   - FP8: MMA_K=64, FRAG_ELTS=32, K_SUB_COLS=64, V_SUB_COLS=64.
   - BF16 regression guard: MMA_K=16, FRAG_ELTS=8,
     K_SUB_COLS=32, V_SUB_COLS=32.
   - K_LAYOUT / V_LAYOUT / ATT_LAYOUT static shapes.

2. K swizzle round-trip (host, no GPU) — for both BF16 (size=2) and
   FP8 (size=1): every `(r, c)` in `(0..32, 0..K_SUB_COLS)` maps to a
   byte offset < sub-block size, and the XOR is self-inverse.

3. K loader round-trip (GPU, MI355) — fills K SMEM with a swizzled
   pattern, calls `MhaMmaOp[float8_e4m3fn, ...].load_K`, dumps each
   lane's 32-elt fragment to global, checks bit-exact against the
   original 2D pattern.

4. V loader round-trip (GPU, MI355) — fills V SMEM in the
   sub-tile-major layout the FP8 V loader actually reads (the
   `SubTileLoaderLDS_st_8x32` DMA image, sub-tiles V_SUB_ROWS×
   V_SUB_COLS in row-major-by-sub-tile order), calls
   `MhaMmaOp[float8_e4m3fn, ...].load_V`, dumps fragments to global,
   checks bit-exact against the original 2D pattern using the
   paired-lane geometry from `kv_buffer.mojo`.

The `mma_QK` FP8 dispatch test is intentionally skipped — it will be
exercised transitively by the kernel-level tests once FP8 is wired
end-to-end.
"""

from max.gpu import lane_id, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.testing import assert_equal

from layout.tile_layout import row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation

from nn.attention.gpu.amd_structured.mha_mma_op import (
    MhaConfigV2,
    MhaMmaOp,
)


# --------------------------------------------------------------------------- #
# Pattern. ((r + c) % 7) is in [0, 7), trivially representable in FP8 e4m3 and
# integer-valued so bit-exact comparison after cast doesn't introduce
# float-round-off ambiguity.
# --------------------------------------------------------------------------- #


@always_inline
def _pattern_fp8(r: Int, c: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn(Float32((r + c) % 7))


# --------------------------------------------------------------------------- #
# Test 1: Comptime constants and layout shapes.
# --------------------------------------------------------------------------- #


def test_comptime_constants[
    depth: Int,
]() raises:
    print("--- comptime constants (depth=", depth, ") ---")

    comptime CFG = MhaConfigV2(
        q_block_size=32,
        kv_block=64,
        depth=depth,
        num_heads=1,
        num_kv_heads=1,
    )

    # FP8 path constants.
    comptime _OpFP8 = MhaMmaOp[.float8_e4m3fn, CFG]
    comptime assert (
        _OpFP8.MMA_K == 64
    ), "FP8 MhaMmaOp.MMA_K must be 64 (v_mfma_scale_f32_32x32x64_f8f6f4)"
    comptime assert (
        _OpFP8.FRAG_ELTS == 32
    ), "FP8 MhaMmaOp.FRAG_ELTS must be 32 (MMA_M*MMA_K/64 = 32*64/64)"
    comptime assert (
        _OpFP8.K_SUB_COLS == 64
    ), "FP8 MhaMmaOp.K_SUB_COLS must be 64 (32 rows * 64 B-equivalent)"
    comptime assert (
        _OpFP8.V_SUB_COLS == 64
    ), "FP8 MhaMmaOp.V_SUB_COLS must be 64"
    comptime assert (
        _OpFP8.K_SUB_ROWS == 32
    ), "FP8 MhaMmaOp.K_SUB_ROWS must be 32 (byte-positional swizzle)"
    comptime assert (
        _OpFP8.V_SUB_ROWS == 8
    ), "FP8 MhaMmaOp.V_SUB_ROWS must stay 8 (parent SMEM geometry shared)"
    comptime assert (
        _OpFP8.ROWL_STRIDE == 32
    ), "FP8 MhaMmaOp.ROWL_STRIDE must be MMA_K/2 = 32"

    # K_LAYOUT.static_shape == (KV_BLOCK/MMA_M, DEPTH/MMA_K, FRAG_ELTS).
    # At KV_BLOCK=64, MMA_M=32 -> 2.
    # depth=128, MMA_K=64 -> 2; depth=64 -> 1.
    comptime assert (
        _OpFP8.K_LAYOUT.static_shape[0] == 64 // 32
    ), "FP8 K_LAYOUT dim 0 must be KV_BLOCK/MMA_M"
    comptime assert (
        _OpFP8.K_LAYOUT.static_shape[1] == depth // 64
    ), "FP8 K_LAYOUT dim 1 must be DEPTH/MMA_K"
    comptime assert (
        _OpFP8.K_LAYOUT.static_shape[2] == 32
    ), "FP8 K_LAYOUT dim 2 must be FRAG_ELTS = 32"

    # V_LAYOUT.static_shape == (KV_BLOCK/MMA_K, DEPTH/MMA_N, FRAG_ELTS).
    # KV_BLOCK=64 / MMA_K=64 = 1; depth=128 / MMA_N=32 = 4; depth=64 = 2.
    comptime assert (
        _OpFP8.V_LAYOUT.static_shape[0] == 64 // 64
    ), "FP8 V_LAYOUT dim 0 must be KV_BLOCK/MMA_K"
    comptime assert (
        _OpFP8.V_LAYOUT.static_shape[1] == depth // 32
    ), "FP8 V_LAYOUT dim 1 must be DEPTH/MMA_N"
    comptime assert (
        _OpFP8.V_LAYOUT.static_shape[2] == 32
    ), "FP8 V_LAYOUT dim 2 must be FRAG_ELTS = 32"

    # ATT_LAYOUT.static_shape == (KV_BLOCK/MMA_M, Q_BLOCK_SIZE/MMA_N, 16).
    # KV_BLOCK=64 / 32 = 2; Q_BLOCK_SIZE=32 / 32 = 1; 16 = 32*32/64.
    comptime assert (
        _OpFP8.ATT_LAYOUT.static_shape[0] == 2
    ), "FP8 ATT_LAYOUT dim 0 must be KV_BLOCK/MMA_M = 2"
    comptime assert (
        _OpFP8.ATT_LAYOUT.static_shape[1] == 1
    ), "FP8 ATT_LAYOUT dim 1 must be Q_BLOCK_SIZE/MMA_N = 1"
    comptime assert (
        _OpFP8.ATT_LAYOUT.static_shape[2] == 16
    ), "FP8 ATT_LAYOUT dim 2 must be 16 FP32 acc elts per lane"

    # BF16 regression guard — the original BF16 kernel values must not move.
    comptime _OpBF16 = MhaMmaOp[.bfloat16, CFG]
    comptime assert (
        _OpBF16.MMA_K == 16
    ), "BF16 MhaMmaOp.MMA_K must stay 16 (v_mfma_f32_32x32x16_bf16)"
    comptime assert (
        _OpBF16.FRAG_ELTS == 8
    ), "BF16 MhaMmaOp.FRAG_ELTS must stay 8"
    comptime assert (
        _OpBF16.K_SUB_COLS == 32
    ), "BF16 MhaMmaOp.K_SUB_COLS must stay 32"
    comptime assert (
        _OpBF16.V_SUB_COLS == 32
    ), "BF16 MhaMmaOp.V_SUB_COLS must stay 32"
    comptime assert (
        _OpBF16.K_SUB_ROWS == 32
    ), "BF16 MhaMmaOp.K_SUB_ROWS must stay 32"
    comptime assert (
        _OpBF16.V_SUB_ROWS == 8
    ), "BF16 MhaMmaOp.V_SUB_ROWS must stay 8"
    comptime assert (
        _OpBF16.ROWL_STRIDE == 8
    ), "BF16 MhaMmaOp.ROWL_STRIDE must stay MMA_K/2 = 8"

    print("  PASSED")


# --------------------------------------------------------------------------- #
# Test 2: Swizzle round-trip — pure-host check that _swizzle_K_sub is
# self-inverse over the sub-block and stays in-bounds. The swizzle is
# byte-positional (XOR on bits 4..10 of the byte offset) — applying it
# twice with the same masks must recover the original byte offset.
# --------------------------------------------------------------------------- #


def test_swizzle_round_trip_fp8() raises:
    print("--- _swizzle_K_sub round-trip (FP8, size=1) ---")

    comptime CFG = MhaConfigV2(
        q_block_size=32, kv_block=64, depth=128, num_heads=1, num_kv_heads=1
    )
    comptime _Op = MhaMmaOp[.float8_e4m3fn, CFG]

    comptime sub_bytes = _Op.K_SUB_ROWS * _Op.K_SUB_COLS * 1  # FP8 size=1
    var max_seen: Int = 0
    for r in range(_Op.K_SUB_ROWS):
        for c in range(_Op.K_SUB_COLS):
            var swz = _Op._swizzle_K_sub(r, c)
            assert_equal(swz >= 0, True)
            assert_equal(swz < sub_bytes, True)
            if swz > max_seen:
                max_seen = swz

            # Self-inverse: applying the same byte-level XOR pair to swz
            # (treating swz as the byte offset) returns the original
            # `r * K_SUB_COLS + c` byte offset.
            # Self-inverse: apply the same `Swizzle(2, 4, 4)` byte-level
            # XOR (bits 4..5 ^= bits 8..9) to swz to recover the
            # original byte offset.
            var inv_swiz = ((swz >> 8) & 3) << 4
            var inv = swz ^ inv_swiz
            var original = r * _Op.K_SUB_COLS + c
            assert_equal(inv, original)

    # FP8 sub-block has 32 * 64 = 2048 byte offsets [0, 2048).
    assert_equal(max_seen < 2048, True)
    print(
        "  swept",
        _Op.K_SUB_ROWS * _Op.K_SUB_COLS,
        "FP8 cells, max byte offset =",
        max_seen,
    )
    print("  PASSED")


def test_swizzle_round_trip_bf16() raises:
    print("--- _swizzle_K_sub round-trip (BF16, size=2) ---")

    comptime CFG = MhaConfigV2(
        q_block_size=32, kv_block=64, depth=128, num_heads=1, num_kv_heads=1
    )
    comptime _Op = MhaMmaOp[.bfloat16, CFG]

    comptime sub_bytes = (_Op.K_SUB_ROWS * _Op.K_SUB_COLS * 2)  # BF16 size=2

    for r in range(_Op.K_SUB_ROWS):
        for c in range(_Op.K_SUB_COLS):
            var swz = _Op._swizzle_K_sub(r, c)
            assert_equal(swz >= 0, True)
            assert_equal(swz < sub_bytes, True)
            # Self-inverse: apply the same `Swizzle(2, 4, 4)` byte-level
            # XOR (bits 4..5 ^= bits 8..9) to swz to recover the
            # original byte offset.
            var inv_swiz = ((swz >> 8) & 3) << 4
            var inv = swz ^ inv_swiz
            var original = 2 * (r * _Op.K_SUB_COLS + c)
            assert_equal(inv, original)

    print("  PASSED")


# --------------------------------------------------------------------------- #
# Test 3: FP8 load_K round-trip.
#
# Strategy: pre-swizzle a (KV_BLOCK, DEPTH) FP8 pattern when writing it
# into SMEM. Then load_K (which unswizzles on read) returns the original
# logical (key, depth) values into the per-lane register fragments.
#
# The lane geometry for load_K FP8 (from mha_mma_op.mojo):
#   row_offset = lid % 32                          (0..31)
#   col_offset = ROWL_STRIDE * (lid // 32) = 32 * (lid // 32) ∈ {0, 32}
# Each base tile is 32 rows × MMA_K=64 cols. For a given
# (register_row, register_col):
#   - The base tile lives in K SMEM sub-block `sub_id = register_row *
#     (DEPTH/K_SUB_COLS) + register_col` (FP8: no col-parity collapse
#     since K_SUB_COLS = MMA_K = 64).
#   - Lane lid owns 32 contiguous columns starting at `col_offset` within
#     the base tile, all within row `row_offset`.
#   - So lane lid's fragment[i] = value at
#       (key_global, depth_global)
#     where
#       key_global  = register_row * MMA_M + row_offset
#       depth_global = register_col * MMA_K + col_offset + i
#     for i in [0, FRAG_ELTS=32).
# --------------------------------------------------------------------------- #


def kernel_load_K_fp8[
    cfg: MhaConfigV2,
    depth: Int,
](
    src_swz_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
    dump_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
):
    """Loads K SMEM from `src_swz_ptr` (already swizzled), calls
    `MhaMmaOp.load_K`, and dumps each lane's fragment to `dump_ptr`.

    `dump_ptr` layout: per-lane stride = FRAG_ELTS * num_base_tiles, where
    num_base_tiles = K_LAYOUT.static_shape[0] * K_LAYOUT.static_shape[1].
    Lane lid's base-tile (rr, rc) fragment lives at
    `lid * total_per_lane + (rr * width + rc) * FRAG_ELTS`.
    """
    comptime _Op = MhaMmaOp[.float8_e4m3fn, cfg]
    comptime _KV_BLOCK = cfg.kv_block
    comptime _DEPTH = depth
    comptime _K_SUB_COLS = _Op.K_SUB_COLS
    comptime _NUM_BLOCK_COLS_K = _DEPTH // _K_SUB_COLS
    comptime _K_SLOT_ROWS = _KV_BLOCK * _NUM_BLOCK_COLS_K
    comptime smem_layout_k = row_major[_K_SLOT_ROWS, _K_SUB_COLS]()

    var k_smem = tt_stack_allocation[.float8_e4m3fn, address_space=.SHARED](
        smem_layout_k
    )

    # Cooperatively fill SMEM from the pre-swizzled global init buffer.
    # Total bytes = _K_SLOT_ROWS * _K_SUB_COLS; 64 threads each write a
    # stride of 64.
    var tid = Int(thread_idx.x)
    comptime _total = _K_SLOT_ROWS * _K_SUB_COLS
    var i = tid
    while i < _total:
        k_smem._storage[i] = src_swz_ptr[i]
        i += 64
    barrier()

    # Allocate register tile and call load_K.
    var k_reg = tt_stack_allocation[.float8_e4m3fn, address_space=.LOCAL](
        _Op.K_LAYOUT
    )
    _Op.load_K(k_reg, k_smem)

    # Dump each lane's fragment to global. Per K_LAYOUT:
    #   shape = (KV_BLOCK/MMA_M, DEPTH/MMA_K, FRAG_ELTS).
    comptime _H = _Op.K_LAYOUT.static_shape[0]
    comptime _W = _Op.K_LAYOUT.static_shape[1]
    comptime _F = _Op.FRAG_ELTS
    comptime _total_per_lane = _H * _W * _F
    var lid = Int(lane_id())
    var k_reg_v = k_reg.vectorize[1, 1, _F]()

    comptime for rr in range(_H):
        comptime for rc in range(_W):
            var frag = k_reg_v[rr, rc, 0]
            comptime for f in range(_F):
                var idx = lid * _total_per_lane + (rr * _W + rc) * _F + f
                dump_ptr[idx] = rebind[Float8_e4m3fn](frag[f])


def test_load_K_fp8[
    depth: Int,
](ctx: DeviceContext) raises:
    print("--- load_K FP8 round-trip (depth=", depth, ") ---")

    comptime KV_BLOCK = 64
    comptime CFG = MhaConfigV2(
        q_block_size=32,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=1,
        num_kv_heads=1,
    )
    comptime _Op = MhaMmaOp[.float8_e4m3fn, CFG]
    comptime _K_SUB_ROWS = _Op.K_SUB_ROWS
    comptime _K_SUB_COLS = _Op.K_SUB_COLS
    comptime _NUM_BLOCK_COLS_K = depth // _K_SUB_COLS
    comptime _K_SLOT_ROWS = KV_BLOCK * _NUM_BLOCK_COLS_K
    comptime _SMEM_SIZE = _K_SLOT_ROWS * _K_SUB_COLS  # bytes (FP8)
    comptime _MMA_M = _Op.MMA_M
    comptime _MMA_K = _Op.MMA_K
    comptime _FRAG_ELTS = _Op.FRAG_ELTS
    comptime _H = _Op.K_LAYOUT.static_shape[0]
    comptime _W = _Op.K_LAYOUT.static_shape[1]
    comptime _total_per_lane = _H * _W * _FRAG_ELTS
    comptime _DUMP_SIZE = 64 * _total_per_lane

    var dev_init = ctx.enqueue_create_buffer[.float8_e4m3fn](_SMEM_SIZE)
    var dev_dump = ctx.enqueue_create_buffer[.float8_e4m3fn](_DUMP_SIZE)

    # Build the pre-swizzled K SMEM init image on host.
    # For each logical (r, c) in (0..KV_BLOCK, 0..DEPTH):
    #   sub_row = r / K_SUB_ROWS, sub_col = c / K_SUB_COLS
    #   sub_id  = sub_row * _NUM_BLOCK_COLS_K + sub_col
    #   in-sub r, c -> _swizzle_K_sub gives byte offset within sub-block
    #   write _pattern_fp8(r, c) at sub_id * sub_bytes + swz
    with dev_init.map_to_host() as host_init:
        # Zero-init the whole image (any holes will show up as 0s).
        for i in range(_SMEM_SIZE):
            host_init[i] = Float8_e4m3fn(0.0)

        var sub_bytes = _K_SUB_ROWS * _K_SUB_COLS  # FP8 size=1
        for r in range(KV_BLOCK):
            for c in range(depth):
                var sub_row = r // _K_SUB_ROWS
                var sub_col = c // _K_SUB_COLS
                var sub_id = sub_row * _NUM_BLOCK_COLS_K + sub_col
                var sr = r % _K_SUB_ROWS
                var sc = c % _K_SUB_COLS
                var swz = _Op._swizzle_K_sub(sr, sc)
                host_init[sub_id * sub_bytes + swz] = _pattern_fp8(r, c)

    ctx.enqueue_function[kernel_load_K_fp8[CFG, depth]](
        dev_init.unsafe_ptr(),
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    var mismatches: Int = 0
    with dev_dump.map_to_host() as host_dump:
        for lid in range(64):
            var row_offset = lid % 32
            var col_offset = _Op.ROWL_STRIDE * (lid // 32)
            for rr in range(_H):
                for rc in range(_W):
                    var key_global = rr * _MMA_M + row_offset
                    var depth_base = rc * _MMA_K + col_offset
                    for f in range(_FRAG_ELTS):
                        var depth_global = depth_base + f
                        var expected = _pattern_fp8(key_global, depth_global)
                        var idx = (
                            lid * _total_per_lane
                            + (rr * _W + rc) * _FRAG_ELTS
                            + f
                        )
                        var got_f32 = host_dump[idx].cast[.float32]()
                        var exp_f32 = expected.cast[.float32]()
                        if got_f32 != exp_f32:
                            if mismatches < 5:
                                print(
                                    "  MISMATCH lid=",
                                    lid,
                                    " rr=",
                                    rr,
                                    " rc=",
                                    rc,
                                    " f=",
                                    f,
                                    " key=",
                                    key_global,
                                    " depth=",
                                    depth_global,
                                    " got=",
                                    got_f32,
                                    " expected=",
                                    exp_f32,
                                )
                            mismatches += 1

    _ = dev_init^
    _ = dev_dump^
    assert_equal(mismatches, 0)
    print("  PASSED")


# --------------------------------------------------------------------------- #
# Test 4: FP8 load_V round-trip.
#
# The FP8 V loader treats V SMEM as `depth/BK` contiguous (BN×BK) depth
# blocks, with BN = KV_BLOCK = 64 and BK = MMA_K = 64 for FP8. The
# logical V[key, depth] maps to byte offset:
#     blk = depth // BK
#     d_in_blk = depth % BK
#     offset = blk * BN * BK + key * BK + d_in_blk
# (No swizzle on the V SMEM image — the test uses the unswizzled path
# matching `MhaPrefillV2` BF16 V geometry.)
#
# load_V per-lane geometry (mirrors `kv_buffer.mojo`):
#   lid = lane_id()
#   lane_in_row = lid % 16
#   pair_idx     = lane_in_row // 2
#   is_odd       = lane_in_row % 2
#   row_in_warp  = lid // 16
#   is_hw1       = lid // 32
#   rel_key      = (pair_idx % 4) + (pair_idx // 4) * 8       (0..7)
#   depth_base   = (row_in_warp % 2) * 16 + is_odd * 8         (0,8,16,24)
#   hw_key_shift = is_hw1 * 4
#
# For each (i, j) in (V_LAYOUT.shape[0], V_LAYOUT.shape[1]):
#   row_offset   = i * 64       (since one bk_tile = 64 keys)
#   depth_offset = j * MMA_M = j * 32
#   blk = depth_offset // BK,   d_in_blk = depth_offset % BK
#
# Per-lane fragment of 32 FP8 elts is formed from 4 `ds_read_tr8_b64`s
# at key_base ∈ {0, 16, 32, 48}, each returning 8 elts. `ds_read_tr8_b64`
# splits a 16-lane row into two 8-lane sub-groups by lane parity (even
# lanes 0/2/.../14 form sub-group A, odd lanes 1/3/.../15 form sub-
# group B). Each sub-group performs an 8x8 byte transpose: the 8x8
# input matrix has row i = pre-transpose lane (2*i for sub-A, 2*i+1
# for sub-B)'s 8-byte fetch. Transpose redistributes the bytes back
# into the row's 16 lanes — sub-A's transposed rows land on lanes
# 0..7, sub-B's land on lanes 8..15. So lane `lane_in_row`:
#   - 0..7  → holds sub-A's transposed row `lane_in_row` → depth =
#             `lane_in_row` (within block, plus row_in_warp*16)
#   - 8..15 → holds sub-B's transposed row `lane_in_row - 8` → depth
#             = `lane_in_row` (sub-B's depth_base 8 plus 0..7 offset)
# In both cases depth_in_block = lane_in_row, and the 8 elts in the
# fragment are KEYS at this one depth:
#   frag[8*k + f] = V[key_k_f, depth_lid]
# where
#   key_k_f   = row_offset + 16*k + (f%4) + (f//4)*8 + hw_key_shift
#   depth_lid = blk*BK + d_in_blk + lane_in_row + (row_in_warp%2)*16
#
# Note: an earlier reading of `kv_buffer.mojo:514-516` was "each lane
# holds 8 contiguous depths from one key"; that wording reflects the
# 16-lane ROW's collective output coverage (16 unique depths total),
# not per-lane semantics. The empirical per-lane pattern is
# "8 keys × 1 depth", verified by this test.
# --------------------------------------------------------------------------- #


def kernel_load_V_fp8[
    cfg: MhaConfigV2,
    depth: Int,
](
    src_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
    dump_ptr: MutPointer[Float8_e4m3fn, MutAnyOrigin],
):
    """Fills V SMEM from `src_ptr` (no swizzle), calls
    `MhaMmaOp.load_V`, and dumps each lane's fragment to `dump_ptr`.

    V SMEM layout follows the parent `MhaPrefillV2` convention:
    `row_major[KV_BLOCK * (DEPTH / V_SUB_COLS), V_SUB_COLS]`. The FP8 V
    loader reads this as the sub-tile-major image the
    `SubTileLoaderLDS_st_8x32` DMA produces — sub-tiles
    (V_SUB_ROWS=8 × V_SUB_COLS=64) in row-major-by-sub-tile order, NOT
    a contiguous (BN×BK) depth-block slab. The host fill in
    `test_load_V_fp8` writes the same sub-tile-major byte offsets.
    """
    comptime _Op = MhaMmaOp[.float8_e4m3fn, cfg]
    comptime _KV_BLOCK = cfg.kv_block
    comptime _DEPTH = depth
    comptime _V_SUB_COLS = _Op.V_SUB_COLS
    comptime _NUM_BLOCK_COLS_V = _DEPTH // _V_SUB_COLS
    comptime _V_SLOT_ROWS = _KV_BLOCK * _NUM_BLOCK_COLS_V
    comptime smem_layout_v = row_major[_V_SLOT_ROWS, _V_SUB_COLS]()

    var v_smem = tt_stack_allocation[.float8_e4m3fn, address_space=.SHARED](
        smem_layout_v
    )

    # Cooperatively fill SMEM from src.
    var tid = Int(thread_idx.x)
    comptime _total = _V_SLOT_ROWS * _V_SUB_COLS
    var i = tid
    while i < _total:
        v_smem._storage[i] = src_ptr[i]
        i += 64
    barrier()

    # Allocate register tile and call load_V.
    var v_reg = tt_stack_allocation[.float8_e4m3fn, address_space=.LOCAL](
        _Op.V_LAYOUT
    )
    _Op.load_V(v_reg, v_smem)

    # Dump each lane's fragment to global.
    comptime _H = _Op.V_LAYOUT.static_shape[0]
    comptime _W = _Op.V_LAYOUT.static_shape[1]
    comptime _F = _Op.FRAG_ELTS
    comptime _total_per_lane = _H * _W * _F
    var lid = Int(lane_id())
    var v_reg_v = v_reg.vectorize[1, 1, _F]()

    comptime for rr in range(_H):
        comptime for rc in range(_W):
            var frag = v_reg_v[rr, rc, 0]
            comptime for f in range(_F):
                var idx = lid * _total_per_lane + (rr * _W + rc) * _F + f
                dump_ptr[idx] = rebind[Float8_e4m3fn](frag[f])


def test_load_V_fp8[
    depth: Int,
](ctx: DeviceContext) raises:
    print("--- load_V FP8 round-trip (depth=", depth, ") ---")

    comptime KV_BLOCK = 64
    comptime CFG = MhaConfigV2(
        q_block_size=32,
        kv_block=KV_BLOCK,
        depth=depth,
        num_heads=1,
        num_kv_heads=1,
    )
    comptime _Op = MhaMmaOp[.float8_e4m3fn, CFG]
    comptime _V_SUB_COLS = _Op.V_SUB_COLS
    comptime _NUM_BLOCK_COLS_V = depth // _V_SUB_COLS
    comptime _V_SLOT_ROWS = KV_BLOCK * _NUM_BLOCK_COLS_V
    comptime _SMEM_SIZE = _V_SLOT_ROWS * _V_SUB_COLS  # bytes (FP8)
    comptime _BK = _Op.MMA_K  # 64 for FP8
    comptime _MMA_M = _Op.MMA_M
    comptime _FRAG_ELTS = _Op.FRAG_ELTS
    comptime _H = _Op.V_LAYOUT.static_shape[0]
    comptime _W = _Op.V_LAYOUT.static_shape[1]
    comptime _total_per_lane = _H * _W * _FRAG_ELTS
    comptime _DUMP_SIZE = 64 * _total_per_lane

    var dev_init = ctx.enqueue_create_buffer[.float8_e4m3fn](_SMEM_SIZE)
    var dev_dump = ctx.enqueue_create_buffer[.float8_e4m3fn](_DUMP_SIZE)

    # Fill V SMEM image on host in the sub-tile-major layout the FP8 V
    # loader actually reads (the `SubTileLoaderLDS_st_8x32` DMA
    # layout `MhaMmaOp.load_V` consumes), NOT a contiguous (BN × BK)
    # depth-block slab. The DMA writes sub-tiles (V_SUB_ROWS=8 ×
    # V_SUB_COLS=64) in row-major-by-sub-tile order, so for logical
    # V[key, depth_v] (key in 0..KV_BLOCK, depth_v in 0..depth):
    #   sub_row    = key // V_SUB_ROWS
    #   sub_col    = depth_v // V_SUB_COLS         (= blk, since
    #                V_SUB_COLS == BK here)
    #   row_in_sub = key % V_SUB_ROWS
    #   slot_row   = (sub_row * (depth/V_SUB_COLS) + sub_col)
    #                  * V_SUB_ROWS + row_in_sub
    #   offset     = slot_row * V_SUB_COLS + depth_v % V_SUB_COLS
    # A contiguous `blk*BN*BK + key*BK + d` fill places the wrong
    # element at the byte the loader reads — e.g. for key=8 the loader
    # reads slot_row 16 (byte 1024), where the contiguous fill stored
    # key=16 — which is exactly the original (key=8 -> got key=16)
    # mismatch this layout fix corrects.
    comptime _SUBTILES_PER_ROW = depth // _V_SUB_COLS
    with dev_init.map_to_host() as host_init:
        for i in range(_SMEM_SIZE):
            host_init[i] = Float8_e4m3fn(0.0)

        for key in range(KV_BLOCK):
            for depth_v in range(depth):
                var sub_row = key // _Op.V_SUB_ROWS
                var sub_col = depth_v // _V_SUB_COLS
                var row_in_sub = key % _Op.V_SUB_ROWS
                var slot_row = (
                    sub_row * _SUBTILES_PER_ROW + sub_col
                ) * _Op.V_SUB_ROWS + row_in_sub
                var off = slot_row * _V_SUB_COLS + depth_v % _V_SUB_COLS
                host_init[off] = _pattern_fp8(key, depth_v)

    ctx.enqueue_function[kernel_load_V_fp8[CFG, depth]](
        dev_init.unsafe_ptr(),
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    var mismatches: Int = 0
    with dev_dump.map_to_host() as host_dump:
        for lid in range(64):
            var lane_in_row = lid % 16
            var row_in_warp = lid // 16
            var is_hw1 = lid // 32
            var hw_key_shift = is_hw1 * 4

            for rr in range(_H):
                for rc in range(_W):
                    var row_off = rr * 64
                    var depth_off = rc * _MMA_M
                    var d_in_blk = depth_off % _BK
                    var blk = depth_off // _BK
                    # `ds_read_tr8_b64` semantics on a 16-lane row:
                    # 16 lanes provide 16 source pointers; the
                    # hardware splits them into two 8-lane sub-groups
                    # by even/odd lane index, performs an 8x8 byte
                    # transpose within each sub-group, and writes the
                    # results back interleaved into the row's 16
                    # lanes.
                    #
                    # Empirically (and as required for MFMA C-output
                    # column 0 of the FP8 V load), after the
                    # transpose:
                    #   - lanes 0..7 hold the even-sub-group's
                    #     transposed data; each lane holds 8 bytes at
                    #     a SINGLE depth value (depth = lane_in_row)
                    #     from 8 different keys.
                    #   - lanes 8..15 hold the odd-sub-group's
                    #     transposed data; each lane holds 8 bytes
                    #     at depth = lane_in_row (i.e., 8..15) from
                    #     the same 8 keys.
                    # The 8 keys at each (k, lane) are
                    #   K[f] = (f % 4) + (f // 4) * 8   for f in 0..8
                    # contributed by pre-transpose pair_idx 0..7.
                    # `row_in_warp` shifts depth by 16 for rows 1/3;
                    # `is_hw1` shifts keys by +4 for hw1.
                    var depth_in_block = lane_in_row + (row_in_warp % 2) * 16
                    for k in range(4):
                        var key_base = 16 * k
                        for f in range(8):
                            var K_f = (f % 4) + (f // 4) * 8
                            var key_global = (
                                row_off + key_base + K_f + hw_key_shift
                            )
                            var depth_global = (
                                blk * _BK + d_in_blk + depth_in_block
                            )
                            var expected = _pattern_fp8(
                                key_global, depth_global
                            )
                            var idx = (
                                lid * _total_per_lane
                                + (rr * _W + rc) * _FRAG_ELTS
                                + (8 * k + f)
                            )
                            var got_f32 = host_dump[idx].cast[.float32]()
                            var exp_f32 = expected.cast[.float32]()
                            if got_f32 != exp_f32:
                                if mismatches < 5:
                                    print(
                                        "  MISMATCH lid=",
                                        lid,
                                        " rr=",
                                        rr,
                                        " rc=",
                                        rc,
                                        " k=",
                                        k,
                                        " f=",
                                        f,
                                        " key=",
                                        key_global,
                                        " depth=",
                                        depth_global,
                                        " got=",
                                        got_f32,
                                        " expected=",
                                        exp_f32,
                                    )
                                mismatches += 1

    _ = dev_init^
    _ = dev_dump^
    assert_equal(mismatches, 0)
    print("  PASSED")


# --------------------------------------------------------------------------- #
# Main: iterate the depth axis here (test parametrization driven from
# `main`). `num_kv_heads=1, num_heads=1, kv_block=64` are baked at the
# call sites.
# --------------------------------------------------------------------------- #


def main() raises:
    print("=" * 60)
    print("MhaMmaOp FP8 component-isolation tests")
    print("=" * 60)

    # Test 1: comptime constants — host, no GPU launch. Both depths
    # exercised so the (depth/MMA_K) and (depth/MMA_N) factors are
    # checked at d=64 (1, 2) and d=128 (2, 4).
    test_comptime_constants[128]()
    test_comptime_constants[64]()

    # Test 2: swizzle round-trip — host, no GPU launch. The swizzle is
    # byte-positional, so depth doesn't matter; one check per dtype.
    test_swizzle_round_trip_fp8()
    test_swizzle_round_trip_bf16()

    # Tests 3 & 4: GPU round-trips via 1-warp kernel.
    with DeviceContext() as ctx:
        test_load_K_fp8[128](ctx)
        test_load_K_fp8[64](ctx)
        test_load_V_fp8[128](ctx)
        test_load_V_fp8[64](ctx)

    print("=" * 60)
    print("ALL FP8 MhaMmaOp TESTS PASSED!")
    print("=" * 60)
