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
"""Lane-layout regression test for `MhaPrefillV2._load_q_and_scale` (FP8).

Locks in the contiguous-K-cols-per-lane invariant the MFMA hardware
expects from the Q (B-operand) fragment of 32x32x64 FP8 MFMA. A prior
implementation tiled the source into two `[Q_BLOCK_SIZE, 32]` halves
split at K-col 32 and issued two separate `RegTileLoader.load()` calls,
which produced a NON-contiguous lane fragment (lane 0 owned K-cols
`{0..15, 32..47}`) that mismatched the K loader's contiguous layout.
The MFMA pairs `A[lane, f]` with `B[lane, f]` for all f; the
misalignment introduced a wrong-pairing for f in `[16, 32)` and cost
~6% cos_sim at production DeepSeek-V3 MLA shape.

Without this regression test, the lane layout is only validated
transitively through end-to-end correctness gates — which let this
specific FP8 mismatch escape 27 phases of debugging because BF16's
smaller fragment (8 BF16 = 16 B, single buffer_load_lds) didn't trigger
the half-split bug.

The test populates Q gmem with a per-column-group pattern
`Q[r, c] = Float8(c // 8)` and asserts that each lane's per-K-tile
fragment matches the contiguous formula
`frag[f] == (lid // 32) * 32 / 8 + f / 8` for `f in [0, 32)`.
"""

from max.gpu import lane_id, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.testing import assert_equal

from layout import TileTensor
from layout.tile_layout import row_major

from nn.attention.gpu.amd_structured.mha_mma_op import MhaConfigV2, MhaMmaOp
from nn.attention.gpu.amd_structured.mha_prefill_v2 import MhaPrefillV2


# Per-col group pattern. Returns `c // 8` packed into FP8 e4m3fn.
# Column groups of 8 keep all distinct values exactly representable in FP8
# at d=128: {0, 1, ..., 15}.
@always_inline
def _pattern_fp8(r: Int, c: Int) -> Float8_e4m3fn:
    return Float8_e4m3fn(Float32(c // 8))


def kernel_load_q_fp8[
    cfg: MhaConfigV2,
    depth: Int,
](
    q_ptr: MutPointer[Scalar[cfg.dtype], MutAnyOrigin],
    dump_ptr: MutPointer[Scalar[cfg.dtype], MutAnyOrigin],
):
    """Calls `MhaPrefillV2._load_q_and_scale` and dumps each lane's
    per-base-tile fragment to gmem.

    Layout of `dump_ptr`: `(64 lanes, num_base_tiles, FRAG_ELTS)`.
    For the MHA kernel at `Q_BLOCK_SIZE=32` / `depth=128` / FP8 32x32x64:
    `Q_LAYOUT = [1, 2, 32]` — 1 Q-row base tile, 2 K-dim base tiles,
    32 FP8 elements per lane per base tile. Total `64 * 1 * 2 * 32`
    = 4096 FP8 elements dumped.
    """
    comptime _Op = MhaMmaOp[cfg.dtype, cfg]

    # Construct a TileTensor wrapping `q_ptr`. Shape mirrors the
    # `q_warp_2d` view the production kernel passes in.
    comptime _layout = row_major[cfg.q_block_size, depth]()
    var q_2d = TileTensor[cfg.dtype, type_of(_layout)](q_ptr, _layout)

    # scale_log2e=1.0 is fine — `prescale_q` is comptime-False for FP8,
    # so the multiply branch is DCE'd.
    var q_reg = MhaPrefillV2[cfg]._load_q_and_scale(q_2d, Float32(1.0))

    comptime _H = _Op.Q_LAYOUT.static_shape[0]
    comptime _W = _Op.Q_LAYOUT.static_shape[1]
    comptime _F = _Op.FRAG_ELTS
    comptime _total_per_lane = _H * _W * _F
    var lid = Int(lane_id())
    var q_reg_v = q_reg.vectorize[1, 1, _F]()

    comptime for rr in range(_H):
        comptime for rc in range(_W):
            var frag = q_reg_v[rr, rc, 0]
            comptime for f in range(_F):
                var idx = lid * _total_per_lane + (rr * _W + rc) * _F + f
                dump_ptr[idx] = rebind[Scalar[cfg.dtype]](frag[f])


def test_q_loader_contiguous_lane_layout(ctx: DeviceContext) raises:
    """Asserts that `_load_q_and_scale` produces a contiguous per-lane
    fragment matching the K loader's layout (which the MFMA hardware
    requires for correct B-operand pairing).
    """
    comptime Q_BLOCK_SIZE = 32
    comptime DEPTH = 128
    comptime CFG = MhaConfigV2(
        q_block_size=Q_BLOCK_SIZE,
        kv_block=64,
        depth=DEPTH,
        num_heads=1,
        num_kv_heads=1,
        dtype=DType.float8_e4m3fn,
    )
    comptime _Op = MhaMmaOp[.float8_e4m3fn, CFG]
    comptime _Q_SIZE = Q_BLOCK_SIZE * DEPTH
    comptime _H = _Op.Q_LAYOUT.static_shape[0]
    comptime _W = _Op.Q_LAYOUT.static_shape[1]
    comptime _F = _Op.FRAG_ELTS
    comptime _total_per_lane = _H * _W * _F
    comptime _DUMP_SIZE = 64 * _total_per_lane

    var dev_q = ctx.enqueue_create_buffer[.float8_e4m3fn](_Q_SIZE)
    var dev_dump = ctx.enqueue_create_buffer[.float8_e4m3fn](_DUMP_SIZE)

    with dev_q.map_to_host() as host_q:
        for r in range(Q_BLOCK_SIZE):
            for c in range(DEPTH):
                host_q[r * DEPTH + c] = _pattern_fp8(r, c)

    ctx.enqueue_function[kernel_load_q_fp8[CFG, DEPTH]](
        dev_q.unsafe_ptr(),
        dev_dump.unsafe_ptr(),
        grid_dim=1,
        block_dim=64,
    )
    ctx.synchronize()

    # Expected layout per the K loader contract (which `_load_q_and_scale`
    # must match for the MFMA B-operand fragment to align with A):
    #
    #   For lane `lid`, base-tile `(rr, rc)`, fragment slot `f`:
    #     col_offset = (lid // 32) * (MMA_K // 2)
    #     k_col      = rc * MMA_K + col_offset + f
    #     value      = k_col // 8       (from `_pattern_fp8`)
    #
    # Lane 0, base-tile (0, 0): k_col = 0..31 contiguous, values
    # [0]*8 + [1]*8 + [2]*8 + [3]*8.
    # Lane 32, base-tile (0, 0): k_col = 32..63, values [4..7]*8 each.
    # Lane 0, base-tile (0, 1): k_col = 64..95, values [8..11]*8 each.
    comptime _MMA_K = _Op.MMA_K  # 64
    comptime _HALF_K = _MMA_K // 2  # 32

    with dev_dump.map_to_host() as host_dump:
        # Probe representative lanes: half-warp boundary (0, 32), warp
        # midpoints (1, 33), edges (31, 63). Each lane's full
        # per-base-tile fragment is asserted.
        var probe_lanes = [0, 1, 31, 32, 33, 63]
        for lid_probe in probe_lanes:
            var col_offset = (lid_probe // 32) * _HALF_K
            for rr in range(_H):
                for rc in range(_W):
                    for f in range(_F):
                        var k_col = rc * _MMA_K + col_offset + f
                        var expected = Float8_e4m3fn(Float32(k_col // 8))
                        var idx = (
                            lid_probe * _total_per_lane
                            + (rr * _W + rc) * _F
                            + f
                        )
                        var got = host_dump[idx]
                        assert_equal(
                            got.cast[.float32](),
                            expected.cast[.float32](),
                        )

    _ = dev_q^
    _ = dev_dump^


def main() raises:
    with DeviceContext() as ctx:
        test_q_loader_contiguous_lane_layout(ctx)
