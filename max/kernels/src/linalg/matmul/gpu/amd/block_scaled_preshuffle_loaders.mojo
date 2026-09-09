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
"""Per-lane DRAM->VGPR loaders for the preshuffled block-scaled MoE matmul.

Both loaders consume buffers produced by `block_scaled_preshuffle_layouts` and emit
one `buffer_load_*` per call, no LDS round-trip. Each lane reads exactly the
fragment / scale word the MFMA needs at its `(lane_nlane, lane_klane)` slot.

`PreshuffledBLoader[N, K_BYTES]`:
    Loads one 16-byte FP4 B fragment per lane via `buffer_load_dwordx4`,
    indexed by logical `(n, k_byte)` through `b_5d_layout`.

`PreshuffledScaleLoader[MN_padded, K_SCALES]`:
    Loads one packed Int32 scale word per lane (4 E8M0 bytes covering
    `MNXdlPack=2 x KXdlPack=2` sub-MMAs) via `buffer_load_dword`, indexed
    by logical `(mn, k_scale)` through `scale_4d_layout`.
"""

from max.gpu import lane_id
from max.gpu.intrinsics import AMDBufferResource
from max.gpu.memory import CacheOperation
from std.memory.unsafe import bitcast

from layout import Coord, Idx, TileTensor, DefaultEngine
from layout._utils import make_amd_buffer_resource

from .block_scaled_preshuffle_layouts import Shuffler


struct PreshuffledBLoader[
    N: Int,
    K_BYTES: Int,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
    lane_bytes: Int = 16,
](TrivialRegisterPassable):
    """Per-lane B fragment loader from preshuffled GMEM (DRAM -> VGPR direct).

    The 5D layout places each lane's 16-byte fragment at a contiguous DRAM
    offset, so a single `buffer_load_dwordx4` per lane delivers the MFMA's
    B operand with no LDS staging. OOB lanes are clamped to zero by the
    buffer-resource bounds.

    Parameters:
        N: Per-expert N dimension (rows of the logical [N, K_BYTES] tile).
        K_BYTES: Per-expert FP4-packed K dimension (= K // 2).
        cache_policy: Cache hint for the B load. Defaults to `ALWAYS` (normal
            cached, flydsl `b_nt=0`); set `STREAMING` (NT=1, flydsl `b_nt=2`)
            to skip caching B fragments that are streamed once and never reused.
        lane_bytes: Bytes one lane feeds the MFMA: 16 for FP4, 24 for FP6, 32
            for FP8. Widths above 16, or not a power of two, are split into
            planes (see `Shuffler.b_plane_byte_off`) and loaded with one
            instruction each.
    """

    comptime num_planes = Shuffler[1].num_planes[Self.lane_bytes]()
    comptime reg_bytes = 16 if Self.lane_bytes <= 16 else 32

    var bc: AMDBufferResource

    @always_inline
    def __init__(
        out self,
        b_gmem_tile: TileTensor[.uint8, ...],
    ):
        """Builds the V# from a preshuffled per-expert B byte buffer.

        Args:
            b_gmem_tile: Preshuffled per-expert B byte buffer holding the
                `[N, K_BYTES]` logical tile, as produced by
                `block_scaled_preshuffle_layouts`.
        """
        self.bc = make_amd_buffer_resource(b_gmem_tile)

    @always_inline
    def load_fragment(
        self, n: Int, k_byte: Int
    ) -> SIMD[.uint8, Self.reg_bytes]:
        """Loads one lane's B fragment at logical `(n, k_byte)`.

        For one MFMA dispatch a lane calls this with
        `(n = warp_n_off + n_mma * 16 + lane % 16,
          k_byte = k_tile * MFMA_K_BYTES + (lane // 16) * lane_bytes)`.

        A single-plane fragment is one `buffer_load_dwordx4`, byte-identical to
        the layout this loader has always used. A multi-plane fragment issues
        one naturally-aligned load per plane and assembles them in registers;
        the payload stays contiguous, which is what the MFMA requires.

        Args:
            n: Logical N row index into the `[N, K_BYTES]` tile.
            k_byte: Logical K byte index into the `[N, K_BYTES]` tile.
        """
        var frag = SIMD[.uint8, Self.reg_bytes](0)

        comptime for p in range(Self.num_planes):
            comptime pb = Shuffler[1].plane_bytes[Self.lane_bytes, p]()
            var off = Int32(
                Shuffler[1].b_plane_byte_off[
                    N=Self.N,
                    K_BYTES=Self.K_BYTES,
                    lane_bytes=Self.lane_bytes,
                    plane=p,
                ](0, n, k_byte)
            )
            frag = frag.insert[offset=p * 16](
                self.bc.load[.uint8, pb, cache_policy=Self.cache_policy](off)
            )
        return frag

    @always_inline
    def lane_plane_off(self, n: Int, lane_k_byte: Int) -> Int32:
        """Returns the K-invariant per-lane part of a single-plane address.

        Pair with `load_at`, which supplies the wave-uniform whole-tile part.
        Splitting the address this way lets a caller hoist the per-lane term
        out of an unrolled K loop instead of rematerialising it per tile.

        Args:
            n: Logical N row index into the `[N, K_BYTES]` tile.
            lane_k_byte: The lane's own K byte offset within its K tile,
                i.e. `(lane // 16) * lane_bytes`.
        """
        comptime assert (
            Self.lane_bytes == Shuffler[1].MFMA_LANE_BYTES
        ), "the split address form assumes a single 16-byte plane per lane"
        return Int32(
            Shuffler[1].b_plane_byte_off[
                N=Self.N,
                K_BYTES=Self.K_BYTES,
                lane_bytes=Self.lane_bytes,
                plane=0,
            ](0, n, lane_k_byte)
        )

    @always_inline
    def load_at(
        self, lane_off: Int32, k_byte_uniform: Int
    ) -> SIMD[.uint8, Shuffler[1].MFMA_LANE_BYTES]:
        """Loads one 16-byte plane at `lane_off` plus a wave-uniform K offset.

        `k_byte_uniform` is a whole number of `(n0, k0)` tiles, so it rides
        `soffset` while `lane_off` stays in `voffset`.

        Args:
            lane_off: The per-lane offset from `lane_plane_off`.
            k_byte_uniform: Logical K byte base of the tile, wave-uniform and
                a multiple of the K0 tile width.
        """
        comptime assert (
            Self.lane_bytes == Shuffler[1].MFMA_LANE_BYTES
        ), "the split address form assumes a single 16-byte plane per lane"
        comptime K0_BYTES = Shuffler[1].MFMA_K_LANES * Self.lane_bytes
        comptime TILE_BYTES = Shuffler[1].MFMA_MN_LANES * K0_BYTES
        return self.bc.load[
            .uint8, Shuffler[1].MFMA_LANE_BYTES, cache_policy=Self.cache_policy
        ](
            lane_off,
            scalar_offset=Int32((k_byte_uniform // K0_BYTES) * TILE_BYTES),
        )


struct PreshuffledScaleLoader[MN_padded: Int, K_SCALES: Int](
    TrivialRegisterPassable
):
    """Per-lane packed-Int32 scale loader from preshuffled GMEM.

    Each i32 cell holds 4 E8M0 bytes packed in `(k_pack, mn_pack)` order;
    the MFMA's `opsel` byte index selects the right byte per sub-MMA.
    OOB lanes (past `MN_padded * K_SCALES`) read as zero.

    Parameters:
        MN_padded: MN dimension rounded up to 32 (the scale-block stride).
        K_SCALES: K // 32 (one E8M0 byte per 32 FP4 elements).
    """

    var bc: AMDBufferResource

    @always_inline
    def __init__(
        out self,
        scale_gmem_tile: TileTensor[.uint8, ...],
    ):
        """Builds the V# from a preshuffled per-expert scale byte buffer.

        Args:
            scale_gmem_tile: Preshuffled per-expert scale byte buffer holding
                the `[MN_padded, K_SCALES]` logical grid of E8M0 bytes, as
                produced by `block_scaled_preshuffle_layouts`.
        """
        self.bc = make_amd_buffer_resource(scale_gmem_tile)

    @always_inline
    def load_packed(self, mn: Int, k_scale: Int) -> Int32:
        """Loads the packed Int32 scale word containing logical `(mn, k_scale)`.

        Pass `(mn, k_scale)` at `(mn_pack=0, k_pack=0)` (the cell base)
        and all 4 bytes of the cell come back in the returned i32. The
        MFMA's `opsel` then selects the byte for each sub-MMA.

        Per-lane usage:
            mn       = warp_mn_off + lane % 16            # mn_lane within block
            k_scale  = k_pair_idx * 8 + (lane // 16)      # k_lane within block

        Args:
            mn: Logical MN index into the `[MN_padded, K_SCALES]` scale grid.
            k_scale: Logical K scale index into the `[MN_padded, K_SCALES]`
                scale grid.
        """
        var byte_off = Int32(
            Shuffler[1].scale_4d_byte_off[
                K_SCALES=Self.K_SCALES, packed_mode=True
            ](mn, k_scale)
        )
        var v = self.bc.load[.uint8, 4](byte_off)
        return bitcast[.int32, 1](v)[0]

    @always_inline
    def load_group[
        GROUP: Int
    ](self, mn_base: Int, k_pair_base: Int) -> SIMD[.uint8, GROUP * 4]:
        """Loads `GROUP` consecutive packed-scale atoms with one VMEM op.

        Consecutive `k_pair` atoms are contiguous 256-byte blocks, so one
        `GROUP * 4`-byte load per lane tiles `GROUP` of them across a wave64.

        The window is lane-transposed: lane `l` holds the words `load_packed`
        would give lanes `GROUP*l .. GROUP*l + GROUP - 1`, and the caller must
        undo that.

        Parameters:
            GROUP: Atoms per window; `GROUP * 4` must be a legal load width.

        Args:
            mn_base: Logical MN index of the window's atom row; must be
                16-aligned so the lane term is the whole in-atom offset.
            k_pair_base: First `k_pair` of the window.
        """
        comptime assert GROUP == 4, "scale group load must be one dwordx4"
        var byte_off = Int32(
            Shuffler[1].scale_4d_byte_off[
                K_SCALES=Self.K_SCALES, packed_mode=True
            ](mn_base, k_pair_base * Shuffler[1].S_K_BLOCK)
            + Int(lane_id()) * (GROUP * 4)
        )
        return self.bc.load[.uint8, GROUP * 4](byte_off)
