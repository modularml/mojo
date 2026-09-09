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

from max.gpu.host import DeviceContext
from std.math import align_up
from std.testing import assert_equal

from layout import Coord, Idx, TileTensor, row_major

from linalg.matmul.gpu.amd import Shuffler


def test_preshuffle_b_round_trip[
    N: Int, K_BYTES: Int
](ctx: DeviceContext) raises:
    var src_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)
    var dst_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)

    var src_db = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    var dst_db = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    ctx.synchronize()

    for i in range(N * K_BYTES):
        src_hb[i] = UInt8(i & 0xFF)

    ctx.enqueue_copy(src_db, src_hb)

    var src_tt = TileTensor(
        src_db, row_major(Coord(Idx[1], Idx[N], Idx[K_BYTES]))
    )

    var dst_tt = TileTensor(
        dst_db, Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES]
    )

    Shuffler[1].preshuffle_b_5d[N=N, K_BYTES=K_BYTES](src_tt, dst_tt, ctx)

    ctx.enqueue_copy(dst_hb, dst_db)
    ctx.synchronize()

    _ = dst_tt
    # Inverse re-index: the output byte at b_5d_grouped_layout(0, n, k_byte)
    # must equal the input byte at [n, k_byte] (row-major).
    for n in range(N):
        for k_byte in range(K_BYTES):
            var src_idx = n * K_BYTES + k_byte
            var dst_idx = Int(
                Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
                    Coord(Idx[0], n, k_byte)
                )
            )
            assert_equal(dst_hb[dst_idx], src_hb[src_idx])

    # Permutation, not duplication: every output position written exactly once.
    var seen_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)
    ctx.synchronize()
    for i in range(N * K_BYTES):
        seen_hb[i] = UInt8(0)
    for n in range(N):
        for k_byte in range(K_BYTES):
            var dst_idx = Int(
                Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
                    Coord(Idx[0], n, k_byte)
                )
            )
            assert_equal(seen_hb[dst_idx], UInt8(0))
            seen_hb[dst_idx] = UInt8(1)
    for i in range(N * K_BYTES):
        assert_equal(seen_hb[i], UInt8(1))


def test_preshuffle_b_planes_round_trip[
    N: Int, K_BYTES: Int, lane_bytes: Int
](ctx: DeviceContext) raises:
    """Round-trips the plane-split preshuffle for a non-power-of-two fragment.

    The write kernel and the read path (`b_plane_byte_off`, used by
    `PreshuffledBLoader`) must agree exactly: a permutation kernel that
    disagrees with its reader produces wrong weights, not a crash. So this
    checks the same two properties as the 16-byte round trip -- every source
    byte lands where the reader will look for it, and every destination byte
    is written exactly once.
    """
    var src_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)
    var dst_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)
    var src_db = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    var dst_db = ctx.enqueue_create_buffer[.uint8](N * K_BYTES)
    ctx.synchronize()

    for i in range(N * K_BYTES):
        src_hb[i] = UInt8((i * 31 + 7) & 0xFF)
    ctx.enqueue_copy(src_db, src_hb)

    var src_tt = TileTensor(
        src_db, row_major(Coord(Idx[1], Idx[N], Idx[K_BYTES]))
    )
    var dst_tt = TileTensor(
        dst_db, row_major(Coord(Idx[1], Idx[N], Idx[K_BYTES]))
    )
    Shuffler[1].preshuffle_b_planes[
        N=N, K_BYTES=K_BYTES, lane_bytes=lane_bytes
    ](src_tt, dst_tt, ctx)

    ctx.enqueue_copy(dst_hb, dst_db)
    ctx.synchronize()

    comptime num_planes = Shuffler[1].num_planes[lane_bytes]()
    var seen_hb = ctx.enqueue_create_host_buffer[.uint8](N * K_BYTES)
    ctx.synchronize()
    for i in range(N * K_BYTES):
        seen_hb[i] = UInt8(0)

    for n in range(N):
        for frag in range(K_BYTES // lane_bytes):
            var k_byte = frag * lane_bytes
            comptime for pl in range(num_planes):
                comptime pb = Shuffler[1].plane_bytes[lane_bytes, pl]()
                var dst_base = Shuffler[1].b_plane_byte_off[
                    N=N, K_BYTES=K_BYTES, lane_bytes=lane_bytes, plane=pl
                ](0, n, k_byte)
                for b in range(pb):
                    var src_idx = n * K_BYTES + k_byte + pl * 16 + b
                    assert_equal(dst_hb[dst_base + b], src_hb[src_idx])
                    assert_equal(seen_hb[dst_base + b], UInt8(0))
                    seen_hb[dst_base + b] = UInt8(1)

    for i in range(N * K_BYTES):
        assert_equal(seen_hb[i], UInt8(1))


def test_preshuffle_scale_round_trip[
    MN: Int, K_SCALES: Int
](ctx: DeviceContext) raises:
    comptime MN_padded = Shuffler[1].scale_padded_mn(MN)
    var src_hb = ctx.enqueue_create_host_buffer[.uint8](MN * K_SCALES)
    var dst_hb = ctx.enqueue_create_host_buffer[.uint8](MN_padded * K_SCALES)
    ctx.synchronize()

    for i in range(MN * K_SCALES):
        src_hb[i] = UInt8(i & 0xFF)

    var src_tt = TileTensor(
        src_hb, row_major(Coord(Idx[1], Idx[MN], Idx[K_SCALES]))
    )
    Shuffler[1].preshuffle_scale_4d[MN=MN, K_SCALES=K_SCALES](src_tt, dst_hb)

    # Valid rows: each input byte appears at the byte-offset-computed dst.
    for mn in range(MN):
        for k_scale in range(K_SCALES):
            var src_idx = mn * K_SCALES + k_scale
            var dst_idx = Shuffler[1].scale_4d_byte_off[K_SCALES=K_SCALES](
                mn, k_scale
            )
            assert_equal(dst_hb[dst_idx], src_hb[src_idx])

    # Pad rows (mn in [MN, MN_padded)) must be zero.
    for mn in range(MN, MN_padded):
        for k_scale in range(K_SCALES):
            var dst_idx = Shuffler[1].scale_4d_byte_off[K_SCALES=K_SCALES](
                mn, k_scale
            )
            assert_equal(dst_hb[dst_idx], UInt8(0))


def _b_offset[N: Int, K_BYTES: Int](n: Int, k_byte: Int) -> Int:
    return Int(
        Shuffler[1].b_5d_grouped_layout[N=N, K_BYTES=K_BYTES](
            Coord(Idx[0], n, k_byte)
        )
    )


def _scale_offset[MN_padded: Int, K_SCALES: Int](mn: Int, k_scale: Int) -> Int:
    return Shuffler[1].scale_4d_byte_off[K_SCALES=K_SCALES](mn, k_scale)


def test_preshuffle_b_offset_modes() raises:
    # Spot-check the 5 layout modes (n0, k0, klane, nlane, kpack) decode
    # to the strides documented in the file header.
    # Strides (bytes): n0=K0*1024, k0=1024, klane=256, nlane=16, kpack=1
    # for the standard 16x16x128 MFMA layout. Verify via incremental deltas.
    comptime N = 32
    comptime K_BYTES = 128  # K0 = 2

    # Same n, same k0: incrementing k_byte within a (klane, nlane, kpack) cell
    # walks contiguous output bytes.
    assert_equal(_b_offset[N, K_BYTES](0, 1), 1)
    assert_equal(_b_offset[N, K_BYTES](0, 15), 15)

    # Crossing kpack=16 boundary while staying in same (klane, n0) bumps nlane.
    # k_byte=16 -> tempk=16 -> klane=1, kpack=0 -> offset = klane*256 = 256.
    assert_equal(_b_offset[N, K_BYTES](0, 16), 256)

    # Crossing klane*kpack=64 boundary bumps k0.
    # k_byte=64 -> k0=1, tempk=0 -> offset = k0*1024 = 1024.
    assert_equal(_b_offset[N, K_BYTES](0, 64), 1024)

    # Crossing nlane=16 boundary with same (k0,klane,kpack) bumps n0.
    # n=15 vs n=16 with k_byte=0: n=15 has nlane=15 -> offset 240;
    # n=16 has n0=1 -> offset = n0 * K0 * 1024 = 2048.
    assert_equal(_b_offset[N, K_BYTES](15, 0), 15 * 16)
    assert_equal(_b_offset[N, K_BYTES](16, 0), 2 * 1024)


def test_preshuffle_scale_offset_modes() raises:
    # Strides: n0=256*K0, k0=256, k_lane=64, mn_lane=4, k_pack=2, mn_pack=1
    # for MNXdlPack=KXdlPack=2, XdlMNThread=16, XdlKThread=4.
    comptime MN_padded = 64
    comptime K_SCALES = 16  # K0 = 2

    # mn_pack=0 mn_lane=0 k_lane=0 k_pack=0 k0=0 n0=0 -> 0.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 0), 0)

    # mn=1: tempn=1, mn_lane=1, mn_pack=0 -> mn_lane*4 + mn_pack = 4.
    assert_equal(_scale_offset[MN_padded, K_SCALES](1, 0), 4)

    # mn=16: tempn=16, mn_lane=0, mn_pack=1 -> mn_pack = 1.
    assert_equal(_scale_offset[MN_padded, K_SCALES](16, 0), 1)

    # mn=32: n0=1, others 0 -> n0 * 256 * K0 = 512.
    assert_equal(_scale_offset[MN_padded, K_SCALES](32, 0), 512)

    # k_scale=1: tempk=1, k_lane=1, k_pack=0 -> k_lane*64 = 64.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 1), 64)

    # k_scale=4: tempk=4, k_lane=0, k_pack=1 -> k_pack*2 = 2.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 4), 2)

    # k_scale=8: k0=1 -> k0 * 256 = 256.
    assert_equal(_scale_offset[MN_padded, K_SCALES](0, 8), 256)


def test_preshuffle_grouped_scale_gpu[
    K_SCALES: Int,
](
    name: String,
    num_tokens_by_expert: List[Int],
    max_num_tokens_per_expert: Int,
    ctx: DeviceContext,
) raises:
    """Validates `Shuffler[1].preshuffle_grouped_scale_4d_gpu`.

    Caller supplies `max_num_tokens_per_expert` as a fixed upper bound
    (mirrors production — the model config sets this, the test never
    derives it from the runtime token list). The slot stride is
    `align_up(max_num_tokens_per_expert, 32)` regardless of the actual
    per-expert counts; any expert with fewer tokens has the slot tail
    zero-filled.
    """
    var num_active = len(num_tokens_by_expert)
    var total_tokens = 0
    var max_tokens = 0
    for ne in num_tokens_by_expert:
        total_tokens += ne
        max_tokens = max(max_tokens, ne)
    if max_tokens > max_num_tokens_per_expert:
        raise Error(
            "test config bug: some expert has more tokens than"
            " max_num_tokens_per_expert"
        )

    var max_padded_M = align_up(
        max_num_tokens_per_expert, Shuffler[1].S_MN_BLOCK
    )
    var slot_bytes = max_padded_M * K_SCALES

    print(
        "  ",
        name,
        " active=",
        num_active,
        " total_tokens=",
        total_tokens,
        " max_padded_M=",
        max_padded_M,
        " K_SCALES=",
        K_SCALES,
    )

    # ---- Host buffers + random-like init ----
    var src_hb = ctx.enqueue_create_host_buffer[.uint8](total_tokens * K_SCALES)
    var a_off_hb = ctx.enqueue_create_host_buffer[.uint32](num_active + 1)
    var dst_hb = ctx.enqueue_create_host_buffer[.uint8](num_active * slot_bytes)
    ctx.synchronize()

    # Deterministic per-token-per-scale fingerprint: byte = (token_idx * 7
    # + k_scale * 13) & 0xFF. Lets us point-check after the shuffle.
    for t in range(total_tokens):
        for k in range(K_SCALES):
            src_hb[t * K_SCALES + k] = UInt8((t * 7 + k * 13) & 0xFF)

    a_off_hb[0] = UInt32(0)
    for e in range(num_active):
        a_off_hb[e + 1] = a_off_hb[e] + UInt32(num_tokens_by_expert[e])

    # Pre-fill dst with a 0xAB sentinel. The persistent kernel's contract
    # is to write exactly `align_up(num_tokens[e], 32) * K_SCALES` bytes
    # per expert — matching the matmul's per-expert V# bound. Higher
    # m_blocks past `ceildiv(num_tokens, S_MN_BLOCK)` are NOT touched
    # and must retain the sentinel. The matmul's tight V# clamps reads
    # past `align_up(num_tokens, 32) * K_SCALES` to 0 in hardware, so
    # the sentinel bytes are never observed in production.
    for i in range(num_active * slot_bytes):
        dst_hb[i] = UInt8(0xAB)

    # ---- Device buffers + upload ----
    var src_db = ctx.enqueue_create_buffer[.uint8](total_tokens * K_SCALES)
    var a_off_db = ctx.enqueue_create_buffer[.uint32](num_active + 1)
    var dst_db = ctx.enqueue_create_buffer[.uint8](num_active * slot_bytes)
    ctx.enqueue_copy(src_db, src_hb)
    ctx.enqueue_copy(a_off_db, a_off_hb)
    ctx.enqueue_copy(dst_db, dst_hb)

    var src_tt = TileTensor[mut=False, ...](
        src_db, row_major(Coord(total_tokens, Idx[K_SCALES]))
    )
    var dst_tt = TileTensor[mut=True, ...](
        dst_db, row_major(Coord(num_active * max_padded_M, Idx[K_SCALES]))
    )
    var a_off_tt = TileTensor[mut=False, ...](
        a_off_db, row_major(Coord(num_active + 1))
    )

    # Persistent grid: pick a small total_wg so the test exercises the
    # grid-stride loop (some CTAs walk multiple tiles).
    var total_wg = 64
    Shuffler[1].preshuffle_grouped_scale_4d_gpu[K_SCALES=K_SCALES](
        src_tt,
        dst_tt,
        a_off_tt,
        num_active,
        max_num_tokens_per_expert,
        total_wg,
        ctx,
    )

    ctx.enqueue_copy(dst_hb, dst_db)
    ctx.synchronize()

    # ---- Validate per-byte (per kernel contract above) ----
    for e in range(num_active):
        var token_start = Int(a_off_hb[e])
        var num_tokens = Int(a_off_hb[e + 1]) - token_start
        var slot_base = e * slot_bytes
        var v_pad_M = align_up(num_tokens, Shuffler[1].S_MN_BLOCK)
        # Real rows: preshuffled bytes equal the source bytes.
        for mn in range(num_tokens):
            for k in range(K_SCALES):
                var src_idx = (token_start + mn) * K_SCALES + k
                var dst_idx = slot_base + Shuffler[1].scale_4d_byte_off[
                    K_SCALES=K_SCALES
                ](mn, k)
                assert_equal(dst_hb[dst_idx], src_hb[src_idx])
        # In-V# pad rows: kernel writes 0 from the cell_bytes init.
        for mn in range(num_tokens, v_pad_M):
            for k in range(K_SCALES):
                var dst_idx = slot_base + Shuffler[1].scale_4d_byte_off[
                    K_SCALES=K_SCALES
                ](mn, k)
                assert_equal(dst_hb[dst_idx], UInt8(0))
        # Higher m_blocks past V#: untouched, sentinel intact.
        for mn in range(v_pad_M, max_padded_M):
            for k in range(K_SCALES):
                var dst_idx = slot_base + Shuffler[1].scale_4d_byte_off[
                    K_SCALES=K_SCALES
                ](mn, k)
                assert_equal(dst_hb[dst_idx], UInt8(0xAB))

    print("    PASS")


def test_preshuffle_scale_padding() raises:
    # Padded MN rounds up to multiple of 32.
    assert_equal(Shuffler[1].scale_padded_mn(1), 32)
    assert_equal(Shuffler[1].scale_padded_mn(31), 32)
    assert_equal(Shuffler[1].scale_padded_mn(32), 32)
    assert_equal(Shuffler[1].scale_padded_mn(33), 64)
    assert_equal(Shuffler[1].scale_padded_mn(127), 128)
    assert_equal(Shuffler[1].scale_padded_mn(128), 128)


def main() raises:
    var ctx = DeviceContext()

    # Stride/offset spot-checks (cheap, run first).
    test_preshuffle_b_offset_modes()
    test_preshuffle_scale_offset_modes()
    test_preshuffle_scale_padding()

    # Round-trip across realistic shapes: tile_n=128, tile_k=256 (Kimi K2-class),
    # plus a small shape and a non-padded scale shape.
    test_preshuffle_b_round_trip[N=16, K_BYTES=64](ctx)
    test_preshuffle_b_round_trip[N=32, K_BYTES=128](ctx)
    test_preshuffle_b_round_trip[N=128, K_BYTES=256](ctx)

    # Plane-split (FP6): 24-byte fragments, K_BYTES a multiple of 96.
    test_preshuffle_b_planes_round_trip[16, 96, 24](ctx)
    test_preshuffle_b_planes_round_trip[32, 192, 24](ctx)
    test_preshuffle_b_planes_round_trip[64, 384, 24](ctx)
    # A 16-byte fragment must still round-trip through the same path.
    test_preshuffle_b_planes_round_trip[32, 128, 16](ctx)

    test_preshuffle_scale_round_trip[MN=32, K_SCALES=8](ctx)
    test_preshuffle_scale_round_trip[MN=64, K_SCALES=16](ctx)
    test_preshuffle_scale_round_trip[MN=128, K_SCALES=8](ctx)
    # Non-32-aligned MN exercises pad-row zero-fill.
    test_preshuffle_scale_round_trip[MN=17, K_SCALES=8](ctx)
    test_preshuffle_scale_round_trip[MN=33, K_SCALES=16](ctx)

    # GPU grouped scale preshuffle. `max_num_tokens_per_expert` is the
    # fixed model-config bound, NOT the runtime max — mirrors production
    # where the allocator sizes for worst case without inspecting the
    # routing.
    test_preshuffle_grouped_scale_gpu[K_SCALES=8]("single-tiny", [3], 16, ctx)
    test_preshuffle_grouped_scale_gpu[K_SCALES=16]("single-mid", [64], 64, ctx)
    test_preshuffle_grouped_scale_gpu[K_SCALES=16](
        "multi-mixed", [16, 32, 7, 48], 64, ctx
    )
    test_preshuffle_grouped_scale_gpu[K_SCALES=8](
        "inactive-M0", [16, 0, 32, 8], 64, ctx
    )
    # Decode-like: many experts with tiny token counts.
    test_preshuffle_grouped_scale_gpu[K_SCALES=8](
        "decode-49",
        [3, 3, 3, 2, 2, 2, 1, 1, 1, 3, 0, 2, 3, 1, 2],
        16,
        ctx,
    )
    # Larger K_SCALES (Kimi-like) with mid-range token counts.
    test_preshuffle_grouped_scale_gpu[K_SCALES=64](
        "kimi-shape", [40, 40, 40, 24], 64, ctx
    )
    # Stress the slot-tail zero-fill: small actual tokens, large config bound.
    test_preshuffle_grouped_scale_gpu[K_SCALES=16](
        "config-much-larger", [3, 7, 1, 0, 5], 128, ctx
    )
