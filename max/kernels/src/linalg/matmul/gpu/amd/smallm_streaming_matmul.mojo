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
"""Small-M weight-streaming matmul over a preshuffled B for MI355X.

Serves decode-band vocab-class projections (huge N, K in the thousands,
M <= 128 — lm_head shards and MTP eh_proj): the GEMM is >99% weight reads, so
the kernel is built to stream B exactly once at HBM rate and spend nothing
else. Each 16-column tile of C is owned by one block; its warps split K and
run mfma 16x16x32 bf16 with the B fragment loaded straight from global memory.

B must be preshuffled into the fragment-major layout produced by
``smallm_preshuffle_b`` (run once at weight-load time): a plain 16B load per
lane then lands a full warp instruction on 1KB of contiguous memory. Reading a
row-major B here is silently wrong — the layout is private to this pair, so
the op must be invoked explicitly, never via generic matmul dispatch.

A stays row-major ``[m, k]`` (contiguous, row stride == K) and is small enough
to be L2-resident. C is ``[m, n]``.
"""

from std.math import ceildiv
from max.gpu import (
    WARP_SIZE,
    block_idx,
    global_idx,
    grid_dim,
    lane_id,
    thread_idx,
)
from std.memory import unsafe_stack_allocation
from std.utils import Index

from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from layout import TensorLayout, TensorEngine, TileTensor
from linalg.utils import elementwise_epilogue_type

comptime _MFMA_K = 32
comptime _TILE_N = 16


@__name(t"smallm_preshuffle_b_{b_type}_w{warps_per_block}_k{k_static}")
def _smallm_preshuffle_b_kernel[
    b_type: DType,
    *,
    k_static: Int,
    warps_per_block: Int,
](
    dst: UnsafePointer[Scalar[b_type], MutAnyOrigin],
    src: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
    n: Int32,
):
    """Permutes a row-major ``[n, k]`` weight into fragment-major order.

    Destination order: 16-column tile -> warp K-slice -> per-mfma 512-element
    chunk -> 16B per lane, matching exactly what
    ``_gemm_smallm_streaming_kernel`` reads. Both layouts keep every 8-element
    (16B) vector contiguous, so this is a pure vector permutation.
    """
    comptime k_per_warp = k_static // warps_per_block
    comptime iters = k_per_warp // _MFMA_K
    comptime region = iters * WARP_SIZE * 8

    var vec = Int(global_idx.x)
    var vecs_per_row = k_static // 8
    if vec >= Int(n) * vecs_per_row:
        return

    var col = vec // vecs_per_row
    var k = (vec % vecs_per_row) * 8

    var warp = k // k_per_warp
    var k_in = k % k_per_warp
    var i = k_in // _MFMA_K
    var g = (k_in // 8) % 4
    var l = g * _TILE_N + col % _TILE_N

    var dst_off = (
        (col // _TILE_N * warps_per_block + warp) * region
        + i * (WARP_SIZE * 8)
        + l * 8
    )
    dst.store(dst_off, src.load[width=8](col * k_static + k))


@__name(t"smallm_shuffle_a_{a_type}_t{m_tiles}_w{warps_per_block}_k{k_static}")
def _smallm_shuffle_a_kernel[
    a_type: DType,
    a_layout: TensorLayout,
    a_engine: TensorEngine,
    *,
    k_static: Int,
    warps_per_block: Int,
    m_tiles: Int,
](
    dst: UnsafePointer[Scalar[a_type], MutAnyOrigin],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    m: Int32,
):
    """Permutes the activation ``[m, k]`` into fragment-major order per call.

    Same fragment geometry as the weight preshuffle, over ``m_tiles * 16``
    rows with rows past ``m`` clamped to the last live row (their products
    land only in store-guarded C rows). A is at most ``128 x k`` (~1.5MB), so
    this is a microsecond-scale prologue — the payoff is that the main
    kernel's A loads become as contiguous as its B loads, instead of
    scattering one cache line per fragment row every chunk.
    """
    comptime k_per_warp = k_static // warps_per_block
    comptime iters = k_per_warp // _MFMA_K
    comptime region = iters * WARP_SIZE * 8
    comptime rows = m_tiles * _TILE_N

    var vec = Int(global_idx.x)
    var vecs_per_row = k_static // 8
    if vec >= rows * vecs_per_row:
        return

    var row = vec // vecs_per_row
    var k = (vec % vecs_per_row) * 8

    var warp = k // k_per_warp
    var k_in = k % k_per_warp
    var i = k_in // _MFMA_K
    var g = (k_in // 8) % 4
    var l = g * _TILE_N + row % _TILE_N

    var dst_off = (
        (row // _TILE_N * warps_per_block + warp) * region
        + i * (WARP_SIZE * 8)
        + l * 8
    )
    var src_row = min(row, Int(m) - 1)
    dst.store(dst_off, a.ptr.load[width=8](src_row * k_static + k))


@__name(
    t"gemm_smallm_streaming_{c_type}_{a_type}_{b_type}_t{m_tiles}_k{k_static}"
)
def _gemm_smallm_streaming_kernel[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    c_engine: TensorEngine,
    *,
    k_static: Int,
    warps_per_block: Int,
    m_tiles: Int,
    a_in_regs: Bool,
    col_tiles: Int = 1,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],  # m * n
    a_shuffled: UnsafePointer[Scalar[a_type], ImmutAnyOrigin],
    b_shuffled: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
    m: Int32,
    n: Int32,
):
    """One warp per (16-column tile, K-slice): stream B once, mfma per m-tile.

    The B fragment for a k-chunk is shared across all ``m_tiles`` row tiles,
    so B traffic is independent of M. Both operands arrive fragment-major (B
    from the load-time weight preshuffle, A from the per-call shuffle
    prologue), so every load is one 16B vector per lane, 1KB contiguous per
    warp. With ``a_in_regs`` (single row tile, shallow K), the warp's whole A
    slice is preloaded into registers and the hot loop is a pure B-stream.
    Each block reduces its warps' K-partials through an LDS scratch with one
    bank-padded 4-float slot per (lane, tile).
    """
    comptime assert (
        a_type == .bfloat16 and b_type == .bfloat16
    ), "the mfma 16x16x32 path is bf16-only"
    comptime k_per_warp = k_static // warps_per_block
    comptime assert (
        k_per_warp % _MFMA_K == 0
    ), "K must divide evenly across the block's warps in mfma-K steps"
    comptime iters = k_per_warp // _MFMA_K
    comptime assert (
        not a_in_regs
    ) or m_tiles == 1, "register-resident A only supports a single row tile"
    comptime assert (
        not a_in_regs
    ) or col_tiles == 1, "the column-loop path amortizes A without col_tiles"
    comptime assert c.flat_rank == 2, "c must be of rank 2"
    comptime region = iters * WARP_SIZE * 8

    var _m = Int(m)
    var _n = Int(n)
    var lane = Int(lane_id())
    var warp_in_block = Int(thread_idx.x) // WARP_SIZE

    # mfma 16x16x32 bf16 lane layout (wave64): lane l serves row/col l%16 and
    # k-group l//16 (4 groups of 8 contiguous k). Fragment-major operands make
    # each fragment a single 16B load at a lane-linear offset.
    var frag_row = lane % _TILE_N
    var a_base = a_shuffled + warp_in_block * region + lane * 8

    # A loads once per block, amortized over the column loop below.
    var a_frags = Array[SIMD[a_type, 8], iters if a_in_regs else 1](fill=0)
    comptime if a_in_regs:
        comptime for i in range(iters):
            a_frags[i] = a_base.load[width=8](i * WARP_SIZE * 8)

    # Pad each lane's reduce slot by one 4-word vector so the per-lane
    # stride is an odd multiple of 4 words: every ds_read_b128 phase of the
    # warp-0 reduction then hits all 64 LDS banks conflict-free.
    comptime lane_slot = (m_tiles * col_tiles + 1) * 4
    var c_smem = unsafe_stack_allocation[
        warps_per_block * WARP_SIZE * lane_slot,
        DType.float32,
        address_space=.SHARED,
    ]()

    # The runtime column loop exists only where its iterations reuse the
    # register-resident A. A runtime loop around the multi-tile body makes
    # the register allocator clamp VGPRs and spill, so per-chunk-A
    # instantiations get a structurally loop-free body, one block per tile.
    var n_tiles = ceildiv(_n, _TILE_N)
    comptime if a_in_regs:
        for ct in range(Int(block_idx.x), n_tiles, Int(grid_dim.x)):
            var b_ptr = (
                b_shuffled
                + (ct * warps_per_block + warp_in_block) * region
                + lane * 8
            )
            var acc = Array[SIMD[.float32, 4], m_tiles](fill=0)

            # Depth-2 B pipeline: the next chunk's load issues before this
            # chunk's mfma work. Deeper pipelining measures slower here.
            var b_next = b_ptr.load[width=8, non_temporal=True](0)
            comptime if a_in_regs:
                comptime for i in range(iters):
                    var b_cur = b_next
                    comptime if i + 1 < iters:
                        b_next = b_ptr.load[width=8, non_temporal=True](
                            (i + 1) * WARP_SIZE * 8
                        )
                    mma(acc[0], a_frags[i], b_cur, acc[0])
            else:
                comptime for i in range(iters):
                    var b_cur = b_next
                    comptime if i + 1 < iters:
                        b_next = b_ptr.load[width=8, non_temporal=True](
                            (i + 1) * WARP_SIZE * 8
                        )
                    comptime for t in range(m_tiles):
                        var a_frag = a_base.load[width=8](
                            (t * warps_per_block) * region + i * WARP_SIZE * 8
                        )
                        mma(acc[t], a_frag, b_cur, acc[t])

            # Reduce the warps' K-partials: fragments to LDS, warp 0 sums.
            if warp_in_block > 0:
                comptime for t in range(m_tiles):
                    c_smem.store(
                        (warp_in_block * WARP_SIZE + lane) * lane_slot + t * 4,
                        acc[t],
                    )
            barrier()

            if warp_in_block == 0:
                comptime for w in range(1, warps_per_block):
                    comptime for t in range(m_tiles):
                        acc[t] += c_smem.load[width=4](
                            (w * WARP_SIZE + lane) * lane_slot + t * 4
                        )

                # D fragment: lane l holds rows (l//16)*4..+3 of column l%16.
                var out_col = ct * _TILE_N + frag_row
                if out_col < _n:
                    comptime for t in range(m_tiles):
                        comptime for r4 in range(4):
                            var row = t * _TILE_N + (lane // _TILE_N) * 4 + r4
                            if row < _m:
                                comptime if elementwise_lambda_fn:
                                    comptime elementwise_lambda = (
                                        elementwise_lambda_fn.value()
                                    )
                                    elementwise_lambda(
                                        Index(row, out_col),
                                        acc[t][r4].cast[c_type](),
                                    )
                                else:
                                    c[row, out_col] = acc[t][r4].cast[c_type]()
            # The LDS scratch is reused by the next column tile.
            barrier()
    else:
        var ct0 = Int(block_idx.x) * col_tiles
        # Read offsets clamp to the last real tile so the tail block's loads
        # stay in bounds; its stores are culled by the out_col guard below.
        var b_offs = Array[Int, col_tiles](fill=0)
        comptime for u in range(col_tiles):
            b_offs[u] = (
                min(ct0 + u, n_tiles - 1) * warps_per_block + warp_in_block
            ) * region + lane * 8
        var acc = Array[SIMD[.float32, 4], m_tiles * col_tiles](fill=0)

        # Depth-2 B pipeline: the next chunk's load issues before this
        # chunk's mfma work. Each A fragment loads once per k-chunk and
        # feeds all col_tiles column tiles.
        var b_next = Array[SIMD[b_type, 8], col_tiles](fill=0)
        comptime for u in range(col_tiles):
            b_next[u] = b_shuffled.load[width=8, non_temporal=True](b_offs[u])
        comptime for i in range(iters):
            var a_frag = Array[SIMD[a_type, 8], m_tiles](fill=0)
            comptime for t in range(m_tiles):
                a_frag[t] = a_base.load[width=8](
                    (t * warps_per_block) * region + i * WARP_SIZE * 8
                )
            comptime for u in range(col_tiles):
                var b_cur = b_next[u]
                comptime if i + 1 < iters:
                    b_next[u] = b_shuffled.load[width=8, non_temporal=True](
                        b_offs[u] + (i + 1) * WARP_SIZE * 8
                    )
                comptime for t in range(m_tiles):
                    mma(
                        acc[u * m_tiles + t],
                        a_frag[t],
                        b_cur,
                        acc[u * m_tiles + t],
                    )

        # Reduce the warps' K-partials: fragments to LDS, then warp 0 sums.
        if warp_in_block > 0:
            comptime for ut in range(m_tiles * col_tiles):
                c_smem.store(
                    (
                        (warp_in_block * WARP_SIZE + lane) * m_tiles * col_tiles
                        + ut
                    )
                    * 4,
                    acc[ut],
                )
        barrier()

        if warp_in_block == 0:
            comptime for w in range(1, warps_per_block):
                comptime for ut in range(m_tiles * col_tiles):
                    acc[ut] += c_smem.load[width=4](
                        ((w * WARP_SIZE + lane) * m_tiles * col_tiles + ut) * 4
                    )

            # D fragment: lane l holds rows (l//16)*4..+3 of column l%16.
            comptime for u in range(col_tiles):
                var out_col = (ct0 + u) * _TILE_N + frag_row
                if out_col < _n:
                    comptime for t in range(m_tiles):
                        comptime for r4 in range(4):
                            var row = t * _TILE_N + (lane // _TILE_N) * 4 + r4
                            if row < _m:
                                comptime if elementwise_lambda_fn:
                                    comptime elementwise_lambda = (
                                        elementwise_lambda_fn.value()
                                    )
                                    elementwise_lambda(
                                        Index(row, out_col),
                                        acc[u * m_tiles + t][r4].cast[c_type](),
                                    )
                                else:
                                    c[row, out_col] = acc[u * m_tiles + t][
                                        r4
                                    ].cast[c_type]()


def smallm_preshuffle_b[
    b_type: DType,
    *,
    k_static: Int,
    warps_per_block: Int = 8,
](
    dst: UnsafePointer[Scalar[b_type], MutAnyOrigin],
    src: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Launches the one-shot weight preshuffle (run at weight-load time).

    Constraints:
        ``n % 16 == 0`` and ``k_static % (warps_per_block * 32) == 0``; both
        also bind the paired matmul.
    """
    if n % _TILE_N != 0:
        raise Error("smallm_preshuffle_b requires n to be a multiple of 16")
    comptime kernel = _smallm_preshuffle_b_kernel[
        b_type, k_static=k_static, warps_per_block=warps_per_block
    ]
    ctx.enqueue_function[kernel](
        dst,
        src,
        Int32(n),
        grid_dim=ceildiv(n * (k_static // 8), 256),
        block_dim=256,
    )


def smallm_streaming_matmul[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    c_engine: TensorEngine,
    a_engine: TensorEngine,
    *,
    k_static: Int,
    warps_per_block: Int = 8,
    max_grid_blocks: Int = 512,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    a: TileTensor[a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
    b_shuffled: UnsafePointer[Scalar[b_type], ImmutAnyOrigin],
    a_scratch: UnsafePointer[Scalar[a_type], MutAnyOrigin],
    m: Int,
    n: Int,
    ctx: DeviceContext,
) raises:
    """Runs the streaming matmul over a ``smallm_preshuffle_b``-layout weight.

    Picks the row-tile instantiation from runtime ``m`` (16 rows per mfma
    tile, up to 128), shuffling the activation into fragment-major order in
    ``a_scratch`` as a microsecond prologue. ``a`` must be contiguous
    row-major ``[m, k_static]``. ``a_scratch`` must hold at least
    ``ceildiv(m, 16) * 16 * k_static`` elements and must be CALLER-owned,
    graph-managed memory: a transient buffer allocated here bakes a stale
    pointer into device-graph-captured launches (replays then read recycled
    memory — silently wrong logits in serving; found the hard way).
    """
    if n % _TILE_N != 0:
        raise Error("smallm_streaming_matmul requires n to be a multiple of 16")
    if m < 1 or m > 128:
        raise Error("smallm_streaming_matmul serves 1 <= m <= 128")

    @__parameter
    def _launch[m_tiles: Int, a_in_regs: Bool, col_tiles: Int = 1]() raises:
        comptime shuffle_kernel = _smallm_shuffle_a_kernel[
            a_type,
            a_layout,
            a_engine,
            k_static=k_static,
            warps_per_block=warps_per_block,
            m_tiles=m_tiles,
        ]
        ctx.enqueue_function[shuffle_kernel](
            a_scratch,
            a,
            Int32(m),
            grid_dim=ceildiv(m_tiles * _TILE_N * (k_static // 8), 256),
            block_dim=256,
        )

        comptime kernel = _gemm_smallm_streaming_kernel[
            c_type,
            a_type,
            b_type,
            c_layout,
            c_engine,
            k_static=k_static,
            warps_per_block=warps_per_block,
            m_tiles=m_tiles,
            a_in_regs=a_in_regs,
            col_tiles=col_tiles,
            elementwise_lambda_fn=elementwise_lambda_fn,
        ]
        var a_src = UnsafePointer[Scalar[a_type], ImmutAnyOrigin](
            unsafe_from_address=Int(a_scratch)
        )
        ctx.enqueue_function[kernel](
            c,
            a_src,
            b_shuffled,
            Int32(m),
            Int32(n),
            # Looped grid only on the register-resident-A instantiation.
            # The cap stays a raw min(): an exactly tile-balanced grid
            # measures slower than an XCD-aligned one with stragglers.
            grid_dim=(
                min(
                    ceildiv(n, _TILE_N), max_grid_blocks
                ) if a_in_regs else ceildiv(ceildiv(n, _TILE_N), col_tiles)
            ),
            block_dim=warps_per_block * WARP_SIZE,
        )

    # Register-resident A pays 4 VGPRs per k-chunk, so gate it to shallow
    # per-warp K (<= 24 chunks = 96 VGPRs); deeper K loads A per chunk.
    comptime shallow_k = (k_static // warps_per_block // _MFMA_K) <= 24
    if m <= 16:
        comptime if shallow_k:
            _launch[1, True]()
        else:
            _launch[1, False]()
    elif m <= 32:
        # Two column tiles per block halve the per-chunk A refetch pressure
        # without starving the grid (the measured bowl: 1/2/4 tiles = 83/76/
        # 106us on the lm_head shard). Register-resident A cannot cover two
        # row tiles: 192 VGPRs spills at 8 waves and 16-wave blocks cap lanes
        # at 128 VGPRs, so per-chunk A streaming is the design here.
        _launch[2, False, col_tiles=2]()
    elif m <= 64:
        _launch[4, False]()
    else:
        _launch[8, False]()
