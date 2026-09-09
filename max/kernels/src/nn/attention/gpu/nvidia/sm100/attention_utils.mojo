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
"""Shared SM100 attention primitives used by both MHA and MLA kernels.

This module contains generic SM100 (Blackwell) GPU primitives including:
- TMEM access helpers (TMemTile, STMatrixLayout)
- Pipeline synchronization (StagedPipeline, RolePipeline, etc.)
- FTZ arithmetic (add_ftz, sub_ftz, mul_ftz, etc.)
- Barrier helpers (FA4MiscMBars)
- MMA building blocks (bulk_mma, SM100TensorAccumulator)
- Masking utilities (apply_mask, apply_oob_mask)
"""

from std.math import ceildiv, exp2, align_up, iota
from std.math.constants import log2e
from std.sys import size_of, _RegisterPackType, get_defined_bool
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import llvm_intrinsic
from std.bit import prev_power_of_two, pop_count
from max.gpu import block_idx
from max.gpu.globals import WARP_SIZE
from max.gpu.primitives.id import cluster_dim
from max.gpu.primitives.warp import broadcast
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    UMMAInsDescriptor,
    UMMAKind,
    MMASmemDescriptorPair,
)
from max.gpu.compute.arch.tcgen05 import tcgen05_ld, tcgen05_st
from layout import (
    IntTuple,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)
from layout.tile_layout import (
    Layout as InternalLayout,
    row_major as tt_row_major,
)
from layout.tma_async import PipelineState, SharedMemBarrier
from std.memory import bitcast
from nn.attention.gpu.nvidia.sm100.attention import FA4Config

# `elect` is defined in the shared NVIDIA module so SM90 and SM100 can both use
# it without a cross-architecture import. Re-exported here for the many SM100
# callers (and tests) that import it from `attention_utils`.
from nn.attention.gpu.nvidia.common import elect
from nn.attention.mha_mask import MHAMask, MASK_VALUE, MaskStrategy
from nn.attention.mha_operand import (
    MHAOperand,
    PagedRowIndices,
    kv_sub_tile_rows,
    kv_num_sub_tiles,
    kv_tma_fold_chunks,
)
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.arch.sm100.mma import smem_descriptor


# IEEE-754 FP32 exponent bias. Clamping `exp2` inputs at
# `-FP32_EXP_BIAS` keeps the result within FP32's representable range
# (the smallest normal positive float32 is `2^-126`, and going much
# more negative just underflows to zero).
comptime FP32_EXP_BIAS = 127


# TileTensor-based aliases for storage (native types)
comptime LocalTensor[
    dtype: DType,
    layout: InternalLayout,
] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutUntrackedOrigin,
    address_space=.LOCAL,
]
comptime SharedMemTensor[dtype: DType, layout: InternalLayout] = TileTensor[
    dtype,
    InternalLayout[
        shape_types=layout.shape_types,
        stride_types=layout.stride_types,
    ],
    MutUntrackedOrigin,
    address_space=.SHARED,
]

# Legacy LayoutTensor aliases for TMA/MMA API boundaries
comptime LocalLT[
    dtype: DType, layout: Layout, element_layout: Layout = Layout(1, 1)
] = LayoutTensor[
    dtype,
    layout,
    MutAnyOrigin,
    address_space=.LOCAL,
    element_layout=element_layout,
]
comptime SharedMemPointer[type: AnyType] = UnsafePointer[
    type, MutAnyOrigin, address_space=.SHARED
]
comptime MBarType = SharedMemPointer[SharedMemBarrier]


# PagedRowIndices, kv_sub_tile_rows, kv_num_sub_tiles are now defined in
# nn.attention.mha_operand and re-exported via the import above.


def extract_power_of_two(N: Int, i: Int) -> Int:
    """Returns the `i`-th power-of-two component when decomposing `N` into decreasing powers of two.
    """
    var pt = prev_power_of_two(N)
    var rem = N
    for _ in range(i):
        rem -= pt
        pt = prev_power_of_two(rem)
    return pt


def cumulative_power_of_two(N: Int, i: Int) -> Int:
    """Returns the cumulative sum of the first `i` power-of-two components of `N`.
    """
    var acc = 0
    var rem = N
    for _ in range(i):
        var pt = prev_power_of_two(rem)
        acc += pt
        rem -= pt
    return acc


# Final call is with `pow_two == 0` (which isn't a power of 2)
# to enable use of this function with pipelining.
@always_inline("nodebug")
def break_into_powers_of_two[
    origins: OriginSet,
    //,
    func: def[pow_two: Int, offset: Int]() capturing[origins] -> None,
    N: Int,
    *,
    max_value: Int = 128,
]():
    """Calls `func` for each power-of-two-sized chunk of `N`, plus a final `pow_two=0` call for pipeline cleanup.

    Parameters:
        origins: Origin set captured by the callback (inferred).
        func: Callback invoked once per power-of-two chunk with the chunk
            size and starting offset, plus a final `pow_two=0` cleanup call.
        N: Total size to decompose into power-of-two chunks.
        max_value: Upper bound on the largest power-of-two chunk size
            (defaults to 128).
    """
    comptime power_of_two = prev_power_of_two(min(max_value, N))

    comptime for offset in range(0, N, power_of_two):
        comptime iter_size = min(N - offset, power_of_two)

        comptime if iter_size == power_of_two:
            func[power_of_two, offset]()
        else:
            comptime for j in range(pop_count(iter_size)):
                comptime pow_two = extract_power_of_two(iter_size, j)
                comptime coffset = offset + cumulative_power_of_two(
                    iter_size, j
                )
                func[pow_two, coffset]()
    # final call for possible pipeline cleanup
    func[0, N]()


struct STMatrixLayout[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_dtype_size: Int,
](TrivialRegisterPassable):
    """
    Layout for using `st_matrix` for writing the final accumulator to smem.

    Parameters:
        BM: Number of rows in the `BM` x `BN` output tile written via
            `st_matrix`.
        BN: Number of columns in the `BM` x `BN` output tile written via
            `st_matrix`.
        num_threads: Number of threads participating in the `st_matrix`
            store, used to derive the warp-group count.
        accum_dtype_size: Size in bytes of the accumulator element dtype,
            used to compute the per-store bit width.
    """

    # We have a BM x BN tile
    #
    # The st_matrix layout wants to map it to threads in 16x8 blocks
    # shape  (2,8), (2,4)
    # stride (0,4), (0,1)
    # Layout = ((2,8),(2,4)):((0,4),(0,1))
    # Where `0` stride indicates that the same thread is repeated across these.
    # We also need a layout for this local memory, which we define here.

    # look at figure 108 https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-stmatrix-fragments

    # That first `2` is
    comptime num_row_blocks_per_mma = 2
    # The second `2` is
    comptime frag_simdwidth: Int = 2

    comptime thread_cols = 4
    # When using tcgen05 ld/st we must repeat across all columns:
    comptime repeat = Self.BN // (Self.thread_cols * Self.frag_simdwidth)

    comptime num_warpgroups = ceildiv(Self.num_threads, 128)
    # 2 = 32 // 16, i.e. we need to load 2 sets of 16
    comptime num_m_tiles_total = ceildiv(2 * Self.BM, 128)
    comptime num_m_tiles = Self.num_m_tiles_total // Self.num_warpgroups

    comptime frag_size = Self.BN * Self.num_row_blocks_per_mma // Self.thread_cols

    comptime elements_per_repeat = Self.frag_simdwidth * Self.num_row_blocks_per_mma

    comptime vec_local_layout: Layout = Layout(
        IntTuple(
            IntTuple(Self.num_row_blocks_per_mma, Self.num_m_tiles),
            IntTuple(Self.repeat),
        ),
        IntTuple(
            IntTuple(
                Self.frag_simdwidth, Self.frag_size
            ),  # distance between vertical m tiles and local fragments
            IntTuple(
                Self.num_row_blocks_per_mma * Self.frag_simdwidth
            ),  # distance between bn repeats
        ),
    )
    comptime element_layout: Layout = Layout.row_major(1, Self.frag_simdwidth)
    comptime TensorType[dtype: DType] = LocalLT[
        dtype, Self.vec_local_layout, Self.element_layout
    ]
    comptime row_of_frags_layout: Layout = Layout.row_major(
        Self.num_m_tiles, Self.frag_size
    )

    comptime bits_per_byte = 8
    comptime bits = Self.bits_per_byte * Self.frag_simdwidth * Self.thread_cols * Self.accum_dtype_size

    @always_inline
    def __init__(out self):
        pass


struct STMatrixOffsets[
    BM: Int,
    BN: Int,
    *,
    num_threads: Int,
    accum_dtype_size: Int,
    curr_repeat: Int,
    cumulative_repeat: Int,
    m_mma: Int,
](TrivialRegisterPassable):
    """Precomputed TMEM and local-fragment offsets for one `st_matrix` repeat column.

    Parameters:
        BM: Number of rows in the `BM` x `BN` output tile (forwarded to
            `STMatrixLayout`).
        BN: Number of columns in the `BM` x `BN` output tile (forwarded to
            `STMatrixLayout`).
        num_threads: Number of threads participating in the `st_matrix` store
            (forwarded to `STMatrixLayout`).
        accum_dtype_size: Size in bytes of the accumulator element dtype
            (forwarded to `STMatrixLayout`).
        curr_repeat: Number of repeat columns in this power-of-two chunk of
            the `st_matrix` store.
        cumulative_repeat: Number of repeat columns already stored before
            this chunk, used as the TMEM column and local-fragment base
            offset.
        m_mma: M-tile index of the `st_matrix` store within the `BM`-row
            tile, selecting the 16-row TMEM quadrant.
    """

    comptime STLayout = STMatrixLayout[
        Self.BM,
        Self.BN,
        num_threads=Self.num_threads,
        accum_dtype_size=Self.accum_dtype_size,
    ]

    comptime tmem_col_offset = Self.cumulative_repeat * Self.STLayout.frag_simdwidth * Self.STLayout.thread_cols
    comptime tmem_row_offset = 16 * Self.m_mma
    comptime tmem_offset = (Self.tmem_row_offset << 16) + Self.tmem_col_offset
    comptime b32_per_repeat = Self.STLayout.elements_per_repeat * Self.accum_dtype_size // 4
    comptime local_frag_size_b32 = Self.curr_repeat * Self.b32_per_repeat
    comptime ptr_offset = Self.b32_per_repeat * (
        Self.STLayout.repeat * Self.m_mma + Self.cumulative_repeat
    )

    @always_inline
    def __init__(out self):
        pass


@always_inline
def o_store_tma_blocks_per_op[
    output_type: DType,
    output_swizzle_mode: TensorMapSwizzle,
    ov_depth: Int,
    group: Int,
    depth_splits: Int,
]() -> Int:
    """Box size (swizzle-granularity blocks per batched O-store TMA).

    The O store splits the contiguous `ov_depth` into swizzle-granularity blocks
    `K = output_swizzle_mode.bytes() // size_of[output_type]` and a single batched
    TMA copies `ceil(n_blocks / depth_splits)` of them (vs `n_blocks` per-block
    copies). `depth_splits` is the number of contiguous depth ranges the issuers
    divide the store into:
      - MHA (`depth_splits == 2`): the descriptor is shared between the 1Q combine
        (2 warpgroups, 1 TMA each over its half) and the single-issuer scale_write
        (2 pipelined TMAs over its two halves), so the box is the half-depth
        `ceil(n_blocks / 2)`.
      - depth512 (`depth_splits == 1`): single issuer, no combine, so the box is
        the full depth `n_blocks` (one TMA).
    The box size is independent of `group`: `RaggedTMA3DTile` folds the
    `(middle_dim, rows)` selectors into one dim, so even fused GQA (`group > 1`)
    fits the *blocks* dimension within the 5D TMA limit (rank-5 batched store) and
    uses the same `ceil(n_blocks / depth_splits)` box.
    Returns 0 (per-block path) only when `output_swizzle_mode != SWIZZLE_NONE`:
    the blocked-smem / identity-layout invariant the batched box relies on holds
    only for SWIZZLE_NONE (e.g. an FP8-QKV MLA variant with a SWIZZLE_128B bf16
    output store stays per-block).
    `group` is retained in the signature for call-site compatibility but no longer
    gates the result. This is the single source of truth for `tma_blocks_per_op`
    across the O-store descriptor type/creation sites.
    """
    comptime if output_swizzle_mode != TensorMapSwizzle.SWIZZLE_NONE:
        return 0
    comptime K = output_swizzle_mode.bytes() // size_of[output_type]()
    comptime n_blocks = align_up(ov_depth, K) // K
    return ceildiv(n_blocks, depth_splits)


@always_inline
def pack_row[
    n: Int, //, output_type: DType, w: Int, start: Int = 0
](o_vals: Array[Float32, n]) -> SIMD[.uint32, 4]:
    """Cast the `w` f32 O lanes `o_vals[start : start + w]` to `output_type` and
    pack them into one 16 B SWIZZLE_NONE store register (exactly four u32).

    `per_u32 = 4 // size_of[output_type]()` output elements pack into each u32
    (2 for a 2-byte bf16/f16 output, 4 for a 1-byte fp8 output), so a full 16 B
    block is `w == 4 * per_u32` f32 lanes (8 for bf16/f16, 16 for fp8). The
    return width is a fixed 4 so the wide-store helper `st_shared_v4_b32` takes
    it without a symbolic-width unification. Scale-free sibling of
    `scale_pack_o_row`, used by the split-K combine where `o_final` is already
    normalized.

    `o_vals` is a `tcgen05_ld` / accumulator result; `start`/`w` window it. Each
    u32 is built from an `SIMD[f32, per_u32]` chunk (f32x2 for bf16 -- wider SIMD
    scalarizes; f32x4 for fp8, mirroring the MLA fp8 store path); only the packed
    u32 store register is built wide.

    Parameters:
        n: Total number of f32 O lanes in `o_vals` (inferred).
        output_type: Target dtype to cast the f32 lanes to; must be a
            1-byte (`fp8`) or 2-byte (`bf16`/`f16`) dtype.
        w: Number of f32 lanes to pack; must equal `4 * per_u32` (8 for
            `bf16`/`f16`, 16 for `fp8`) to fill one 16 B block.
        start: Starting index into `o_vals` for the window (defaults to
            0).

    Args:
        o_vals: `tcgen05_ld` or accumulator result holding the f32 O
            lanes; the window `[start, start + w)` is packed.
    """
    comptime assert (
        size_of[output_type]() == 1 or size_of[output_type]() == 2
    ), "pack_row supports a 1-byte (fp8) or 2-byte (bf16/f16) output dtype"
    comptime per_u32 = 4 // size_of[output_type]()
    comptime assert w == 4 * per_u32, (
        "pack_row packs exactly one 16 B SWIZZLE_NONE block (four u32); `w`"
        " must equal 4 * (4 // size_of[output_type]()) -- 8 for bf16/f16, 16"
        " for fp8."
    )
    var packed = SIMD[.uint32, 4]()
    comptime for c in range(4):
        var chunk = SIMD[.float32, per_u32]()
        comptime for k in range(per_u32):
            chunk[k] = o_vals[start + per_u32 * c + k]
        packed[c] = bitcast[.uint32, 1](chunk.cast[output_type]())
    return packed


@always_inline
def blasst_vote_unanimous(
    blasst_vote: SharedMemPointer[UInt8], wg: UInt32, phase: UInt32
) -> Bool:
    """Reads the BLASST per-warp skip votes for `(wg, phase)` and ANDs them.

    Shared by the softmax-side vote publisher (`blasst_observe`, which reads
    back its own WG's vote right after publishing it) and the MMA-side
    consumer (`blasst_should_skip`, which reads a vote published earlier by
    the S-consumer pipeline's own synchronization). Only the read-back
    itself is shared here -- each caller's surrounding synchronization
    differs and stays local to that caller.
    """
    # One aligned 32-bit read replaces 4 byte reads + a short-circuit AND
    # chain (which serializes the loads behind 3 branches). Legal because a
    # `(wg, phase)` group's 4 slots are consecutive bytes at a 4-byte-aligned
    # address (`blasst_vote_byte_offset` follows the UInt32 `tmem_addr`, and
    # `base` is a multiple of 4) and every slot only ever holds 0 or 1
    # (`kernel.mojo` zero-inits the region; `blasst_observe` writes 0/1), so
    # "all four nonzero" == "the word is 0x01010101".
    var base = (wg * UInt32(2) + phase) * UInt32(4)
    return (blasst_vote + base).bitcast[UInt32]()[0] == UInt32(0x01010101)


@always_inline
def scale_pack_o_row[
    n: Int, //, output_type: DType, w: Int, start: Int = 0
](o_vals: Array[Float32, n], inv_row_sum: Float32) -> SIMD[
    DType.uint32, w // 2
]:
    """Scale the `w` f32 O lanes `o_vals[start : start + w]` by `inv_row_sum`,
    cast to the 2-byte `output_type`, and pack into `w // 2` u32 lanes (the
    row-major 16 B SWIZZLE_NONE store register).

    `o_vals` is a `tcgen05_ld` result; `start`/`w` window it so one wide TMEM
    load can feed several stores (depth512 loads 16 lanes, stores two 8-lane
    blocks). Compute stays in f32x2 (64-bit) chunks because LLVM scalarizes
    wider SIMD here; only the packed u32 store register is built wide. Shared by
    the SM100 O-store writeback helpers (`fa4_scale_write_output`,
    `depth512_scale_write_output`).

    Parameters:
        n: Total number of f32 O lanes in `o_vals` (inferred).
        output_type: Target 2-byte dtype (`bf16`/`f16`) to cast the
            scaled lanes to.
        w: Number of f32 lanes to scale and pack; must equal the width
            of one 16 B SWIZZLE_NONE block.
        start: Starting index into `o_vals` for the window (defaults to
            0).

    Args:
        o_vals: `tcgen05_ld` result holding the f32 O lanes; the window
            `[start, start + w)` is scaled and packed.
        inv_row_sum: Inverse of the softmax row sum, multiplied into
            each lane to normalize the output.
    """
    comptime assert size_of[output_type]() == 2
    var packed = SIMD[.uint32, w // 2]()
    comptime for c in range(w // 2):
        var pair = (
            SIMD[.float32, 2](o_vals[start + 2 * c], o_vals[start + 2 * c + 1])
            * inv_row_sum
        ).cast[output_type]()
        packed[c] = bitcast[.uint32, 1](pair)
    return packed


@always_inline
def combine_pack_o_row[
    n: Int, //, output_type: DType
](
    own: Array[Float32, n],
    peer: Array[Float32, n],
    scale_own: Float32,
    scale_peer: Float32,
) -> SIMD[.uint32, n // 2]:
    """LSE-combine `own * scale_own + peer * scale_peer` over `n` f32 O lanes,
    cast to the 2-byte `output_type`, and pack into `n // 2` u32 lanes.

    `own`/`peer` are `tcgen05_ld` results. f32x2 compute / wide-only store, as
    in `scale_pack_o_row`; the fused `peer.fma(scale_peer, own * scale_own)`
    form matches the per-element combine it replaces. Used by
    `fa4_lse_combine_write`.
    """
    comptime assert size_of[output_type]() == 2
    var packed = SIMD[.uint32, n // 2]()
    comptime for c in range(n // 2):
        var own_c = SIMD[.float32, 2](own[2 * c], own[2 * c + 1])
        var peer_c = SIMD[.float32, 2](peer[2 * c], peer[2 * c + 1])
        var comb = peer_c.fma(
            SIMD[.float32, 2](scale_peer), own_c * scale_own
        ).cast[output_type]()
        packed[c] = bitcast[.uint32, 1](comb)
    return packed


@always_inline
def st_shared_v4_b32[
    dtype: DType,
    //,
](
    dst: UnsafePointer[mut=True, Scalar[dtype], _, address_space=.SHARED],
    elem_off: Int,
    packed: SIMD[.uint32, 4],
):
    """Explicit 16 B `st.shared.v4.b32` (one `STS.128`) to `dst[elem_off]`.

    Forces the wide vector store regardless of how `packed`'s four words were
    produced. A plain `.store()` of a `SIMD[uint32, 4]` scalarizes to 4x
    `STS.32` when the words come from a long-lived accumulator (the split-K
    combine's `o_final`): ptxas cannot fuse the non-contiguous in-place `F2FP`
    pack outputs, and the resulting 4 B stores hit only every 4th bank (4-way
    conflict, 4x wavefronts). The `v4.b32` operand mandates a contiguous
    register quad, so this stays one bank-conflict-free 16 B transaction.

    `dtype` is the shared buffer's element type -- any 1-byte (fp8), 2-byte
    (bf16/f16), or 4-byte (f32) output; `elem_off` is in `dtype` elements.
    `packed` is a fixed 16 B (four u32) for every `dtype`: `pack_row` folds the
    per-u32 element count (2 for bf16, 4 for fp8, 1 for f32) into that width, so
    one call stores one SWIZZLE_NONE block (`fa4_ws_intracta_combine` stores its
    f32 output this way).

    Parameters:
        dtype: Element dtype of the shared buffer; any 1-byte (`fp8`),
            2-byte (`bf16`/`f16`), or 4-byte (`f32`) output dtype (inferred).

    Args:
        dst: Shared-memory pointer to the store target.
        elem_off: Element offset into `dst` in `dtype`-element units.
        packed: Four `u32` words forming one 16 B SWIZZLE_NONE block to
            store.
    """
    var dst_ptr = dst + elem_off
    _ = inlined_assembly[
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        NoneType,
        constraints="l,r,r,r,r",
        has_side_effect=True,
    ](dst_ptr, packed[0], packed[1], packed[2], packed[3])


@always_inline
def store_p_quadrant[
    cols: Int,  # quadrant width; must equal BN // 4
    p_type: DType,
    //,
    *,
    BN: Int,
    # Physical row count of the P SMEM tile. The layout is chunk-outer with
    # chunk stride `p_tile_rows * sw_K`, so this MUST match the tile the P@V
    # A-descriptor is built over (`BM` for the shared-key datapath).
    p_tile_rows: Int,
](
    p_smem: UnsafePointer[
        mut=True, Scalar[p_type], _, address_space=AddressSpace.SHARED
    ],
    p: Array[Scalar[DType.float32], cols],
    row: UInt32,
    warp_in_wg: UInt32,
):
    """Row-local P quadrant writer (Layout-G register map -> SWIZZLE_NONE SMEM).

    Lane identity = row (< 32); warp `w` holds columns [w*BN/4, (w+1)*BN/4).
    Casts the fp32 quadrant to `p_type` and stores it in the k-major layout the
    SS P@V A-descriptor reads: element (r, k) sits in block `k // sw_K` at block
    base `(k // sw_K) * p_tile_rows * sw_K`, at within-block offset `r * sw_K`.
    16 B stores; the caller owns the `fence_async_view_proxy` + barrier before
    the MMA reads P.

    Parameters:
        cols: Quadrant width in elements; must equal `BN // 4` (inferred).
        p_type: Element dtype of the P SMEM tile (inferred).
        BN: Key-block width of the full P tile.
        p_tile_rows: Physical row count of the P SMEM tile.

    Args:
        p_smem: Base of this warpgroup's P tile in shared memory.
        p: This thread's fp32 P quadrant register array (one row, `cols`
            wide).
        row: This thread's row within the tile (lane identity, < p_tile_rows).
        warp_in_wg: This warp's index within the warpgroup (0..3).
    """
    # The SWIZZLE_NONE box is 16 B; spelled as `16 // size_of` for the same
    # reason `fa4_scale_write_output` spells its `o_sw_K` that way.
    comptime sw_K = 16 // size_of[p_type]()  # bf16 -> 8, fp8 -> 16
    comptime group_elems = 16 // size_of[p_type]()  # one st.shared.v4.b32
    comptime num_groups = cols // group_elems
    comptime assert cols * 4 == BN, "quadrant width must be BN // 4"
    comptime assert cols % group_elems == 0, "quadrant not 16B-store aligned"
    comptime assert p_tile_rows % 8 == 0, "8-row core matrices"

    var block_base = (Int(warp_in_wg) * cols) // sw_K
    var within = Int(row) * sw_K

    comptime for g in range(num_groups):
        # `sw_K == group_elems`, so group `g` IS block `block_base + g`.
        var off = (block_base + g) * (p_tile_rows * sw_K) + within
        st_shared_v4_b32(
            p_smem,
            off,
            pack_row[p_type, w=group_elems, start=g * group_elems](p),
        )


@always_inline
def _tmem_offset(dtype_size: Int, *, MMA_N: Int, m_mma: Int, n_mma: Int) -> Int:
    var row = 16 * m_mma
    var col = (MMA_N * n_mma * dtype_size) // 4
    return (row << 16) + col


@always_inline
def _tmem_offset[dtype: DType, *, MMA_N: Int, m_mma: Int, n_mma: Int]() -> Int:
    comptime linear = _tmem_offset(
        size_of[dtype](), MMA_N=MMA_N, m_mma=m_mma, n_mma=n_mma
    )
    return linear


struct TMemTile[
    dtype_: DType,
    BM: Int,
    BN: Int,
](TrivialRegisterPassable):
    """Represents a tile in SM100 tensor memory (TMEM) and provides async load/store helpers.

    Parameters:
        dtype_: Element dtype of the TMEM tile.
        BM: Number of rows in the tile, in elements; must be a multiple
            of 64.
        BN: Number of columns in the tile, in elements.
    """

    comptime dtype: DType = Self.dtype_
    comptime dtype_size = size_of[Self.dtype]()
    comptime num_m_tiles = Self.BM // 64

    var tmem_addr: UInt32

    @always_inline
    def __init__(out self, tmem_addr: UInt32):
        self.tmem_addr = tmem_addr

    @always_inline
    def __getitem__(self, i: UInt32) -> Self:
        return {self.tmem_addr + i * UInt32(Self.BN)}

    @always_inline
    def offset[m_mma: Int, n_mma: Int](self) -> UInt32:
        comptime if m_mma == 0 and n_mma == 0:
            return self.tmem_addr
        else:
            comptime linear = _tmem_offset[
                Self.dtype, MMA_N=Self.BN, m_mma=m_mma, n_mma=n_mma
            ]()

            return self.tmem_addr + UInt32(linear)

    @staticmethod
    @always_inline
    def allocate_register_tile[
        *, num_threads: Int
    ](
        out res: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        res = type_of(res).stack_allocation()

    @always_inline
    def store_async[
        *, num_threads: Int
    ](
        self,
        src: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        comptime assert Self.dtype_size <= 4
        var ptr = src.ptr.bitcast[UInt32]()
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ]
        comptime assert st_mat_layout.bits == 128 or st_mat_layout.bits == 256

        @__parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            # pow_two is current repeat, offset total so far
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_dtype_size=Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=offset,
                        m_mma=m_mma,
                    ]()
                    var tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
                    var frag = Array[_, offsets.local_frag_size_b32](
                        fill_with_unrolled=lambda [i: Int]() -> UInt32: (
                            ptr.load(offsets.ptr_offset + i)
                        )
                    )
                    # 16 x 256b results in repeated 8x4 matrix of <1,2> vector pattern
                    tcgen05_st[
                        datapaths=16,  # first dimension of the shape
                        bits=st_mat_layout.bits,  # second dimension of the shape
                        repeat=pow_two,
                        pack=False,
                    ](tmem, frag)

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=store_fn, N=st_mat_layout.repeat, max_value=max_value
        ]()

    @always_inline
    def load_async_with_st_matrix_layout[
        *, num_threads: Int
    ](
        self,
        out dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        comptime assert (
            Self.dtype_size <= 4
        ), "Loading for st matrix requires elements to be <= 4 bytes."
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ]()
        comptime assert (st_mat_layout.num_m_tiles == 1) or (
            st_mat_layout.num_m_tiles == 2
        ), (
            "Only 1 or 2 m tiles are supported, but"
            " st_mat_layout.num_m_tiles == "
            + String(st_mat_layout.num_m_tiles)
        )

        dst = type_of(dst).stack_allocation()
        self.load_st_matrix_chunk[
            num_threads=num_threads,
            start_repeat=0,
            num_repeats=st_mat_layout.repeat,
        ](dst)

    @always_inline
    def load_st_matrix_chunk[
        *, num_threads: Int, start_repeat: Int, num_repeats: Int
    ](
        self,
        dst: STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ].TensorType[Self.dtype],
    ):
        """Load a range of repeat columns from tmem into a pre-allocated
        tensor.

        Parameters:
            num_threads: Number of threads in the warp group.
            start_repeat: First repeat index to load (0-based).
            num_repeats: Number of repeats to load.

        Args:
            dst: Pre-allocated register tensor.
        """
        comptime st_mat_layout = STMatrixLayout[
            Self.BM,
            Self.BN,
            num_threads=num_threads,
            accum_dtype_size=Self.dtype_size,
        ]()
        comptime load_dtype = DType.uint32
        var ptr = rebind[
            UnsafePointer[
                Scalar[load_dtype], MutAnyOrigin, address_space=.LOCAL
            ]
        ](dst.ptr)

        @__parameter
        @always_inline
        def load_fn[pow_two: Int, local_offset: Int]():
            comptime assert pow_two + local_offset <= num_repeats
            comptime if pow_two > 0:
                comptime for m_mma in range(st_mat_layout.num_m_tiles):
                    comptime offsets = STMatrixOffsets[
                        Self.BM,
                        Self.BN,
                        num_threads=num_threads,
                        accum_dtype_size=Self.dtype_size,
                        curr_repeat=pow_two,
                        cumulative_repeat=start_repeat + local_offset,
                        m_mma=m_mma,
                    ]()
                    var tmem = self.tmem_addr + UInt32(offsets.tmem_offset)
                    var frag = tcgen05_ld[
                        datapaths=16,
                        bits=st_mat_layout.bits,
                        repeat=pow_two,
                        dtype=load_dtype,
                        pack=False,
                        width=offsets.local_frag_size_b32,
                    ](tmem)

                    comptime for _i in range(offsets.local_frag_size_b32):
                        ptr.store(offsets.ptr_offset + _i, frag[_i])

        comptime max_value = 64 if st_mat_layout.bits == 128 else 32
        break_into_powers_of_two[
            func=load_fn, N=num_repeats, max_value=max_value
        ]()

    @always_inline
    def load_async(
        self,
        out dst: Array[Scalar[Self.dtype], Self.BN],
    ):
        dst = Array[Scalar[Self.dtype], Self.BN](uninitialized=True)
        # The uint32 bitcast path below assumes dtype_size == 4.
        # Sub-32-bit types (bf16, f16) pack multiple elements per uint32
        # and would need unpacking logic not yet implemented.
        comptime assert (
            Self.dtype_size == 4
        ), "load_async only supports 32-bit dtypes"
        comptime repeat = Self.dtype_size * Self.BN // 4
        comptime dtype = Self.dtype if Self.dtype_size == 4 else DType.uint32

        @__parameter
        @always_inline
        def load_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime if dtype == Self.dtype:
                    var frag0 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=Self.dtype,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))

                    comptime for _i in range(pow_two):
                        dst[offset + _i] = frag0[_i]
                else:
                    var frag1 = tcgen05_ld[
                        datapaths=32,  # first dimension of the shape
                        bits=32,  # second dimension of the shape
                        repeat=pow_two,
                        dtype=DType.uint32,
                        pack=False,
                        width=pow_two,
                    ](self.tmem_addr + UInt32(offset))

                    comptime for _i in range(pow_two):
                        dst[offset + _i] = bitcast[Self.dtype](frag1[_i])

        break_into_powers_of_two[func=load_fn, N=repeat, max_value=128]()

    @always_inline
    def store_async[
        src_type: DType
    ](self, src: LocalTensor[src_type, row_major[Self.BN]()]):
        @__parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime frag_width = pow_two * Self.dtype_size // 4
                var frag = Array[UInt32, frag_width](uninitialized=True)

                comptime if src_type == Self.dtype:
                    comptime for _i in range(frag_width):
                        frag[_i] = src.ptr.bitcast[UInt32]().load(offset + _i)
                elif pow_two > 1:
                    comptime size_ratio = size_of[src_type]() // Self.dtype_size
                    comptime cast_width = min(
                        4 if size_ratio
                        >= 4 else (2 if size_ratio >= 2 else pow_two),
                        pow_two,
                    )
                    comptime u32_per_cast = cast_width * Self.dtype_size // 4
                    comptime num_casts = pow_two // cast_width

                    comptime if u32_per_cast >= 1:
                        comptime for _i in range(num_casts):
                            var src_vec = src.raw_load[width=cast_width](
                                offset + _i * cast_width
                            )
                            var dst_vec = src_vec.cast[Self.dtype]()
                            var packed_chunk = bitcast[
                                DType.uint32, u32_per_cast
                            ](dst_vec)
                            comptime for _j in range(u32_per_cast):
                                frag[_i * u32_per_cast + _j] = packed_chunk[_j]
                    else:
                        var packed = bitcast[.uint32, frag_width](
                            src.raw_load[width=pow_two](offset).cast[
                                Self.dtype
                            ]()
                        )
                        comptime for _i in range(frag_width):
                            frag[_i] = packed[_i]
                else:
                    frag[0] = bitcast[.uint32](src[0].cast[Self.dtype]())

                tcgen05_st[
                    datapaths=32,  # first dimension of the shape
                    bits=32,  # second dimension of the shape
                    repeat=pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + UInt32(offset * Self.dtype_size // 4), frag)

        break_into_powers_of_two[func=store_fn, N=Self.BN, max_value=128]()

    @always_inline
    def store_async[
        src_type: DType,
        src_len: Int,
        src_offset: Int = 0,
    ](self, src: Array[Scalar[src_type], src_len]):
        @__parameter
        @always_inline
        def store_fn[pow_two: Int, offset: Int]():
            comptime if pow_two > 0:
                comptime frag_width = pow_two * Self.dtype_size // 4
                var frag = Array[UInt32, frag_width](uninitialized=True)

                comptime if src_type == Self.dtype:
                    comptime for _i in range(frag_width):
                        frag[_i] = bitcast[.uint32](
                            src[src_offset + offset + _i]
                        )
                else:
                    comptime sub_elements = 4 // Self.dtype_size
                    comptime size_ratio = size_of[src_type]() // Self.dtype_size
                    comptime cast_width = 4 if size_ratio >= 4 else (
                        2 if size_ratio >= 2 else 1
                    )

                    comptime for _i in range(frag_width):
                        var x: SIMD[Self.dtype, sub_elements] = {}
                        comptime if cast_width >= 2:
                            comptime for _j in range(
                                0, sub_elements, cast_width
                            ):
                                var src_vec: SIMD[src_type, cast_width] = {}
                                comptime for _k in range(cast_width):
                                    comptime idx = (
                                        src_offset
                                        + offset
                                        + _i * sub_elements
                                        + _j
                                        + _k
                                    )
                                    src_vec[_k] = src[idx]
                                var dst_vec = src_vec.cast[Self.dtype]()
                                comptime for _k in range(cast_width):
                                    x[_j + _k] = dst_vec[_k]
                        else:
                            comptime for _j in range(sub_elements):
                                comptime idx = (
                                    src_offset + offset + _i * sub_elements + _j
                                )
                                x[_j] = src[idx].cast[Self.dtype]()
                        frag[_i] = bitcast[.uint32, 1](x)
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=pow_two * Self.dtype_size // 4,
                    pack=False,
                ](self.tmem_addr + UInt32(offset * Self.dtype_size // 4), frag)

        break_into_powers_of_two[func=store_fn, N=Self.BN, max_value=128]()


struct SM100TensorAccumulator[
    operand_type: DType,
    accum_dtype: DType,
    MMA_M: Int,
    MMA_N: Int,
    BK: Int,
    *,
    a_tmem: Bool,
    mma_kind: UMMAKind = UMMAKind.KIND_F16,
    swizzle_a: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    swizzle_b: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    transpose_b: Bool = True,
    cta_group: Int = 1,
    num_stages: Int = 1,
    b_page_dense: Bool = False,
    # ANDed into `use_3_then_1_split` below (default True: byte-identical for
    # every existing caller). Layout-E's P@V reduction-split accumulator
    # (`mma_warp.mojo`) sets this False so its `num_stages==2` P sub-stage
    # split stays an EVEN 2-then-2 of each reduction chunk's own `BK`, instead
    # of the 3-then-1 split calibrated for Layout-G's (non-reduction-split)
    # P write cadence.
    allow_3_then_1_split: Bool = True,
](TrivialRegisterPassable):
    """Performs the `C = A @ B` tensor contraction on SM100 using `tcgen05.mma` instructions.

    The A operand is either an SMEM tile (`a_tmem=False`, the "SS" contraction) or a TMEM tile (`a_tmem=True`, the "TS" contraction); B is always an SMEM descriptor. When `cta_group == 1 and MMA_M <= 64`, the warp-specialized `.ws` datapath is used.

    Every `mma*` helper below takes a `c_scale`, lowered as
    `setp.ne.b32 %ps, c_scale, 0` feeding UMMA's `enable-input-d`: ZERO
    OVERWRITES the accumulator (`D = A@B`) -- pass 0 on the first block -- and
    NONZERO ACCUMULATES (`D = A@B + D`).

    Parameters:
        operand_type: Element dtype of the A and B input operands.
        accum_dtype: Element dtype of the output accumulator `C`.
        MMA_M: M dimension (rows) of the output tile in MMA units.
        MMA_N: N dimension (columns) of the output tile in MMA units.
        BK: K dimension (contraction axis) tile size in elements.
        a_tmem: Whether the A operand resides in TMEM (`True`, the TS
            contraction) or SMEM (`False`, the SS contraction).
        mma_kind: `UMMAKind` selecting the `tcgen05.mma` instruction variant
            (defaults to `UMMAKind.KIND_F16`).
        swizzle_a: SMEM swizzle mode for the A tile (defaults to
            `SWIZZLE_128B`); meaningful only when `a_tmem` is `False`.
        swizzle_b: SMEM swizzle mode for the B tile (defaults to
            `SWIZZLE_128B`).
        transpose_b: Whether B is stored k-major (`True`) or mn-major
            (`False`) (defaults to `True`).
        cta_group: Number of cooperating CTAs, 1 or 2 (defaults to 1). When
            1 and `MMA_M <= 64`, the `.ws` datapath is used.
        num_stages: Number of K-dimension pipeline stages for latency
            hiding (defaults to 1).
        b_page_dense: Whether B uses the row-major page-fold layout
            (defaults to `False`).
        allow_3_then_1_split: Whether a `num_stages == 2` A-in-TMEM
            contraction with `num_k_blocks % 4 == 0` may use the 3-then-1
            K sub-stage split (defaults to `True`, the historical behavior).
            Set `False` by Layout-E's P@V reduction-split accumulator so the
            P sub-stage chunks stay an even 2-then-2 aligned with the V
            reduction chunks.
    """

    # This performs C = A @ B
    # where A is BM x BK and B is BN x BK if k major, else BK x BN.
    # `BK` is broken into `num_stages` and pipelined.
    #
    # The A operand is either an SMEM tile referenced through an
    # `MMASmemDescriptorPair` (`a_tmem=False`, the "SS" contraction, e.g.
    # Q@K' producing the unweighted score input of the `softmax`) or a
    # TMEM tile referenced through a raw TMEM address (`a_tmem=True`, the
    # "TS" contraction, e.g. P@V). B is always an SMEM descriptor.
    # `swizzle_a` is meaningful only for `a_tmem=False`.
    # The benefit of setting `stages > 1` is that this can hide latency.
    #
    # When `cta_group == 1 and MMA_M <= 64`, MMAs are issued as
    # `tcgen05.mma.ws` (warp-specialized), which uses the packed-TMEM
    # (1x4 / Layout G) datapath: the hardware subpartition folds
    # `m_pack = 128 // MMA_M` row-groups onto the same physical TMEM
    # columns, so the accumulator occupies only `MMA_N / m_pack` physical
    # columns and all 128 datapath lanes stay busy. For MMA_M == 64 the
    # non-ws form is also legal in hardware but uses only half the
    # datapaths, so this struct ALWAYS chooses ws there. The A (P) operand
    # in TMEM (`a_tmem=True`) follows the same packed convention -- its
    # producer must write it accordingly -- and consumers must read the
    # accumulator with the packed layout: all `m_pack` warps issue
    # tcgen05_ld/st against the SAME TMEM column address (no per-warp
    # column offsets), because the HW subpartition does the lane folding.
    comptime use_ws = Self.cta_group == 1 and Self.MMA_M <= 64
    comptime tcgen05_mma_type = "tcgen05.mma.ws.cta_group::1."
    comptime operand_t = Self.operand_type
    comptime operand_size = size_of[Self.operand_t]()
    comptime accum_t = Self.accum_dtype
    comptime MMA_K = 16 if Self.operand_type.is_half_float() else 32
    # The TS quadrant requires BK % MMA_K == 0 (P columns are produced in
    # whole MMA_K blocks); the SS quadrant counts a ragged tail block.
    comptime num_k_mmas = (Self.BK // Self.MMA_K) if Self.a_tmem else ceildiv(
        Self.BK, Self.MMA_K
    )
    comptime swizzle_granularity = (
        Self.swizzle_b.bytes() if Self.a_tmem else max(
            Self.swizzle_a.bytes(), Self.swizzle_b.bytes()
        )
    ) // size_of[Self.operand_t]()
    # TMEM A (P) is written at exactly BK columns, so no swizzle padding
    # applies; SMEM A/B tiles are padded to the swizzle granularity.
    comptime padded_BK = Self.BK if Self.a_tmem else align_up(
        Self.BK, Self.swizzle_granularity
    )
    comptime num_k_blocks = Self.padded_BK // Self.MMA_K
    comptime use_3_then_1_split: Bool = Self.allow_3_then_1_split and Self.a_tmem and Self.num_stages == 2 and Self.num_k_blocks % 4 == 0
    comptime num_k_blocks_per_stage = Self.num_k_blocks // (
        4 if Self.use_3_then_1_split else Self.num_stages
    )

    # With cta_group > 1, each CTA's SMEM holds MMA_M/cta_group rows (A)
    # and MMA_N/cta_group columns (B). The K-offset arithmetic in
    # `_build_mma` (SS path) uses these layouts, so BMN must match per-CTA
    # dimensions to keep addresses within each CTA's SMEM tile.
    #
    # For k_major A the outer-K stride is BMN * swizzle_width; halving BMN
    # halves that stride so K offsets stay in the per-CTA buffer.
    # For k_major B (transpose_b) the cross-swizzle-chunk K stride is
    # BMN * swizzle_width too (tile_layout_k_major's (0, k) offset
    # scales with BMN), so per-CTA BMN is REQUIRED in both quadrants:
    # a full-MMA_N layout doubles every chunk-crossing K offset at
    # cta_group=2, reading wrong SMEM (and past the tile) from the
    # second 64-column chunk onward.
    #
    # The mn-major TS quadrant (the P@V users) historically builds
    # b_layout with the full MMA_N (no cta_group division); preserved
    # as-is.
    comptime a_bmn: Int = align_up(Self.MMA_M // Self.cta_group, 8)
    comptime a_layout = tile_layout_k_major[
        Self.operand_t, Self.a_bmn, Self.padded_BK, Self.swizzle_a
    ]()
    comptime b_bmn: Int = Self.MMA_N if (
        Self.a_tmem and not Self.transpose_b
    ) else (Self.MMA_N // Self.cta_group)
    comptime b_layout = tile_layout_k_major[
        Self.operand_t,
        Self.b_bmn,
        Self.padded_BK,
        Self.swizzle_b,
        page_dense=Self.b_page_dense,
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.operand_t,
        Self.b_bmn,
        Self.padded_BK,
        Self.swizzle_b,
        page_dense=Self.b_page_dense,
    ]()

    comptime idesc = UMMAInsDescriptor[Self.mma_kind].create[
        Self.accum_t,
        Self.operand_t,
        Self.operand_t,
        Index[dtype=DType.uint32](Self.MMA_M, Self.MMA_N),
        transpose_b=Self.transpose_b,
    ]()

    comptime AType: TrivialRegisterPassable = TMemTile[
        Self.operand_type, Self.MMA_M, Self.BK
    ] if Self.a_tmem else MMASmemDescriptorPair
    # The runtime argument type of `a` in `mma`/`mma_maybe_partial_k`:
    # a raw TMEM address for the TS quadrant, an SMEM descriptor pair
    # for the SS quadrant.
    comptime AInput: TrivialRegisterPassable = UInt32 if Self.a_tmem else MMASmemDescriptorPair
    comptime BType = MMASmemDescriptorPair
    comptime CType = TMemTile[Self.accum_t, Self.MMA_M, Self.MMA_N]

    @staticmethod
    @always_inline("nodebug")
    def mma[
        *, stage_idx: Int = 0
    ](
        a: Self.AInput,
        b: Self.BType,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
    ):
        comptime assert (not Self.use_ws) or Self.MMA_M in (
            32,
            64,
        ), "ws path requires MMA_M in (32, 64)"

        comptime if Self.num_stages == 1:
            # Original single-stage behavior
            comptime if Self.a_tmem:
                var a_ = rebind[UInt32](a)
                comptime if Self.use_ws:
                    bulk_mma_ws_ts[
                        Self.mma_kind,
                        Self.operand_t,
                        b_BMN=Self.MMA_N,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](Self.idesc, a_, b, c, c_scale, elect)
                else:
                    bulk_mma[
                        Self.b_layout,
                        mma_k=Self.MMA_K,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](Self.idesc, a_, b, c, c_scale, elect)
            else:
                var a_ = rebind[MMASmemDescriptorPair](a)
                comptime if Self.use_ws:
                    bulk_mma_ws[
                        Self.mma_kind,
                        Self.operand_t,
                        Self.operand_t,
                        a_BMN=Self.a_bmn,
                        a_BK=Self.padded_BK,
                        a_swizzle=Self.swizzle_a,
                        a_is_k_major=True,
                        b_BMN=Self.b_bmn,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=Self.num_k_mmas,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](Self.idesc, a_, b, c, c_scale, elect)
                else:
                    bulk_mma[
                        Self.a_layout,
                        Self.b_layout,
                        num_k_mmas=Self.num_k_mmas,
                        mma_k=Self.MMA_K,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](Self.idesc, a_, b, c, c_scale, elect)
        else:
            comptime start = 3 * stage_idx if Self.use_3_then_1_split else stage_idx
            comptime end = stage_idx + 3 if Self.use_3_then_1_split else stage_idx + 1
            comptime k_batch_start = Self.num_k_blocks_per_stage * start
            comptime k_batch_end = min(
                Self.num_k_blocks_per_stage * end, Self.num_k_mmas
            )
            comptime k_offset = k_batch_start * Self.MMA_K
            # Offset both A and B operands by k_offset.
            # B (smem) offset: move by k_offset rows of the descriptor.
            comptime b_byte_offset = (
                Self.b_layout(IntTuple(0, k_offset)) * Self.operand_size
            )
            var scale: UInt32

            comptime if stage_idx == 0:
                scale = c_scale
            else:
                scale = 1
            comptime if Self.a_tmem:
                # A (tmem) offset: P is MMA_M x BK, so the column offset is
                # k_offset * dtype_size / 4 (in tmem units).
                comptime a_tmem_offset = (k_offset * Self.operand_size) // 4
                var a_ = rebind[UInt32](a) + UInt32(a_tmem_offset)
                comptime if Self.use_ws:
                    bulk_mma_ws_ts[
                        Self.mma_kind,
                        Self.operand_t,
                        b_BMN=Self.MMA_N,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
                else:
                    bulk_mma[
                        Self.b_layout,
                        mma_k=Self.MMA_K,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
            else:
                # A (smem) offset: move by k_offset rows of the descriptor.
                comptime a_byte_offset = (
                    Self.a_layout(IntTuple(0, k_offset)) * Self.operand_size
                )
                var a_ = rebind[MMASmemDescriptorPair](a) + UInt32(
                    a_byte_offset
                )
                comptime if Self.use_ws:
                    bulk_mma_ws[
                        Self.mma_kind,
                        Self.operand_t,
                        Self.operand_t,
                        a_BMN=Self.a_bmn,
                        a_BK=Self.padded_BK,
                        a_swizzle=Self.swizzle_a,
                        a_is_k_major=True,
                        b_BMN=Self.b_bmn,
                        b_BK=Self.padded_BK,
                        b_swizzle=Self.swizzle_b,
                        b_is_k_major=Self.transpose_b,
                        num_k_mmas=k_batch_end - k_batch_start,
                        operand_size=Self.operand_size,
                        tcgen05_mma_type=Self.tcgen05_mma_type,
                        mma_k=Self.MMA_K,
                        b_page_dense=Self.b_page_dense,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )
                else:
                    bulk_mma[
                        Self.a_layout,
                        Self.b_layout,
                        num_k_mmas=k_batch_end - k_batch_start,
                        mma_k=Self.MMA_K,
                        operand_size=Self.operand_size,
                        cta_group=Self.cta_group,
                    ](
                        Self.idesc,
                        a_,
                        b + UInt32(b_byte_offset),
                        c,
                        scale,
                        elect,
                    )

    @staticmethod
    @always_inline("nodebug")
    def mma_maybe_partial_k[
        *, stage_idx: Int = 0
    ](
        a: Self.AInput,
        b: Self.BType,
        c: UInt32,
        *,
        c_scale: UInt32,
        elect: Int32,
        valid_k_mmas: UInt32,
    ):
        # Contraction for the last KV tile, where only `valid_k_mmas`
        # MMA_K-blocks hold real data (for TS, the loaded V pages; requires
        # page_size % MMA_K == 0 so the loaded boundary is MMA_K-aligned --
        # enforced by FA4Config.supported()). Skipping the unloaded tail
        # blocks is bit-identical to the full contraction (for TS their P is
        # exactly 0 after masking) AND avoids reading uninitialized SMEM
        # (the `0 * NaN = NaN` bug). For a full last tile, calling with
        # `valid_k_mmas = num_k_mmas` degenerates to the plain contraction
        # (every `@!%pv` validity guard is never-true).

        # comptime k-block range owned by this stage -- mirror `mma`'s stage
        # split as top-scope ternaries (NOT a `comptime if` block, whose
        # branch scope would hide ks_start/ks_end).
        comptime _multi = Self.num_stages != 1
        comptime _start = 3 * stage_idx if Self.use_3_then_1_split else stage_idx
        comptime _end = (
            stage_idx + 3
        ) if Self.use_3_then_1_split else stage_idx + 1
        comptime ks_start = (
            Self.num_k_blocks_per_stage * _start
        ) if _multi else 0
        comptime ks_end = min(
            Self.num_k_blocks_per_stage * _end, Self.num_k_mmas
        ) if _multi else Self.num_k_mmas

        # Issue this stage's k-blocks as one fused inline-asm sequence. The
        # partial primitives predicate each block `jj` on a SEPARATE,
        # warp-uniform validity guard (`@!%pv`, run iff `jj < valid_k_mmas`)
        # while passing `elect` through UNMODIFIED, so the elect codegen
        # matches the full-tile path (no BSYNC.RECONVERGENT). `a`/`b` are the
        # stage-0 bases (unlike `mma`'s multi-stage path, no offsets are
        # applied here); the builders apply absolute per-block offsets, so no
        # stage-offset swizzle-linearity is assumed. `c_scale` initializes
        # the accumulator on stage 0's first block (jj 0; always valid since
        # valid_k_mmas >= 1); every later block accumulates.
        comptime assert (not Self.use_ws) or Self.MMA_M in (
            32,
            64,
        ), "ws path requires MMA_M in (32, 64)"

        comptime if Self.a_tmem:
            var a_ = rebind[UInt32](a)
            comptime if Self.use_ws:
                bulk_mma_ws_ts_partial[
                    Self.mma_kind,
                    Self.operand_t,
                    b_BMN=Self.MMA_N,
                    b_BK=Self.padded_BK,
                    b_swizzle=Self.swizzle_b,
                    b_is_k_major=Self.transpose_b,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    tcgen05_mma_type=Self.tcgen05_mma_type,
                    mma_k=Self.MMA_K,
                    k_start=ks_start,
                    b_page_dense=Self.b_page_dense,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
            else:
                bulk_mma_partial[
                    Self.b_layout,
                    mma_k=Self.MMA_K,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    k_start=ks_start,
                    cta_group=Self.cta_group,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
        else:
            var a_ = rebind[MMASmemDescriptorPair](a)
            comptime if Self.use_ws:
                bulk_mma_ws_partial[
                    Self.mma_kind,
                    Self.operand_t,
                    Self.operand_t,
                    a_BMN=Self.a_bmn,
                    a_BK=Self.padded_BK,
                    a_swizzle=Self.swizzle_a,
                    a_is_k_major=True,
                    b_BMN=Self.b_bmn,
                    b_BK=Self.padded_BK,
                    b_swizzle=Self.swizzle_b,
                    b_is_k_major=Self.transpose_b,
                    num_k_mmas=ks_end - ks_start,
                    operand_size=Self.operand_size,
                    tcgen05_mma_type=Self.tcgen05_mma_type,
                    mma_k=Self.MMA_K,
                    k_start=ks_start,
                    b_page_dense=Self.b_page_dense,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )
            else:
                bulk_mma_ss_partial[
                    Self.a_layout,
                    Self.b_layout,
                    num_k_mmas=ks_end - ks_start,
                    mma_k=Self.MMA_K,
                    operand_size=Self.operand_size,
                    k_start=ks_start,
                    cta_group=Self.cta_group,
                ](
                    Self.idesc,
                    a_,
                    b,
                    c,
                    c_scale,
                    elect,
                    valid_k_mmas,
                )


def _build_mma[
    *, a_tmem: Bool, ws: Bool, partial: Bool
](
    kind: String,
    layout_a: Layout,
    layout_b: Layout,
    *,
    operand_size: Int,
    mma_k: Int,
    num_k_mmas: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
    tcgen05_mma_type: String = "",
) -> String:
    # Unified PTX builder for the tcgen05 MMA contraction, parameterized over the
    # three axes that previously spawned eight near-duplicate builders:
    #   * `a_tmem`  -- A operand source: TS (TMEM address, `[$7]`/`[%rab]`) vs
    #                  SS (SMEM descriptor pair in `%rda` from `$7`/`$8`).
    #   * `ws`      -- warp-specialized datapath: `tcgen05_mma_type` instruction
    #                  with NO zero-column mask, vs non-ws
    #                  `tcgen05.mma.cta_group::N.` with the `{$1,...}` mask.
    #   * `partial` -- partial-K tail: each block carries a SEPARATE warp-uniform
    #                  validity guard `%pv = (valid_k_mmas <= jj)`.
    #
    # `layout_a` is consulted only for SS (`a_tmem=False`); TS computes the A
    # column stride directly. `cta_group` matters only for non-ws (`ws=False`);
    # `tcgen05_mma_type` only for ws. `k_start` offsets the absolute k-index
    # (partial validity guards, or a k-slice of a full contraction).
    #
    # PREDICATION (the one rule that protects elect codegen): the form depends
    # ONLY on `partial`, never on `ws`.
    #   * full    -> single-instruction predication `@!%pj <instr>`. Keeping the
    #                MMA a straight-line predicated instruction is what lets the
    #                compiler recognize the single-lane `elect` and avoid emitting
    #                a `BSYNC.RECONVERGENT` into the SASS.
    #   * partial -> `@%pj bra skip{k}` + a SEPARATE `@!%pv` guard on the MMA. Two
    #                guards are needed (elect AND validity) and PTX allows one
    #                predicate per instruction, so elect uses the branch form
    #                while validity rides `%pv`. `%pv` is warp-uniform, so it never
    #                diverges and needs no reconvergence; `%pj` stays a pure
    #                function of the unmodified `elect`, preserving the codegen.
    # Blocks use ABSOLUTE k-index `jj = k_start + k` (for full, `k_start=0`).
    #
    # Plain `if` (not `comptime if`) is used throughout: the whole function is
    # comptime-evaluated, so a `comptime if` would add only a scope.
    # Pre-reserve so `mma` is heap-backed from the start: the comptime
    # interpreter cannot memcpy into a String's inline (SSO) buffer, so
    # appending a small fragment to a still-small string fails to interpret
    # ("can't get dst memory"). A heap-backed destination interprets fine.
    var mma = String(capacity_bytes=64)
    mma += "{\n"
    if not a_tmem:
        mma += ".reg .b64 %rda;\n"
    mma += ".reg .b64 %rdb;\n"
    mma += ".reg .s32 %ra;\n"
    if a_tmem:
        mma += ".reg .b32 %rab;\n"
    mma += ".reg .s32 %rb;\n"
    mma += ".reg .pred %pj;\n"
    mma += ".reg .pred %ps;\n"
    if partial:
        mma += ".reg .pred %pv;\n"
    mma += "setp.eq.s32 %pj, $6, 0;\n"

    # Instruction mnemonic (no predicate prefix; that is applied per-block below).
    var instr = tcgen05_mma_type + kind if ws else (
        "tcgen05.mma.cta_group::" + String(cta_group) + "." + kind
    )
    # Non-ws zero-column mask operand; absent for ws.
    var mask = (
        "{$1, $1, $1, $1}" if cta_group
        == 1 else "{$1, $1, $1, $1, $1, $1, $1, $1}"
    )
    # TMEM A column stride per k-mma (TS only).
    var a_stride = mma_k * operand_size // 4
    # Operand slot holding the warp-uniform `valid_k_mmas` (partial only): A
    # consumes `$7,$8` for SS but only `$7` for TS, so the next free slot differs.
    var valid_op = 8 if a_tmem else 9

    var operands: String
    for k in range(num_k_mmas):
        var jj = k_start + k
        # Warp-uniform validity guard: true once an absolute k-index lands past
        # the loaded region. Block jj == 0 is always loaded (valid_k_mmas >= 1).
        if partial and jj != 0:
            mma += String("setp.le.u32 %pv, $", valid_op, ", ", jj, ";\n")

        # A/B descriptor setup + the enable-input-d (`%ps`) scale predicate.
        if jj == 0:
            if not a_tmem:
                mma += "mov.b64 %rda, {$7, $8};\n"
            mma += "mov.b64 %rdb, {$4, $5};\n"
            # Absolute first block initializes the accumulator from c_scale ($3).
            mma += "setp.ne.b32 %ps, $3, 0;\n"
        else:
            var b_offset = (
                layout_b(IntTuple(0, mma_k * jj)) * operand_size
            ) >> 4
            if a_tmem:
                var a_offset = a_stride * jj
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += "mov.b32 %rab, %ra;\n"
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            elif partial:
                # SS-partial interleaving: A descriptor, then B descriptor.
                var a_offset = (
                    layout_a(IntTuple(0, mma_k * jj)) * operand_size
                ) >> 4
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += "mov.b64 %rda, {%ra, $8};\n"
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            else:
                # SS-full interleaving: both `add`s first, then both `mov`s.
                var a_offset = (
                    layout_a(IntTuple(0, mma_k * jj)) * operand_size
                ) >> 4
                mma += String("add.s32 %ra, $7, ", a_offset, ";\n")
                mma += String("add.s32 %rb, $4, ", b_offset, ";\n")
                mma += "mov.b64 %rda, {%ra, $8};\n"
                mma += "mov.b64 %rdb, {%rb, $5};\n"
            # First accumulate block (of the whole tile, or of a later stage)
            # pins %ps = 1; it then stays set for every subsequent block.
            if k == 0 or jj == 1:
                mma += "setp.ne.b32 %ps, 1, 0;\n"

        # Result + operand list.
        if a_tmem:
            var a_op = "$7" if jj == 0 else "%rab"
            operands = String(" [$0], [", a_op, "], %rdb, $2, ")
        else:
            operands = String(" [$0], %rda, %rdb, $2, ")
        if not ws:
            operands += mask + ", "
        operands += "%ps;\n"

        # Predication (form depends ONLY on `partial`; see header).
        if partial:
            mma += String("@%pj bra skip", k, ";\n")
            if jj != 0:
                mma += "@!%pv "
            mma += instr + operands
            mma += String("skip", k, ":\n")
        else:
            mma += "@!%pj " + instr + operands
    return mma + "}"


@always_inline("nodebug")
def bulk_mma[
    kind: UMMAKind,
    //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    mma_k: Int,
    operand_size: Int,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    """Issues a full-tile SS (both operands in SMEM) non-warp-specialized `tcgen05.mma` contraction.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma` instruction variant.
        layout_a: SMEM layout of the A operand tile, used to compute
            per-K-block A descriptor offsets.
        layout_b: SMEM layout of the B operand tile, used to compute
            per-K-block B descriptor offsets.
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over.
        mma_k: K-dimension tile size per MMA block, in elements.
        operand_size: Size in bytes of the A and B operand elements.
        cta_group: Number of cooperating CTAs, 1 or 2 (defaults to 1).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: SMEM descriptor pair for the A operand.
        b: SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
    """
    # Full-tile SS (both operands SMEM descriptors), non-ws contraction.
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=False, ws=False, partial=False](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


@always_inline("nodebug")
def bulk_mma[
    kind: UMMAKind,
    //,
    layout_b: Layout,
    *,
    mma_k: Int,
    num_k_mmas: Int,
    operand_size: Int,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    """Issues a full-tile TS (A in TMEM, B in SMEM) non-warp-specialized `tcgen05.mma` contraction.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma` instruction variant.
        layout_b: SMEM layout of the B operand tile, used to compute
            per-K-block B descriptor offsets.
        mma_k: K-dimension tile size per MMA block, in elements.
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over.
        operand_size: Size in bytes of the A and B operand elements.
        cta_group: Number of cooperating CTAs, 1 or 2 (defaults to 1).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: TMEM base address of the A operand.
        b: SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
    """
    # Full-tile TS (A in TMEM, B an SMEM descriptor), non-ws contraction.
    # `_build_mma` ignores `layout_a` for TS, so `layout_b` fills that slot.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=True, ws=False, partial=False](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a
    )


@always_inline("nodebug")
def bulk_mma_partial[
    kind: UMMAKind,
    //,
    layout_b: Layout,
    *,
    mma_k: Int,
    num_k_mmas: Int,
    operand_size: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    """Issues a partial-K TS contraction for a partially-loaded last KV tile, non-warp-specialized.

    Each block's MMA carries a warp-uniform validity guard derived from `valid_k_mmas`, kept separate from the `elect` predicate to preserve identical elect codegen.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma` instruction variant.
        layout_b: SMEM layout of the B operand tile, used to compute
            per-K-block B descriptor offsets.
        mma_k: K-dimension tile size per MMA block, in elements.
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over in this stage.
        operand_size: Size in bytes of the A and B operand elements.
        k_start: Absolute K-block index of the first block in this
            stage (defaults to 0).
        cta_group: Number of cooperating CTAs, 1 or 2 (defaults to 1).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: Un-offset (stage-0) TMEM base address of the A operand.
        b: Un-offset (stage-0) SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
        valid_k_mmas: Count of loaded `mma_k`-sized blocks; blocks whose
            absolute index reaches or exceeds this count are predicated
            off.
    """
    # P@V contraction for a partially-loaded last KV tile (TS, non-ws). Issues
    # this stage's `num_k_mmas` k-blocks (absolute indices
    # `k_start ..< k_start + num_k_mmas`) as a SINGLE fused inline-asm sequence.
    # Each block's MMA carries a warp-uniform validity guard derived from
    # `valid_k_mmas` (the count of loaded MMA_K blocks) kept entirely SEPARATE
    # from the `elect` predicate -- so the elect codegen is identical to the
    # full-tile `bulk_mma` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=True, ws=False, partial=True](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a, valid_k_mmas
    )


@always_inline("nodebug")
def bulk_mma_ss_partial[
    kind: UMMAKind,
    //,
    layout_a: Layout,
    layout_b: Layout,
    *,
    num_k_mmas: Int,
    mma_k: Int,
    operand_size: Int,
    k_start: Int = 0,
    cta_group: Int = 1,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    """Issues a partial-K SS contraction for a partially-loaded last KV tile, non-warp-specialized.

    Both A and B come from SMEM descriptors; each block's MMA carries a warp-uniform validity guard derived from `valid_k_mmas`, kept separate from `elect`.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma` instruction variant.
        layout_a: SMEM layout of the A operand tile, used to compute
            per-K-block A descriptor offsets.
        layout_b: SMEM layout of the B operand tile, used to compute
            per-K-block B descriptor offsets.
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over in this stage.
        mma_k: K-dimension tile size per MMA block, in elements.
        operand_size: Size in bytes of the A and B operand elements.
        k_start: Absolute K-block index of the first block in this
            stage (defaults to 0).
        cta_group: Number of cooperating CTAs, 1 or 2 (defaults to 1).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: Un-offset (stage-0) SMEM descriptor pair for the A operand.
        b: Un-offset (stage-0) SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
        valid_k_mmas: Count of loaded `mma_k`-sized blocks; blocks whose
            absolute index reaches or exceeds this count are predicated
            off.
    """
    # Contraction over a partially-loaded last KV tile, SS (non-ws) variant:
    # both A and B come from SMEM descriptors. Each block's MMA carries a
    # warp-uniform validity guard derived from `valid_k_mmas` kept entirely
    # SEPARATE from `elect` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime assert num_k_mmas >= 1
    comptime assert cta_group in (1, 2)
    comptime mma_string = _build_mma[a_tmem=False, ws=False, partial=True](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        cta_group=cta_group,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem),
        0,
        idesc,
        c_scale,
        b.lo,
        b.hi,
        elect,
        a.lo,
        a.hi,
        valid_k_mmas,
    )


# ------------------------------------------------------------------------------
# SM100 warp-specialized (.ws) MMA building blocks
# ------------------------------------------------------------------------------


@always_inline
def bulk_mma_ws[
    kind: UMMAKind,
    a_dtype: DType,
    b_dtype: DType,
    *,
    a_BMN: Int,
    a_BK: Int,
    a_swizzle: TensorMapSwizzle,
    a_is_k_major: Bool,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    b_page_dense: Bool = False,
    k_start: Int = 0,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    """Issues a full-tile SS (both operands in SMEM) warp-specialized `tcgen05.mma.ws` contraction.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma.ws` instruction
            variant.
        a_dtype: Element dtype of the A operand, used to derive the A
            SMEM tile layout.
        b_dtype: Element dtype of the B operand, used to derive the B
            SMEM tile layout.
        a_BMN: M (or N) dimension of the A operand tile in elements,
            used to derive the A layout.
        a_BK: K dimension of the A operand tile in elements, used to
            derive the A layout.
        a_swizzle: SMEM swizzle mode for the A tile.
        a_is_k_major: Whether A is stored k-major (`True`) or
            mn-major (`False`).
        b_BMN: M (or N) dimension of the B operand tile in elements,
            used to derive the B layout.
        b_BK: K dimension of the B operand tile in elements, used to
            derive the B layout.
        b_swizzle: SMEM swizzle mode for the B tile.
        b_is_k_major: Whether B is stored k-major (`True`) or
            mn-major (`False`).
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over.
        operand_size: Size in bytes of the A and B operand elements.
        tcgen05_mma_type: `tcgen05.mma.ws` instruction string prefix,
            including the CTA-group selector.
        mma_k: K-dimension tile size per MMA block, in elements
            (defaults to 16).
        b_page_dense: Whether B uses the row-major page-fold layout
            (defaults to `False`).
        k_start: Absolute K-block index of the first block in this tile
            (defaults to 0).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: SMEM descriptor pair for the A operand.
        b: SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
    """
    # Full-tile SS, warp-specialized. The tile layouts are computed from the
    # dtype/tile params (`_build_mma` takes `Layout` directly). `b_page_dense`
    # selects the row-major page-fold layout for the B operand (K / Q@K' is
    # k-major; the advance crosses a depth chunk by `_CM_NUM_ROWS*gran` instead
    # of `BN*gran`, derived from this layout). `k_start` issues a slice of the
    # contraction (absolute k-mmas `k_start ..< k_start + num_k_mmas`) against
    # the un-offset full-tile descriptors; slices with `k_start > 0` always
    # accumulate (`c_scale` only applies to the absolute first k-mma).
    comptime layout_a = tile_layout_k_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]() if a_is_k_major else tile_layout_mn_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]()
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=False, ws=True, partial=False](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a.lo, a.hi
    )


# ---- TS (TMEM-SMEM) .ws MMA building blocks ----


@always_inline
def bulk_mma_ws_ts[
    kind: UMMAKind,
    b_dtype: DType,
    *,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
):
    """Issues a full-tile TS (A in TMEM, B in SMEM) warp-specialized `tcgen05.mma.ws` contraction.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma.ws` instruction
            variant.
        b_dtype: Element dtype of the B operand, used to derive the B
            SMEM tile layout.
        b_BMN: M (or N) dimension of the B operand tile in elements,
            used to derive the B layout.
        b_BK: K dimension of the B operand tile in elements, used to
            derive the B layout.
        b_swizzle: SMEM swizzle mode for the B tile.
        b_is_k_major: Whether B is stored k-major (`True`) or
            mn-major (`False`).
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over.
        operand_size: Size in bytes of the A and B operand elements.
        tcgen05_mma_type: `tcgen05.mma.ws` instruction string prefix,
            including the CTA-group selector.
        mma_k: K-dimension tile size per MMA block, in elements
            (defaults to 16).
        b_page_dense: Whether B uses the row-major page-fold layout
            (defaults to `False`).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator
            and operand dtypes and the output tile shape.
        a: TMEM base address of the A operand.
        b: SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
    """
    # Full-tile TS, warp-specialized. `a` is a single TMEM base ($7); `_build_mma`
    # computes each k-tile's column offset in-PTX (`add.s32 %ra, $7, k*stride`),
    # so the old per-tile operand ladder is gone.
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=True, ws=True, partial=False](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a
    )


# ---- partial-K (.ws) MMA building blocks ----


@always_inline
def bulk_mma_ws_partial[
    kind: UMMAKind,
    a_dtype: DType,
    b_dtype: DType,
    *,
    a_BMN: Int,
    a_BK: Int,
    a_swizzle: TensorMapSwizzle,
    a_is_k_major: Bool,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    k_start: Int = 0,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: MMASmemDescriptorPair,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    """Issues a partial-K SS warp-specialized contraction for a partially-loaded last KV tile.

    Both A and B come from SMEM descriptors; each block's MMA carries a warp-uniform validity guard derived from `valid_k_mmas`, kept separate from `elect`.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma.ws` instruction
            variant.
        a_dtype: Element dtype of the A operand, used to derive the A
            SMEM tile layout.
        b_dtype: Element dtype of the B operand, used to derive the B
            SMEM tile layout.
        a_BMN: M (or N) dimension of the A operand tile in elements,
            used to derive the A layout.
        a_BK: K dimension of the A operand tile in elements, used to
            derive the A layout.
        a_swizzle: SMEM swizzle mode for the A tile.
        a_is_k_major: Whether A is stored k-major (`True`) or
            mn-major (`False`).
        b_BMN: M (or N) dimension of the B operand tile in elements,
            used to derive the B layout.
        b_BK: K dimension of the B operand tile in elements, used to
            derive the B layout.
        b_swizzle: SMEM swizzle mode for the B tile.
        b_is_k_major: Whether B is stored k-major (`True`) or
            mn-major (`False`).
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over in this stage.
        operand_size: Size in bytes of the A and B operand elements.
        tcgen05_mma_type: `tcgen05.mma.ws` instruction string prefix,
            including the CTA-group selector.
        mma_k: K-dimension tile size per MMA block, in elements
            (defaults to 16).
        k_start: Absolute K-block index of the first block in this
            stage (defaults to 0).
        b_page_dense: Whether B uses the row-major page-fold layout
            (defaults to `False`).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator and
            operand dtypes and the output tile shape.
        a: Un-offset (stage-0) SMEM descriptor pair for the A operand.
        b: Un-offset (stage-0) SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
        valid_k_mmas: Count of loaded `mma_k`-sized blocks; blocks whose
            absolute index reaches or exceeds this count are predicated
            off.
    """
    # P@V contraction for a partially-loaded last KV tile, SS warp-specialized:
    # both A and B come from SMEM descriptors. Each block's MMA carries a
    # warp-uniform validity guard derived from `valid_k_mmas` kept entirely
    # SEPARATE from `elect` (no BSYNC.RECONVERGENT). `a`/`b` are the un-offset
    # (stage-0) bases; the builder applies absolute per-block offsets.
    comptime layout_a = tile_layout_k_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]() if a_is_k_major else tile_layout_mn_major[
        a_dtype, a_BMN, a_BK, a_swizzle
    ]()
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=False, ws=True, partial=True](
        String(kind),
        layout_a,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem),
        0,
        idesc,
        c_scale,
        b.lo,
        b.hi,
        elect,
        a.lo,
        a.hi,
        valid_k_mmas,
    )


@always_inline
def bulk_mma_ws_ts_partial[
    kind: UMMAKind,
    b_dtype: DType,
    *,
    b_BMN: Int,
    b_BK: Int,
    b_swizzle: TensorMapSwizzle,
    b_is_k_major: Bool,
    num_k_mmas: Int,
    operand_size: Int,
    tcgen05_mma_type: String,
    mma_k: Int = 16,
    k_start: Int = 0,
    b_page_dense: Bool = False,
](
    idesc: UMMAInsDescriptor[kind],
    a: UInt32,
    b: MMASmemDescriptorPair,
    c_tmem: UInt32,
    c_scale: UInt32,
    elect: Int32,
    valid_k_mmas: UInt32,
):
    """Issues a partial-K TS warp-specialized contraction for a partially-loaded last KV tile.

    `a` is the un-offset TMEM base; each block's absolute column offset is computed in-PTX, and a `%pv` validity guard is kept separate from `elect`.

    Parameters:
        kind: `UMMAKind` selecting the `tcgen05.mma.ws` instruction
            variant.
        b_dtype: Element dtype of the B operand, used to derive the B
            SMEM tile layout.
        b_BMN: M (or N) dimension of the B operand tile in elements,
            used to derive the B layout.
        b_BK: K dimension of the B operand tile in elements, used to
            derive the B layout.
        b_swizzle: SMEM swizzle mode for the B tile.
        b_is_k_major: Whether B is stored k-major (`True`) or
            mn-major (`False`).
        num_k_mmas: Number of `mma_k`-sized K-dimension blocks to
            contract over in this stage.
        operand_size: Size in bytes of the A and B operand elements.
        tcgen05_mma_type: `tcgen05.mma.ws` instruction string prefix,
            including the CTA-group selector.
        mma_k: K-dimension tile size per MMA block, in elements
            (defaults to 16).
        k_start: Absolute K-block index of the first block in this
            stage (defaults to 0).
        b_page_dense: Whether B uses the row-major page-fold layout
            (defaults to `False`).

    Args:
        idesc: UMMA instruction descriptor encoding the accumulator
            and operand dtypes and the output tile shape.
        a: Un-offset (stage-0) TMEM base address of the A operand.
        b: Un-offset (stage-0) SMEM descriptor pair for the B operand.
        c_tmem: TMEM base address of the output accumulator `C`.
        c_scale: Accumulator overwrite/accumulate selector; 0 overwrites. See
            the struct docstring.
        elect: `elect()` result selecting the single thread that issues
            the MMA.
        valid_k_mmas: Count of loaded `mma_k`-sized blocks; blocks
            whose absolute index reaches or exceeds this count are
            predicated off.
    """
    # P@V contraction for a partially-loaded last KV tile, TS warp-specialized.
    # `a` is the un-offset (stage-0) TMEM base ($7); `_build_mma` computes each
    # block's ABSOLUTE column offset (`a_stride * (k_start + k)`) in-PTX, so the
    # old per-tile operand ladder is gone. `valid_k_mmas` rides $8 (A uses only
    # $7) and gates each block via a `%pv` guard kept SEPARATE from `elect` (no
    # BSYNC.RECONVERGENT).
    comptime assert num_k_mmas >= 1 and num_k_mmas <= 16
    comptime layout_b = tile_layout_k_major[
        b_dtype, b_BMN, b_BK, b_swizzle
    ]() if b_is_k_major else tile_layout_mn_major[
        b_dtype, b_BMN, b_BK, b_swizzle, page_dense=b_page_dense
    ]()
    comptime mma_string = _build_mma[a_tmem=True, ws=True, partial=True](
        String(kind),
        layout_b,
        layout_b,
        operand_size=operand_size,
        mma_k=mma_k,
        num_k_mmas=num_k_mmas,
        k_start=k_start,
        tcgen05_mma_type=tcgen05_mma_type,
    )

    inlined_assembly[mma_string, NoneType, constraints="r,r,r,r,r,r,r,r,r"](
        broadcast(c_tmem), 0, idesc, c_scale, b.lo, b.hi, elect, a, valid_k_mmas
    )


@always_inline
def llvm_opaque_tid() -> UInt32:
    """Returns the opaque thread ID via the `llvm.nvvm.read.ptx.sreg.tid.x` intrinsic.
    """
    return llvm_intrinsic[
        "llvm.nvvm.read.ptx.sreg.tid.x", UInt32, has_side_effect=True
    ]()


@always_inline
def intrin_ftz[intrin: String](a: Float32, b: Float32) -> Float32:
    """Wraps a flush-to-zero (FTZ) binary float32 PTX intrinsic."""
    return inlined_assembly[
        String(intrin, ".ftz.f32 $0, $1, $2;"),
        Float32,
        constraints="=f,f,f",
        has_side_effect=False,
    ](a, b)


@always_inline
def intrin[intrin: String](a: Float32, b: Float32, c: Float32) -> Float32:
    """Wraps a ternary float32 PTX intrinsic (e.g. `max.f32`)."""
    return inlined_assembly[
        String(intrin, ".f32 $0, $1, $2, $3;"),
        Float32,
        constraints="=f,f,f,f",
        has_side_effect=False,
    ](a, b, c)


@always_inline
def intrin_ftz_x2[
    intrin: String
](a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Wraps a flush-to-zero (FTZ) binary `f32x2` PTX intrinsic."""
    return inlined_assembly[
        String(intrin, ".ftz.f32x2 $0, $1, $2;"),
        SIMD[.float32, 2],
        constraints="=l,l,l",
        has_side_effect=False,
    ](a, b)


@always_inline
def add_ftz(a: Float32, b: Float32) -> Float32:
    """Returns the flush-to-zero sum of two float32 values.

    Args:
        a: First float32 addend.
        b: Second float32 addend.
    """
    return intrin_ftz["add"](a, b)


@always_inline
def sub_ftz(a: Float32, b: Float32) -> Float32:
    """Returns the flush-to-zero difference of two float32 values.

    Args:
        a: The minuend float32 value.
        b: The subtrahend float32 value.
    """
    return intrin_ftz["sub"](a, b)


@always_inline
def mul_ftz(a: Float32, b: Float32) -> Float32:
    """Returns the flush-to-zero product of two float32 values."""
    return intrin_ftz["mul"](a, b)


@always_inline
def max_ftz(a: Float32, b: Float32) -> Float32:
    """Returns the flush-to-zero maximum of two float32 values."""
    return intrin_ftz["max"](a, b)


@always_inline
def max_ftz(a: Float32, b: Float32, c: Float32) -> Float32:
    """Returns the flush-to-zero maximum of three float32 values."""
    return intrin["max.ftz"](a, b, c)


@always_inline
def add_ftz(a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Returns the flush-to-zero sum of two `f32x2` vectors.

    Args:
        a: First `f32x2` addend vector.
        b: Second `f32x2` addend vector.
    """
    return intrin_ftz_x2["add"](a, b)


@always_inline
def sub_ftz(a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Returns the flush-to-zero difference of two `f32x2` vectors.

    Args:
        a: First `f32x2` minuend vector.
        b: Second `f32x2` subtrahend vector.
    """
    return intrin_ftz_x2["sub"](a, b)


@always_inline
def mul_ftz(a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Returns the flush-to-zero product of two `f32x2` vectors."""
    return intrin_ftz_x2["mul"](a, b)


@always_inline
def add_ftz_rm(a: SIMD[.float32, 2], b: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Returns the round-to-nearest-even flush-to-zero sum of two `f32x2` vectors.

    Args:
        a: First `f32x2` addend vector.
        b: Second `f32x2` addend vector.
    """
    return intrin_ftz_x2["add.rm"](a, b)


@always_inline
def fma_ftz(a: Float32, b: Float32, c: Float32) -> Float32:
    return intrin["fma.rn.ftz"](a, b, c)


@always_inline
def fma_ftz(
    a: SIMD[.float32, 2],
    b: SIMD[.float32, 2],
    c: SIMD[.float32, 2],
) -> SIMD[.float32, 2]:
    """Returns the flush-to-zero fused multiply-add `a * b + c` for `f32x2` vectors.
    """
    return inlined_assembly[
        "fma.rn.ftz.f32x2 $0, $1, $2, $3;",
        SIMD[.float32, 2],
        constraints="=l,l,l,l",
        has_side_effect=False,
    ](a, b, c)


def _ptx_f32_literal[value: Float32]() -> String:
    """Formats `value` as a PTX `0f`-prefixed 32-bit float literal.

    PTX spells a float immediate as its raw IEEE-754 bit pattern in exactly
    eight hex digits, so this pads to width rather than using `hex()` (which
    trims leading zeros and carries an `0x` prefix).

    Parameters:
        value: The float32 constant to encode.

    Returns:
        The PTX literal, e.g. `0fC61C4000` for -10000.0.
    """
    comptime bits = bitcast[.uint32](value)
    var lit = String("0f")

    comptime for i in range(8):
        comptime nibble = Int((bits >> UInt32(4 * (7 - i))) & UInt32(0xF))
        # '0'-'9' then 'A'-'F'.
        comptime code = (48 + nibble) if nibble < 10 else (55 + nibble)
        lit += chr(code)

    return lit


def _mask_select8_asm[byte_idx: Int]() -> String:
    """Builds the PTX body for `mask_select8`.

    Emits bits 0-6 as 7 `and.b32` + `setp.eq.u32 ...,0` followed by 7
    `selp.f32`, then bit 7 as a separate `and`/`setp`/`selp` that reuses %p0, so
    at most 7 predicates (the full P0-P6 file) are ever live, never 8. Everything
    stays inside one `{ ... }` block so the bit-extraction is adjacent to the
    selects. This mirrors the cold region ptxas already emits for this mask:
    `R2P ...,0x7f` for the 7-bit group + `LOP3.LUT ...,0x80` for the 8th bit,
    both consumed by `selp.f32 ...,<MASK_VALUE>,score`.

    Parameters:
        byte_idx: Which mask byte (0..3) this block applies.

    Returns:
        The assembled PTX string.
    """
    comptime mv = _ptx_f32_literal[Float32(MASK_VALUE)]()
    # Bits 0-6: the R2P group. 7 (`and` + `setp.eq...,0`) then 7 `selp`, so at
    # most 7 predicates are live; ptxas folds the 7 `setp` into `R2P ...,0x7f`.
    var asm = String("{\n.reg .pred %p<7>;\n.reg .b32 %t<7>;\n")

    comptime for j in range(7):
        asm += String(
            "and.b32 %t",
            j,
            ", $16, ",
            hex(UInt32(1) << UInt32(8 * byte_idx + j)),
            ";\nsetp.eq.u32 %p",
            j,
            ", %t",
            j,
            ", 0;\n",
        )

    comptime for j in range(7):
        asm += String("selp.f32 $", j, ", ", mv, ", $", 8 + j, ", %p", j, ";\n")

    # Bit 7 (the 8th lane): reuse %p0/%t0, free after the selps above, so this
    # never pushes liveness to 8. This is the cold region's `LOP3.LUT ...,0x80`.
    asm += String(
        "and.b32 %t0, $16, ",
        hex(UInt32(1) << UInt32(8 * byte_idx + 7)),
        ";\nsetp.eq.u32 %p0, %t0, 0;\nselp.f32 $7, ",
        mv,
        ", $15, %p0;\n",
    )

    asm += "}"
    return asm


@always_inline
def mask_select8[
    byte_idx: Int
](
    s0: Float32,
    s1: Float32,
    s2: Float32,
    s3: Float32,
    s4: Float32,
    s5: Float32,
    s6: Float32,
    s7: Float32,
    mask_bits: UInt32,
) -> _RegisterPackType[
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
    Float32,
]:
    """Masks 8 contiguous scores against one byte of a 32-column bitmask.

    Lane `j` keeps its score if bit `8*byte_idx + j` of `mask_bits` is set,
    otherwise it becomes `MASK_VALUE`. The 8 `and`/`setp`/`selp` are
    confined to a single opaque PTX block so the bit-extraction sits adjacent to
    the selects: the predicate live-set stays bounded (avoiding the wide
    up-front bit pre-extraction that spills) and the shape stays `R2P`-eligible.

    Parameters:
        byte_idx: Which mask byte to apply (0..3), i.e. columns
            `8*byte_idx .. 8*byte_idx + 7`.

    Args:
        s0: Already-scaled score for lane 0.
        s1: Already-scaled score for lane 1.
        s2: Already-scaled score for lane 2.
        s3: Already-scaled score for lane 3.
        s4: Already-scaled score for lane 4.
        s5: Already-scaled score for lane 5.
        s6: Already-scaled score for lane 6.
        s7: Already-scaled score for lane 7.
        mask_bits: Packed 32-column visibility mask.

    Returns:
        The 8 masked scores, register-packed (index `[0] .. [7]`).
    """
    comptime asm = _mask_select8_asm[byte_idx]()
    return inlined_assembly[
        asm,
        _RegisterPackType[
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
            Float32,
        ],
        constraints="=f,=f,=f,=f,=f,=f,=f,=f,f,f,f,f,f,f,f,f,r",
        has_side_effect=False,
    ](s0, s1, s2, s3, s4, s5, s6, s7, mask_bits)


@always_inline
def exp2_emulation[
    use_exp2_emulation: Bool = True
](x: SIMD[.float32, 2]) -> SIMD[.float32, 2]:
    """Computes `2^x` for an `f32x2` vector via a degree-3 polynomial approximation.

    When `use_exp2_emulation` is False, falls back to the standard `exp2` intrinsic.
    """
    comptime if use_exp2_emulation:
        comptime fp32_round_int = SIMD[.float32, 2]((1 << 23) + (1 << 22))
        var clamped = max(x, -FP32_EXP_BIAS)
        # We want to round down here, so that the fractional part is in [0, 1)
        var rounded = add_ftz_rm(clamped, fp32_round_int)
        var rounded_back = sub_ftz(rounded, fp32_round_int)
        var frac = sub_ftz(clamped, rounded_back)
        # Degree-3 polynomial approximation of `2^x` on `x ∈ [0, 1)`.
        # Coefficients lifted from Tri Dao's FlashAttention-3
        # `exp2_emulated` (Dao-AILab/flash-attention, `flash_fwd_kernel*`)
        # — fit by minimax over the unit interval.
        # Tri Dao assumes x <= 127.0 and y <= 127.0
        var frac_ex2 = fma_ftz(
            fma_ftz(
                fma_ftz(
                    0.077119089663028717041015625,
                    frac,
                    0.227564394474029541015625,
                ),
                frac,
                0.695146143436431884765625,
            ),
            frac,
            1.0,
        )
        # The integer floor of x & y are now in the last 8 bits of xy_rounded
        # We want the next 2 ops to round to nearest even. The rounding mode is important.
        return bitcast[.float32](
            bitcast[.int32](frac_ex2) + (bitcast[.int32](rounded) << 23)
        )
    else:
        return exp2(x)


@always_inline
def elect_mma_arrive[
    cta_group: Int = 1
](mbar_ptr: UnsafePointer[address_space=.SHARED, ...], elect: Int32,):
    """Arrive at the mbar pointer for the MMA instruction.

    Parameters:
        cta_group: Number of ctas used by MMA.

    Args:
        mbar_ptr: Pointer to the mbar.
        elect: `elect()`.
    """

    comptime assert cta_group in (1, 2), String(
        "Unsupported cta group: ", cta_group
    )

    comptime type = mbar_ptr.T
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

    inlined_assembly[
        """{
        .reg .pred %p;
        setp.eq.s32  %p, $1, 0;
        @!%p tcgen05.commit.cta_group::"""
        + String(cta_group)
        + """.mbarrier::arrive::one.shared::cluster.b64 [$0];
        }""",
        NoneType,
        constraints="r, r",
    ](Int32(Int(mbar_ptr)), elect)


@always_inline
def expect_bytes_pred(
    mbar_ptr: UnsafePointer[address_space=.SHARED, ...],
    bytes: Int32,
    pred: Int32,
):
    """Issue `mbarrier.arrive.expect_tx.shared::cta.b64` predicated on
    `pred != 0`.

    Equivalent to:

        if pred != 0:
            mbar_ptr[].expect_bytes(bytes)

    but folds the runtime branch into a single PTX `@%p` predicate
    on the `mbarrier.arrive.expect_tx` instruction: no Mojo-level
    `if`, no SASS branch, no warp divergence.

    Args:
        mbar_ptr: Pointer to the shared-memory mbarrier (8-byte slot).
        bytes: Expected transaction byte count for this barrier.
        pred: Runtime predicate (typically the result of `elect()` or
            an `elect()`-derived `elect_mask`); the PTX instruction is
            skipped when this is 0.
    """

    comptime type = mbar_ptr.T
    comptime assert size_of[type]() == 8, "mbar_ptr must be 8 bytes"

    inlined_assembly[
        """{
        .reg .pred %p;
        .reg .b64 %state;
        setp.ne.s32 %p, $2, 0;
        @%p mbarrier.arrive.expect_tx.shared::cta.b64 %state, [$0], $1;
        }""",
        NoneType,
        constraints="r,r,r",
    ](Int32(Int(mbar_ptr)), bytes, pred)


@always_inline
def store_global_pred[
    dtype: DType,
    address_space: AddressSpace,
    //,
](
    ptr: UnsafePointer[mut=True, Scalar[dtype], _, address_space=address_space],
    value: Scalar[dtype],
    pred: Int32,
):
    """Issue a global store predicated on `pred != 0`.

    Equivalent to:

        if pred != 0:
            ptr[] = value

    but folds the guard into a PTX `@%p` predicate on the store, so ptxas emits
    a single predicated `STG` instead of the `BSSY`/`BRA`/`NOP`/`BSYNC`
    reconvergence quartet an `if` costs.

    That is a real trade, not a free win. The `if` form gives ptxas a basic
    block boundary, which caps the scheduling region; this form does not, so
    ptxas may hoist the producers of `value` across neighbouring code and spill.
    Measure spill counts, not just the instruction count, when switching a site
    to this helper.

    There is no `~{memory}` clobber, matching `expect_bytes_pred` and the
    `st.shared.v4.b32` helper above: adding one would force LLVM to reload every
    value it has cached from memory at each call site. Callers must therefore
    not read back what they store here without an explicit fence.

    Parameters:
        dtype: Element dtype of the store; must be 4 or 8 bytes (inferred).
        address_space: Address space of `ptr` (inferred).

    Args:
        ptr: Destination address. Must point into global memory -- the
            instruction is `st.global`, so a generic pointer into shared or
            local memory is undefined behaviour rather than a compile error.
        value: The value to store when `pred` is nonzero.
        pred: Runtime predicate; the store is skipped when this is 0.
    """
    comptime assert (
        address_space == .GENERIC or address_space == .GLOBAL
    ), "store_global_pred emits `st.global`; the pointer must address gmem"
    comptime size = size_of[dtype]()
    comptime assert size in (4, 8), (
        "store_global_pred handles 4- and 8-byte scalars; got a "
        + String(size)
        + "-byte dtype"
    )
    comptime bits = "b32" if size == 4 else "b64"
    # Bitcast to the same-width unsigned integer so one `.bXX`/register-class
    # pair covers float and integer dtypes alike; `st.global.bXX` is
    # bit-preserving.
    comptime word_type = DType.uint32 if size == 4 else DType.uint64
    comptime word_constraint = "r" if size == 4 else "l"

    inlined_assembly[
        """{
        .reg .pred %p;
        setp.ne.s32 %p, $2, 0;
        @%p st.global."""
        + bits
        + """ [$0], $1;
        }""",
        NoneType,
        constraints="l," + word_constraint + ",r",
    ](ptr, bitcast[word_type](value), pred)


@always_inline
def maximum[
    BN: Int, //, *, width: Int = 4
](x: Array[Float32, BN], out res: StaticTuple[Float32, width],):
    """Reduces `BN` float32 scores into `width` lane-maxima using FTZ max."""
    comptime assert 3 * width <= BN
    res = {}

    comptime for w in range(width):
        res[w] = max_ftz(
            x[3 * w],
            x[3 * w + 1],
            x[3 * w + 2],
        )

    # max idx = 3 * (width-1) + 2 = 3*width - 1
    comptime remaining_iters = BN - 3 * width
    comptime num_iters = remaining_iters // (2 * width)

    comptime for i in range(num_iters):
        comptime col = i * 2 * width + 3 * width

        comptime for w in range(width):
            res[w] = max_ftz(
                res[w],
                x[col + 2 * w],
                x[col + 2 * w + 1],
            )

    comptime remainder_base = 3 * width + 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    comptime for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            x[remainder_base + 2 * w],
            x[remainder_base + 2 * w + 1],
        )

    comptime if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], x[BN - 1])


@always_inline
def maximum[
    BN: Int, //, *, width: Int = 4
](
    x: Array[Float32, BN],
    init: StaticTuple[Float32, width],
    out res: StaticTuple[Float32, width],
):
    """Reduces `BN` float32 scores into `width` lane-maxima, seeded from `init`.
    """
    res = init

    # unroll (using SIMD) to break up dependency chain
    comptime num_iters = BN // (2 * width)

    comptime for i in range(num_iters):
        comptime for w in range(width):
            comptime j = i * 2 * width + 2 * w
            res[w] = max_ftz(res[w], x[j], x[j + 1])

    comptime remainder_base = 2 * width * num_iters
    comptime end_iters = (BN - remainder_base) // 2

    comptime for w in range(end_iters):
        res[w] = max_ftz(
            res[w],
            x[remainder_base + 2 * w],
            x[remainder_base + 2 * w + 1],
        )

    comptime if (BN - remainder_base) % 2 == 1:
        res[end_iters] = max_ftz(res[end_iters], x[BN - 1])


@always_inline
def maximum(x: StaticTuple[Float32, 4]) -> Float32:
    """Returns the maximum of four float32 values packed in a `StaticTuple`."""
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3])


@always_inline
def maximum(x: StaticTuple[Float32, 4], init: Float32) -> Float32:
    """Returns the FTZ maximum of a `StaticTuple[4]` and an initial value."""
    return max_ftz(max_ftz(x[0], x[1], x[2]), x[3], init)


@always_inline
def maximum(x: StaticTuple[Float32, 8]) -> Float32:
    """Returns the maximum of eight float32 values packed in a `StaticTuple`."""
    var a = max_ftz(x[0], x[1], x[2])
    var b = max_ftz(x[3], x[4], x[5])
    var c = max_ftz(x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
def maximum(x: StaticTuple[Float32, 8], init: Float32) -> Float32:
    """Returns the FTZ maximum of a `StaticTuple[8]` and an initial value."""
    var a = max_ftz(init, x[0], x[1])
    var b = max_ftz(x[2], x[3], x[4])
    var c = max_ftz(x[5], x[6], x[7])
    return max_ftz(a, b, c)


@always_inline
def sum[
    dtype: DType, BN: Int, //, *, width: Int = 8
](x: LocalTensor[dtype, row_major[BN]()]) -> SIMD[dtype, 2]:
    """Reduces a `BN`-element local tensor into a width-2 SIMD vector via vectorized accumulation.

    Parameters:
        dtype: Element dtype of the input tensor (inferred).
        BN: Number of elements in the input tensor; must be divisible by
            `width` (inferred).
        width: Vectorization width for the accumulation (defaults to 8).

    Args:
        x: Local tensor of `BN` elements to reduce.
    """
    comptime assert BN % width == 0
    var vx = x.vectorize[width]()
    var acc = vx[0]

    # unroll (using SIMD) to break up dependency chain
    comptime for i in range(1, BN // width):
        acc += vx[i]

    return acc.reduce_add[size_out=2]()
    # return rebind[SIMD[dtype,width]](acc)


struct StagedPipeline[num_kv_stages: Int, num_qk_stages: Int = 1](
    TrivialRegisterPassable
):
    """
    Unified pipeline for K, V, and KV tile barrier management.

    `num_kv_stages` refers to how many KV tile buffers we have for pipelining.
    `num_qk_stages` controls K loading staging for Q@K' MMA:
      - K can be loaded in num_qk_stages chunks, allowing MMA to start earlier
      - V always uses qk_stages=1 (complete tile required)

    Total stages = num_kv_stages * num_qk_stages.

    Parameters:
        num_kv_stages: Number of double-buffered KV tile buffers used for
            pipelining.
        num_qk_stages: Number of K-loading sub-stages per KV stage for Q@K'
            MMA staging; V always uses 1 (defaults to 1).
    """

    comptime num_stages: Int = Self.num_kv_stages * Self.num_qk_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def producer_mbar[qk_stage: Int = 0](self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.mbar + UInt32(Self.num_qk_stages) * idx + qk_stage

    @always_inline
    def consumer_mbar[qk_stage: Int = 0](self, idx: UInt32) -> MBarType:
        comptime const_offset = qk_stage + Self.num_stages
        return self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset

    @always_inline
    def consumer_mbar[qk_stage: Int = 0](self) -> MBarType:
        return self.consumer_mbar[qk_stage](self.state.index())

    @always_inline("nodebug")
    def producer_acquire[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait until consumer has released the buffer for this stage.

        Parameters:
            qk_stage: K-loading sub-stage to acquire (defaults to the last
                sub-stage).
        """
        self.consumer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_wait[qk_stage: Int = Self.num_qk_stages - 1](self):
        """Wait for producer to complete this stage.

        Parameters:
            qk_stage: K-loading sub-stage to wait on (defaults to the last
                sub-stage).
        """
        self.producer_mbar[qk_stage]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def consumer_release[
        qk_stage: Int = Self.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release the buffer after consuming this stage.

        Parameters:
            qk_stage: K-loading sub-stage to release (defaults to the last
                sub-stage).

        Args:
            e: `elect()` result selecting the single thread that arrives on
                the mbarrier.
        """
        elect_mma_arrive(self.consumer_mbar[qk_stage](), e)

        comptime if qk_stage == Self.num_qk_stages - 1:
            self.state.step()

    @always_inline("nodebug")
    def consumer_release_at(self, idx: UInt32, e: Int32):
        """Release a specific stage without stepping the pipeline state.

        Used for deferred V release in shared KV mode: V_{n-1} must be
        released while holding K_n, which is at a different pipeline index.

        Args:
            idx: Pipeline index of the stage to release.
            e: `elect()` result selecting the single thread that arrives
                on the mbarrier.
        """
        comptime qk_stage = Self.num_qk_stages - 1
        comptime const_offset = qk_stage + Self.num_stages
        var mbar = self.mbar + UInt32(Self.num_qk_stages) * idx + const_offset
        elect_mma_arrive(mbar, e)

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.num_qk_stages * Self.num_kv_stages)


# Backward-compatible type aliases
comptime KPipeline = StagedPipeline
comptime VPipeline = StagedPipeline[_, 1]
comptime KVPipeline = StagedPipeline


struct TMADestination[dtype: DType, smem_elems: Int](TrivialRegisterPassable):
    """Pairs a shared memory TileTensor with a barrier for TMA operations.

    The stored TileTensor uses a flat `row_major[smem_elems]()` layout.
    TMA only uses `.ptr`.

    Parameters:
        dtype: Element dtype of the shared memory tile.
        smem_elems: Number of elements in the flat shared memory buffer
            used by the TileTensor.
    """

    comptime SmemType = TileTensor[
        Self.dtype,
        type_of(tt_row_major[Self.smem_elems]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ]

    @__allow_legacy_any_origin_fields
    var mbar: MBarType

    @__allow_legacy_any_origin_fields
    var smem: Self.SmemType

    @always_inline
    def __init__(
        out self,
        mbar: MBarType,
        smem: Self.SmemType,
    ):
        self.mbar = mbar
        self.smem = smem


struct TMAProducerPipeline[dtype: DType, config: FA4Config, is_k: Bool = True](
    TrivialRegisterPassable
):
    """Unified producer pipeline for K and V TMA loading.

    K loading (is_k=True): Can be staged (num_qk_stages chunks), uses k_major layout.
    V loading (is_k=False): Always complete (qk_stage=0), uses mn_major layout.

    Parameters:
        dtype: Element dtype of the K or V tile being loaded via TMA.
        config: FlashAttention-4 configuration providing tile sizes,
            swizzle mode, and stage counts.
        is_k: Whether this pipeline loads the K operand (`True`, k-major
            layout with staged `qk_stages`) or the V operand (`False`,
            mn-major layout with `qk_stage=0`); defaults to `True`.
    """

    # Compute layout first using comptime, then use it in type.
    # For pair-CTA: K uses k_rows_per_cta (BN/2), V uses v_cols_per_cta (ov/2).
    comptime tile_layout: Layout = tile_layout_k_major[
        Self.dtype,
        Self.config.k_rows_per_cta(),
        Self.config.BK0,
        Self.config.swizzle_mode,
    ]() if Self.is_k else tile_layout_mn_major[
        Self.dtype,
        Self.config.v_cols_per_cta(),
        Self.config.BK1,
        Self.config.swizzle_mode,
    ]()

    comptime PairType = TMADestination[Self.dtype, Self.tile_layout.size()]
    comptime elements: Int = Self.tile_layout.size()
    comptime elements_full: Int = Self.elements * Self.config.num_qk_stages if Self.is_k else Self.elements
    comptime tile_bytes: Int = Self.elements * size_of[Self.dtype]()
    # Backward-compatible aliases
    comptime bytes = Self.tile_bytes
    comptime SMemType = SharedMemPointer[Scalar[Self.dtype]]

    # K uses full staging, V uses qk_stages=1
    comptime num_qk_stages_effective: Int = Self.config.num_qk_stages if Self.is_k else 1

    var pipeline: StagedPipeline[
        Self.config.num_kv_stages, Self.num_qk_stages_effective
    ]

    @__allow_legacy_any_origin_fields
    var smem: Self.SMemType

    @always_inline
    def __init__(out self, mbar: MBarType, smem: Self.SMemType):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_qk_depth % Self.config.num_qk_stages == 0
            ), "padded_qk_depth must be divisible by num_qk_stages"
        self.pipeline = {mbar}
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    def __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: Self.SMemType,
    ):
        comptime if Self.is_k:
            comptime assert (
                Self.config.padded_qk_depth % Self.config.num_qk_stages == 0
            ), "padded_qk_depth must be divisible by num_qk_stages"
        self.pipeline = pipeline
        self.smem = smem
        self.pipeline.state._phase = 1

    @always_inline
    def get_smem[*, qk_stage: Int = 0](self) -> Self.SMemType:
        """Get smem pointer for current stage.

        Parameters:
            qk_stage: K-loading sub-stage whose smem offset to return
                (defaults to 0).
        """

        comptime if Self.is_k:
            comptime stage_offset = qk_stage * Self.elements
            var dyn_offset: UInt32 = (
                UInt32(Self.elements_full) * self.pipeline.state.index()
            )
            return self.smem + stage_offset + dyn_offset
        else:
            var dyn_offset: UInt32 = (
                UInt32(Self.elements) * self.pipeline.state.index()
            )
            return self.smem + dyn_offset

    @always_inline
    def get_tile[*, qk_stage: Int = 0](self) -> Self.PairType:
        """Get TMA destination for this stage.

        Parameters:
            qk_stage: K-loading sub-stage whose TMA destination to return
                (defaults to 0).
        """
        var p_mbar = self.pipeline.producer_mbar[qk_stage]()
        var smem = Self.PairType.SmemType(
            self.get_smem[qk_stage=qk_stage](),
            tt_row_major[Self.PairType.smem_elems](),
        )
        return {p_mbar, smem}

    @always_inline
    def get_tile[*, qk_stage: Int = 0](self, e: Int32) -> Self.PairType:
        """Get TMA destination with optional expect_bytes.

        Parameters:
            qk_stage: K-loading sub-stage whose TMA destination to return
                (defaults to 0).

        Args:
            e: `elect()` result; when nonzero, issues `expect_bytes` on
                the producer mbarrier before returning.
        """
        var p_mbar = self.pipeline.producer_mbar[qk_stage]()
        if e != 0:
            p_mbar[].expect_bytes(Int32(Self.tile_bytes))
        var smem = Self.PairType.SmemType(
            self.get_smem[qk_stage=qk_stage](),
            tt_row_major[Self.PairType.smem_elems](),
        )
        return {p_mbar, smem}

    @always_inline
    def acquire[*, qk_stage: Int = 0](self):
        """Wait for consumer to release the buffer.

        Parameters:
            qk_stage: K-loading sub-stage to acquire (defaults to 0).
        """
        self.pipeline.producer_acquire[qk_stage]()

    @always_inline
    def commit_step(mut self):
        """Step the pipeline. Commit is handled by tma_op.async_copy."""
        self.pipeline.state.step()

    # Backward-compatible K methods (for KProducerPipeline)
    comptime KPairType = Self.PairType  # Alias for backward compatibility

    @always_inline
    def get_k_smem[*, qk_stage: Int](self) -> Self.SMemType:
        return self.get_smem[qk_stage=qk_stage]()

    @always_inline
    def get_k[*, qk_stage: Int](self) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage]()

    @always_inline
    def get_k[*, qk_stage: Int](self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=qk_stage](e)

    @always_inline
    def acquire_k[*, qk_stage: Int](self):
        self.acquire[qk_stage=qk_stage]()

    @always_inline
    def get_v_smem(self) -> Self.SMemType:
        return self.get_smem[qk_stage=0]()

    @always_inline
    def get_v(self, e: Int32) -> Self.PairType:
        return self.get_tile[qk_stage=0](e)

    @always_inline
    def acquire_v(self):
        self.acquire[qk_stage=0]()


# Backward-compatible type aliases
comptime KProducerPipeline = TMAProducerPipeline[_, _, True]
comptime VProducerPipeline = TMAProducerPipeline[_, _, False]


struct TMAConsumerPipeline[dtype: DType, config: FA4Config, is_k: Bool = True](
    TrivialRegisterPassable
):
    """Unified consumer pipeline for K and V TMA consumption.

    K consumption (is_k=True): Uses k_major layout, supports staged qk_stages.
    V consumption (is_k=False): Uses mn_major layout, always uses qk_stage=0.

    This follows the order of Tri Dao and Cutlass implementations
    (modulo any rotation of the ops through the iterations).

    We consume/produce in the following order:
        0. S0 <- Q0 @ Kn'
        1. O1 <- O1 + P1 @ V{n-1}
        2. S1 <- Q1 @ Kn'
        3. O0 <- O0 + P0 @ Vn

    Note that we have two MMA between calculating Si and consuming Pi,
    maximizing the overlap between MMAs and softmax calculation.

    Parameters:
        dtype: Element dtype of the consumed K or V tile.
        config: FlashAttention-4 configuration providing tile sizes, swizzle
            mode, and stage counts.
        is_k: Whether this pipeline consumes the K operand (`True`, k-major
            layout with staged `qk_stages`) or the V operand (`False`,
            mn-major layout with `qk_stage=0`).
    """

    # K stage stride uses the K_nope width (`padded_nope_depth`), not the
    # V/output depth — they differ when `v_head_dim != qk_nope_head_dim`. V
    # stage stride uses `v_cols_per_cta()` (= padded_ov_depth). Equal for
    # DeepSeek and MHA (nope == ov).
    comptime full_kv_bytes = (
        Self.config.k_rows_per_cta()
        * Self.config.padded_nope_depth
        * size_of[Self.dtype]()
        + Self.config.k_rows_per_cta()
        * Self.config.rope_depth()
        * Self.config.rope_dtype_size
    ) if Self.is_k else (
        Self.config.BN * Self.config.v_cols_per_cta() * size_of[Self.dtype]()
    )
    comptime staged_k_bytes = Self.config.k_rows_per_cta() * Self.config.BK0 * size_of[
        Self.dtype
    ]()

    # K uses full staging, V uses qk_stages=1
    comptime num_qk_stages_effective: Int = Self.config.num_qk_stages if Self.is_k else 1

    # Descriptor parameters differ by role
    comptime BMN: Int = Self.config.k_rows_per_cta() if Self.is_k else Self.config.v_cols_per_cta()
    comptime BK: Int = Self.config.BK0 if Self.is_k else Self.config.BK1
    comptime is_k_major: Bool = Self.is_k
    # Page-dense (row-major) layout: K (Q@K', k-major) gated by k_row_major(),
    # V (P@V, mn-major) by v_row_major(). `is_k_major=Self.is_k` (below) routes
    # the flag to the matching `tile_layout_*` branch in `smem_descriptor`.
    comptime page_dense: Bool = (
        Self.config.k_row_major() if Self.is_k else Self.config.v_row_major()
    )

    var pipeline: StagedPipeline[
        Self.config.num_kv_stages, Self.num_qk_stages_effective
    ]
    var smem_desc: MMASmemDescriptorPair

    @always_inline
    def __init__(
        out self,
        pipeline: StagedPipeline[
            Self.config.num_kv_stages, Self.num_qk_stages_effective
        ],
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        self.pipeline = pipeline
        self.smem_desc = smem_descriptor[
            BMN=Self.BMN,
            BK=Self.BK,
            swizzle_mode=Self.config.swizzle_mode,
            is_k_major=Self.is_k_major,
            page_dense=Self.page_dense,
        ](smem)

    @always_inline
    def __init__(
        out self,
        mbar: MBarType,
        smem: SharedMemPointer[Scalar[Self.dtype]],
    ):
        return Self(type_of(self.pipeline)(mbar), smem)

    @always_inline("nodebug")
    def get(self) -> MMASmemDescriptorPair:
        """Get smem descriptor for current stage."""
        var dyn_offset: UInt32 = (
            UInt32(Self.full_kv_bytes) * self.pipeline.state.index()
        )
        return self.smem_desc + dyn_offset

    @always_inline("nodebug")
    def wait[*, qk_stage: Int = 0](self):
        """Wait for tile from producer.

        Parameters:
            qk_stage: K-loading sub-stage to wait on (defaults to 0).
        """
        self.pipeline.consumer_wait[qk_stage]()

    @always_inline("nodebug")
    def release[*, qk_stage: Int = 0](mut self, e: Int32):
        """Release buffer after consuming.

        Parameters:
            qk_stage: K-loading sub-stage to release (defaults to 0).

        Args:
            e: `elect()` result selecting the single thread that arrives
                on the mbarrier.
        """
        self.pipeline.consumer_release[qk_stage](e)

    # Backward-compatible K methods (for KConsumerPipeline)
    @always_inline("nodebug")
    def get_k(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    def wait_k[*, qk_stage: Int = Self.config.num_qk_stages - 1](mut self):
        """Wait on K stage from the producer.

        Parameters:
            qk_stage: K-loading sub-stage to wait on (defaults to the last
                sub-stage).
        """
        self.wait[qk_stage=qk_stage]()

    @always_inline("nodebug")
    def release_k[
        *, qk_stage: Int = Self.config.num_qk_stages - 1
    ](mut self, e: Int32):
        """Release K buffer after consuming this stage.

        Parameters:
            qk_stage: K-loading sub-stage to release (defaults to the last
                sub-stage).

        Args:
            e: `elect()` result selecting the single thread that arrives
                on the mbarrier.
        """
        self.release[qk_stage=qk_stage](e)

    # Backward-compatible V methods (for VConsumerPipeline)
    @always_inline("nodebug")
    def get_v(self) -> MMASmemDescriptorPair:
        return self.get()

    @always_inline("nodebug")
    def wait_v(self):
        """Wait for V tile."""
        self.wait[qk_stage=0]()

    @always_inline("nodebug")
    def release_v(mut self, e: Int32):
        """Release V buffer after consuming.

        Args:
            e: `elect()` result selecting the single thread that arrives
                on the mbarrier.
        """
        self.release[qk_stage=0](e)


# Backward-compatible type aliases
comptime KConsumerPipeline = TMAConsumerPipeline[_, _, True]
comptime VConsumerPipeline = TMAConsumerPipeline[_, _, False]


struct RolePipeline[
    number_of_stages: Int,
    is_producer: Bool = True,
    producer_sub_stages: Int = 1,
    consumer_sub_stages: Int = 1,
    cta_group: Int = 1,
](TrivialRegisterPassable):
    """
    Unified producer/consumer pipeline for barrier synchronization.

    Producer role: Starts with phase=1, uses acquire/commit methods.
    Consumer role: Starts with phase=0, uses wait/release methods.

    Sub-stages allow multiple barriers per stage:
    - Total producer barriers: num_stages * producer_sub_stages
    - Total consumer barriers: num_stages * consumer_sub_stages

    Synchronization behavior (example with num_stages=1):

    Producer:
    p0. consumer_mbar.wait(phase=1)  # 1 != 0: falls through
    p1. producer_mbar.commit()       # producer_mbar.phase=1
    p2. step()                       # phase = 0
    p3. consumer_mbar.wait(phase=0)  # 0 == 0: blocked until c1
    ...

    Consumer:
    c0. producer_mbar.wait(phase=0)  # 0 == 0: blocked until p1
    c1. consumer.release()           # consumer_mbar.phase=1
    c2. step()                       # phase = 1
    ...

    Parameters:
        number_of_stages: Number of double-buffered pipeline stages.
        is_producer: Whether this instance is the producer role (defaults
            to `True`).
        producer_sub_stages: Number of producer mbarriers per stage
            (defaults to 1).
        consumer_sub_stages: Number of consumer mbarriers per stage
            (defaults to 1).
        cta_group: Number of cooperating CTAs for MMA commit arrivals
            (defaults to 1).
    """

    comptime num_stages: Int = Self.number_of_stages

    @__allow_legacy_any_origin_fields
    var producer_mbar_base: MBarType

    @__allow_legacy_any_origin_fields
    var consumer_mbar_base: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(
        out self, producer_mbar_base: MBarType, consumer_mbar_base: MBarType
    ):
        self.producer_mbar_base = producer_mbar_base
        self.consumer_mbar_base = consumer_mbar_base
        self.state = {}

        comptime if Self.is_producer:
            # Producer starts with phase=1 so initial waits fall through
            self.state._phase = 1

    @always_inline
    def producer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
        """Get producer mbar for current stage and optional sub-stage.

        Parameters:
            sub_stage_idx: Sub-stage index (0 to producer_sub_stages-1).
        """
        comptime assert (
            sub_stage_idx < Self.producer_sub_stages
        ), "sub_stage_idx out of range"
        return (
            self.producer_mbar_base
            + self.state.index() * UInt32(Self.producer_sub_stages)
            + sub_stage_idx
        )

    @always_inline
    def consumer_mbar[sub_stage_idx: Int = 0](self) -> MBarType:
        """Get consumer mbar for current stage and optional sub-stage.

        Parameters:
            sub_stage_idx: Sub-stage index (0 to consumer_sub_stages-1).
        """
        comptime assert (
            sub_stage_idx < Self.consumer_sub_stages
        ), "sub_stage_idx out of range"
        return (
            self.consumer_mbar_base
            + self.state.index() * UInt32(Self.consumer_sub_stages)
            + sub_stage_idx
        )

    # Producer methods
    @always_inline("nodebug")
    def acquire[sub_stage_idx: Int = 0](self):
        """Wait until consumer has released the buffer. Producer-only.

        Parameters:
            sub_stage_idx: Consumer sub-stage barrier to wait on (defaults
                to 0).
        """
        self.consumer_mbar[sub_stage_idx]()[].wait(self.state.phase())

    @always_inline("nodebug")
    def commit(mut self):
        """Commit production and step. Producer-only."""
        _ = self.producer_mbar()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    def commit_mma(self):
        """Commit via MMA arrive using elected thread. Producer-only."""
        var mbar = self.producer_mbar()
        elect_mma_arrive[cta_group=Self.cta_group](mbar, elect())

    @always_inline("nodebug")
    def commit_mma(self, elect: Int32):
        """Commit via MMA arrive with explicit elect value. Producer-only.

        Args:
            elect: `elect()` result selecting the single arriving thread.
        """
        var mbar = self.producer_mbar()
        elect_mma_arrive[cta_group=Self.cta_group](mbar, elect)

    # Consumer methods
    @always_inline("nodebug")
    def wait(self):
        """Wait for producer to complete. Consumer-only."""
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline("nodebug")
    def release[sub_stage_idx: Int = 0](mut self):
        """Release buffer at sub-stage and step. Consumer-only.

        Parameters:
            sub_stage_idx: Consumer sub-stage barrier to arrive on
                (defaults to 0).
        """
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()
        self.state.step()

    @always_inline("nodebug")
    def release_no_step[sub_stage_idx: Int = 0](self):
        """Release buffer without stepping. For multi-sub-stage release.

        Parameters:
            sub_stage_idx: Consumer sub-stage barrier to arrive on
                (defaults to 0).
        """
        _ = self.consumer_mbar[sub_stage_idx]()[].arrive()

    # Shared method
    @always_inline("nodebug")
    def step(mut self):
        self.state.step()


# Backward-compatible type aliases
comptime ProducerPipeline = RolePipeline[_, True, _, _, _]
comptime ConsumerPipeline = RolePipeline[_, False, _, _, _]


struct MBarPipeline[number_of_stages: Int](TrivialRegisterPassable):
    """Manages a paired set of producer/consumer mbarriers for pipeline synchronization.

    Parameters:
        number_of_stages: Number of double-buffered pipeline stages, each
            with one producer and one consumer mbarrier.
    """

    comptime num_stages: Int = Self.number_of_stages

    # mbars are ordered in {producer, consumer} pairs
    @__allow_legacy_any_origin_fields
    var mbar: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(out self, mbar: MBarType):
        self.mbar = mbar
        self.state = {}

    @always_inline
    def init[*, num_producer: UInt32 = 1, num_consumer: UInt32 = 1](self):
        comptime for i in range(Self.number_of_stages):
            self.mbar[i].init(Int32(Int(num_producer)))

        comptime for i in range(Self.number_of_stages):
            self.mbar[i + Self.number_of_stages].init(Int32(Int(num_consumer)))

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(2 * Self.number_of_stages)


@always_inline
def apply_oob_mask[
    *,
    mask_strategy: MaskStrategy,
    apply_log2e_after_mask: Bool,
](
    s_arg: SIMD[.float32, 2],
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: Int32,
    max_seq_len: UInt32,
    num_keys: Int32,
    score_row: Int32,
    score_col: Int32,
) -> SIMD[.float32, 2]:
    """Applies the out-of-bounds key mask to a pair of attention scores.

    Scores for columns at or beyond `num_keys` are replaced with `MASK_VALUE`; optionally scales by `log2e` before masking.

    Parameters:
        mask_strategy: `MaskStrategy` bitset selecting which masking
            strategies to apply; the out-of-bounds clip runs only when
            `OUT_OF_BOUNDS` is set.
        apply_log2e_after_mask: Whether to multiply the scores by `log2e`
            before masking.

    Args:
        s_arg: Pair of attention scores to mask.
        prompt_idx: Index of the prompt in the batch.
        q_head_idx: Index of the query head.
        kv_tile_start_row: Starting row of the KV tile in the key
            dimension.
        max_seq_len: Maximum sequence length.
        num_keys: Number of valid keys; columns at or beyond this index
            are masked with `MASK_VALUE`.
        score_row: Row index of the score in the query dimension.
        score_col: Starting column index of the score pair; columns from
            this index onward are compared to `num_keys`.
    """
    var s: SIMD[.float32, 2] = s_arg

    comptime if apply_log2e_after_mask:
        s = mul_ftz(s, log2e)

    comptime if MaskStrategy.OUT_OF_BOUNDS in mask_strategy:
        s = (
            iota[.int32, 2](score_col)
            .lt(num_keys)
            .select(s, MASK_VALUE)
            # .select(s, min_or_neg_inf[DType.float32]())
        )

    return s


@always_inline
def apply_mask[
    BN: Int,
    MaskType: MHAMask,
    //,
    *,
    mask_strategy: MaskStrategy,
    skip_scale: Bool = False,
](
    mut srow: Array[Float32, BN],
    mask: MaskType,
    scale_log2e: Float32,
    *,
    prompt_idx: UInt32,
    q_head_idx: UInt32,
    kv_tile_start_row: Int32,
    max_seq_len: UInt32,
    num_keys: Int32,
    score_row: Int32,
):
    """Applies bitmask, computed, and out-of-bounds masking strategies to a row of `BN` attention scores.

    Scales by `scale_log2e` (unless `skip_scale`), then applies the mask strategy: the bitmask path uses `mask_select8` per 32-column batch, the computed path calls `mask.mask`, and both paths apply the out-of-bounds clip via `apply_oob_mask`.

    Parameters:
        BN: Number of scores in the row; must be a multiple of 32 for
            the bitmask path.
        MaskType: `MHAMask` type providing the `apply_log2e_after_mask`
            flag and the `mask_bits` or `mask` masking primitives.
        mask_strategy: `MaskStrategy` bitset selecting which masking
            strategies to apply (BITMASK, COMPUTED, OUT_OF_BOUNDS).
        skip_scale: Whether to skip the `scale_log2e` pre-scaling
            (defaults to `False`).

    Args:
        srow: Row of `BN` attention scores to mask in place.
        mask: Mask object providing bitmask or computed mask values.
        scale_log2e: Softmax scale factor in log2 base, applied to
            scores before masking unless `skip_scale` is set.
        prompt_idx: Index of the prompt in the batch.
        q_head_idx: Index of the query head.
        kv_tile_start_row: Starting row of the KV tile in the key
            dimension, the absolute column offset of the first key in
            this tile.
        max_seq_len: Maximum sequence length.
        num_keys: Number of valid keys; columns at or beyond this index
            are out of bounds.
        score_row: Row index of the score in the query dimension.
    """
    comptime simd_size = 2
    comptime F32x2 = SIMD[.float32, simd_size]

    comptime if MaskStrategy.BITMASK in mask_strategy or (
        MaskStrategy.OUT_OF_BOUNDS in mask_strategy
        and MaskStrategy.COMPUTED not in mask_strategy
    ):
        # Mask-driven bitmask path: each 32-col batch is masked by a 32-bit
        # visibility pattern. Either the mask provides it via `mask_bits()`
        # (`BITMASK`), or the kernel hardcodes "clip at num_keys" by setting
        # `OUT_OF_BOUNDS` alone — used by softmax warps that fast-path the
        # runtime-`NO_MASK` case.
        comptime num_batches = BN // 32
        comptime assert (BN % 32) == 0

        comptime for batch in range(num_batches):
            var col_start: Int32 = kv_tile_start_row + Int32(32 * batch)
            var mask_bits: UInt32

            comptime if MaskStrategy.BITMASK in mask_strategy:
                mask_bits = mask.mask_bits(
                    prompt_idx, score_row, col_start, num_keys
                )
            else:
                # OUT_OF_BOUNDS alone: low `n_valid_oob` bits set, where
                # n_valid_oob = max(num_keys - col_start, 0).
                var n_valid_oob: Int32 = max(num_keys - col_start, 0)
                mask_bits = (
                    (UInt32(1) << UInt32(n_valid_oob)) - UInt32(1)
                ) if n_valid_oob < 32 else UInt32(0xFFFF_FFFF)

            comptime for byte_idx in range(4):
                # One opaque `mask_select8` block per mask byte: 8 contiguous
                # lanes `srow[base .. base+7]` gated by bits `8*byte_idx .. +7`.
                # Confining each byte's bit-extraction + selects to one block
                # keeps the predicate live-set bounded (the spill fix).
                comptime base = 32 * batch + 8 * byte_idx

                # Gather + scale OUTSIDE the asm, reusing the x2 `mul_ftz` so the
                # ftz scaling matches the scalar path byte-for-byte.
                var p0: F32x2 = F32x2(srow[base + 0], srow[base + 1])
                var p1: F32x2 = F32x2(srow[base + 2], srow[base + 3])
                var p2: F32x2 = F32x2(srow[base + 4], srow[base + 5])
                var p3: F32x2 = F32x2(srow[base + 6], srow[base + 7])
                comptime if not skip_scale:
                    p0 = mul_ftz(p0, scale_log2e)
                    p1 = mul_ftz(p1, scale_log2e)
                    p2 = mul_ftz(p2, scale_log2e)
                    p3 = mul_ftz(p3, scale_log2e)

                var r = mask_select8[byte_idx](
                    p0[0],
                    p0[1],
                    p1[0],
                    p1[1],
                    p2[0],
                    p2[1],
                    p3[0],
                    p3[1],
                    mask_bits,
                )

                var o0 = F32x2(r[0], r[1])
                var o1 = F32x2(r[2], r[3])
                var o2 = F32x2(r[4], r[5])
                var o3 = F32x2(r[6], r[7])
                comptime if MaskType.apply_log2e_after_mask:
                    o0 = mul_ftz(o0, log2e)
                    o1 = mul_ftz(o1, log2e)
                    o2 = mul_ftz(o2, log2e)
                    o3 = mul_ftz(o3, log2e)

                srow[base + 0] = o0[0]
                srow[base + 1] = o0[1]
                srow[base + 2] = o1[0]
                srow[base + 3] = o1[1]
                srow[base + 4] = o2[0]
                srow[base + 5] = o2[1]
                srow[base + 6] = o3[0]
                srow[base + 7] = o3[1]

    else:
        comptime block_size = BN // simd_size

        comptime for n in range(block_size):
            # score_col = mask_frag_col + j * 8
            comptime frag_col = simd_size * n
            var s: F32x2

            comptime if skip_scale:
                s = F32x2(srow[frag_col], srow[frag_col + 1])
            else:
                s = mul_ftz(
                    F32x2(srow[frag_col], srow[frag_col + 1]), scale_log2e
                )
            var score_col: Int32 = kv_tile_start_row + Int32(frag_col)

            comptime if MaskStrategy.COMPUTED in mask_strategy:
                s = mask.mask(
                    IndexList[4, element_type=.uint32](
                        Int(prompt_idx),
                        Int(q_head_idx),
                        Int(score_row),
                        Int(score_col),
                    ),
                    s,
                )

            var result = apply_oob_mask[
                mask_strategy=mask_strategy,
                apply_log2e_after_mask=MaskType.apply_log2e_after_mask,
            ](
                s,
                prompt_idx=prompt_idx,
                q_head_idx=q_head_idx,
                kv_tile_start_row=kv_tile_start_row,
                max_seq_len=max_seq_len,
                num_keys=num_keys,
                score_row=score_row,
                score_col=score_col,
            )
            srow[frag_col] = result[0]
            srow[frag_col + 1] = result[1]


@always_inline
def clusters_per_wave[cluster_size: Int, sm_count: Int]() -> Int:
    """Number of size-`cluster_size` thread-block clusters that fit on the target
    Blackwell datacenter GPU in ONE wave, honoring GPC co-residency.

    A cluster's CTAs must all live inside a single GPC, so a size-`C` cluster
    occupies `C` SMs within one GPC and only `floor(gpc_sm / C)` such clusters
    fit per GPC. The usable count is therefore *below* the flat `sm_count / C`;
    it is the per-GPC histogram `sum_g floor(gpc_sm[g] / C)`. GPC layout is not
    queryable (neither `DeviceAttribute` nor `GPUInfo` exposes it), so it is
    hardcoded per chip; the whole expression folds at comptime since both
    parameters are comptime.

    B200 (148 SMs) has 11 GPCs with an irregular layout -- three 20s, four 18s,
    one 10, three 2s (`2x` the pair counts `10,10,10,9,9,9,9,5,1,1,1`). B300 /
    Blackwell Ultra (160 SMs) is the full die with 8 GPCs of a UNIFORM 20 SMs
    each, so the histogram collapses to `8 * (20 // C)`.

    Non-increasing in `cluster_size`, so scanning candidate sizes largest-first
    yields the largest that fits. Only B200 (148) and B300 (160) are modeled;
    the `else` branch is a comptime error on any other chip.
    """
    comptime C = cluster_size
    comptime if sm_count == 148:
        # B200: irregular 11-GPC layout -- three 20s, four 18s, one 10, three 2s.
        return 3 * (20 // C) + 4 * (18 // C) + (10 // C) + 3 * (2 // C)
    elif sm_count == 160:
        # B300 / Blackwell Ultra: uniform 8 GPCs x 20 SMs.
        return 8 * (20 // C)
    else:
        comptime assert (
            False
        ), "clusters_per_wave: only B200 (148) / B300 (160) modeled"


@always_inline
def splitk_p_ladder[sm_count: Int]() -> List[Int]:
    """The rung ladder of split-K partition counts `P`, shared by the producer
    and the consumer of a workspace split-K launch.

    Single source of truth for two consumers that MUST agree:

    * `dispatch.mojo`'s `_bucket_ws` snaps a desired count UP to a rung (then
      caps it), so every `P` the production auto-route picks is a rung of this
      list or the cap.
    * `fa4_splitk_combine`'s launcher compiles one comptime-unrolled combine
      specialization per rung, and falls back to a generic runtime-`P` kernel
      for anything off it.

    Keeping them as two hand-copied lists silently costs the specialization
    whenever one side gains a rung the other does not (which is exactly what
    happened to the sub-12 rungs).

    `sm_count` is the top rung: `_bucket_ws` reaches it exactly when
    `raw_grid == 1`.
    """
    return [2, 4, 6, 8, 10, 12, 16, 18, 20, 24, 32, 48, 64, 96, sm_count]


@always_inline
def splitk_num_partitions[
    config: FA4Config
](ws_num_partitions: UInt32) -> UInt32:
    """The split-K partition count `P` this CTA must divide its KV range by.

    Single source of truth for the two split-K mechanisms. All four FA4 warps
    (`load`, `mma`, `softmax`, `correction`) feed this to `splitk_window` and
    MUST derive the same window: a disagreement makes the producer over- or
    under-fill relative to its consumers, which HANGS rather than producing a
    wrong number.

    * Cluster/DSMEM split-K bakes `P` into the launch cluster, so it is the
      comptime `config.splitk_partitions` (or `cluster_dim.x` once the cluster
      dimension becomes dynamic).
    * Workspace (traditional/unfused) split-K keeps `config.splitk_partitions
      == 1` -- no launch cluster -- and carries `P` at runtime by over-launching
      `grid.x`; `ws_num_partitions` is that count.

    Parameters:
        config: The FA4 config, supplying the comptime cluster partition count.

    Args:
        ws_num_partitions: Runtime partition count for the workspace scheme.
            Ignored when `config.splitk_partitions > 1`. Its `1` default makes
            this a no-op for a caller with no split-K at all.
    """
    comptime if config.splitk_partitions > 1:
        comptime if config.dynamic_cluster_dim:
            return UInt32(cluster_dim.x)
        else:
            return UInt32(config.splitk_partitions)
    else:
        return ws_num_partitions


@always_inline
def splitk_partition_idx(splitk_partitions: UInt32) -> UInt32:
    """This CTA's split-K partition index `[0, splitk_partitions)`.

    Derived from the grid coordinate (`block_idx.x % splitk_partitions`),
    NOT `block_rank_in_cluster()`: the scheduler maps `block_idx.x //
    cluster_size -> tile` (cluster_size == splitk_partitions for split-K,
    since it forces pair_cta=False), so the low bits are the partition. This
    is correct and CTA-uniform whether or not the launch forms a hardware
    cluster -- M2 has no cross-CTA traffic, so it does not depend on cluster
    co-residency. (M4's DSMEM combine will additionally require a real
    cluster; that is where `block_rank_in_cluster()` / cluster co-residency
    re-enters.)

    Args:
        splitk_partitions: Number of split-K partitions (`P`); the
            partition index is `block_idx.x % splitk_partitions`.
    """
    return UInt32(block_idx.x) % splitk_partitions


@always_inline
def splitk_window(
    T: UInt32, num_partitions: UInt32, partition_idx: UInt32
) -> Tuple[UInt32, UInt32]:
    """Front-loaded balanced split of the combined K-tile range `[0, T)`.

    Partition `p` owns the BN-tile window `[cb, ce)` where the first
    `r = T % num_partitions` partitions get `q+1 = ceil(T/P)` tiles and the
    rest get `q = floor(T/P)`:

        cb = p*q     + min(p,   r)
        ce = (p+1)*q + min(p+1, r)

    The chunks differ by at most one tile (balanced load), but tiles are
    filled *leading* partition first: for `T >= 1` partition 0 is always
    non-empty (it owns tile 0), and only *trailing* partitions go empty
    (`cb == ce`) once `T < num_partitions`. Keeping the writer (rank 0)
    non-empty lets the cross-CTA combine (which hardcodes own = rank 0) stay
    valid; empty trailing partitions stage a neutral identity and the writer
    weights them to zero. M6 routes idle CTAs (`partition_idx >=
    num_partitions`) through that same neutral path.

    `T` is a tile count (small) and `num_partitions <= sm_count`, so the
    products cannot overflow `UInt32`. On the cluster split-K path
    `num_partitions` is comptime and the `//`/`%` lower to multiply-shift; the
    workspace (unfused) path passes a runtime `P`, so it pays one real `divmod`
    -- once per warp per kernel, outside every key-tile loop.

    Args:
        T: Total number of K-tiles to divide across partitions.
        num_partitions: Number of split-K partitions (`P`).
        partition_idx: This CTA's partition index `[0, num_partitions)`.
    """
    var q, r = divmod(T, num_partitions)
    var cb: UInt32 = partition_idx * q + min(partition_idx, r)
    var ce: UInt32 = (partition_idx + UInt32(1)) * q + min(
        partition_idx + UInt32(1), r
    )
    return (cb, ce)


@always_inline
def peel_mask[
    num_sets: Int,
    //,
    mask_strategies: StaticTuple[MaskStrategy, num_sets],
    load_fn: def[mask_strategy: MaskStrategy](UInt32) capturing -> Float32,
](mut mask_iters: StaticTuple[UInt32, num_sets], kv_row: UInt32,) -> Float32:
    """Determine which mask strategy applies to the peeled first iteration.

    Walks through mask sets to find the first with remaining iterations,
    calls load_fn with the corresponding strategy, and decrements the counter.
    Prevents UInt32 underflow when early sets are empty (e.g.
    SlidingWindowCausalMask with num_sets=3 and small sequences).

    Parameters:
        num_sets: Number of mask sets to walk through (inferred).
        mask_strategies: `MaskStrategy` per set, in evaluation order;
            the first set with remaining iterations supplies the
            strategy.
        load_fn: Callback invoked as
            `load_fn[strategy](kv_row)` to load the mask value for the
            selected strategy.

    Args:
        mask_iters: Remaining iteration count per set; the selected
            set's count is decremented in place.
        kv_row: KV row index forwarded to `load_fn`.
    """
    comptime assert num_sets in (1, 2, 3)
    comptime if num_sets == 1:
        mask_iters[0] -= 1
        return load_fn[mask_strategies[0]](kv_row)
    elif num_sets == 2:
        if mask_iters[0] > 0:
            mask_iters[0] -= 1
            return load_fn[mask_strategies[0]](kv_row)
        else:
            mask_iters[1] -= 1
            return load_fn[mask_strategies[1]](kv_row)
    else:
        if mask_iters[0] > 0:
            mask_iters[0] -= 1
            return load_fn[mask_strategies[0]](kv_row)
        elif mask_iters[1] > 0:
            mask_iters[1] -= 1
            return load_fn[mask_strategies[1]](kv_row)
        else:
            mask_iters[2] -= 1
            return load_fn[mask_strategies[2]](kv_row)


struct FA4MiscMBars[
    *,
    num_qk_stages: Int = 1,
    num_pv_stages: Int = 1,
    num_kv_stages: Int = 2,
    use_order_barriers: Bool = True,
    use_shared_kv: Bool = False,
    pair_cta: Bool = False,
    num_q: Int = 2,
    splitk_partitions: Int = 1,
    BM: Int = 128,
    use_ws: Bool = False,
    crossp: Bool = False,
](TrivialRegisterPassable):
    """Manages all mbarrier resources for FA4.

    This struct consolidates all mbarrier management including:
    - S barriers (score MMA synchronization)
    - C barriers (correction synchronization)
    - Order barriers (softmax ordering)
    - Q1Sync barriers (Q tile synchronization)
    - K/V pipeline barriers (separate K and V)
    - O pipeline barriers

    Parameters:
        num_qk_stages: Number of stages for Q@K' MMA (K loading can be staged).
        num_pv_stages: Number of stages for P@V MMA (P writing can be staged).
        num_kv_stages: Number of KV buffer stages for double/triple buffering.
        use_order_barriers: When True, allocate order barriers to prevent softmax
            warp group overlap. When False, order barriers are omitted.
        use_shared_kv: Whether the K and V share the same pipeline, or separate.
        pair_cta: Whether to use 1-cta or 2-cta implementation.
        num_q: Number of Q tiles per CTA. When 1, the `Q1Sync` slot is
            collapsed and `K_offset` shifts down by `num_qk_stages`. Must
            be 2 for any caller of `q1_wait_mbar()`.
        splitk_partitions: Number of split-K partitions (P). When
            `num_q == 1` and this exceeds 1, a single publish barrier is
            added so the cross-CTA O combine writer observes all `P`
            partitions' staged partials. Otherwise no extra barrier is
            allocated, keeping a byte-identical mbar layout.
        BM: Block size (rows per CTA). For 1Q split-K this is the number of
            WG0 rows that each arrive on the publish barrier, so its count is
            `BM * P` (every row of every partition). Only used to size the
            publish barrier; defaults to 128 (== `WARPGROUP_SIZE` on the 1Q
            path) for non-split-K callers.
        use_ws: Warp-specialized packed-TMEM (MMA_M=32) datapath. When True,
            `num_kv_stages` counts depth-split 256x64 sub-tile ring slots
            ("Convention B"), so `K_barriers = 2 * num_kv_stages` (the
            `num_qk_stages` depth factor is already folded into the slot count).
            When False (default), the non-WS full-depth-tile count applies and
            the layout is byte-identical.
        crossp: Whether cross-stage P applies to the config this type was
            built from, i.e. `FA4Config.crossp_on()`. Threaded in rather than
            read from the `FA4_TMEM_CROSS_P` define, because that define
            defaults ON and this type cannot see the config fields that scope
            it to the MHA 2Q shape. False (default) keeps the mbar layout
            byte-identical to cross-P-off.

    Memory layout (count=128 first, then count=1):
        [S0_cons] [S1_cons] [C0] [C1] [Order*] | [S0_prod] [S1_prod] [Q1Sync**] [K] [V] [O_prod]
        *Order barriers only present when use_order_barriers=True
        **Q1Sync barriers only present when num_q == 2
    """

    @__allow_legacy_any_origin_fields
    var mbar_base: MBarType

    # ---- Count=128 section (first in smem) ----
    # S consumer barriers: num_pv_stages per warp group
    comptime S0_consumer_offset = 0
    comptime S1_consumer_offset = Self.num_pv_stages
    # C barriers: 2 per warp group (producer + consumer, both count=128)
    comptime C0_offset = 2 * Self.num_pv_stages
    comptime C1_offset = Self.C0_offset + 2
    # Order barriers: 1 per warp group (count=128), conditional on use_order_barriers
    comptime num_order_barriers: Int = 2 if Self.use_order_barriers else 0
    comptime order_offset = Self.C1_offset + 2
    # ---- Count=1 section ----
    # S producer barriers: 1 per warp group
    comptime S0_producer_offset = Self.order_offset + Self.num_order_barriers
    comptime S1_producer_offset = Self.S0_producer_offset + 1
    # Q1Sync barriers (collapsed when num_q == 1; q1_wait_mbar() is
    # then unsafe to call — see the comptime assert in q1_wait_mbar().)
    comptime Q1SyncIdx = Self.S1_producer_offset + 1
    comptime Q1Sync_count: Int = Self.num_qk_stages if Self.num_q == 2 else 0
    # K pipeline barriers
    comptime K_offset = Self.Q1SyncIdx + Self.Q1Sync_count
    # Non-WS: each of the `num_kv_stages` full-depth K tiles is depth-chunked into
    # `num_qk_stages` Q@K' sub-loads, each needing a producer+consumer barrier.
    # WS ("Convention B"): `num_kv_stages` already counts depth-split 256x64
    # sub-tile ring slots (the depth split is folded into the slot count), so a
    # slot is one ring entry with just 2 barriers — the `num_qk_stages` factor
    # must NOT be applied again. `use_ws == False` folds to the non-WS count.
    comptime K_barriers: Int = (
        2
        * Self.num_kv_stages if Self.use_ws else 2
        * Self.num_qk_stages
        * Self.num_kv_stages
    )
    # V pipeline barriers (separate from K, only in non-shared mode)
    comptime V_offset: Int = Self.K_offset + Self.K_barriers
    comptime V_barriers: Int = 0 if Self.use_shared_kv else 2 * Self.num_kv_stages
    # O producer barriers (count=1)
    comptime O_producer_offset = Self.V_offset + Self.V_barriers
    # Split-K publish barrier (count=1 section, but count=P): one slot used by
    # the 1Q split-K cross-CTA O combine so the writer observes all P partitions'
    # staged partials. Present only for 1Q split-K; otherwise zero-sized so every
    # other config keeps a byte-identical mbar layout.
    comptime Publish_offset = Self.O_producer_offset + 2
    comptime Publish_count: Int = (
        1 if (Self.num_q == 1 and Self.splitk_partitions > 1) else 0
    )

    # Cross-stage P: 4 handshakes (sfree0/sfree1 + s0_p1/s1_p0, see the
    # accessors below) appended after Publish, so OFF `size` is byte-identical.
    # 2Q + non-WS only -- mutually exclusive with Publish (num_q==1 only), so
    # appending after it never collides.
    # `crossp` is `FA4Config.crossp_on()`, threaded in by whoever built this
    # type, so this count and `FA4Config.smem_used` always agree. Do NOT read
    # the `FA4_TMEM_CROSS_P` define here: it defaults ON, and this struct
    # cannot see the config fields that scope it to the MHA 2Q shape.
    comptime CrossP_enabled: Bool = (
        Self.crossp and Self.num_q == 2 and not Self.use_ws
    )
    comptime CrossP_offset = Self.Publish_offset + Self.Publish_count
    comptime InplaceDepth: Int = 4
    comptime CrossP_count: Int = (
        2 + 2 * Self.InplaceDepth
    ) if Self.CrossP_enabled else 0

    # Total size includes all barriers
    comptime size = Self.CrossP_offset + Self.CrossP_count
    comptime number_warpgroup_count = Self.S0_producer_offset

    @always_inline
    def __init__(out self, mbar_base: MBarType):
        self.mbar_base = mbar_base

    @staticmethod
    def _init_count(lane_idx: Int32) -> Int32:
        """Return the mbarrier thread count for the given barrier index.

        S0_consumer[0] and S1_consumer[0] get count=256 (combined softmax +
        correction), other S_consumer barriers get count=128 (softmax only).
        In pair-CTA mode, S_consumer counts double (both CTAs arrive).
        C and Order barriers are CTA-local and always count=128.
        """
        comptime cta_mult: Int = 2 if Self.pair_cta else 1
        # S_consumer[0] = combined P+O: softmax + correction from each CTA.
        if lane_idx == Int32(Self.S0_consumer_offset) or lane_idx == Int32(
            Self.S1_consumer_offset
        ):
            return Int32(256 * cta_mult)
        # Other S_consumer barriers: softmax only, scaled by cta_mult.
        if lane_idx < Int32(Self.C0_offset):
            return Int32(128 * cta_mult)
        # C and Order barriers: CTA-local, always 128.
        if lane_idx < Int32(Self.number_warpgroup_count):
            return 128
        # Split-K publish barrier: every WG0 row (BM of them) of every partition
        # CTA arrives, so the COUNT is `BM * splitk_partitions`. Each split-K
        # kernel is compiled once per static partition count `P`, so
        # `Self.splitk_partitions` is the exact launch cluster size (comptime) —
        # no ceiling/launch mismatch to guard against. Per-row (rather than one
        # leader per CTA) lets the publish sites drop their CTA-local
        # `named_barrier`: each row's arrive already happens-after that row's own
        # staging write. The slot itself is gated comptime on Publish_count.
        # ONLY round-1 (phase 0) uses this barrier now -- it makes peers' staged
        # O_cta + (max,sum) visible before the DSMEM reads. There is no round-2:
        # the combine packs its bf16 into its OWN-band dead f32 slice (no peer
        # reads it), and the kernel's terminal `cluster_sync()` keeps the
        # peer-read bands alive through reads.
        comptime if Self.Publish_count > 0:
            if lane_idx == Int32(Self.Publish_offset):
                return Int32(Self.BM * Self.splitk_partitions)
        # Cross-stage P handshakes (appended after Publish): all WG-collective
        # (softmax commit / softmax<->softmax), count=128.
        comptime if Self.CrossP_enabled:
            if lane_idx >= Int32(Self.CrossP_offset):
                return 128
        return 1

    @always_inline
    def init(self, *, lane_idx: Int32):
        comptime if Self.size < WARP_SIZE:
            if lane_idx < Int32(Self.size):
                self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
        elif Self.size == WARP_SIZE:
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
        else:
            comptime assert Self.size <= 2 * WARP_SIZE, String(
                "Total barrier count = ",
                Self.size,
                " exceeds 2 * WARP_SIZE = ",
                2 * WARP_SIZE,
            )
            # Wave 1: first 32 barriers (all lanes participate).
            self.mbar_base[lane_idx].init(Self._init_count(lane_idx))
            # Wave 2: remaining barriers past index 32. Use `_init_count` (not a
            # hardcoded 1) so a count != 1 barrier that lands past index 32 — the
            # split-K publish barrier (count=P) — is initialized correctly.
            if lane_idx < Int32(Self.size - WARP_SIZE):
                self.mbar_base[Int32(WARP_SIZE) + lane_idx].init(
                    Self._init_count(Int32(WARP_SIZE) + lane_idx)
                )

    # S pipeline type: 1 producer sub-stage, num_pv_stages consumer sub-stages
    comptime SPipelineProducer = RolePipeline[1, True, 1, Self.num_pv_stages]
    comptime SPipelineConsumer = RolePipeline[1, False, 1, Self.num_pv_stages]

    @always_inline
    def producer_s0(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 0."""
        return {
            self.mbar_base + Self.S0_producer_offset,
            self.mbar_base + Self.S0_consumer_offset,
        }

    @always_inline
    def producer_s1(self) -> Self.SPipelineProducer:
        """Get S producer for warp group 1."""
        return {
            self.mbar_base + Self.S1_producer_offset,
            self.mbar_base + Self.S1_consumer_offset,
        }

    @always_inline
    def consumer_s(self, wg_idx: UInt32) -> Self.SPipelineConsumer:
        """Get S consumer for given warp group.

        Args:
            wg_idx: Warp group index (0 or 1) selecting the S consumer
                pipeline.
        """
        return {
            self.mbar_base + Self.S0_producer_offset + wg_idx,
            self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx,
        }

    @always_inline
    def consumer_c0(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C0_offset,
            self.mbar_base + Self.C0_offset + 1,
        }

    @always_inline
    def consumer_c1(self) -> ConsumerPipeline[1]:
        return {
            self.mbar_base + Self.C1_offset,
            self.mbar_base + Self.C1_offset + 1,
        }

    @always_inline
    def producer_c(self, wg_idx: UInt32) -> ProducerPipeline[1]:
        var base = UInt32(Self.C0_offset) + 2 * wg_idx
        return {self.mbar_base + base, self.mbar_base + base + 1}

    @always_inline
    def pipeline_order_wait(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + Self.order_offset + wg_idx

    @always_inline
    def pipeline_order_arrive(self, wg_idx: UInt32) -> MBarType:
        return self.mbar_base + (Self.order_offset + 1) - wg_idx

    @always_inline
    def q1_wait_mbar(self) -> MBarType:
        comptime assert Self.num_q == 2, (
            "q1_wait_mbar() requires num_q == 2; the Q1Sync slot is"
            " collapsed when num_q == 1."
        )
        return self.mbar_base + Self.Q1SyncIdx

    # K/V/O barrier accessors
    @always_inline("nodebug")
    def get_k_mbars(self) -> MBarType:
        """Returns base pointer for K pipeline barriers."""
        return self.mbar_base + Self.K_offset

    @always_inline("nodebug")
    def get_v_mbars(self) -> MBarType:
        """Returns base pointer for V pipeline barriers.
        In shared mode, returns the same as get_k_mbars (shared pipeline).
        """
        comptime if Self.use_shared_kv:
            return self.mbar_base + Self.K_offset
        else:
            return self.mbar_base + Self.V_offset

    @always_inline("nodebug")
    def combined_p_o_consumer(self, wg_idx: UInt32) -> MBarType:
        """Combined P+O consumer barrier for given warp group.

        Arrived at by BOTH softmax (P ready) and correction (O rescaled).
        Returns S_consumer[0] for wg_idx=0 or wg_idx=1.

        Args:
            wg_idx: Warp group index (0 or 1) selecting the consumer
                barrier slot.
        """
        return self.mbar_base + UInt32(Self.num_pv_stages) * wg_idx

    # ---- Cross-stage P handshake accessors (CrossP_enabled, 2Q, non-WS) ----
    comptime CrossPProducer = RolePipeline[1, True, 1, 1]
    comptime CrossPConsumer = RolePipeline[1, False, 1, 1]
    # Inplace pipes are DEPTH-4 one-sided (only the full mbar side is wired):
    # producer_mbar_base == consumer_mbar_base so both producer.commit() and
    # consumer.wait() cycle the SAME 4 full mbars via state.index(); the empty
    # side is never allocated/armed.
    comptime InplaceProducer = RolePipeline[Self.InplaceDepth, True, 1, 1]
    comptime InplaceConsumer = RolePipeline[Self.InplaceDepth, False, 1, 1]

    # sfree{wg}: softmax commits "S{wg} scores consumed" (producer-only, 1
    # mbar, consumer_mbar aliased/unused -- natural throttle); MMA QK{wg}
    # acquires it (consumer .wait/.step) before overwriting S{wg}.
    @always_inline
    def sfree_producer(self, wg: UInt32) -> Self.CrossPProducer:
        var m = self.mbar_base + UInt32(Self.CrossP_offset) + wg
        return {m, m}

    @always_inline
    def sfree_consumer(self, wg: UInt32) -> Self.CrossPConsumer:
        var m = self.mbar_base + UInt32(Self.CrossP_offset) + wg
        return {m, m}

    # inplace{k}: k=0 -> s0_p1 (softmax0 produces "S0 window free", softmax1
    # consumes before storing P1 into S0's window); k=1 -> s1_p0 (mirror).
    # DEPTH-4 one-sided: 4 full mbars at base; both producer and consumer
    # cycle them via state.index() (producer_mbar_base == consumer_mbar_base).
    @always_inline
    def inplace_producer(self, k: UInt32) -> Self.InplaceProducer:
        var base = (
            self.mbar_base
            + UInt32(Self.CrossP_offset + 2)
            + k * UInt32(Self.InplaceDepth)
        )
        return {base, base}

    @always_inline
    def inplace_consumer(self, k: UInt32) -> Self.InplaceConsumer:
        var base = (
            self.mbar_base
            + UInt32(Self.CrossP_offset + 2)
            + k * UInt32(Self.InplaceDepth)
        )
        return {base, base}

    # O pipeline convenience methods
    @always_inline("nodebug")
    def consumer_o(self) -> RolePipeline[2, False, 1, Self.num_pv_stages]:
        """Get O consumer pipeline.

        Wait side: O_producer barriers (stride 1, indexed by stage).
        Release side: combined S+O barriers (S_consumer[0] per wg,
        stride num_pv_stages).
        """
        return {
            self.mbar_base + Self.O_producer_offset,
            self.mbar_base,
        }

    @always_inline("nodebug")
    def consumer_o0(self) -> RolePipeline[1, False, 1, Self.num_pv_stages]:
        """Single-O (1Q wide-V) O consumer: a ONE-stage pipeline on WG0's
        O-producer barrier only.

        The standard `consumer_o()` is a 2-stage pipeline that alternates
        between the two per-WG O-producer barriers (`O_producer_offset+0`
        for WG0, `+1` for WG1). The single-O path runs a single warp group
        (WG0) that accumulates ALL K-tiles into the single (aliased) O0, so
        the correction warp must wait on ONLY `O_producer_offset+0` with an
        incrementing phase, never the never-produced `+1` (which would
        deadlock). Release side is WG0's combined P+O consumer barrier, as
        in `producer_o0`.
        """
        return {
            self.mbar_base + Self.O_producer_offset,
            self.combined_p_o_consumer(0),
        }

    @always_inline("nodebug")
    def producer_o0(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 0."""
        return {
            self.mbar_base + Self.O_producer_offset,
            self.combined_p_o_consumer(0),
        }

    @always_inline("nodebug")
    def producer_o1(self) -> ProducerPipeline[1]:
        """Get O producer for warp group 1."""
        return {
            self.mbar_base + Self.O_producer_offset + 1,
            self.combined_p_o_consumer(1),
        }

    @always_inline("nodebug")
    def publish_mbar(self) -> MBarType:
        """Split-K cross-CTA O-combine publish barrier (count=`BM * P`).

        Every WG0 row of every partition CTA `arrive_cluster`s on every peer's
        copy (BM rows × P partitions arrivals per copy); the softmax threads
        `wait` on it before the writer's DSMEM peer reads. Per-row arrivals mean
        the publish sites need no CTA-local `named_barrier` to collect rows. Only
        present for 1Q split-K (`Publish_count > 0`).
        """
        return self.mbar_base + Self.Publish_offset

    @staticmethod
    @always_inline
    def num_mbars() -> UInt32:
        return UInt32(Self.size)
