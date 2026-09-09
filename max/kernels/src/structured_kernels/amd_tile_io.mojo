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
"""TileTensor data movement and AMD GPU hardware operations.

Provides reusable building blocks for TileTensor-based DMA, LDS reads,
and MMA operand loads on AMD CDNA GPUs (gfx950+).

Low-level LDS read primitives:
    ds_read_tr16_b64_row  - 4x16 transposed LDS read (raw rocdl intrinsic).
    ds_read_tr16_b64_warp - Warp-level transposed LDS read.
    load_lds_fragment     - Generic MFMA-fragment LDS read with swizzle.

DRAM→LDS cooperative DMA loaders (expert objects, structurally composed):
    _load_to_lds     - Single-instruction alias-scoped buffer-to-LDS DMA.
    TileLoaderLDS     - Warp-group cooperative, coord-indexed tile iteration
                        (half-tile BK-wide steps, per-iter swizzle). Matmul's
                        pattern. Uses stdlib `AMDBufferResource.load_to_lds`.
    SubTileLoaderLDS  - Single sub-tile DMA, TileTensor-indexed. Attention's
                        pattern. Uses `rocdl.raw.ptr.buffer.load.lds` with
                        the `amdgpu.AsyncCopies` alias scope so consumers
                        carrying `noalias_scopes=_alias_scope_attr` can skip
                        `s_waitcnt vmcnt(0)` (PR #74537).

SMEM→register MMA-fragment loader (expert object, static methods):
    TiledMmaLoader   - Sibling to `TiledMmaOp`. Parameterized by operand
                       dtype, MMA shape, and optional swizzle. Static
                       `load_b`, `load_b_tr`, `load_v_fp8_strip` methods
                       cover the B-operand and V-operand MFMA-fragment
                       load patterns (attention's QK / PV matmuls).

DRAM↔register loaders:
    RegTileLoader   - AMD buffer-resource load from DRAM to registers.
    RegTileWriter   - AMD buffer-resource store from registers to DRAM.
                      Buffer-resource OOB clamping handles the M
                      boundary cleanly but cannot distinguish row/col
                      straddle, so use this only when N is BN-aligned
                      and no fused lambda is needed.
    RegTileEpilogue - Per-lane epilogue writer with optional fused
                      elementwise lambda. Caller passes
                      (m_global, n_global) per call; the writer
                      handles the fully-in-bounds chunk store, the
                      partial-chunk-straddling-N per-element fallback,
                      and the lambda dispatch. Use this for any kernel
                      that needs to support N-misaligned shapes or a
                      fused epilogue.

Register→LDS writer (expert object, static methods):
    RegTileWriterLDS - Sibling to `RegTileLoader` / `RegTileWriter`.
        Stateless; parameterized by `thread_layout + swizzle`.
        `.copy` handles plain SMEM; `.copy_blocked[block_cols]` handles
        the `blocked_product`-mismatched-layout case.

SMEM layout helpers:
    smem_subtile / smem_mma_subtile / smem_mma_subtile_offset - blocked
        SMEM navigation (TileTensor views + offset math).
"""

from std.sys import (
    align_of,
    get_defined_bool,
    is_amd_gpu,
    simd_width_of,
    size_of,
)
from max.gpu import lane_id, thread_idx, WARP_SIZE
from max.gpu.intrinsics import (
    AMDBufferResource,
    ds_read_tr8_b64,
)
from max.gpu._utils import to_i32, to_i64
from max.gpu.memory import CacheOperation
from std.memory import AddressSpace
from std.memory.unsafe import bitcast
from std.math import ceildiv, min
from std.math.uutils import udivmod, umod, ufloordiv
from std.sys._assembly import inlined_assembly
from std.sys.intrinsics import readfirstlane
from std.utils import IndexList
from layout import Coord, Idx, TileTensor, TensorLayout
from layout.coord import crd2idx
from layout._utils import make_amd_buffer_resource
from layout.tile_layout import Layout, row_major, col_major
from layout.swizzle import Swizzle
from layout.tensor_engine import DefaultEngine
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from std.itertools import product


comptime elementwise_epilogue_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](IndexList[2], SIMD[dtype, width]) capturing -> None
"""Type alias for a fused elementwise epilogue lambda.

Local re-declaration of `linalg.utils.elementwise_epilogue_type`.
`structured_kernels` is a *dependency* of `linalg`, so we cannot
import the canonical definition without creating a cyclic bazel dep.
Mojo function-pointer types are structural, so this duplicate alias
is interchangeable with the canonical one at every call site that
hands a lambda across the package boundary.
"""


comptime _alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
comptime _no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`


# ===----------------------------------------------------------------------=== #
# _load_to_lds: Alias-scoped DRAM-to-LDS DMA
# ===----------------------------------------------------------------------=== #


@always_inline
def _load_to_lds[
    dtype: DType,
    //,
    width: Int,
    cache_policy: CacheOperation = CacheOperation.ALWAYS,
](
    bc: AMDBufferResource,
    vector_offset: Int32,
    shared_ptr: UnsafePointer[Scalar[dtype], _, address_space=.SHARED],
    scalar_offset: Int32 = 0,
):
    """Loads one raw-buffer vector into LDS with TileIO's async-copy scope.

    The hardware writes lane `l`'s `width` elements at
    `shared_ptr + l * width`. `shared_ptr` lowers to M0 and must be
    wave-uniform, so callers pass the wave's tile base and put the per-lane
    source term in `vector_offset` only.

    The DMA carries the `amdgpu.AsyncCopies` alias scope so consumer LDS reads
    through `_load_from_lds` can overlap it. This makes waits software-owned:
    issue `s_waitcnt vmcnt(0)` before consuming the tile, plus a workgroup
    barrier unless the tile is wave-private. Drain prior LDS reads before
    overwriting a reused tile.

    Parameters:
        dtype: Element type of both the LDS destination and source allocation.
            The caller must ensure these match because `AMDBufferResource` is
            dtype-erased.
        width: Number of elements loaded by each lane.
        cache_policy: AMD cache policy for the source load.

    Args:
        bc: Buffer resource descriptor for the source allocation.
        vector_offset: Per-lane source offset in elements.
        shared_ptr: Wave-uniform LDS tile base; hardware adds the lane stride.
        scalar_offset: Wave-uniform source offset in elements.
    """
    comptime assert is_amd_gpu(), "_load_to_lds is AMD-only"
    comptime num_bytes_per_lane = size_of[dtype]() * width
    comptime assert num_bytes_per_lane in (1, 2, 4, 12, 16), (
        "buffer_load_lds supports 1/2/4 bytes per lane on gfx9 and "
        "additionally 12/16 on gfx950"
    )
    comptime assert cache_policy in (
        CacheOperation.ALWAYS,
        CacheOperation.STREAMING,
    ), "_load_to_lds supports ALWAYS or STREAMING cache policy"
    var desc_ptr = UnsafePointer[
        BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
    ].unsafe_dangling()
    var ptr_to_ptr = UnsafePointer(to=desc_ptr)
    var ptr_to_simd = UnsafePointer(to=bc.desc)
    ptr_to_ptr[0] = ptr_to_simd.bitcast[
        UnsafePointer[BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE]
    ]()[0]
    var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<8>`
    ](desc_ptr)
    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](shared_ptr)
    var vector_offset_bytes = vector_offset * Int32(size_of[dtype]())
    var scalar_offset_bytes = scalar_offset * Int32(size_of[dtype]())

    # The cache policy is an I32Attr. The parameterized stdlib emitter lowers
    # the aux parameter as an index attribute in this toolchain, so keep the
    # two accepted policies as literal local emissions.
    comptime if cache_policy == CacheOperation.STREAMING:
        __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
            alias_scopes=_alias_scope_attr,
            aux=__mlir_attr.`2 : i32`,
            _type=None,
        ](
            desc_ptr_llvm,
            shared_ptr3,
            to_i32(Int32(num_bytes_per_lane)),
            to_i32(vector_offset_bytes),
            to_i32(scalar_offset_bytes),
            to_i32(0),
        )
    else:
        __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
            alias_scopes=_alias_scope_attr,
            aux=__mlir_attr.`0 : i32`,
            _type=None,
        ](
            desc_ptr_llvm,
            shared_ptr3,
            to_i32(Int32(num_bytes_per_lane)),
            to_i32(vector_offset_bytes),
            to_i32(scalar_offset_bytes),
            to_i32(0),
        )


# ===----------------------------------------------------------------------=== #
# LDS transposed reads
# ===----------------------------------------------------------------------=== #


@always_inline
def ds_read_tr16_b64_row(
    tile: TileTensor[_, _, address_space=.SHARED, ...],
) -> SIMD[tile.dtype, 4]:
    """4x16 transposed LDS read via rocdl.ds.read.tr16.b64.

    Each 16-lane "row" loads a 4x16 tile, with per-lane exchange so each
    lane gets a column of the tile as SIMD[dtype, 4].

    Args:
        tile: A 4x16 TileTensor in shared memory (2-byte element type).

    Returns:
        A SIMD[dtype, 4] vector with one column of the transposed tile.
    """
    comptime assert size_of[tile.dtype]() == 2
    comptime assert type_of(tile).static_shape[0] == 4
    comptime assert type_of(tile).static_shape[1] == 16

    comptime thread_layout = row_major[4, 4]()
    var lane_in_row = umod(lane_id(), 16)
    var dist_result = tile.vectorize[1, 4]().distribute_with_offset[
        thread_layout
    ](lane_in_row)
    var offset = dist_result[2]
    var ptr = tile.ptr + offset

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](ptr)

    var llvm_res = __mlir_op.`rocdl.ds.read.tr16.b64`[
        _type=__mlir_type.`vector<4 x bf16>`,
        noalias_scopes=_alias_scope_attr,
        alias_scopes=_no_alias_scope_attr,
    ](shared_ptr3)

    return rebind[SIMD[tile.dtype, 4]](
        __mlir_op.`pop.cast_from_builtin`[_type=SIMD[tile.dtype, 4]._mlir_type](
            llvm_res
        )
    )


@always_inline
def ds_read_tr16_b64_warp[
    mma_shape: IndexList[3],
](tile: TileTensor[_, _, address_space=.SHARED, ...],) -> SIMD[tile.dtype, 4]:
    """Warp-level transposed LDS read distributing across 16-lane rows.

    For 32x32x16 MMA: 2x2 row distribution over 8x32 tile.
    For 16x16x32 MMA: 4x1 row distribution over 16x16 tile.

    Parameters:
        mma_shape: MMA instruction shape (M, N, K).

    Args:
        tile: A TileTensor in shared memory sized for the MMA shape.

    Returns:
        A SIMD[dtype, 4] vector with transposed data for one lane.
    """
    comptime row_dim0 = 2 if mma_shape[0] == 32 else 4
    comptime row_dim1 = 2 if mma_shape[0] == 32 else 1

    comptime assert tile.dtype == .bfloat16
    comptime assert type_of(tile).static_shape[0] == row_dim0 * 4
    comptime assert type_of(tile).static_shape[1] == row_dim1 * 16

    var row_idx = ufloordiv(lane_id(), 16)
    var coord0, coord1 = divmod(row_idx, row_dim1)
    var shared_b_tile = tile.tile[4, 16](coord0, coord1)
    return ds_read_tr16_b64_row(shared_b_tile)


# ===----------------------------------------------------------------------=== #
# MMA operand loads from SMEM
# ===----------------------------------------------------------------------=== #


struct TiledMmaLoader[
    in_type: DType,
    mma_shape: IndexList[3],
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    swizzle2: Optional[Swizzle] = Optional[Swizzle](),
]:
    """SMEM→register loader expert for MFMA operand fragments.

    Sibling to `TiledMmaOp` (static MFMA compute). Stateless: all
    methods are `@staticmethod`. Parameterized by operand dtype, MMA
    instruction shape, and optional vector-space swizzle. Reusable
    wherever a kernel issues MMA-tile-shaped SMEM reads (attention's
    QK / PV matmuls, potential future matmul variants).

    Static methods:

    - `load_b`: full B-operand load from a warp-sized SMEM tile.
      M-outer iteration; handles BF16 single-load and FP8 two-half-load
      (`num_packs` branch) with optional vector-space swizzle.
    - `load_b_tr`: transposed single-MMA-tile load via
      `ds_read_tr16_b64_warp` halves + join (BF16 double-rate shapes:
      32x32x16 and 16x16x32).
    - `load_v_fp8_strip`: FP8 V-operand per-strip load via
      `ds_read_tr8_b64` with paired-lane addressing (16x16x128 FP8 PV
      matmul).

    `_load_b_tile` is a private helper used by `load_b`.

    Parameters:
        in_type: Operand element type.
        mma_shape: MMA instruction shape [M, N, K].
        swizzle: Optional vector-space swizzle for `load_b`.
        swizzle2: Optional second vector-space swizzle, applied AFTER
            `swizzle`. Use to compose two-XOR swizzles (e.g., the
            reference `st_32x32` `bit5^=bit9` + `bit4^=bit10` byte-level
            pair, which are not expressible as a single Swizzle).
    """

    @staticmethod
    @always_inline
    def load_b[
        num_mmas: Int,
        simd_width: Int,
        imm_offset_bytes: Int = 0,
    ](
        src: TileTensor[Self.in_type, _, address_space=.SHARED, ...],
    ) -> Array[
        SIMD[Self.in_type, simd_width], num_mmas
    ]:
        """Full B operand load from a SMEM warp tile.

        Loads all MMA tiles from a WN x BK SMEM warp tile and returns
        them as an Array of SIMD fragments (one per MMA tile).

        Parameters:
            num_mmas: Number of MMA tiles to load.
            simd_width: SIMD vector width for the element type.
            imm_offset_bytes: Comptime byte offset added to each ds_read
                via the `n` (numeric immediate) constraint, bypassing
                the AMDGPU instruction selector for the address-fold
                step. See `_load_from_lds[imm_offset_bytes]`. Cost:
                per-read `s_waitcnt lgkmcnt(0)` serializes LDS reads.

        Args:
            src: A WN x BK TileTensor in shared memory.

        Returns:
            An Array of SIMD fragments, one per MMA tile.
        """
        comptime MMA_M = Self.mma_shape[0]
        comptime MMA_K = Self.mma_shape[2]
        comptime BK = type_of(src).static_shape[1]
        comptime M = type_of(src).static_shape[0] // MMA_M
        comptime N = BK // MMA_K
        comptime load_width = simd_width_of[Self.in_type]()
        comptime mma_frag_width = (MMA_M * MMA_K) // WARP_SIZE
        comptime num_packs = mma_frag_width // load_width
        comptime assert num_packs == 1 or num_packs == 2

        var result = Array[SIMD[Self.in_type, simd_width], num_mmas](
            uninitialized=True
        )
        comptime for i in range(M):
            comptime for j in range(N):
                var src_row = src.tile[MMA_M, BK](Int(i), 0)
                comptime if num_packs == 1:
                    # BF16: single load covers the full fragment.
                    result[Int(i) + Int(j) * M] = rebind[
                        SIMD[Self.in_type, simd_width]
                    ](
                        Self._load_b_tile[
                            Self.mma_shape, Int(j), imm_offset_bytes
                        ](src_row)
                    )
                else:
                    # FP8: MMA K (128) = 2 * SMEM load width (64 elements
                    # per 16-element lane SIMD). Load two [MMA_M, MMA_K/2]
                    # halves and join so the K-dim permutation matches Q.
                    comptime half_k_shape = IndexList[3](
                        MMA_M, Self.mma_shape[1], MMA_K // 2
                    )
                    var lo = Self._load_b_tile[
                        half_k_shape, Int(j) * 2, imm_offset_bytes
                    ](src_row)
                    var hi = Self._load_b_tile[
                        half_k_shape, Int(j) * 2 + 1, imm_offset_bytes
                    ](src_row)
                    result[Int(i) + Int(j) * M] = rebind[
                        SIMD[Self.in_type, simd_width]
                    ](lo.join(hi))
        return result^

    @staticmethod
    @always_inline
    def load_b_tr(
        tile: TileTensor[Self.in_type, _, address_space=.SHARED, ...],
    ) -> SIMD[Self.in_type, 8]:
        """Transposed B operand load for double-rate MFMA shapes.

        Splits the tile along the K dimension into two halves and
        concatenates the results.

        Args:
            tile: A MMA_K x MMA_N TileTensor in shared memory.

        Returns:
            A SIMD[in_type, 8] vector with both halves concatenated.
        """
        comptime assert Self.mma_shape in (
            IndexList[3](32, 32, 16),
            IndexList[3](16, 16, 32),
        )
        comptime assert Self.in_type == .bfloat16
        comptime MMA_K = Self.mma_shape[2]
        comptime MMA_N = Self.mma_shape[1]
        comptime half_k = MMA_K // 2
        comptime assert type_of(tile).static_shape[0] == MMA_K
        comptime assert type_of(tile).static_shape[1] == MMA_N

        var part_1 = ds_read_tr16_b64_warp[Self.mma_shape](
            tile.tile[half_k, MMA_N](0, 0)
        )
        var part_2 = ds_read_tr16_b64_warp[Self.mma_shape](
            tile.tile[half_k, MMA_N](1, 0)
        )
        return part_1.join(part_2)

    @staticmethod
    @always_inline
    def load_v_fp8_strip[
        BN: Int,
        BK: Int,
        bk_tile: Int,
        dt: Int,
    ](
        v_base: UnsafePointer[
            Scalar[Self.in_type], MutAnyOrigin, address_space=.SHARED
        ],
        rel_key: Int,
        hw_key_shift: Int,
        depth_base: Int,
    ) -> SIMD[Self.in_type, 32]:
        """FP8 V per-strip `ds_read_tr8_b64` load for one (bk_tile, dt).

        Paired-lane addressing: issues 4 `ds_read_tr8_b64` calls (at
        `key_base = 0, 16, 32, 48`) and joins the results into one
        32-element SIMD matching the MFMA C-output column pattern for
        the 16x16x128 FP8 V operand in the PV matmul.

        Caller is responsible for precomputing the per-lane coords
        (`rel_key`, `hw_key_shift`, `depth_base`) ONCE before the outer
        (bk_tile, dt) loop: they're lane-only, not
        (bk_tile, dt)-dependent, so hoisting saves redundant address
        math per iteration.

        Parameters:
            BN: V block height in elements.
            BK: V block width in elements.
            bk_tile: Which BK-tall row strip (0..num_k_tiles - 1).
            dt: Which depth-tile within the strip (0..depth/MMA_M - 1).

        Args:
            v_base: Pointer to V SMEM stage base (block 0 of
                `num_repeats`).
            rel_key: Per-lane relative key index within the 16-lane row.
            hw_key_shift: +4 for lanes in hw1, +0 for hw0.
            depth_base: Per-lane depth sub-range offset
                (0 or 8 or 16 or 24).

        Returns:
            `SIMD[in_type, 32]` for this lane's (bk_tile, dt) strip.
        """
        comptime MMA_M = Self.mma_shape[0]
        comptime row_offset = bk_tile * 64
        comptime depth_offset = dt * MMA_M
        comptime blk = depth_offset // BK
        comptime d_in_blk = depth_offset % BK
        var block_base = v_base + blk * BN * BK

        # Match K's swizzle on V reads.  In `mla_kv_alias` mode K and V
        # share one SMEM image: K writes via `SubTileLoaderLDS.load`,
        # which permutes write positions by `swizzle(vec_idx) * simd_w`
        # where `vec_idx = byte_offset / simd_w` (16B for FP8) and
        # `swizzle` XORs bits 4..6 of `vec_idx` into bits 0..2.  To read
        # the same element back, V must apply the same permutation.
        #
        # An element at logical `byte_offset` was written to
        #     swizzle(byte_offset / simd_w) * simd_w + (byte_offset % simd_w)
        # because the swizzle operates only on the vec-aligned part —
        # the low `log2(simd_w)` bits (here 0..3) are the within-vec
        # byte position, which is the same for writer and reader.
        # Paired V lanes use `depth_base ∈ {0, 8, 16, 24}`, i.e. an
        # 8-byte sub-vec offset, so preserving the low 4 bits is
        # required for the 8-byte `ds_read_tr8_b64` to land correctly.
        comptime simd_w = simd_width_of[Self.in_type]()

        @always_inline
        @__parameter
        def _load_keys[key_base: Int]() -> SIMD[Self.in_type, 8]:
            var key = row_offset + key_base + rel_key + hw_key_shift
            var byte_offset = key * BK + d_in_blk + depth_base
            comptime if Self.swizzle:
                # Decompose byte_offset = vec_idx * simd_w + sub_vec,
                # permute vec_idx, then reassemble.
                var vec_idx = byte_offset // simd_w
                var sub_vec = byte_offset & (simd_w - 1)
                var swizzled_vec = Self.swizzle.value()(vec_idx)
                var addr = block_base + swizzled_vec * simd_w + sub_vec
                return ds_read_tr8_b64(addr)
            else:
                return ds_read_tr8_b64(block_base + byte_offset)

        var r0 = _load_keys[0]()
        var r1 = _load_keys[16]()
        var r2 = _load_keys[32]()
        var r3 = _load_keys[48]()
        return r0.join(r1).join(r2.join(r3))

    @staticmethod
    @always_inline
    def load_v_fp8_strip_16[
        BN: Int,
        block_width: Int,
        bk_tile: Int,
        dt: Int,
    ](
        v_base: UnsafePointer[
            Scalar[Self.in_type], MutAnyOrigin, address_space=.SHARED
        ],
        key_group: Int,
        pair_idx: Int,
        is_odd: Int,
    ) -> SIMD[Self.in_type, 32]:
        """FP8 V per-strip `ds_read_tr8_b64` load for one (bk_tile, dt),
        sized for the 16x16x128 MFMA A-operand fragment layout.

        Lane partition for a 64-lane wave (lane id `l`):
          - key_group g    = l // 16     (0..3: 16-lane "rows")
          - pair_idx  p    = (l % 16)/2  (0..7: pair within the row)
          - is_odd    o    = l % 2       (0 or 1)

        Per (bk_tile, dt), one MFMA tile of V is 16 depths * 128 keys
        = 2048 FP8 = 64 lanes * 32 FP8/lane.  Four `ds_read_tr8_b64`
        calls at `key_base ∈ {0, 8, 16, 24}` deliver 8 keys per lane
        each, totaling 32 contiguous keys per lane at one depth.

        Within one `ds_read_tr8_b64` call, each 16-lane row performs
        **two interleaved 8x8 byte transposes** (one over the 8 even
        lanes, one over the 8 odd lanes).  Per the AMD ISA, paired
        even/odd lanes share a key and read 8 depths each:
          - Even lane (p, o=0) reads V[g*32 + key_base + p, depth 0..7]
          - Odd lane  (p, o=1) reads V[g*32 + key_base + p, depth 8..15]
        After the transpose, even lane at pair_idx p in the row gets 8
        keys (key_base..key_base+7 within the group) at depth p; odd
        lane at pair_idx p gets the same 8 keys at depth p+8.

        The per-lane output matches the scalar gather it replaces:
        lane l holds V[key=g*32..g*32+31, depth=butterfly(l%16) + dt*16],
        where `butterfly(p) = (p/2) + (p%2)*8`.  The depth axis is a
        butterfly permutation of the linear ordering: the MFMA's
        A-operand lane->m_h mapping for the 16x16x128 shape consumes
        this permuted layout directly (no post-load permute needed).

        Parameters:
            BN: V block height in elements (keys per block).
            block_width: SMEM block width in depth elements (caller's
                `bk_smem`: not `BK` if the K-split path is active and
                `bk_smem < BK`).
            bk_tile: Which BK-tall row strip (always 0 here; kept for
                API symmetry with the 32x32x64 variant).
            dt: Which depth-tile within the strip
                (0 .. depth/MMA_M - 1).

        Args:
            v_base: Pointer to V SMEM stage base (block 0).
            key_group: Per-lane key-group index (lane_id // 16).
            pair_idx: Per-lane pair index ((lane_id % 16) // 2).
            is_odd: Per-lane parity (lane_id % 2).

        Returns:
            `SIMD[in_type, 32]`: 32 contiguous keys at one depth for
            this lane's (bk_tile, dt).
        """
        comptime assert (
            Self.mma_shape[0] == 16
        ), "load_v_fp8_strip_16 requires MMA_M == 16"
        # The 4 × `ds_read_tr8_b64` schedule below (key_base ∈
        # {0, 8, 16, 24}) hard-encodes the 16x16x128 lane geometry —
        # `num_matrix_reg[16, 128] = 32` per-lane elements as four
        # 8-element joins.  A future shape change (e.g. 16x16x64)
        # would need a different schedule.
        comptime assert (
            Self.mma_shape[2] == 128
        ), "load_v_fp8_strip_16 requires MMA_K == 128"
        # num_k_tiles == 1 for V in the current 16x16x128 wiring, so
        # bk_tile is always 0.  Keep the parameter for API symmetry with
        # the 32x32x64 variant, but assert until a caller actually
        # exercises bk_tile > 0 and the row-stride math is verified.
        comptime assert (
            bk_tile == 0
        ), "load_v_fp8_strip_16: bk_tile > 0 path is not yet validated"
        comptime MMA_M = Self.mma_shape[0]
        # Per-call key span is 128 (= 4 groups * 32 keys), so bk_tile
        # shifts by 128 in key space.
        comptime row_offset = bk_tile * 128
        comptime depth_offset = dt * MMA_M
        comptime blk = depth_offset // block_width
        comptime d_in_blk = depth_offset % block_width
        var block_base = v_base + blk * BN * block_width

        comptime simd_w = simd_width_of[Self.in_type]()
        # Even lanes (is_odd=0) read the low-half-depth-byte octet
        # (d_in_blk + 0..7); odd lanes read the high-half octet
        # (d_in_blk + 8..15).  The transpose then spreads these 16
        # depth bytes across the 16 lanes of the row.
        var depth_base = d_in_blk + is_odd * 8

        @always_inline
        @__parameter
        def _load_keys[key_base: Int]() -> SIMD[Self.in_type, 8]:
            var key = row_offset + key_group * 32 + key_base + pair_idx
            var byte_offset = key * block_width + depth_base
            comptime if Self.swizzle:
                var vec_idx, sub_vec = udivmod(byte_offset, simd_w)
                var swizzled_vec = Self.swizzle.value()(vec_idx)
                var addr = block_base + swizzled_vec * simd_w + sub_vec
                return ds_read_tr8_b64(addr)
            else:
                return ds_read_tr8_b64(block_base + byte_offset)

        var r0 = _load_keys[0]()
        var r1 = _load_keys[8]()
        var r2 = _load_keys[16]()
        var r3 = _load_keys[24]()
        return r0.join(r1).join(r2.join(r3))

    @staticmethod
    @always_inline
    def _load_b_tile[
        tile_mma_shape: IndexList[3],
        k_tile_idx: Int,
        imm_offset_bytes: Int = 0,
    ](
        src: TileTensor[Self.in_type, _, address_space=.SHARED, ...],
    ) -> SIMD[
        Self.in_type, simd_width_of[Self.in_type]()
    ]:
        """Private helper for `load_b`: single MMA sub-tile load.

        Takes the MMA shape as a parameter (rather than reading Self.mma_shape)
        because `load_b` may need to issue loads with a half-K MMA shape on
        the FP8 num_packs==2 path, distinct from the struct's MMA shape.
        """
        comptime MMA_M = tile_mma_shape[0]
        comptime MMA_K = tile_mma_shape[2]
        comptime assert type_of(src).static_shape[0] == MMA_M
        comptime simd_width = simd_width_of[Self.in_type]()
        comptime assert MMA_M == 32 or MMA_M == 16

        # Pick up the sub-tile's element offset from `tile_with_offset`,
        # then add the lane's per-element offset from
        # `distribute_with_offset`. Both offsets are already in
        # `Self.in_type` scalar elements, so no ptr-to-int byte math is
        # needed.
        var sub_tile_res = src.tile_with_offset[MMA_M, MMA_K](
            Coord(Idx[0], k_tile_idx)
        )
        var sub_tile = sub_tile_res[0]
        comptime idx_type = src.linear_idx_type
        var sub_offset = Scalar[idx_type](sub_tile_res[2])

        # BF16: col_major(32, 2) — 32 lanes along M, 2 groups along K.
        # FP8:  col_major(16, 4) — 16 lanes along M, 4 groups along K.
        var offset: Scalar[idx_type]
        comptime if MMA_M == 32:
            comptime thread_layout = col_major[32, 2]()
            var dist_res = sub_tile.vectorize[
                1, simd_width
            ]().distribute_with_offset[thread_layout](lane_id())
            offset = sub_offset + Scalar[idx_type](dist_res[2])
        else:
            comptime thread_layout = col_major[16, 4]()
            var dist_res = sub_tile.vectorize[
                1, simd_width
            ]().distribute_with_offset[thread_layout](lane_id())
            offset = sub_offset + Scalar[idx_type](dist_res[2])

        comptime if Self.swizzle:
            offset = Self.swizzle.value()(
                offset // Scalar[idx_type](simd_width)
            ) * Scalar[idx_type](simd_width)
        comptime if Self.swizzle2:
            offset = Self.swizzle2.value()(
                offset // Scalar[idx_type](simd_width)
            ) * Scalar[idx_type](simd_width)

        return _load_from_lds[
            width=simd_width, imm_offset_bytes=imm_offset_bytes
        ](src.ptr + offset)


# ===----------------------------------------------------------------------=== #
# _load_from_lds: Alias-scoped LDS load
# ===----------------------------------------------------------------------=== #


@always_inline
def _load_from_lds[
    dtype: DType,
    //,
    width: Int = 1,
    imm_offset_bytes: Int = 0,
    typed_imm_offset_bytes: Int = 0,
](
    shared_ptr: UnsafePointer[Scalar[dtype], _, address_space=.SHARED],
) -> SIMD[
    dtype, width
]:
    """Alias-scoped LDS load via LLVM intrinsic with noalias annotations.

    When `imm_offset_bytes != 0`, routes through `ds_read_b128_imm_u32x4`
    inline-asm path with `s_waitcnt lgkmcnt(0)` baked in. This forces
    a comptime byte offset into ds_read's `offset:imm` field: the
    compiler's instruction selector sometimes fails to fold buried
    comptime offsets (e.g., K SMEM stage stride 0x4000 hidden inside
    `select(stage, k_smem_0, k_smem_1)`).

    Trade-off: the inline-asm path serializes LDS reads (per-read
    lgkmcnt(0) defeats pipelining). Use only when the missed offset
    fold cost > the serialization cost.

    When `typed_imm_offset_bytes != 0`, applies the comptime byte
    offset via Mojo pointer arithmetic BEFORE the `llvm.load`. The
    AMDGPU backend's address-fold pattern matcher then folds the
    constant GEP into `ds_read offset:imm`: same hardware
    instruction as the inline-asm path but visible to
    `SIInsertWaitcnts` + `GCNHazardRecognizer` + register allocator.
    Use this in lieu of `imm_offset_bytes` when the per-read
    `lgkmcnt(0)` cost outweighs the codegen benefit (manual inline-asm
    baking bypasses the AMDGPU waitcnt insertion and costs extra spills).

    Structural-clarity primitive: lets callers express
    "ONE hoisted base + per-cell comptime immediate" at the source
    level instead of relying on the LLVM AMDGPU backend to recover
    that shape from per-cell `subtile.ptr` materializations.
    Codegen-equivalent to the equivalent `subtile.ptr` form on the
    MLA K-side (the AMDGPU backend recovers the same `ds_read
    offset:imm`), but makes the hoist explicit at the source level.
    """
    comptime if imm_offset_bytes != 0:
        comptime if dtype == .bfloat16 and width == 8:
            var bf16_ptr = shared_ptr.bitcast[BFloat16]()
            var raw = ds_read_b128_imm_u32x4[offset_bytes=imm_offset_bytes](
                bf16_ptr
            )
            return rebind[SIMD[dtype, width]](bitcast[.bfloat16, 8](raw))
        else:
            comptime assert (
                False
            ), "_load_from_lds[imm_offset_bytes != 0]: only BF16 width=8"
    comptime alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.AsyncCopies", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`
    comptime no_alias_scope_attr = __mlir_attr.`[#llvm.alias_scope<id= "amdgpu.LocalLoads", domain=#llvm.alias_scope_domain<id = "amdgpu.AsyncOps">>]`

    # Apply comptime byte offset via Mojo pointer arithmetic BEFORE
    # the !llvm.ptr<3> conversion. The backend sees the GEP constant
    # and folds it into ds_read's offset:imm field.
    comptime assert (
        typed_imm_offset_bytes % size_of[dtype]() == 0
    ), "_load_from_lds: typed_imm_offset_bytes must be a multiple of dtype size"
    comptime _offset_elts = typed_imm_offset_bytes // size_of[dtype]()
    var shifted_ptr = shared_ptr + _offset_elts

    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](shifted_ptr)

    comptime load_bytes = width * size_of[dtype]()
    comptime alignment = min(load_bytes, 16)

    comptime if dtype == .bfloat16 and width == 4:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<4 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[.bfloat16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == .bfloat16 and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x bf16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[.bfloat16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype == .float16 and width == 4:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<4 x f16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[.float16, 4]._mlir_type
            ](llvm_res)
        )
    elif dtype == .float16 and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x f16>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        return rebind[SIMD[dtype, width]](
            __mlir_op.`pop.cast_from_builtin`[
                _type=SIMD[.float16, 8]._mlir_type
            ](llvm_res)
        )
    elif dtype.is_float8() and width == 8:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<8 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[.uint8, 8]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](rebind[SIMD[.uint8, width]](uint8_vec))
    elif dtype.is_float8() and width == 16:
        var llvm_res = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var uint8_vec = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[.uint8, 16]._mlir_type
        ](llvm_res)
        return bitcast[dtype, width](rebind[SIMD[.uint8, width]](uint8_vec))
    elif dtype.is_float8() and width == 32:
        var llvm_res0 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3)
        var shared_ptr_offset = shifted_ptr + 16
        var shared_ptr3_hi = __mlir_op.`builtin.unrealized_conversion_cast`[
            _type=__mlir_type.`!llvm.ptr<3>`
        ](shared_ptr_offset)
        var llvm_res1 = __mlir_op.`llvm.load`[
            _type=__mlir_type.`vector<16 x i8>`,
            alignment=to_i64(Int64(alignment)),
            noalias_scopes=alias_scope_attr,
            alias_scopes=no_alias_scope_attr,
        ](shared_ptr3_hi)
        var uint8_vec0 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[.uint8, 16]._mlir_type
        ](llvm_res0)
        var uint8_vec1 = __mlir_op.`pop.cast_from_builtin`[
            _type=SIMD[.uint8, 16]._mlir_type
        ](llvm_res1)
        var uint8_vec = rebind[SIMD[.uint8, 16]](uint8_vec0).join(
            rebind[SIMD[.uint8, 16]](uint8_vec1)
        )
        return bitcast[dtype, width](uint8_vec)
    else:
        comptime assert False, "Unsupported dtype/width for _load_from_lds"


# ===----------------------------------------------------------------------=== #
# ds_read_b128_imm: ds_read with comptime imm offset, hazards + lgkmcnt wait
# ===----------------------------------------------------------------------=== #
#
# Inline-asm `ds_read_b128 ... offset:imm` with three guards baked in:
#   1. Pre-issue `s_nop 15 + s_nop 4` (21 cycles, > worst-case 20-cycle
#      XDL→LDS hazard, §7.6).
#   2. Post-issue `s_waitcnt lgkmcnt(0)` so the result VGPRs are
#      definitively populated when the asm returns. INLINEASM is
#      opaque to `SIInsertWaitcnts`, so it may not insert this
#      waitcnt before downstream consumers — bake it in to be safe.
#   3. Pre-issue `s_waitcnt vmcnt(0)` to ensure any pending DMA-to-LDS
#      writes have committed before we read.
# Returns native `<4 x i32>` (the natural ds_read_b128 destination
# layout); caller bitcasts.


@always_inline
def ds_read_b128_imm_u32x4[
    offset_bytes: Int,
](
    base_ptr: UnsafePointer[BFloat16, _, address_space=.SHARED],
) -> SIMD[
    .uint32, 4
]:
    """Issues `ds_read_b128` with a comptime immediate offset and returns the
    loaded 128 bits as `SIMD[.uint32, 4]`.

    Includes pre-issue `s_nop` hazard guard, pre-issue `s_waitcnt vmcnt(0)`
    (DMA completion), and POST-issue `s_waitcnt lgkmcnt(0)` (LDS read
    completion). Caller bitcasts the i32 result.
    """
    comptime assert (
        offset_bytes >= 0 and offset_bytes <= 65535
    ), "ds_read_b128 offset:imm is u16 (0..65535)"
    var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
        _type=__mlir_type.`!llvm.ptr<3>`
    ](base_ptr)
    var addr_i32_raw = __mlir_op.`llvm.ptrtoint`[_type=__mlir_type.i32](
        shared_ptr3
    )
    var addr_i32 = Int32(
        mlir_value=__mlir_op.`pop.cast_from_builtin`[_type=Int32._mlir_type](
            addr_i32_raw
        )
    )
    # Per-read `s_waitcnt lgkmcnt(0)` baked in: required because
    # GCNHazardRecognizer treats INLINEASM as opaque and cannot see
    # the MFMA-write → ds_read-write WAR hazards that would arise
    # when the compiler reuses VGPRs across consecutive reads. A
    # coarser waitcnt (one per load_b call) lets the compiler
    # pipeline reads (using lgkmcnt(N)) but breaks correctness
    # when the destination VGPRs of a new ds_read overlap with the
    # source VGPRs of a still-in-flight MFMA. Tested empirically
    # 2026-04-30 — fine-grained per-read waitcnt is the correct
    # trade-off for kernels with concurrent MFMA + LDS-read flows.
    return inlined_assembly[
        "ds_read_b128 $0, $1 offset:$2",
        SIMD[.uint32, 4],
        constraints="=v,v,n",
        has_side_effect=True,
    ](addr_i32, Int32(offset_bytes))


# ===----------------------------------------------------------------------=== #
# load_lds_fragment: MMA LDS→register load (element-granularity swizzle)
# ===----------------------------------------------------------------------=== #


@always_inline
def load_lds_fragment[
    smem_layout: TensorLayout,
    reg_layout: TensorLayout,
    //,
    MMA_K: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
](
    smem_tile: SMemTile[mut=False, _, smem_layout, _],
    reg_tile: RegTile[mut=True, smem_tile.dtype, reg_layout, _],
):
    """Load MMA fragments from SMEM to registers using hardware access pattern.

    Dimensions are derived from the tile layouts:
        - num_mmas = reg rows, MMA_M = smem rows / num_mmas
        - lds_frag_width = MMA_M * MMA_K / WARP_SIZE
        - lds_row_stride: MMA_K (BF16 dense), smem stride (FP8 or strided)
        - num_iterations = reg flat elements / lds_frag_width

    Parameters:
        smem_layout: Inferred layout of the SMEM source tile.
        reg_layout: Inferred layout of the register destination tile.
        MMA_K: MMA K dimension (hardware instruction width).
        swizzle: Optional element-space swizzle.

    Args:
        smem_tile: Source [num_mmas * MMA_M, K] in SHARED.
        reg_tile: Destination [num_mmas, K_frags * frag_width] in LOCAL.
    """
    comptime dtype = smem_tile.dtype
    comptime smem_rows = smem_layout.static_shape[0]
    comptime smem_cols = smem_layout.static_shape[1]
    comptime smem_stride = smem_layout.static_stride[0]
    comptime num_mmas = reg_layout.static_shape[0]
    comptime reg_cols = reg_layout.static_shape[1]
    comptime reg_stride = reg_layout.static_stride[0]
    comptime MMA_M = smem_rows // num_mmas
    comptime assert MMA_M >= 1 and smem_rows % num_mmas == 0, (
        "load_lds_fragment: MMA_M = smem_rows // num_mmas must be >= 1"
        " and divide evenly."
    )
    comptime mma_frag_width = MMA_M * MMA_K // WARP_SIZE
    comptime assert mma_frag_width >= 1, (
        "load_lds_fragment: mma_frag_width = MMA_M * MMA_K // WARP_SIZE"
        " floored to 0 — MMA shape too small for the wavefront."
    )
    comptime use_fp8_split = (
        dtype.is_float8() and MMA_M == 16 and MMA_K == 128
    )
    comptime lds_frag_width = 16 if use_fp8_split else mma_frag_width
    comptime num_iterations = (num_mmas * reg_cols) // lds_frag_width
    comptime assert num_iterations >= 1, (
        "load_lds_fragment: num_iterations = (num_mmas * reg_cols) //"
        " lds_frag_width floored to 0 — destination register tile too"
        " small for the fragment width."
    )

    # SMEM row stride: when the smem tile is a narrow sub-tile of a wider
    # allocation (stride > cols), use the physical stride. Otherwise use the
    # MMA-native stride: smem_cols for FP8 (contiguous BK), MMA_K for BF16
    # (mma_access_layout packs MMA_K elements per logical row).
    comptime smem_is_subtile = smem_stride > smem_cols
    comptime lds_row_stride = (
        smem_stride if smem_is_subtile else (
            smem_cols if dtype.is_float8() else MMA_K
        )
    )

    # Register stride: strided sub-tile (stride > cols) spaces fragments at
    # row stride; dense tile packs fragments contiguously.
    comptime _reg_stride = (
        reg_stride if reg_stride > reg_cols else lds_frag_width
    )

    var smem_ptr = smem_tile.ptr
    var reg_ptr = reg_tile.ptr
    comptime FragElement = SIMD[dtype, lds_frag_width]

    comptime col_groups = WARP_SIZE // MMA_M
    comptime assert col_groups >= 1, (
        "load_lds_fragment: col_groups = WARP_SIZE // MMA_M floored to 0"
        " (MMA_M > WARP_SIZE)."
    )
    var lane = lane_id()
    var lane_offset = (
        Int(lane % MMA_M) * lds_row_stride + Int(lane // MMA_M) * lds_frag_width
    )

    comptime elements_per_iter = col_groups * lds_frag_width
    comptime use_split_k = lds_row_stride > elements_per_iter

    comptime if use_split_k:
        comptime k_splits = lds_row_stride // elements_per_iter
        comptime m_positions = num_iterations // k_splits
        comptime assert k_splits >= 1 and m_positions >= 1, (
            "load_lds_fragment split-K: k_splits and m_positions must be"
            " >= 1. Geometry inconsistent — would emit zero LDS reads."
        )
        comptime k_stride = elements_per_iter
        comptime m_stride = lds_row_stride * MMA_M

        comptime for m_idx in range(m_positions):
            comptime for k_idx in range(k_splits):
                var iter_base = m_idx * m_stride + k_idx * k_stride
                var full_offset = iter_base + lane_offset

                comptime if swizzle:
                    full_offset = swizzle.value()(full_offset)

                comptime frag_idx = m_idx * k_splits + k_idx
                reg_ptr.store[width=lds_frag_width](
                    frag_idx * _reg_stride,
                    rebind[FragElement](
                        _load_from_lds[width=lds_frag_width](
                            smem_ptr + full_offset
                        )
                    ),
                )
    else:
        comptime for i in range(num_iterations):
            var iter_base = i * WARP_SIZE * lds_frag_width
            var full_offset = iter_base + lane_offset

            comptime if swizzle:
                full_offset = swizzle.value()(full_offset)

            reg_ptr.store[width=lds_frag_width](
                i * _reg_stride,
                rebind[FragElement](
                    _load_from_lds[width=lds_frag_width](smem_ptr + full_offset)
                ),
            )


# ===----------------------------------------------------------------------=== #
# Blocked SMEM navigation
# ===----------------------------------------------------------------------=== #


@always_inline
def smem_subtile[
    tile_rows: Int,
    tile_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=.SHARED],
    tile_row: Int,
    tile_col: Int,
) -> TileTensor[
    dtype,
    type_of(row_major[tile_rows, tile_cols]()),
    MutAnyOrigin,
    address_space=.SHARED,
]:
    """Creates a flat TileTensor sub-view of a blocked SMEM layout.

    The blocked layout has num_repeats contiguous BN x BK blocks. This
    function computes the physical offset for a block-aligned tile and
    returns a row-major TileTensor view with strides (tile_cols, 1).

    Correct only when tile_cols == BK (tiles don't cross block boundaries
    in the column dimension).

    Parameters:
        tile_rows: Height of the sub-tile.
        tile_cols: Width of the sub-tile (must equal BK for block alignment).
        BN: Number of rows per block (full block height).
        BK: Number of columns per block (full block width).
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation.
        tile_row: Tile row index (0-based, in units of tile_rows).
        tile_col: Tile column index (0-based, in units of tile_cols).

    Returns:
        A TileTensor view into the specified sub-tile region.
    """
    comptime block_size = BN * BK
    var offset = tile_row * tile_rows * BK + tile_col * block_size
    return TileTensor[
        dtype,
        type_of(row_major[tile_rows, tile_cols]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ](smem_ptr + offset, row_major[tile_rows, tile_cols]())


@always_inline
def smem_mma_subtile_offset[
    mma_rows: Int,
    mma_cols: Int,
    BN: Int,
    BK: Int,
](bk_tile: Int, k_sub: Int, mma_idx: Int) -> Int:
    """Element offset of an MMA sub-tile within a blocked (BN x BK) SMEM region.

    The physical SMEM layout is `num_repeats` contiguous `BN x BK`
    blocks. This helper computes the scalar-element offset (from the
    stage base) of the MMA sub-tile at `(bk_tile, k_sub, mma_idx)`.

    The offset is layout-agnostic: callers that need a `TileTensor`
    view pair it with whatever within-tile stride their load pattern
    requires (see `smem_mma_subtile` for the default row-major form).
    """
    comptime block_size = BN * BK
    comptime tiles_per_block = BK // mma_cols
    var block_idx = mma_idx // tiles_per_block
    var col_in_block = (mma_idx % tiles_per_block) * mma_cols
    return (
        bk_tile * BK * BK
        + k_sub * mma_rows * BK
        + block_idx * block_size
        + col_in_block
    )


@always_inline
def smem_mma_subtile[
    mma_rows: Int,
    mma_cols: Int,
    BN: Int,
    BK: Int,
    dtype: DType,
](
    smem_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin, address_space=.SHARED],
    bk_tile: Int,
    k_sub: Int,
    mma_idx: Int,
) -> TileTensor[
    dtype,
    type_of(row_major[mma_rows, mma_cols]()),
    MutAnyOrigin,
    address_space=.SHARED,
]:
    """Creates a flat TileTensor for an MMA-sized sub-tile in blocked SMEM.

    Used by the non-transposed (V buffer) load_from_shared path. The V
    buffer's SMEM has shape (BN, depth) with blocked layout
    (num_repeats x BN x BK blocks). Each MMA tile is mma_rows x mma_cols
    within one block. The returned TileTensor uses plain
    `row_major[mma_rows, mma_cols]` strides: only correct when the
    physical row stride equals `mma_cols`. For `mma_cols < BK`, callers
    must pair `smem_mma_subtile_offset` with an explicit-stride layout
    (e.g. `MixedLayout((mma_rows, mma_cols), (BK, 1))`).

    Parameters:
        mma_rows: MMA tile height (e.g., MMA_K=16).
        mma_cols: MMA tile width (e.g., MMA_M=32).
        BN: Block height.
        BK: Block width.
        dtype: Element data type.

    Args:
        smem_ptr: Base pointer to the SMEM allocation for this buffer stage.
        bk_tile: Which BK-tall row group (0..depth/BK-1).
        k_sub: Which MMA_K sub-row within the BK group (0..BK/MMA_K-1).
        mma_idx: Linear MMA tile index across the full depth dimension.

    Returns:
        A TileTensor view into the MMA-sized sub-tile.
    """
    var offset = smem_mma_subtile_offset[mma_rows, mma_cols, BN, BK](
        bk_tile, k_sub, mma_idx
    )
    return TileTensor[
        dtype,
        type_of(row_major[mma_rows, mma_cols]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ](smem_ptr + offset, row_major[mma_rows, mma_cols]())


# ===----------------------------------------------------------------------=== #
# TileLoader trait
# ===----------------------------------------------------------------------=== #


trait TileLoader(TrivialRegisterPassable):
    """DRAM→LDS DMA loader contract for `tile_rows × tile_cols` half-tiles.

    Implementations cooperate as a warp group to fill an SMEM half-tile
    via `buffer_load_*_lds`. The kernel walks coords in `(m_offset,
    k_offset)` GEMM-space; the loader translates them to physical
    addresses internally. Conformers must be `TrivialRegisterPassable`
    so the kernel can pass them by value through closures.

    Two conformers ship today:

    - `TileLoaderLDS`: linear 2D source. Used by matmul A/B operands
      and by conv's B (filter) operand. The address math is
      `addr = (m_offset * stride) + k_offset`.
    - `TileLoaderLDSIm2col`: NHWC + in-line im2col. Used by conv's A
      (input) operand. The address math decomposes
      `m_offset → (n, h_out, w_out)` and `k_offset → (kh, kw, c)` at
      load time; conv geometry (R, S, H, W, stride, dilation, pad) is
      loader-internal state.

    The kernel doesn't have to know which loader is in use; it just
    advances `(m_offset, k_offset)` through the K-loop. That's the
    point of the trait: the conv body and matmul body can share
    everything except which loader they instantiate.
    """

    comptime dtype: DType
    comptime tile_rows: Int
    comptime tile_cols: Int

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[Self.dtype, _, _],
        m_offset: Int,
        k_offset: Int,
    ):
        """Loads a half-tile from global memory into the SMEM dst.

        Issues `num_iterations` `buffer_load_*_lds` bursts (per lane)
        that together fill the `tile_rows × tile_cols` SMEM half-tile.
        Each iteration costs one vmcnt-tracked outstanding load per
        lane: the 4-wave software pipeline relies on this exact
        accounting.

        Args:
            dst: Destination half-tile in SHARED address space, sized
                `tile_rows × tile_cols`.
            m_offset: Row offset (M dimension) of the sub-tile origin
                in GEMM space.
            k_offset: Column offset (K dimension) of the sub-tile
                origin in GEMM space.
        """
        ...


# ===----------------------------------------------------------------------=== #
# TileLoaderLDS: Cooperative Global→LDS loader
# ===----------------------------------------------------------------------=== #


struct TileLoaderLDS[
    dtype_: DType,
    tile_rows_: Int,
    tile_cols_: Int,
    stride: Int,
    num_loading_warps: Int,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    load_width: Int = simd_width_of[dtype_](),
    use_full_tile_width: Bool = False,
](TileLoader):
    """DRAM→LDS DMA expert for warp-group cooperative coord-indexed loads.

    Sibling of `SubTileLoaderLDS` (single-sub-tile TileTensor-indexed).
    This one coordinates a warp group (typically 8 warps) to cooperatively
    fill a half-tile via coord-indexed iteration: `load_tile(dst,
    m_offset, k_offset)` steps through `num_iterations` BK-wide rows,
    optionally applying a per-iteration byte-space swizzle for LDS
    bank-conflict avoidance. Matmul's DRAM→LDS pattern (ping-pong, etc.).

    Uses stdlib `AMDBufferResource.load_to_lds` directly: no alias scope
    attached. Matmul's scheduling uses `s_sched_group_barrier` hints,
    which don't qualify as the runtime fence required by the
    `SIInsertWaitcnts` vmcnt-relaxation contract; attaching the scope
    would miscompile (see the `async_copies` docstring on
    `load_to_lds`). For attention patterns that DO satisfy the contract
    via explicit `s_waitcnt vmcnt(0) + s_barrier` fences, use
    `SubTileLoaderLDS` instead.

    Parameters:
        dtype_: Element data type. Re-bound to `dtype` at body scope to
            match the `TileLoader` trait alias.
        tile_rows_: Height of each half-tile to load. Re-bound to
            `tile_rows` at body scope.
        tile_cols_: Width (K dimension) of each half-tile. Re-bound to
            `tile_cols` at body scope.
        stride: Row stride of the source GMEM tensor.
        num_loading_warps: Warps cooperating on each load (typically 8).
        swizzle: Optional byte-space swizzle for LDS bank conflicts.
        load_width: Elements per load (SIMD width).
        use_full_tile_width: FP8 row-major mode.
    """

    # Body-level aliases re-bind the parametric `dtype_`/`tile_rows_`/
    # `tile_cols_` (trailing-underscore naming required to avoid the
    # struct-param vs trait-alias name clash) to the names the trait
    # declares. Lets the `TileLoader` conformance machinery match, and
    # lets the rest of this struct keep its `Self.dtype` / `Self.tile_*`
    # references unchanged.
    comptime dtype: DType = Self.dtype_
    comptime tile_rows: Int = Self.tile_rows_
    comptime tile_cols: Int = Self.tile_cols_

    comptime subtile_cols = Self.tile_cols if Self.use_full_tile_width else 32
    comptime threads_per_row = Self.subtile_cols // Self.load_width
    comptime thread_rows = WARP_SIZE // Self.threads_per_row

    comptime num_warp_cols = Self.tile_cols // Self.subtile_cols
    comptime num_warp_rows = Self.num_loading_warps // Self.num_warp_cols

    comptime elements_per_warp = WARP_SIZE * Self.load_width
    comptime rows_per_warp = Self.elements_per_warp // Self.tile_cols

    comptime loading_threads = Self.num_loading_warps * WARP_SIZE
    comptime rows_per_iteration = Self.loading_threads // (
        Self.tile_cols // Self.load_width
    )
    # `ceildiv` (not floor-div) so that under-supplied sub-tiles —
    # `tile_rows < rows_per_iteration` (e.g. half_BM=32 + BK=32 + bf16
    # where 256 loading threads can transfer the whole 32x32 sub-tile in
    # ~half a wave) — get `num_iterations == 1`. Floor-div rounded this
    # to 0, the `comptime for i in range(0)` unrolled 0 times, and the
    # loader silently emitted zero `buffer_load_lds` — LDS uninit, MMA
    # garbage. The `load_tile` body gates over-supplied warps via
    # `warp_id < active_warps_this_iter` (computed per-iter from
    # `total_warp_rows`) so warps mapped past the sub-tile boundary
    # skip the load (vmcnt unaffected for them — s_waitcnt is a no-op).
    comptime num_iterations = ceildiv(
        Self.total_warp_rows, Self.num_loading_warps
    )
    # Total warp-rows of work across all iterations; clamped against
    # `num_iterations * num_loading_warps` from above. The per-iter
    # `active_warps_this_iter` derives from this.
    comptime total_warp_rows = ceildiv(Self.tile_rows, Self.rows_per_warp)

    comptime warp_subtile_bytes = Self.rows_per_warp * Self.tile_cols * size_of[
        Self.dtype
    ]()
    comptime lane_load_bytes = Self.load_width * size_of[Self.dtype]()
    comptime row_bytes = Self.tile_cols * size_of[Self.dtype]()

    comptime _needs_per_iter_swizzle = Bool(
        Self.swizzle
    ) and Self.use_full_tile_width

    var buffer: AMDBufferResource
    var thread_row: Int
    var thread_col: Int
    var warp_id: Int
    var lane_id: Int
    # Block anchor in (M, K) GEMM space. Caller-supplied at construction.
    # `load_tile(m_offset, k_offset)` addresses `(m_anchor + m_offset,
    # k_anchor + k_offset)` against the loader's SRD. Lets callers point
    # the SRD at the full A/B tensor and absorb the per-block origin into
    # the loader, so the K-loop callsite is uniform across split-K /
    # multi-block kernels and across the conv `TileLoaderLDSIm2col`
    # sibling. Defaults to 0 — passing a per-block-sliced `src` with
    # zero anchors yields the legacy SRD-bounds-to-block behavior.
    var m_anchor: Int
    var k_anchor: Int

    @always_inline
    def __init__(
        out self,
        src: TileTensor[Self.dtype, ...],
        warp_id: Int,
        lane_id: Int,
        *,
        m_anchor: Int = 0,
        k_anchor: Int = 0,
    ):
        """Builds the loader.

        Args:
            src: GMEM tile to source from. Pass the full A/B tensor and
                set `m_anchor`/`k_anchor` to the per-block origin, or
                pass a pre-sliced block tile with zero anchors (legacy
                behavior). The full-tensor form lets the SRD's
                `num_records` bound the actual allocation rather than
                the block view: required for split-K kernels and
                for parity with `TileLoaderLDSIm2col`.
            warp_id: Warp identifier within the loading warp group.
            lane_id: Lane identifier within the warp.
            m_anchor: M-coordinate (row dim) of the block origin in
                the loader's SRD coordinate system. Added to
                `m_offset` at load time. Defaults to 0.
            k_anchor: K-coordinate (column dim) of the block origin.
                Added to `k_offset` at load time. Defaults to 0.
        """
        # === Tier 2/3 sanity asserts: catch silently-zeroed counts ===
        # The class body computes a chain of integer divisions
        # (`subtile_cols // load_width`, `tile_cols // subtile_cols`,
        # ..., `tile_rows // rows_per_iteration`). When any link
        # flooris to 0, downstream `comptime for` loops unroll 0
        # times — loader emits no `buffer_load_lds` at all, LDS stays
        # uninitialized, MMA reads garbage. These asserts make every
        # link's invariant explicit.
        comptime assert Self.threads_per_row >= 1, (
            "threads_per_row = subtile_cols // load_width must be >= 1"
            " (subtile_cols >= load_width)."
        )
        comptime assert (
            Self.subtile_cols % Self.load_width == 0
        ), "subtile_cols must be a multiple of load_width."
        comptime assert Self.num_warp_cols >= 1, (
            "num_warp_cols = tile_cols // subtile_cols must be >= 1"
            " (tile_cols >= subtile_cols)."
        )
        comptime assert (
            Self.tile_cols % Self.subtile_cols == 0
        ), "tile_cols must be a multiple of subtile_cols."
        comptime assert Self.num_loading_warps % Self.num_warp_cols == 0, (
            "num_loading_warps must be a multiple of num_warp_cols"
            " (otherwise num_warp_rows loses warps)."
        )
        comptime assert Self.rows_per_warp >= 1, (
            "rows_per_warp = WARP_SIZE * load_width // tile_cols must be"
            " >= 1 (each warp must cover at least one row)."
        )
        comptime assert Self.rows_per_iteration >= 1, (
            "rows_per_iteration must be >= 1 (loading_threads /"
            " (tile_cols / load_width)). Sub-tile too narrow for"
            " 4-wave coverage."
        )
        comptime assert Self.num_iterations >= 1, (
            "num_iterations = ceildiv(total_warp_rows, num_loading_warps)"
            " == 0 — tile_rows must be >= 1."
        )
        comptime assert Self.total_warp_rows >= 1, (
            "total_warp_rows = ceildiv(tile_rows, rows_per_warp) == 0"
            " — tile_rows must be >= 1."
        )
        self.buffer = make_amd_buffer_resource(src)
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.m_anchor = m_anchor
        self.k_anchor = k_anchor

        var effective_lane = lane_id

        comptime if Self.swizzle and not Self._needs_per_iter_swizzle:
            var lds_write_bytes = (
                lane_id * Self.load_width * size_of[Self.dtype]()
            )
            var swizzled_bytes = Self.swizzle.value()(lds_write_bytes)
            effective_lane = swizzled_bytes // (
                Self.load_width * size_of[Self.dtype]()
            )

        var warp_row, warp_col = divmod(warp_id, Self.num_warp_cols)
        var subtile_row, subtile_col_idx = divmod(
            effective_lane, Self.threads_per_row
        )
        var subtile_col = subtile_col_idx * Self.load_width

        self.thread_row = warp_row * Self.thread_rows + subtile_row
        self.thread_col = warp_col * Self.subtile_cols + subtile_col

    @always_inline
    def load_tile(
        self,
        dst: SMemTile[Self.dtype, _, _],
        m_offset: Int,
        k_offset: Int,
    ):
        """Loads a half-tile from GMEM into SMEM dst via `load_to_lds`.

        The effective GEMM-space coordinate is `(m_anchor + m_offset,
        k_anchor + k_offset)`, so callers using the legacy
        pre-sliced-block form (anchors=0) keep their address math
        unchanged.

        Args:
            dst: Destination TileTensor in SHARED (half-tile sized).
            m_offset: Row offset (M dim) within the block.
            k_offset: Column (K dim) offset within the block.
        """
        comptime SmemPtr = Pointer[
            Scalar[Self.dtype], MutAnyOrigin, address_space=.SHARED
        ]

        comptime if not Self._needs_per_iter_swizzle:
            comptime assert (
                Self.rows_per_iteration
                == Self.num_loading_warps * Self.rows_per_warp
            ), (
                "rows_per_iteration must equal num_loading_warps *"
                " rows_per_warp on the non-swizzled path; a mismatch means the"
                " iteration stride and the rows actually issued disagree,"
                " silently under-covering the tile."
            )
        comptime assert Self.tile_rows % Self.rows_per_warp == 0, (
            "rows_per_warp must divide tile_rows, else the last warp-tile runs"
            " past the destination."
        )

        var m_eff = self.m_anchor + m_offset
        var k_eff = self.k_anchor + k_offset

        comptime if Self._needs_per_iter_swizzle:
            var lane_byte = self.lane_id * Self.lane_load_bytes

            comptime for i in range(Self.num_iterations):
                # When `tile_rows < num_loading_warps * rows_per_warp`,
                # the last (or only) iteration covers fewer than the
                # full warp grid. Gate over-supplied warps so they
                # don't issue a load_to_lds past the sub-tile end.
                # See `total_warp_rows` doc for rationale.
                comptime active_warps_this_iter = min(
                    Self.num_loading_warps,
                    Self.total_warp_rows - i * Self.num_loading_warps,
                )
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var full_byte = tile_idx * Self.warp_subtile_bytes + lane_byte
                var swizzled_byte = Self.swizzle.value()(full_byte)

                var swizzled_row = swizzled_byte // Self.row_bytes
                var swizzled_col = (swizzled_byte % Self.row_bytes) // size_of[
                    Self.dtype
                ]()

                var lane_offset = swizzled_col + swizzled_row * Self.stride
                var uniform_offset = k_eff + m_eff * Self.stride

                comptime if active_warps_this_iter == Self.num_loading_warps:
                    self.buffer.load_to_lds[width=Self.load_width](
                        Int32(lane_offset),
                        smem_ptr,
                        scalar_offset=Int32(uniform_offset),
                    )
                else:
                    # Wavefront-uniform SGPR branch (warp_id is warp-
                    # scope). Gated warps never issue → vmcnt stays 0,
                    # downstream s_waitcnt vmcnt(N) is a no-op for them.
                    if Int(self.warp_id) < active_warps_this_iter:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(lane_offset),
                            smem_ptr,
                            scalar_offset=Int32(uniform_offset),
                        )
        else:
            var lane_offset = self.thread_col + self.thread_row * Self.stride

            comptime for i in range(Self.num_iterations):
                comptime active_warps_this_iter = min(
                    Self.num_loading_warps,
                    Self.total_warp_rows - i * Self.num_loading_warps,
                )
                var tile_idx = i * Self.num_loading_warps + self.warp_id
                var warp_tile = dst.tile[Self.rows_per_warp, Self.tile_cols](
                    tile_idx, 0
                )
                var smem_ptr = readfirstlane(rebind[SmemPtr](warp_tile.ptr))

                var tile_row = m_eff + i * Self.rows_per_iteration
                var uniform_offset = k_eff + tile_row * Self.stride

                comptime if active_warps_this_iter == Self.num_loading_warps:
                    self.buffer.load_to_lds[width=Self.load_width](
                        Int32(lane_offset),
                        smem_ptr,
                        scalar_offset=Int32(uniform_offset),
                    )
                else:
                    if Int(self.warp_id) < active_warps_this_iter:
                        self.buffer.load_to_lds[width=Self.load_width](
                            Int32(lane_offset),
                            smem_ptr,
                            scalar_offset=Int32(uniform_offset),
                        )


# ===----------------------------------------------------------------------=== #
# SubTileLoaderLDS
# ===----------------------------------------------------------------------=== #


struct SubTileLoaderLDS[
    dtype: DType,
    swizzle: Optional[Swizzle] = Optional[Swizzle](),
    swizzle2: Optional[Swizzle] = Optional[Swizzle](),
](TrivialRegisterPassable):
    """DRAM→LDS DMA expert for single-sub-tile TileTensor-indexed loads.

    Sibling of `TileLoaderLDS` (warp-group cooperative coord-indexed).
    This one issues one `buffer_load_*_lds` burst per `.load()` call for
    a single source sub-tile. Attention's KV-cache warp DMA pattern:
    each warp claims a `(warp_tile_rows, BK)` slice of a
    `(BN, K-span)` DRAM tile and streams it into its SMEM lane.

    The AMD `buffer_load_*_lds` intrinsic is emitted with the
    `amdgpu.AsyncCopies` alias scope via `rocdl.raw.ptr.buffer.load.lds`
    so consumer-side LDS reads tagged with
    `noalias_scopes=_alias_scope_attr` (see `ds_read_tr*` at lines 96,
    419-480) can skip `s_waitcnt vmcnt(0)`: LLVM PR #74537's
    `SIInsertWaitcnts` vmcnt-relaxation handshake. Safe because
    attention kernels also maintain an explicit
    `s_waitcnt vmcnt(0) + s_barrier` fence at DMA/compute boundaries.

    Why not `stdlib load_to_lds[async_copies=True]`: stdlib's
    `async_copies=True` attaches its OWN `alias_scope` MLIR attribute
    which is textually identical to `_alias_scope_attr` but an
    MLIR-distinct object; `ScopedNoAliasAA` matches by identity, so
    the DMA and LDS-consumer scopes don't match and the relaxation is
    silently disabled (MLA regresses 0.76 abs at output[0,0,0,0], the
    same signature as `b7b68a00290`). Keeping the intrinsic emission
    local to this file so producer + consumer share the exact same
    `_alias_scope_attr` object. If stdlib ever exports the scope as a
    shareable symbol, collapse this body to `bc.load_to_lds[
    async_copies=True]`.

    Constructs the `AMDBufferResource` once from a DRAM tile (which may
    carry `Scalar` valid_rows for bounds clamping via `MixedLayout`).
    Each `load()` call reuses the descriptor: one shared bc per tile,
    zero per-warp overhead for buffer resource construction. SRD bounds
    computed by `make_amd_buffer_resource` via `_get_bounds`; hardware
    clamps OOB reads to zero.

    Parameters:
        dtype: Element data type.
        swizzle: Optional swizzle for bank conflict reduction.
        swizzle2: Optional second swizzle, applied AFTER `swizzle`. Use
            to compose two-XOR swizzles (e.g., the reference `st_32x32`
            `bit5^=bit9` + `bit4^=bit10` byte-level pair, which are not
            expressible as a single Swizzle).
    """

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM access."""

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[Self.dtype, Engine=DefaultEngine[], ...],
    ):
        """Create a loader from a DRAM tile.

        The tile's layout carries the valid row count (via Scalar
        dim[0] in MixedLayout). make_amd_buffer_resource reads that
        dimension to compute the SRD size.

        Args:
            gmem_tile: The full DRAM tile from KVCacheIterator.
        """
        self.bc = make_amd_buffer_resource(gmem_tile)

    @always_inline
    def load[
        hoist_scalar_offset: Bool = False,
    ](
        self,
        dst: TileTensor[Self.dtype, _, _, address_space=.SHARED, ...],
        src: TileTensor[Self.dtype, ...],
        scalar_offset: Int = 0,
        worker_base: Int = 0,
    ):
        """Load a warp sub-tile from DRAM to LDS.

        The src tile should be a warp-sized sub-tile of the original DRAM
        tile. Offsets are computed relative to the bc's base pointer, so
        the src pointer must be within the original tile's address range.

        Comptime `hoist_scalar_offset` selects which scalar-offset
        codegen path the inner loop takes:

        * `False` (default, legacy codegen): `scalar_offset` is ignored.
          Each iteration recomputes `Int(src_partitions.ptr) - dram_base`
          where `src_partitions` is the per-iteration sub-tile of `src`.
          Matches the pre-refactor inline DMA emission: `s_add` of the
          per-iter pointer base + bc-base subtract. This is what
          `MhaPrefillV2`, `KVBuffer`, and `_MlaKDmaPair` at KV<128 want:
          the legacy SGPR pressure profile that benches verified at
          KV=64 (no -17% regression).
        * `True` (opt-in hoist): the caller's `scalar_offset` is used
          directly + comptime `partition_offset_bytes`. The runtime
          piece is computed ONCE at the call site and shared across the
          inner loop iterations. This is what `_MlaKDmaPair` at KV>=128
          wants: one SGPR carries the hoisted base across both
          dma_nope+dma_rope, giving the +7% KV=128 lift.

        Args:
            dst: Destination TileTensor in shared memory.
            src: Source TileTensor in global memory (warp sub-tile).
            scalar_offset: Wave-uniform byte offset of `src` relative
                to the buffer-resource base. Only consumed when
                `hoist_scalar_offset` is `True`; pass `0` (or any
                value, it is dead-code-eliminated) when `False`.
            worker_base: Sub-tile row-strip index for cooperative
                half-sub-block loads (N-warps-per-subblock partition at
                depths < 128). When a caller splits a `BM`-row sub-block
                across N warps and passes each warp its own `M = BM/N`-row
                strip, the loader's internal `m_sub_tile` collapses to
                `{0}` and the swizzle would be computed as if the strip
                were the FIRST sub-row, dropping the
                `m_sub_tile * WARP_SIZE` worker offset that the two-XOR
                `st_32x32_s` swizzle needs (the `Swizzle(1,0,6)` bit-0
                flip keys off worker bit 6). Pass the strip's absolute
                sub-row index here so the swizzle matches the consumer
                read (`MhaMmaOp.load_K`). Default 0 = full sub-block load
                (depth 128), unchanged.

        Parameters:
            hoist_scalar_offset: Comptime flag selecting between legacy
                per-iter codegen (`False`, default) and the explicit
                caller-supplied hoisted base (`True`).
        """
        comptime M = type_of(src).static_shape[0]
        comptime N = type_of(src).static_shape[1]
        # `BM` is the outer warp-strip height in rows. Default is 32
        # (one reference `st_32x32_s` sub-block). For half-sub-block loads
        # (N-warps-per-subblock partition at depths < 128), the caller
        # passes M=16 — clamp BM accordingly so the outer
        # `range(M // BM)` is 1 instead of 0.
        comptime BM = 32 if M >= 32 else M
        comptime BN = N
        # Adapt thread layout to keep bytes/lane ≤ 16
        # (buffer_load_dwordx4_lds limit).  bf16 BK=64 and fp8 BK=128
        # would load 32 B/lane with the base 16×4 layout, so widen to
        # 8×8.
        comptime raw_load_bytes = (N * size_of[Self.dtype]()) // 4
        comptime thread_rows = 16 if raw_load_bytes <= 16 else 8
        comptime thread_cols = 4 if raw_load_bytes <= 16 else 8
        comptime thread_layout = row_major[thread_rows, thread_cols]()
        comptime load_width = BN // thread_cols
        comptime BM_SUB = thread_rows if BM >= thread_rows else BM

        var worker_idx = lane_id()
        # `dram_base` is needed only by the legacy non-hoisted path.
        # When `hoist_scalar_offset=True` the caller has already
        # subtracted bc.base from src.ptr (or some hoisted parent
        # tile's ptr), so the SGPR read here is dead and DCE'd.
        var dram_base = self.bc.get_base_ptr()

        comptime dst_stride0 = type_of(dst).static_stride[0]
        comptime dst_stride1 = type_of(dst).static_stride[1]
        comptime assert dst_stride1 == 1
        comptime assert dst_stride0 == BN

        # The comptime partition offset is folded into the `s_add`
        # immediate alongside the runtime piece in BOTH codegen modes;
        # only the source of the runtime piece differs (hoisted caller
        # value vs per-iter `Int(src_partitions.ptr) - dram_base`).
        comptime src_stride0_bytes = (
            type_of(src).static_stride[0] * size_of[Self.dtype]()
        )
        comptime row_bytes = BN * size_of[Self.dtype]()

        comptime for n_tile, m_tile, m_sub_tile in product(
            range(N // BN), range(M // BM), range(BM // BM_SUB)
        ):
            var dst_partitions = dst.tile[BM, BN](m_tile, n_tile).tile[
                BM_SUB, BN
            ](m_sub_tile, 0)
            var src_partitions = src.tile[BM, BN](m_tile, n_tile).tile[
                BM_SUB, BN
            ](m_sub_tile, 0)
            var worker_idx_with_offset = (
                worker_idx + (m_sub_tile + worker_base) * WARP_SIZE
            )
            var swizzled_worker_idx = worker_idx_with_offset
            comptime if Self.swizzle:
                swizzled_worker_idx = Self.swizzle.value()(swizzled_worker_idx)
            comptime if Self.swizzle2:
                swizzled_worker_idx = Self.swizzle2.value()(swizzled_worker_idx)
            var src_dist = src_partitions.vectorize[1, load_width]().distribute[
                thread_layout
            ](umod(swizzled_worker_idx, WARP_SIZE))
            var dst_ptr = dst_partitions.ptr.address_space_cast[.SHARED]()

            var desc_ptr_ = UnsafePointer[
                BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
            ].unsafe_dangling()
            var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
            var ptr_to_simd = UnsafePointer(to=self.bc.desc)
            ptr_to_ptr[0] = ptr_to_simd.bitcast[
                UnsafePointer[
                    BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
                ]
            ]()[0]
            var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<8>`
            ](desc_ptr_)
            var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<3>`
            ](dst_ptr)

            comptime num_bytes_per_lane = size_of[Self.dtype]() * load_width
            # The partition offset within `src` is fully comptime — it's
            # the sum of the outer tile coords times the comptime
            # `src_stride0_bytes` and `row_bytes`. Adding it as a Mojo
            # `Int` comptime constant lets the AMDGPU backend fold it
            # into the `s_add` immediate alongside the runtime piece.
            comptime partition_offset_bytes = (
                (m_tile * BM + m_sub_tile * BM_SUB) * src_stride0_bytes
                + n_tile * row_bytes
            )
            var vector_offset_bytes = Int(src_dist.ptr) - Int(
                src_partitions.ptr
            )
            # Branch by comptime `hoist_scalar_offset`. Both paths
            # produce the SAME numeric scalar_offset_bytes; they differ
            # only in WHICH SSA value the AMDGPU backend sees as the
            # runtime piece — `src_partitions.ptr` (rematerialized per
            # iteration, legacy) or the caller-supplied `scalar_offset`
            # (hoisted across iterations and potentially across
            # `.load()` calls sharing the same loader).
            var scalar_offset_bytes: Int

            comptime if hoist_scalar_offset:
                scalar_offset_bytes = scalar_offset + partition_offset_bytes
            else:
                scalar_offset_bytes = Int(src_partitions.ptr) - dram_base

            __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
                alias_scopes=_alias_scope_attr,
                aux=__mlir_attr.`0 : i32`,  # default cache policy
                _type=None,
            ](
                desc_ptr_llvm,
                shared_ptr3,
                to_i32(Int32(num_bytes_per_lane)),
                to_i32(Int32(vector_offset_bytes)),
                to_i32(Int32(scalar_offset_bytes)),
                to_i32(0),
            )


# ===----------------------------------------------------------------------=== #
# SubTileLoaderLDS_st_8x32: reference st_8x32_s-aligned cooperative DMA
# ===----------------------------------------------------------------------=== #


struct SubTileLoaderLDS_st_8x32[
    dtype: DType,
    BN: Int,
    depth: Int,
    BK: Int,
    num_threads: Int,
    v_full_v227: Bool = False,
](TrivialRegisterPassable):
    """DRAM→LDS DMA for the reference `st_8x32_s` SMEM layout (V operand).

    Mirrors the reference's group-level cooperative `load()`:

      * Each thread (laneid 0..63 across all `num_threads / 64` warps)
        writes `bytes_per_thread = 16` bytes per iteration directly to
        LDS at the natural byte offset
        `lane_byte_offset = thread_id * 16 + iter * num_threads * 16`.
      * `thread_id` is `warp_id * WARP_SIZE + lane_id`, so the LDS bytes
        cover successive subtiles in the reference's row-major-by-block-col
        ordering (`subtile_id = subtile_row * subtiles_per_row +
        subtile_col`, with subtile shape 8×BK BF16).
      * Each lane reads its source position in DRAM via the swizzle ↔
        position bijection: subtile_lane_byte_offset → (row, col)
        within the 8×BK subtile, which then unpacks back into a
        (global_row, global_col) DRAM byte address. For the reference
        `st_8x32` BF16 the swizzle is the identity, so the global position
        is just the natural subtile-local position.
      * Writes go via `rocdl.raw.ptr.buffer.load.lds` with the same
        `_alias_scope_attr` SubTileLoaderLDS uses, so consumer-side
        `ds_read_tr*` LDS reads tagged `noalias_scopes=_alias_scope_attr`
        can skip `s_waitcnt vmcnt(0)` (LLVM PR #74537's
        `SIInsertWaitcnts` vmcnt-relaxation handshake), provided the
        kernel maintains an explicit `s_waitcnt vmcnt(0) + s_barrier`
        fence at DMA/compute boundaries.

    The layout is hard-coded to the reference's BF16 `st_8x32_s`:
      * `subtile_rows = 8`
      * `subtile_cols = BK` (32 for the V2 attention kernels)
      * No swizzle (st_8x32 BF16 returns the offset unchanged)

    For the K operand (the reference uses `st_32x32_s` with a two-XOR
    swizzle), see `SubTileLoaderLDS` + `swizzle/swizzle2` plumbing instead.

    Parameters:
        dtype: Element data type (must be BF16: the `st_8x32_s`
            specialization assumes 2-byte elements; FP32 would use a
            different shape).
        BN: KV block height in elements (= 64 for the V2 attention
            kernels).
        depth: V tile column span in elements (= D for the model's
            head_dim; 64, 128, or 256 for the V2 attention kernels).
        BK: Subtile column span in elements (= 32 for the reference
            `st_8x32_s`).
        num_threads: Total threads in the cooperative load (= 8 warps ×
            64 lanes = 512 for the V2 attention kernels). Used to compute
            `bytes_per_iter`.
        v_full_v227: Reference `v227` V LDS layout (Bool). Default False →
            byte-identical, the production `st_8x32` contiguous
            fill. When True, the WRITE side of the reference V adapter: each
            cooperative-DMA 16-byte run (one key's 16 depth cols) is written
            to the LDS byte the reference `v227` `ds_read_b64_tr_b8` read
            expects, instead of the natural `st_8x32` contiguous byte. The DRAM
            source (`global_byte_in_tile`) is UNCHANGED: only the LDS
            destination is remapped. The closed form (FP8 32×32×64, DEPTH=128,
            KV=128, with `key = global_row 0..127` and
            `depth = global_col 0..127`) is
            `lds_byte = c*0x410 + Lp*16 + depth`, where
            `c = (((key>>1)&1)<<1 | ((key>>2)&1)) + ((key>>3)&1)*4 +
            ((key>>6)&1)*8` and
            `Lp = (key&1)*8 + ((key>>4)&1)*16 + ((key>>5)&1)*32`.
            This is the `W` of the adapter `W∘R` pair. `R` is
            `MhaMmaOp.precompute_v_lane_base[v_full_v227=True]` (the `v227`
            per-lane base) + `load_V_frag[v_full_v227=True]` (the faithful
            readout cell `i_strip*0x2080 + j_depth*0x20 + r*0x100`). The two
            compose to the IDENTITY fragment: `W` is derived as the byte
            permutation `pi: ours_read_addr -> ref_read_addr` over the slot,
            PROVEN a bijection that leaves the `ds_read_tr8_b64` transpose
            invariant (both reads issue the identical tr8 op + 4-subread join,
            differing only in the LDS address, so the transpose cancels). The
            consumer MUST set `v_full_v227=True` too or V scrambles. The slot
            must hold ≥ 16624 B (max `lds_byte` 16623): the `_V_SLOT_PAD_ROWS`
            (256 B / 4 rows) padding `MlaPrefillV2` already allocates.
            Used ONLY by `MlaPrefillV2` (the reference research kernel), where
            it is the default-on reference V LDS adapter. The production V2 MHA
            / MLA loaders build this type at `v_full_v227=False`
            (byte-identical).
    """

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM access."""

    comptime _v227_layout = get_defined_bool["v227_layout", False]()
    """CuTe-style Layout-Algebra spelling of the `v_full_v227` V LDS adapter
    (`-D v227_layout`, default False / GATED OFF). Drives BOTH halves: the
    WRITE branch here expresses the SAME source byte + LDS M0 via `crd2idx` over
    per-bit `Coord`s (whose strides carry the chunk->key bit-permutation +
    skews), and `MlaPrefillV2` reads the same flag and threads it into the
    READ base (`precompute_v_lane_base[v227_layout]`). The two spellings are
    numerically equivalent; the Layout form is a clarity/enablement choice,
    not a codegen win: `crd2idx`'s generic divmod is heavier than the hand
    bit-ops, so the hand path is the default. `-D v227_layout=true`
    selects the Layout spelling on both sides; the default is the hand path."""

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[Self.dtype, Engine=DefaultEngine[], ...],
    ):
        """Create a loader from a DRAM tile.

        Args:
            gmem_tile: The full DRAM tile from KVCacheIterator (carries
                a Scalar valid_rows for clamping bounds).
        """
        self.bc = make_amd_buffer_resource(gmem_tile)

    @always_inline
    def load(
        self,
        v_smem_slot: SMemTile[Self.dtype, ...],
        v_gmem_tile: TileTensor[Self.dtype, ...],
        warp_id_uniform: Int,
        lane_id_local: Int,
        scalar_offset: Int,
    ):
        """Cooperatively DMA one V tile from DRAM into LDS.

        `scalar_offset` is the runtime-uniform byte offset of
        `v_gmem_tile` relative to the buffer-resource's base. Caller
        computes this once (typically
        `Int(v_gmem_tile.ptr) - Int(self.bc.get_base_ptr())`) and passes
        the value here. The method uses it as the
        `rocdl.raw.ptr.buffer.load.lds` scalar-offset argument.

        For callers that construct the loader from the same
        `v_gmem_tile` they pass here, `scalar_offset` is 0: both
        common production paths (`MhaPrefillV2._dma_v` and
        `MlaPrefillV2Core._dma_v`) hit this case.

        Args:
            v_smem_slot: Destination V SMEM tile (must hold at least
                `BN * depth * size_of[dtype]` bytes; the loader uses
                `.ptr` as the LDS base).
            v_gmem_tile: TileTensor view of the BN × depth source tile
                in DRAM.
            warp_id_uniform: Wave-uniform warp index (0..num_warps-1).
                Caller must pass an SGPR-class value (e.g.,
                `readfirstlane(warp_id())`).
            lane_id_local: Per-lane index (0..WARP_SIZE-1) from
                `lane_id()`.
            scalar_offset: Wave-uniform byte offset of `v_gmem_tile`
                relative to the buffer-resource base.
        """
        var v_smem_base = v_smem_slot.ptr
        comptime _bytes_per_thread = 16
        comptime _bytes_per_warp_iter = _bytes_per_thread * WARP_SIZE
        comptime _bytes_per_iter = _bytes_per_thread * Self.num_threads
        comptime _subtile_rows = 8
        comptime _subtile_cols = Self.BK
        comptime _subtile_bytes = (
            _subtile_rows * _subtile_cols * size_of[Self.dtype]()
        )
        comptime _subtiles_per_row = Self.depth // _subtile_cols
        comptime _subtile_row_bytes = _subtile_cols * size_of[Self.dtype]()
        comptime _total_bytes = (Self.BN * Self.depth * size_of[Self.dtype]())
        comptime _num_iters = _total_bytes // _bytes_per_iter

        comptime assert (
            Self.dtype == .bfloat16 or Self.dtype == .float8_e4m3fn
        ), (
            "SubTileLoaderLDS_st_8x32: dtype must be BF16 or FP8 e4m3."
            " The byte-level `buffer_load_lds` DMA is byte-equivalent for"
            " both: 8 rows * BK cols * size_of[dtype] = 512 B per sub-tile"
            " (BF16 BK=32, FP8 BK=64). Callers must pass the dtype-"
            " appropriate BK to keep the sub-tile byte size invariant."
        )
        comptime assert (
            _total_bytes % _bytes_per_iter == 0
        ), "BN*depth*sizeof(dtype) must be a multiple of bytes_per_iter"

        var tile_byte_offset = scalar_offset
        var thread_id = warp_id_uniform * WARP_SIZE + lane_id_local

        # Row stride (in elements) taken from the gmem tile's layout so
        # per-(batch, kv_head) slices of a (B, N, NUM_KV_HEADS, D) tensor
        # advance by `NUM_KV_HEADS * depth` per row, not just `depth`.
        comptime _v_row_stride = type_of(v_gmem_tile).static_stride[0]

        comptime for i in range(_num_iters):
            var lane_byte_offset = (
                thread_id * _bytes_per_thread + Int(i) * _bytes_per_iter
            )
            var subtile_id, subtile_lane_byte_offset = divmod(
                lane_byte_offset, _subtile_bytes
            )
            var subtile_row, subtile_col = divmod(subtile_id, _subtiles_per_row)
            var row_in_subtile, col_byte_in_subtile = divmod(
                subtile_lane_byte_offset, _subtile_row_bytes
            )
            var col_in_subtile = col_byte_in_subtile // size_of[Self.dtype]()
            # st_8x32 BF16 swizzle = identity (no XOR). Global position
            # is just the natural subtile-local position.
            var global_row = subtile_row * _subtile_rows + row_in_subtile
            var global_col = subtile_col * _subtile_cols + col_in_subtile
            var global_byte_in_tile = Int32(
                (global_row * _v_row_stride + global_col)
                * size_of[Self.dtype]()
            )

            var lds_warp_byte = Int32(
                warp_id_uniform * _bytes_per_warp_iter
                + Int(i) * _bytes_per_iter
            )
            # Reference `v227` V adapter WRITE (the `W` of the `W∘R` pair).
            # Reorganizes the cooperative DMA into the reference's chunk-stepped
            # LDS layout so the consumer's `v227` `ds_read_b64_tr_b8` reads the
            # standard PV fragment bank-conflict-free. 16 contiguous 1024-B
            # chunks (== the 16 (warp, iter) bursts the default fill already
            # emits — SAME burst count + granularity, no DMA inflation), at
            # LDS chunk stride 0x410 (the reference's M0 step `s_add m0,
            # {0,0x410,0x820,...}`). Warp w + iter i owns chunk `w*2 + i`;
            # the hardware adds `lane*16` to the M0 we set. Per lane (0..63):
            #   g = lane // 8 ; key = key_base(chunk) + key_off(g)
            #   depth = (lane % 8) * 16
            #   key_base(ci) = (ci1<<1)|(ci0<<2)|(ci2<<3)|(ci3<<6)
            #   key_off(g)   = (g&1) | ((g>>1)&1)<<4 | ((g>>2)&1)<<5
            # The DRAM source is `key*v_row_stride + depth` (our ragged V is
            # key-major, same as the reference's source). The byte permutation
            # `pi: ours_read_addr -> ref_read_addr` is a bijection over the
            # whole slot that leaves the tr8 transpose invariant. FP8
            # 32×32×64, DEPTH=128, KV_BLOCK=128 only (the assert below).
            # Default False → this whole block is comptime-elided (byte-id).
            comptime if Self.v_full_v227:
                comptime assert (
                    Self.dtype.is_float8()
                    and Self.depth == 128
                    and Self.BN == 128
                    and Self.BK == 64
                ), (
                    "SubTileLoaderLDS_st_8x32[v_full_v227]: FP8 32x32x64"
                    " DEPTH=128 KV_BLOCK=128 adapter only"
                )
                # chunk owned by (warp, iter); 8 warps * 2 iters = 16 chunks.
                var _ci = warp_id_uniform * Int(_num_iters) + Int(i)
                comptime if Self._v227_layout:
                    # ----- Layout-Algebra expression of the SAME geometry
                    # (default, `-D v227_layout`; off restores hand) ----
                    # The v227 source byte is BIT-LINEAR over the bit-decomposed
                    # (chunk, lane) index (proven: characterize_layout.py), so
                    # it is `crd2idx` against a per-bit `Layout` whose strides
                    # carry the chunk->key bit-PERMUTATION + the depth/key-off
                    # skews declaratively. Per-bit shapes are 2; strides:
                    #   chunk bit k -> key_base bit-permutation * _v_row_stride
                    #     key_base = (ci1<<1)|(ci0<<2)|(ci2<<3)|(ci3<<6) =>
                    #     chunk bit 0->key bit 2 (stride 4), 1->1 (2),
                    #     2->3 (8), 3->6 (64), times _v_row_stride.
                    #   lane bits 0..2 (= lane%8) -> depth, stride 16,32,64.
                    #   lane bits 3..5 (= g)      -> key_off bits 0,4,5
                    #     (stride 1,16,32) times _v_row_stride.
                    # The bit-DECOMPOSITION (idx2crd over shape (2,...)) is
                    # what stays explicit; the Layout cleanly carries only the
                    # stride ASSIGNMENT (permutation + skews). XOR `Swizzle`
                    # CANNOT realize this (key_base is a non-identity bit
                    # permutation; Swizzle is XOR-only). The `0x410` LDS chunk
                    # stride is a plain Layout STRIDE (not a Swizzle — it is
                    # not a power of two; only Swizzle needs that).
                    # Built as comptime `Coord`s (all
                    # `Idx[...]` = `ComptimeInt`), applied to the runtime
                    # `(chunk, lane)` via the free `crd2idx` (the same op
                    # `Layout.__call__`/`mask_op.idx2crd` use). A `Scalar`
                    # leaf is the runtime-coord form (`Idx[...]` is comptime
                    # only).
                    comptime _SRC_CHUNK_SHAPE = Coord(2, 2, 2, 2)
                    comptime _SRC_CHUNK_STRIDE = Coord(
                        4 * _v_row_stride,
                        2 * _v_row_stride,
                        8 * _v_row_stride,
                        64 * _v_row_stride,
                    )
                    comptime _SRC_LANE_SHAPE = Coord(2, 2, 2, 2, 2, 2)
                    comptime _SRC_LANE_STRIDE = Coord(
                        16,
                        32,
                        64,
                        1 * _v_row_stride,
                        16 * _v_row_stride,
                        32 * _v_row_stride,
                    )
                    comptime _LDS_CHUNK_SHAPE = Coord(Int32(_num_iters) * 8)
                    comptime _LDS_CHUNK_STRIDE = Coord(0x410)
                    # Coordinates applied at the layout's default `out_type`
                    # (int64); the plain form is used (a `uint32` narrow-first
                    # variant, mirroring `_distribute`'s `linear_idx_type`, was
                    # evaluated and rejected). This Layout spelling is the
                    # gated study path — see the `_v227_layout` field doc.
                    global_byte_in_tile = Int32(
                        crd2idx(
                            Int32(_ci),
                            _SRC_CHUNK_SHAPE,
                            _SRC_CHUNK_STRIDE,
                        )
                        + crd2idx(
                            Int32(lane_id_local),
                            _SRC_LANE_SHAPE,
                            _SRC_LANE_STRIDE,
                        )
                    ) * Int32(size_of[Self.dtype]())
                    lds_warp_byte = Int32(
                        crd2idx(
                            Int32(_ci),
                            _LDS_CHUNK_SHAPE,
                            _LDS_CHUNK_STRIDE,
                        )
                    )
                else:
                    # ----- DEFAULT: hand-rolled runtime bit arithmetic --------
                    var _ci0 = _ci & 1
                    var _ci1 = (_ci >> 1) & 1
                    var _ci2 = (_ci >> 2) & 1
                    var _ci3 = (_ci >> 3) & 1
                    var _key_base = (
                        (_ci1 << 1) | (_ci0 << 2) | (_ci2 << 3) | (_ci3 << 6)
                    )
                    var _g = lane_id_local // 8
                    var _key = (
                        _key_base
                        + (_g & 1)
                        + (((_g >> 1) & 1) << 4)
                        + (((_g >> 2) & 1) << 5)
                    )
                    var _depth = (lane_id_local % 8) * 16
                    # DRAM source for this lane (override the st_8x32 position).
                    global_byte_in_tile = Int32(
                        _key * _v_row_stride + _depth
                    ) * Int32(size_of[Self.dtype]())
                    # LDS M0 for the burst = chunk*0x410 bytes (hardware adds
                    # lane*_bytes_per_thread). _bytes_per_thread == 16, matching
                    # the within-chunk lane stride the closed form assumes.
                    lds_warp_byte = Int32(_ci * 0x410)
            var v_smem_warp_ptr = v_smem_base + lds_warp_byte // Int32(
                size_of[Self.dtype]()
            )

            var shared_ptr3 = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<3>`
            ](v_smem_warp_ptr)

            var desc_ptr_ = UnsafePointer[
                BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
            ].unsafe_dangling()
            var ptr_to_ptr = UnsafePointer(to=desc_ptr_)
            var ptr_to_simd = UnsafePointer(to=self.bc.desc)
            ptr_to_ptr[0] = ptr_to_simd.bitcast[
                UnsafePointer[
                    BFloat16, MutAnyOrigin, address_space=.BUFFER_RESOURCE
                ]
            ]()[0]
            var desc_ptr_llvm = __mlir_op.`builtin.unrealized_conversion_cast`[
                _type=__mlir_type.`!llvm.ptr<8>`
            ](desc_ptr_)

            __mlir_op.`rocdl.raw.ptr.buffer.load.lds`[
                alias_scopes=_alias_scope_attr,
                aux=__mlir_attr.`0 : i32`,  # default cache policy
                _type=None,
            ](
                desc_ptr_llvm,
                shared_ptr3,
                to_i32(Int32(_bytes_per_thread)),
                to_i32(Int32(global_byte_in_tile)),
                to_i32(Int32(tile_byte_offset)),
                to_i32(0),
            )


# ===----------------------------------------------------------------------=== #
# RegTileLoader
# ===----------------------------------------------------------------------=== #


struct RegTileLoader[
    dtype: DType,
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    warp_scope: Bool = False,
](TrivialRegisterPassable):
    """AMD buffer-resource load from DRAM to registers.

    Pre-builds the AMDBufferResource from a DRAM TileTensor once.
    Each load() call distributes a source tile across threads and
    issues buffer_load intrinsics to fill a LOCAL register TileTensor.

    The dst register tile uses row-major element ordering (per-thread
    (M, N) fragment stored with strides (N, 1)) so that dst row i is
    m_mma=i's contiguous fragment. `RegTileWriterLDS.copy` reads in the
    same row-major order; the two are paired and must agree.

    Parameters:
        dtype: Element data type.
        thread_layout: Thread distribution layout (e.g. row_major[r, c]()
            or col_major[r, c]()).
        num_threads: Total threads in the block. When the block has more
            threads than thread_layout.size(), extra threads are idled.
            Only needed when the block size differs from the layout size
            (e.g. attention uses a warp-sized layout within a larger block).
            Defaults to thread_layout.size().
        warp_scope: If True, uses lane_id() as worker index (warp scope).
            If False, uses thread_idx.x (block scope).
    """

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM loads."""
    var base_ptr_as_int: Int
    """Integer address of the DRAM tile base pointer.

    Captured at construction so the per-thread base offset in
    `_buffer_load_impl` is computed relative to the buffer resource's
    base, not relative to `src.ptr`. `src` passed to `load()` may be
    a sub-tile of `gmem_tile` (matmul iterates `a_blockrow.tile[BK,
    BM](k, 0)` over k); the offset between the two pointers must
    fold into the per-thread `vector_offset` for buffer_load to
    address the correct rows.
    """

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[Self.dtype, ...],
    ):
        """Creates a loader from a DRAM tile.

        The TileTensor may carry Scalar for any masked dimension
        (e.g. valid_rows in MixedLayout) so that make_amd_buffer_resource
        computes correct OOB clamping bounds.

        Args:
            gmem_tile: The DRAM tile as TileTensor.
        """
        self.bc = make_amd_buffer_resource(gmem_tile)
        self.base_ptr_as_int = Int(gmem_tile.ptr)

    @always_inline
    def __init__(
        out self,
        gmem_tile: TileTensor[Self.dtype, ...],
        *,
        bounds_from: TileTensor[Self.dtype, ...],
    ):
        """Creates a loader with OOB bounds from a full (pre-tiled) tensor.

        TileTensor.tile produces compile-time shapes that are never clipped
        to the actual tensor extent. This overload derives the buffer
        resource clamping range from bounds_from (which carries runtime
        dimensions), so OOB loads return zero for partial edge blocks.

        Args:
            gmem_tile: Block-row tile (provides base pointer for loads).
            bounds_from: Full tensor with runtime dims for OOB bounds.
        """
        from layout._utils import _get_bounds

        var off = (Int(gmem_tile.ptr) - Int(bounds_from.ptr)) // size_of[
            Self.dtype
        ]()
        # A tile based entirely past `bounds_from` makes the remaining extent
        # negative, which `AMDBufferResource` narrows to a UInt32 byte count —
        # wrapping to ~4 GiB and clamping nothing. Saturate so it reads as zero.
        self.bc = AMDBufferResource(
            readfirstlane(gmem_tile.ptr),
            readfirstlane(max(0, _get_bounds(bounds_from) - off)),
        )
        self.base_ptr_as_int = Int(gmem_tile.ptr)

    @always_inline
    def load(
        self,
        dst: TileTensor[mut=True, Self.dtype, _, _, address_space=.LOCAL, ...],
        src: TileTensor[Self.dtype, ...],
    ):
        """Loads DRAM tile data into a LOCAL register tile.

        Distributes src across threads, reads row-major from DRAM,
        stores row-major into dst (matched by `RegTileWriterLDS.copy`).

        Args:
            dst: Destination register TileTensor (LOCAL address space).
            src: Source DRAM TileTensor (vectorized).
        """
        comptime _DistType = type_of(
            src.distribute_with_offset[Self.thread_layout](0)[0]
        )
        comptime M = _DistType.static_shape[0]
        comptime N = _DistType.static_shape[1]

        _buffer_load_impl[
            Self.thread_layout,
            Self.num_threads,
            Self.warp_scope,
        ](
            dst,
            src,
            self.bc,
            self.base_ptr_as_int,
            dst_layout=row_major[M, N](),
        )


# ===----------------------------------------------------------------------=== #
# RegTileWriter
# ===----------------------------------------------------------------------=== #


struct RegTileWriter[
    dtype: DType,
    thread_rows: Int,
    thread_cols: Int,
](TrivialRegisterPassable):
    """AMD buffer-resource store for writing register tiles to DRAM.

    Pre-builds the AMDBufferResource from the full DRAM output tile once.
    Each `store()` call writes a warp sub-tile's worth of register data
    to DRAM via the pre-built descriptor; OOB lanes (past the recorded
    byte bound) are silently dropped by the hardware clamp.

    Pure TileTensor implementation: uses TileTensor distribute_with_offset
    directly (no LayoutTensor conversion). The distribute operation divides
    shape by thread_shape and multiplies strides by thread_shape, producing
    identical offsets to LayoutTensor's zipped_divide for flat 2D layouts.

    A single `store[mfma32: Bool = False]` method handles both:
    - mfma32=False: Generic path using the src tile's own layout indexing
      (any MMA shape).
    - mfma32=True: 32×32 MFMA path with hardware-specific register
      permutation (`src[4*n + 16*m]` → fragment position `4*m + n`).

    See `RegTileWriterLDS.copy` for the matching row-major register reader
    used in DRAM→reg→SMEM pipelines.

    The buffer-resource OOB clamp bounds the store by the destination
    tensor's TOTAL byte extent, not by a per-row column extent, so a
    SIMD chunk that straddles an N boundary (last column block when
    `N % BN != 0`) will spill into the next row of the same buffer
    instead of being clipped. Use `RegTileEpilogue` instead for kernels
    that need to support N-misaligned shapes (or a fused lambda).

    Parameters:
        dtype: Element data type for DRAM destination.
        thread_rows: Number of rows in the col-major thread distribution.
        thread_cols: Number of columns in the col-major thread distribution.
    """

    comptime thread_layout = col_major[Self.thread_rows, Self.thread_cols]()

    var bc: AMDBufferResource
    """The 128-bit buffer resource descriptor for DRAM stores."""
    var base_ptr_as_int: Int
    """Integer address of the full DRAM tile base pointer."""

    @always_inline
    def __init__(out self, dst_base: TileTensor[...]):
        """Create a writer from the full DRAM output tile.

        The TileTensor must carry Scalar for any masked dimension
        (e.g. valid_rows) so that make_amd_buffer_resource computes
        correct OOB clamping bounds.

        Args:
            dst_base: The full DRAM output tile as TileTensor.
        """
        self.bc = make_amd_buffer_resource(dst_base)
        self.base_ptr_as_int = Int(dst_base.ptr)

    @always_inline
    def store[
        mfma32: Bool = False
    ](
        self,
        dst_warp_tile: TileTensor[Self.dtype, ...],
        src_tile: TileTensor[_, _, _, address_space=.LOCAL, ...],
    ):
        """Write register tile data to a DRAM warp sub-tile.

        The distribute + base-offset prologue is identical across MMA
        shapes; only the `(iteration index i, src scalar offset)` pair
        differs:

        - `mfma32=False`: iterate `i in range(dst_shape0 * dst_shape1)`,
          read src at `(i // src_cols) * src_stride0 + (i % src_cols) *
          elem_size` (source's natural row-major layout).
        - `mfma32=True`: iterate `(m, n)` over `(src_shape0, src_shape1 /
          elem_size)`, read src at `4*n + 16*m` (32×32 MFMA register
          permutation) and map to fragment position `i = 4*m + n`.

        Parameters:
            mfma32: Select the 32×32 MFMA register permutation instead of
                the src tile's natural layout.

        Args:
            dst_warp_tile: Vectorized DRAM warp sub-tile.
            src_tile: Register TileTensor with MMA output data.
        """
        comptime elem_size = type_of(dst_warp_tile).element_size

        # Distribute dst among threads and compute the per-lane base
        # offset (in scalar units).
        var dist_result = dst_warp_tile.distribute_with_offset[
            Self.thread_layout
        ](lane_id())
        var dst_dist = dist_result[0]
        var lane_offset = (Int(dst_dist.ptr) - self.base_ptr_as_int) // size_of[
            Self.dtype
        ]()
        var base_offset = Int32(lane_offset)

        comptime dst_shape1 = type_of(dst_dist).static_shape[1]
        comptime dst_stride0 = type_of(dst_dist).static_stride[0]
        comptime dst_stride1 = type_of(dst_dist).static_stride[1]

        comptime if mfma32:
            # 32×32 MFMA hardware register permutation.
            comptime M = type_of(src_tile).static_shape[0]
            comptime N = type_of(src_tile).static_shape[1] // elem_size
            comptime for n in range(N):
                comptime for m in range(M):
                    comptime src_offset = 4 * n + 16 * m
                    comptime i = 4 * m + n
                    comptime dr, dc = divmod(i, dst_shape1)
                    comptime dst_elem_offset = (
                        dr * dst_stride0 + dc * dst_stride1
                    )
                    var data = src_tile.raw_load[width=elem_size](src_offset)
                    self.bc.store(
                        base_offset + Int32(dst_elem_offset),
                        data.cast[Self.dtype](),
                    )
        else:
            # Generic path: src uses its own row-major layout.
            comptime dst_shape0 = type_of(dst_dist).static_shape[0]
            comptime src_stride0 = type_of(src_tile).static_stride[0]
            comptime src_cols = type_of(src_tile).static_shape[1] // elem_size
            comptime for i in range(dst_shape0 * dst_shape1):
                comptime sr, sc = divmod(i, src_cols)
                comptime src_offset = sr * src_stride0 + sc * elem_size
                comptime dr, dc = divmod(i, dst_shape1)
                comptime dst_elem_offset = (dr * dst_stride0 + dc * dst_stride1)
                var data = src_tile.raw_load[width=elem_size](src_offset)
                self.bc.store(
                    base_offset + Int32(dst_elem_offset),
                    data.cast[Self.dtype](),
                )


# ===----------------------------------------------------------------------=== #
# RegTileEpilogue
# ===----------------------------------------------------------------------=== #


struct RegTileEpilogue[
    c_type: DType,
    chunk_width: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
](TrivialRegisterPassable):
    """Per-lane MFMA epilogue writer with optional fused elementwise lambda.

    Encapsulates the per-lane `(m_global, n_global) → store / lambda`
    handoff at the end of an AMD matmul kernel. Each `store()` call
    writes one SIMD chunk of `chunk_width` columns at a single row:
    the natural shape of an AMD MFMA output fragment for one lane.

    Per-lane bound handling:
    - In-bounds chunk (`n + chunk_width <= n_total`): one SIMD store
      or one lambda call.
    - Partial chunk (`n < n_total < n + chunk_width`): per-element
      fallback. The SIMD-of-`chunk_width` store would otherwise spill
      into the next row of the buffer (where stride==N), so we degrade
      to up to `chunk_width` scalar stores or scalar lambda calls,
      each gated on `col < n_total`. This is what makes the writer
      correct for N-misaligned outputs.
    - Fully OOB column (`n >= n_total`): skip silently.

    The caller is responsible for the M bound check before calling
    `store()`: a split-K matmul kernel passes a workspace row that
    differs from the logical output row, so the writer cannot derive
    a single M bound that applies to both DRAM and lambda modes.

    With `elementwise_lambda_fn=None` writes go to DRAM at
    `c_ptr + m * row_stride + n` directly (no buffer-resource clamp;
    the partial-chunk fallback gates by `n_total` explicitly). With a
    lambda set the lambda receives global `(m, n)` and the SIMD chunk;
    DRAM is left untouched. Lambda mode therefore requires the caller
    to pass `m` as the LOGICAL output row, incompatible with a
    per-split workspace write. Kernels that use both split-K and a
    fused lambda should not set the lambda on the per-split matmul
    kernel; instead run a non-fused split-K and apply the lambda in
    the reduce kernel that consumes the partials.

    Parameters:
        c_type: Output element type.
        chunk_width: Number of contiguous columns per lane per call.
            For 16x16x* MFMA on AMD this is `MMA_M * MMA_N // WARP_SIZE
            = 4`. For 32x32x* MFMA the natural per-lane fragment is 16
            elements but they are spread across non-contiguous columns,
            so callers should fan out into per-element calls
            (`chunk_width = 1`) instead.
        elementwise_lambda_fn: Optional fused epilogue.
    """

    var c_ptr_as_int: Int
    """Integer address of the destination's base pointer. Stored as
    Int rather than `Pointer` because the dst tile's origin
    may be any mutable origin and the writer is reused across
    kernels with different origin types."""

    var row_stride: Int
    """Element stride between consecutive rows of dst."""

    var n_total: Int
    """N dimension of the output, used for the chunk-boundary
    detection and the per-element OOB gate."""

    @always_inline
    def __init__(out self, dst: TileTensor[mut=True, Self.c_type, ...]):
        """Build from the (mutable) destination DRAM tile.

        For non-split-K: `dst` is the logical output tensor; `m` in
        subsequent `store()` calls is the logical output row, which
        is also the DRAM row.

        For split-K matmul kernels: `dst` is the
        `(num_splits * M, N)` workspace; `m` in `store()` is the
        workspace row (`split_id * M + pid_m * BM + ...`). Callers
        must keep `elementwise_lambda_fn` unset in that case: see
        the struct doc.

        Args:
            dst: Destination DRAM tile (must be mutable).
        """
        self.c_ptr_as_int = Int(dst.ptr)
        self.row_stride = Int(dst.layout.stride[0]().value())
        self.n_total = Int(dst.dim[1]())

    @always_inline
    def _ptr(self) -> UnsafePointer[Scalar[Self.c_type], MutAnyOrigin]:
        return UnsafePointer[Scalar[Self.c_type], MutAnyOrigin](
            unsafe_from_address=self.c_ptr_as_int
        )

    @always_inline
    def store(
        self,
        v: SIMD[Self.c_type, Self.chunk_width],
        *,
        m: Int,
        n: Int,
    ):
        """Write a SIMD chunk at `(m, n)` of dst.

        The caller has already checked the M bound. If the chunk
        straddles `n_total` (a partial block at the column boundary)
        the writer falls back to per-element stores or lambda calls.

        Args:
            v: SIMD value to write (already cast to `Self.c_type`).
            m: Destination row (DRAM row for split-K workspace, or
                logical output row for non-split-K / reduce-kernel
                lambda mode).
            n: Destination starting column. Caller has typically
                offset by `lane_group * chunk_width` already.
        """
        if n + Self.chunk_width <= self.n_total:
            comptime if Bool(Self.elementwise_lambda_fn):
                comptime epilogue_fn = Self.elementwise_lambda_fn.value()
                epilogue_fn[
                    alignment=align_of[SIMD[Self.c_type, Self.chunk_width]]()
                ](IndexList[2](m, n), v)
            else:
                self._ptr().store[
                    alignment=align_of[SIMD[Self.c_type, Self.chunk_width]]()
                ](m * self.row_stride + n, v)
        elif n < self.n_total:
            for e in range(Self.chunk_width):
                var col = n + e
                if col < self.n_total:
                    comptime if Bool(Self.elementwise_lambda_fn):
                        comptime epilogue_fn = Self.elementwise_lambda_fn.value()
                        epilogue_fn[alignment=align_of[Scalar[Self.c_type]]()](
                            IndexList[2](m, col),
                            SIMD[Self.c_type, 1](v[e]),
                        )
                    else:
                        self._ptr()[m * self.row_stride + col] = v[e]


@always_inline
def _buffer_load_impl[
    thread_layout: Layout,
    num_threads: Int = thread_layout.size(),
    warp_scope: Bool = False,
](
    dst: TileTensor[mut=True, _, _, _, address_space=.LOCAL, ...],
    src: TileTensor[dst.dtype, ...],
    bc: AMDBufferResource,
    base_ptr_as_int: Int,
    dst_layout: Layout,
):
    """Load DRAM data into registers with a caller-specified storage layout.

    Distributes src across threads via thread_layout, reads row-major from
    DRAM (for cache locality), stores into dst using dst_layout strides.
    The dst_layout controls how the M x N per-thread fragment is packed
    into registers; `RegTileLoader` pairs row_major dst with
    `RegTileWriterLDS.copy`'s row-major reads.

    The per-thread base offset is computed via pointer subtraction
    `Int(dist.ptr) - base_ptr_as_int`, which captures BOTH the
    `src.ptr - gmem_tile.ptr` sub-tile offset (when callers pass a sliced
    sub-tile to `load()`) AND the per-thread distribute offset. The
    AMDGPU backend folds this to i32 ops via algebraic simplification
    (both pointers share the same gmem base), so there is no carry-chain
    cost. Recomputing the offset from `dist_tup[2]` alone would drop the
    sub-tile offset and silently produce wrong loads for matmul-style
    callers that iterate over `gmem_tile.tile[...](...)` slices.

    Parameters:
        thread_layout: Thread distribution layout (row_major or col_major).
        num_threads: Total threads; threads beyond layout size are idle.
        warp_scope: If True, uses lane_id() (warp scope).

    Args:
        dst: Destination register tile (LOCAL).
        src: Source DRAM tile (vectorized).
        bc: AMD buffer resource descriptor with OOB bounds.
        base_ptr_as_int: Integer address of the buffer-resource base
            pointer (= the `gmem_tile` passed to `RegTileLoader.__init__`).
            All per-thread offsets are relative to this base, NOT to
            `src.ptr`, which may be a slice with a different pointer.
        dst_layout: Layout controlling register storage order. Shape must
            match the per-thread fragment dimensions (M, N).
    """
    var worker_idx = Int(lane_id()) if warp_scope else Int(thread_idx.x)

    comptime if num_threads > thread_layout.size():
        if worker_idx >= thread_layout.size():
            return

    var dist = src.distribute_with_offset[thread_layout](worker_idx)[0]
    var base_offset = Int32(
        (Int(dist.ptr) - base_ptr_as_int) // size_of[dst.dtype]()
    )

    comptime elem_size = type_of(src).element_size
    comptime M = type_of(dist).static_shape[0]
    comptime N = type_of(dist).static_shape[1]
    comptime src_s0 = type_of(dist).static_stride[0]
    comptime src_s1 = type_of(dist).static_stride[1]
    comptime dst_s0 = dst_layout.static_stride[0]
    comptime dst_s1 = dst_layout.static_stride[1]

    comptime for i in range(M):
        comptime for j in range(N):
            dst.raw_store[width=elem_size](
                (i * dst_s0 + j * dst_s1) * elem_size,
                bc.load[dst.dtype, elem_size](
                    base_offset,
                    scalar_offset=Int32(i * src_s0 + j * src_s1),
                ),
            )


# ===----------------------------------------------------------------------=== #
# RegTileWriterLDS
# ===----------------------------------------------------------------------=== #


struct RegTileWriterLDS[
    thread_layout: Layout,
    swizzle: Optional[Swizzle] = None,
    num_threads: Int = thread_layout.size(),
]:
    """Stateless register→LDS copy expert.

    Sibling to `RegTileLoader` / `RegTileWriter` (DRAM↔reg) and
    `TileLoaderLDS` / `SubTileLoaderLDS` (DRAM→LDS). Writes register
    tiles to shared memory via thread-distributed element stores.

    Parameters:
        thread_layout: Thread distribution layout across the tile.
        swizzle: Optional SMEM swizzle for bank-conflict avoidance.
        num_threads: Number of threads to participate (threads past
            `thread_layout.size()` early-exit).

    Static methods:
        copy         - Standard plain-SMEM write (rank-2 or rank-3
                       distributed layouts); reads src in row-major
                       order to match `RegTileLoader`'s storage.
        copy_blocked - `blocked_product` SMEM write with its own
                       `block_cols` param. Used when thread_layout and
                       SMEM layout have mismatched blocked structure
                       that `distribute_with_offset` can't resolve.
    """

    @staticmethod
    @always_inline
    def copy(
        dst: TileTensor[mut=True, _, _, _, address_space=.SHARED, ...],
        src: TileTensor[_, _, _, address_space=.LOCAL, ...],
    ):
        """Copy register data to SMEM, distributed across threads.

        Reads src registers in row-major element order to match the
        storage convention of `RegTileLoader`. Supports both flat
        (rank 2) and hierarchical (rank 3) distributed layouts.

        Args:
            dst: Destination TileTensor in shared memory.
            src: Source TileTensor in local (register) memory.
        """
        comptime num_busy_threads = Self.thread_layout.size()
        comptime elem_size = type_of(dst).element_size

        var worker_idx = Int(thread_idx.x)

        comptime if Self.num_threads > num_busy_threads:
            if worker_idx >= num_busy_threads:
                return

        var dist_result = dst.distribute_with_offset[
            Self.thread_layout, Self.swizzle
        ](worker_idx)
        var dst_dist = dist_result[0]

        comptime dist_type = type_of(dst_dist)
        comptime DstVec = SIMD[dist_type.dtype, elem_size]
        comptime rank = dist_type.LayoutType.flat_rank

        # Row-major iteration order matches RegTileLoader's storage.
        # Handles both flat (rank 2) and hierarchical Coord layouts (rank 3).
        comptime if rank == 2:
            comptime R0 = dist_type.static_shape[0]
            comptime R1 = dist_type.static_shape[1]
            comptime s0 = dist_type.static_stride[0]
            comptime s1 = dist_type.static_stride[1]
            comptime for i in range(R0):
                comptime for j in range(R1):
                    comptime src_idx = i * R1 + j
                    comptime dst_off = i * s0 + j * s1
                    dst_dist.raw_store[width=elem_size](
                        dst_off,
                        rebind[DstVec](
                            src.raw_load[width=elem_size](src_idx * elem_size)
                        ),
                    )
        elif rank == 3:
            comptime R0 = dist_type.static_shape[0]
            comptime R1 = dist_type.static_shape[1]
            comptime R2 = dist_type.static_shape[2]
            comptime s0 = dist_type.static_stride[0]
            comptime s1 = dist_type.static_stride[1]
            comptime s2 = dist_type.static_stride[2]
            comptime for i in range(R0):
                comptime for j in range(R1):
                    comptime for k in range(R2):
                        comptime src_idx = i * R1 * R2 + j * R2 + k
                        comptime dst_off = i * s0 + j * s1 + k * s2
                        dst_dist.raw_store[width=elem_size](
                            dst_off,
                            rebind[DstVec](
                                src.raw_load[width=elem_size](
                                    src_idx * elem_size
                                )
                            ),
                        )
        else:
            comptime assert (
                False
            ), "RegTileWriterLDS.copy: unsupported flat_rank"

    @staticmethod
    @always_inline
    def copy_blocked[
        block_cols: Int,
    ](dst: SMemTile[mut=True, _, _, _], src: RegTile[dst.dtype, _, _]):
        """Copy register tile to blocked_product SMEM layout.

        Handles structural mismatches between `thread_layout` and SMEM
        layout by computing per-element SMEM offsets using the
        `blocked_product` formula. Reads registers sequentially as
        `simd_width`-wide vectors; this is invariant to col- vs row-major
        flat ordering when each per-thread row equals one SIMD vector.

        The SMEM layout is `blocked_product` with blocks of
        `dst.shape[0] x block_cols`. `thread_layout` distributes a 2D
        grid of `(data_rows, data_cols/simd_width)` vector positions
        across threads.

        Parameters:
            block_cols: Cols per SMEM block in `blocked_product` layout.

        Args:
            dst: Destination `[block_rows, data_cols]` in SHARED.
            src: Source register tile in LOCAL (row-major elements).
        """
        comptime block_rows = type_of(dst).static_shape[0]
        comptime data_cols = type_of(dst).static_shape[1]
        comptime simd_width = simd_width_of[dst.dtype]()

        var worker_idx = Int(thread_idx.x)

        comptime if Self.num_threads > Self.thread_layout.size():
            if worker_idx >= Self.thread_layout.size():
                return

        # Thread grid dimensions (flat rows/cols in the thread grid).
        # Cols are in vector units (each = simd_width elements).
        comptime tgr = Self.thread_layout.static_shape[0] * (
            1 if Self.thread_layout.flat_rank
            == 2 else Self.thread_layout.static_shape[1]
        )
        comptime tgc = (
            Self.thread_layout.static_shape[1] if Self.thread_layout.flat_rank
            == 2 else Self.thread_layout.static_shape[2]
            * Self.thread_layout.static_shape[3]
        )

        # Number of vector positions per thread in each dimension.
        comptime data_vcols = data_cols // simd_width
        comptime vectors_per_thread = (block_rows * data_vcols) // (tgr * tgc)
        comptime cols_per_blk = block_cols // simd_width

        # Distribute thread ID → (row, vcol) using UInt32 bitwise ops.
        # All grid dimensions are power-of-2 so divmod compiles to shift/mask.
        var tid = UInt32(thread_idx.x)
        var base_row, base_vcol = divmod(tid, UInt32(tgc))

        # blocked_product base address: compute within-block super-element
        # index, apply swizzle once, then use compile-time row deltas for
        # subsequent stores.  Inter-row stride bits are above the swizzle
        # range, so swz(base + delta) == swz(base) + delta.
        var blk, col_in_blk = divmod(base_vcol, UInt32(cols_per_blk))
        var local_idx = base_row * UInt32(cols_per_blk) + col_in_blk
        comptime if Self.swizzle:
            comptime swizzle_fn = Self.swizzle.value()
            local_idx = UInt32(swizzle_fn(Int(local_idx)))

        var base_offset = blk * UInt32(
            block_rows * block_cols
        ) + local_idx * UInt32(simd_width)

        # Compile-time vector stride (above swizzle range for typical configs).
        comptime row_delta = tgr * cols_per_blk * simd_width

        comptime for v in range(vectors_per_thread):
            dst.raw_store[width=simd_width](
                Int(base_offset + UInt32(v * row_delta)),
                src.raw_load[width=simd_width](v * simd_width),
            )


# ===----------------------------------------------------------------------=== #
# Tile type aliases
# ===----------------------------------------------------------------------=== #


comptime GMemTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin]
"""Global memory tile. Alias for TileTensor in default (GENERIC) address space."""


comptime SMemTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin, address_space=.SHARED]
"""Shared memory tile. Alias for TileTensor in SHARED address space."""


comptime RegTile[
    mut: Bool,
    //,
    dtype: DType,
    LayoutType: TensorLayout,
    origin: Origin[mut=mut],
] = TileTensor[dtype, LayoutType, origin, address_space=.LOCAL]
"""Register tile. Alias for TileTensor in LOCAL address space."""


# ===----------------------------------------------------------------------=== #
# Stack allocators — thin specializations of layout.tile_tensor's
# stack_allocation with address_space pre-set. The returned origin is
# always MutUntrackedOrigin (the only origin tt_stack_allocation can
# produce), so callers don't need to spell it.
# ===----------------------------------------------------------------------=== #


@always_inline("nodebug")
def reg_alloc[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    alignment: Int = align_of[dtype](),
](var layout: LayoutType) -> RegTile[
    dtype, LayoutType, MutUntrackedOrigin
] where LayoutType.all_dims_known:
    """Stack-allocate a register tile (LOCAL address space) with the given layout.
    """
    return tt_stack_allocation[
        dtype, address_space=.LOCAL, alignment=alignment
    ](layout)


@always_inline("nodebug")
def smem_alloc[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    alignment: Int = align_of[dtype](),
](var layout: LayoutType) -> SMemTile[
    dtype, LayoutType, MutUntrackedOrigin
] where LayoutType.all_dims_known:
    """Stack-allocate a shared memory tile (SHARED address space) with the given layout.
    """
    return tt_stack_allocation[
        dtype, address_space=.SHARED, alignment=alignment
    ](layout)
