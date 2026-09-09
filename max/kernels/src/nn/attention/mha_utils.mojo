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
"""Shared configuration types and dispatch helpers for MHA GPU kernels.

Provides `MHAConfig` (tile/pipeline configuration), `FlashAttentionAlgorithm`
(algorithm variant selector), mask-dispatch helpers, and partition-scheme
types used by both prefill and decode attention kernels.
"""

from std.math import align_up, ceildiv
from std.math.uutils import ufloordiv, ualign_up
from std.collections import OptionalReg
from std.sys import (
    CompilationTarget,
    align_of,
    get_defined_bool,
    get_defined_int,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
    is_amd_gpu,
    is_nvidia_gpu,
    simd_width_of,
    size_of,
)
from std.sys.info import _accelerator_arch

from std.bit import prev_power_of_two
from max.gpu import WARP_SIZE, lane_id
from max.gpu.host import DeviceBuffer
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout.layout_tensor import LayoutTensorIter
from layout.swizzle import make_ldmatrix_swizzle
from nn.attention.mha_mask import (
    CausalMask,
    ChunkedCausalMask,
    ChunkedMask,
    MaskName,
    MaterializedMask,
    MHAMask,
    NullMask,
    RelativeLogitsMask,
    SlidingWindowCausalMask,
    SlidingWindowNonCausalMask,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import min_or_neg_inf
from max.gpu.primitives.grid_controls import PDLLevel

# ===-----------------------------------------------------------------------===#
# Multi-Head Attention
# ===-----------------------------------------------------------------------===#


comptime is_sm90 = "sm_90" in _accelerator_arch()
comptime is_sm100 = "sm_100" in _accelerator_arch() or "sm_103" in _accelerator_arch()
comptime is_sm90or100 = is_sm90 or is_sm100

# Programmatic Dependent Launch level for the split-K decode producer/consumer
# (the split-K attention kernels and `mha_splitk_reduce`). On by default so
# back-to-back grids in the stream overlap launch/prologue latency; disable
# with `-D MHA_PDL=false`. When > OFF, those kernels emit
# `wait_on_dependent_grids()` / `launch_dependent_grids()` and their dispatches
# attach the PROGRAMMATIC_STREAM_SERIALIZATION launch attribute.
comptime MHA_PDL_LEVEL = PDLLevel.OVERLAP_AT_END if get_defined_bool[
    "MHA_PDL", True
]() else PDLLevel.OFF


@always_inline
def as_dynamic_row_major_1d[
    dtype: DType
](
    tensor: LayoutTensor[mut=False, dtype, address_space=.GENERIC, ...],
) -> LayoutTensor[dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]:
    """Reinterprets a generic-address `LayoutTensor` as a 1-D dynamic row-major tensor.

    The pointer and total element count are preserved; the result has an
    unknown-value row-major layout so it can be passed to routines that
    require a 1-D runtime-layout tensor without copying data.

    Parameters:
        dtype: The element data type of the input tensor.

    Args:
        tensor: The immutable generic-address tensor to reinterpret.

    Returns:
        A 1-D `LayoutTensor` with a `row_major(UNKNOWN_VALUE)` layout backed
        by the same storage as `tensor`.
    """
    return {
        tensor.ptr.as_imm().as_unsafe_any_origin(),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            tensor.get_shape()
        ),
    }


struct FlashAttentionAlgorithm(Defaultable, TrivialRegisterPassable, Writable):
    """Identifies which flash-attention algorithm variant to use for a kernel launch.

    The four variants range from a naive reference implementation to the
    latest warp-specialized FA3 pipeline. The default constructed value is
    `FLASH_ATTENTION_3`. Use `init()` to resolve an unspecified (`-1`) value
    to the best algorithm for the target dtype and GPU architecture.
    """

    var _value: Int32

    comptime NAIVE = Self(0)
    comptime FLASH_ATTENTION_1 = Self(1)
    comptime FLASH_ATTENTION_2 = Self(2)
    comptime FLASH_ATTENTION_3 = Self(3)

    def __init__(out self):
        self._value = 3

    def __init__(out self, value: Int):
        self._value = Int32(value)

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    def __eq__(self, version: Int) -> Bool:
        return self._value == Int32(version)

    @always_inline
    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value

    @always_inline
    def init(self, dtype: DType) -> Self:
        if self._value == -1:
            comptime if is_sm90or100:
                return FlashAttentionAlgorithm(
                    2
                    + Int(
                        dtype.is_half_float()
                        or (is_sm100 and dtype.is_float8())
                    )
                )
            else:
                return FlashAttentionAlgorithm(2)
        else:
            return self

    @always_inline
    def write_to(self, mut writer: Some[Writer]):
        if self._value == 0:
            writer.write("naive-attention")
        elif self._value == 1:
            writer.write("flash-attention-1")
        elif self._value == 2:
            writer.write("flash-attention-2")
        elif self._value == 3:
            writer.write("flash-attention-3")
        else:
            writer.write("invalid algorithm")


struct MHAConfig[dtype: DType](TrivialRegisterPassable, Writable):
    """Compile-time and runtime tile-shape configuration for MHA GPU kernels.

    Stores the tile dimensions (BM, BN, BK), warp tile dimensions (WM, WN),
    pipeline depth, and algorithm variant used when launching flash-attention
    kernels. The constructor auto-selects sensible defaults based on `dtype`
    and the detected GPU architecture when optional fields are left as `None`.

    Parameters:
        dtype: The element data type shared by Q, K, V, and the output tensor.
    """

    # Q, K, V, output should have the same type.
    var num_heads: Int
    var depth: Int
    var padded_depth: Int
    var num_queries_per_block: Int
    var num_keys_per_block: Int
    var BK: Int  # tile size in depth dimension
    var WM: Int
    var WN: Int
    var num_pipeline_stages: Int
    var k_group_size: Int

    var algorithm: FlashAttentionAlgorithm
    var swizzle_mode: TensorMapSwizzle

    def block_m(self) -> Int:
        return self.num_queries_per_block

    def block_n(self) -> Int:
        return self.num_keys_per_block

    def block_k(self) -> Int:
        return self.BK

    def warp_m(self) -> Int:
        return self.WM

    def warp_n(self) -> Int:
        return self.WN

    def num_warps_m(self) -> Int:
        return ufloordiv(self.block_m(), self.warp_m())

    def num_warps_n(self) -> Int:
        return ufloordiv(self.block_n(), self.warp_n())

    def num_consumer_threads(self) -> Int:
        return self.num_warps_m() * self.num_warps_n() * WARP_SIZE

    def num_producer_threads[
        producer_consumer_kernel: Bool = False
    ](self) -> Int:
        return 128 if (producer_consumer_kernel and self.algorithm == 3) else 0

    def num_threads[producer_consumer_kernel: Bool = False](self) -> Int:
        return (
            self.num_consumer_threads()
            + self.num_producer_threads[producer_consumer_kernel]()
        )

    def swizzle_granularity(self) -> Int:
        return ufloordiv(self.swizzle_mode.bytes(), size_of[self.dtype]())

    def q_smem_size(self, fa3: Bool = False, persistent: Bool = False) -> Int:
        var q_size = self.block_m() * self.padded_depth
        var num_q = 2 if fa3 and persistent else 1
        return num_q * q_size

    def kv_smem_size(self, fa3: Bool = False) -> Int:
        if fa3:
            return self.num_pipeline_stages * self.block_n() * self.padded_depth
        else:
            return self.block_n() * self.padded_depth

    def k_smem_size(self, fa3: Bool = False) -> Int:
        if fa3:
            return self.kv_smem_size(True)
        else:
            return self.block_n() * self.padded_depth

    def v_smem_size(self, fa3: Bool = False) -> Int:
        if fa3:
            return self.kv_smem_size(True)
        else:
            var BN = self.block_n()
            return BN * BN

    def p_smem_size(self) -> Int:
        return self.block_m() * self.block_n()

    def warp_scratch_smem_size(self) -> Int:
        var n_warps_n = self.num_warps_n()
        return 2 * n_warps_n * self.block_m() if n_warps_n > 1 else 0

    def shared_mem_bytes[
        shared_kv: Bool = False, sm_90: Bool = False
    ](self) -> Int:
        if not has_nvidia_gpu_accelerator():
            return 0

        comptime persistent = (
            get_defined_int["USE_EXPERIMENTAL_KERNELS", 0]() != 0
        ) and sm_90
        var sm_90_fa3 = sm_90 and (self.algorithm == 3)

        var num_smem_elements: Int
        comptime if shared_kv:
            num_smem_elements = (
                self.q_smem_size(sm_90_fa3, persistent)
                + self.kv_smem_size(sm_90_fa3)
                + self.warp_scratch_smem_size()
            )
        else:
            num_smem_elements = (
                self.q_smem_size(sm_90_fa3, persistent)
                + self.k_smem_size(sm_90_fa3)
                + self.v_smem_size(sm_90_fa3)
                + self.warp_scratch_smem_size()
            )

        if self.num_warps_n() > 1 or has_amd_gpu_accelerator():
            num_smem_elements += self.p_smem_size()

        var num_smem_bytes = size_of[self.dtype]() * num_smem_elements
        if sm_90_fa3:
            comptime i64_size = size_of[DType.int64]()
            num_smem_bytes += (2 * self.num_pipeline_stages) * i64_size + (
                4 * i64_size + 2 * size_of[DType.uint32]() if persistent else 0
            )
        return num_smem_bytes

    def __init__(
        out self,
        num_heads: Int,
        depth: Int,
        num_queries_per_block: Optional[Int] = None,
        num_keys_per_block: Optional[Int] = None,
        BK: Optional[Int] = None,
        WM: Optional[Int] = None,
        WN: Optional[Int] = None,
        num_pipeline_stages: Int = 4,
        k_group_size: Int = 1,
        algorithm: FlashAttentionAlgorithm = FlashAttentionAlgorithm(-1),
        swizzle_mode: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    ):
        self.num_heads = num_heads
        self.depth = depth
        var swizzle_granularity = (
            swizzle_mode.bytes() // size_of[DType.bfloat16]()
        )
        self.padded_depth = ualign_up(depth, swizzle_granularity)
        self.num_pipeline_stages = num_pipeline_stages
        self.k_group_size = k_group_size
        self.algorithm = algorithm.init(Self.dtype)
        self.swizzle_mode = swizzle_mode
        # Not all of these have to be `OptionalReg`, only
        # those that depend on `depth`.
        # Currently, all are `OptionalReg` for consistency.
        if (
            is_sm90or100
            and (
                Self.dtype.is_half_float()
                or (is_sm100 and Self.dtype.is_float8())
            )
            and self.algorithm == FlashAttentionAlgorithm(3)
        ):
            # BM
            self.num_queries_per_block = num_queries_per_block.or_else(128)
            var reg_per = 224 if self.num_queries_per_block > 64 else 256
            if num_keys_per_block:
                self.num_keys_per_block = num_keys_per_block.value()
            # FIXME: for depth == 64, larger num_keys_per_block values currently
            #        trigger correctness issues; this hardcoded value is a
            #        temporary workaround and should be revisited.
            elif depth == 64:
                self.num_keys_per_block = 64
            else:
                # BN
                # reg use per warp is at least
                # 16*BN//32 + 16*depth//32 + 16*BN//64 + 4
                # reg_per >= 16*BN//32 + 16*depth//32 + 16*BN//64 + 4
                # (reg_per - depth//2 - 4) >= 3*BN//4
                # BN <= (4*reg_per - 2*depth - 16)//3
                var reg_upper_bound = (4 * reg_per - 2 * depth - 16) // 3
                comptime persistent: Bool = (
                    get_defined_int["USE_EXPERIMENTAL_KERNELS", 0]() != 0
                )
                var smem_total = 227000
                # smem_total >= 2*(BN * depth * pipeline_stages + BM*depth*(1+persistent))
                #                 + 16*pipeline_stages + 40*persistent
                # smem_total - 2*BM*depth*(1+persistent) - 16*pipeline_stages - 40*persistent
                #        >= 2*depth*pipeline_stages*BN
                # BN <= (smem_total//2 - BM*depth*(1+persistent) - 8*pipeline_stages
                #        - 20*persistent) // (depth*pipeline_stages)
                var smem_upper_bound = (
                    smem_total // 2
                    - (
                        self.num_queries_per_block
                        * depth
                        * (1 + Int(persistent))
                    )
                    - 8 * num_pipeline_stages
                    - 20 * Int(persistent)
                ) // (depth * num_pipeline_stages)
                # divide and multiply by 16 to get a multiple of MMA_K
                var min_upper_bound = 16 * (
                    min(reg_upper_bound, smem_upper_bound) // 16
                )
                # FIXME: add support for non-power-of-twos?
                self.num_keys_per_block = max(
                    prev_power_of_two(min_upper_bound), 64
                )
            self.BK = BK.or_else(64)
            self.WN = WN.or_else(min(self.num_keys_per_block, 256))
        else:
            # BN
            self.num_keys_per_block = num_keys_per_block.or_else(
                (
                    32 if depth == 512 else 64
                ) if has_amd_gpu_accelerator() else depth
            )
            # BM
            self.num_queries_per_block = num_queries_per_block.or_else(
                32 if Self.dtype
                == .float32 else (128 if has_amd_gpu_accelerator() else 64)
            )
            var bk_arch_factor = 2 if num_pipeline_stages <= 2 else 1
            var bk_type_factor = 1 if Self.dtype == DType.float32 else 2
            self.BK = BK.or_else(
                16 * bk_arch_factor * bk_type_factor
            ) if has_nvidia_gpu_accelerator() else BK.or_else(
                64 if Self.dtype.is_float8() else 32
            )
            self.WN = WN.or_else(
                32 if Self.dtype == .float32 else self.num_keys_per_block
            )
        self.WM = WM.or_else(
            32 if Self.dtype
            == .float32 else (32 if has_amd_gpu_accelerator() else 16)
        )

    def write_to(self, mut writer: Some[Writer]):
        if self.algorithm == 2:
            writer.write("ampere_")
        else:
            writer.write("fa3_")
        writer.write(self.dtype, "_")
        # Use BNxBM to match MatmulConfig, which matches cublas
        writer.write(self.block_n(), "x", self.block_m(), "_")
        writer.write(self.block_k(), "x")
        writer.write(self.num_pipeline_stages)
        writer.write(",depth = ", self.depth)
        writer.write(",padded_depth = ", self.padded_depth)
        writer.write(",num_attention_heads = ", self.num_heads)


@always_inline
def indexer_key_bound[
    kpool: Int = 1
](num_keys: Int, seq_len: Int, tok_local: Int, causal: Int) -> Int:
    """Candidate rows the sparse indexer defines for token `tok_local` of a row.

    `num_keys` is the row's total key count (`cache_len + seq_len`); the
    result is all of them without a causal mask, `cache_len + tok_local + 1`
    with one. Branchless multiply form: a branch in the scorer
    epilogues' unrolled token loop measured +4-9% on the non-causal path from
    codegen alone.

    With `kpool > 1` the cache holds one pooled key per `kpool` consecutive
    tokens, and the result counts *pools* instead of tokens. A pool covering
    tokens `[kpool*p, kpool*p + kpool)` is a candidate only once its last
    member is visible, which is exactly `visible // kpool` pools -- so the
    pooled bound is the token bound floored by `kpool`, and no separate
    validity rule is needed. A non-positive token bound stays non-positive,
    which both call sites already treat as "no rows".

    Read side of the indexer's write/read contract: the SM100 scorers
    (`sparse_index_fp8_sm100[_prefill].mojo`) write score slots `[0, bound)`
    for each token and nothing else, computing this same bound inline in
    their store guards (their operands are `Int32`), and the bounded top-k
    (`topk_row_bounds_kernel` in `mla_index_fp8.mojo` feeding
    `persistent_topk_block_split`) reads exactly that range with no `-inf`
    prefill between them. If either side drifts, the top-k reads score slots
    the scorer never wrote.
    """
    comptime assert kpool >= 1, "kpool must be positive"
    return (num_keys - (seq_len - 1 - tok_local) * causal) // kpool


@always_inline
def _kernel_mask[
    dtype: DType, width: SIMDLength
](
    coord: IndexList[2, ...], bound: IndexList[2, ...], vec: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    var masked_vec = SIMD[dtype, width]()

    # TODO: use `select` to see if it generates the same code.
    comptime for i in range(width):
        masked_vec[i] = (
            vec[i] if coord[0] < bound[0]
            and UInt32(coord[1]) + UInt32(i)
            < UInt32(bound[1]) else min_or_neg_inf[dtype]()
        )

    return masked_vec


@always_inline
def _copy_frag_to_smem_nvidia[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    frag_simd_width: Int,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        mut=True, type0, layout0, address_space=.SHARED, ...
    ],
    p_reg_tile: LayoutTensor[type1, layout1, _, address_space=.LOCAL],
    warp_x: UInt32,
    warp_y: UInt32,
):
    """Copy p fragments to shared memory.

    Logically P has shape BM x BN. It's sharded across threads in 16x8 mma layout.
    The BM x BN matrix is divided to BM x BK tiles, each tile is CONTIGUOUS for
    the 2nd mma. This function maps each fragment to the right BM x BK tile and
    swizzle to avoid bank conflict.
    """

    comptime simd_width = simd_width_of[p_smem_tile.dtype]()
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    # This tile is used for offset computation because 1st mma output is organized
    # for BM x BN output tile. The layout for 2nd mma is in p_smem_iter.
    # Use ImmutAnyOrigin so distance() call below does not see aliased writable args.
    var p_smem_tile = LayoutTensor[
        mut=False,
        p_smem_iter.dtype,
        Layout.row_major(BM, BN),
        address_space=.SHARED,
    ](p_smem_iter.ptr)
    var p_smem_warp_tile = p_smem_tile.tile[WM, WN](Int(warp_y), Int(warp_x))
    var p_reg_vecs = p_reg_tile.vectorize[1, frag_simd_width]()

    comptime swizzle_fn = make_ldmatrix_swizzle[p_smem_tile.dtype, BK]()

    comptime for n_mma in range(num_n_mmas):
        comptime for m_mma in range(num_m_mmas):
            var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                m_mma, n_mma
            ).vectorize[1, frag_simd_width]()
            var p_smem_frag = p_smem_mma_tile.distribute[
                Layout.row_major(8, 4)
            ](lane_id())
            var frag_offset = p_smem_frag.distance(p_smem_tile)

            comptime for i in range(p_reg_vecs.shape[1]()):
                comptime offset_in_frag = type_of(p_smem_frag).layout(i)

                # Translate offset in BM x BN matrix to the right BM x BK tile.
                comptime OffsetType = type_of(frag_offset)
                var offset_BMxBN = frag_offset + type_of(frag_offset)(
                    offset_in_frag
                )
                var offset_BMxBK = (
                    offset_BMxBN // OffsetType(BN)
                ) * OffsetType(BK) + offset_BMxBN % OffsetType(BK)
                # Convert offset to vectorized domain, since BM x BK will be loaded
                # by vectors in 2nd mma, and swizzle
                var swizzle_offset = swizzle_fn(
                    offset_BMxBK // OffsetType(simd_width)
                )
                # Convert offset back to where the frag will be stored.
                offset_BMxBK = swizzle_offset * OffsetType(
                    simd_width
                ) + offset_BMxBK % OffsetType(simd_width)
                # E.g. fp32x2 -> bf16x2 for bf16 mma.
                var vec = p_reg_vecs[n_mma * num_m_mmas + m_mma, i].cast[
                    p_smem_tile.dtype
                ]()
                # Grep the right BMxBK tile and store the casted vec.
                var tile_BMxBK = p_smem_iter.next_unsafe(
                    p_smem_iter.linear_uint_type(
                        Int((offset_BMxBN % OffsetType(BN)) // OffsetType(BK))
                    )
                )[]
                comptime align = align_of[
                    SIMD[p_smem_iter.dtype, frag_simd_width]
                ]()
                tile_BMxBK.ptr.store[alignment=align](offset_BMxBK, vec)


@always_inline
def _copy_frag_to_smem_amd[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    frag_simd_width: Int,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        mut=True, type0, layout0, address_space=.SHARED, ...
    ],
    p_reg_tile: LayoutTensor[type1, layout1, _, address_space=.LOCAL],
    warp_x: UInt32,
    warp_y: UInt32,
):
    """Copy p fragments to shared memory.
    Logically P has shape BM x BN. It's sharded across threads in 16x16 mma layout.
    The BM x BN matrix is divided to BM x BK tiles, each tile is CONTIGUOUS for
    the 2nd mma. This function maps each fragment to the right BM x BK tile.
    """
    comptime simd_width = 1
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    # This tile is used for offset computation because 1st mma output is organized
    # for BM x BN output tile. The layout for 2nd mma is in p_smem_iter.
    # Use ImmutAnyOrigin so distance() call below does not see aliased writable args.
    var p_smem_tile = LayoutTensor[
        mut=False,
        p_smem_iter.dtype,
        Layout.row_major(BM, BN),
        address_space=.SHARED,
    ](p_smem_iter.ptr)

    var p_smem_warp_tile = p_smem_tile.tile[WM, WN](Int(warp_y), Int(warp_x))
    var p_reg_vecs = p_reg_tile.vectorize[1, frag_simd_width]()

    comptime for n_mma in range(num_n_mmas):
        comptime for m_mma in range(num_m_mmas):
            var p_smem_mma_tile = p_smem_warp_tile.tile[MMA_M, MMA_N](
                m_mma, n_mma
            ).vectorize[frag_simd_width, 1]()
            var p_smem_frag = p_smem_mma_tile.distribute[
                Layout.row_major(4, 16)
            ](lane_id())
            var frag_offset = p_smem_frag.distance(p_smem_tile)

            comptime for i in range(frag_simd_width):
                comptime offset_in_frag = BN * i
                # Translate offset in BM x BN matrix to the right BM x BK tile.
                comptime OffsetType = type_of(frag_offset)
                var offset_BMxBN = frag_offset + OffsetType(offset_in_frag)
                var offset_BMxBK = (
                    offset_BMxBN // OffsetType(BN)
                ) * OffsetType(BK) + offset_BMxBN % OffsetType(BK)

                var vec = p_reg_vecs[n_mma * num_m_mmas + m_mma, 0][i].cast[
                    p_smem_tile.dtype
                ]()
                # Grep the right BMxBK tile and store the casted vec.
                var tile_BMxBK = p_smem_iter.next_unsafe(
                    p_smem_iter.linear_uint_type(
                        Int((offset_BMxBN % OffsetType(BN)) // OffsetType(BK))
                    )
                )[]
                tile_BMxBK.ptr.store(offset_BMxBK, vec)


@always_inline
def _copy_frag_to_smem[
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    MMA_M: Int,
    MMA_N: Int,
    frag_simd_width: Int,
    *,
    type0: DType,
    layout0: Layout,
    type1: DType,
    layout1: Layout,
](
    p_smem_iter: LayoutTensorIter[
        mut=True, type0, layout0, address_space=.SHARED, ...
    ],
    p_reg_tile: LayoutTensor[type1, layout1, _, address_space=.LOCAL],
    warp_x: UInt32,
    warp_y: UInt32,
):
    comptime if is_nvidia_gpu():
        _copy_frag_to_smem_nvidia[
            BM, BN, BK, WM, WN, MMA_M, MMA_N, frag_simd_width
        ](p_smem_iter, p_reg_tile, warp_x, warp_y)
    elif is_amd_gpu():
        _copy_frag_to_smem_amd[
            BM, BN, BK, WM, WN, MMA_M, MMA_N, frag_simd_width
        ](p_smem_iter, p_reg_tile, warp_x, warp_y)
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def get_start_and_end_for_partitions[
    tile_size: Int
](num_keys: Int, num_partitions: Int, partition_idx: Int) -> Tuple[Int, Int]:
    """Calculate start and end indices for a partition.

    Non-empty partitions are packed at low indices `0..N-1` with
    `partition_size = max(tile_size, align_up(ceildiv(num_keys, num_partitions),
    tile_size))`; partitions `>= N` are empty (start == end == num_keys).

    Parameters:
        tile_size: Alignment granularity, in elements, for partition
            boundaries. Each non-empty partition spans a multiple of
            `tile_size` keys so that tile-aligned loads cover whole tiles.

    Args:
        num_keys: Total number of keys (sequence length).
        num_partitions: Number of partitions to split keys into.
        partition_idx: Index of current partition (0 to num_partitions-1).

    Returns:
        Tuple of (start_idx, end_idx) for the partition, aligned to tile_size.
    """
    var num_keys_per_partition = ceildiv(num_keys, num_partitions)
    var partition_size = max(
        tile_size, align_up(num_keys_per_partition, tile_size)
    )
    var start = partition_idx * partition_size
    if start >= num_keys:
        return (num_keys, num_keys)
    var end = min(num_keys, start + partition_size)
    return (start, end)


comptime callback_fn_type = def[mask_t: MHAMask](mask: mask_t) raises -> None


@always_inline
def dispatch_mask[
    mask_type: String,
    local_window_size: Int = -1,
](callback_fn: Some[callback_fn_type],) raises -> None:
    """Instantiate an `MHAMask` by name and invoke a callback with it.

    Resolves the mask string to one of the built-in `MHAMask` implementations
    at compile time and calls `callback_fn` with a concrete mask instance.
    This lets callers write mask-agnostic kernels while still specialising the
    generated code per mask type.

    Parameters:
        mask_type: Name of the mask (e.g. `"causal"`, `"null"`,
            `"sliding_window_causal"`). Must match one of the `MaskName`
            constants.
        local_window_size: Sliding-window or chunk size for masks that
            require it. Must be `-1` for masks that ignore it and positive
            for masks that require it.

    Args:
        callback_fn: Parametric callback invoked with the resolved mask.

    Raises:
        Compile-time assertion error if `mask_type` is unrecognised or if
        `local_window_size` is inconsistent with the selected mask.
    """

    @always_inline
    def outer_wrapper[mask_t: MHAMask](mask: mask_t) raises {imm}:
        return callback_fn(mask)

    # TODO: attach string constants to mask types themselves.
    comptime if MaskName.CAUSAL == mask_type:
        return outer_wrapper(CausalMask())
    elif MaskName.CHUNKED == mask_type:
        comptime assert (
            local_window_size > 0
        ), "You must specify local_window_size for ChunkedMask"
        return outer_wrapper(ChunkedMask[local_window_size]())
    elif MaskName.NULL == mask_type:
        return outer_wrapper(NullMask())
    elif MaskName.SLIDING_WINDOW_CAUSAL == mask_type:
        comptime assert (
            local_window_size > 0
        ), "You must specify local_window_size for SlidingWindowCausalMask"
        return outer_wrapper(SlidingWindowCausalMask[local_window_size]())
    elif MaskName.SLIDING_WINDOW_NONCAUSAL == mask_type:
        comptime assert (
            local_window_size > 0
        ), "You must specify local_window_size for SlidingWindowNonCausalMask"
        return outer_wrapper(SlidingWindowNonCausalMask[local_window_size]())
    elif MaskName.CHUNKED_CAUSAL == mask_type:
        comptime assert (
            local_window_size > 0
        ), "You must specify local_window_size for ChunkedCausalMask"
        return outer_wrapper(ChunkedCausalMask[local_window_size]())
    else:
        comptime assert False, "Unsupported mask type: " + mask_type


@always_inline
def dispatch_materialized_mask[
    dtype: DType,
    layout: Layout,
    //,
](
    mask_nd: LayoutTensor[mut=False, dtype, layout, _],
    callback_fn: Some[callback_fn_type],
    start_pos_nd: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
) raises -> None:
    """Wrap a dense mask tensor in a `MaterializedMask` and invoke a callback.

    Constructs a `MaterializedMask` from the provided tensor and optional
    per-sequence start-position tensor, then calls `callback_fn` with the
    resulting mask. Use this when the mask is provided as an explicit tensor
    (e.g. an ALiBi or relative-positional-encoding bias) rather than a
    compute-on-the-fly strategy.

    Parameters:
        dtype: Element type of the mask tensor.
        layout: Layout of the mask tensor.

    Args:
        mask_nd: The mask values tensor with shape `(batch, heads, q, k)` or
            compatible broadcast shape.
        callback_fn: Parametric callback invoked with the `MaterializedMask`.
        start_pos_nd: Optional per-sequence start positions used to offset the
            key dimension.
    """

    var mask = MaterializedMask(mask_nd, start_pos_nd)
    return callback_fn(mask)


@always_inline
def dispatch_relative_logits_mask[
    dtype: DType,
    layout: Layout,
    //,
    local_window_size: Int = -1,
](
    bias_nd: LayoutTensor[mut=False, dtype, layout, _],
    cache_lengths: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    input_row_offsets: LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
    ],
    callback_fn: Some[callback_fn_type],
) raises -> None:
    """Wrap `bias_nd` in a `RelativeLogitsMask` and invoke `callback_fn`.

    Like `dispatch_materialized_mask`, this carries runtime state (the bias
    table plus the tensors that recover its ragged-flat row), so it lives
    outside `dispatch_mask`'s zero-arg string dispatch. `local_window_size`
    picks the visibility mask: `<= 0` (canonically `-1`, the graph-level
    "no window" value) -> `CausalMask`, else
    `SlidingWindowCausalMask[local_window_size]`. The mask structs
    themselves report "no window" as `sliding_window_size() == 0`; this
    function is where the two conventions meet.
    """
    comptime if local_window_size <= 0:
        return callback_fn(
            RelativeLogitsMask[visibility=CausalMask()](
                bias_nd, cache_lengths, input_row_offsets
            )
        )
    else:
        return callback_fn(
            RelativeLogitsMask[
                visibility=SlidingWindowCausalMask[local_window_size]()
            ](bias_nd, cache_lengths, input_row_offsets)
        )


# The motivation here is to be able to pass `StaticInt[1]()`
# to indicate `decoding=True`, and have this not generate any code
# when passing as a function argument.
# That is, we want different specializations of a function to have
# different numbers of arguments post-compilation.
trait OptionallyStaticInt(Copyable, Intable, TrivialRegisterPassable):
    """Trait for integer values that may be statically known at compile time.

    Implementors carry a `comptime static_value` that is `Some` when the
    integer is a compile-time constant and `None` when it is a runtime value.
    This lets callers specialise code at compile time (eliminating kernel
    arguments) while retaining a uniform runtime interface through `__int__`
    and `as_uint32()`.
    """

    comptime static_value: Optional[Int]

    def as_uint32(self) -> UInt32:
        ...


struct StaticInt[value: Int](
    Defaultable, OptionallyStaticInt, TrivialRegisterPassable
):
    """A compile-time constant integer that satisfies `OptionallyStaticInt`.

    Because the value is fully known at compile time, no runtime storage is
    needed and no kernel argument is generated for this type. Use
    `StaticInt[1]()` to represent the decoding mode without passing an
    extra argument to GPU kernels.

    Parameters:
        value: The compile-time integer value.
    """

    comptime static_value: Optional[Int] = Optional[Int](Self.value)

    @always_inline("nodebug")
    def __init__(out self):
        pass

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return Self.value

    @always_inline("nodebug")
    def as_uint32(self) -> UInt32:
        return UInt32(Self.value)


struct DynamicInt(OptionallyStaticInt, TrivialRegisterPassable):
    """A runtime integer value that satisfies `OptionallyStaticInt`.

    Unlike `StaticInt`, the value is not known until runtime and is stored
    in a `UInt32` field. Use this when the integer must vary between
    invocations, such as a dynamic sequence length or partition index.
    """

    var value: UInt32
    comptime static_value: Optional[Int] = None

    @always_inline("nodebug")
    def __init__(out self, value: Int):
        self.value = UInt32(value)

    @always_inline("nodebug")
    def __int__(self) -> Int:
        return Int(self.value)

    @always_inline("nodebug")
    def as_uint32(self) -> UInt32:
        return self.value


@always_inline
def _is_decoding[int_t: OptionallyStaticInt]() -> Bool:
    return int_t.static_value.or_else(0) == 1


trait OptionalPointer(Copyable, TrivialRegisterPassable):
    """Abstracts over nullable pointers, providing a uniform interface for `NonNullPointer` and `NullPointer`.
    """

    comptime dtype: DType
    comptime is_null: Bool
    comptime address_space: AddressSpace

    @always_inline
    def value(
        self,
    ) -> UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]:
        ...


struct NonNullPointer[dtype_: DType, address_space_: AddressSpace = .GENERIC](
    OptionalPointer
):
    """A pointer with a compile-time guarantee of being non-null.

    Parameters:
        dtype_: Element type of the pointed-to values.
        address_space_: GPU address space of the pointer (defaults to
            `AddressSpace.GENERIC`).
    """

    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = False
    comptime address_space: AddressSpace = Self.address_space_
    comptime PtrType = UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]

    @__allow_legacy_any_origin_fields
    var ptr: Self.PtrType

    @always_inline
    def __init__(out self, ptr: Self.PtrType):
        self.ptr = ptr

    @always_inline
    def __init__(out self, ptr: DeviceBuffer[Self.dtype]):
        comptime assert Self.address_space == .GENERIC
        self.ptr = rebind[Self.PtrType](ptr.unsafe_ptr())

    @always_inline
    def value(self) -> Self.PtrType:
        assert Int(self.ptr) != 0, (
            "NonNullPointer is supposed to provide a compile-time guarantee"
            " of being non-null"
        )
        return self.ptr


struct NullPointer[dtype_: DType, address_space_: AddressSpace = .GENERIC](
    OptionalPointer
):
    """A pointer known at compile time to be null, used when an optional pointer argument is absent.

    Parameters:
        dtype_: Element type of the pointed-to values.
        address_space_: GPU address space of the pointer (defaults to
            `AddressSpace.GENERIC`).
    """

    comptime dtype: DType = Self.dtype_
    comptime is_null: Bool = True
    comptime address_space: AddressSpace = Self.address_space_
    comptime PtrType = UnsafePointer[
        Scalar[Self.dtype], ImmutAnyOrigin, address_space=Self.address_space
    ]

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def value(self) -> Self.PtrType:
        # NullPointer.value() should never be called at runtime — it exists
        # only for trait conformance. Return dangling as a safe sentinel.
        return Self.PtrType.unsafe_dangling()


trait MHAPartitionScheme(Copyable, TrivialRegisterPassable):
    """Trait describing how the key-value sequence is partitioned for split-K decoding.

    Implementations either skip partitioning entirely (`NoPartition`) or
    divide the key sequence across multiple CTAs and accumulate partial
    softmax statistics in a separate reduction pass (`SplitKPartition`).
    The `do_partition` compile-time flag lets the compiler eliminate the
    reduction kernel when no partitioning is needed.
    """

    comptime do_partition: Bool
    comptime accum_dtype: DType
    # Null exactly when `do_partition` is False, so a scheme that owns no
    # partial-statistics buffer cannot hand out a pointer to one.
    comptime LSEPointerType: OptionalPointer

    @always_inline
    def num_partitions(self) -> UInt32:
        ...

    # The number of partition CTAs the decode grid is launched with. This is an
    # upper bound on num_partitions() that is independent of num_keys, so the
    # launched grid shape is stable across num_keys (one CUDA graph per batch
    # size). CTAs with partition index >= num_partitions() early-return. Equal
    # to num_partitions() when the scheme does not over-launch.
    @always_inline
    def max_num_partitions(self) -> UInt32:
        ...

    # Base of the partial softmax statistics buffer. Callers that need to write
    # through it must first establish `do_partition` (see
    # `MHAPosition.exp_sum_qk_max_ptr`), which is what makes the mutability
    # laundering at those sites sound.
    @always_inline
    def lse_pointer(self) -> Self.LSEPointerType:
        ...


struct NoPartition[dtype: DType](
    Defaultable, MHAPartitionScheme, TrivialRegisterPassable
):
    """A single-partition (non-split-K) scheme for MHA decoding.

    Uses the standard single-pass flash-attention decode without any
    inter-CTA reduction. `do_partition` is `False`, so the split-K
    reduction kernel is compiled away entirely.

    Parameters:
        dtype: The accumulator element type (same as the output type).
    """

    comptime do_partition: Bool = False
    comptime accum_dtype: DType = Self.dtype
    comptime LSEPointerType = NullPointer[Self.accum_dtype]

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def num_partitions(self) -> UInt32:
        return 1

    @always_inline
    def max_num_partitions(self) -> UInt32:
        return 1

    @always_inline
    def lse_pointer(self) -> Self.LSEPointerType:
        return {}


struct SplitKPartition[dtype: DType](
    MHAPartitionScheme, TrivialRegisterPassable
):
    """A multi-partition split-K scheme for MHA decoding over long sequences.

    Divides the key sequence across `num_partitions` CTAs. Each CTA writes
    its partial softmax numerator/denominator to the buffer pointed to by
    `ptr`, and a separate reduction kernel merges the results. Over-launches
    up to `max_num_partitions` CTAs so the grid shape is stable across
    varying key lengths (enabling CUDA graph capture).

    Parameters:
        dtype: The accumulator element type used for the partial statistics
            buffer and the final output.
    """

    comptime do_partition: Bool = True
    comptime accum_dtype: DType = Self.dtype
    comptime LSEPointerType = NonNullPointer[Self.accum_dtype]

    @__allow_legacy_any_origin_fields
    var ptr: UnsafePointer[Scalar[Self.accum_dtype], MutAnyOrigin]
    var num_partitions_value: UInt32
    var max_num_partitions_value: UInt32

    @always_inline
    def __init__(
        out self,
        ptr: UnsafePointer[Scalar[Self.accum_dtype], MutAnyOrigin],
        num_partitions_value: UInt32,
        max_num_partitions_value: UInt32,
    ):
        self.ptr = ptr
        self.num_partitions_value = num_partitions_value
        self.max_num_partitions_value = max_num_partitions_value

    @always_inline
    def num_partitions(self) -> UInt32:
        return self.num_partitions_value

    @always_inline
    def max_num_partitions(self) -> UInt32:
        return self.max_num_partitions_value

    @always_inline
    def lse_pointer(self) -> Self.LSEPointerType:
        return {self.ptr.as_imm().as_unsafe_any_origin()}
