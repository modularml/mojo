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
"""Attention struct for RDNA Wave32 MHA kernels (prefill + decode).

TileTensor throughout. Wave32 + 16x16x16 WMMA + wave-cooperative
fragments. Constructor surface matches `amd_structured/Attention` so
the dispatcher (`nn/attention/gpu/mha.mojo`) can branch on
`_is_amd_rdna()` without restructuring its call sites.
"""

from std.collections import OptionalReg
from std.math import ceildiv, recip
from std.math.uutils import umod, ufloordiv
from std.math.constants import log2e
from std.algorithm.functional import unswitch
from max.gpu import block_idx, lane_id, thread_idx
from std.memory import unsafe_stack_allocation
from std.sys import align_of, simd_width_of
from std.utils import IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf

from layout import (
    ComptimeInt,
    Coord,
    Idx,
    MixedLayout,
    TileTensor,
)
from layout.tile_layout import row_major
from layout.tensor_core import TiledTensorCore

from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig

from .buffers import (
    QRegisterBufferRDNA,
    OutputRegisterBufferRDNA,
    PRegisterBufferRDNA,
    RDNA_AB_FRAG_SIZE,
    RDNA_CD_FRAG_SIZE,
)
from .config import (
    MHAAttentionConfigRDNA,
    RDNA_MMA_M,
    RDNA_MMA_N,
    RDNA_MMA_K,
)
from .softmax import SoftmaxRDNA
from .utils import get_warp_coords, pad

# RDNA k_group_size is always 1 (full K per WMMA fragment).
comptime RDNA_K_GROUP_SIZE = 1


@always_inline
def _mask_apply_rdna[
    masked: Bool,
    accum_type: DType,
    token_gen: Bool,
    mma_shape: IndexList[3],
    num_m_mmas: Int,
    num_n_mmas: Int,
    mask_t: MHAMask,
    group: Int,
    frag_num_rows: Int,
    use_exp2: Bool = False,
](
    kv_tile_start_row: UInt32,
    kv_tile_num_rows: UInt32,
    start_pos: UInt32,
    seq_len: UInt32,
    num_keys: UInt32,
    mask_block_row: UInt32,
    mask_warp_row: UInt32,
    mask_warp_col: UInt32,
    scale: Float32,
    mask: mask_t,
    p_reg_tile: TileTensor[
        mut=True, accum_type, _, _, address_space=.LOCAL, linear_idx_type=_
    ],
    not_last_iter: Bool,
    cache_start_pos: UInt32 = 0,
):
    """Apply `mask` to the per-lane QK score fragments in place."""
    comptime output_frag_size = frag_num_rows
    var p_reg_vectorized = p_reg_tile.vectorize[1, output_frag_size]()
    comptime assert p_reg_vectorized.flat_rank == 2

    var lane = lane_id()
    var scale_log2e: Scalar[accum_type] = scale.cast[accum_type]() * (
        log2e if use_exp2
        and not mask_t.apply_log2e_after_mask else Scalar[accum_type](1)
    )

    # RDNA WMMA C/D: lane l, elem v -> D[row=v*2+l//16, col=l%16].
    var lane_seq_offset = umod(lane, 16)
    var lane_key_group = ufloordiv(lane, 16)

    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime mma_id = n_mma * num_m_mmas + m_mma
            p_reg_vectorized[mma_id, 0] = (
                p_reg_vectorized[mma_id, 0] * scale_log2e
            )

            var mma_seq_base = mask_warp_row + UInt32(m_mma * mma_shape[0])
            var mma_key_base = (
                mask_warp_col
                + UInt32(n_mma * mma_shape[1])
                + (kv_tile_start_row if token_gen else 0)
            )

            var score_seq = (
                (num_keys - 1) if token_gen else mask_block_row
                + mma_seq_base
                + UInt32(lane_seq_offset)
            )
            var score_seq_with_start_pos = score_seq + start_pos

            comptime if masked:
                comptime for j in range(output_frag_size):
                    var score_key = mma_key_base + UInt32(
                        j * 2 + lane_key_group
                    )
                    var score_key_with_cache_start_pos = (
                        score_key + cache_start_pos
                    )
                    # Use lane % group to match q_head_idx().
                    var group_idx = umod(lane, group)
                    var q_head_idx = (
                        block_idx.y * group
                        + group_idx if token_gen else block_idx.x
                    )
                    p_reg_vectorized[mma_id, 0][j] = mask.mask(
                        IndexList[4, element_type=.uint32](
                            block_idx.z,
                            q_head_idx,
                            Int(score_seq_with_start_pos),
                            Int(score_key_with_cache_start_pos),
                        ),
                        p_reg_vectorized[mma_id, 0][j],
                    )

            comptime if mask_t.apply_log2e_after_mask:
                p_reg_vectorized[mma_id, 0] = (
                    p_reg_vectorized[mma_id, 0] * log2e
                )

            # OOB key clamp: CausalMask sets `mask_out_of_bound=False`
            # for the AMD path, so apply the row/col bound mask here
            # rather than upstream.
            var bound_seq = num_keys if token_gen else seq_len

            if score_seq >= bound_seq:
                comptime for j in range(output_frag_size):
                    p_reg_vectorized[mma_id, 0][j] = min_or_neg_inf[
                        accum_type
                    ]()
            elif not not_last_iter or token_gen:
                var bound_key = (
                    kv_tile_start_row
                    + kv_tile_num_rows if token_gen else num_keys
                )

                comptime for j in range(output_frag_size):
                    var score_key = mma_key_base + UInt32(
                        j * 2 + lane_key_group
                    )
                    if score_key >= bound_key:
                        p_reg_vectorized[mma_id, 0][j] = min_or_neg_inf[
                            accum_type
                        ]()


struct AttentionRDNA[
    output_type: DType,
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    config: MHAConfig,
    group: Int,
    sink: Bool,
    token_gen: Bool = False,
    q_depth: Int = config.depth,
    cache_depth: Int = config.depth,
    output_depth: Int = config.depth,
]:
    """RDNA Wave32 multi-head attention tile driver for prefill and decode.

    Holds the Q/K/V register and shared-memory buffers, online-softmax state,
    and output accumulator, and exposes the per-iteration mask, softmax, and
    store hooks that the prefill and decode kernels in `mha.mojo` invoke.

    Parameters:
        output_type: The `DType` of the output tensor (inferred).
        q_type: The `DType` of the query tensor (inferred).
        k_t: The `MHAOperand` type of the key operand (inferred).
        v_t: The `MHAOperand` type of the value operand (inferred).
        mask_t: The `MHAMask` type applied to attention scores (inferred).
        config: The MHA configuration holding tile, warp, and head counts.
        group: Number of query heads sharing each KV head in GQA.
        sink: Whether to initialize the softmax with attention sink
            weights.
        token_gen: Whether the kernel runs in decode (token generation)
            mode rather than prefill (defaults to `False`).
        q_depth: Head dimension of the query (defaults to `config.depth`).
        cache_depth: Head dimension of each KV cache entry (defaults to
            `config.depth`).
        output_depth: Head dimension of the output projection (defaults
            to `config.depth`).
    """

    comptime attention_config = MHAAttentionConfigRDNA[
        Self.token_gen, Self.config, Self.group
    ]

    comptime BM = Self.config.block_m()
    comptime BN = Self.config.block_n()
    comptime BK = Self.config.block_k()
    comptime WM = Self.config.warp_m()
    comptime WN = Self.config.warp_n()
    comptime num_threads = Self.config.num_threads()
    comptime num_heads = Self.config.num_heads
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps_m = Self.BM // Self.WM
    comptime depth = Self.config.depth
    comptime accum_type = get_accum_type[Self.q_type]()

    comptime mma_shape = IndexList[3](RDNA_MMA_M, RDNA_MMA_N, RDNA_MMA_K)

    # Per-lane C/D fragment: 1 row x 8 cols.
    comptime frag_num_rows = 1
    comptime output_frag_size = RDNA_CD_FRAG_SIZE

    comptime num_m_mmas = ceildiv(Self.WM, Self.mma_shape[0])
    comptime num_n_mmas = ceildiv(Self.WN, Self.mma_shape[1])
    comptime num_n_mmas_output = ceildiv(
        Self.output_depth // Self.num_warps_n, Self.mma_shape[1]
    )

    comptime swap_a_b = True
    comptime use_exp2 = True

    comptime k_group_size = RDNA_K_GROUP_SIZE
    comptime num_k_mmas2 = ceildiv(
        Self.BK, Self.mma_shape[2] * Self.k_group_size
    )

    comptime num_stages = 2

    comptime OutputRegisterBufferType = OutputRegisterBufferRDNA[
        Self.accum_type,
        Self.num_m_mmas,
        Self.num_n_mmas_output,
    ]

    comptime PRegisterBufferType = PRegisterBufferRDNA[
        Self.accum_type,
        Self.q_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.WM,
        Self.WN,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.mma_shape,
        Self.k_group_size,
    ]

    comptime QRegisterBufferType = QRegisterBufferRDNA[
        dtype=Self.q_type,
        mma_shape=Self.mma_shape,
        k_group_size=Self.k_group_size,
        WM=Self.WM,
        WN=Self.WN,
        BN=Self.BN,
        BK=Self.BK,
        depth=Self.q_depth,
    ]

    comptime SoftmaxType = SoftmaxRDNA[
        Self.accum_type,
        Self.num_m_mmas,
        Self.num_n_mmas,
        Self.num_warps_m,
        Self.num_warps_n,
        use_exp2=Self.use_exp2,
    ]

    # Q / output / KV DRAM tile layouts. Scalar rows + ComptimeInt
    # depth/strides; the runtime row count drives SRD OOB clamping.

    comptime kv_num_heads = Self.num_heads // Self.group

    comptime _q_stride0 = (
        Self.num_heads * Self.q_depth if not Self.token_gen else Self.q_depth
    )
    comptime QTileLayout = MixedLayout[
        Coord[Int64, ComptimeInt[Self.q_depth]].element_types,
        Coord[ComptimeInt[Self._q_stride0], ComptimeInt[1]].element_types,
    ]

    comptime _output_stride0 = (
        Self.num_heads
        * Self.output_depth if not Self.token_gen else Self.output_depth
    )
    comptime OutputTileLayout = MixedLayout[
        Coord[Int64, ComptimeInt[Self.output_depth]].element_types,
        Coord[ComptimeInt[Self._output_stride0], ComptimeInt[1]].element_types,
    ]

    comptime _kv_stride0 = Self.kv_num_heads * Self.depth
    comptime KvTileLayout = MixedLayout[
        Coord[Int64, ComptimeInt[Self.depth]].element_types,
        Coord[ComptimeInt[Self._kv_stride0], ComptimeInt[1]].element_types,
    ]

    comptime _smem_alignment = align_of[
        SIMD[Self.q_type, simd_width_of[Self.q_type]()]
    ]()
    # K SMEM: BN x BK (one strip at a time).
    comptime _k_smem_size = Self.BN * Self.BK
    # V SMEM: pad(depth) x BK (transposed).
    comptime _v_smem_size = Self._padded_depth() * Self.BK
    # P SMEM (decode only): BM x BN, A operand staging for PV.
    comptime _p_smem_size = Self.BM * Self.BN if Self.token_gen else 0

    # Online-softmax cross-warp scratch; reuses K SMEM (writes happen
    # at iter boundaries, after K is consumed).
    comptime _warp_scratch_layout = row_major[
        2 * Int(Self.num_warps_n), Int(Self.BM)
    ]()

    var out_reg_buffer: Self.OutputRegisterBufferType
    var p_reg_buffer: Self.PRegisterBufferType

    var softmax: Self.SoftmaxType

    var k_smem_ptr: UnsafePointer[
        Scalar[Self.k_t.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]
    var v_smem_ptr: UnsafePointer[
        Scalar[Self.v_t.dtype], MutUntrackedOrigin, address_space=.SHARED
    ]

    var q_buffer: Self.QRegisterBufferType

    @__allow_legacy_any_origin_fields
    var output_tile: TileTensor[
        Self.output_type, Self.OutputTileLayout, MutAnyOrigin
    ]

    var batch_idx: Int

    var k: Self.k_t
    var v: Self.v_t
    var mask: Self.mask_t

    var mask_block_row: UInt32
    var mask_warp_row: UInt32
    var mask_warp_col: UInt32

    var scale: Float32

    var seq_len: Int
    var num_keys: Int
    var start_pos: Int
    var cache_start_pos: Int

    @staticmethod
    @always_inline
    def _padded_depth() -> Int:
        return pad[Self.v_t.dtype, Self.depth, Self.depth]()

    @staticmethod
    @always_inline
    def q_head_idx() -> Int:
        return Self.attention_config.q_head_idx()

    @staticmethod
    @always_inline
    def q_tile_idx() -> Int:
        return Self.attention_config.q_tile_idx()

    @staticmethod
    @always_inline
    def kv_head_idx() -> Int:
        return Self.attention_config.kv_head_idx()

    @always_inline
    def zero_p_buffer(self):
        self.p_reg_buffer.zero()

    @always_inline
    def get_batch_idx(self) -> Int:
        return self.batch_idx

    @staticmethod
    @always_inline
    def get_tensor_core_mma_qk(
        out result: TiledTensorCore[
            get_accum_type[Self.q_type](),
            Self.q_type,
            Self.mma_shape,
            group_size=Self.k_group_size,
            transpose_b=True,
        ],
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    def get_tensor_core_mma_pv(
        out result: TiledTensorCore[
            get_accum_type[Self.q_type](),
            Self.q_type,
            Self.mma_shape,
            group_size=Self.k_group_size,
            transpose_b=False,
        ],
    ):
        return type_of(result)()

    @staticmethod
    @always_inline
    def make_kv_tile[
        operand_t: MHAOperand,
        //,
    ](
        operand: operand_t,
        batch_idx: UInt32,
        start_tok_idx: UInt32,
        head_idx: UInt32,
        kv_tile_num_rows: UInt32,
    ) -> TileTensor[operand_t.dtype, Self.KvTileLayout, ImmutAnyOrigin]:
        return operand.block_paged_tile[Int(Self.BN)](
            batch_idx,
            start_tok_idx,
            head_idx,
            Self.KvTileLayout(
                Coord(
                    Int64(kv_tile_num_rows),
                    Idx[Self.depth],
                ),
                Coord(Idx[Self._kv_stride0], Idx[1]),
            ),
        )

    # `mma_qk` / `mma_pv` are inlined into the prefill / decode kernels
    # where the concrete K/V buffer types are known.

    @always_inline
    def mask_status(
        self,
        kv_tile_start_row: UInt32,
    ) -> TileMaskStatus:
        comptime if Self.token_gen:
            return self.mask.status(
                UInt32(self.batch_idx),
                IndexList[2, element_type=.uint32](
                    Int(self.num_keys - 1),
                    Int(kv_tile_start_row),
                ),
                IndexList[2, element_type=.uint32](1, Self.BN),
            )
        else:
            return self.mask.status(
                UInt32(self.batch_idx),
                IndexList[2, element_type=.uint32](
                    Int(self.mask_block_row + UInt32(self.start_pos)),
                    Int(kv_tile_start_row + UInt32(self.cache_start_pos)),
                ),
                IndexList[2, element_type=.uint32](Self.BM, Self.BN),
            )

    @always_inline
    def mask_advance(mut self):
        comptime if not Self.token_gen:
            self.mask_warp_col += UInt32(Self.BN)

    @always_inline
    def mask_skip_tile(self, status: TileMaskStatus) -> Bool:
        return status == TileMaskStatus.FULL_MASK

    @always_inline
    def mask_skip_and_advance(
        mut self,
        kv_tile_start_row: UInt32,
    ) -> Bool:
        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var status = self.mask_status(kv_tile_start_row)
            if self.mask_skip_tile(status):
                self.mask_advance()
                return True
        return False

    @always_inline
    def mask_apply(
        mut self,
        kv_tile_start_row: UInt32,
        kv_tile_num_rows: UInt32,
        not_last_iter: Bool,
    ):
        @always_inline
        def _mask_apply_impl[masked: Bool]() {imm}:
            _mask_apply_rdna[
                masked=masked,
                accum_type=Self.accum_type,
                token_gen=Self.token_gen,
                mma_shape=Self.mma_shape,
                num_m_mmas=Self.num_m_mmas,
                num_n_mmas=Self.num_n_mmas,
                mask_t=Self.mask_t,
                group=Self.group,
                frag_num_rows=RDNA_CD_FRAG_SIZE,
                use_exp2=Self.use_exp2,
            ](
                kv_tile_start_row,
                kv_tile_num_rows,
                UInt32(self.start_pos),
                UInt32(self.seq_len),
                UInt32(self.num_keys),
                self.mask_block_row,
                self.mask_warp_row,
                self.mask_warp_col,
                self.scale,
                self.mask,
                self.p_reg_buffer.reg_tile,
                not_last_iter,
                UInt32(self.cache_start_pos),
            )

        comptime if not Self.token_gen or Self.mask_t.check_mask_during_decoding:
            var mask_status = self.mask_status(kv_tile_start_row)
            unswitch(
                mask_status == TileMaskStatus.PARTIAL_MASK, _mask_apply_impl
            )
        else:
            _mask_apply_impl[masked=True]()
        self.mask_advance()

    @always_inline
    def __init__(
        out self,
        output_ptr: UnsafePointer[Scalar[Self.output_type], MutAnyOrigin],
        q: UnsafePointer[Scalar[Self.q_type], ImmutAnyOrigin],
        k: Self.k_t,
        v: Self.v_t,
        mask: Self.mask_t,
        sink_weights: OptionalReg[
            UnsafePointer[Scalar[Self.q_type], ImmutAnyOrigin]
        ],
        batch_idx: Int,
        scale: Float32,
        seq_len: Int,
        num_keys: Int,
        start_pos: Int,
        cache_start_pos: Int = 0,
    ):
        # Online softmax + output accumulator (registers).
        self.softmax = Self.SoftmaxType()
        self.out_reg_buffer = Self.OutputRegisterBufferType()
        self.out_reg_buffer.zero()

        # SMEM allocations.
        self.k_smem_ptr = unsafe_stack_allocation[
            Self._k_smem_size,
            Self.k_t.dtype,
            address_space=.SHARED,
            alignment=Self._smem_alignment,
        ]()
        self.v_smem_ptr = unsafe_stack_allocation[
            Self._v_smem_size,
            Self.v_t.dtype,
            address_space=.SHARED,
            alignment=Self._smem_alignment,
        ]()

        # P buffer: dedicated SMEM for prefill (BM*BK), borrows decode-mode
        # P SMEM region (BM*BN) otherwise.
        comptime if not Self.token_gen:
            var p_ptr = unsafe_stack_allocation[
                Self.BM * Self.BK,
                Self.q_type,
                address_space=.SHARED,
            ]()
            self.p_reg_buffer = Self.PRegisterBufferType(
                p_ptr.as_unsafe_any_origin()
            )
        else:
            var p_ptr = unsafe_stack_allocation[
                Self._p_smem_size,
                Self.q_type,
                address_space=.SHARED,
            ]()
            self.p_reg_buffer = Self.PRegisterBufferType(
                p_ptr.as_unsafe_any_origin()
            )

        # Q tile: pre-offset and wrapped as TileTensor with Scalar rows.
        var valid_rows: UInt32 = UInt32(Self.group) if Self.token_gen else min(
            UInt32(Self.BM),
            UInt32(seq_len) - UInt32(Self.q_tile_idx()) * UInt32(Self.BM),
        )
        var q_offset = Self.attention_config.get_q_offset[Self.q_depth]()
        var output_offset = Self.attention_config.get_output_offset[
            Self.output_depth
        ]()

        var q_tile = TileTensor[Self.q_type, Self.QTileLayout, ImmutAnyOrigin](
            ptr=q + Int(q_offset),
            layout=Self.QTileLayout(
                Coord(
                    Int64(valid_rows),
                    Idx[Self.q_depth],
                ),
                Coord(Idx[Self._q_stride0], Idx[1]),
            ),
        )
        self.q_buffer = Self.QRegisterBufferType(q_tile, Int(valid_rows))

        self.output_tile = TileTensor[
            Self.output_type, Self.OutputTileLayout, MutAnyOrigin
        ](
            ptr=output_ptr + Int(output_offset),
            layout=Self.OutputTileLayout(
                Coord(
                    Int64(valid_rows),
                    Idx[Self.output_depth],
                ),
                Coord(
                    Idx[Self._output_stride0],
                    Idx[1],
                ),
            ),
        )

        self.k = k
        self.v = v
        self.mask = mask

        self.mask_block_row = UInt32(self.q_tile_idx() * Self.BM)
        var warp_coords = get_warp_coords[Self.BN, Self.WN]()
        self.mask_warp_row = UInt32(warp_coords[0] * Self.WM)
        self.mask_warp_col = UInt32(warp_coords[1] * Self.WN)

        self.batch_idx = batch_idx
        self.scale = scale

        self.seq_len = seq_len
        self.num_keys = num_keys
        self.start_pos = start_pos
        self.cache_start_pos = cache_start_pos

        comptime if Self.sink:
            assert Bool(
                sink_weights
            ), "expect sink_weights to be non-null when sink=true"
            var sink_weight = (
                sink_weights.value()[self.q_head_idx()].cast[Self.accum_type]()
                * log2e
            )
            _ = self.softmax.rowmax_tensor.fill(sink_weight)
            _ = self.softmax.rowsum_tensor.fill(1)
        else:
            _ = self.softmax.rowmax_tensor.fill(
                min_or_neg_inf[Self.accum_type]()
            )
            _ = self.softmax.rowsum_tensor.fill(0)

    @always_inline
    def online_softmax(mut self):
        """One online softmax iteration: max → exp → sum → correction
        → update output."""
        var warp_row: Int = get_warp_coords[Self.BN, Self.WN]()[0]

        # Cross-warp reduction scratch reuses K SMEM (writes only happen
        # at iteration boundaries, after K has been consumed).
        var warp_scratch = TileTensor[
            Self.accum_type,
            type_of(Self._warp_scratch_layout),
            address_space=.SHARED,
        ](
            self.k_smem_ptr.bitcast[Scalar[Self.accum_type]](),
            Self._warp_scratch_layout,
        ).tile[
            2 * Self.num_warps_n, Self.WM
        ](
            0, warp_row
        )

        self.softmax.full(
            self.out_reg_buffer.reg_tile,
            self.p_reg_buffer.reg_tile,
            warp_scratch,
        )

    @always_inline
    def apply_softmax_denominator(self):
        """Divide the output accumulator by the softmax row sum, in-place."""
        comptime for m_mma in range(Self.num_m_mmas):
            var rowsum_inv = recip(self.softmax.rowsum_tensor[m_mma, 0][0])
            comptime for n_mma in range(Self.num_n_mmas_output):
                comptime for i in range(Self.output_frag_size):
                    self.out_reg_buffer.reg_tile[
                        n_mma * Self.num_m_mmas + m_mma, i
                    ] *= rowsum_inv

    @always_inline
    def store_output(self):
        """Store output from registers to global memory."""
        var warp_row: Int = get_warp_coords[Self.BN, Self.WN]()[0]
        var warp_col: Int = get_warp_coords[Self.BN, Self.WN]()[1]

        var reg_tile = self.out_reg_buffer.get_reg_tile()
        var lane = lane_id()
        var row_group = lane // 16
        var col_within_mma = lane % 16

        # Decode rows = heads in GQA group (stride=output_depth).
        # Prefill rows = seq positions (stride=num_heads*output_depth).
        comptime row_stride = Self.output_depth if Self.token_gen else Self.num_heads * Self.output_depth
        var row_bound = Self.group if Self.token_gen else self.seq_len

        comptime for depth_tile in range(Self.num_n_mmas_output):
            comptime for seq_tile in range(Self.num_m_mmas):
                comptime mma_idx = depth_tile * Self.num_m_mmas + seq_tile

                comptime for elem in range(Self.output_frag_size):
                    var seq_in_mma = col_within_mma
                    var depth_in_mma = elem * 2 + row_group

                    var global_seq = (
                        warp_row * Self.WM
                        + seq_tile * Self.mma_shape[0]
                        + Int(seq_in_mma)
                    )
                    var global_depth = (
                        warp_col * Self.output_depth // Self.num_warps_n
                        + depth_tile * Self.mma_shape[1]
                        + Int(depth_in_mma)
                    )

                    if global_seq < row_bound:
                        var output_offset = (
                            global_seq * row_stride + global_depth
                        )
                        comptime reg_offset = (
                            mma_idx * Self.output_frag_size + elem
                        )
                        var val = reg_tile.ptr[reg_offset]
                        self.output_tile.ptr[output_offset] = val.cast[
                            Self.output_type
                        ]()

    @always_inline
    def copy_fragment_to_smem[chunk_idx: Int](self):
        """Copy one chunk of P to shared memory.

        Parameters:
            chunk_idx: The compile-time index of the P fragment chunk to
                copy.
        """
        self.p_reg_buffer.copy_to_shared[chunk_idx]()

    @always_inline
    def store_partition_info(
        self,
        num_partitions: Int,
        exp_sum_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
        qk_max_ptr: UnsafePointer[
            Scalar[get_accum_type[Self.q_type]()], MutAnyOrigin
        ],
    ):
        comptime if not Self.token_gen:
            return

        var q_head_idx = self.q_head_idx()
        if num_partitions > 1:
            if thread_idx.x < Self.group:
                var row_sum = self.softmax.rowsum_tensor[0, 0][0]
                var row_max = self.softmax.rowmax_tensor[0, 0][0]
                exp_sum_ptr[q_head_idx] = row_sum
                qk_max_ptr[q_head_idx] = row_max
