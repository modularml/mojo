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

from std.math import ceildiv
from std.sys import simd_width_of, size_of

from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.gpu.nvidia.mha_tile_scheduler import MHATileScheduler, SeqInfo
from nn.attention.gpu.nvidia.sm100.attention import (
    FA4Config,
    EnableForcedOrdering,
    SM100_RESERVED_SMEM_BYTES,
)
from nn.attention.gpu.nvidia.sm100.attention_utils import (
    SM100TensorAccumulator,
    FA4MiscMBars,
    SharedMemPointer,
    TMADestination,
    MBarType,
    elect,
    elect_mma_arrive,
)
from nn.attention.gpu.nvidia.common import (
    OptionalPointer,
    MHAPosition,
)
from nn.attention.mha_utils import OptionallyStaticInt, MHAPartitionScheme

from layout import TileTensor
from layout.tma_async import PipelineState
from layout.swizzle import Swizzle
from layout.tile_layout import row_major as tt_row_major
from layout.tensor_core_async import (
    tile_layout_k_major_typed,
    tile_layout_mn_major_typed,
)

from linalg.arch.sm100.mma import smem_descriptor

from max.gpu.host.info import B200
from max.gpu.globals import WARP_SIZE, WARPGROUP_SIZE
from max.gpu.memory import fence_async_view_proxy
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.compute.arch.mma_nvidia_sm100 import (
    MMASmemDescriptorPair,
    UMMAKind,
)
import max.gpu.primitives.warp as warp
from max.gpu.sync import named_barrier

from std.utils.index import Index


# rope_dtype is the dtype of the MMA!
struct MLAConfig[
    qkv_dtype: DType,
    *,
    rope_gmem_dtype: DType,
    rope_mma_dtype: DType,
    scale_dtype_: Optional[DType] = None,
](TrivialRegisterPassable):
    # Concrete scale dtype for `Scalar[...]`/`TMATensorTile[...]` reads. Falls
    # back to `qkv_dtype` when unset; presence flows to `FA4Config` via the
    # optional `scale_dtype_`.
    comptime scale_dtype = Self.scale_dtype_.or_else(Self.qkv_dtype)
    var fa4_config: FA4Config[
        Self.qkv_dtype,
        rope_dtype_=Self.rope_mma_dtype,
        scale_dtype_=Self.scale_dtype_,
    ]
    var MMA_M: Int
    var BM: Int
    var BN: Int
    var BK0: Int  # BK for MMA0
    var BK1: Int  # BK for MMA1
    var qk_depth: Int
    var rope_depth: Int
    var nope_depth: Int
    var cache_depth: Int
    var padded_qk_depth: Int  # align_up(k_depth, swizzle_elems)
    var group: Int
    var num_q_heads: Int
    var num_kv_heads: Int
    comptime TMEM_S0: Int = 0
    var TMEM_S1: Int
    var TMEM_O0: Int
    var TMEM_O1: Int
    var TMEM_P0: Int
    var TMEM_P1: Int
    var tmem_used: Int
    var num_kv_stages: Int
    var num_qk_stages: Int  # Stages for Q@K' (K loading pipelining)
    var num_pv_stages: Int  # Stages for P@V (P writing pipelining)
    var smem_used: Int
    comptime num_threads: Int = 512  # 2x softmax, 1x correction, 1x other
    var qkv_swizzle_mode: TensorMapSwizzle
    var rope_mma_swizzle_mode: TensorMapSwizzle
    var rope_gmem_swizzle_mode: TensorMapSwizzle
    var output_swizzle_mode: TensorMapSwizzle

    comptime qkv_dtype_size: Int = size_of[Self.qkv_dtype]()
    comptime rope_mma_dtype_size: Int = size_of[Self.rope_mma_dtype]()
    comptime rope_gmem_dtype_size: Int = size_of[Self.rope_gmem_dtype]()
    comptime sm100_smem_carveout = (
        B200.shared_memory_per_multiprocessor - SM100_RESERVED_SMEM_BYTES
    )
    comptime sm100_tmem_cols = 512
    comptime mbar_size = size_of[DType.int64]()
    comptime num_correction_cols = 1

    def __init__(
        out self,
        *,
        num_q_heads: Int,
        group: Int,
        depth: Int,
        page_size: Int,
        v_depth: Int = -1,
        num_q: Int = 2,
        single_o: Bool = False,
        bn_cap: Int = 0,
    ):
        # DSv3.2 absorbed-MLA dims: KV cache row width is
        # `kv_lora_rank (512) + qk_rope_head_dim (64) = 576`; the depth-64
        # RoPE tail participates in QK but not in PV, so
        # `nope_depth = qk_depth - rope_depth = 512`.
        #
        # `v_depth` is the per-head V / output width (`v_head_dim`). It is
        # DECOUPLED from `nope_depth`: DeepSeek has `v_head_dim == qk_nope`
        # (so v_depth == nope_depth) but GLM-style MLA has
        # `v_head_dim != qk_nope`. `v_depth < 0` (default) means "V width equals
        # the nope width" (the DeepSeek shape), preserved for back-compat.
        comptime rope_depth = 64
        comptime cache_depth = 576
        var nope_depth = depth - rope_depth
        var ov_depth = nope_depth if v_depth < 0 else v_depth
        debug_assert(depth > rope_depth, "MLA q depth must exceed rope_depth")
        debug_assert(ov_depth > 0, "MLA v_depth must be > 0")

        comptime if Self.qkv_dtype_size == 1:
            self.qkv_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.qkv_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        comptime if Self.rope_mma_dtype_size == 1:
            self.rope_mma_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.rope_mma_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        comptime if Self.rope_gmem_dtype_size == 1:
            self.rope_gmem_swizzle_mode = TensorMapSwizzle.SWIZZLE_64B
        else:
            self.rope_gmem_swizzle_mode = TensorMapSwizzle.SWIZZLE_128B

        # O output store is row-major SWIZZLE_NONE (decoupled from the swizzled
        # QKV/RoPE buffers). The softmax warp loads O one-row-per-thread and
        # writes it row-major, avoiding cross-thread shuffles and swizzling
        # while staying bank-conflict-free.
        self.output_swizzle_mode = TensorMapSwizzle.SWIZZLE_NONE

        # `ov_depth` (V/output width) is decoupled from `nope_depth` (the
        # non-rope Q/K width). For DeepSeek they coincide; for GLM-style MLA
        # they differ (`v_head_dim != qk_nope_head_dim`). The FA4Config uses
        # `nope_depth` for the Q@K' / Q_nope geometry and `ov_depth` for the
        # P@V / V / output geometry.
        self.fa4_config = {
            num_q_heads = num_q_heads,
            group = group,
            qk_depth = depth,
            ov_depth = ov_depth,
            swizzle_mode = self.qkv_swizzle_mode,
            page_size = page_size,
            is_mla = True,
            # FA4Config's primary knob is now BM; MLA is single-CTA so
            # num_q=2 -> BM=256, num_q=1 -> BM=128.
            BM = 256 if num_q == 2 else 128,
            nope_depth = nope_depth,
            single_o = single_o,
            bn_cap = bn_cap,
        }

        self.MMA_M = self.fa4_config.MMA_M
        self.BM = self.fa4_config.BM
        self.BN = self.fa4_config.BN
        self.BK0 = self.fa4_config.BK0
        self.BK1 = self.fa4_config.BK1
        self.qk_depth = self.fa4_config.qk_depth
        self.rope_depth = rope_depth
        self.nope_depth = self.qk_depth - self.rope_depth
        self.cache_depth = cache_depth
        self.padded_qk_depth = self.fa4_config.padded_qk_depth
        self.tmem_used = self.fa4_config.tmem_used
        self.num_kv_stages = self.fa4_config.num_kv_stages
        self.num_qk_stages = self.fa4_config.num_qk_stages
        self.num_pv_stages = self.fa4_config.num_pv_stages
        self.smem_used = self.fa4_config.smem_used
        self.group = self.fa4_config.group
        self.num_q_heads = self.fa4_config.num_q_heads
        self.num_kv_heads = self.fa4_config.num_kv_heads
        self.TMEM_S1 = self.fa4_config.TMEM_S1
        self.TMEM_O0 = self.fa4_config.TMEM_O0
        self.TMEM_O1 = self.fa4_config.TMEM_O1
        self.TMEM_P0 = self.fa4_config.TMEM_P0
        self.TMEM_P1 = self.fa4_config.TMEM_P1

    @always_inline
    def num_q(self) -> Int:
        return self.fa4_config.num_q

    @always_inline
    def q_tile_rows(self) -> Int:
        """Rows per Q TMA tile / per-half MMA — `BM // num_q`.

        128 in both modes: one of two BM=256 halves in 2Q, the single full
        BM=128 tile in 1Q. The Q (and per-token q_scale) TMA boxes and the
        ragged output store all fold to this value, which is why their op
        types match across the 1Q/2Q configs.
        """
        return self.BM // self.fa4_config.num_q

    @always_inline
    def with_num_q(self, num_q: Int) -> Self:
        """Reconstruct this config with a different `num_q` (single-CTA).

        Mirrors `FA4Config.with_num_q`, but simpler: MLA pins
        `num_qk_stages == 1` (is_mla), so there is no staging knob to
        match between the 1Q and 2Q variants.

        `v_depth` (the V/output head dim, carried by `fa4_config.ov_depth`)
        MUST be re-passed: otherwise the rebuilt config defaults to
        `v_depth == nope_depth`, so the 1Q variant's V/output geometry would
        diverge from the 2Q config's (a type mismatch in the shared O-store /
        V TMA tile when `v_head_dim != qk_nope_head_dim`).
        """
        return Self(
            num_q_heads=self.num_q_heads,
            group=self.group,
            depth=self.qk_depth,
            page_size=self.fa4_config.page_size,
            v_depth=self.fa4_config.ov_depth,
            num_q=num_q,
            # Preserve single-O only when reconstructing as 1Q (it implies 1Q).
            single_o=self.fa4_config.single_o and num_q == 1,
        )

    @always_inline
    def switch_1q_config(self) -> Self:
        """The 1Q variant used by the in-kernel per-sequence 1Q/2Q switch.

        Identical to `with_num_q(1)` (see `with_num_q` for why MLA has
        no staging-pinning concern, unlike `FA4Config.switch_1q_config`).
        """
        return self.with_num_q(1)

    @always_inline
    def can_switch_to_1q(self) -> Bool:
        """Whether a 2Q-launched kernel may dispatch to the 1Q body at
        runtime.

        True only when this is a 2Q config AND the 1Q variant is valid.
        The TMA-op types fold between the two configs by construction:
        the Q TMA / ragged-store `BM // num_q` is 128 in both modes, and
        the K_nope/K_rope/V TMA shapes are BM-independent (BN's formula
        does not reference `num_q`).
        """
        if self.fa4_config.num_q != 2 or self.fa4_config.pair_cta:
            return False
        var cfg1 = self.switch_1q_config()
        return cfg1.supported() and cfg1.fa4_config.supported()

    @always_inline
    def launch_smem_used(self) -> Int:
        """Dynamic smem to reserve when launching this config's kernel.

        When the launched kernel may dispatch to the 1Q body at runtime
        (`can_switch_to_1q()`), it constructs the 1Q `SM100AttentionSMem`
        over the same dynamic smem region, so the launch must reserve the
        max of both footprints. Otherwise this is just `smem_used`.
        """
        if self.can_switch_to_1q():
            return max(self.smem_used, self.switch_1q_config().smem_used)
        return self.smem_used

    @always_inline
    def launch_num_threads(self) -> Int:
        """Threads to launch for this config's kernel.

        The generic single-O (wide-V) path drops the redundant 2nd softmax
        warpgroup -- WG1 is a full no-op there (see the single-O serial-KV
        accumulation) -- so it launches 3 warpgroups (Softmax0 + Correction +
        MMA/Load/Empty) instead of 4. Every other config keeps the standard
        4-warpgroup (`num_threads` = 512) layout. Only the generic kernel calls
        this; the per-token-scale / blockscale siblings read the `num_threads`
        field directly and stay at 512 even for their own single-O configs
        (they keep the 2nd softmax WG).
        """
        if self.fa4_config.single_o:
            return 3 * WARPGROUP_SIZE
        return self.num_threads

    @always_inline
    def prefer_1q(
        self,
        max_prompt_len: UInt32,
        num_partitions: UInt32,
        batch_size: UInt32,
        sm_count: Int,
    ) -> Bool:
        """Runtime 1Q-vs-2Q grid heuristic for a 2Q config (mirrors the MHA
        heuristic in `dispatch.mojo`): prefer 1Q when (a) `max_prompt_len`
        fits a single 1Q tile (`q_tile_rows()`), so 2Q's BM=256 would waste
        >= 50% of Q rows, or (b) the unclamped 2Q grid only fills <= half
        the SMs, so halving BM doubles the grid without oversubscribing.
        """
        var tiles_2q = ceildiv(max_prompt_len, UInt32(self.BM))
        var raw_grid_2q = (
            tiles_2q * num_partitions * UInt32(self.num_q_heads) * batch_size
        )
        return max_prompt_len <= UInt32(
            self.q_tile_rows()
        ) or raw_grid_2q <= UInt32(sm_count // 2)

    @always_inline
    def num_rope_buffers(self) -> Int:
        return self.fa4_config.num_rope_buffers()

    def supported(self) -> Bool:
        return (
            self.qk_depth >= 64
            and self.BN >= 64
            and self.num_kv_stages >= 2
            and self.tmem_used <= Self.sm100_tmem_cols
            and self.smem_used <= Self.sm100_smem_carveout
        )

    def correction_smem_elements(self) -> Int:
        return self.BM * Self.num_correction_cols


@always_inline
def select_mla_prefill_config[
    qkv_dtype: DType,
    *,
    rope_gmem_dtype: DType,
    rope_mma_dtype: DType,
    scale_dtype_: Optional[DType] = None,
](
    *,
    num_q_heads: Int,
    group: Int,
    depth: Int,
    page_size: Int,
    v_depth: Int,
) -> MLAConfig[
    qkv_dtype,
    rope_gmem_dtype=rope_gmem_dtype,
    rope_mma_dtype=rope_mma_dtype,
    scale_dtype_=scale_dtype_,
]:
    """Selects the supported SM100 MLA-prefill config for these dims.

    Shared by the generic / blockscale / per-token-scale prefill kernels, which
    differ only in the config's dtype parameters, so the single-O fallback
    policy lives in ONE place instead of three hand-synced copies.

    `v_depth` is the per-head V / output width (`v_head_dim`); `-1` means "V
    width == nope width" (the DeepSeek shape). The standard 2-O config is tried
    first, so when `v_head_dim == qk_nope_head_dim` the result is byte-identical
    to the pre-decoupling path. A wide V (e.g. `v_head_dim=256`) overflows the
    2-O TMEM layout (standard BN=0), so fall back to a single-O (`num_q=1`)
    config at the TMEM-max BN, then to a BN capped at the `supported()` floor so
    >= 2 KV stages still fit shared memory. If none is supported, return the
    standard config so the caller's `supported()` assert reports the real dims.
    """
    comptime Config = MLAConfig[
        qkv_dtype,
        rope_gmem_dtype=rope_gmem_dtype,
        rope_mma_dtype=rope_mma_dtype,
        scale_dtype_=scale_dtype_,
    ]
    # `bn_floor` == the `MLAConfig.supported()` BN floor (`self.BN >= 64`): the
    # largest MMA_K-aligned BN cap that still admits >= 2 KV stages for a wide V.
    var bn_floor = 64
    var standard = Config(
        num_q_heads=num_q_heads,
        group=group,
        depth=depth,
        page_size=page_size,
        v_depth=v_depth,
    )
    var singleo_max = Config(
        num_q_heads=num_q_heads,
        group=group,
        depth=depth,
        page_size=page_size,
        v_depth=v_depth,
        num_q=1,
        single_o=True,
    )
    var singleo_floor = Config(
        num_q_heads=num_q_heads,
        group=group,
        depth=depth,
        page_size=page_size,
        v_depth=v_depth,
        num_q=1,
        single_o=True,
        bn_cap=bn_floor,
    )
    return standard if standard.fa4_config.supported() else (
        singleo_max if singleo_max.fa4_config.supported() else (
            singleo_floor if singleo_floor.fa4_config.supported() else standard
        )
    )


@always_inline
def split_smem[
    first_size: Int, second_size: Int, first_dtype: DType, second_dtype: DType
](tensor: TileTensor[address_space=.SHARED, ...]) -> Tuple[
    TileTensor[
        first_dtype,
        type_of(tt_row_major[first_size]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ],
    TileTensor[
        second_dtype,
        type_of(tt_row_major[second_size]()),
        MutAnyOrigin,
        address_space=.SHARED,
    ],
]:
    """Split a shared memory tensor into two TileTensors at the boundary
    of `first_size` elements.

    TMA only uses .ptr — flat row_major layout avoids needing
    InternalLayout equivalents of swizzled layouts.
    """
    comptime SmemPtr[dt: DType] = UnsafePointer[
        Scalar[dt], MutAnyOrigin, address_space=.SHARED
    ]
    var ptr = rebind[SmemPtr[first_dtype]](tensor.ptr)
    comptime first_layout = tt_row_major[first_size]()
    comptime second_layout = tt_row_major[second_size]()
    return {
        TileTensor[
            first_dtype,
            type_of(first_layout),
            MutAnyOrigin,
            address_space=.SHARED,
        ](ptr, first_layout),
        TileTensor[
            second_dtype,
            type_of(second_layout),
            MutAnyOrigin,
            address_space=.SHARED,
        ](
            rebind[SmemPtr[second_dtype]](ptr + first_size),
            second_layout,
        ),
    }


struct MLAPositionSummary(TrivialRegisterPassable):
    var num_keys: UInt32
    var score_row: UInt32

    @always_inline
    def __init__(out self, num_keys: UInt32, score_row: UInt32):
        self.num_keys = num_keys
        self.score_row = score_row

    @staticmethod
    @always_inline
    def get_num_keys_and_start_pos[
        KRopeType: MHAOperand,
        //,
        _ndbuffer_mha_operand: Bool,
    ](k_rope_lut: KRopeType, seq_info: SeqInfo) -> Tuple[UInt32, UInt32]:
        var num_keys: UInt32
        var start_pos: UInt32
        comptime if _ndbuffer_mha_operand:
            num_keys = UInt32(
                warp.broadcast(
                    k_rope_lut.cache_length(Int(seq_info.prompt_idx))
                )
            )
            start_pos = UInt32(num_keys - warp.broadcast(seq_info.seq_len))
        else:
            start_pos = UInt32(
                warp.broadcast(
                    k_rope_lut.cache_length(Int(seq_info.prompt_idx))
                )
            )
            num_keys = start_pos + warp.broadcast(seq_info.seq_len)
        return {num_keys, start_pos}

    @staticmethod
    @always_inline
    def get_score_row(seq_info: SeqInfo, start_pos: UInt32) -> UInt32:
        return start_pos + warp.broadcast(seq_info.prompt_offset)

    @staticmethod
    @always_inline
    def create[
        KRopeType: MHAOperand,
        //,
        _ndbuffer_mha_operand: Bool,
    ](k_rope_lut: KRopeType, seq_info: SeqInfo,) -> MLAPositionSummary:
        var num_keys, start_pos = Self.get_num_keys_and_start_pos[
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ](k_rope_lut, seq_info)
        var score_row = Self.get_score_row(seq_info, start_pos)
        return {num_keys, score_row}


struct MLAKVLayouts[
    k_nope_dtype: DType,
    k_rope_dtype: DType,
    kv_scale_dtype: Optional[DType],
    config: MLAConfig,
]:
    """Comptime layout and size metadata for MLA K/V tiles."""

    comptime k_nope_tma_layout = tile_layout_k_major_typed[
        Self.k_nope_dtype,
        Self.config.BN,
        Self.config.nope_depth,
        Self.config.qkv_swizzle_mode,
    ].static_product
    comptime k_rope_tma_layout = tile_layout_k_major_typed[
        Self.k_rope_dtype,
        Self.config.BN,
        Self.config.rope_depth,
        Self.config.rope_gmem_swizzle_mode,
    ].static_product
    comptime k_tma_layout = tile_layout_k_major_typed[
        Self.k_nope_dtype,
        Self.config.BN,
        Self.config.BK0,
        Self.config.qkv_swizzle_mode,
    ].static_product
    # V tile is (v_depth x BN) mn-major for P@V; the MN dim is the V head dim
    # (`ov_depth`), NOT `nope_depth` — they differ when v_head_dim != qk_nope.
    comptime v_tma_layout = tile_layout_mn_major_typed[
        Self.k_nope_dtype,
        Self.config.fa4_config.ov_depth,
        Self.config.BK1,
        Self.config.qkv_swizzle_mode,
    ].static_product

    comptime KPairType = TMADestination[Self.k_nope_dtype, Self.k_tma_layout]
    comptime VPairType = TMADestination[Self.k_nope_dtype, Self.v_tma_layout]
    comptime k_elements = Self.k_nope_tma_layout + Self.k_rope_tma_layout
    comptime v_elements = Self.v_tma_layout
    comptime k_nope_bytes = Self.k_nope_tma_layout * size_of[
        Self.k_nope_dtype
    ]()
    comptime k_rope_bytes = Self.k_rope_tma_layout * size_of[
        Self.k_rope_dtype
    ]()
    comptime k_bytes = Self.k_nope_bytes + Self.k_rope_bytes
    comptime v_bytes = Self.v_elements * size_of[Self.k_nope_dtype]()
    comptime SMemType = SharedMemPointer[Scalar[Self.k_nope_dtype]]


struct TMAtoCvtPipeline[
    num_kv_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    @__allow_legacy_any_origin_fields
    var consumer_mbars: MBarType

    @__allow_legacy_any_origin_fields
    var producer_mbars: MBarType
    var state: PipelineState[Self.num_kv_stages]

    @always_inline
    def __init__(out self, consumer_mbars: MBarType, producer_mbars: MBarType):
        self.consumer_mbars = consumer_mbars
        self.producer_mbars = producer_mbars
        self.state = {}

    @always_inline
    def init(self):
        comptime for i in range(Self.num_kv_stages):
            self.consumer_mbars[i].init(Int32(Self.num_consumer))
            self.producer_mbars[i].init(Int32(Self.num_producer))

    @always_inline
    def producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    def consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    def producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    def consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    def producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    def consumer_release(mut self):
        _ = self.consumer_mbar()[].arrive()
        self.step()

    @always_inline
    def step(mut self):
        self.state.step()


struct CvtToMMAPipeline[
    num_stages: Int,
    num_producer: Int,
    num_consumer: Int,
](TrivialRegisterPassable):
    @__allow_legacy_any_origin_fields
    var producer_mbars: MBarType

    @__allow_legacy_any_origin_fields
    var consumer_mbars: MBarType
    var state: PipelineState[Self.num_stages]

    @always_inline
    def __init__(out self, producer_mbars: MBarType, consumer_mbars: MBarType):
        self.producer_mbars = producer_mbars
        self.consumer_mbars = consumer_mbars
        self.state = {}

    @always_inline
    def init(self):
        comptime for i in range(Self.num_stages):
            self.producer_mbars[i].init(Int32(Self.num_producer))
            self.consumer_mbars[i].init(Int32(Self.num_consumer))

    @always_inline
    def producer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.producer_mbars + idx

    @always_inline
    def consumer_mbar(self) -> MBarType:
        var idx: UInt32 = self.state.index()
        return self.consumer_mbars + idx

    @always_inline
    def producer_acquire(self):
        self.consumer_mbar()[].wait(self.state.phase())

    @always_inline
    def consumer_wait(self):
        self.producer_mbar()[].wait(self.state.phase())

    @always_inline
    def producer_commit(mut self):
        _ = self.producer_mbar()[].arrive()
        self.step()

    @always_inline
    def consumer_release(mut self, elect: Int32):
        elect_mma_arrive(self.consumer_mbar(), elect)
        self.step()

    @always_inline
    def step(mut self):
        self.state.step()


@always_inline
def cvt_block_fp8_to_bf16_with_scale[
    input_type: DType,
    output_dtype: DType,
    KRopeType: MHAOperand,
    //,
    swizzle_fp8: Swizzle,
    swizzle_bf16: Swizzle,
](
    input: TileTensor[input_type, _, address_space=.SHARED, ...],
    mut output: TileTensor[
        mut=True, output_dtype, _, address_space=.SHARED, ...
    ],
    k_rope_lut: KRopeType,
    seq_info: SeqInfo,
    kv_start_row: UInt32,
    num_keys: UInt32,
    tid: UInt32,
):
    """TileTensor overload — standalone implementation using `.ptr` and
    comptime `static_shape`/`static_stride` directly."""
    comptime assert (
        input_type == .float8_e4m3fn and output_dtype == .bfloat16
    ), "Only support float8_e4m3fn to bfloat16 conversion"

    comptime num_regs = (
        type_of(input).static_shape[0] * type_of(input).static_shape[1]
    ) // WARP_SIZE
    comptime row_stride = type_of(input).static_stride[0]

    var t_row, t_col = divmod(tid, 16)

    var fp8_regs = SIMD[input_type, num_regs](0)

    comptime for i in range(num_regs // 4):
        var row = UInt32(i * 2) + t_row
        var col = t_col * 4
        var elem_offset = row * UInt32(row_stride) + col
        var fp8x4 = (input.ptr + Int(swizzle_fp8(elem_offset))).load[width=4]()
        fp8_regs = fp8_regs.insert[offset=i * 4](fp8x4)

    # make sure all the fp8_regs are loaded
    named_barrier[64](6)

    var scale: Scalar[KRopeType.scale_dtype]
    comptime for i in range(num_regs // 4):
        var row = UInt32(i * 2) + t_row
        var col = t_col * 4
        var elem_offset = row * UInt32(row_stride) + col

        comptime if KRopeType.quantization_enabled:
            var tok_idx = kv_start_row + row
            if tok_idx < num_keys:
                scale = k_rope_lut.load_scale[width=1](
                    batch_idx=Int(seq_info.prompt_idx),
                    start_tok_idx=Int(tok_idx),
                    head_idx=0,
                    head_dim_idx=576 - 64,
                )
            else:
                scale = SIMD[KRopeType.scale_dtype, 1](1)

            var fp32x4 = fp8_regs.slice[4, offset=i * 4]().cast[
                KRopeType.scale_dtype
            ]()
            fp32x4 = fp32x4 * scale
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](
                fp32x4.cast[output_dtype]()
            )
        else:
            var fp16x4 = fp8_regs.slice[4, offset=i * 4]().cast[output_dtype]()
            (output.ptr + Int(swizzle_bf16(elem_offset))).store[width=4](fp16x4)

    fence_async_view_proxy()


struct SM100MLA[
    KVLUTType: MHAOperand,
    KRopeType: MHAOperand,
    output_dtype: DType,
    MaskType: MHAMask,
    SchedulerType: MHATileScheduler,
    config: MLAConfig,
    ValidLengthType: OptionalPointer,
    SinkType: OptionalPointer,
    KVRowOffsetsType: OptionalPointer,
    MaxSeqLenType: OptionallyStaticInt,
    PartitionType: MHAPartitionScheme,
    _ndbuffer_mha_operand: Bool,
](TrivialRegisterPassable):
    comptime qkv_dtype = Self.KVLUTType.dtype
    comptime rope_mma_dtype = Self.config.rope_mma_dtype
    comptime rope_gmem_dtype = Self.KRopeType.dtype
    comptime accum_dtype = DType.float32
    comptime simd_size: Int = simd_width_of[Self.qkv_dtype]()

    comptime cta_group = 1  # TODO: support 2
    comptime BM = Self.config.BM
    comptime BN = Self.config.BN
    comptime qk_depth = Self.config.qk_depth  # 192
    comptime padded_depth = Self.config.padded_qk_depth  # 192
    comptime num_q_heads = Self.config.num_q_heads
    comptime group = Self.config.group
    comptime page_size = Self.KVLUTType.page_size

    comptime rope_depth = Self.config.rope_depth
    comptime nope_depth = Self.config.nope_depth
    # V / output head dim (`v_head_dim`). Equals `nope_depth` for DeepSeek but
    # differs for GLM-style MLA. Used for the P@V output width and V geometry.
    comptime ov_depth = Self.config.fa4_config.ov_depth
    comptime padded_nope_depth = Self.config.fa4_config.padded_nope_depth
    comptime padded_ov_depth = Self.config.fa4_config.padded_ov_depth
    comptime cache_depth = Self.config.cache_depth

    # 128 in both modes: 2Q has BM=256 split into two per-half MMAs;
    # 1Q has BM=128 covered by a single full-BM MMA (mirrors kernel.mojo).
    comptime MMA_M = Self.config.fa4_config.MMA_M
    comptime qkv_dt_size = size_of[Self.qkv_dtype]()

    comptime num_qk_stages = Self.config.num_qk_stages
    comptime num_pv_stages = Self.config.num_pv_stages

    comptime nope_mma_kind = (
        UMMAKind.KIND_F16 if Self.qkv_dtype.is_half_float() else UMMAKind.KIND_F8F6F4
    )
    comptime rope_mma_kind = (
        UMMAKind.KIND_F16 if Self.rope_mma_dtype.is_half_float() else UMMAKind.KIND_F8F6F4
    )
    # use_shared_kv means we use a shared kv pipeline in shared memory
    # that forces us to put the k nope and rope in separate regions of smem
    # preventing us from fusing the nope and rope parts of UMMA0
    comptime fused_umma0 = (Self.qkv_dtype == Self.rope_mma_dtype) and (
        not Self.config.fa4_config.use_shared_kv
    )
    comptime BK0 = Self.qk_depth if Self.fused_umma0 else Self.nope_depth

    # First MMA is Q@K' (can be staged by num_qk_stages)
    # (BM x depth) @ (BN x depth)' -> (BM x BN)
    comptime UMMA0Type = SM100TensorAccumulator[
        Self.qkv_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,  # generally 128
        MMA_N=Self.BN,
        BK=Self.BK0,  # BK in memory depth
        a_tmem=False,
        mma_kind=Self.nope_mma_kind,
        swizzle_a=Self.config.qkv_swizzle_mode,
        swizzle_b=Self.config.qkv_swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
    ]
    comptime UMMA0RopeType = SM100TensorAccumulator[
        Self.rope_mma_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,
        MMA_N=Self.BN,
        BK=Self.rope_depth,
        a_tmem=False,
        mma_kind=Self.rope_mma_kind,
        swizzle_a=Self.config.rope_mma_swizzle_mode,
        swizzle_b=Self.config.rope_mma_swizzle_mode,
        transpose_b=True,
        num_stages=Self.num_qk_stages,
    ]
    # Second MMA is P@V
    # (BM x BN) @ (BN x v_depth) -> (BM x v_depth). The output width is the V
    # head dim (`ov_depth`), which equals `nope_depth` only when
    # `v_head_dim == qk_nope_head_dim` (DeepSeek).
    comptime UMMA1Type = SM100TensorAccumulator[
        Self.qkv_dtype,
        Self.accum_dtype,
        MMA_M=Self.MMA_M,
        MMA_N=Self.ov_depth,
        BK=Self.BN,
        a_tmem=True,
        mma_kind=Self.nope_mma_kind,
        swizzle_b=Self.config.qkv_swizzle_mode,
        transpose_b=False,
        num_stages=Self.num_pv_stages,
    ]

    # Byte offset within Q's smem tile where Q_rope columns begin.
    # Q is stored as tile_layout_k_major(BM/2, BK0), column-major atoms.
    # Q_nope occupies (BM/2) * padded_nope_depth elements, then Q_rope follows.
    # The Q_nope width is the non-rope Q/K depth (`padded_nope_depth`), not the
    # V/output depth — they differ when `v_head_dim != qk_nope_head_dim`.
    comptime q_rope_byte_offset: Int = (
        Self.MMA_M
        * Self.config.fa4_config.padded_nope_depth
        * size_of[Self.qkv_dtype]()
    )

    comptime PositionType = MHAPosition[
        Self.config.BM,
        Self.config.BN,
        Self.config.qk_depth,
        Self.config.padded_qk_depth,
        Self.config.num_q_heads,
        Self.config.group,
        False,
    ]
    # Unified misc barriers type managing all barriers including KV/O pipelines.
    # Use fa4_config fields so that the type expression matches
    # SM100AttentionSMem[config.fa4_config, ...].MiscMBarsType.
    comptime MiscMBarsType = FA4MiscMBars[
        num_qk_stages=Self.config.fa4_config.num_qk_stages,
        num_pv_stages=Self.config.fa4_config.num_pv_stages,
        num_kv_stages=Self.config.fa4_config.num_kv_stages,
        use_order_barriers=EnableForcedOrdering,
        use_shared_kv=Self.config.fa4_config.use_shared_kv,
        pair_cta=Self.config.fa4_config.pair_cta,
        num_q=Self.config.fa4_config.num_q,
        splitk_partitions=Self.config.fa4_config.splitk_partitions,
        BM=Self.config.fa4_config.BM,
        # MLA is never warp-specialized (use_ws is always False here), but the
        # param must be threaded so this MiscMBarsType matches the one built by
        # `SM100AttentionSMem.MiscMBarsType` (same `...use_ws` expression) — an
        # omitted param defaults to literal `False`, a DIFFERENT type from the
        # `config.fa4_config.use_ws` expression, and the two would not convert.
        use_ws=Self.config.fa4_config.use_ws,
        # Same expression as SM100AttentionSMem.MiscMBarsType, for the same
        # type-identity reason as use_ws above. MLA always resolves this to
        # False (rope_depth() > 0), which is what keeps MLA's mbar accounting
        # byte-identical to cross-P-off.
        crossp=Self.config.fa4_config.crossp_on(),
    ]

    @staticmethod
    @always_inline
    def mask_status(
        mask: Self.MaskType,
        seq_id: UInt32,
        score_row: UInt32,
        kv_row: UInt32,
    ) -> TileMaskStatus:
        return mask.status(
            seq_id,
            Index[dtype=DType.int32](
                Int(score_row),
                Int(kv_row),
            ),
            Index[dtype=DType.int32](Self.BM, Self.BN),
        )

    @staticmethod
    @always_inline
    def descriptor_q(
        q_smem: SharedMemPointer[Scalar[Self.qkv_dtype]],
    ) -> MMASmemDescriptorPair:
        # `BM // num_q` = 128 in both modes: one of two Q halves in 2Q,
        # the single full-BM Q tile in 1Q.
        return smem_descriptor[
            BMN=Self.config.q_tile_rows(),
            BK=Self.config.nope_depth,
            swizzle_mode=Self.config.qkv_swizzle_mode,
            is_k_major=True,
        ](q_smem)

    @always_inline
    @staticmethod
    def descriptor_q_rope(
        q_smem: SharedMemPointer[Scalar[Self.rope_mma_dtype]],
    ) -> MMASmemDescriptorPair:
        return smem_descriptor[
            BMN=Self.config.q_tile_rows(),
            BK=Self.config.rope_depth,
            swizzle_mode=Self.config.rope_mma_swizzle_mode,
            is_k_major=True,
        ](q_smem)
