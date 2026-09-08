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

"""GPU kernels for Multi-head Latent Attention (MLA) decoding and prefill.

Provides the `flare_mla_decoding` and `flare_mla_prefill` entrypoints plus
their platform-specific dispatch and kernel implementations targeting NVIDIA
(SM80/SM100) and AMD (gfx950) GPUs, including split-K reduction, multi-token
prediction (MTP) query folding, per-token scale, and sparse-attention support.
"""

from std.collections import Array, OptionalReg
from std.math import align_up, ceildiv, recip
from std.math.uutils import umod, ufloordiv, udivmod
from nn.attention.mha_utils import DynamicInt, MHA_PDL_LEVEL
from std.math.constants import log2e
from std.sys import (
    align_of,
    has_nvidia_gpu_accelerator,
    has_amd_gpu_accelerator,
    get_defined_int,
    simd_width_of,
    size_of,
    is_nvidia_gpu,
    is_amd_gpu,
    CompilationTarget,
)

from nn.attention.gpu.mha import (
    mha_splitk_reduce,
    q_num_matrix_view_rows,
)
from nn.attention.gpu.mha_decode_partition_heuristic import (
    mha_decoding_num_partitions,
)
import max.gpu.primitives.warp as warp
from max.gpu.primitives.grid_controls import pdl_launch_attributes
from std.algorithm.functional import (
    tile_and_unswitch,
    unswitch,
)

from max.algorithm.functional import (
    _elementwise_impl_gpu,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    thread_idx,
    block_idx,
    global_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import (
    DeviceContext,
    FuncAttribute,
    get_gpu_target,
    DeviceBuffer,
    Dim as LaunchDim,
)
from max.gpu.host.info import A100, H100, B200, _is_sm10x_gpu
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_all,
    external_memory,
)
from kv_cache.types import KVCacheT
from layout import (
    Coord,
    Idx,
    IntTuple,
    LayoutTensor,
    RowMajorLayout,
    RuntimeLayout,
    RuntimeTuple,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from layout.layout import *
from layout.layout_tensor import (
    LayoutTensorIter,
    copy_dram_to_sram_async,
    copy_local_to_dram,
    copy_local_to_shared,
    copy_sram_to_dram,
)
from layout.swizzle import make_swizzle
from layout.tensor_core import get_fragment_size, get_mma_shape
from layout.tile_tensor import NullableTileTensor
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from linalg.matmul.gpu._multistage_gemm_gpu import multistage_mma
from std.memory import unsafe_stack_allocation
from nn._ragged_utils import get_batch_from_row_offsets
from nn.attention.mha_mask import MHAMask, TileMaskStatus
from nn.attention.mha_operand import (
    KVCacheMHAOperand,
    MHAOperand,
    LayoutTensorMHAOperand,
    RaggedMHAOperand,
)
from nn.attention.mha_utils import (
    FlashAttentionAlgorithm,
    MHAConfig,
    _copy_frag_to_smem,
    _kernel_mask,
    DynamicInt,
)
from max.runtime.tracing import Trace, TraceLevel, trace_arg

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from nn.attention.mha_utils import get_start_and_end_for_partitions
from nn.softmax import (
    _exp2_concrete,
    _exp_concrete,
    _online_softmax_iter_for_mma_output,
)
from .amd_structured.mla_decode import Attention
from .amd_structured.mla_prefill import Attention
from .nvidia.sm100.mla_prefill import mla_sm100_prefill
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    mla_decode_sm100_dispatch,
)
from .nvidia.sm100.mla_prefill_per_token_scale import (
    mla_sm100_prefill_per_token_scale,
)


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) decoding implementations
# ===-----------------------------------------------------------------------===#


# Max query tokens per sequence (S) the decode branch handles (MTP) —
# the coarse graph-routing ceiling (`mla_graph.mojo` decode-vs-prefill
# split). The precise per-call gate is `AMD_MLA_DECODE_FOLD_M_MAX` below. AMD
# folds H*S query rows into the MMA M dimension up to M=128, with S independently
# capped at this value. At num_heads=16 the two caps coincide (16*8==128); for
# num_heads<16 the S<=8 cap binds first (e.g. H=8 reaches M=64 at S=8), so the
# chokepoint enforces both. Under-routing is unsafe: it sends S>1 to the MLA
# prefill kernel, which can't read the absorbed-latent decode KV cache (NaN +
# OOB). FP8-only; BF16 S>1 is rejected at the chokepoint.
comptime MLA_DECODE_MAX_SEQ_LEN = 8


# AMD MLA decode S>1 (MTP) token-fold is wired only in the num_heads<=16 dispatch
# arm of `flare_mla_decoding_dispatch`; the larger-head arms hardcode q_seq_len=1.
# Both the host-side fold gate and the `mla_graph` decode-vs-prefill router key on
# this, so an S>1 batch with num_heads>16 (e.g. DP-attention Kimi: 64 heads/rank)
# routes to MLA prefill instead of the nonexistent large-head decode fold.
comptime AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS = 16


@always_inline
def mla_decode_max_seq_len[dtype: DType, num_heads: Int]() -> Int:
    """Max query tokens (S) the MLA *decode* branch can fold for this config.

    On AMD the S>1 fold is FP8-only AND num_heads<=AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS
    only (the larger-head dispatch arm is S=1-only), so a BF16 cache or num_heads>16
    routes S>1 to MLA prefill; NVIDIA folds up to `MLA_DECODE_MAX_SEQ_LEN` for both.
    Mirrors the host-side fold gate in `flare_mla_decoding_dispatch` so the router
    never hands the gate an S>1 batch it would reject.

    Parameters:
        dtype: Element type of the query tensor. On AMD the S>1 fold is
            FP8-only, so non-FP8 routes S>1 to MLA prefill.
        num_heads: Number of query attention heads. On AMD the S>1 fold
            requires `num_heads <= AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS`.
    """
    return 1 if (
        has_amd_gpu_accelerator()
        and (
            not dtype.is_float8()
            or num_heads > AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS
        )
    ) else MLA_DECODE_MAX_SEQ_LEN


# Precise AMD MLA-decode token-fold cap: M = num_heads * S must be <= this for
# the warp-local geometry (one 16-row MFMA M-tile per warp, BM = align16(M), up
# to 8 warps). Enforced by a host-side `raise` (active in DEFAULT builds). At
# BM=128 total LDS is ~89KB < the 160KB gfx950 limit (KV SMEM ~72KB is
# BM-independent, Q is register-resident).
#
# Occupancy: BM <= 64 fits 1 block/CU at ~510 VGPR with no spill; BM >= 80 forces
# >= 2 waves/SIMD, capping VGPR at 256 so the working set spills (~1KB).
# Correctness-neutral. A register-resident-P path would relieve this.
comptime AMD_MLA_DECODE_FOLD_M_MAX = 128


# entrypoint for MLA decoding kernels
# Disable nested-origin exclusivity checking: MLA decode reads (does not write)
# the paged KV cache, but the cache's buffer views and the mutable `output`
# carry tracked origins the exclusivity checker cannot prove disjoint, so it
# rejects them as separately-writable arguments. They are distinct allocations,
# so the check is a false positive here (proper fix: give the cache views
# provably-disjoint origins instead of sharing the collection's).
@__unsafe_nested_origins_read_only
@always_inline
def flare_mla_decoding[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    //,
    config: MHAConfig[dtype],
    ragged: Bool = False,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: True selects the BF16-rope sparse kernel
    # (split FP8 nope + BF16 rope). False (default) selects the all-FP8
    # sparse kernel. This is the production default; True is only used by
    # internal BF16-rope-kernel tests.
    rope_aware_kv_sparse: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141): set True (compile-time, from
    # the model's index_share signal) when the folded q positions share one
    # identical topk list, so the sparse fp8 decode gathers it ONCE. False
    # (default) -> unchanged per-position behavior.
    fold_shared_index: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
    q_max_seq_len: OptionalReg[Int] = None,
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
    num_partitions: Optional[Int] = None,
    # Per-token Q scale pointer: float32 array with one scale per Q token.
    # sigma_Q[q_token_idx] is folded into scale_log2e inside the Softmax function.
    # Default is null (sigma_Q = 1.0, no effect).
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Sparse indices: when non-null, the kernel uses gather4 TMA with
    # pre-computed physical row indices instead of page-table lookups.
    # d_indices[batch * indices_stride + token] = physical KV row index.
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    # Per-batch topk lengths: when non-null, topk_lengths[batch_idx] gives
    # the actual number of valid sparse indices for that batch. indices_stride
    # is the allocation stride (max topk across all batches).
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Extra KV: separate always-attend cache. Tokens from extra_k are
    # appended after the topk tokens in a unified attention loop.
    extra_k: OptionalReg[cache_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar from the Python resolver. When set, the
    # SM100 dispatcher uses this instead of recomputing num_partitions
    # at grid time.
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """MLA decoding kernel that would only be called in the optimized compute
    graph.

    The Q input has a shape of [seq_len, num_heads, depth].
    The K input has a shape of [seq_len, 1, depth].
    The V tensor is derived by reusing K, where V = K[:, :, :depth_v].

    Specifically, for DeepSeek V2/3, depth = 576 and depth_v = 512.

    When per_token_scale_rope_aware is True, Q and KV cache have an interleaved
    FP8+BF16 layout: FP8 content (512 bytes) + BF16 rope (128 bytes) = 640
    bytes/row. Q's last dimension is 640 (FP8 elements) but represents 576
    logical dimensions (512 nope + 64 rope).

    This kernel computes attention without needing to load V twice. This kernel
    only handles decoding requests. In this case q_max_seq_len = 1.

    This kernel handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.

    Parameters:
        rank: Tensor rank of `q` and `output` (inferred). Must be 4 for
            padded inputs or 3 for `ragged` inputs.
        cache_t: Paged KV cache type backing `k` (inferred). Carries the
            KV layout, dtype, and head geometry.
        mask_t: Mask functor type applied to attention scores (inferred).
        dtype: Element type of `q` (inferred). Must be a half-float or
            `float8_e4m3fn`.
        config: MLA attention config carrying `num_heads` and `depth`
            (576 for DeepSeek V2/3).
        ragged: Whether `q` is a ragged rank-3 tensor with batch offsets
            in `valid_length` instead of a padded rank-4 tensor (defaults
            to `False`).
        decoding_warp_split_k: Whether to use warp-level split-K
            reduction within the decode kernel (defaults to `False`).
        per_token_scale_rope_aware: Whether `q` and the KV cache use the
            interleaved FP8+BF16 rope-aware layout (640 bytes/row, 576
            logical dims) (defaults to `False`).
        sparse: Whether to use sparse attention with pre-computed
            physical KV row indices via gather4 TMA (defaults to
            `False`).
        rope_aware_kv_sparse: Sparse-only routing flag: `True` selects
            the BF16-rope sparse kernel (split FP8 nope + BF16 rope);
            `False` selects the all-FP8 sparse kernel (defaults to
            `False`).
        fold_shared_index: Whether to use the read-once shared-index fold
            that packs folded output/LSE slots into one CTA (defaults to
            `False`).

    Args:
        output: Output tensor with shape `[batch, num_heads, depth_v]`
            (or ragged equivalent). Dtype is `bfloat16` when `q` is
            `float8_e4m3fn`, else matches `q`.
        q: Query tensor. Padded rank-4 shape
            `[batch, q_seq_len, num_heads, depth]` (or rank-3 ragged).
            For `per_token_scale_rope_aware`, the last dim is 640 FP8
            elements representing 576 logical dims.
        k: Paged KV cache operand. V is derived as `K[:, :, :depth_v]`,
            so V is not loaded separately.
        mask_functor: Mask functor instance applied to attention scores.
        valid_length: Per-batch `uint32` tensor of valid (pre-padding)
            sequence lengths. For ragged inputs, carries cumulative row
            offsets.
        scale: Softmax scale factor applied to QK^T.
        ctx: Device context used to enqueue the kernel.
        scalar_args_buf: Optional GPU buffer of pre-computed scalar
            dispatch args for capturable graph launches; null for the
            legacy host-computed path.
        q_max_seq_len: Maximum query sequence length (query tokens per
            batch). Defaults to the cache's `max_prompt_length` when
            `None`.
        kv_input_row_offsets: Optional per-batch row offsets into a
            ragged KV layout; `None` for non-ragged inputs.
        num_partitions: Optional explicit split-K partition count;
            `None` selects via heuristic.
        q_scale_ptr: Optional per-token Q scale array (`float32`); folded
            into the softmax as `sigma_Q[token_idx]`. `None` means scale
            1.0.
        d_indices: Optional sparse indices: `d_indices[batch *
            indices_stride + token]` gives the physical KV row index.
            `None` for non-sparse.
        indices_stride: Allocation stride (max topk across all batches)
            for `d_indices` (defaults to 0).
        topk_lengths: Optional per-batch array of actual valid sparse
            index counts; `None` for non-sparse.
        attn_sink_ptr: Optional attention-sink scale pointer
            (`float32`); `None` to disable.
        extra_k: Optional separate always-attend KV cache operand,
            appended after the topk tokens in the attention loop; `None`
            to disable.
        extra_d_indices: Optional sparse indices for `extra_k`, same
            layout as `d_indices`; `None` for non-sparse extra KV.
        extra_indices_stride: Allocation stride for `extra_d_indices`
            (defaults to 0).
        extra_topk_lengths: Optional per-batch valid index counts for
            `extra_k`; `None` for non-sparse extra KV.
        extra_scales_ptr: Optional per-token scale pointer for
            `extra_k`; `None` to disable.
        num_partitions_in: Optional capturable-graph scalar from the
            Python resolver; when set, the SM100 dispatcher uses it
            instead of recomputing `num_partitions` at grid time.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
    """
    comptime assert (
        ragged or rank == 4
    ), "only support rank 4 inputs for non-ragged inputs."
    comptime assert (
        not ragged or rank == 3
    ), "only support rank 3 inputs for ragged inputs."
    # Q and output may differ for native FP8 path: Q is float8_e4m3fn,
    # output is bfloat16. Both half-float Q (bfloat16) and FP8 Q are valid.
    comptime assert q.dtype == output.dtype or (
        q.dtype == .float8_e4m3fn and output.dtype == .bfloat16
    ), (
        "Q and output must have same type, or Q=float8_e4m3fn with"
        " output=bfloat16."
    )

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_decoding",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        comptime kv_num_heads = cache_t.kv_params.num_heads

        var max_prompt_len: Int
        var num_keys = Int(k.max_context_length())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k.max_prompt_length())

        var k_operand = KVCacheMHAOperand(k)

        # For per_token_scale_rope_aware: Q's last dim is 640 (interleaved FP8+BF16)
        # but the logical depth is 576. Override config to use 576.
        comptime if per_token_scale_rope_aware:
            comptime rope_aware_config = MHAConfig[dtype](config.num_heads, 576)
            # Build extra_k_operand when extra_k is provided.
            var extra_k_operand: OptionalReg[type_of(k_operand)] = None
            if extra_k is not None:
                extra_k_operand = KVCacheMHAOperand(extra_k.value())

            flare_mla_decoding_dispatch[
                kv_num_heads=kv_num_heads,
                config=rope_aware_config,
                ragged=ragged,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=True,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                output,
                q,
                k_operand,
                mask_functor,
                valid_length,
                max_prompt_len,
                num_keys,
                scale,
                ctx,
                scalar_args_buf,
                kv_input_row_offsets,
                num_partitions,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k_operand,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                num_partitions_in=num_partitions_in,
                logical_indices=logical_indices,
            )
        else:
            # Build extra_k_operand when extra_k is provided.
            var extra_k_operand: OptionalReg[type_of(k_operand)] = None
            if extra_k is not None:
                extra_k_operand = KVCacheMHAOperand(extra_k.value())

            flare_mla_decoding_dispatch[
                kv_num_heads=kv_num_heads,
                config=config,
                ragged=ragged,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=False,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                output,
                q,
                k_operand,
                mask_functor,
                valid_length,
                max_prompt_len,
                num_keys,
                scale,
                ctx,
                scalar_args_buf,
                kv_input_row_offsets,
                num_partitions,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k_operand,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                num_partitions_in=num_partitions_in,
                logical_indices=logical_indices,
            )


# entrypoint for TileTensor as K input, used by tests and benchmarks.
def flare_mla_decoding[
    mask_t: MHAMask,
    dtype: DType,
    //,
    config: MHAConfig[dtype],
    decoding_warp_split_k: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: TileTensor[address_space=.GENERIC, ...],
    mask_functor: mask_t,
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
    # if not set, we select num_partitions based on heuristics
    num_partitions: Optional[Int] = None,
) raises:
    comptime assert q.rank == 4, "only support rank 4 inputs."

    comptime kv_num_heads = type_of(k).static_shape[2]

    # Runtime dimensions.
    var num_keys = Int(k.dim[1]())

    # Re-view `k` as a row-major TileTensor directly (no throwaway
    # LayoutTensor round-trip). `shape_coord()` preserves the static/runtime
    # dim types, and the operand infers `buffer_layout` from this TileTensor.
    var k_operand = LayoutTensorMHAOperand(
        TileTensor(k.ptr, row_major(k.layout.shape_coord()))
    )

    var valid_length = TileTensor(
        UnsafePointer[UInt32, MutUntrackedOrigin].unsafe_dangling(),
        row_major(Coord(Idx[0])),
    )

    flare_mla_decoding_dispatch[
        kv_num_heads=kv_num_heads,
        config=config,
        ragged=False,
        _is_cache_length_accurate=True,
        _use_valid_length=False,
        decoding_warp_split_k=decoding_warp_split_k,
    ](
        output,
        q,
        k_operand,
        mask_functor,
        valid_length,
        Int(q.dim[1]()),
        num_keys,
        scale,
        ctx,
        scalar_args_buf,
        kv_input_row_offsets=None,
        num_partitions=num_partitions,
    )


@always_inline
def flare_mla_decoding_dispatch[
    k_t: MHAOperand,
    mask_t: MHAMask,
    dtype: DType,
    //,
    kv_num_heads: Int,
    config: MHAConfig[dtype],
    ragged: Bool = False,
    # Work arounds to unify KVCache and TileTensor inputs:
    # Differentiate two cases, KV cache's length is before adding the latest
    # tokens e.g. zero for CE, and KV TileTensor's length is the latest length
    # e.g. prompt length for CE.
    _is_cache_length_accurate: Bool = False,
    # valid_length is needed for KV cache inputs and is empty for TileTensor
    # inputs to avoid overhead in benchmark.
    _use_valid_length: Bool = True,
    decoding_warp_split_k: Bool = False,
    per_token_scale_rope_aware: Bool = False,
    sparse: Bool = False,
    # Sparse-only routing flag: True selects the BF16-rope sparse kernel,
    # False (default) selects the all-FP8 sparse kernel.
    rope_aware_kv_sparse: Bool = False,
    # Read-once shared-index MTP fold (KERN-3141); see flare_mla_decoding.
    fold_shared_index: Bool = False,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: k_t,
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    max_prompt_len: Int,
    max_cache_valid_length: Int,
    scale: Float32,
    ctx: DeviceContext,
    scalar_args_buf: NullableTileTensor[
        DType.int64, address_space=.GENERIC, ...
    ],
    kv_input_row_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin]
    ] = None,
    num_partitions: Optional[Int] = None,
    q_scale_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    indices_stride: Int = 0,
    topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    attn_sink_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Extra KV: separate always-attend cache operand.
    extra_k: OptionalReg[k_t] = None,
    extra_d_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_indices_stride: Int = 0,
    extra_topk_lengths: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
    extra_scales_ptr: OptionalReg[UnsafePointer[Float32, MutAnyOrigin]] = None,
    # Capturable-graph scalar: forwarded by the Python resolver so grid-time
    # dispatch matches the kernel's device-side divmod.
    num_partitions_in: Optional[Int] = None,
    # Logical sparse indices for position-based causal masking; `None` keeps
    # the prior slot-count behavior. See mla_decode_utils.mojo.
    logical_indices: OptionalReg[UnsafePointer[Int32, MutAnyOrigin]] = None,
) raises:
    """Dispatches an MLA decoding request to the platform-specific kernel.

    Routes to the SM100 (`mla_decode_sm100_dispatch`), AMD, or legacy NVIDIA
    decode path based on the target GPU, selects the block-M tile geometry
    (including the AMD MTP token-fold and BM=32/64 heuristic), and launches the
    `mla_decoding` kernel with optional split-K reduction via
    `mla_splitk_reduce`.

    Parameters:
        k_t: KV cache operand type backing `k` (inferred). Either
            `KVCacheMHAOperand` (paged cache) or
            `LayoutTensorMHAOperand` (TileTensor input).
        mask_t: Mask functor type applied to attention scores (inferred).
        dtype: Element type of `q` (inferred). Must be a half-float or
            `float8_e4m3fn`.
        kv_num_heads: Number of KV attention heads. Must be 1 for MLA.
        config: MLA attention config carrying `num_heads` and `depth`
            (576 for DeepSeek V2/3).
        ragged: Whether `q` is a ragged rank-3 tensor with batch offsets
            in `valid_length` instead of a padded rank-4 tensor (defaults
            to `False`).
        _is_cache_length_accurate: Workaround unifying KVCache and
            TileTensor inputs: `True` when `max_cache_valid_length` is the
            accurate latest length (TileTensor case); `False` when it
            precedes the latest tokens, for example zero for continuous
            execution (KV cache case) (defaults to `False`).
        _use_valid_length: Whether `valid_length` is needed for masking;
            `False` skips it for TileTensor inputs to avoid benchmark
            overhead (defaults to `True`).
        decoding_warp_split_k: Whether to use warp-level split-K
            reduction within the decode kernel (defaults to `False`).
        per_token_scale_rope_aware: Whether `q` and the KV cache use the
            interleaved FP8+BF16 rope-aware layout (640 bytes/row, 576
            logical dims) (defaults to `False`).
        sparse: Whether to use sparse attention with pre-computed
            physical KV row indices via gather4 TMA (defaults to
            `False`).
        rope_aware_kv_sparse: Sparse-only routing flag: `True` selects
            the BF16-rope sparse kernel (split FP8 nope + BF16 rope);
            `False` selects the all-FP8 sparse kernel (defaults to
            `False`).
        fold_shared_index: Whether to use the read-once shared-index fold
            that packs folded output/LSE slots into one CTA (defaults to
            `False`).

    Args:
        output: Output tensor with shape `[batch, num_heads, depth_v]`
            (or ragged equivalent). Dtype is `bfloat16` when `q` is
            `float8_e4m3fn`, else matches `q`.
        q: Query tensor. Padded rank-4 shape
            `[batch, q_seq_len, num_heads, depth]` (or rank-3 ragged).
            For `per_token_scale_rope_aware`, the last dim is 640 FP8
            elements representing 576 logical dims.
        k: KV cache operand. V is derived as `K[:, :, :depth_v]`, so V is
            not loaded separately.
        mask_functor: Mask functor instance applied to attention scores.
        valid_length: Per-batch `uint32` tensor of valid (pre-padding)
            sequence lengths. For ragged inputs, carries cumulative row
            offsets.
        max_prompt_len: Maximum query sequence length (query tokens per
            batch); the MTP token-fold ceiling.
        max_cache_valid_length: Total number of cached KV entries (the
            cache context length).
        scale: Softmax scale factor applied to QK^T.
        ctx: Device context used to enqueue the kernel.
        scalar_args_buf: Optional GPU buffer of pre-computed scalar
            dispatch args for capturable graph launches; null for the
            legacy host-computed path.
        kv_input_row_offsets: Optional per-batch row offsets into a
            ragged KV layout; `None` for non-ragged inputs.
        num_partitions: Optional explicit split-K partition count;
            `None` selects via heuristic.
        q_scale_ptr: Optional per-token Q scale array (`float32`); folded
            into the softmax as `sigma_Q[token_idx]`. `None` means scale
            1.0.
        d_indices: Optional sparse indices: `d_indices[batch *
            indices_stride + token]` gives the physical KV row index.
            `None` for non-sparse.
        indices_stride: Allocation stride (max topk across all batches)
            for `d_indices` (defaults to 0).
        topk_lengths: Optional per-batch array of actual valid sparse
            index counts; `None` for non-sparse.
        attn_sink_ptr: Optional attention-sink scale pointer
            (`float32`); `None` to disable.
        extra_k: Optional separate always-attend KV cache operand,
            appended after the topk tokens in the attention loop; `None`
            to disable.
        extra_d_indices: Optional sparse indices for `extra_k`, same
            layout as `d_indices`; `None` for non-sparse extra KV.
        extra_indices_stride: Allocation stride for `extra_d_indices`
            (defaults to 0).
        extra_topk_lengths: Optional per-batch valid index counts for
            `extra_k`; `None` for non-sparse extra KV.
        extra_scales_ptr: Optional per-token scale pointer for
            `extra_k`; `None` to disable.
        num_partitions_in: Optional capturable-graph scalar from the
            Python resolver; when set, the SM100 dispatcher uses it
            instead of recomputing `num_partitions` at grid time.
        logical_indices: Logical sparse indices for position-based causal
            masking; `None` keeps the prior slot-count behavior.
    """
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // kv_num_heads
    comptime assert num_heads == type_of(q).static_shape[q.rank - 2]

    # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
    comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

    # For per_token_scale_rope_aware: Q's physical last dim is 640 (interleaved
    # FP8+BF16) but the logical depth (from config) is 576. Only validate
    # the config depth; the Q physical dim is checked separately.
    comptime if per_token_scale_rope_aware:
        comptime assert (
            depth == 576
        ), "per_token_scale_rope_aware requires logical depth == 576."
        comptime assert (
            type_of(q).static_shape[q.rank - 1] == 640
        ), "per_token_scale_rope_aware requires Q physical dim == 640."
    else:
        # 576 = 512 latent + 64 rotary. A NoPE model stores the 512 latent alone.
        # The kernels keep SMEM at 576 and zero-fill the tail, so both rows run.
        comptime assert depth == type_of(q).static_shape[q.rank - 1] and (
            depth == 576 or depth == 512
        ), (
            "flareMLA_decoding only supports head_dim 576 (rotary) or 512"
            " (NoPE)."
        )
        # The mixed BF16-Q / FP8-KV converter narrowed its TMA widths but not
        # the byte counts its barriers wait for, so a 512 row hangs rather
        # than returns wrong. Refuse it here while the combination is legible.
        comptime assert (
            depth == 576 or q.dtype != .bfloat16 or k_t.dtype == .bfloat16
        ), (
            "the NoPE 512-wide row is supported with a BF16 KV cache (BF16 Q)"
            " or with an all-FP8 QKV cache. The BF16-Q / FP8-KV combination"
            " still expects the 576-wide rotary row"
        )
    comptime assert (
        kv_num_heads == 1
    ), "flareMLA_decoding only supports kv_num_heads == 1."
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_decoding currently only supports Nvidia and AMD GPUs."

    comptime assert (
        q.dtype.is_half_float() or q.dtype == .float8_e4m3fn
    ), "Only support half precision or float8_e4m3fn Q."

    # AMD MLA decode folds H*S query rows into the QK^T MMA M dimension up to
    # M = num_heads * S <= 128 (legacy (1,4) for M <= 16, else warp-local
    # (M/16, 1)). S=1 is always supported (large-H S=1 uses the multi-tile grid.y
    # path, not the fold).
    #
    # The host-side `raise` below is the only gate (the dispatch fn is `raises`,
    # so it fires in every build). A `debug_assert` backstop would be wrong: under
    # `-D ASSERT=all` it aborts before the catchable `raise` and breaks the
    # `assert_raises` unsupported-fold tests.
    comptime if has_amd_gpu_accelerator():
        # Hard launch-time rejection, and the single source of truth for the
        # fold envelope. Supported: S == 1 (any), or S > 1 with FP8 Q, num_heads
        # <= 16, S <= MLA_DECODE_MAX_SEQ_LEN, and num_heads*S <= 128. The other
        # arms would silently downgrade S>1 to S=1 and drop tokens: the BF16 arm
        # and the num_heads>16 arms both hardcode q_seq_len=1 (only the
        # FP8/num_heads<=16 arm threads it). Routing S>1 to MLA prefill is also
        # unsafe (it can't read the absorbed-latent decode KV cache → NaN + OOB),
        # so raise loudly. The S<=MLA_DECODE_MAX_SEQ_LEN term matters for
        # num_heads<16, where num_heads*S<=128 alone would admit S>8 (e.g. H=8,
        # S=10 → M=80); without it such a call falls through to the dispatch
        # ladder's backstop raise instead of failing here.
        if max_prompt_len > 1 and (
            not dtype.is_float8()
            or num_heads > AMD_MLA_DECODE_FOLD_MAX_NUM_HEADS
            or max_prompt_len > MLA_DECODE_MAX_SEQ_LEN
            or num_heads * max_prompt_len > AMD_MLA_DECODE_FOLD_M_MAX
        ):
            raise Error(
                t"AMD MLA decode (MTP) supports S=1 (any num_heads), or S>1"
                t" with FP8 Q, num_heads <= 16, S <= {MLA_DECODE_MAX_SEQ_LEN},"
                t" and num_heads*S <= {AMD_MLA_DECODE_FOLD_M_MAX}."
                t" Requested S={max_prompt_len} with num_heads={num_heads}"
                t" (M = num_heads*S = {num_heads * max_prompt_len}) is not"
                t" supported."
            )

        # Variable per-sequence query length is supported: the comptime
        # `q_seq_len` (= `max_prompt_len`) is only the padding ceiling, while the
        # runtime length `valid_length[b+1] - valid_length[b]` drives the SRD
        # clamp, causal offset, split-K stat writer, and ragged reducer — so
        # short sequences' pad rows are dropped. Uniform batches are
        # byte-equivalent (seq_len == q_seq_len).

    # TileTensor always has static shapes for the last two dims.

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        if scalar_args_buf.ptr:
            # Capturable path: GPU buffer is pre-computed, compute host-side
            # dispatch args from inputs.
            var batch_size: Int
            comptime if ragged:
                batch_size = Int(valid_length.dim[0]()) - 1
            else:
                batch_size = Int(q.dim(0))
            if batch_size == 0:
                return
            mla_decode_sm100_dispatch[
                q.dtype,
                k_t,
                output.dtype,
                mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                q,
                k,
                output,
                scale,
                valid_length,
                mask_functor,
                scalar_args_buf.value(),
                batch_size,
                max_prompt_len,
                max_cache_valid_length,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
                num_partitions_in=num_partitions_in,
                logical_indices=logical_indices,
            )
        else:
            # Legacy path: compute dispatch params and GPU buffer from inputs.
            var batch_size: Int
            comptime if ragged:
                batch_size = Int(valid_length.dim[0]()) - 1
            else:
                batch_size = Int(q.dim(0))
            if batch_size == 0:
                return

            comptime num_heads_val = type_of(q).static_shape[q.rank - 2]
            comptime _is_fp8_kv = (k_t.dtype == DType.float8_e4m3fn)
            var local_args = MLADispatchScalarArgs[
                num_heads=num_heads_val,
                _is_cache_length_accurate=_is_cache_length_accurate,
                is_fp8_kv=_is_fp8_kv,
                fold_shared_index=fold_shared_index,
            ](batch_size, max_cache_valid_length, max_prompt_len, ctx)
            mla_decode_sm100_dispatch[
                q.dtype,
                k_t,
                output.dtype,
                mask_t,
                config=config,
                depth=depth,
                num_heads=num_heads,
                group=group,
                ragged=ragged,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=decoding_warp_split_k,
                per_token_scale_rope_aware=per_token_scale_rope_aware,
                sparse=sparse,
                rope_aware_kv_sparse=rope_aware_kv_sparse,
                fold_shared_index=fold_shared_index,
            ](
                q,
                k,
                output,
                scale,
                valid_length,
                mask_functor,
                TileTensor(
                    rebind[UnsafePointer[Int64, origin=MutAnyOrigin]](
                        local_args.gpu_buf.unsafe_ptr()
                    ),
                    row_major[3](),
                ),
                local_args.batch_size,
                local_args.q_max_seq_len,
                max_cache_valid_length,
                ctx,
                q_scale_ptr,
                d_indices,
                indices_stride,
                topk_lengths,
                attn_sink_ptr,
                extra_k=extra_k,
                extra_d_indices=extra_d_indices,
                extra_indices_stride=extra_indices_stride,
                extra_topk_lengths=extra_topk_lengths,
                extra_scales_ptr=extra_scales_ptr,
            )
            _ = local_args^

    else:
        var batch_size: Int
        comptime if ragged:
            batch_size = Int(valid_length.dim[0]()) - 1
        else:
            batch_size = Int(q.dim(0))

        if batch_size == 0:
            return

        # only A100 or H100 have the enough smem to store the full BM * head_dim Q tensor.
        comptime has_enough_smem = ctx.default_device_info == A100 or ctx.default_device_info == H100

        # AMD FP8 MLA decode uses 32x32x64 MMA (depth=576 is not a multiple
        # of 128, so the 16x16x128 path is unusable). That requires BM>=32
        # and BK>=64. BF16 AMD MLA keeps BM=16, BK=32 (16x16x32 MMA).
        # For amd_fp8, BM=64 packs all 64 q-heads into one block per
        # (batch, partition), halving K DRAM reads vs BM=32 (which splits
        # heads across 2 blocks that each re-read the same K tile). The
        # win is real only when K traffic actually overflows L2 AND the
        # kernel reaches the bandwidth-bound regime; otherwise BM=64's 2×
        # per-block compute (more MFMAs, more SMEM round-trip) loses out.
        # See heuristic dispatch below.
        comptime amd_fp8 = has_amd_gpu_accelerator() and q.dtype.is_float8()

        @always_inline
        @__parameter
        def launch_with_BM[
            BM: Int,
            q_seq_len: Int = 1,
            WM: Int = BM,
            WN: Int = (16 if has_nvidia_gpu_accelerator() else 32),
        ]() raises:
            # `q_seq_len` (S) folds H*S query rows into the MMA M dimension.
            # Default 1 = single-token decode. `WM`/`WN` default to the legacy
            # (1,4) geometry (WM=BM, WN=32 AMD / 16 NVIDIA), so S=1 call sites are
            # byte-identical; warp-local passes `WM=16, WN=128` (num_warps_m=
            # BM//16, num_warps_n=1, one 16-row tile per warp).
            comptime BN = 64 if has_nvidia_gpu_accelerator() else 128
            # AMD-structured config picks 16x16x128 for MLA decode when
            # `num_heads <= 16` (Kimi-K2.5 TP=4) or `depth % 128 == 0`;
            # both need BK=128 so each MFMA call consumes 128 elements
            # of depth (depth=576 → 4 full tiles + 1 partial tile
            # zero-padded in registers).
            comptime amd_fp8_16x16x128 = amd_fp8 and (
                num_heads <= 16 or depth % 128 == 0
            )
            comptime BK = 128 if amd_fp8_16x16x128 else (
                64 if (has_nvidia_gpu_accelerator() or amd_fp8) else 32
            )  # 8 mma_tile per row resolves bank conflict on nvidia
            # num warps in M and N, multiplied by warp size.
            comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

            comptime accum_type = get_accum_type[q.dtype]()
            comptime num_pipeline_stages = 6
            # smem for q
            var shared_mem_bytes = BM * depth * size_of[q.dtype]()

            shared_mem_bytes += BN * depth * size_of[k_t.dtype]()

            comptime num_warps = ceildiv(num_threads, WARP_SIZE)

            # smem for p and warp_scratch
            shared_mem_bytes += (
                BM * BN * size_of[k_t.dtype]()
                + 2 * num_warps * BM * size_of[accum_type]()
            )

            shared_mem_bytes = (
                shared_mem_bytes if has_nvidia_gpu_accelerator() else 0
            )

            # M-based, not num_heads-based: the fold makes M = num_heads *
            # q_seq_len, so grid.y must cover all H*S rows. At S=1 this is
            # `ceildiv(num_heads, BM)`, byte-identical (the S=1 large-H path keeps
            # grid.y>1).
            comptime _fold_m = num_heads * q_seq_len
            comptime num_blocks_y = ceildiv(_fold_m, BM)
            comptime assert BM % 16 == 0
            # The S>1 fold packs all H*S rows into one block's M tile, so M must
            # fit BM (= align16(M) for warp-local).
            comptime assert (q_seq_len == 1) or (_fold_m <= BM), (
                "S>1 requires M = num_heads*q_seq_len <= BM (the folded query"
                " rows must fit the block M tile)."
            )

            comptime depth_v = type_of(output).static_shape[output.rank - 1]

            comptime kernel = mla_decoding[
                q.dtype,
                k_t,
                output.dtype,
                mask_t,
                type_of(valid_length).LayoutType,
                BM=BM,
                BN=BN,
                BK=BK,
                WM=WM,
                WN=WN,
                depth=depth,
                depth_v=depth_v,
                num_heads=num_heads,
                num_threads=num_threads,
                num_pipeline_stages=num_pipeline_stages,
                group=group,
                ragged=ragged,
                _use_valid_length=_use_valid_length,
                _is_cache_length_accurate=_is_cache_length_accurate,
                decoding_warp_split_k=decoding_warp_split_k,
                q_seq_len=q_seq_len,
            ]

            # Pick num_partitions for split-K. Only AMD has a tuned heuristic
            # today; non-SM10x NVIDIA stays at 1 to preserve existing behavior.
            # Partition count is independent of BM; both BM variants share it.
            var num_partitions_value: Int
            if num_partitions:
                num_partitions_value = num_partitions.value()
            else:
                comptime if has_amd_gpu_accelerator():
                    # MLA: kv_num_heads == 1, so heads_per_group == num_heads.
                    num_partitions_value = mha_decoding_num_partitions(
                        batch_size,
                        max_cache_valid_length,
                        num_heads,
                        ctx,
                        is_mla=True,
                    )
                else:
                    num_partitions_value = 1

            # Token fold (S>1): split-K is re-keyed by query row, so the
            # intermediate + stat workspaces are sized by `_split_k_rows =
            # num_heads * q_seq_len` (= M, heads-inner: row = token*H + head),
            # `store_partition_info` writes one stat per row, and the reducer runs
            # one CTA per row. At S=1 == num_heads, byte-identical.
            comptime _split_k_rows = num_heads * q_seq_len

            var q_device = DeviceBuffer[q.dtype](
                ctx, q.ptr, q.num_elements(), owning=False
            )
            var output_device = DeviceBuffer[output.dtype](
                ctx, output.ptr, output.num_elements(), owning=False
            )

            if num_partitions_value == 1:
                # Single-partition fast path — no intermediate buffers, no reduce.
                var nullptr_device = DeviceBuffer[accum_type].empty(ctx)
                ctx.enqueue_function[kernel](
                    q_device,
                    k,
                    output_device,
                    nullptr_device,
                    nullptr_device,
                    scale,
                    Int32(batch_size),
                    Int32(num_partitions_value),
                    Int32(max_cache_valid_length),
                    valid_length.as_immut(),
                    mask_functor,
                    grid_dim=(1, num_blocks_y, batch_size),
                    block_dim=(num_threads, 1, 1),
                    shared_mem_bytes=shared_mem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(
                            ctx.default_device_info.shared_memory_per_multiprocessor
                            - 4096
                        )
                    ),
                )
            else:
                # Split-K: per-partition output (depth_v) and softmax stats.
                comptime intermediate_dtype = output.dtype

                var output_intermediate_data = ctx.enqueue_create_buffer[
                    intermediate_dtype
                ](_split_k_rows * depth_v * batch_size * num_partitions_value)

                var data_len = _split_k_rows * batch_size * num_partitions_value
                var exp_sum_qk_max_data = ctx.enqueue_create_buffer[accum_type](
                    2 * data_len
                )
                var exp_sum_device = DeviceBuffer[accum_type](
                    ctx,
                    exp_sum_qk_max_data.unsafe_ptr(),
                    data_len,
                    owning=False,
                )
                var qk_max_device = DeviceBuffer[accum_type](
                    ctx,
                    exp_sum_qk_max_data.unsafe_ptr() + data_len,
                    data_len,
                    owning=False,
                )

                ctx.enqueue_function[kernel](
                    q_device,
                    k,
                    output_intermediate_data,
                    exp_sum_device,
                    qk_max_device,
                    scale,
                    Int32(batch_size),
                    Int32(num_partitions_value),
                    Int32(max_cache_valid_length),
                    valid_length.as_immut(),
                    mask_functor,
                    grid_dim=(
                        num_partitions_value,
                        num_blocks_y,
                        batch_size,
                    ),
                    block_dim=(num_threads, 1, 1),
                    shared_mem_bytes=shared_mem_bytes,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(
                            ctx.default_device_info.shared_memory_per_multiprocessor
                            - 4096
                        )
                    ),
                )

                # AMD softmax always uses exp2; CUDA non-FA3 path uses exp.
                comptime reduce_use_exp2 = has_amd_gpu_accelerator()
                comptime if has_amd_gpu_accelerator():
                    # W per kernel: target parts_per_warp = MAX_PARTITIONS/W
                    # = 8, the sweet spot for step-2 software pipelining
                    # (~HBM_latency / FMA_throughput loads in flight). So
                    # the 64-kernel uses W=8 and the 128-kernel uses W=16.
                    # D_TILES picks per num_heads: tiny grids
                    # (num_heads<=16, e.g. Kimi-K2.5 TP=4) need D=8 to
                    # spread CTAs; large grids (num_heads>=64) already
                    # saturate CUs at D=4, so going higher just adds
                    # launch overhead.
                    comptime D_TILES_DEFAULT = 8 if num_heads <= 16 else 4
                    comptime D_TILES = get_defined_int[
                        "MODULAR_MLA_REDUCE_DTILES", D_TILES_DEFAULT
                    ]()
                    comptime W_PARTS_64 = get_defined_int[
                        "MODULAR_MLA_REDUCE_WPARTS_64", 8
                    ]()
                    comptime W_PARTS_128 = get_defined_int[
                        "MODULAR_MLA_REDUCE_WPARTS_128", 16
                    ]()
                    # 256-partition reducer for launch-starved large-num_keys
                    # bs=1 decode (nk>=64K): W_PARTS=16 => 1024 threads/CTA
                    # (the block-dim ceiling), giving parts_per_warp=16 (2x
                    # the 8 sweet spot, but the np=128->256 decode-parallelism
                    # win dominates here — the decode grid was launch-starved
                    # at ~128 blocks / 256 CUs).
                    comptime W_PARTS_256 = get_defined_int[
                        "MODULAR_MLA_REDUCE_WPARTS_256", 16
                    ]()
                    # Two specializations: 64-kernel covers np <= 64
                    # (short context); 128-kernel covers np > 64 (long
                    # context). Picking per dispatch keeps O(MAX_PARTITIONS)
                    # per-CTA work proportional to the actual partition
                    # count bucket.
                    # `ragged` + input_row_offsets let the reducer remap the
                    # final store to the ragged `[total_q_tokens, H, depth]`
                    # layout and skip short-sequence pad rows. Comptime-dead at
                    # S=1 / non-ragged (byte-identical store).
                    comptime kernel_reduce_64 = mla_splitk_reduce[
                        intermediate_dtype,
                        output.dtype,
                        type_of(valid_length).LayoutType,
                        depth=depth_v,
                        num_heads=num_heads,
                        D_TILES=D_TILES,
                        W_PARTS=W_PARTS_64,
                        MAX_PARTITIONS=64,
                        use_exp2=reduce_use_exp2,
                        q_seq_len=q_seq_len,
                        ragged=ragged,
                    ]
                    comptime kernel_reduce_128 = mla_splitk_reduce[
                        intermediate_dtype,
                        output.dtype,
                        type_of(valid_length).LayoutType,
                        depth=depth_v,
                        num_heads=num_heads,
                        D_TILES=D_TILES,
                        W_PARTS=W_PARTS_128,
                        MAX_PARTITIONS=128,
                        use_exp2=reduce_use_exp2,
                        q_seq_len=q_seq_len,
                        ragged=ragged,
                    ]
                    comptime kernel_reduce_256 = mla_splitk_reduce[
                        intermediate_dtype,
                        output.dtype,
                        type_of(valid_length).LayoutType,
                        depth=depth_v,
                        num_heads=num_heads,
                        D_TILES=D_TILES,
                        W_PARTS=W_PARTS_256,
                        MAX_PARTITIONS=256,
                        use_exp2=reduce_use_exp2,
                        q_seq_len=q_seq_len,
                        ragged=ragged,
                    ]
                    if num_partitions_value <= 64:
                        ctx.enqueue_function[kernel_reduce_64](
                            output_intermediate_data,
                            output_device,
                            exp_sum_device,
                            qk_max_device,
                            Int32(batch_size),
                            Int32(num_partitions_value),
                            valid_length.as_immut(),
                            grid_dim=(D_TILES, _split_k_rows, batch_size),
                            block_dim=(W_PARTS_64 * WARP_SIZE, 1, 1),
                        )
                    elif num_partitions_value <= 128:
                        ctx.enqueue_function[kernel_reduce_128](
                            output_intermediate_data,
                            output_device,
                            exp_sum_device,
                            qk_max_device,
                            Int32(batch_size),
                            Int32(num_partitions_value),
                            valid_length.as_immut(),
                            grid_dim=(D_TILES, _split_k_rows, batch_size),
                            block_dim=(W_PARTS_128 * WARP_SIZE, 1, 1),
                        )
                    else:
                        ctx.enqueue_function[kernel_reduce_256](
                            output_intermediate_data,
                            output_device,
                            exp_sum_device,
                            qk_max_device,
                            Int32(batch_size),
                            Int32(num_partitions_value),
                            valid_length.as_immut(),
                            grid_dim=(D_TILES, _split_k_rows, batch_size),
                            block_dim=(W_PARTS_256 * WARP_SIZE, 1, 1),
                        )
                else:
                    comptime kernel_reduce = mha_splitk_reduce[
                        intermediate_dtype,
                        output.dtype,
                        depth=depth_v,
                        num_heads=num_heads,
                        num_threads=WARP_SIZE,
                        use_exp2=reduce_use_exp2,
                    ]
                    ctx.enqueue_function[kernel_reduce](
                        output_intermediate_data,
                        output_device,
                        exp_sum_device,
                        qk_max_device,
                        Int32(batch_size),
                        Int32(num_partitions_value),
                        grid_dim=(1, num_heads, batch_size),
                        block_dim=(WARP_SIZE, 1, 1),
                        attributes=pdl_launch_attributes(MHA_PDL_LEVEL),
                    )
                _ = exp_sum_qk_max_data^
                _ = output_intermediate_data^

        # ---- BM dispatch ----
        # AMD FP8: pick BM=32 vs BM=64 from a runtime predicate. Empirically
        # (MI355X, num_heads=64, see kern-2876 regression sweep), BM=64 wins
        # iff BOTH:
        #   1) cache_len >= 32K  — long enough for the kernel to reach the
        #      bandwidth-bound regime where K traffic dominates per-block
        #      compute. Below this, per-block setup/sync of BM=64's 2×
        #      MFMAs and 2× SMEM round-trip overwhelms the K-traffic savings
        #      regardless of total bytes (e.g. bs=128 cache=4096 is 302 MB
        #      yet still regresses -8% under BM=64).
        #   2) bs * cache_len * depth >= ~0.4 × L2  — total K bytes large
        #      enough that BM=32's redundant per-block_y K reads actually
        #      hit DRAM. When K fits in L2, the second block_y's reads are
        #      L2 hits and BM=32's "doubled" traffic is free.
        # Together these capture every meaningful win in the sweep
        # (+14%..+64%) with zero false positives. Thresholds tuned for the
        # MI355X (256 MB L2); generalizes by physical interpretation.
        # For num_heads <= 32, BM=32 already covers all heads in one m_mma,
        # so BM=64 has no head-packing benefit — keep BM=32.
        comptime if amd_fp8:
            # `num_heads <= 16` triggers the 16x16x128 MFMA shape (see
            # `AMDStructuredConfig.get_mma_shape`); pair it with BM=WM=16
            # so each warp packs one MFMA tile of valid heads (no m_mma=1
            # doing wasted work on OOB rows). Without this, BM=32 with
            # mma_shape[0]=16 gives num_m_mmas=2 and the second m_mma is
            # wasted for any num_heads <= 16.
            comptime if num_heads <= 16:
                # 16x16x128 MFMA. The runtime `max_prompt_len` (S) selects the
                # comptime `q_seq_len`, folding M = num_heads*S rows into the MMA
                # M dimension: legacy (1,4) for M <= 16, else warp-local (M/16, 1)
                # (BM=align16(M), WM=16, WN=128 — one 16-row tile per warp).
                # `get_mma_shape()` is 16x16x128 for any BM here, so WM=16 gives
                # num_m_mmas=1.
                #
                # The chokepoint `raise` already rejected M > 128, so each
                # supported S maps to its own kernel/q_seq_len — never downgraded.
                # `_s_max` is capped by both limits so no dead S kernel is built;
                # the trailing `raise` is a backstop if the chokepoint weakens.
                comptime _s_max = min(
                    AMD_MLA_DECODE_FOLD_M_MAX // num_heads,
                    MLA_DECODE_MAX_SEQ_LEN,
                )
                comptime for s in range(1, _s_max + 1):
                    if max_prompt_len == s:
                        comptime _m = num_heads * s
                        comptime if _m <= 16:
                            # legacy (1,4) — WM/WN default to BM/32.
                            launch_with_BM[16, s]()
                        else:
                            # warp-local (M/16, 1): one 16-row MFMA M-tile per
                            # warp over align16(M) rows (W = BM//16 warps),
                            # full N=128 KV, warp-local softmax.
                            comptime _bm = align_up(_m, 16)
                            launch_with_BM[_bm, s, WM=16, WN=128]()
                        return
                raise Error(
                    "AMD MLA decode (MTP): num_heads*S must be <= 128 (the"
                    " 8-tile fold cap) and S <= MLA_DECODE_MAX_SEQ_LEN; larger"
                    " folds must not be downgraded."
                )
            elif num_heads > 32:
                # MI355X L2 = 256 MB; threshold ≈ 0.4 × L2 = 100 MB leaves
                # headroom for Q + V + working state in L2.
                comptime L2_K_BYTES_THRESHOLD = 100 * 1024 * 1024
                comptime CACHE_LEN_THRESHOLD = 32768
                var k_bytes_total = batch_size * max_cache_valid_length * depth
                if (
                    max_cache_valid_length >= CACHE_LEN_THRESHOLD
                    and k_bytes_total >= L2_K_BYTES_THRESHOLD
                ):
                    launch_with_BM[64]()
                else:
                    launch_with_BM[32]()
            else:
                launch_with_BM[32]()
        else:
            # BF16 AMD or non-AMD: keep original BM choice.
            comptime preferred_BM_default = (
                16 if (not has_enough_smem or has_amd_gpu_accelerator()) else 32
            )
            comptime BM_default = preferred_BM_default if (
                preferred_BM_default <= num_heads
            ) else num_heads
            launch_with_BM[BM_default]()


# Split-K combine for MLA decode. Splits work two ways vs the generic
# mha_splitk_reduce (1 warp per head, ~25% CU coverage on MI355):
#   - W_PARTS warps per CTA over the partition axis (in-CU latency hiding)
#   - D_TILES CTAs per (batch, head) over the depth axis (CU coverage).
# Depth tiles are disjoint outputs, so no cross-CTA reduction is needed;
# warp 0 sums per-warp partition partials in SMEM.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(W_PARTS * WARP_SIZE)
    )
)
@__name(
    t"mla_splitk_reduce_{intermediate_type}_{output_type}",
)
def mla_splitk_reduce[
    intermediate_type: DType,
    output_type: DType,
    ValidLT: TensorLayout,
    depth: Int,
    num_heads: Int,
    D_TILES: Int,
    W_PARTS: Int,
    MAX_PARTITIONS: Int,
    use_exp2: Bool = False,
    # Token fold (S>1): split-K is keyed by query row, so num_rows =
    # num_heads * q_seq_len (heads-inner: row = token*H + head) and each reduce
    # CTA owns one row. At S=1 == num_heads, byte-identical.
    q_seq_len: Int = 1,
    # Variable per-sequence query length: when True the final store is remapped
    # through `valid_length` to the ragged `[total_q_tokens, H, depth]` layout
    # (the padded intermediate/stat reads stay keyed by (batch_idx, row_idx)).
    # False (and S=1) keeps the padded-dense `[B, num_rows, depth]` store
    # byte-identically.
    ragged: Bool = False,
](
    intermediate_ptr: UnsafePointer[Scalar[intermediate_type], ImmutAnyOrigin],
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], MutAnyOrigin
    ],
    qk_max_ptr: UnsafePointer[
        Scalar[get_accum_type[output_type]()], MutAnyOrigin
    ],
    batch_size: Int32,
    num_partitions: Int32,
    # input_row_offsets `[batch_size + 1]`. Read only on the `ragged and
    # q_seq_len > 1` path (remap the final store, skip short-sequence pad rows);
    # a zero-length placeholder otherwise.
    valid_length_tt: TileTensor[.uint32, ValidLT, ImmutAnyOrigin],
):
    var _batch_size = Int(batch_size)
    var _num_partitions = Int(num_partitions)
    comptime assert depth > 0, "depth must be positive"
    comptime assert (
        depth % (D_TILES * WARP_SIZE) == 0
    ), "depth must be divisible by D_TILES * WARP_SIZE"
    comptime assert (
        MAX_PARTITIONS % WARP_SIZE == 0
    ), "MAX_PARTITIONS must be a multiple of WARP_SIZE"
    comptime assert (
        MAX_PARTITIONS % W_PARTS == 0
    ), "MAX_PARTITIONS must be divisible by W_PARTS"
    comptime assert (
        W_PARTS >= 1 and D_TILES >= 1
    ), "W_PARTS and D_TILES must be positive"

    # Runtime invariant: the partition heuristic always returns >= 1, but
    # the clamps in step 1/step 2 compute `_num_partitions - 1` so a zero
    # would silently OOB. Catch it in debug builds.
    debug_assert(
        _num_partitions > 0,
        "mla_splitk_reduce requires _num_partitions > 0",
    )

    comptime accum_type = get_accum_type[output_type]()
    # Folded query-row count (M = num_heads * q_seq_len). At S=1 == num_heads.
    comptime num_rows = num_heads * q_seq_len
    comptime depth_per_cta = depth // D_TILES
    comptime elems_per_lane = depth_per_cta // WARP_SIZE
    comptime parts_per_warp = MAX_PARTITIONS // W_PARTS
    # Per-lane partition stride: when MAX_PARTITIONS > WARP_SIZE the same lane
    # owns multiple partitions. e.g. MAX_PARTITIONS=128 → each lane handles 2.
    comptime parts_per_lane = MAX_PARTITIONS // WARP_SIZE

    var qk_max_tt = TileTensor(
        qk_max_ptr,
        row_major((_num_partitions, _batch_size, Idx[num_rows])),
    )
    var exp_sum_tt = TileTensor(
        exp_sum_ptr,
        row_major((_num_partitions, _batch_size, Idx[num_rows])),
    )
    var intermediate_tt = TileTensor(
        intermediate_ptr,
        row_major(
            (
                _num_partitions,
                _batch_size,
                Idx[num_rows],
                Idx[depth],
            )
        ),
    )
    # Final output as a flat [rows, depth] tensor. Padded-dense keys the row by
    # `batch_idx * num_rows + row_idx` (same linear address as the old
    # `[B, num_rows, depth]`); ragged re-keys by `start_of_seq*H + row_idx`. The
    # leading dim only sizes the layout — the store address is the row index — so
    # `_batch_size * num_rows` bounds both (ragged rows are <= that).
    var output_tt = TileTensor(
        output_ptr,
        row_major((_batch_size * num_rows, Idx[depth])),
    )

    var scales_tt = tt_stack_allocation[
        dtype=accum_type, address_space=.SHARED
    ](row_major[MAX_PARTITIONS]())
    var warp_partial_tt = tt_stack_allocation[
        dtype=accum_type, address_space=.SHARED
    ](row_major[W_PARTS, depth_per_cta]())

    var d_tile_idx = block_idx.x
    # block_idx.y indexes a folded query row (token*H + head) in [0, num_rows);
    # == head when q_seq_len == 1.
    var row_idx = block_idx.y
    var batch_idx = block_idx.z
    var warp_idx = warp_id()
    var lane_idx = lane_id()
    var depth_in_tile = lane_idx * elems_per_lane
    var depth_global = d_tile_idx * depth_per_cta + depth_in_tile

    # Absolute output row for the final store (padded-dense; the
    # intermediate/stat reads below stay keyed by (batch_idx, row_idx)).
    var out_row = batch_idx * num_rows + row_idx

    # Variable per-sequence query length: on the ragged S>1 path remap the store
    # to `[total_q_tokens, H, depth]`. `valid_length` is the input_row_offsets,
    # so `seq_len_rt = valid_length[b+1] - valid_length[b]`; skip pad rows
    # (row_idx >= H*seq_len_rt, never written by the decode kernel) and send live
    # rows to `start_of_seq*H + row_idx`. Comptime-dead at S=1 / non-ragged
    # (byte-identical `out_row`).
    comptime if ragged and q_seq_len > 1:
        var valid_length = valid_length_tt.to_layout_tensor()
        var start_of_seq = Int(valid_length[batch_idx])
        var seq_len_rt = Int(valid_length[batch_idx + 1]) - start_of_seq
        if Int(row_idx) >= num_heads * seq_len_rt:
            return
        out_row = start_of_seq * num_heads + Int(row_idx)

    # Step 1: warp 0 computes per-partition scales. Each lane owns
    # `parts_per_lane` partitions strided by WARP_SIZE; `parts_per_lane`
    # is comptime so the `comptime for` unrolls to the right shape (and
    # to a single iteration when MAX_PARTITIONS == WARP_SIZE).
    if warp_idx == 0:
        comptime exp_fn = _exp2_concrete if use_exp2 else _exp_concrete
        var np_last = _num_partitions - 1
        var lse_lane = SIMD[accum_type, parts_per_lane](
            min_or_neg_inf[accum_type]()
        )
        var local_max: Scalar[accum_type] = min_or_neg_inf[accum_type]()
        # Clamp the partition index so the load stays in-bounds for lanes
        # whose partition_idx >= _num_partitions; ternary-select to -inf
        # for OOB lanes so neither lse_lane nor local_max are polluted.
        comptime for k in range(parts_per_lane):
            var partition_idx = Int(lane_idx) + k * WARP_SIZE
            var pi_safe = min(partition_idx, np_last)
            var v_raw = qk_max_tt[pi_safe, batch_idx, row_idx]
            var v = (
                v_raw if partition_idx
                < _num_partitions else min_or_neg_inf[accum_type]()
            )
            lse_lane[k] = v
            local_max = max(local_max, v)

        var qk_max_global = warp.max(local_max)
        # exp_sum == 0 only if every partition had qk_max == -inf;
        # emit scale=0 instead of NaN so step 2 produces clean zero.
        if qk_max_global == min_or_neg_inf[accum_type]():
            qk_max_global = 0

        var rescaled_lane = SIMD[accum_type, parts_per_lane](0)
        var local_sum: Scalar[accum_type] = 0
        # No predicate needed: for OOB lanes lse_lane[k] is -inf so
        # exp_fn(-inf - qk_max_global) == 0, making r == 0 regardless of
        # the (clamped) exp_sum value loaded.
        comptime for k in range(parts_per_lane):
            var partition_idx = Int(lane_idx) + k * WARP_SIZE
            var pi_safe = min(partition_idx, np_last)
            var r = exp_sum_tt[pi_safe, batch_idx, row_idx] * exp_fn(
                lse_lane[k] - qk_max_global
            )
            rescaled_lane[k] = r
            local_sum += r

        var exp_sum = warp.sum(local_sum)
        var inv_global_exp_sum: Scalar[accum_type] = 0
        if exp_sum > 0:
            inv_global_exp_sum = recip(exp_sum)

        # partition_idx is in [0, MAX_PARTITIONS) by construction, so no
        # bounds check is needed.
        comptime for k in range(parts_per_lane):
            var partition_idx = Int(lane_idx) + k * WARP_SIZE
            scales_tt[partition_idx] = rescaled_lane[k] * inv_global_exp_sum

    barrier()

    # Step 2: per-warp partition accumulation.
    #
    # Warp-level bail: when `part_start_warp >= _num_partitions` the entire
    # warp has no real work, so it just leaves `acc = 0` and jumps to the
    # step-3 barrier. This matches the old per-iter-predicated code's
    # behavior for fully-OOB warps (critical for small-np / large-grid
    # shapes like num_heads=128 bs=8 where np=16 → most warps would
    # otherwise issue clamped loads + zero-FMAs for nothing).
    #
    # Warps that pass the bail issue all `parts_per_warp` partitions
    # together (no per-iter predicate) so the compiler can
    # software-pipeline the HBM loads with a single `vmcnt` drain. For
    # the boundary warp (some real partitions, some OOB):
    #   - The HBM index is clamped to `_num_partitions - 1` so the load
    #     stays in-bounds. The clamped data is irrelevant because:
    #   - `scales_tt[p]` is 0 (step 1 writes 0 for slots whose lse stayed
    #     at -inf, i.e. partitions outside [0, _num_partitions)), and
    #   - the `scale > 0` mask zeros out the contribution.
    var part_start_warp = warp_idx * parts_per_warp
    var acc = SIMD[accum_type, elems_per_lane](0)

    if part_start_warp < _num_partitions:
        var np_last_s2 = _num_partitions - 1
        var xs = Array[SIMD[accum_type, elems_per_lane], parts_per_warp](
            fill=SIMD[accum_type, elems_per_lane](0)
        )
        var scales_local = Array[Scalar[accum_type], parts_per_warp](
            fill=Scalar[accum_type](0)
        )
        comptime for k in range(parts_per_warp):
            var p = part_start_warp + k
            var p_safe = min(p, np_last_s2)
            scales_local[k] = scales_tt[p]
            xs[k] = intermediate_tt.load[width=elems_per_lane](
                Coord(
                    p_safe,
                    batch_idx,
                    row_idx,
                    depth_global,
                )
            ).cast[accum_type]()
        comptime for k in range(parts_per_warp):
            var scale_k = scales_local[k]
            var safe = SIMD[.bool, elems_per_lane](fill=scale_k > 0).select(
                xs[k], type_of(xs[k])(0)
            )
            acc += safe * type_of(safe)(scale_k)

    # Step 3: cross-warp reduction and output store (`out_row` set above).
    comptime if W_PARTS == 1:
        output_tt.store(
            Coord(out_row, depth_global),
            acc.cast[output_type](),
        )
    else:
        warp_partial_tt.store(Coord(warp_idx, depth_in_tile), acc)
        barrier()

        if warp_idx == 0:
            var final_acc = SIMD[accum_type, elems_per_lane](0)
            comptime for w in range(W_PARTS):
                final_acc += warp_partial_tt.load[width=elems_per_lane](
                    Coord(Idx[w], depth_in_tile)
                )
            output_tt.store(
                Coord(out_row, depth_global),
                final_acc.cast[output_type](),
            )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"mla_decoding_{q_type}_{output_type}_{BM}x{BN}x{BK}_{ragged}_nqh{num_heads}_nkvh{num_heads // group}",
)
def mla_decoding[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    ValidLT: TensorLayout,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    depth_v: Int,
    num_heads: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    group: Int = 1,
    ragged: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    decoding_warp_split_k: Bool = False,
    # MTP (multi-token prediction): number of query tokens (S) folded into the
    # MMA M dimension on AMD. Default 1 = single-token decode. NVIDIA ignores
    # it (its fold path is selected separately).
    q_seq_len: Int = 1,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    batch_size_dev: Int32,
    num_partitions_dev: Int32,
    max_cache_valid_length_dev: Int32,  # longest KV cache entry
    valid_length_tt: TileTensor[
        .uint32, ValidLT, ImmutAnyOrigin
    ],  # valid length per batch
    mask: mask_t,
):
    """MLA decoding kernel that computes single-batch attention per CTA.

    Sets up split-K offsets, batch indexing, and ragged or dense query strides,
    then delegates to `mla_decoding_single_batch` on NVIDIA or
    `Attention.mla_decode` on AMD to perform the actual QK and PV computation
    for one decode step (with optional MTP query folding on AMD).

    Parameters:
        q_type: Element type of the query tensor `q_ptr`. Must be a
            half-float or `float8_e4m3fn`.
        k_t: KV cache operand type backing `k`. Carries the KV layout,
            dtype, and head geometry.
        output_type: Element type of the output tensor `output_ptr`.
            `bfloat16` when `q_type` is `float8_e4m3fn`, else matches
            `q_type`.
        mask_t: Mask functor type applied to attention scores.
        ValidLT: Compile-time layout of the `valid_length_tt` tensor.
        BM: Number of query rows per block (the M-tile height).
        BN: Number of key rows per block (the N-tile width).
        BK: Tile size in the depth (head) dimension.
        WM: Warp tile height in the M dimension.
        WN: Warp tile width in the N dimension.
        depth: Total head dimension of Q and K (576 for DeepSeek V2/3).
        depth_v: V head dimension; V is derived as `K[:, :, :depth_v]`
            (512 for DeepSeek V2/3).
        num_heads: Number of query attention heads.
        num_threads: Number of threads per CTA, derived from the warp
            tile geometry.
        num_pipeline_stages: Number of software-pipelined MMA stages.
        group: GQA group size (`num_heads // kv_num_heads`); for MLA
            `kv_num_heads == 1` so `group == num_heads` (defaults to 1).
        ragged: Whether `q_ptr` is a ragged tensor with batch offsets in
            `valid_length_tt` instead of a padded tensor (defaults to
            `False`).
        _use_valid_length: Whether to read `valid_length_tt` for per-batch
            sequence lengths; `False` treats all sequences as `q_seq_len`
            long (defaults to `False`).
        _is_cache_length_accurate: Whether `max_cache_valid_length` is the
            accurate latest length; `False` adds `seq_len` for continuous
            execution (defaults to `False`).
        decoding_warp_split_k: Whether to use warp-level split-K
            reduction within the decode kernel (defaults to `False`).
        q_seq_len: Number of query tokens (S) folded into the MMA M
            dimension for MTP; 1 for single-token decode (defaults to 1).

    Args:
        q_ptr: Pointer to the query tensor with shape `[batch,
            q_seq_len, num_heads, depth]` (or ragged equivalent).
        k: KV cache operand. V is derived as `K[:, :, :depth_v]`, so V
            is not loaded separately.
        output_ptr: Pointer to the output tensor with shape `[batch,
            q_seq_len, num_heads, depth_v]` (or ragged equivalent).
        exp_sum_ptr: Pointer to the per-row softmax exp-sum stats
            buffer, used for split-K reduction.
        qk_max_ptr: Pointer to the per-row softmax max stats buffer,
            used for split-K reduction.
        scale: Softmax scale factor applied to QK^T.
        batch_size_dev: Number of sequences in the batch.
        num_partitions_dev: Split-K partition count; 1 disables split-K
            reduction.
        max_cache_valid_length_dev: Total number of cached KV entries (the
            cache context length).
        valid_length_tt: Per-batch `uint32` tensor of valid (pre-padding)
            sequence lengths. For ragged inputs, carries cumulative row
            offsets.
        mask: Mask functor instance applied to attention scores.
    """
    var batch_size = Int(batch_size_dev)
    var num_partitions = Int(num_partitions_dev)
    var max_cache_valid_length = Int(max_cache_valid_length_dev)
    var valid_length = valid_length_tt.to_layout_tensor()
    var batch_idx = block_idx.z

    # split-k offsets
    var partition_idx = block_idx.x
    # Output is [B, S, H, depth_v]. Split-K re-keys the intermediate workspaces
    # by query row, so both the intermediate output and the softmax stats stride
    # by M = q_seq_len * num_heads (heads-inner: row = token*H + head). At S=1
    # every q_seq_len factor is 1, byte-identical to the num_heads-keyed offsets.
    var output_batch_offset: Int = (
        q_seq_len * depth_v * num_heads * batch_idx
        + q_seq_len * depth_v * num_heads * batch_size * partition_idx
    )
    var qk_max_offset = (
        q_seq_len * num_heads * batch_idx
        + q_seq_len * num_heads * batch_size * partition_idx
    )
    var exp_sum_offset = qk_max_offset

    # split-k intermediate buffers — only used when num_partitions > 1
    var qk_max_batch_ptr = qk_max_ptr
    if num_partitions > 1:
        qk_max_batch_ptr = qk_max_ptr + qk_max_offset

    var exp_sum_batch_ptr = exp_sum_ptr
    if num_partitions > 1:
        exp_sum_batch_ptr = exp_sum_ptr + exp_sum_offset

    var seq_len: Int
    var q_batch_offset: Int

    comptime if ragged:
        # treat valid_lengths as a input_row_offsets
        var start_of_seq = Int(valid_length[batch_idx])
        var end_of_seq = Int(valid_length[batch_idx + 1])
        seq_len = end_of_seq - start_of_seq
        q_batch_offset = start_of_seq * depth * num_heads

        # Variable per-sequence query length: at num_partitions == 1 `output_ptr`
        # is the final ragged output, so the per-batch base is the ragged row
        # start `start_of_seq * H * depth_v`, not the comptime-uniform
        # `q_seq_len * H * depth_v * batch_idx`. (The kernel still reads M =
        # q_seq_len*H rows; the mask + SRD clamp drop short-sequence pad rows.)
        # At num_partitions > 1 the output is the padded intermediate workspace —
        # keep the dense offset; the reducer does the ragged remap. Comptime-dead
        # at S=1.
        comptime if q_seq_len > 1:
            if num_partitions == 1:
                output_batch_offset = start_of_seq * depth_v * num_heads
    elif _use_valid_length:
        # treat valid_lengths as valid lengths
        # Q is [B, S, H, depth] → batch stride = q_seq_len * H * depth.
        q_batch_offset = q_seq_len * depth * num_heads * batch_idx
        seq_len = Int(valid_length[batch_idx])
    else:
        # No valid-length path (standalone S>1 tests): dense Q is [B, S, H,
        # depth], seq_len is the comptime fold S. Bit-identical at S=1.
        seq_len = q_seq_len
        q_batch_offset = q_seq_len * depth * num_heads * batch_idx

    var num_keys = k.cache_length(batch_idx)

    comptime if not _is_cache_length_accurate:
        num_keys += seq_len

    comptime if is_nvidia_gpu():
        mla_decoding_single_batch[
            BM=BM,
            BN=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            depth=depth,
            depth_v=depth_v,
            num_threads=num_threads,
            num_pipeline_stages=num_pipeline_stages,
            decoding_warp_split_k=decoding_warp_split_k,
        ](
            q_ptr + q_batch_offset,
            k,
            output_ptr + output_batch_offset,
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            scale,
            num_keys,
            num_partitions,
            mask,
            batch_idx,
        )
    elif is_amd_gpu():
        comptime config = MHAConfig[q_type](
            num_heads,
            depth,
            num_queries_per_block=BM,
            num_keys_per_block=BN,
            BK=BK,
            WM=WM,
            WN=WN,
            num_pipeline_stages=num_pipeline_stages,
            k_group_size=group,
        )

        var attention = Attention[
            config,
            group,
            False,  # sink
            token_gen=True,
            q_depth=depth,
            output_depth=depth_v,
            mla_mode=True,
            # K==V in MLA — load once, let PV reuse K's SMEM.
            mla_kv_alias=True,
            # MTP token fold: M = num_heads * q_seq_len query rows.
            q_seq_len=q_seq_len,
        ](
            output_ptr + output_batch_offset,
            q_ptr + q_batch_offset,
            k,
            k,
            mask,
            None,
            batch_idx,
            scale,
            seq_len,
            num_keys,
            0,
        )
        attention.mla_decode(
            exp_sum_batch_ptr,
            qk_max_batch_ptr,
            num_partitions,
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def mla_decoding_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    BM: Int,  # number of queries per block
    BN: Int,  # number of keys per block
    BK: Int,  # tile size in depth dimension
    WM: Int,
    WN: Int,
    depth: Int,
    depth_v: Int,
    num_threads: Int,
    num_pipeline_stages: Int,
    decoding_warp_split_k: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    exp_sum_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    qk_max_ptr: UnsafePointer[Scalar[get_accum_type[q_type]()], MutAnyOrigin],
    scale: Float32,
    num_keys: Int,
    num_partitions: Int,
    mask: mask_t,
    batch_idx: Int,
):
    """Flash attention v2 algorithm.

    Computes single-batch MLA attention for one decode step on NVIDIA
    GPUs: Q @ K^T with online softmax, then P @ V where V is derived as
    `K[:, :depth_v]`. Split-K partitions the key range across
    `block_idx.x`; `block_idx.y` selects the query head group.

    Parameters:
        q_type: Element type of the query tensor `q_ptr` (inferred).
            Must be a half-float or `float8_e4m3fn`.
        k_t: KV cache operand type backing `k` (inferred). Carries the
            KV layout, dtype, and head geometry.
        output_type: Element type of the output tensor `output_ptr`
            (inferred). `bfloat16` when `q_type` is `float8_e4m3fn`,
            else matches `q_type`.
        mask_t: Mask functor type applied to attention scores
            (inferred).
        BM: Number of query rows per block (the M-tile height).
        BN: Number of key rows per block (the N-tile width).
        BK: Tile size in the depth (head) dimension.
        WM: Warp tile height in the M dimension.
        WN: Warp tile width in the N dimension.
        depth: Total head dimension of Q and K (576 for DeepSeek V2/3).
        depth_v: V head dimension; V is derived as `K[:, :, :depth_v]`
            (512 for DeepSeek V2/3).
        num_threads: Number of threads per CTA, derived from the warp
            tile geometry.
        num_pipeline_stages: Number of software-pipelined MMA stages.
        decoding_warp_split_k: Whether to use warp-level split-K
            reduction within the decode kernel (defaults to `False`).
            Currently unsupported; must be `False`.

    Args:
        q_ptr: Pointer to the query tensor for this batch, with
            `num_heads` rows of `depth` elements each, indexed by
            `block_idx.y` in `BM`-row blocks.
        k: KV cache operand. V is derived as `K[:, :, :depth_v]`, so V
            is not loaded separately.
        output_ptr: Pointer to the output tensor for this batch, with
            `num_heads` rows of `depth_v` elements each, indexed by
            `block_idx.y` in `BM`-row blocks.
        exp_sum_ptr: Pointer to the per-row softmax exp-sum stats
            buffer, used for split-K reduction.
        qk_max_ptr: Pointer to the per-row softmax max stats buffer,
            used for split-K reduction.
        scale: Softmax scale factor applied to QK^T.
        num_keys: Total number of cached KV entries for this batch.
        num_partitions: Split-K partition count; 1 disables split-K
            reduction.
        mask: Mask functor instance applied to attention scores.
        batch_idx: Batch index into the paged KV cache.
    """
    comptime k_type = k_t.dtype
    comptime assert q_type == k_type

    comptime simd_size = simd_width_of[q_type]()

    comptime WN_O = 128
    comptime nope_dim = depth_v
    comptime rope_dim = depth - depth_v

    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN

    comptime assert num_warps_m * num_warps_n == (
        num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    comptime assert (
        not decoding_warp_split_k
    ), "mla_decoding doesn't support warp split-k."

    var tid = thread_idx.x
    var warp_id = warp.broadcast(ufloordiv(tid, WARP_SIZE))
    var lane = lane_id()

    # Coordinates of the current warp.
    var warp_y, warp_x = udivmod(warp_id, num_warps_n)

    # The entire query block (BM x depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )

    comptime kv_smem_size = BN * depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()

    # For MLA, We define V = K[:, :nope_dim], thus we split the K tensor
    # in two parts when storing it in the smem: K[:, :nope_dim] and
    # K[:, nope_dim:(nope_dim+rope_dim)].
    # Instead of initializing the tiled iterator with a row-major layout
    # (BN, BK) like standard mha kernels, we manually set the following
    # layout. This ensures that once Q @ K calculation is complete, the
    # K[:, :nope_dim] tensor stored continuously in the smem.
    comptime IteratorTypeKV = LayoutTensorIter[
        k_type,
        Layout(IntTuple(BN, BK), IntTuple(nope_dim, 1)),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var kv_nope_smem_iter = IteratorTypeKV(
        k_smem,
        IteratorTypeKV.layout_uint_type(nope_dim),
        stride=IteratorTypeKV.layout_uint_type(BK),
    )

    # view the K[:, :nope_dim] as V tensor.
    comptime IteratorTypeV = LayoutTensorIter[
        k_type,
        Layout.row_major(BK, nope_dim),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        k_smem, IteratorTypeV.layout_uint_type(BN * nope_dim)
    )

    # smem for the last rope_dim of each head, will only be used during
    # Q @ K calculation.
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var k_rope_smem_iter = IteratorTypeK(
        k_smem + BN * nope_dim, IteratorTypeK.layout_uint_type(BN * rope_dim)
    )

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    comptime num_output_rows = num_m_mmas * (WN_O // MMA_N)  # num_n_mmas
    comptime num_output_rows_full = num_output_rows
    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_output_rows_full, p_frag_size),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0.0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()
    var rowsum = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()

    comptime for i in range(WM):
        rowmax[i] = min_or_neg_inf[accum_type]()
        rowsum[i] = 0.0

    # Shared memory for P = Q * K^t
    var p_smem = (k_smem + kv_smem_size).bitcast[Scalar[k_type]]()
    comptime p_smem_size = BM * BN
    comptime IteratorTypeP = LayoutTensorIter[
        k_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=.SHARED,
    ]((p_smem + BM * BN).bitcast[Scalar[accum_type]]())

    comptime kv_num_heads = 1
    comptime kv_head_idx = 0
    var q_head_group = block_idx.y

    var q_offset = depth * BM * q_head_group

    comptime q_gmem_layout = Layout.row_major(BM, depth)
    var q_gmem_block = LayoutTensor[q_type, q_gmem_layout](q_ptr + q_offset)
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)

    var start, end = get_start_and_end_for_partitions[BN](
        num_keys, num_partitions, block_idx.x
    )

    # Mask global memory iterator, seq_len = 1
    comptime seq_len = 1
    var mask_warp_col = warp_x * WN + start

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.layout_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        q_gmem_iter._incr()

    @always_inline
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int) {
        mut mask_warp_col,
        mut kv_nope_smem_iter,
        mut k_rope_smem_iter,
        mut v_smem_iter,
        imm,
    }:
        var k_ptr = k.block_paged_ptr[BN](
            UInt32(batch_idx), UInt32(kv_tile_start_row), kv_head_idx, 0
        )

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(kv_num_heads * depth, 1),
        )
        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type=.int32, linear_idx_type=.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=.int32](
                kv_num_heads * depth, 1
            ),
        )

        _ = p_reg_tile.fill(0)

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            k_ptr,
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        # load K[:, nope_dim:(nope_dim+rope_dim)], this would be used later
        comptime k_rope_num_ves = BN * rope_dim // simd_size
        comptime async_copy_k_rope_layout = Layout.row_major(
            ufloordiv(
                min(num_threads, k_rope_num_ves) * simd_size,
                k_rope_smem_iter.layout.shape[1].value(),
            ),
            k_rope_smem_iter.layout.shape[1].value() // simd_size,
        )

        comptime for k_id in range(ufloordiv(rope_dim, BK)):
            var k_rope_smem_tile = k_rope_smem_iter.next_unsafe(
                k_rope_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_rope_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_rope_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter.next(ufloordiv(nope_dim, BK) + k_id)[].vectorize[
                    1, simd_size
                ](),
            )

        # Calculate Q[:, :nope_dim] @ K[:, :nope_dim] (K transposed)
        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=True,
            static_num_iters=ufloordiv(nope_dim, BK),
        ](
            p_reg_tile,
            q_smem_iter,
            k_gmem_iter,
            q_smem_iter,
            kv_nope_smem_iter,
            ufloordiv(nope_dim, BK),
        )

        # Calculate the last `rope_dim` part of Q @ K
        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            1,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(rope_dim, BK),
        ](
            p_reg_tile,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(ufloordiv(nope_dim, BK))
            ),
            k_rope_smem_iter,
            q_smem_iter.next_unsafe(
                q_smem_iter.linear_uint_type(ufloordiv(nope_dim, BK))
            ),
            k_rope_smem_iter,
            ufloordiv(nope_dim, BK),
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        def _apply_mask[masked: Bool]() {imm}:
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var q_head_idx = q_head_group * (BM + m_mma) * MMA_M
                    var mask_frag_col = mask_warp_col + n_mma * MMA_N

                    # Offset to current thread's fragment
                    mask_frag_col += umod(lane * p_frag_simdwidth, MMA_N)

                    # Offset to current thread's head idx
                    q_head_idx += ufloordiv(
                        lane, ufloordiv(MMA_N, p_frag_simdwidth)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_col = mask_frag_col

                        var score_head_idx = q_head_idx + (i * MMA_M // 2)

                        var score_row_with_start_pos = num_keys - 1
                        var score_row = (
                            0  # this is a decoding kernel with seq_len = 1
                        )

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    block_idx.z,
                                    score_head_idx,
                                    score_row_with_start_pos,
                                    score_col,
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type=.uint32](
                                    score_row, score_col
                                ),
                                IndexList[2, element_type=.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch(
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    num_keys,
                    kv_tile_start_row,
                ),
                Index[dtype=DType.uint32](1, BN),
            )
            == TileMaskStatus.PARTIAL_MASK,
            _apply_mask,
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += BN

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )

        comptime output_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * (WN_O // MMA_N), 2
        )
        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, warp_y),
            rowmax,
            rowsum,
        )

        # Copy score fragments to shared memory with swizzling to resolve bank
        # conflicts for ldmatrix in the 2nd matmul.
        # warp_split_k does not need the copy as warps don't perform reduction
        # iterating across tiles, but use extra registers to perform MMAs
        # with warp-local data.
        _copy_frag_to_smem[BM, BN, BK, WM, WN, MMA_M, MMA_N, p_frag_simdwidth](
            p_smem_iter, p_reg_tile, UInt32(warp_x), UInt32(warp_y)
        )

        async_copy_wait_all()
        barrier()

        # S[m, :] @ V[:, (0:WN) + n*WN]
        multistage_mma[
            BM,
            nope_dim,
            BK,
            WM,
            WN_O,
            num_threads,
            num_pipeline_stages,
            False,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=ufloordiv(BN, BK),
        ](
            output_reg_tile,
            p_smem_iter,
            v_smem_iter,
            p_smem_iter,
            v_smem_iter,
            ufloordiv(BN, BK),
        )

        barrier()

    tile_and_unswitch[[BN]](start, end, loop_over_kvcache)

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(WN_O // 8):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    var o_offset = nope_dim * BM * q_head_group

    comptime output_gmem_layout = Layout(
        IntTuple(BM, nope_dim), IntTuple(nope_dim, 1)
    )
    var output_gmem_tile = LayoutTensor[output_type, output_gmem_layout](
        output_ptr + o_offset,
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN_O](warp_y, warp_x)

    # Write to global memory.
    comptime if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=nope_dim, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, nope_dim),
            address_space=.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, WN_O](
            warp_y, warp_x
        )

        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                WARP_SIZE * simd_size // WN_O, WN_O // simd_size
            ),
            swizzle=swizzle,
        ](
            output_gmem_warp_tile.vectorize[1, simd_size](),
            accum_smem_warp_tile.vectorize[1, simd_size](),
        )

    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# GPU Multi-head Latent Attention (MLA) prefill implementations
# ===-----------------------------------------------------------------------===#


@always_inline
def _ragged_kv_view(
    src: TileTensor[address_space=.GENERIC, ...],
) -> TileTensor[
    src.dtype,
    RowMajorLayout[*Coord[Int, Int, Int].element_types],
    ImmutAnyOrigin,
]:
    """Rebuild a ragged K/V buffer as a rank-3 row-major TileTensor view.

    The MLA prefill K/V operands are contiguous row-major
    `[total_keys, num_heads, depth]` tensors. Rebuilding the view directly
    from the source pointer (instead of round-tripping through a
    `LayoutTensor` + `lt_to_tt`) normalizes the origin, address space, and
    linear-index type to the canonical GENERIC + `ImmutAnyOrigin` form the
    `RaggedMHAOperand` buffer field expects.
    """
    return TileTensor(
        rebind[UnsafePointer[Scalar[src.dtype], ImmutAnyOrigin]](src.ptr),
        row_major(
            Coord(Int(src.dim[0]()), Int(src.dim[1]()), Int(src.dim[2]()))
        ),
    )


@always_inline
def _ragged_offsets_view(
    src: TileTensor[.uint32, address_space=.GENERIC, ...],
) -> TileTensor[
    .uint32, RowMajorLayout[*Coord[Int].element_types], ImmutAnyOrigin
]:
    """Rebuild ragged cache-row-offsets as a rank-1 row-major TileTensor view.

    Mirrors `_ragged_kv_view` for the rank-1 `cache_row_offsets` operand.
    """
    return TileTensor(
        rebind[UnsafePointer[UInt32, ImmutAnyOrigin]](src.ptr),
        row_major(Coord(Int(src.dim[0]()))),
    )


@always_inline
def _ragged_scales_view(
    src: TileTensor[address_space=.GENERIC, ...],
) -> TileTensor[
    src.dtype, RowMajorLayout[*Coord[Int, Int].element_types], ImmutAnyOrigin
]:
    """Rebuild a ragged per-token scale buffer as a rank-2 row-major view.

    Mirrors `_ragged_kv_view` for the rank-2 `[total_keys, 1]` `k_scales`
    operand, normalizing to the GENERIC + `ImmutAnyOrigin` scale-buffer form.
    """
    return TileTensor(
        rebind[UnsafePointer[Scalar[src.dtype], ImmutAnyOrigin]](src.ptr),
        row_major(Coord(Int(src.dim[0]()), Int(src.dim[1]()))),
    )


# entrypoint for MLA prefill kernels
@always_inline
def flare_mla_prefill[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    output_type: DType,
    //,
](
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: LayoutTensor[mut=False, _, address_space=.GENERIC, ...],
    v: LayoutTensor[mut=False, _, address_space=.GENERIC, ...],
    k_rope: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_row_offsets: TileTensor[.uint32, address_space=.GENERIC, ...],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    """MLA prefill kernel that would only be called in the optimized compute
    graph. Only supports ragged Q/K/V inputs.

    The Q input has a shape of [seq_len, num_heads, q_depth].
    The K and V input has a shape of [cache_len, num_heads, depth].
    The K_rope input is retrieved from the KV cache, with a shape of
    [cache_len, 1, q_depth - depth].

    Specifically, for DeepSeek V2/3, depth = 128 and q_depth = 192.

    When computing attention scores (Q @ K), each head of K is smaller than Q
    head. The missing 64 elements of K are retrieved from the K cache, and
    broadcasted to all the heads. This kernel also handles that output has
    reduced dimension compared to input Q.

    This kernel handles batches with different valid lengths (i.e., before the
    padding). Such lengths are passed in valid_length argument.

    Parameters:
        rank: Tensor rank of `q` and `output` (inferred). Must be 3 for
            ragged inputs.
        cache_t: Paged KV cache type backing `k_rope` (inferred).
            Carries the KV layout, dtype, and head geometry.
        mask_t: Mask functor type applied to attention scores (inferred).
        dtype: Element type of `q` (inferred). Must be `bfloat16` or
            `float8_e4m3fn`.
        output_type: Element type of `output` (inferred). `bfloat16` when
            `q` is `float8_e4m3fn`, else matches `q`.

    Args:
        output: Output tensor with ragged rank-3 shape
            `[total_q_tokens, num_heads, depth]`. Dtype is `bfloat16`
            when `q` is `float8_e4m3fn`, else matches `q`.
        q: Query tensor with ragged rank-3 shape
            `[total_q_tokens, num_heads, q_depth]`. For DeepSeek V2/3,
            `q_depth` is 192.
        k: Key tensor (nope part) with shape
            `[cache_len, num_heads, depth]`. For DeepSeek V2/3, `depth`
            is 128.
        v: Value tensor with shape `[cache_len, num_heads, depth]`.
            Same `depth` as `k`.
        k_rope: Paged KV cache operand providing the rope part of K,
            with shape `[cache_len, 1, q_depth - depth]`. For DeepSeek
            V2/3, the last dim is 64.
        mask_functor: Mask functor instance applied to attention scores.
        valid_length: Per-batch `uint32` tensor of cumulative row offsets
            into the ragged Q layout; `offsets[b]` to `offsets[b+1]`
            gives batch `b`'s token range.
        cache_row_offsets: Per-batch `uint32` tensor of row offsets into
            the ragged KV layout, used to build the ragged K/V operands.
        scale: Softmax scale factor applied to QK^T.
        ctx: Device context used to enqueue the kernel.
        q_max_seq_len: Optional maximum query sequence length (tokens per
            batch); defaults to the cache's `max_prompt_length` when
            `None`.
        cache_offsets: Optional per-batch `uint32` tensor of starting
            offsets into the paged KV cache; `None` when the cache is
            contiguous.
    """
    comptime assert rank == 3, "only support ragged inputs"

    comptime if q.dtype == .bfloat16 and cache_t.dtype == .bfloat16:
        comptime assert (
            q.dtype == k.dtype == v.dtype == cache_t.dtype == output.dtype
        ), "Q, K, V, output should have same type if q.dtype is bfloat16"
    elif q.dtype == .bfloat16 and cache_t.dtype == .float8_e4m3fn:
        comptime assert q.dtype == k.dtype == v.dtype == output.dtype, (
            "Q, K, V, output should have same type if q.dtype is bfloat16 and"
            " k_rope.dtype is float8_e4m3fn"
        )
    elif q.dtype == .float8_e4m3fn and cache_t.dtype == .float8_e4m3fn:
        comptime assert (
            q.dtype == k.dtype == v.dtype == cache_t.dtype
            and output.dtype == .bfloat16
        ), (
            "Q, K, V, output should have same type if q.dtype is float8_e4m3fn"
            " and k_rope.dtype is float8_e4m3fn and output.dtype is bfloat16"
        )
    else:
        comptime assert False, "Q, K, V, output dtype combination not supported"

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg("k", k.runtime_layout.shape.value),
                    trace_arg("v", v.runtime_layout.shape.value),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        var cro_buf = _ragged_offsets_view(cache_row_offsets)
        var k_operand = RaggedMHAOperand(
            TileTensor(
                k.ptr,
                row_major(
                    Coord(Int(k.dim[0]()), Int(k.dim[1]()), Int(k.dim[2]()))
                ),
            ),
            cro_buf,
        )
        var v_operand = RaggedMHAOperand(
            TileTensor(
                v.ptr,
                row_major(
                    Coord(Int(v.dim[0]()), Int(v.dim[1]()), Int(v.dim[2]()))
                ),
            ),
            cro_buf,
        )
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        comptime kv_num_heads = cache_t.kv_params.num_heads
        comptime cache_depth = cache_t.kv_params.head_size
        comptime q_depth = type_of(q).static_shape[rank - 1]

        comptime num_keys_per_block = 64

        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],  # num_heads
            Int(k.layout.shape[rank - 1]),  # depth
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )

        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            config=mha_config,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


# entrypoint for TileTensor as K_rope input, used by tests.
@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: TileTensor[address_space=.GENERIC, ...],
    v: TileTensor[address_space=.GENERIC, ...],
    k_rope: TileTensor[address_space=.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_row_offsets: TileTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    comptime assert rank == 3, "only support ragged inputs"

    comptime if q.dtype == .bfloat16 and k_rope.dtype == .bfloat16:
        comptime assert (
            q.dtype == k.dtype == v.dtype == k_rope.dtype == output.dtype
        ), "Q, K, V, output should have same type if q.dtype is bfloat16"
    elif q.dtype == .bfloat16 and k_rope.dtype == .float8_e4m3fn:
        comptime assert q.dtype == k.dtype == v.dtype == output.dtype, (
            "Q, K, V, output should have same type if q.dtype is bfloat16 and"
            " k_rope.dtype is float8_e4m3fn"
        )
    elif q.dtype == .float8_e4m3fn and k_rope.dtype == .float8_e4m3fn:
        comptime assert (
            q.dtype == k.dtype == v.dtype == k_rope.dtype
            and output.dtype == .bfloat16
        ), (
            "Q, K, V, output should have same type if q.dtype is float8_e4m3fn"
            " and k_rope.dtype is float8_e4m3fn and output.dtype is bfloat16"
        )
    else:
        comptime assert False, "Q, K, V, output dtype combination not supported"

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q.dim[0]())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cro_buf = _ragged_offsets_view(cache_row_offsets)
        var k_operand = RaggedMHAOperand(_ragged_kv_view(k), cro_buf)
        var v_operand = RaggedMHAOperand(_ragged_kv_view(v), cro_buf)
        var k_rope_operand = LayoutTensorMHAOperand(
            TileTensor(k_rope.ptr, row_major(k_rope.layout.shape_coord()))
        )

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q).static_shape[q.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            config=mha_config,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=True,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


# entrypoint for TileTensor as K_rope input with scales, used by tests.
@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: TileTensor[address_space=.GENERIC, ...],
    v: TileTensor[address_space=.GENERIC, ...],
    k_rope: TileTensor[address_space=.GENERIC, ...],
    k_rope_scales: TileTensor[address_space=.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_row_offsets: TileTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    comptime assert rank == 3, "only support ragged inputs"
    comptime assert (
        q.dtype == k.dtype == v.dtype == k_rope.dtype == output.dtype
    ) if k_rope.dtype == .bfloat16 else (
        q.dtype == k.dtype == v.dtype == output.dtype
    ), (
        "Q, K, V, output should have same type if k_rope.dtype is bfloat16,"
        " otherwise only Q, K, V should have same type."
    )
    comptime assert (
        q.dtype == .float32 or q.dtype.is_half_float()
    ), "Only support single and half precision."

    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q",
                        coord_to_index_list(q.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q.dim[0]())

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        var cro_buf = _ragged_offsets_view(cache_row_offsets)
        var k_operand = RaggedMHAOperand(_ragged_kv_view(k), cro_buf)
        var v_operand = RaggedMHAOperand(_ragged_kv_view(v), cro_buf)
        var k_rope_operand = LayoutTensorMHAOperand(
            TileTensor(k_rope.ptr, row_major(k_rope.layout.shape_coord())),
            TileTensor(
                k_rope_scales.ptr,
                row_major(k_rope_scales.layout.shape_coord()),
            ),
        )

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q).static_shape[q.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        flare_mla_prefill_dispatch[
            kv_num_heads=kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            config=mha_config,
            _ndbuffer_mha_operand=True,
        ](
            output,
            q,
            k_operand,
            v_operand,
            k_rope_operand,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            ctx,
            cache_offsets,
        )


@always_inline
def flare_mla_prefill[
    rank: Int,
    mask_t: MHAMask,
    dtype: DType,
    scale_dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q_nope: TileTensor[dtype, address_space=.GENERIC, ...],
    q_rope: TileTensor[address_space=.GENERIC, ...],
    q_scale: TileTensor[scale_dtype, address_space=.GENERIC, ...],
    k: TileTensor[address_space=.GENERIC, ...],
    k_scales: TileTensor[scale_dtype, address_space=.GENERIC, ...],
    v: TileTensor[address_space=.GENERIC, ...],
    k_rope: TileTensor[address_space=.GENERIC, ...],
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_row_offsets: TileTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q_nope",
                        coord_to_index_list(q_nope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_rope",
                        coord_to_index_list(q_rope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int = Int(q_nope.dim[0]())

        comptime assert q_scale.rank == 2, (
            "q_scale must be a per token scale 2D tensor of [batch_size *"
            " seq_len, 1]"
        )
        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()

        comptime assert k_scales.rank == 2, (
            "k_scales must be a per token scale 2D tensor of [batch_size *"
            " num_keys, 1]"
        )

        var q_rope_lt = q_rope.to_layout_tensor()
        var q_scale_lt = q_scale.to_layout_tensor()
        var cro_buf = _ragged_offsets_view(cache_row_offsets)
        var k_operand = RaggedMHAOperand(
            _ragged_kv_view(k),
            _ragged_scales_view(k_scales),
            cro_buf,
        )

        var v_operand = RaggedMHAOperand(_ragged_kv_view(v), cro_buf)
        var k_rope_operand = LayoutTensorMHAOperand(
            TileTensor(k_rope.ptr, row_major(k_rope.layout.shape_coord()))
        )

        var batch_size: Int = Int(valid_length.dim[0]()) - 1

        if batch_size == 0 or max_prompt_len == 0:
            return

        comptime output_type = output.dtype
        comptime kv_num_heads = type_of(k_rope).static_shape[2]
        comptime cache_depth = type_of(k_rope).static_shape[3]
        comptime q_depth = type_of(q_nope).static_shape[
            q_nope.rank - 1
        ] + type_of(q_rope).static_shape[q_rope.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q_nope).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        mla_sm100_prefill_per_token_scale[
            config=mha_config,
            group=mha_config.num_heads // kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=True,
            v_depth=type_of(output).static_shape[rank - 1],
        ](
            output,
            q_nope,
            q_rope_lt,
            q_scale_lt,
            k_operand,
            k_rope_operand,
            v_operand,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )


# entrypoint for paged K_rope (KVCacheT) with per-token Q/K scales,
# used by tests. Mirrors the contiguous per-token-scale entrypoint
# above but swaps K_rope from a TileTensor to a paged KVCacheT
# operand (matching the generic paged entrypoint at ~:1517).
@always_inline
def flare_mla_prefill[
    rank: Int,
    cache_t: KVCacheT,
    mask_t: MHAMask,
    dtype: DType,
    scale_dtype: DType,
    //,
](
    output: TileTensor[mut=True, address_space=.GENERIC, ...],
    q_nope: TileTensor[dtype, address_space=.GENERIC, ...],
    q_rope: TileTensor[address_space=.GENERIC, ...],
    q_scale: TileTensor[scale_dtype, address_space=.GENERIC, ...],
    k: TileTensor[address_space=.GENERIC, ...],
    k_scales: TileTensor[scale_dtype, address_space=.GENERIC, ...],
    v: TileTensor[address_space=.GENERIC, ...],
    k_rope: cache_t,
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    cache_row_offsets: TileTensor[
        mut=True, .uint32, address_space=.GENERIC, ...
    ],
    scale: Float32,
    ctx: DeviceContext,
    q_max_seq_len: OptionalReg[Int] = None,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    @always_inline
    def description_fn() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg(
                        "q_nope",
                        coord_to_index_list(q_nope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "q_rope",
                        coord_to_index_list(q_rope.layout.shape_coord()),
                    ),
                    trace_arg(
                        "k",
                        coord_to_index_list(k.layout.shape_coord()),
                    ),
                    trace_arg(
                        "v",
                        coord_to_index_list(v.layout.shape_coord()),
                    ),
                    trace_arg(
                        "output",
                        coord_to_index_list(output.layout.shape_coord()),
                    ),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=ctx.default_device_info.api](
        "flare_mla_prefill",
        Trace[
            TraceLevel.OP, target=ctx.default_device_info.api
        ]._get_detail_str(description_fn),
        task_id=Int(ctx.id()),
    ):
        var max_prompt_len: Int

        if q_max_seq_len:
            max_prompt_len = q_max_seq_len.value()
        else:
            max_prompt_len = Int(k_rope.max_prompt_length())

        comptime assert q_scale.rank == 2, (
            "q_scale must be a per token scale 2D tensor of [batch_size *"
            " seq_len, 1]"
        )
        comptime assert k_scales.rank == 2, (
            "k_scales must be a per token scale 2D tensor of [batch_size *"
            " num_keys, 1]"
        )

        var q_rope_lt = q_rope.to_layout_tensor()
        var q_scale_lt = q_scale.to_layout_tensor()
        var cro_buf = _ragged_offsets_view(cache_row_offsets)
        var k_operand = RaggedMHAOperand(
            _ragged_kv_view(k),
            _ragged_scales_view(k_scales),
            cro_buf,
        )

        var v_operand = RaggedMHAOperand(_ragged_kv_view(v), cro_buf)
        var k_rope_operand = KVCacheMHAOperand(k_rope)

        var batch_size: Int = Int(valid_length.dim[0]()) - 1

        if batch_size == 0 or max_prompt_len == 0:
            return

        comptime kv_num_heads = cache_t.kv_params.num_heads
        comptime cache_depth = cache_t.kv_params.head_size
        comptime q_depth = type_of(q_nope).static_shape[
            q_nope.rank - 1
        ] + type_of(q_rope).static_shape[q_rope.rank - 1]
        comptime num_keys_per_block = 64
        comptime mha_config = MHAConfig[dtype](
            type_of(q_nope).static_shape[rank - 2],
            type_of(k).static_shape[rank - 1],
            num_keys_per_block=num_keys_per_block,
            WN=num_keys_per_block,
            algorithm=FlashAttentionAlgorithm.FLASH_ATTENTION_2,
        )
        mla_sm100_prefill_per_token_scale[
            config=mha_config,
            group=mha_config.num_heads // kv_num_heads,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=False,
            v_depth=type_of(output).static_shape[rank - 1],
        ](
            output,
            q_nope,
            q_rope_lt,
            q_scale_lt,
            k_operand,
            k_rope_operand,
            v_operand,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )


@always_inline
def flare_mla_prefill_dispatch[
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    mask_t: MHAMask,
    dtype: DType,
    output_type: DType,
    //,
    kv_num_heads: Int,
    config: MHAConfig[dtype],
    q_depth: Int = 192,
    cache_depth: Int = 576,
    _ndbuffer_mha_operand: Bool = False,
](
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    q: TileTensor[dtype, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    mask_functor: mask_t,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    max_prompt_len: Int,
    scale: Float32,
    ctx: DeviceContext,
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ] = None,
) raises:
    """Dispatches an MLA prefill request to the platform-specific kernel.

    Routes to `mla_sm100_prefill` on SM100 GPUs or enqueues the generic
    `mla_prefill` kernel on other NVIDIA and AMD targets, computing the grid
    and shared-memory layout from the MHA config.

    Parameters:
        k_t: Key operand type backing `k` (inferred). Either
            `RaggedMHAOperand` or `LayoutTensorMHAOperand`.
        v_t: Value operand type backing `v` (inferred). Either
            `RaggedMHAOperand` or `LayoutTensorMHAOperand`.
        k_rope_t: KV cache operand type backing `k_rope` (inferred).
            Either `KVCacheMHAOperand` or `LayoutTensorMHAOperand`.
        mask_t: Mask functor type applied to attention scores (inferred).
        dtype: Element type of `q` (inferred). Must be `bfloat16` or
            `float8_e4m3fn`.
        output_type: Element type of `output` (inferred). `bfloat16`
            when `q` is `float8_e4m3fn`, else matches `q`.
        kv_num_heads: Number of KV attention heads. Must be 1 for MLA.
        config: MLA attention config carrying `num_heads`, `depth`, and
            tile geometry (`block_m`, `block_n`, `block_k`).
        q_depth: Q head dimension in elements (defaults to 192). For
            DeepSeek V2/3, 192 = 128 nope + 64 rope.
        cache_depth: KV cache head dimension in elements (defaults to
            576). The absorbed-latent width of the paged MLA cache.
        _ndbuffer_mha_operand: Whether the K/V/`k_rope` operands use
            ND-buffer layout instead of the default TileTensor layout
            (defaults to `False`).

    Args:
        output: Output tensor with ragged rank-3 shape
            `[total_q_tokens, num_heads, v_depth]`. Dtype is
            `bfloat16` when `q` is `float8_e4m3fn`, else matches `q`.
        q: Query tensor with ragged rank-3 shape
            `[total_q_tokens, num_heads, q_depth]`.
        k: Key operand (nope part). Carries the ragged K tensor and
            row offsets.
        v: Value operand. Carries the ragged V tensor and row offsets.
        k_rope: KV cache operand providing the rope part of K, with
            shape `[cache_len, 1, q_depth - depth]`.
        mask_functor: Mask functor instance applied to attention scores.
        valid_length: Per-batch `uint32` tensor of cumulative row offsets
            with `batch_size + 1` entries; `offsets[b]` to
            `offsets[b+1]` gives batch `b`'s token range.
        max_prompt_len: Maximum query sequence length (tokens per batch
            across all batches); drives the grid's M dimension.
        scale: Softmax scale factor applied to QK^T.
        ctx: Device context used to enqueue the kernel.
        cache_offsets: Optional per-batch `uint32` tensor of starting
            offsets into the paged KV cache; `None` when the cache is
            contiguous.
    """
    comptime num_heads = config.num_heads
    comptime depth = config.depth
    comptime group = config.num_heads // kv_num_heads
    comptime rank = output.rank

    # Per-head V / output width (`v_head_dim`), read from the output tensor's
    # last dim. Decoupled from the non-rope Q/K width: DeepSeek has them equal
    # (so this is byte-identical to the old behavior), GLM-style MLA does not.
    # The decode path already derives this as `depth_v`.
    comptime v_depth = type_of(output).static_shape[rank - 1]

    comptime assert q_depth == type_of(q).static_shape[rank - 1]
    comptime assert num_heads == type_of(q).static_shape[rank - 2]
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "flareMLA_prefill currently only supports Nvidia and AMD GPUs."

    var batch_size: Int = Int(valid_length.dim[0]()) - 1

    if batch_size == 0 or max_prompt_len == 0:
        return

    comptime q_half_float = dtype in (DType.float16, DType.bfloat16)

    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()

    comptime q_smem = BM * q_depth
    comptime k_smem = BN * q_depth
    comptime v_smem = BN * depth

    comptime smem_use = (q_smem + k_smem + v_smem) * size_of[
        config.dtype
    ]() if has_nvidia_gpu_accelerator() else 0

    comptime if _is_sm10x_gpu(ctx.default_device_info):
        comptime assert (
            k_rope_t.dtype == .bfloat16 or k_rope_t.dtype == .float8_e4m3fn
        ), "Only support bfloat16 or float8_e4m3fn for SM100"

        mla_sm100_prefill[
            config=config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
            v_depth=v_depth,
        ](
            output,
            q,
            k,
            v,
            k_rope,
            mask_functor,
            valid_length,
            DynamicInt(max_prompt_len),
            scale,
            batch_size,
            ctx,
        )

    else:
        comptime assert (
            k_rope_t.dtype == .bfloat16 or has_amd_gpu_accelerator()
        ), (
            "Only support bfloat16 for non-SM100 Nvidia GPUs; AMD supports"
            " bfloat16 and float8_e4m3fn"
        )

        var q_device = DeviceBuffer[q.dtype](
            ctx, q.ptr, q.num_elements(), owning=False
        )
        var output_device = DeviceBuffer[output.dtype](
            ctx, output.ptr, output.num_elements(), owning=False
        )

        comptime kernel = mla_prefill[
            config.dtype,
            k_t,
            v_t,
            k_rope_t,
            output.dtype,
            mask_t,
            type_of(valid_length).LayoutType,
            config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
        ]
        var grid_dim = LaunchDim(
            ceildiv(max_prompt_len, BM),
            config.num_heads,
            batch_size,
        ) if has_nvidia_gpu_accelerator() else LaunchDim(
            config.num_heads,
            ceildiv(max_prompt_len, BM),
            batch_size,
        )
        ctx.enqueue_function[kernel](
            q_device,
            k,
            v,
            k_rope,
            output_device,
            scale,
            Int32(batch_size),
            Int32(max_prompt_len),
            valid_length.as_immut(),
            cache_offsets,
            mask_functor,
            grid_dim=grid_dim,
            block_dim=(config.num_threads(), 1, 1),
            shared_mem_bytes=smem_use,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(smem_use)
            ),
        )


@__llvm_metadata(`rocdl.waves_per_eu`=SIMDLength(2))
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(config.num_threads())
    )
)
@__name(
    t"mla_prefill_{q_type}_{output_type}_{q_depth}_{cache_depth}_nqh{config.num_heads}_nkvh{config.num_heads // group}",
)
def mla_prefill[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    valid_layout: TensorLayout,
    config: MHAConfig,
    group: Int = 128,
    q_depth: Int = 192,
    cache_depth: Int = 576,
    _ndbuffer_mha_operand: Bool = False,
](
    q_ptr: UnsafePointer[Scalar[q_type], MutAnyOrigin],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[Scalar[output_type], MutAnyOrigin],
    scale: Float32,
    batch_size: Int32,
    seq_len_arg: Int32,
    valid_length_tt: TileTensor[.uint32, valid_layout, ImmutAnyOrigin],
    cache_offsets: OptionalReg[
        LayoutTensor[.uint32, Layout.row_major(UNKNOWN_VALUE), MutAnyOrigin]
    ],
    mask: mask_t,
):
    var _batch_size = Int(batch_size)
    var _seq_len_arg = Int(seq_len_arg)
    var valid_length = valid_length_tt.to_layout_tensor()
    comptime depth = config.depth
    var batch_idx = block_idx.z

    # mha inputs
    var seq_len: Int
    var max_seq_len = _seq_len_arg
    var num_keys: Int
    var start_pos: UInt32 = 0
    var cache_start_pos: UInt32 = 0

    # treat valid_lengths as a input_row_offsets
    var start_of_seq = Int(valid_length[batch_idx])
    var end_of_seq = Int(valid_length[batch_idx + 1])
    seq_len = end_of_seq - start_of_seq

    @always_inline
    def q_block_idx() -> Int:
        return block_idx.x if is_nvidia_gpu() else block_idx.y

    # @always_inline
    # def head_idx() -> Int:
    #     return block_idx.y if is_nvidia_gpu() else block_idx.x

    if seq_len < q_block_idx() * config.block_m():
        return

    comptime if _ndbuffer_mha_operand:
        num_keys = k_rope.cache_length(batch_idx)
        start_pos = UInt32(num_keys - seq_len)

    else:
        start_pos = UInt32(k_rope.cache_length(batch_idx))
        num_keys = Int(start_pos) + seq_len

    if cache_offsets:
        var cache_offsets_nd = cache_offsets.value()
        cache_start_pos = cache_offsets_nd[batch_idx][0]

    var q_batch_offset = start_of_seq * q_depth * config.num_heads
    var o_batch_offset = start_of_seq * depth * config.num_heads

    comptime if is_nvidia_gpu():
        mla_prefill_single_batch[
            config=config,
            group=group,
            q_depth=q_depth,
            cache_depth=cache_depth,
        ](
            q_ptr + q_batch_offset,
            k,
            v,
            k_rope,
            output_ptr + o_batch_offset,
            scale,
            seq_len,
            max_seq_len,
            start_pos,
            cache_start_pos,
            num_keys,
            mask,
            batch_idx,
        )
    elif is_amd_gpu():
        var attention = Attention[config, 1, False, q_depth=q_depth](
            output_ptr + o_batch_offset,
            q_ptr + q_batch_offset,
            k,
            v,
            mask,
            None,
            batch_idx,
            scale,
            seq_len,
            num_keys,
            Int(start_pos),
            Int(cache_start_pos),
        )
        attention.mla_prefill(
            k_rope,
        )
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


@always_inline
def mla_prefill_single_batch[
    q_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    k_rope_t: MHAOperand,
    output_type: DType,
    mask_t: MHAMask,
    *,
    config: MHAConfig,
    group: Int = 1,
    q_depth: Int = 192,
    cache_depth: Int = 576,
](
    q_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    k: k_t,
    v: v_t,
    k_rope: k_rope_t,
    output_ptr: UnsafePointer[mut=True, Scalar[output_type], _],
    scale: Float32,
    seq_len: Int,  # valid sequence length i.e. w/o padding.
    max_seq_len: Int,  # sequence length after padding.
    start_pos: UInt32,
    cache_start_pos: UInt32,
    num_keys: Int,
    mask: mask_t,
    batch_idx: Int,
):
    """MLA for encoding where seqlen > 1.

    Parameters:
        q_type: Element type of the query tensor `q_ptr` (inferred).
            Must be a half-float or `float8_e4m3fn`.
        k_t: Key operand type backing `k` (inferred). Provides the
            nope part of K.
        v_t: Value operand type backing `v` (inferred).
        k_rope_t: KV cache operand type backing `k_rope` (inferred).
            Provides the rope part of K and the per-batch cache
            length.
        output_type: Element type of `output_ptr` (inferred).
            `bfloat16` when `q_type` is `float8_e4m3fn`, else matches
            `q_type`.
        mask_t: Mask functor type applied to attention scores
            (inferred).
        config: MLA attention config carrying `num_heads`, `depth`,
            and tile geometry (`block_m`, `block_n`, `block_k`).
        group: GQA group size, `num_heads // num_kv_heads` (defaults
            to 1).
        q_depth: Q head dimension in elements (defaults to 192). For
            DeepSeek V2/3, 192 = 128 nope + 64 rope.
        cache_depth: KV cache head dimension in elements (defaults
            to 576). The absorbed-latent width of the paged MLA
            cache.

    Args:
        q_ptr: Pointer to this batch's query rows with shape
            `[seq_len, num_heads, q_depth]`.
        k: Key operand providing the nope (latent) part of K.
        v: Value operand.
        k_rope: KV cache operand providing the rope part of K;
            `k_rope.cache_length(batch_idx)` gives the cached
            length.
        output_ptr: Pointer to this batch's output rows with shape
            `[seq_len, num_heads, depth]` where `depth` is
            `config.depth`.
        scale: Softmax scale factor applied to QK^T.
        seq_len: Valid sequence length (without padding) for this
            batch.
        max_seq_len: Padded sequence length ceiling for this batch.
        start_pos: Starting position of the query within the
            sequence (the cached length before this prefill).
        cache_start_pos: Starting offset into the paged KV cache for
            this batch.
        num_keys: Total number of KV keys (cached plus new tokens)
            for this batch.
        mask: Mask functor instance applied to attention scores.
        batch_idx: Index of the current batch in the request.
    """
    comptime k_type = k_t.dtype
    comptime v_type = v_t.dtype
    comptime k_rope_type = k_rope_t.dtype
    comptime assert (
        q_type == k_type and k_type == v_type and k_type == k_rope_type
    )

    comptime simd_size = simd_width_of[q_type]()

    comptime num_warps_m = config.num_warps_m()
    comptime num_warps_n = config.num_warps_n()
    comptime num_threads = config.num_threads()
    comptime BM = config.block_m()
    comptime BN = config.block_n()
    comptime BK = config.block_k()
    comptime num_heads = config.num_heads
    comptime depth = config.depth

    comptime rope_depth = q_depth - depth

    comptime cache_num_heads = num_heads // group

    comptime assert (
        num_warps_m * num_warps_n == num_threads // WARP_SIZE
    ), "Number of warps doesn't match warp tile sizes."

    var tid: Int = thread_idx.x
    var warp_id = UInt32(warp_id[broadcast=True]())
    var lane = UInt32(lane_id())

    # Coordinates of the current warp.
    var warp_y, warp_x = divmod(warp_id, UInt32(num_warps_n))

    # The entire query block (BM x q_depth) is tiled in shared memory.
    comptime alignment = align_of[SIMD[q_type, simd_size]]()
    comptime q_smem_size = BM * q_depth
    var q_smem = external_memory[
        Scalar[q_type],
        address_space=.SHARED,
        alignment=alignment,
    ]()
    comptime IteratorTypeQ = LayoutTensorIter[
        q_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        alignment=alignment,
    ]
    var q_smem_iter = IteratorTypeQ(
        rebind[
            type_of(
                LayoutTensorIter[
                    q_type,
                    Layout.row_major(BM, BK),
                    q_smem.origin,
                    address_space=.SHARED,
                    alignment=alignment,
                ]().ptr
            )
        ](q_smem),
        IteratorTypeQ.layout_uint_type(q_smem_size),
    )
    # There is one pre-allocated dynamic shared buffer.
    # Need to explicitly offset key after at query's end.
    comptime k_smem_size = BN * q_depth
    var k_smem = (q_smem + q_smem_size).bitcast[Scalar[k_type]]()
    comptime IteratorTypeK = LayoutTensorIter[
        k_type,
        Layout.row_major(BN, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var k_smem_iter = IteratorTypeK(
        k_smem, IteratorTypeK.layout_uint_type(k_smem_size)
    )

    comptime v_smem_size = BN * depth
    var v_smem = (k_smem + k_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeV = LayoutTensorIter[
        v_type,
        Layout.row_major(BK, depth),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var v_smem_iter = IteratorTypeV(
        v_smem, IteratorTypeV.layout_uint_type(v_smem_size)
    )

    var head_idx = UInt32(block_idx.y)
    var q_tile_idx = UInt32(block_idx.x)

    # Query global memory iterator
    comptime q_gmem_layout = Layout(
        IntTuple(BM, q_depth),
        IntTuple(num_heads * q_depth, 1),
    )
    var q_tile_num_rows = min(
        UInt32(BM), UInt32(seq_len) - q_tile_idx * UInt32(BM)
    )
    var q_offset = UInt32(q_depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )

    var q_gmem_block = LayoutTensor[
        q_type,
        q_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        q_ptr + Int(q_offset),
        RuntimeLayout[element_type=.int32, linear_idx_type=.int32](
            RuntimeTuple[q_gmem_layout.shape, element_type=.int32](
                Int(q_tile_num_rows), q_depth
            ),
            RuntimeTuple[q_gmem_layout.stride, element_type=.int32](
                num_heads * q_depth, 1
            ),
        ),
    )
    var q_gmem_iter = q_gmem_block.tiled_iterator[BM, BK, axis=1](0, 0)
    # q tile has valid shape q_tile_num_rows x q_depth
    # q_tile_num_rows could be less than BM when seqlen % BM != 0

    comptime mma_shape = get_mma_shape[q_type, get_accum_type[q_type]()]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]
    comptime WM = config.WM
    comptime WN = config.WN
    comptime WN_O = depth
    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime num_n_mmas_output = WN_O // MMA_N

    comptime accum_type = get_accum_type[q_type]()
    comptime frag_size = get_fragment_size[mma_shape]()
    comptime p_frag_size = frag_size[2]
    comptime p_frag_simdwidth = p_frag_size // 2

    var p_reg_tile = LayoutTensor[
        accum_type,
        Layout.row_major(num_m_mmas * num_n_mmas, p_frag_size),
        MutAnyOrigin,
        address_space=.LOCAL,
    ].stack_allocation()

    var output_reg_tile = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas_output, p_frag_size),
            MutAnyOrigin,
            address_space=.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # Rowwise max and sum for online softmax
    comptime row_alignment = align_of[
        SIMD[accum_type, simd_width_of[accum_type]()]
    ]()
    var rowmax = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()
    var rowsum = unsafe_stack_allocation[
        WM, accum_type, alignment=row_alignment
    ]()

    comptime for i in range(0, WM, 2):
        rowmax.store(i, SIMD[accum_type, 2](min_or_neg_inf[accum_type]()))
        rowsum.store(i, SIMD[accum_type, 2](0))

    # Shared memory for P = Q * K^t
    # This overlaps key tile but are used at the same time i.e. no race condition.
    var p_smem = (v_smem + v_smem_size).bitcast[Scalar[v_type]]()
    comptime IteratorTypeP = LayoutTensorIter[
        v_type,
        Layout.row_major(BM, BK),
        _,
        address_space=.SHARED,
        circular=True,
    ]
    var p_smem_iter = IteratorTypeP(
        p_smem, IteratorTypeP.layout_uint_type(BM * BN)
    )

    # Scratch shared memory for reduction across warps.
    var warp_scratch = LayoutTensor[
        accum_type,
        Layout.row_major(2 * num_warps_n, BM),
        address_space=.SHARED,
    ](
        (p_smem + (BM * BN if num_warps_n > 1 else 0)).bitcast[
            Scalar[accum_type]
        ]()
    )

    # Mask global memory iterator.
    var mask_block_row = q_tile_idx * UInt32(BM)
    var mask_warp_row = warp_y * UInt32(WM)
    var mask_warp_col = warp_x * UInt32(WN)

    comptime num_pipeline_stages = config.num_pipeline_stages

    comptime q_num_vecs = BM * BK // simd_size

    comptime async_copy_q_layout = Layout.row_major(
        min(num_threads, q_num_vecs) * simd_size // BK,
        BK // simd_size,
    )

    comptime for q_id in range(q_depth // BK):
        var q_smem_tile = q_smem_iter.next_unsafe(
            q_smem_iter.linear_uint_type(q_id)
        )[]

        copy_dram_to_sram_async[
            thread_layout=async_copy_q_layout,
            swizzle=True,
            num_threads=num_threads,
        ](
            q_smem_tile.vectorize[1, simd_size](),
            q_gmem_iter[].vectorize[1, simd_size](),
        )

        async_copy_commit_group()

        q_gmem_iter._incr()

    # Iterate over KV, equivalent to the following with if hoisted out.
    #   ```
    #   for i in range(kv_tile_start_row, seq_len, tile_size):
    #     if i + tile_size >= seq_len:
    #       loop_over_kvcache[tile_size, False]
    #     else:
    #       loop_over_kvcache[tile_size, True]
    #   ```
    # Only the last iteration is doing boundary check.
    @always_inline
    def loop_over_kvcache[
        tile_size: Int, not_last_iter: Bool
    ](kv_tile_start_row: Int, end: Int) {
        var seq_len,
        var num_keys,
        var start_pos,
        mut mask_warp_col,
        mut k_smem_iter,
        mut v_smem_iter,
        imm,
    }:
        if (
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    Int(UInt32(kv_tile_start_row) + cache_start_pos),
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.FULL_MASK
        ):
            return

        comptime kv_gmem_layout = Layout(
            IntTuple(BN, depth),
            IntTuple(num_heads * depth, 1),
        )

        var kv_tile_num_rows = min(tile_size, end - kv_tile_start_row)

        # kv cache gmem has to clip num rows as runtime layout
        var kv_runtime_layout = RuntimeLayout[
            element_type=.int32, linear_idx_type=.int32
        ](
            RuntimeTuple[kv_gmem_layout.shape, element_type=.int32](
                kv_tile_num_rows, depth
            ),
            RuntimeTuple[kv_gmem_layout.stride, element_type=.int32](
                num_heads * depth, 1
            ),
        )

        var k_gmem_block = LayoutTensor[
            k_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            k.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx)),
                0,
            ),
            kv_runtime_layout,
        )
        var k_gmem_iter = k_gmem_block.tiled_iterator[BN, BK, axis=1](0, 0)

        var v_gmem_block = LayoutTensor[
            v_type,
            kv_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            v.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row),
                UInt32(Int(head_idx)),
                0,
            ),
            kv_runtime_layout,
        )
        var v_gmem_iter = v_gmem_block.tiled_iterator[BK, depth, axis=0](0, 0)

        # here we set up variables for k_rope tensor
        comptime k_rope_gmem_layout = Layout(
            IntTuple(BN, cache_depth),
            IntTuple(cache_num_heads * cache_depth, 1),
        )

        var k_rope_runtime_layout = RuntimeLayout[
            element_type=.int32, linear_idx_type=.int32
        ](
            RuntimeTuple[k_rope_gmem_layout.shape, element_type=.int32](
                kv_tile_num_rows, cache_depth
            ),
            RuntimeTuple[k_rope_gmem_layout.stride, element_type=.int32](
                cache_num_heads * cache_depth, 1
            ),
        )

        var k_rope_gmem_block = LayoutTensor[
            k_rope_type,
            k_rope_gmem_layout,
            layout_int_type=.int32,
            linear_idx_type=.int32,
            masked=not not_last_iter,
        ](
            k_rope.block_paged_ptr[BN](
                UInt32(batch_idx),
                UInt32(kv_tile_start_row) + cache_start_pos,
                UInt32(Int(head_idx // UInt32(group))),
                UInt32(cache_depth - rope_depth),
            ),
            k_rope_runtime_layout,
        )
        var k_rope_gmem_iter = k_rope_gmem_block.tiled_iterator[BN, BK, axis=1](
            0, 0
        )

        # P = Q @ K, register tile holding mma result.
        _ = p_reg_tile.fill(0)

        @always_inline
        def _mask_tensor_row(
            tensor: LayoutTensor, num_rows: Int, out result: type_of(tensor)
        ) {imm}:
            return {
                tensor.ptr,
                {{num_rows, tensor.dim[1]()}, tensor.runtime_layout.stride},
            }

        comptime kv_num_vecs = BN * BK // simd_size
        comptime async_copy_k_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // k_smem_iter.layout.stride[0].value(),
            k_smem_iter.layout.stride[0].value() // simd_size,
        )

        # load K tile into smem
        comptime for k_id in range(depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.layout_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_gmem_iter[].vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            k_gmem_iter._incr()

        comptime for k_id in range(depth // BK, q_depth // BK):
            var k_smem_tile = k_smem_iter.next_unsafe(
                k_smem_iter.linear_uint_type(k_id)
            )[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_k_layout,
                swizzle=True,
                num_threads=num_threads,
            ](
                k_smem_tile.vectorize[1, simd_size](),
                k_rope_gmem_iter[].vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            k_rope_gmem_iter._incr()

        # synchronize here since we can overlap q tile and first k tile copy
        async_copy_wait_all()
        barrier()

        multistage_mma[
            BM,
            BN,
            BK,
            WM,
            WN,
            num_threads,
            num_pipeline_stages,
            True,  # transpose_b
            swizzle_a=True,
            prefetch_init=False,
            static_num_iters=q_depth // BK,
            k_group_size=config.k_group_size,
        ](
            p_reg_tile,
            q_smem_iter,
            k_smem_iter,
            q_smem_iter,
            k_smem_iter,
            q_depth // BK,
        )

        # Vectorize by 2.
        var p_reg_vec2 = p_reg_tile.vectorize[1, p_frag_simdwidth]()

        def _apply_mask[masked: Bool]() {imm}:
            var scale_log2e: Scalar[accum_type] = (
                scale.cast[
                    accum_type
                ]() if mask_t.apply_log2e_after_mask else scale.cast[
                    accum_type
                ]()
                * log2e
            )

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    comptime mma_id = n_mma * num_m_mmas + m_mma

                    # Coordinates in mask for current mma tile.
                    var mask_frag_row = mask_warp_row + UInt32(m_mma * MMA_M)
                    var mask_frag_col = mask_warp_col + UInt32(n_mma * MMA_N)

                    # Offset to current thread's fragment
                    mask_frag_row += lane // UInt32(MMA_N // p_frag_simdwidth)
                    mask_frag_col += (
                        lane * UInt32(p_frag_simdwidth) % UInt32(MMA_N)
                    )

                    comptime for i in range(2):
                        # The row in score matrix of shape seq_len x num_keys.
                        # Mask col is score col since we don't partition in col.
                        var score_row = (
                            mask_block_row
                            + mask_frag_row
                            + UInt32(i * MMA_M // 2)
                        )
                        var score_col = mask_frag_col

                        var score_row_with_start_pos = score_row + start_pos
                        var score_col_with_cache_start_pos = (
                            score_col + cache_start_pos
                        )

                        comptime if masked:
                            p_reg_vec2[mma_id, i] = mask.mask(
                                IndexList[4, element_type=.uint32](
                                    block_idx.z,
                                    block_idx.y,
                                    Int(score_row_with_start_pos),
                                    Int(score_col_with_cache_start_pos),
                                ),
                                p_reg_vec2[mma_id, i] * scale_log2e,
                            )
                        else:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * scale_log2e
                            )

                        comptime if mask_t.apply_log2e_after_mask:
                            p_reg_vec2[mma_id, i] = (
                                p_reg_vec2[mma_id, i] * log2e
                            )

                        if not not_last_iter:
                            p_reg_vec2[mma_id, i] = _kernel_mask(
                                IndexList[2, element_type=.uint32](
                                    Int(score_row), Int(score_col)
                                ),
                                IndexList[2, element_type=.uint32](
                                    seq_len,
                                    num_keys,
                                ),
                                p_reg_vec2[mma_id, i],
                            )

        unswitch(
            mask.status(
                UInt32(batch_idx),
                Index[dtype=DType.uint32](
                    Int(q_tile_idx * UInt32(BM) + start_pos),
                    UInt32(kv_tile_start_row) + cache_start_pos,
                ),
                Index[dtype=DType.uint32](BM, BN),
            )
            == TileMaskStatus.PARTIAL_MASK,
            _apply_mask,
        )

        # Increment mask to next BM x BN block.
        mask_warp_col += UInt32(BN)

        comptime reg_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas, 2
        )
        comptime reg_output_layout_by_mma_unit = Layout.row_major(
            2 * num_m_mmas * num_n_mmas_output, 2
        )

        _online_softmax_iter_for_mma_output[
            accum_type,
            # score layout by mma unit
            # TODO: generalize beyond 16x8 layout
            Layout.row_major(2 * num_m_mmas, num_n_mmas),
            # threads layout by warp
            Layout.row_major(num_warps_m, num_warps_n),
            Layout.row_major(8, 4),
            use_exp2=True,
        ](
            output_reg_tile.reshape[reg_output_layout_by_mma_unit]().vectorize[
                1, 2
            ](),
            p_reg_tile.reshape[reg_layout_by_mma_unit]().vectorize[1, 2](),
            warp_scratch.tile[num_warps_n, WM](0, Int(warp_y)),
            rowmax,
            rowsum,
        )

        comptime async_copy_v_layout = Layout.row_major(
            min(num_threads, kv_num_vecs)
            * simd_size
            // v_smem_iter.layout.stride[0].value(),
            v_smem_iter.layout.stride[0].value() // simd_size,
        )

        var v_tensor: type_of(v_gmem_iter[])
        # load V tile into smem
        comptime for v_id in range(BN // BK):
            var v_smem_tile = v_smem_iter.next_unsafe(
                v_smem_iter.layout_uint_type(v_id)
            )[]

            comptime if not not_last_iter:
                var num_rows_bound = min(
                    BK, end - (kv_tile_start_row + v_id * BK)
                )
                v_tensor = _mask_tensor_row(v_gmem_iter[], num_rows_bound)
            else:
                v_tensor = v_gmem_iter[]

            copy_dram_to_sram_async[
                thread_layout=async_copy_v_layout,
                swizzle=v_smem_tile.dtype.is_half_float(),
                num_threads=num_threads,
            ](
                v_smem_tile.vectorize[1, simd_size](),
                v_tensor.vectorize[1, simd_size](),
            )

            async_copy_commit_group()

            v_gmem_iter._incr()

        comptime if num_warps_n > 1:
            # Pack the per-thread fragments in shared memory for 2nd mma.
            _copy_frag_to_smem[
                BM,
                BN,
                BK,
                WM,
                WN,
                MMA_M,
                MMA_N,
                p_frag_simdwidth,
            ](p_smem_iter, p_reg_tile, warp_x, warp_y)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                depth,
                BK,
                WM,
                depth,
                num_threads,
                num_pipeline_stages,
                False,  # transpose_b
                swizzle_a=True,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                output_reg_tile,
                p_smem_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

        else:
            # Reuse 1st mma output (MMA_M, MMA_N) as 2nd mma's input (MMA_M, MMA_K).
            # The num_n_mmas dim becomes "num_k_mmas" for 2nd mma.
            var p_reg_iter = p_reg_tile.tiled_iterator[
                MMA_K // MMA_N * num_m_mmas, p_frag_size
            ](0, 0)

            async_copy_wait_all()
            barrier()

            multistage_mma[
                BM,
                depth,
                BK,
                WM,
                depth,
                num_threads,
                2,
                False,  # transpose_b
                swizzle_a=False,
                prefetch_init=False,
                static_num_iters=ufloordiv(BN, BK),
                k_group_size=config.k_group_size,
            ](
                output_reg_tile,
                p_reg_iter,
                v_smem_iter,
                p_smem_iter,
                v_smem_iter,
                ufloordiv(BN, BK),
            )

    tile_and_unswitch[[BN]](0, num_keys, loop_over_kvcache)

    comptime output_gmem_layout = Layout(
        IntTuple(BM, depth), IntTuple(num_heads * depth, 1)
    )

    var output_offset = UInt32(depth) * (
        head_idx + UInt32(num_heads) * q_tile_idx * UInt32(BM)
    )
    var output_gemm_runtime_layout = RuntimeLayout[
        element_type=.int32, linear_idx_type=.int32
    ](
        RuntimeTuple[output_gmem_layout.shape, element_type=.int32](
            Int(q_tile_num_rows), depth
        ),
        RuntimeTuple[output_gmem_layout.stride, element_type=.int32](
            num_heads * depth, 1
        ),
    )
    var output_gmem_tile = LayoutTensor[
        output_type,
        output_gmem_layout,
        layout_int_type=.int32,
        linear_idx_type=.int32,
        masked=True,
    ](
        output_ptr + Int(output_offset),
        output_gemm_runtime_layout,
    )
    var output_gmem_warp_tile = output_gmem_tile.tile[WM, WN_O](
        Int(warp_y), Int(warp_x)
    )

    # Apply softmax denumerator.
    comptime for m_mma in range(num_m_mmas):
        var rowsum_inv0 = recip(rowsum[2 * m_mma])
        var rowsum_inv1 = recip(rowsum[2 * m_mma + 1])

        comptime for n_mma in range(num_n_mmas_output):
            comptime for i in range(p_frag_size // 2):
                output_reg_tile[n_mma * num_m_mmas + m_mma, i] *= rowsum_inv0
                output_reg_tile[
                    n_mma * num_m_mmas + m_mma, i + p_frag_size // 2
                ] *= rowsum_inv1

    # Write to global memory.
    comptime if output_type.is_half_float():
        comptime swizzle = make_swizzle[
            num_rows=MMA_M // 2, row_size=depth, access_size=MMA_N
        ]()
        # Reuse a_smem for c tile in smem
        var accum_smem_tile = LayoutTensor[
            output_type,
            Layout.row_major(BM, depth),
            address_space=.SHARED,
        ](q_smem.bitcast[Scalar[output_type]]())

        var accum_smem_warp_tile = accum_smem_tile.tile[WM, depth](
            Int(warp_y), Int(warp_x)
        )
        copy_local_to_shared[
            thread_layout=Layout.row_major(8, 4), swizzle=swizzle
        ](
            accum_smem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )

        # Guard writing to shared memory.
        barrier()

        # Vectorized copy from shared to global memory, during which every 2 FP32
        # are cast to 2 BF16 so that 2 4xFP32 vectors are merged into 1 8xBF16
        # vector and stored using 16B store instruction.
        copy_sram_to_dram[
            thread_layout=Layout.row_major(
                num_threads * simd_size // depth,
                depth // simd_size,
            ),
            swizzle=swizzle,
        ](
            output_gmem_tile.vectorize[1, simd_size](),
            accum_smem_tile.vectorize[1, simd_size](),
        )
    else:
        copy_local_to_dram[dst_thread_layout=Layout.row_major(8, 4)](
            output_gmem_warp_tile.vectorize[1, 2](),
            output_reg_tile.vectorize[1, 2]().transpose(),
        )


# ===-----------------------------------------------------------------------===#
# Helper function that creates cache_row_offsets for the MLA prefill kernel
# ===-----------------------------------------------------------------------===#


def set_buffer_lengths_to_zero[
    BufferLengthsLayoutType: TensorLayout,
](
    buffer_lengths: TileTensor[
        mut=True, .int32, BufferLengthsLayoutType, MutUntrackedOrigin
    ],
):
    """Zeroes out every element of a 1D buffer-lengths tensor.

    Used as the empty-batch fallback in `mla_prefill_plan` so downstream
    prefill iterations see no work to process.

    Parameters:
        BufferLengthsLayoutType: Layout type of the `buffer_lengths`
            tensor (inferred).

    Args:
        buffer_lengths: 1D `int32` tensor of per-chunk buffer lengths
            to be zeroed. Must have `flat_rank == 1`.
    """
    comptime assert buffer_lengths.flat_rank == 1
    comptime MAX_CHUNKS = buffer_lengths.static_shape[0]

    comptime for chunk_idx in range(MAX_CHUNKS):
        buffer_lengths[chunk_idx] = 0


@always_inline
def mla_prefill_plan[
    cache_t: KVCacheT,
](
    buffer_row_offsets: TileTensor[mut=True, .uint32, ...],
    cache_offsets: TileTensor[mut=True, .uint32, ...],
    buffer_lengths: TileTensor[mut=True, .int32, ...],
    input_row_offsets: TileTensor[mut=False, .uint32, ...],
    k_cache: cache_t,
    buffer_token_size: UInt32,
    ctx: DeviceContext,
) raises:
    """
    This calls a GPU kernel that plans how to process a batch of sequences with
    varying lengths using a fixed-size buffer.

    Each sequence in the batch has some existing cached tokens and new input
    tokens. The kernel divides the total tokens into chunks of buffer_token_size.

    For each chunk (iteration), it calculates:
        1. Buffer offsets for each sequence in each chunk
        2. Cache offsets for each sequence in each chunk
        3. Total buffer lengths for each processing iteration

    Parameters:
        cache_t: Paged KV cache type backing `k_cache` (inferred).
            Carries the KV layout, dtype, and page size.

    Args:
        buffer_row_offsets: Output `[MAX_CHUNKS, batch_size + 1]`
            tensor of per-chunk buffer row offsets.
            `buffer_row_offsets[chunk_idx, seq_idx]` is the starting
            row offset within the buffer for sequence `seq_idx` in
            chunk `chunk_idx`.
        cache_offsets: Output `[MAX_CHUNKS, batch_size]` tensor of
            per-chunk cache offsets. `cache_offsets[chunk_idx,
            seq_idx]` is the starting offset into the paged KV cache
            for sequence `seq_idx` in chunk `chunk_idx`.
        buffer_lengths: Output `[MAX_CHUNKS]` tensor of per-chunk
            total buffer lengths. `buffer_lengths[chunk_idx]` is the
            total number of valid rows across all sequences in
            chunk `chunk_idx`; -1 marks unused chunks.
        input_row_offsets: Input `[batch_size + 1]` tensor of
            cumulative row offsets; `offsets[b]` to `offsets[b+1]`
            gives batch `b`'s new token range.
        k_cache: Paged KV cache providing per-sequence cache lengths.
        buffer_token_size: Fixed buffer size in tokens; each chunk
            processes at most this many tokens per sequence.
        ctx: Device context used to enqueue the kernel.
    """
    var batch_size: Int = Int(input_row_offsets.dim[0]()) - 1

    if batch_size == 0:
        # Fill buffer lengths with 0
        comptime kernel = set_buffer_lengths_to_zero[buffer_lengths.LayoutType,]
        ctx.enqueue_function[kernel](buffer_lengths, grid_dim=1, block_dim=1)
    else:
        comptime kernel = mla_prefill_plan_kernel[
            buffer_row_offsets.LayoutType,
            cache_offsets.LayoutType,
            buffer_lengths.LayoutType,
            input_row_offsets.LayoutType,
            cache_t,
        ]

        ctx.enqueue_function[kernel](
            buffer_row_offsets,
            cache_offsets,
            buffer_lengths,
            input_row_offsets.as_immut(),
            k_cache,
            buffer_token_size,
            grid_dim=(ceildiv(batch_size, 128), 1, 1),
            block_dim=(128, 1, 1),
        )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"mla_prefill_plan")
def mla_prefill_plan_kernel[
    BufferRowOffsetsLayoutType: TensorLayout,
    CacheOffsetsLayoutType: TensorLayout,
    BufferLengthsLayoutType: TensorLayout,
    InputRowOffsetsLayoutType: TensorLayout,
    cache_t: KVCacheT,
](
    buffer_row_offsets: TileTensor[
        mut=True, .uint32, BufferRowOffsetsLayoutType, MutUntrackedOrigin
    ],
    cache_offsets: TileTensor[
        mut=True, .uint32, CacheOffsetsLayoutType, MutUntrackedOrigin
    ],
    buffer_lengths: TileTensor[
        mut=True, .int32, BufferLengthsLayoutType, MutUntrackedOrigin
    ],
    input_row_offsets: TileTensor[
        .uint32, InputRowOffsetsLayoutType, ImmUntrackedOrigin
    ],
    k_cache: cache_t,
    buffer_token_size: UInt32,
):
    """Plans how to process a batch of varying-length sequences through a fixed-size buffer.

    For each sequence, computes the per-chunk buffer row offsets, cache offsets,
    and total buffer lengths needed to divide the cached plus new tokens into
    fixed-size chunks aligned to the page size, enabling the MLA prefill kernel
    to process sequences that exceed the buffer in multiple iterations.

    Parameters:
        BufferRowOffsetsLayoutType: Compile-time layout of
            `buffer_row_offsets` (inferred).
        CacheOffsetsLayoutType: Compile-time layout of `cache_offsets`
            (inferred).
        BufferLengthsLayoutType: Compile-time layout of `buffer_lengths`
            (inferred).
        InputRowOffsetsLayoutType: Compile-time layout of
            `input_row_offsets` (inferred).
        cache_t: Paged KV cache type backing `k_cache` (inferred).
            Carries the KV layout, dtype, and page size.

    Args:
        buffer_row_offsets: Output `[MAX_CHUNKS, batch_size + 1]`
            tensor of per-chunk buffer row offsets.
            `buffer_row_offsets[chunk_idx, seq_idx]` is the starting
            row offset within the buffer for sequence `seq_idx` in
            chunk `chunk_idx`.
        cache_offsets: Output `[MAX_CHUNKS, batch_size]` tensor of
            per-chunk cache offsets. `cache_offsets[chunk_idx,
            seq_idx]` is the starting offset into the paged KV cache
            for sequence `seq_idx` in chunk `chunk_idx`.
        buffer_lengths: Output `[MAX_CHUNKS]` tensor of per-chunk
            total buffer lengths. `buffer_lengths[chunk_idx]` is the
            total number of valid rows across all sequences in
            chunk `chunk_idx`; -1 marks unused chunks.
        input_row_offsets: Input `[batch_size + 1]` tensor of
            cumulative row offsets; `offsets[b]` to `offsets[b+1]`
            gives batch `b`'s new token range.
        k_cache: Paged KV cache providing per-sequence cache lengths.
        buffer_token_size: Fixed buffer size in tokens; each chunk
            processes at most this many tokens per sequence.
    """
    comptime assert buffer_row_offsets.flat_rank == 2
    comptime assert cache_offsets.flat_rank == 2
    comptime assert buffer_lengths.flat_rank == 1
    comptime assert input_row_offsets.flat_rank == 1

    var seq_idx = global_idx.x
    var seq_start_pos = 0
    var batch_size: Int = Int(input_row_offsets.dim[0]()) - 1
    var buffer_size: Int = Int(buffer_token_size)

    comptime MAX_CHUNKS = buffer_lengths.static_shape[0]
    comptime page_size = cache_t.page_size_
    comptime assert page_size != 0, "Only PagedKVCache is supported."

    if seq_idx >= batch_size:
        return

    # Calculate starting position for this sequence.
    # Note: the total cache length of each sequence is aligned to the page size.
    var prev_row_offset = Int(input_row_offsets[0])
    for i in range(seq_idx):
        # The cache length that has been prefilled in previous forward passes.
        var cache_length = k_cache.cache_length(i)

        # account for the new input tokens.
        var row_offset_i = Int(input_row_offsets[i + 1])
        cache_length += row_offset_i - prev_row_offset
        prev_row_offset = row_offset_i

        seq_start_pos += align_up(cache_length, page_size)

    var curr_seq_len = align_up(
        k_cache.cache_length(seq_idx)
        + Int(input_row_offsets[seq_idx + 1])
        - prev_row_offset,
        page_size,
    )

    # which chunk this sequence starts in
    var start_chunk = seq_start_pos // buffer_size
    var processed_seq_len = UInt32(0)
    var seq_len_left = curr_seq_len

    # Fill buffer offsets for this sequence
    comptime for chunk_idx in range(MAX_CHUNKS):
        if chunk_idx < start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = UInt32(buffer_size)
        elif chunk_idx == start_chunk:
            buffer_row_offsets[chunk_idx, seq_idx] = UInt32(
                seq_start_pos % buffer_size
            )
        else:
            buffer_row_offsets[chunk_idx, seq_idx] = 0

        cache_offsets[chunk_idx, seq_idx] = processed_seq_len

        var chunk_len = min(
            seq_len_left,
            buffer_size - Int(buffer_row_offsets[chunk_idx, seq_idx]),
        )
        processed_seq_len += UInt32(chunk_len)
        seq_len_left -= chunk_len

    # If this is the last sequence in the batch
    if seq_idx == batch_size - 1:
        var seq_end_pos = seq_start_pos + curr_seq_len
        var end_chunk = ceildiv(seq_end_pos, buffer_size) - 1

        # Set buffer lengths for all chunks
        comptime for chunk_idx in range(MAX_CHUNKS):
            if chunk_idx < end_chunk:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = UInt32(buffer_size)
                buffer_lengths[chunk_idx] = Int32(buffer_size)
            elif chunk_idx == end_chunk:
                var last_chunk_len = seq_end_pos - end_chunk * buffer_size
                buffer_row_offsets[chunk_idx, seq_idx + 1] = UInt32(
                    last_chunk_len
                )
                buffer_lengths[chunk_idx] = Int32(last_chunk_len)
            else:
                buffer_row_offsets[chunk_idx, seq_idx + 1] = 0
                buffer_lengths[chunk_idx] = -1


# ===-----------------------------------------------------------------------===#
# Helper function that copies K cache to a contiguous buffer
# ===-----------------------------------------------------------------------===#


@always_inline
def _k_cache_to_buffer[
    dtype: DType,
    cache_t: KVCacheT,
    BufferRowOffsetsLayoutType: TensorLayout,
    CacheOffsetsLayoutType: TensorLayout,
](
    buffer_row_offsets: TileTensor[.uint32, BufferRowOffsetsLayoutType, ...],
    cache_offsets: TileTensor[.uint32, CacheOffsetsLayoutType, ...],
    k_cache: cache_t,
    length: Int32,
    buffer: TileTensor[mut=True, dtype, ...],
    context: DeviceContext,
) raises:
    comptime num_heads = cache_t.kv_params.num_heads
    comptime assert num_heads == 1, "num_heads should be equal to 1"
    comptime assert buffer.rank == 2, "buffer should be rank 2"
    comptime assert buffer_row_offsets.flat_rank == 1
    comptime assert cache_offsets.flat_rank == 1

    @always_inline
    @__parameter
    @__copy_capture(k_cache, buffer_row_offsets, cache_offsets)
    def copy_fn[
        width: Int, rank: Int, alignment: Int = 1
    ](idx: IndexList[rank]):
        comptime assert rank == 2, "rank should be equal to 2"

        var global_token_idx = idx[0]

        var batch_idx: Int = get_batch_from_row_offsets(
            buffer_row_offsets, global_token_idx
        )

        var token_idx = (
            global_token_idx
            - Int(buffer_row_offsets[batch_idx])
            + Int(cache_offsets[batch_idx])
        )

        var head_dim_idx = idx[1]

        var cache_val = rebind[SIMD[dtype, width]](
            k_cache.load[width=width](
                batch_idx, 0, token_idx, head_dim_idx
            ).cast[dtype]()
        )

        buffer.store_linear(idx, cache_val)

    var launch_shape = Coord(Int(length), Int(buffer.dim[1]()))
    comptime target_simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    def copy_fn_unified[width: Int, alignment: Int = 1](idx: Coord):
        copy_fn[width, idx.rank, alignment](coord_to_index_list(idx))

    _elementwise_impl_gpu[
        simd_width=target_simd_width,
        trace_description="mla_k_cache_copy",
    ](copy_fn_unified, shape=launch_shape, ctx=context)
