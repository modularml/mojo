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
"""SM100 (Blackwell) MLA prefill kernels.

Provides dense and sparse multi-latent attention (MLA) prefill entry points
for NVIDIA SM100 GPUs, including the generic (non-blockscale), blockwise-scale,
and sparse (with optional FP8 KV cache) variants.
"""

from std.memory import UnsafePointer

from kv_cache.types import KVCacheT
from nn.attention.mha_operand import MHAOperand
from nn.attention.mha_utils import MHAConfig, OptionallyStaticInt
from nn.attention.mha_mask import MHAMask
from max.gpu.host import DeviceContext
from layout import TileTensor
from .mla_prefill_generic import mla_sm100_prefill_generic
from .mla_prefill_blockscale import mla_sm100_prefill_blockscale
from .mla_prefill_sparse_utils import MLASparseConfig
from .mla_prefill_sparse import mla_prefill_sparse
from .mla_prefill_sparse_kv_fp8 import mla_prefill_sparse_fp8


@always_inline
def mla_sm100_prefill[
    output_type: DType,
    q_type: DType,
    KVType: MHAOperand,
    VType: MHAOperand,
    KRopeType: MHAOperand,
    MaskType: MHAMask,
    MaxPromptLenType: OptionallyStaticInt,
    //,
    config: MHAConfig,
    group: Int,
    q_depth: Int,
    cache_depth: Int,
    _ndbuffer_mha_operand: Bool,
    blockwise_scale: Int = 0,
    # -1 => V width == nope width (DeepSeek); resolved in `MLAConfig.__init__`.
    v_depth: Int = -1,
](
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    k: KVType,
    v: VType,
    k_rope: KRopeType,
    mask_functor: MaskType,
    valid_length: TileTensor[.uint32, address_space=.GENERIC, ...],
    max_prompt_len: MaxPromptLenType,
    scale: Float32,
    batch_size: Int,
    ctx: DeviceContext,
) raises:
    """Dense MLA prefill dispatcher for SM100 (Blackwell).

    Routes to ``mla_sm100_prefill_generic`` when ``blockwise_scale`` is zero
    and the query, key, value, and rope dtypes all match, otherwise routes
    to ``mla_sm100_prefill_blockscale``. Asserts that the output is BF16 and
    that the key and value share an element dtype.

    Parameters:
        output_type: Output element type (must be ``DType.bfloat16``).
        q_type: Query element type.
        KVType: Key operand type (an ``MHAOperand``).
        VType: Value operand type (an ``MHAOperand``).
        KRopeType: Rope key operand type (an ``MHAOperand``).
        MaskType: Attention mask functor type (an ``MHAMask``).
        MaxPromptLenType: Maximum prompt length type (an
            ``OptionallyStaticInt``).
        config: MHA configuration struct.
        group: Number of query heads per KV head (GQA group size).
        q_depth: Per-head query depth.
        cache_depth: Per-head KV cache depth.
        _ndbuffer_mha_operand: Whether operands are ND buffers.
        blockwise_scale: Blockwise quantization scale size; zero disables
            blockwise scaling and selects the generic path.
        v_depth: Per-head value depth; ``-1`` resolves to the nope width
            (DeepSeek convention) inside ``MLAConfig.__init__``.

    Args:
        output: Output tile tensor with shape
            ``[total_q_tokens, num_q_heads, v_depth]``.
        q: Query tile tensor.
        k: Key operand.
        v: Value operand.
        k_rope: Rope-applied key operand.
        mask_functor: Attention mask functor.
        valid_length: Per-sequence valid lengths (uint32).
        max_prompt_len: Maximum prompt length (static or dynamic).
        scale: Softmax scale.
        batch_size: Number of sequences in the batch.
        ctx: GPU device context.
    """
    comptime assert (
        output_type == .bfloat16
    ), "Only support bfloat16 output for SM100 MLA prefill"
    comptime assert (
        KVType.dtype == VType.dtype
    ), "k and v must share an element dtype for SM100 MLA prefill"

    comptime if blockwise_scale == 0 and (
        KRopeType.dtype == KVType.dtype == q.dtype
    ):
        comptime assert (
            blockwise_scale == 0
        ), "blockwise_scale is not supported for generic MLA prefill"
        mla_sm100_prefill_generic[
            config=config,
            group=Int(group),
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
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )
    else:
        mla_sm100_prefill_blockscale[
            config=config,
            group=Int(group),
            q_depth=q_depth,
            cache_depth=cache_depth,
            _ndbuffer_mha_operand=_ndbuffer_mha_operand,
            blockwise_scale=blockwise_scale,
            v_depth=v_depth,
        ](
            output,
            q,
            k,
            v,
            k_rope,
            mask_functor,
            valid_length,
            max_prompt_len,
            scale,
            batch_size,
            ctx,
        )


@always_inline
def mla_sm100_prefill_sparse[
    output_type: DType,
    q_type: DType,
    cache_t: KVCacheT,
    //,
    num_q_heads: Int,
    qk_depth: Int,
    v_depth: Int,
    indices_stride: Int,
](
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """Sparse MLA prefill (DSv3.2 absorbed shape, BF16, SM100).

    Thin wrapper around ``mla_prefill_sparse`` that builds the
    ``MLASparseConfig`` from the passed dimensions so callers don't have to
    reach into the kernel's config type. The kernel targets the DSv3.2
    absorbed/latent shape and comptime-asserts ``qk_depth==576``,
    ``num_q_heads`` is 128 or 64, and ``num_kv_heads==1``, plus equal
    output/query bit-width. ``v_depth`` is assumed to be 512 (=
    ``kv_lora_rank``) by that shape but is not asserted.

    Parameters:
        output_type: Output element type (must be the same width as
            ``q_type``; the kernel asserts this).
        q_type: Query element type (BF16 in the supported DSv3.2 shape).
        cache_t: KV cache type (typically a paged MLA cache obtained from
            ``kv_collection.get_key_cache(layer_idx)``).
        num_q_heads: Number of query heads (asserted to be 128 or 64).
        qk_depth: Per-head Q/K depth (must be 576 = ``kv_lora_rank(512) +
            qk_rope_head_dim(64)``).
        v_depth: Per-head V depth (512 = ``kv_lora_rank`` for the DSv3.2
            absorbed shape; not asserted).
        indices_stride: Per-query indices buffer stride (= the indexer's
            ``index_topk``). Also used as the runtime ``indices_stride`` to
            the kernel.

    Args:
        output: Output tile tensor with shape
            ``[total_q_tokens, num_q_heads, v_depth]``.
        q: Query tile tensor with shape
            ``[total_q_tokens, num_q_heads, qk_depth]``.
        kv_cache: Paged MLA KV cache for the current layer.
        indices: Per-query gather4 indices, encoded as
            ``Int32(physical_block_id * page_size + token_offset_within_page)``
            (reinterpreted via the ``uint32`` tile-tensor view; ``-1``-bit-pattern
            sentinels are masked out by the kernel's k-valid producer).
        topk_lengths: Per-query effective top-k count (``[total_q_tokens]``).
        attn_sink_ptr: Optional attention sink (one ``Float32`` per query head).
            Pass `None` to skip the sink term in the softmax epilogue.
        scale: Softmax scale (``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim) *
            mscale^2``; for DSv3.2 with mscale=1, ``1 / sqrt(192)``).
        ctx: GPU device context.
    """
    # head128 uses 2SM (cta_group=2, B_TOPK=128); head64 uses single-CTA WS
    # (cta_group=1, B_TOPK=64), which fits SMEM where a 2SM split would not.
    comptime cta_group = 2 if num_q_heads == 128 else 1
    comptime b_topk = 128 if num_q_heads == 128 else 64
    comptime config = MLASparseConfig[
        q_type, b_topk_=b_topk, cta_group_=cta_group
    ](
        num_q_heads=num_q_heads,
        num_kv_heads=1,
        qk_depth=qk_depth,
        v_depth=v_depth,
        indices_stride=indices_stride,
        group=num_q_heads,
    )
    mla_prefill_sparse[
        config=config,
        group=num_q_heads,
        q_depth=qk_depth,
    ](
        output,
        q,
        kv_cache,
        indices,
        topk_lengths,
        attn_sink_ptr,
        scale,
        Int32(indices_stride),
        ctx,
    )


# SM100 host launch wrapper. Deliberately NOT `@always_inline`: this wrapper's
# transitively-inlined FP8 host launch path (the INT64-packed gather4 K+V TMA
# descriptor construction, `mla_prefill_sparse_fp8` below) is a large store
# chain. If it inlines into the model's giant fused host `region_0`, LLVM's SLP
# vectorizer goes quadratic (`BoUpSLP::calculateDependencies` O(n^2) alias
# queries) and the GLM-5.2 FP8-cache model compile blows up (~66 min vs ~15 for
# bf16-cache). `@no_inline` keeps the store chain in its own small function
# where SLP is cheap. Host-side only — the device kernel body is unchanged.
@no_inline
def mla_sm100_prefill_sparse_fp8[
    output_type: DType,
    q_type: DType,
    cache_t: KVCacheT,
    //,
    num_q_heads: Int,
    qk_depth: Int,
    v_depth: Int,
    indices_stride: Int,
    scale_block_size: Int,
](
    output: TileTensor[output_type, address_space=.GENERIC, ...],
    q: TileTensor[q_type, address_space=.GENERIC, ...],
    kv_cache: cache_t,
    indices: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[UnsafePointer[Float32, ImmutAnyOrigin]],
    scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
    scale: Float32,
    ctx: DeviceContext,
) raises:
    """FP8 KV-cache variant of ``mla_sm100_prefill_sparse``.

    Thin wrapper around ``mla_prefill_sparse_fp8`` that builds the
    ``MLASparseConfig`` from the passed dimensions.  The FP8 KV cache
    uses ``DType.float8_e4m3fn`` storage with one Float32 scale per
    physical KV token supplied via ``scales_ptr``.

    Parameters:
        output_type: Output element type (BF16).
        q_type: Query element type (BF16).
        cache_t: FP8 KV cache type (``DType.float8_e4m3fn``).
        num_q_heads: Number of query heads (128 for DSv3.2).
        qk_depth: Per-head Q/K depth (576 for DSv3.2).
        v_depth: Per-head V depth (512 for DSv3.2).
        indices_stride: Per-query indices buffer stride (= top-k count).
        scale_block_size: Quantization block size along the depth axis.
            ``0`` selects no-scale mode: FP8 latents are read at unit scale
            (no dequant, ``scales_ptr`` unused / may be null), mirroring the
            sparse-decode read of today's scale-less MLA latent cache — the
            current DSv3.2 production path. ``>= qk_depth`` selects tensorwise
            scaling (one scale per KV token). A smaller positive value (e.g.
            32) selects cache-native blockwise scaling: MLA stores one
            ``ceildiv(qk_depth, scale_block_size)``-wide scale vector per token
            over the full latent, K consumes all of it and V the first
            ``ceildiv(v_depth, scale_block_size)`` blocks, so a single
            ``scales_ptr`` (stride ``ceildiv(qk_depth, sbs)``) serves both.

    Args:
        output: Output tile tensor ``[total_q_tokens, num_q_heads, v_depth]``.
        q: Query tile tensor ``[total_q_tokens, num_q_heads, qk_depth]``.
        kv_cache: Paged MLA FP8 KV cache for the current layer.
        indices: Per-query gather4 indices (uint32).
        topk_lengths: Per-query effective top-k count.
        attn_sink_ptr: Optional attention sink (pass ``None`` to skip).
        scales_ptr: FP8 dequantization scales in ``get_tma_row`` space, laid
            out ``[total_phys_rows, ceildiv(qk_depth, scale_block_size)]``
            (i.e. ``k.scales_raw_ptr()`` for the paged cache).
        scale: Softmax scale.
        ctx: GPU device context.
    """
    comptime cta_group = 2 if num_q_heads == 128 else 1
    comptime b_topk = 128 if num_q_heads == 128 else 64
    comptime config = MLASparseConfig[
        q_type, b_topk_=b_topk, cta_group_=cta_group
    ](
        num_q_heads=num_q_heads,
        num_kv_heads=1,
        qk_depth=qk_depth,
        v_depth=v_depth,
        indices_stride=indices_stride,
        group=num_q_heads,
    )
    mla_prefill_sparse_fp8[
        config=config,
        group=num_q_heads,
        q_depth=qk_depth,
        scale_block_size=scale_block_size,
    ](
        output,
        q,
        kv_cache,
        indices,
        topk_lengths,
        attn_sink_ptr,
        scales_ptr,
        scale,
        Int32(indices_stride),
        ctx,
    )
