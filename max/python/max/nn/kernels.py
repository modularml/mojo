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
"""Helper functions for wrapping custom kv cache/attention related ops."""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

import numpy as np
from max._core.dialects import builtin, kgen, mo
from max.driver import accelerator_api, accelerator_architecture_name
from max.dtype import DType
from max.graph import (
    AlgebraicDim,
    BufferValue,
    BufferValueLike,
    DeviceKind,
    DeviceRef,
    Dim,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    TensorValueLike,
    Type,
    Value,
    ops,
)
from max.graph.ops import assert_same_device
from max.graph.ops.quantized import repack_gguf_quantized_weights
from max.graph.quantization import QuantizationConfig, QuantizationEncoding
from max.nn.quant_config import InputScaleSpec, QuantConfig, WeightScaleSpec

from .attention.mask_config import AttentionMaskVariant, MHAMaskVariant
from .kv_cache import KVCacheParams, MHAKVCacheParams, PagedCacheValues

# Elements sharing one MX block scale, matching `MXFP8_SF_VECTOR_SIZE` /
# `MXFP4_SF_VECTOR_SIZE` in `linalg` and the `sf_vector_size=32` the MX
# quantize wrappers take.
_MX_SF_VECTOR_SIZE = 32

_MHA_MASK_VARIANT_TO_ATTENTION_MASK = {
    MHAMaskVariant.CAUSAL_MASK: AttentionMaskVariant.CAUSAL_MASK,
    MHAMaskVariant.NULL_MASK: AttentionMaskVariant.NULL_MASK,
    MHAMaskVariant.CHUNKED_CAUSAL_MASK: (
        AttentionMaskVariant.CHUNKED_CAUSAL_MASK
    ),
    MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK: (
        AttentionMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
    ),
    MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK: (
        AttentionMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
    ),
}

KEY_CACHE_INDEX = 0
VALUE_CACHE_INDEX = 1


def _check_dtype(expected: DType, **tensors: TensorValue | BufferValue) -> None:
    """Raises ``ValueError`` if any tensor kwarg does not have dtype ``expected``

    Note: The kwarg names are used in the error message, so naming matters.
    """
    for name, t in tensors.items():
        if t.dtype != expected:
            raise ValueError(
                f"expected {name} to have dtype {expected.name}, was {t.dtype}"
            )


def _check_rank(expected: int, **tensors: TensorValue | BufferValue) -> None:
    """Raises ``ValueError`` if any tensor kwarg does not have rank ``expected``

    Note: The kwarg names are used in the error message, so naming matters.
    """
    for name, t in tensors.items():
        if t.rank != expected:
            raise ValueError(
                f"expected {name} to have rank {expected}, was {t.rank}"
            )


def _check_same_dtype(**tensors: TensorValue | BufferValue) -> None:
    """Raises ``ValueError`` unless all tensor kwargs share the same dtype;

    Note: The kwarg names are used in the error message, so naming matters.
    """
    first_name, first = next(iter(tensors.items()))
    for name, t in list(tensors.items())[1:]:
        if t.dtype != first.dtype:
            raise ValueError(
                f"expected {first_name} and {name} to have the same dtype, "
                f"but got {first.dtype} and {t.dtype}, respectively."
            )


def _check_same_device(**tensors: TensorValue | BufferValue) -> None:
    """Raises ``ValueError`` unless all tensor kwargs share the same device;

    Note: The kwarg names are used in the error message, so naming matters.
    """
    first_name, first = next(iter(tensors.items()))
    for name, t in list(tensors.items())[1:]:
        if t.device != first.device:
            raise ValueError(
                f"expected {first_name} and {name} to have the same device, "
                f"but got {first.device} and {t.device}, respectively."
            )


def _mask_str(mask_variant: MHAMaskVariant) -> str:
    return _MHA_MASK_VARIANT_TO_ATTENTION_MASK[mask_variant].value


def _mha_parameters(
    mask_variant: MHAMaskVariant,
    *,
    local_window_size: int | None = None,
) -> dict[str, int | str | DType]:
    parameters: dict[str, int | str | DType] = {
        "mask_str": _mask_str(mask_variant)
    }
    if local_window_size is not None:
        parameters["local_window_size"] = local_window_size
    return parameters


def ceildiv(n: Dim, d: Dim) -> Dim:
    """Ceiling division.

    Args:
        n: The numerator.
        d: The denominator.

    Returns:
        The ceiling of dividing n by d.
    """
    return (n + d - 1) // d


def fused_qkv_padded_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    n_heads: int,
) -> TensorValue:
    """Computes fused query, key, and value projections with padded input.

    This is for non-ragged (padded batch) inputs where sequences may have
    different actual lengths but are padded to a uniform shape.

    Args:
        kv_params: KV cache parameters.
        input: Input tensor with shape [batch_size, seq_len, hidden_dim].
        wqkv: Weight tensor for Q, K, V projections.
        kv_collection: Paged KV cache collection.
        layer_idx: Layer index for cache lookup (must be uint32).
        valid_lengths: Buffer of shape [batch] containing the valid length for each
            sequence (must be uint32). K and V are only written to cache for
            positions within these lengths.
        n_heads: Number of attention heads.

    Returns:
        Query projections tensor. K and V projections are written to cache.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 3
    _check_rank(input_rank_expected, input=input)

    _check_dtype(DType.uint32, layer_idx=layer_idx, valid_lengths=valid_lengths)

    _check_rank(1, valid_lengths=valid_lengths)

    return ops.inplace_custom(
        "mo.fused_qkv_matmul.padded.paged",
        device=input.device,
        values=[
            input,
            wqkv,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            valid_lengths,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def fused_qkv_ragged_matmul(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [total_seq_len, hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            request in the input tensor with shape [batch_size + 1].
        wqkv: The concatenated Q, K and V projection weights.
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of Query attention heads.
        bias: Optional bias vector concatenated as [q, k, v].
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Returns:
        Query projection tensor.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    op_name = "mo.fused_qkv_matmul.ragged.paged"
    values = [
        input,
        input_row_offsets,
        wqkv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]

    if bias is not None:
        op_name += ".bias"
        values.append(bias)

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
    )[0].tensor


def rope_split_store_ragged(
    kv_params: KVCacheParams,
    qkv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    interleaved: bool = True,
    position_ids: TensorValue | None = None,
    mrope_section: list[int] | None = None,
    fuse: bool = True,
    q_out_dtype: DType | None = None,
    q_norm_weight: TensorValue | None = None,
    k_norm_weight: TensorValue | None = None,
    rms_norm_eps: float | None = None,
    k_eq_v: bool = False,
) -> TensorValue:
    """Apply rope to Q and K from flat QKV buffer, store K/V to cache.

    Reads from a flat QKV matmul output, applies RoPE to Q and K regions,
    stores K/V to the paged KV cache, and writes roped Q to the output.

    Args:
        kv_params: KV cache parameters.
        qkv: Flat QKV matmul output [total_seq_len, q_dim + k_dim + v_dim].
        input_row_offsets: Ragged offsets [batch_size + 1].
        freqs_cis: RoPE frequencies [max_seq_len, head_dim].
        kv_collection: Paged KV cache.
        layer_idx: Layer index.
        n_heads: Number of query attention heads.
        interleaved: Whether freqs_cis uses interleaved (re, im) format.
        position_ids: Optional ragged 2D array of position IDs. If None,
            defaults to cache_length + token_idx for each token. When
            ``num_sections > 1``, ``mrope_section`` must be provided.
            Shape: [num_sections, total_seq_len].
        mrope_section: Optional list of ints indicating the section of the
            head_dim to apply RoPE to. Must be used with ``position_ids``.
        fuse: If True (default), emit a single fused custom op. If False,
            emit separate split, rope, and store ops for testing graph
            compiler fusion.
        q_out_dtype: Dtype for the roped Q output. Defaults to ``qkv.dtype``.
        q_norm_weight: Optional per-head RMSNorm gamma ``[head_dim]`` for Q. When
            given (with ``k_norm_weight`` and ``rms_norm_eps``), the per-head
            Q/K/V RMS-norm is fused into the op (q/k use their gammas, v is a bare
            norm), removing the separate norm ops. Mutually exclusive with
            ``position_ids``.
        k_norm_weight: Per-head RMSNorm gamma ``[head_dim]`` for K (see
            ``q_norm_weight``).
        rms_norm_eps: Epsilon for the fused qk-norm; required when
            ``q_norm_weight`` is set.
        k_eq_v: When True (only valid with ``q_norm_weight``), V has no own
            projection and reuses K's: ``qkv`` is ``[q|k]`` (no V region) and the
            kernel reads the K head for both the K and V stores, sharing the norm
            reduction. When False (default), ``qkv`` is ``[q|k|v]``.

    Returns:
        Roped Q output [total_seq_len, n_heads * head_dim].
    """
    _check_rank(2, qkv=qkv)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    if kv_params.quantized_kv_cache:
        assert kv_params.kvcache_quant_config is not None
        raise ValueError(
            "rope_split_store_ragged does not support a scaled quantized KV"
            f" cache (dtype={kv_params.dtype},"
            f" scale_dtype={kv_params.kvcache_quant_config.scale_dtype})."
        )

    _check_rank(2, freqs_cis=freqs_cis)

    head_dim = kv_params.head_dim
    q_dim = n_heads * head_dim

    if not fuse:
        return _rope_split_store_ragged_unfused(
            kv_params=kv_params,
            qkv=qkv,
            input_row_offsets=input_row_offsets,
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=n_heads,
            interleaved=interleaved,
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
    }

    if mrope_section is not None and position_ids is None:
        raise ValueError("mrope_section requires position_ids to be provided")

    if position_ids is not None:
        _check_dtype(DType.uint32, position_ids=position_ids)
        _check_rank(2, position_ids=position_ids)
        if mrope_section is not None:
            if len(mrope_section) != position_ids.shape[0]:
                raise ValueError(
                    "expected mrope_section to have length"
                    f" {position_ids.shape[0]}, was {len(mrope_section)}"
                )
            scaled = [x * 2 for x in mrope_section]
            prefix_sums = [sum(scaled[: i + 1]) for i in range(len(scaled))]
            parameters["mrope_section"] = "_".join(str(x) for x in prefix_sums)
        else:
            parameters["mrope_section"] = ""

    if (q_norm_weight is None) != (k_norm_weight is None):
        raise ValueError(
            "q_norm_weight and k_norm_weight must be provided together"
        )
    if q_norm_weight is not None and position_ids is not None:
        raise ValueError(
            "qk-norm fusion and position_ids are not supported together"
        )

    if position_ids is not None:
        op_name = "mo.rope_split_store.ragged.paged.with_position_id"
        values = [
            qkv,
            input_row_offsets,
            freqs_cis,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            position_ids,
            layer_idx,
        ]
    elif q_norm_weight is not None:
        # Fused per-head Q/K/V RMS-norm folded into the RoPE+store op: q/k use
        # the learned gammas, v is a bare norm. `eps` is passed as its integer
        # reciprocal (custom-op params reject float; eps is negligible vs
        # mean(x^2), so this is ample precision).
        assert k_norm_weight is not None
        if rms_norm_eps is None:
            raise ValueError("rms_norm_eps is required with q_norm_weight")
        op_name = "mo.rope_split_store.ragged.paged.with_qk_norm"
        parameters["eps_recip"] = round(1.0 / rms_norm_eps)
        # When `k_eq_v`, V has no projection: `qkv` is `[q|k]` and the kernel
        # reads the K region for both K and V (sharing one norm reduction)
        # rather than a duplicated V region.
        parameters["k_eq_v"] = k_eq_v
        values = [
            qkv,
            input_row_offsets,
            freqs_cis,
            q_norm_weight,
            k_norm_weight,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ]
    else:
        op_name = "mo.rope_split_store.ragged.paged"
        values = [
            qkv,
            input_row_offsets,
            freqs_cis,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ]

    return ops.inplace_custom(
        op_name,
        device=qkv.device,
        values=values,
        out_types=[
            TensorType(
                dtype=q_out_dtype if q_out_dtype is not None else qkv.dtype,
                shape=qkv.shape[:-1] + [q_dim],
                device=qkv.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def store_k_scale_cache_ragged(
    kv_collection: PagedCacheValues,
    x_k_scale: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    quantization_granularity: int,
) -> None:
    """Store key scale tensor into the paged KV cache."""
    if kv_collection.kv_scales is None:
        raise ValueError(
            "kv_collection.kv_scales is None, expected a buffer value"
        )
    ops.inplace_custom(
        "mo.kv_cache.store_k_scales.paged.ragged",
        device=x_k_scale.device,
        values=[
            x_k_scale,
            kv_collection.kv_blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            input_row_offsets,
            kv_collection.max_prompt_length,
            kv_collection.max_cache_length,
            kv_collection.kv_scales,
            kv_collection.scales_lookup_table or kv_collection.lookup_table,
            layer_idx,
        ],
        parameters={
            "quantization_granularity": quantization_granularity,
        },
    )


def _rope_split_store_ragged_unfused(
    kv_params: KVCacheParams,
    qkv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    interleaved: bool,
) -> TensorValue:
    """Unfused rope + split + store for testing graph compiler fusion.

    Emits separate slice, rope, and store ops instead of a single fused
    custom op, so the graph compiler can attempt to fuse them.
    """
    head_dim = kv_params.head_dim
    assert isinstance(kv_params, MHAKVCacheParams)
    n_kv_heads = kv_params.n_kv_heads
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    # Split QKV into Q, K, V.
    x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)

    # Reshape to [total_seq_len, num_heads, head_dim] for rope.
    x_q = x_q.reshape((-1, n_heads, head_dim))
    x_k = x_k.reshape((-1, n_kv_heads, head_dim))
    x_v = x_v.reshape((-1, n_kv_heads, head_dim))

    # Apply RoPE to Q and K individually.
    xq_rope = rope_ragged(
        x_q,
        input_row_offsets,
        kv_collection.cache_lengths,
        freqs_cis,
        interleaved=interleaved,
    )
    xk_rope = rope_ragged(
        x_k,
        input_row_offsets,
        kv_collection.cache_lengths,
        freqs_cis,
        interleaved=interleaved,
    )

    # Store K and V to cache individually.
    kv_blocks = kv_collection.kv_blocks
    cache_lengths = kv_collection.cache_lengths
    lookup_table = kv_collection.lookup_table
    max_prompt_length = kv_collection.max_prompt_length
    max_cache_length = kv_collection.max_cache_length
    ops.inplace_custom(
        "mo.kv_cache.store.paged.ragged",
        device=xk_rope.device,
        values=[
            xk_rope,
            kv_blocks,
            cache_lengths,
            lookup_table,
            input_row_offsets,
            max_prompt_length,
            max_cache_length,
            layer_idx,
        ],
        parameters={"key_or_value": 0},
    )
    ops.inplace_custom(
        "mo.kv_cache.store.paged.ragged",
        device=x_v.device,
        values=[
            x_v,
            kv_blocks,
            cache_lengths,
            lookup_table,
            input_row_offsets,
            max_prompt_length,
            max_cache_length,
            layer_idx,
        ],
        parameters={"key_or_value": 1},
    )

    # Return flat roped Q [total_seq_len, n_heads * head_dim].
    return xq_rope.reshape((-1, q_dim))


def _fused_qkv_ragged_matmul_scaled_float8(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    bias: TensorValue | None = None,
    quant_config: QuantConfig | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with scaled float8 input and weights.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        wqkv: TensorValue representing the weight tensor with shape
            [N=(num_heads + 2 * num_kv_heads) * head_dim, K=hidden_dim].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of attention heads.
        input_scale: TensorValue representing the input scale tensor. Shape
            varies depending on the quantization config.
        weight_scale: TensorValue representing the weight scale tensor. Shape
            varies depending on the quantization config.
        bias: Optional bias vector concatenated as [q, k, v].
        quant_config: Optional QuantConfig object containing scaled
            quantization parameters. If not provided, the quantization config
            will be inferred from the input and weight scale shapes.
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    # Device check - all tensors must be on the same device
    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]
    if bias is not None:
        tensors_to_check.append(bias)

    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            "expected all tensors to be on the same device as input"
            f" ({input.device}), but got:\n  wqkv={wqkv.device}\n "
            f" input_row_offsets={input_row_offsets.device}\n "
            f" input_scale={input_scale.device}\n "
            f" weight_scale={weight_scale.device}"
            + ("" if bias is None else f"\n  bias={bias.device}")
        )

    # layer_idx must be a scalar on CPU as it's used for indexing
    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            "expected layer_idx to be on CPU device, but got"
            f" {layer_idx.device}"
        )

    # for per-tensor quantization, the scale is a scalar. We view it as a 1x1
    # rank-2 tensor so that we can use the same kernel for per-tensor and
    # per-channel quantization.
    if input_scale.shape in [[], [1]]:
        input_scale = input_scale.reshape([1, 1])

    if weight_scale.shape in [[], [1]]:
        weight_scale = weight_scale.reshape([1, 1])

    # Try to infer the quantization config
    if quant_config is not None:
        scales_granularity_mnk = quant_config.scales_granularity_mnk
    else:
        # with out quant_config, we either use per-tensor or per-channel quantization
        # both dynamic and static tensor wise quantization have weight shape [1, 1]
        if (
            input_scale.shape[0] == 1
            and input_scale.shape[1] == 1
            and weight_scale.shape[0] == 1
            and weight_scale.shape[1] == 1
        ):
            scales_granularity_mnk = (-1, -1, -1)  # per-tensor quantization
        elif input_scale.shape[0] == 1 and weight_scale.shape[1] == 1:
            scales_granularity_mnk = (1, 1, -1)  # per-channel quantization
        else:
            raise ValueError(
                (
                    "Can not infer the quantization config from the input"
                    " tensor shapes"
                ),
                "Please provide a quant_config",
            )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "kv_type": kv_params.dtype,
        "m_scale_granularity": scales_granularity_mnk[0],
        "n_scale_granularity": scales_granularity_mnk[1],
        "k_scale_granularity": scales_granularity_mnk[2],
    }

    op_name = "mo.fused_qkv_matmul.ragged.paged.scale"
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]
    if bias is not None:
        op_name += ".bias"
        values.append(bias)

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def _fused_qkv_ragged_matmul_scaled_float4(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    tensor_sf: float | TensorValue,
    kv_scales: TensorValue | None = None,
    sf_vector_size: int = 16,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with scaled float4 input and weights.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        wqkv: TensorValue representing the weight tensor with shape
            [N=(num_heads + 2 * num_kv_heads) * head_dim, K=hidden_dim].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        n_heads: Number of attention heads.
        input_scale: TensorValue representing the input scale tensor. Shape
            for blockwise scaling is 5D, for example, [2, 3, 32, 4, 4].
        weight_scale: TensorValue representing the weight scale tensor. Shape
            for blockwise scaling is 5D, for example, [2, 34, 32, 4, 4]
        tensor_sf: Buffer-wise scaling factor equal to weight_scale_2 * input_scale (pre-quantization, non-inverted).
        kv_scales: TBD, used in NVFP4 KV cache, see: https://github.com/NVIDIA/TensorRT-LLM/blob/0ffa77af51b272ba27424564ed253096d6f0f11a/tensorrt_llm/_torch/modules/linear.py#L690
        _output_dim: Optional output dimension. If not provided, the output
            dimension will be [n_heads * head_dim].

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    # Device check - all tensors must be on the same device
    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]

    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            "expected all tensors to be on the same device as input"
            f" ({input.device}), but got:\n  wqkv={wqkv.device}\n "
            f" input_row_offsets={input_row_offsets.device}\n "
            f" input_scale={input_scale.device}\n "
            f" weight_scale={weight_scale.device}"
        )

    # layer_idx must be a scalar on CPU as it's used for indexing
    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            "expected layer_idx to be on CPU device, but got"
            f" {layer_idx.device}"
        )

    # tensor_sf must be a scalar on CPU as it's used for per-tensor scaling
    if isinstance(tensor_sf, float):
        tensor_sf = ops.constant(
            tensor_sf, DType.float32, device=DeviceRef.CPU()
        )
    elif isinstance(tensor_sf, TensorValue):
        tensor_sf = (
            tensor_sf.cast(DType.float32).to(DeviceRef.CPU()).reshape(())
        )
    else:
        raise ValueError(
            "tensor_sf must be either float or a float32 CPU tensor of rank 0."
        )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "dtype": DType.uint8,
        "scale_type": DType.float8_e4m3fn,
        "kv_type": kv_params.dtype,
        "SF_VECTOR_SIZE": sf_vector_size,
    }

    op_name = "mo.fused_qkv_matmul.ragged.paged.scale.float4"
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        tensor_sf,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def _fused_qkv_ragged_matmul_scaled_mxfp8(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    _output_dim: int | None = None,
) -> TensorValue:
    """Computes fused QKV projections with MXFP8 block-scaled input and weights.

    The MXFP8 sibling of :func:`_fused_qkv_ragged_matmul_scaled_float4`:
    ``input`` and ``wqkv`` carry ``float8_e4m3fn`` data with E8M0
    (``float8_e8m0fnu``) block scales over 32-element K blocks. The Q
    projection is returned, while K and V are written in place into
    ``kv_collection``.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: Activation tensor, ``float8_e4m3fn`` with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        wqkv: Weight tensor, ``float8_e4m3fn`` with shape
            [N=(num_heads + 2 * num_kv_heads) * head_dim, K=hidden_dim].
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: Layer index, expected to have dtype uint32 and live on CPU.
        n_heads: Number of attention heads.
        input_scale: E8M0 input block scales; rank-5 SF-atom layout on SM100,
            rank-2 ``[M, K // 32]`` on CDNA4.
        weight_scale: E8M0 weight block scales; rank-5 SF-atom layout on SM100,
            rank-2 ``[N, K // 32]`` on CDNA4.
        _output_dim: Optional output dimension. Defaults to
            ``n_heads * head_dim``.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]
    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            "expected all tensors to be on the same device as input"
            f" ({input.device}), but got:\n  wqkv={wqkv.device}\n "
            f" input_row_offsets={input_row_offsets.device}\n "
            f" input_scale={input_scale.device}\n "
            f" weight_scale={weight_scale.device}"
        )

    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            "expected layer_idx to be on CPU device, but got"
            f" {layer_idx.device}"
        )

    # MXFP8 block scales fully describe the quantization, but the kernel still
    # multiplies by a per-tensor scale, so pass an identity scale on CPU.
    tensor_sf = ops.constant(1.0, DType.float32, device=DeviceRef.CPU())

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "dtype": DType.float8_e4m3fn,
        "scale_type": DType.float8_e8m0fnu,
        "kv_type": kv_params.dtype,
        "SF_VECTOR_SIZE": 32,
    }

    # The two ops share every operand but the scale layout: SM100 takes the
    # rank-5 SF-atom interleave, CDNA4 the checkpoint's rank-2 E8M0 scales.
    op_name = (
        "mo.fused_qkv_matmul.ragged.paged.scale.mxfp8.amd"
        if _is_amd_gpu()
        else "mo.fused_qkv_matmul.ragged.paged.scale.mxfp8"
    )
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        tensor_sf,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]

    output_dim = (
        _output_dim if _output_dim is not None else n_heads * kv_params.head_dim
    )

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def _fused_qkv_index_ragged_matmul_scaled_mxfp8(
    kv_params: KVCacheParams,
    index_kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    index_kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    num_index_heads: int,
    idx_head_dim: int,
    input_scale: TensorValue,
    weight_scale: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Computes MiniMax-M3's fused QKV + index-QK projections in one MXFP8 GEMM.

    A 5-way fusion: ``input`` and ``wqkv`` carry ``float8_e4m3fn`` data with
    E8M0 (``float8_e8m0fnu``) block scales over 32-element K blocks. ``wqkv`` is
    the concatenation ``[Wq | Wk | Wv | Wiq | Wik]`` along the output dimension.
    The single matmul output columns route as:

    - ``Q``       -> returned as the first output, shape ``[M, q_dim]``.
    - ``K`` / ``V`` -> scattered in place into the MAIN ``kv_collection``.
    - ``IndexQ``  -> returned as the second output, shape ``[M, iq_dim]``.
    - ``IndexK``  -> scattered in place into the INDEX ``index_kv_collection``
      (MLA cache: single latent head, head 0, K only).

    The fusion is bit-exact to separate QKV and IndexQK matmuls because every
    band boundary lands on a 128-element scale-block boundary for M3.

    Args:
        kv_params: KVCacheParams for the MAIN (K, V) cache (GQA/MHA, non-MLA).
        index_kv_params: KVCacheParams for the INDEX (IndexK) cache; MLA with a
            single latent head (``is_mla=True``, ``n_kv_heads=1`` for M3).
        input: Activation tensor, ``float8_e4m3fn`` with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: Ragged offsets ``[batch_size + 1]``, uint32.
        wqkv: Concatenated weight ``[Wq | Wk | Wv | Wiq | Wik]``,
            ``float8_e4m3fn``, shape [N_total, K=hidden_dim] where
            ``N_total = q_dim + 2 * kv_dim + iq_dim + ik_dim``.
        kv_collection: PagedCacheValues for the MAIN cache.
        index_kv_collection: PagedCacheValues for the INDEX cache.
        layer_idx: Layer index, uint32 on CPU.
        n_heads: Number of (main) attention heads. ``q_dim = n_heads *
            head_dim``.
        num_index_heads: Number of index Q heads. ``iq_dim = num_index_heads *
            idx_head_dim``.
        idx_head_dim: Index head dimension; also the single-head IndexK width.
        input_scale: E8M0 input block scales in the rank-5 SF-atom layout.
        weight_scale: E8M0 weight block scales in the rank-5 SF-atom layout.

    Returns:
        A tuple ``(q, index_q)`` of bf16 tensors: ``q`` is ``[M, q_dim]`` and
        ``index_q`` is ``[M, iq_dim]``.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    tensors_to_check = [wqkv, input_row_offsets, input_scale, weight_scale]
    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            "expected all tensors to be on the same device as input"
            f" ({input.device}), but got:\n  wqkv={wqkv.device}\n "
            f" input_row_offsets={input_row_offsets.device}\n "
            f" input_scale={input_scale.device}\n "
            f" weight_scale={weight_scale.device}"
        )

    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            "expected layer_idx to be on CPU device, but got"
            f" {layer_idx.device}"
        )

    # MXFP8 block scales fully describe the quantization, but the kernel still
    # multiplies by a per-tensor scale, so pass an identity scale on CPU.
    tensor_sf = ops.constant(1.0, DType.float32, device=DeviceRef.CPU())

    assert kv_params.page_size is not None
    assert index_kv_params.page_size is not None
    iq_dim = num_index_heads * idx_head_dim
    parameters: dict[str, int | str | DType] = {
        "dtype": DType.float8_e4m3fn,
        "scale_type": DType.float8_e8m0fnu,
        "kv_type": kv_params.dtype,
        "index_kv_type": index_kv_params.dtype,
        "SF_VECTOR_SIZE": 32,
        "IQ_DIM": iq_dim,
    }

    # The two ops share every operand but the scale layout: SM100 takes the
    # rank-5 SF-atom interleave, CDNA4 the checkpoint's rank-2 E8M0 scales.
    op_name = (
        "mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8.amd"
        if _is_amd_gpu()
        else "mo.fused_qkv_index_matmul.ragged.paged.scale.mxfp8"
    )
    values = [
        input,
        input_row_offsets,
        wqkv,
        input_scale,
        weight_scale,
        tensor_sf,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        *index_kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]

    # Two separate outputs: Q [M, q_dim] and IndexQ [M, iq_dim]. The kernel's
    # store-redirect epilogue routes the Q band to the first output and the
    # IndexQ band to the second, so the downstream reshapes stay contiguous
    # views (no split/copy).
    q_dim = n_heads * kv_params.head_dim

    results = ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [q_dim],
                device=input.device,
            ),
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape[:-1] + [iq_dim],
                device=input.device,
            ),
        ],
        parameters=parameters,
    )
    return (results[0].tensor, results[1].tensor)


def _fused_qkv_index_ragged_matmul(
    kv_params: KVCacheParams,
    index_kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    index_kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    num_index_heads: int,
    idx_head_dim: int,
) -> TensorValue:
    """Computes MiniMax-M3's fused QKV + index-QK projections in one BF16 GEMM.

    Non-scaled BF16 analog of ``_fused_qkv_index_ragged_matmul_scaled_mxfp8``: a
    5-way fusion over the concatenated weight ``[Wq | Wk | Wv | Wiq | Wik]``
    (along the output dimension), with no block-scaling operands. ``input`` and
    ``wqkv`` are uniform ``bfloat16`` (attention in M3 is not quantized). The
    single matmul output columns route as:

    - ``Q``       -> returned combined output, columns ``[0, q_dim)``.
    - ``K`` / ``V`` -> scattered in place into the MAIN ``kv_collection``.
    - ``IndexQ``  -> returned combined output, columns
      ``[q_dim, q_dim + iq_dim)`` (packed right after ``Q``).
    - ``IndexK``  -> scattered in place into the INDEX ``index_kv_collection``
      (MLA cache: single latent head, head 0, K only).

    The model code splits the returned tensor into ``Q`` and ``IndexQ`` via
    ``ops.split``.

    Args:
        kv_params: KVCacheParams for the MAIN (K, V) cache (GQA/MHA, non-MLA).
        index_kv_params: KVCacheParams for the INDEX (IndexK) cache; MLA with a
            single latent head (``is_mla=True``, ``n_kv_heads=1`` for M3).
        input: Activation tensor, ``bfloat16`` with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: Ragged offsets ``[batch_size + 1]``, uint32.
        wqkv: Concatenated weight ``[Wq | Wk | Wv | Wiq | Wik]``, ``bfloat16``,
            shape [N_total, K=hidden_dim] where
            ``N_total = q_dim + 2 * kv_dim + iq_dim + ik_dim``.
        kv_collection: PagedCacheValues for the MAIN cache.
        index_kv_collection: PagedCacheValues for the INDEX cache.
        layer_idx: Layer index, uint32 on CPU.
        n_heads: Number of (main) attention heads. ``q_dim = n_heads *
            head_dim``.
        num_index_heads: Number of index Q heads. ``iq_dim = num_index_heads *
            idx_head_dim``.
        idx_head_dim: Index head dimension; also the single-head IndexK width.

    Returns:
        Combined ``[M, q_dim + iq_dim]`` tensor (Q then IndexQ), dtype = input's.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(input=input, wqkv=wqkv)

    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    tensors_to_check = [wqkv, input_row_offsets]
    if not all(t.device == input.device for t in tensors_to_check):
        raise ValueError(
            "expected all tensors to be on the same device as input"
            f" ({input.device}), but got:\n  wqkv={wqkv.device}\n "
            f" input_row_offsets={input_row_offsets.device}"
        )

    if layer_idx.device != DeviceRef.CPU():
        raise ValueError(
            "expected layer_idx to be on CPU device, but got"
            f" {layer_idx.device}"
        )

    assert kv_params.page_size is not None
    assert index_kv_params.page_size is not None
    iq_dim = num_index_heads * idx_head_dim
    parameters: dict[str, int | str | DType] = {
        "kv_type": kv_params.dtype,
        "index_kv_type": index_kv_params.dtype,
        "IQ_DIM": iq_dim,
    }

    op_name = "mo.fused_qkv_index_matmul.ragged.paged"
    values = [
        input,
        input_row_offsets,
        wqkv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        *index_kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]

    # Combined output: Q (n_heads * head_dim) then IndexQ (iq_dim).
    output_dim = n_heads * kv_params.head_dim + iq_dim

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [output_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def unfused_qkv_ragged_matmul_gguf_quantized(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    q_weight: TensorValue,
    k_weight: TensorValue,
    v_weight: TensorValue,
    quantization_encoding_q: QuantizationEncoding,
    quantization_encoding_k: QuantizationEncoding,
    quantization_encoding_v: QuantizationEncoding,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input and
    quantized weight matrices. A ``quantization_config`` must be provided.

    ``input`` and ``input_row_offsets`` are used together to implement the ragged
    tensor.
    ``input_row_offsets`` indicates where each batch starts and ends in ``input``

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(DType.float32, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    if (
        not quantization_encoding_q.is_gguf
        or not quantization_encoding_k.is_gguf
        or not quantization_encoding_v.is_gguf
    ):
        raise ValueError(
            "expected quantization_encoding_q, quantization_encoding_k, and"
            " quantization_encoding_v to be gguf, was"
            f" {quantization_encoding_q}, {quantization_encoding_k}, and"
            f" {quantization_encoding_v}"
        )

    assert kv_params.page_size is not None
    parameters: dict[str, int | str | DType] = {
        "quantization_encoding_q": quantization_encoding_q.name,
        "quantization_encoding_k": quantization_encoding_k.name,
        "quantization_encoding_v": quantization_encoding_v.name,
    }

    return ops.inplace_custom(
        "mo.unfused_qkv_matmul.ragged.paged.gguf_quantized",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            repack_gguf_quantized_weights(q_weight, quantization_encoding_q),
            repack_gguf_quantized_weights(k_weight, quantization_encoding_k),
            repack_gguf_quantized_weights(v_weight, quantization_encoding_v),
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qkv_ragged_matmul_quantized(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    n_heads: int,
    quantization_config: QuantizationConfig,
    perm_idx: TensorValue | None = None,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes fused query, key, and value projections with ragged input and
    quantized weight matrices. A ``quantization_config`` must be provided.

    ``input`` and ``input_row_offsets`` are used together to implement the ragged
    tensor.
    ``input_row_offsets`` indicates where each batch starts and ends in ``input``

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    input_rank_expected = 2
    _check_rank(input_rank_expected, input=input)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    # In the group-wise quantization scheme, every `group_size` quantized weights
    # share the same scale. If `has_zp` is `True`, there is also a group-wise zero
    # point that need to be subtracted from the quantized weights.
    # Since the new extensibility API doesn't currently support `bool` type parameters,
    # we pass `has_zp` as an integer (`has_zp_int`).
    # For GPTQ, `has_zp_int` will always be 0.
    parameters: dict[str, int | str | DType] = {
        "group_size": quantization_config.group_size,
        "has_zp_int": 0,
    }
    if perm_idx:
        input = ops.gather(input, TensorValue(perm_idx), axis=1)
        perm_idx = perm_idx.to(input.type.device or DeviceRef.CPU())
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128_desc_act",
            wqkv.device,
            [wqkv, perm_idx],
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                    device=input.type.device or DeviceRef.CPU(),
                )
            ],
        )[0].tensor
    else:
        wqkv = ops.custom(
            "GPTQ_gpu_repack_b4_g128",
            wqkv.device,
            [
                wqkv,
            ],
            out_types=[
                TensorType(
                    DType.uint8,
                    ((wqkv.shape[1], wqkv.shape[0])),
                    device=input.type.device or DeviceRef.CPU(),
                )
            ],
        )[0].tensor

    args = [
        input,
        input_row_offsets,
        wqkv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
    ]
    if bias is not None:
        args.append(bias)
        bias_name_str = "bias."
    else:
        bias_name_str = ""

    op_name = f"mo.fused_qkv_matmul.ragged.paged.{bias_name_str}quantized"

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=args,
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=input.shape[:-1] + [n_heads * kv_params.head_dim],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def matmul_kv_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> None:
    """Computes key and value projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    _check_same_dtype(hidden_states=hidden_states, weight=weight)

    hidden_states_rank_expected = 2
    _check_rank(hidden_states_rank_expected, hidden_states=hidden_states)

    _check_dtype(DType.uint32, input_row_offsets=input_row_offsets)

    op_name = "mo.kv_matmul.ragged.paged"

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
    )


def matmul_k_cache_ragged(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
) -> None:
    """Computes key projections with ragged input.

    `hidden_states` and `input_row_offsets` are used together to
    implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`
    """
    _check_same_dtype(hidden_states=hidden_states, weight=weight)

    hidden_states_rank_expected = 2
    _check_rank(hidden_states_rank_expected, hidden_states=hidden_states)

    _check_dtype(DType.uint32, input_row_offsets=input_row_offsets)

    op_name = "mo.k_matmul.ragged.paged"

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
    )


def matmul_k_cache_ragged_scaled_float8(
    kv_params: KVCacheParams,
    hidden_states: TensorValue,
    input_row_offsets: TensorValue,
    weight: TensorValue,
    input_scale: TensorValue,
    weight_scale: TensorValue,
    kv_collection: PagedCacheValues,
    scales_granularity_mnk: tuple[int, int, int],
    layer_idx: TensorValue,
) -> None:
    """Computes key projections with ragged input with FP8 block scaling.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        hidden_states: TensorValue representing the input tensor with shape
            [M=total_seq_len, K=hidden_dim].
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape [batch_size + 1].
        weight: TensorValue representing the weight tensor with shape
            [N=num_heads, K=hidden_dim].
        input_scale: TensorValue representing the input scale tensor with shape
            [ceildiv(K / BLOCK_SIZE_K), ceildiv(M / BLOCK_SIZE_M)].
        weight_scale: TensorValue representing the weight scale tensor with
            shape [ceildiv(N / BLOCK_SIZE_N), ceildiv(K / BLOCK_SIZE_K)].
        kv_collection: PagedCacheValues object for managing key-value cache.
        scales_granularity_mnk: tuple[int, int, int] representing the
            scaling (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K).
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_same_dtype(hidden_states=hidden_states, weight=weight)

    hidden_states_rank_expected = 2
    _check_rank(hidden_states_rank_expected, hidden_states=hidden_states)

    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    op_name = "mo.k_matmul.ragged.paged.scale"

    parameters: dict[str, bool | int | str | DType] = {
        "m_scale_granularity": scales_granularity_mnk[0],
        "n_scale_granularity": scales_granularity_mnk[1],
        "k_scale_granularity": scales_granularity_mnk[2],
    }

    ops.inplace_custom(
        name=op_name,
        device=hidden_states.device,
        values=[
            hidden_states,
            input_row_offsets,
            weight,
            input_scale,
            weight_scale,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
        parameters=parameters,
    )


def fused_qk_ragged_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    interleaved: bool = True,
    position_ids: TensorValue | None = None,
    mrope_section: list[int] | None = None,
) -> TensorValue:
    """Computes fused query-key attention with rotary positional encodings and ragged inputs.

    Args:
        kv_params: KV cache parameters
        input: [batch_size * seq_len, n_heads, head_dim]
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        kv_collection: KV cache collection
        freqs_cis: tensor of shape (max_seq_len * 2, head_dim)
        layer_idx: Layer index for KV cache
        interleaved: Whether to use interleaved RoPE pattern
        position_ids: Optional ragged 2D array of position IDs. If None, defaults to
                     cache_length + token_idx for each token. When `num_sections` > 1,
                     `mrope_section` must be provided to indicate each section of the head_dim
                     to apply RoPE to. Shape: [num_sections, total_seq_len]
        mrope_section: Optional list of integers indicating the section of the head_dim to
        apply RoPE to. Must be used in conjunction with `position_ids`.

    `input` and `input_row_offsets` are used together to implement the ragged tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`. If `input`
    is not of the same dtype as `freqs_cis`, it will be cast to the dtype of `freqs_cis`
    for the computation, and cast back to the original dtype after the computation is
    finished.

    When `position_ids` and `mrope_section` are provided, it replaces the default position
    calculation (cache_length + token_idx) with explicit position values. This is useful for
    3D RoPE in models like Qwen2.5-VL that need custom position encoding.
    """
    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
        "cache_dtype": kv_params.dtype,
    }

    if position_ids is not None:
        _check_dtype(DType.uint32, position_ids=position_ids)
        _check_rank(2, position_ids=position_ids)
        if mrope_section is not None:
            if len(mrope_section) != position_ids.shape[0]:
                raise ValueError(
                    "expected mrope_section to have length"
                    f" {position_ids.shape[0]}, was {len(mrope_section)}"
                )
            # multiplied by 2 because the kernel expects the section to be in terms of head_dim,
            # then calculate the prefix sum of the section
            mrope_section = [x * 2 for x in mrope_section]
            mrope_section = [
                sum(mrope_section[: i + 1]) for i in range(len(mrope_section))
            ]
            # convert mrope_section to a string, with each element separated by "_"
            parameters["mrope_section"] = "_".join(
                str(x) for x in mrope_section
            )
        else:
            parameters["mrope_section"] = ""

    if position_ids is not None:
        op_name = "mo.fused_qk_rope.ragged.paged.with_position_id"
        values = [
            input,
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            freqs_cis,
            position_ids,
            layer_idx,
        ]
    else:
        op_name = "mo.fused_qk_rope.ragged.paged"
        values = [
            input,
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            freqs_cis,
            layer_idx,
        ]

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qk_ragged_rms_norm(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    q_gamma: TensorValue,
    k_gamma: TensorValue,
    epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    weight_offset: float | np.floating[Any],
    multiply_before_cast: bool = True,
) -> TensorValue:
    """Computes fused query-key RMSNorm with ragged inputs and paged KV cache.

    This function applies per-head RMSNorm to the query tensor and to the new
    key entries already written into the paged KV cache. The query tensor is
    returned as a new tensor with the same shape and dtype as ``input``. The key
    cache is normalized in place for only the newly written entries described by
    ``input_row_offsets``.

    Args:
        kv_params: The KV cache parameters.
        input: The query tensor of shape ``[total_seq_len, n_heads, head_dim]``.
        input_row_offsets: The ragged tensor offsets indicating where each
            batch starts and ends in ``input``. Must have dtype ``uint32``.
        kv_collection: The paged KV cache collection containing the key cache.
        q_gamma: The rank-1 query RMSNorm weight. Its size must match
            ``kv_params.head_dim``.
        k_gamma: The rank-1 key RMSNorm weight. Its size must match
            ``q_gamma`` and ``kv_params.head_dim``.
        epsilon: The RMSNorm epsilon value.
        layer_idx: The layer index for the KV cache. Must have dtype ``uint32``.
        weight_offset: The constant offset added to each RMSNorm weight.
        multiply_before_cast: Whether to multiply by the effective weight before
            casting to the output dtype.

    Returns:
        The normalized query tensor with the same shape and dtype as ``input``.

    Raises:
        ValueError: This includes when the input ranks are invalid, the row
            offset or layer index dtypes are invalid, the gamma weights have
            different sizes, or the gamma size does not match the head
            dimension.
    """
    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )
    _check_rank(3, input=input)
    _check_rank(1, q_gamma=q_gamma, k_gamma=k_gamma)

    if q_gamma.shape[0] != k_gamma.shape[0]:
        raise ValueError(
            "expected q_gamma and k_gamma to have the same size, got"
            f" {q_gamma.shape[0]} and {k_gamma.shape[0]}"
        )
    if q_gamma.shape[0] != kv_params.head_dim:
        raise ValueError(
            "fused_qk_ragged_rms_norm requires full per-head normalization;"
            f" expected gamma size {kv_params.head_dim} but got"
            f" {q_gamma.shape[0]}"
        )
    if input.shape[2] != kv_params.head_dim:
        raise ValueError(
            "expected input head_dim to match kv_params.head_dim, got"
            f" {input.shape[2]} and {kv_params.head_dim}"
        )

    parameters: dict[str, int | str | DType | bool] = {
        "multiply_before_cast": multiply_before_cast,
    }
    assert kv_params.page_size is not None

    return ops.inplace_custom(
        "mo.fused_qk_rms_norm.ragged.paged",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            q_gamma,
            k_gamma,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
            ops.constant(weight_offset, input.dtype, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_qk_rms_norm_rope_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    q_gamma: TensorValue,
    k_gamma: TensorValue,
    freqs_cis: TensorValue,
    epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    weight_offset: float | np.floating[Any],
    interleaved: bool = True,
    multiply_before_cast: bool = True,
    q_out_dtype: DType | None = None,
) -> TensorValue:
    """Computes fused per-head RMSNorm and RoPE with ragged inputs and paged KV cache.

    This fuses :obj:`fused_qk_ragged_rms_norm` and :obj:`fused_qk_ragged_rope`
    into a single GPU launch. It applies per-head RMSNorm to the query tensor
    and to the new key entries written into the paged KV cache, then applies
    RoPE to the normalized values. The query tensor is returned as a new tensor
    with the same shape as ``input``, at ``q_out_dtype`` (default
    ``input.dtype``); the key cache is updated in place for the newly written
    entries.

    The RoPE dimension is taken from ``freqs_cis.shape[1]``. When it is smaller
    than the head dimension, RoPE is applied only to the prefix
    ``[0, rope_dim)`` of each head (non-interleaved layout) and the suffix is
    left un-roped, matching :obj:`fused_qk_ragged_rope`.

    Args:
        kv_params: The KV cache parameters.
        input: The query tensor of shape ``[total_seq_len, n_heads, head_dim]``.
        input_row_offsets: The ragged tensor offsets indicating where each
            batch starts and ends in ``input``. Must have dtype ``uint32``.
        kv_collection: The paged KV cache collection containing the key cache.
        q_gamma: The rank-1 query RMSNorm weight. Its size must match
            ``kv_params.head_dim``.
        k_gamma: The rank-1 key RMSNorm weight. Its size must match ``q_gamma``
            and ``kv_params.head_dim``.
        freqs_cis: The RoPE frequency tensor. Its second dimension determines
            the RoPE dimension. Must share ``input``'s dtype, or be
            ``float32`` (the kernel upcasts freqs to the fp32 accumulator, so a
            higher-precision table is consumed losslessly).
        epsilon: The RMSNorm epsilon value.
        layer_idx: The layer index for the KV cache. Must have dtype ``uint32``.
        weight_offset: The constant offset added to each RMSNorm weight.
        interleaved: Whether to use the interleaved RoPE pattern.
        multiply_before_cast: Whether to multiply by the effective weight
            before rounding the RMSNorm result to ``input``'s dtype. Governs
            that round only, not the ``q_out_dtype`` cast.
        q_out_dtype: Dtype of the returned query tensor (default
            ``input.dtype``). Q rounds to ``input.dtype`` first, so this equals
            a separate ``ops.cast`` bit for bit.

    Returns:
        The normalized and RoPE-applied query tensor with the same shape as
        ``input``, at ``q_out_dtype``.

    Raises:
        ValueError: If the input ranks are invalid, the row offset or layer
            index dtypes are invalid, the gamma weights have different sizes,
            the gamma size does not match the head dimension, or ``freqs_cis``
            dtype neither matches ``input`` nor is ``float32``.
    """
    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )
    _check_rank(3, input=input)
    _check_rank(1, q_gamma=q_gamma, k_gamma=k_gamma)
    _check_rank(2, freqs_cis=freqs_cis)

    if q_gamma.shape[0] != k_gamma.shape[0]:
        raise ValueError(
            "expected q_gamma and k_gamma to have the same size, got"
            f" {q_gamma.shape[0]} and {k_gamma.shape[0]}"
        )
    if q_gamma.shape[0] != kv_params.head_dim:
        raise ValueError(
            "fused_qk_rms_norm_rope_ragged requires full per-head"
            f" normalization; expected gamma size {kv_params.head_dim} but got"
            f" {q_gamma.shape[0]}"
        )
    if input.shape[2] != kv_params.head_dim:
        raise ValueError(
            "expected input head_dim to match kv_params.head_dim, got"
            f" {input.shape[2]} and {kv_params.head_dim}"
        )
    # The kernel is freq-dtype-parametric: it loads freqs_cis and upcasts to
    # the fp32 accumulator before the RoPE rotation, so a higher-precision
    # (fp32) freqs table is consumed losslessly regardless of the input dtype.
    # Allow fp32 freqs with a lower-precision input; otherwise require a match.
    if freqs_cis.dtype != input.dtype and freqs_cis.dtype != DType.float32:
        raise ValueError(
            "expected freqs_cis dtype to match input dtype (or be float32),"
            f" got {freqs_cis.dtype} and {input.dtype}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
        "multiply_before_cast": multiply_before_cast,
        "cache_dtype": kv_params.dtype,
    }
    assert kv_params.page_size is not None

    return ops.inplace_custom(
        "mo.fused_qk_rms_norm_rope.ragged.paged",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            q_gamma,
            k_gamma,
            freqs_cis,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
            ops.constant(weight_offset, input.dtype, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=q_out_dtype or input.dtype,
                shape=input.shape,
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def fused_dual_qk_rms_norm_rope_ragged(
    main_kv_params: KVCacheParams,
    index_kv_params: KVCacheParams,
    main_input: TensorValue,
    index_input: TensorValue,
    input_row_offsets: TensorValue,
    main_kv_collection: PagedCacheValues,
    index_kv_collection: PagedCacheValues,
    q_main_gamma: TensorValue,
    k_main_gamma: TensorValue,
    q_index_gamma: TensorValue,
    k_index_gamma: TensorValue,
    freqs_cis: TensorValue,
    main_epsilon: float | np.floating[Any],
    index_epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    weight_offset: float | np.floating[Any],
    interleaved: bool = True,
    multiply_before_cast: bool = True,
    q_main_out_dtype: DType | None = None,
) -> tuple[TensorValue, TensorValue]:
    """Fuses two :obj:`fused_qk_rms_norm_rope_ragged` launches into one.

    MiniMax-M3 sparse layers apply the fused per-head RMSNorm+RoPE op twice back
    to back: once for the main GQA Q / K cache and once for the lightning
    indexer's IndexQ / index-K cache. Both bands read (disjoint) slices of the
    same combined QKV+IndexQ matmul output, share ``input_row_offsets``, and
    share ``freqs_cis``, so this runs them in a single GPU launch. Each band's Q
    is returned as a separate tensor; each band's key cache is updated in place.

    All per-head RoPE geometry (dtype, ``freqs_cis.shape[1]`` rope dim,
    ``interleaved``, head dim) must be identical across the two bands; a
    divergence trips a compile-time assert in the kernel rather than silently
    mis-roping a band. The two caches may differ in KV-head count, so this is
    bit-exact to two separate :obj:`fused_qk_rms_norm_rope_ragged` calls,
    including when ``q_main_out_dtype`` narrows the main band.

    Args:
        main_kv_params: KV cache parameters for the main (GQA) cache.
        index_kv_params: KV cache parameters for the index-K cache.
        main_input: The main Q tensor ``[total_seq_len, n_heads, head_dim]``.
        index_input: The indexer Q tensor
            ``[total_seq_len, num_index_heads, head_dim]``.
        input_row_offsets: Ragged offsets shared by both bands. Dtype ``uint32``.
        main_kv_collection: Paged cache holding the main key cache.
        index_kv_collection: Paged cache holding the index-K cache.
        q_main_gamma: Rank-1 main-Q RMSNorm weight (size ``head_dim``).
        k_main_gamma: Rank-1 main-K RMSNorm weight (size ``head_dim``).
        q_index_gamma: Rank-1 index-Q RMSNorm weight (size ``head_dim``).
        k_index_gamma: Rank-1 index-K RMSNorm weight (size ``head_dim``).
        freqs_cis: The shared RoPE frequency table. Its second dimension is the
            RoPE dim. Must share the input dtype or be ``float32``.
        main_epsilon: RMSNorm epsilon for the main band.
        index_epsilon: RMSNorm epsilon for the indexer band.
        layer_idx: The layer index for both caches. Dtype ``uint32``.
        weight_offset: Constant offset added to each RMSNorm weight.
        interleaved: Whether to use the interleaved RoPE pattern (both bands).
        multiply_before_cast: Whether to multiply by the effective weight
            before rounding the RMSNorm result to ``input``'s dtype. Governs
            that round only, not the ``q_main_out_dtype`` cast.
        q_main_out_dtype: Dtype of the returned main query tensor (default
            ``main_input.dtype``); the index band is not retypable. Q rounds to
            ``main_input.dtype`` first, so this equals a separate ``ops.cast``.

    Returns:
        A tuple ``(q_main, q_index)`` of the normalized + RoPE-applied query
        tensors, shaped like ``main_input`` / ``index_input``. ``q_index`` keeps
        ``index_input``'s dtype; ``q_main`` is at ``q_main_out_dtype``.

    Raises:
        ValueError: On invalid ranks/dtypes, mismatched gamma sizes, a gamma
            size that does not match its head dim, a head-dim mismatch between
            the two bands, or a ``freqs_cis`` dtype that neither matches the
            input nor is ``float32``.
    """
    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, layer_idx=layer_idx
    )
    _check_rank(3, main_input=main_input, index_input=index_input)
    _check_rank(
        1,
        q_main_gamma=q_main_gamma,
        k_main_gamma=k_main_gamma,
        q_index_gamma=q_index_gamma,
        k_index_gamma=k_index_gamma,
    )
    _check_rank(2, freqs_cis=freqs_cis)

    for name, q_gamma, k_gamma, kv_params, input in (
        ("main", q_main_gamma, k_main_gamma, main_kv_params, main_input),
        ("index", q_index_gamma, k_index_gamma, index_kv_params, index_input),
    ):
        if q_gamma.shape[0] != k_gamma.shape[0]:
            raise ValueError(
                f"expected {name} q_gamma and k_gamma to have the same size,"
                f" got {q_gamma.shape[0]} and {k_gamma.shape[0]}"
            )
        if q_gamma.shape[0] != kv_params.head_dim:
            raise ValueError(
                "fused_dual_qk_rms_norm_rope_ragged requires full per-head"
                f" normalization; expected {name} gamma size"
                f" {kv_params.head_dim} but got {q_gamma.shape[0]}"
            )
        if input.shape[2] != kv_params.head_dim:
            raise ValueError(
                f"expected {name} input head_dim to match kv_params.head_dim,"
                f" got {input.shape[2]} and {kv_params.head_dim}"
            )

    if main_kv_params.head_dim != index_kv_params.head_dim:
        raise ValueError(
            "fused_dual_qk_rms_norm_rope_ragged requires both bands to share"
            f" head_dim, got {main_kv_params.head_dim} (main) and"
            f" {index_kv_params.head_dim} (index)"
        )
    # The kernel loads freqs_cis and upcasts to the fp32 accumulator, so an fp32
    # table is consumed losslessly regardless of the input dtype.
    if freqs_cis.dtype != main_input.dtype and freqs_cis.dtype != DType.float32:
        raise ValueError(
            "expected freqs_cis dtype to match input dtype (or be float32),"
            f" got {freqs_cis.dtype} and {main_input.dtype}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
        "multiply_before_cast": multiply_before_cast,
        "main_cache_dtype": main_kv_params.dtype,
        "index_cache_dtype": index_kv_params.dtype,
    }
    assert main_kv_params.page_size is not None
    assert index_kv_params.page_size is not None

    results = ops.inplace_custom(
        "mo.fused_qk_rms_norm_rope.ragged.paged.dual",
        device=main_input.device,
        values=[
            main_input,
            index_input,
            input_row_offsets,
            *main_kv_collection.flatten_without_attention_dispatch_metadata(),
            *index_kv_collection.flatten_without_attention_dispatch_metadata(),
            q_main_gamma,
            k_main_gamma,
            q_index_gamma,
            k_index_gamma,
            freqs_cis,
            ops.constant(main_epsilon, DType.float32, device=DeviceRef.CPU()),
            ops.constant(index_epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
            ops.constant(
                weight_offset, main_input.dtype, device=DeviceRef.CPU()
            ),
        ],
        out_types=[
            TensorType(
                dtype=q_main_out_dtype or main_input.dtype,
                shape=main_input.shape,
                device=main_input.device,
            ),
            TensorType(
                dtype=index_input.dtype,
                shape=index_input.shape,
                device=index_input.device,
            ),
        ],
        parameters=parameters,
    )
    return (results[0].tensor, results[1].tensor)


def fused_qk_padded_rope(
    kv_params: KVCacheParams,
    input: TensorValue,
    kv_collection: PagedCacheValues,
    freqs_cis: TensorValue,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    interleaved: bool = True,
) -> TensorValue:
    """Computes fused query-key RoPE with padded inputs and paged KV cache.

    This function applies Rotary Positional Embeddings (RoPE) to both Q and K tensors,
    where K is stored in the paged KV cache. This is the padded equivalent of
    fused_qk_ragged_rope.

    Args:
        kv_params: KV cache parameters.
        input: Query tensor of shape [batch, seq_len, n_heads, head_dim].
        kv_collection: Paged KV cache collection.
        freqs_cis: Frequency tensor of shape (max_seq_len * 2, head_dim).
        layer_idx: Layer index for KV cache (must be uint32 on CPU).
        valid_lengths: Buffer of shape [batch] containing the valid length for each
            sequence (must be uint32). RoPE is only applied to positions within
            these lengths.
        interleaved: Whether to use interleaved RoPE pattern.

    Returns:
        Query tensor with RoPE applied, same shape as input.

    Note:
        Unlike fused_qk_ragged_rope which requires ragged inputs, this function
        works with padded batch inputs where sequences may have different actual
        lengths but are padded to a uniform shape.
    """
    _check_dtype(DType.uint32, layer_idx=layer_idx, valid_lengths=valid_lengths)

    _check_rank(4, input=input)

    _check_rank(1, valid_lengths=valid_lengths)

    parameters: dict[str, bool | int | str | DType] = {
        "interleaved": interleaved,
    }

    # Use custom op that calls the Mojo fused_qk_rope kernel with paged cache
    return ops.inplace_custom(
        "mo.fused_qk_rope.padded.paged",
        device=input.device,
        values=[
            input,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            freqs_cis,
            layer_idx,
            valid_lengths,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype, shape=input.shape, device=input.device
            )
        ],
        parameters=parameters,
    )[0].tensor


def _validate_kv_cache_store_common(
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    key_or_value: int,
) -> None:
    _check_dtype(DType.uint32, layer_idx=layer_idx)
    _check_rank(0, layer_idx=layer_idx)
    _check_rank(6, kv_blocks=kv_collection.kv_blocks)
    _check_rank(1, cache_lengths=kv_collection.cache_lengths)
    _check_rank(2, lookup_table=kv_collection.lookup_table)
    _check_rank(
        1,
        max_prompt_length=kv_collection.max_prompt_length,
        max_cache_length=kv_collection.max_cache_length,
    )
    if key_or_value not in (KEY_CACHE_INDEX, VALUE_CACHE_INDEX):
        raise ValueError(
            "expected key_or_value to be KEY_CACHE_INDEX or VALUE_CACHE_INDEX, "
            f"was {key_or_value}"
        )


def kv_cache_store_paged_ragged(
    kv_collection: PagedCacheValues,
    x_cache: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
    *,
    key_or_value: int,
) -> None:
    """Stores key or value tensor into the paged KV cache (ragged inputs)."""
    _check_dtype(DType.uint32, input_row_offsets=input_row_offsets)
    _check_rank(3, x_cache=x_cache)
    _check_rank(1, input_row_offsets=input_row_offsets)
    _validate_kv_cache_store_common(kv_collection, layer_idx, key_or_value)

    parameters: dict[str, int | str | DType] = {
        "key_or_value": key_or_value,
    }

    ops.inplace_custom(
        "mo.kv_cache.store.paged.ragged",
        device=x_cache.device,
        values=[
            x_cache,
            kv_collection.kv_blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            input_row_offsets,
            kv_collection.max_prompt_length,
            kv_collection.max_cache_length,
            layer_idx,
        ],
        parameters=parameters,
    )


def store_k_cache_ragged(
    kv_collection: PagedCacheValues,
    x_k: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores the key tensor into the paged KV cache for ragged inputs.

    Args:
        kv_collection: The paged KV cache collection to write into.
        x_k: The key tensor of rank 3 containing the new key projections.
        input_row_offsets: Ragged tensor row offsets of shape ``[batch + 1]``
            indicating where each sequence starts and ends. Must have dtype
            ``uint32``.
        layer_idx: The scalar layer index (dtype ``uint32``) identifying which
            transformer layer's cache to update.
    """
    kv_cache_store_paged_ragged(
        kv_collection,
        x_k,
        input_row_offsets,
        layer_idx,
        key_or_value=KEY_CACHE_INDEX,
    )


def store_v_cache_ragged(
    kv_collection: PagedCacheValues,
    x_v: TensorValue,
    input_row_offsets: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores the value tensor into the paged KV cache for ragged inputs.

    Args:
        kv_collection: The paged KV cache collection to write into.
        x_v: The value tensor of rank 3 containing the new value projections.
        input_row_offsets: Ragged tensor row offsets of shape ``[batch + 1]``
            indicating where each sequence starts and ends. Must have dtype
            ``uint32``.
        layer_idx: The scalar layer index (dtype ``uint32``) identifying which
            transformer layer's cache to update.
    """
    kv_cache_store_paged_ragged(
        kv_collection,
        x_v,
        input_row_offsets,
        layer_idx,
        key_or_value=VALUE_CACHE_INDEX,
    )


def kv_cache_store_paged_padded(
    kv_collection: PagedCacheValues,
    x_cache: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
    *,
    key_or_value: int,
) -> None:
    """Stores key or value tensor into the paged KV cache (padded inputs)."""
    _check_dtype(DType.uint32, valid_lengths=valid_lengths)
    _check_rank(4, x_cache=x_cache)
    _check_rank(1, valid_lengths=valid_lengths)
    _validate_kv_cache_store_common(kv_collection, layer_idx, key_or_value)

    parameters: dict[str, int | str | DType] = {
        "key_or_value": key_or_value,
    }

    ops.inplace_custom(
        "mo.kv_cache.store.paged.padded",
        device=x_cache.device,
        values=[
            x_cache,
            kv_collection.kv_blocks,
            kv_collection.cache_lengths,
            kv_collection.lookup_table,
            valid_lengths,
            kv_collection.max_prompt_length,
            kv_collection.max_cache_length,
            layer_idx,
        ],
        parameters=parameters,
    )


def store_k_cache_padded(
    kv_collection: PagedCacheValues,
    x_k: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores the key tensor into the paged KV cache for padded inputs.

    Args:
        kv_collection: The paged KV cache collection to write into.
        x_k: The key tensor of rank 4 containing the new key projections.
        valid_lengths: Buffer of shape ``[batch]`` (dtype ``uint32``)
            indicating the actual (non-padded) sequence length for each
            batch element.
        layer_idx: The scalar layer index (dtype ``uint32``) identifying which
            transformer layer's cache to update.
    """
    kv_cache_store_paged_padded(
        kv_collection,
        x_k,
        valid_lengths,
        layer_idx,
        key_or_value=KEY_CACHE_INDEX,
    )


def store_v_cache_padded(
    kv_collection: PagedCacheValues,
    x_v: TensorValue,
    valid_lengths: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """Stores the value tensor into the paged KV cache for padded inputs.

    Args:
        kv_collection: The paged KV cache collection to write into.
        x_v: The value tensor of rank 4 containing the new value projections.
        valid_lengths: Buffer of shape ``[batch]`` (dtype ``uint32``)
            indicating the actual (non-padded) sequence length for each
            batch element.
        layer_idx: The scalar layer index (dtype ``uint32``) identifying which
            transformer layer's cache to update.
    """
    kv_cache_store_paged_padded(
        kv_collection,
        x_v,
        valid_lengths,
        layer_idx,
        key_or_value=VALUE_CACHE_INDEX,
    )


def rope_ragged(
    input: TensorValue,
    input_row_offsets: TensorValue,
    start_pos: TensorValue,
    freqs_cis: TensorValue,
    *,
    interleaved: bool = True,
    rope_first: bool = False,
    output_dtype: DType | None = None,
) -> TensorValue:
    """Applies RoPE to ragged input using the standard rope kernel.

    When ``freqs_cis`` is narrower than ``input``'s head dimension, only that
    many columns of each head are rotated and the rest pass through. Those
    rotated columns are the trailing ones by default; set ``rope_first`` to
    rotate the leading ones instead.
    """
    _check_dtype(
        DType.uint32, input_row_offsets=input_row_offsets, start_pos=start_pos
    )
    _check_rank(3, input=input)
    _check_rank(1, input_row_offsets=input_row_offsets, start_pos=start_pos)
    _check_rank(2, freqs_cis=freqs_cis)

    # The rope kernel runs on ``input.device`` (a GPU). If ``freqs_cis`` lives
    # on a different device -- commonly a CPU-resident frequency table that the
    # caller sliced (e.g. ``freqs_cis[:seq_len]``) before passing it in --
    # handing the cross-device value straight to ``ops.custom`` lets the graph
    # compiler fuse the CPU view (the ``mo.slice``) directly into the GPU
    # consumer. That fused view races on the implicit transfer's lifetime and
    # intermittently reads out of bounds under host-side timing jitter. Insert
    # an explicit transfer so the device crossing is a hard fusion barrier and
    # the kernel reads a materialized on-device buffer instead of a fused view.
    if freqs_cis.device != input.device:
        freqs_cis = freqs_cis.to(input.device)

    return Graph.current._add_op_generated(
        mo.CompositeRopeRaggedOp,
        result=TensorType(
            dtype=output_dtype if output_dtype is not None else input.dtype,
            shape=input.shape,
            device=input.device,
        ),
        input=input,
        input_row_offsets=input_row_offsets,
        start_pos=start_pos,
        freqs_cis=freqs_cis,
        interleaved=builtin.BoolAttr(interleaved),
        rope_first=builtin.BoolAttr(rope_first),
        output_param_decls=kgen.ParamDeclArrayAttr([]),
    )[0].tensor


def _apply_rope_with_freqs_cis(
    input: TensorValue,
    freqs_cis: TensorValue,
    *,
    interleaved: bool = True,
) -> TensorValue:
    """Applies RoPE using per-token freqs_cis (no KV cache coupling)."""
    if freqs_cis.rank == 2:
        head_dim = input.shape[-1]
        freqs_cis = freqs_cis.reshape((freqs_cis.shape[0], head_dim // 2, 2))
    freqs_cis = ops.cast(freqs_cis, input.dtype)
    freqs_cis = ops.unsqueeze(freqs_cis, 1)  # [T, 1, D/2, 2]

    if interleaved:
        x_complex = ops.as_interleaved_complex(input)
        x_re = x_complex[..., 0]
        x_im = x_complex[..., 1]
    else:
        half_dim = input.shape[-1] // 2
        x_re = input[..., :half_dim]
        x_im = input[..., half_dim:]

    freqs_re = freqs_cis[..., 0]
    freqs_im = freqs_cis[..., 1]
    rope_re = (x_re * freqs_re) - (x_im * freqs_im)
    rope_im = (x_re * freqs_im) + (x_im * freqs_re)

    if interleaved:
        rope_complex = ops.stack([rope_re, rope_im], axis=-1)
    else:
        rope_complex = ops.concat((rope_re, rope_im), axis=-1)

    return ops.cast(ops.reshape(rope_complex, input.shape), input.dtype)


def _freqs_cis_from_position_ids(
    freqs_cis: TensorValue,
    position_ids: TensorValue,
    *,
    mrope_section: list[int] | None = None,
) -> TensorValue:
    """Builds per-token freqs_cis from a freqs table and explicit position_ids."""
    _check_dtype(DType.uint32, position_ids=position_ids)
    if position_ids.rank == 1:
        position_ids = ops.unsqueeze(position_ids, 0)
    if position_ids.rank != 2:
        raise ValueError(
            "expected position_ids to be 1D or 2D, got rank"
            f" {position_ids.rank}"
        )

    freqs_by_section = ops.gather(input=freqs_cis, indices=position_ids, axis=0)
    if mrope_section is None:
        if position_ids.shape[0] != 1:
            raise ValueError(
                "mrope_section must be provided when position_ids has multiple"
                " sections"
            )
        return freqs_by_section[0]

    if len(mrope_section) != int(position_ids.shape[0]):
        raise ValueError(
            "expected mrope_section to have length "
            f"{position_ids.shape[0]}, was {len(mrope_section)}"
        )

    head_dim = freqs_cis.shape[-1]
    freqs_by_section = freqs_by_section.reshape(
        (position_ids.shape[0], position_ids.shape[1], head_dim // 2, 2)
    )
    freqs_t = freqs_by_section[0]

    h_offset = 1
    w_offset = 2
    step = 3
    h_length = mrope_section[h_offset] * step
    w_length = mrope_section[w_offset] * step

    h_indices = ops.range(
        h_offset,
        h_length,
        step,
        device=position_ids.device,
        dtype=DType.int64,
        out_dim=(h_length + 1) // step,
    )
    w_indices = ops.range(
        w_offset,
        w_length,
        step,
        device=position_ids.device,
        dtype=DType.int64,
        out_dim=(w_length + 1) // step,
    )

    total_seq_len = position_ids.shape[1]
    freqs_h_selected = ops.gather(
        input=freqs_by_section[h_offset], indices=h_indices, axis=1
    )
    h_indices_for_scatter = ops.tile(
        ops.unsqueeze(h_indices, 0), (total_seq_len, 1)
    )
    freqs_t = ops.scatter(
        input=freqs_t,
        updates=freqs_h_selected,
        indices=h_indices_for_scatter,
        axis=1,
    )

    freqs_w_selected = ops.gather(
        input=freqs_by_section[w_offset], indices=w_indices, axis=1
    )
    w_indices_for_scatter = ops.tile(
        ops.unsqueeze(w_indices, 0), (total_seq_len, 1)
    )
    freqs_t = ops.scatter(
        input=freqs_t,
        updates=freqs_w_selected,
        indices=w_indices_for_scatter,
        axis=1,
    )

    return ops.reshape(freqs_t, (total_seq_len, head_dim))


def rope_ragged_with_position_ids(
    input: TensorValue,
    freqs_cis: TensorValue,
    position_ids: TensorValue,
    *,
    mrope_section: list[int] | None = None,
    interleaved: bool = True,
) -> TensorValue:
    """Applies RoPE using explicit position_ids (no KV cache coupling)."""
    _check_dtype(DType.uint32, position_ids=position_ids)
    if position_ids.rank == 1:
        position_ids = ops.unsqueeze(position_ids, 0)
    if position_ids.rank != 2:
        raise ValueError(
            "expected position_ids to be 1D or 2D, got rank"
            f" {position_ids.rank}"
        )

    # Fast path: invoke kernel directly when mrope_section is not used.
    if mrope_section is None:
        # Materialize freqs_cis on the kernel's device before handing it to
        # ``ops.custom``; see the note in ``rope_ragged``. A cross-device
        # (e.g. CPU-resident, sliced) freqs_cis fused into the GPU consumer
        # races on the implicit transfer and reads out of bounds.
        if freqs_cis.device != input.device:
            freqs_cis = freqs_cis.to(input.device)
        total_tokens = ops.cast(
            ops.shape_to_tensor(input.shape)[0], DType.uint32
        ).to(input.device)
        row_offsets = ops.stack(
            [
                ops.constant(0, dtype=DType.uint32, device=input.device),
                total_tokens,
            ],
            axis=0,
        )
        start_pos = ops.constant([0], dtype=DType.uint32, device=input.device)
        return ops.custom(
            "mo.rope.ragged.with_position_id",
            device=input.device,
            values=[
                input,
                row_offsets,
                start_pos,
                freqs_cis,
                position_ids,
            ],
            out_types=[
                TensorType(
                    dtype=input.dtype, shape=input.shape, device=input.device
                )
            ],
            parameters={"interleaved": interleaved},
        )[0].tensor

    # Fallback path for mRoPE sections, keep existing graph implementation.
    per_token_freqs = _freqs_cis_from_position_ids(
        freqs_cis,
        position_ids,
        mrope_section=mrope_section,
    )
    return _apply_rope_with_freqs_cis(
        input, per_token_freqs, interleaved=interleaved
    )


def flash_attention_padded_kv_cache(
    kv_params: KVCacheParams,
    q: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    valid_lengths: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
) -> TensorValue:
    """Computes flash attention with padded inputs and paged KV cache.

    Args:
        kv_params: KV cache parameters
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        kv_collection: Paged KV cache collection
        layer_idx: Layer index for cache lookup
        valid_lengths: Buffer of shape [batch] with dtype uint32 indicating
            actual (non-padded) sequence lengths for each batch element
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention

    Returns:
        Output tensor of shape [batch, seq_len, num_heads, head_dim]

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if valid_lengths.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 valid_lengths but got {valid_lengths.dtype}"
        )

    if valid_lengths.rank != 1:
        raise ValueError(
            f"expected valid_lengths to be rank 1, got {valid_lengths.rank}"
        )

    if valid_lengths.shape[0] != q.shape[0]:
        raise ValueError(
            f"valid_lengths batch size ({valid_lengths.shape[0]}) must match "
            f"q batch size ({q.shape[0]})"
        )

    parameters = _mha_parameters(
        mask_variant, local_window_size=local_window_size
    )

    return ops.inplace_custom(
        "mo.mha.padded.paged",
        device=q.device,
        values=[
            q,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            valid_lengths,
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[TensorType(dtype=q.dtype, shape=q.shape, device=q.device)],
        parameters=parameters,
    )[0].tensor


def _validate_argument_tensor(
    name: str,
    tensor: TensorValue | BufferValue,
    dtype: DType | None = None,
    rank: int | None = None,
    device: DeviceRef | None = None,
    device_type: DeviceKind | None = None,
) -> None:
    errors = []
    if dtype is not None and tensor.dtype != dtype:
        errors.append(
            f"{name}.dtype was expected to be {dtype} but got {tensor.dtype}"
        )
    if rank is not None and tensor.rank != rank:
        errors.append(
            f"{name}.rank was expected to be {rank} but got {tensor.rank}"
        )
    if device is not None and tensor.device != device:
        errors.append(
            f"{name}.device was expected to be {device} but got {tensor.device}"
        )
    if device_type is not None and tensor.device.device_type != device_type:
        errors.append(
            f"{name}'s device type was expected to be {device_type} but got"
            f" {tensor.device.device_type}"
        )
    if errors:
        raise ValueError("\n".join(errors))


def mla_fp8_index_top_k(
    q: TensorValue,
    q_s: TensorValue,
    input_row_offsets: TensorValue,
    k_collection: PagedCacheValues,
    layer_idx: TensorValue,
    top_k: int,
    quantization_granularity: int,
    mask_variant: MHAMaskVariant = MHAMaskVariant.CAUSAL_MASK,
    kpool: int = 1,
) -> TensorValue:
    """Computes top-k indices for MLA FP8 indexed attention scores.

    This function computes FP8 matmul between queries and cached keys (with scales),
    applies masking, and returns the indices of the top-k highest-scoring keys per token.
    Scores are aggregated (summed) across all attention heads.

    Args:
        q: Query tensor of shape [total_seq_len, num_heads, head_dim] in FP8.
        q_s: Query scales tensor of shape [total_seq_len, num_heads] in float32.
        input_row_offsets: Input row offsets tensor of shape [batch_size + 1].
        k_collection: Paged KV cache collection. Must be FP8 quantized with scales.
        layer_idx: Layer index for cache lookup.
        top_k: Requested number of top indices per token.
        quantization_granularity: Quantization granularity for the K cache.
        mask_variant: The mask variant to use (NULL or CAUSAL_MASK).
        kpool: Tokens per row of ``k_collection``. ``1`` scores one row per
            cached token. ``k > 1`` means the cache holds one pooled key per
            ``k`` consecutive tokens, so ``top_k`` counts pools and the
            returned indices are pool ids that the caller expands back to
            token positions.

    Returns:
        Output tensor of shape [total_seq_len, effective_k] containing top-k key
        indices per token, where effective_k = min(top_k, max_num_keys).
        Invalid positions are filled with -1. With ``kpool > 1`` the entries are
        pool ids rather than token positions.

    Raises:
        ValueError: If ``top_k`` or ``kpool`` is not positive, or if
            ``mask_variant`` is neither NULL nor CAUSAL.
    """
    _validate_argument_tensor(
        "q", q, dtype=DType.float8_e4m3fn, rank=3, device_type=DeviceKind.GPU
    )
    _validate_argument_tensor(
        "q_s", q_s, dtype=DType.float32, rank=2, device=q.device
    )

    _validate_argument_tensor(
        "input_row_offsets",
        input_row_offsets,
        dtype=DType.uint32,
        rank=1,
        device=q.device,
    )
    _validate_argument_tensor(
        "k_collection.kv_blocks",
        k_collection.kv_blocks,
        dtype=DType.float8_e4m3fn,
        rank=6,
        device=q.device,
    )
    assert k_collection.kv_scales is not None, (
        "FP8 k_collection must have kv_scales"
    )
    _validate_argument_tensor(
        "k_collection.kv_scales",
        k_collection.kv_scales,
        dtype=DType.float32,
        rank=6,
        device=q.device,
    )

    _validate_argument_tensor(
        "layer_idx", layer_idx, dtype=DType.uint32, device=DeviceRef.CPU()
    )
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than 0, got {top_k}")
    if kpool < 1:
        raise ValueError(f"kpool must be at least 1, got {kpool}")

    # Validate mask_variant is supported
    if mask_variant not in (
        MHAMaskVariant.NULL_MASK,
        MHAMaskVariant.CAUSAL_MASK,
    ):
        raise ValueError(
            f"mask_variant must be NULL_MASK or CAUSAL_MASK, got {mask_variant}"
        )

    mask_str = _mask_str(mask_variant)
    result = ops.inplace_custom(
        "mo.mla.indexer.ragged.float8.paged",
        device=q.device,
        values=[
            q,
            q_s,
            input_row_offsets,
            *k_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=(q.shape[0], top_k),
                device=q.device,
            )
        ],
        parameters={
            "num_heads": int(q.shape[1]),
            "depth": int(q.shape[2]),
            "k": top_k,
            "quantization_granularity": quantization_granularity,
            "mask_str": mask_str,
            "kpool": kpool,
        },
    )[0].tensor

    return result


def msa_sparse_indexer(
    kv_params: KVCacheParams,
    index_q: TensorValue,
    input_row_offsets: TensorValue,
    prefix_lens: TensorValue,
    index_kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    score_scratch: BufferValue,
    *,
    num_index_heads: int,
    idx_head_dim: int,
    block_size: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    scale: float,
) -> TensorValue:
    """Selects the top-k key *blocks* per token for MiniMax-M3 sparse attention.

    Computes per-block QK scores against the index-K paged cache (BF16 or
    scale-free FP8 e4m3) and returns, for each (index head, query token), the
    ids of the ``topk`` highest-scoring 128-token blocks. The forward
    sparse-attention op consumes these block ids to gather a sparse KV band.
    The op selects the prefill or decode kernel at runtime from the index-K
    cache's ``max_seq_length``, so the same call serves both paths.

    Args:
        kv_params: Key-value cache parameters for the index-K cache.
        index_q: Index queries. Prefill: ``[total_q, num_index_heads,
            idx_head_dim]``; decode: ``[batch, num_index_heads, idx_head_dim]``.
            BF16.
        input_row_offsets: Ragged query offsets ``[batch + 1]`` uint32 (used on
            the prefill path; on decode it is ``[0, 1, ..., batch]``).
        prefix_lens: Per-row cached-key count ``[batch]`` (the index-K
            ``cache_lengths``). uint32.
        index_kv_collection: Paged index-K cache (BF16 or scale-free e4m3, no
            scales).
        layer_idx: Layer index, uint32, on CPU.
        score_scratch: Persistent FP32 scratch buffer for decode scoring, shape
            ``[num_index_heads, max_rows, MAX_NUM_BLOCKS]``. ``max_rows`` must
            cover ``total_q`` for every decode step served: ``max_batch`` for
            single-token decode, and ``max_batch * (num_draft_tokens + 1)`` for
            the multi-token (MTP / speculative) routes, which write at the
            ragged row ``input_row_offsets[b] + t``. It is persistent because
            decode runs inside the graph-capture region, so the kernel cannot
            allocate a correctly sized buffer per step; a scratch too narrow for
            a multi-token route makes that step fall back to the prefill
            indexer, while single-token decode raises.
        num_index_heads: Number of index (query) heads.
        idx_head_dim: Index head dimension.
        block_size: KV block size in tokens (== page size).
        topk: Number of blocks to select per token.
        init_blocks: Always-keep leading blocks.
        local_blocks: Always-keep trailing/local blocks.
        scale: QK scale.

    Returns:
        Block indices, int32, ``-1``-padded. Prefill: ``[num_index_heads,
        total_q, topk]``; decode: ``[num_index_heads, batch, topk]``.
    """
    _validate_argument_tensor(
        "index_q",
        index_q,
        dtype=DType.bfloat16,
        rank=3,
        device_type=DeviceKind.GPU,
    )
    _validate_argument_tensor(
        "input_row_offsets",
        input_row_offsets,
        dtype=DType.uint32,
        rank=1,
        device=index_q.device,
    )
    _validate_argument_tensor(
        "prefix_lens",
        prefix_lens,
        dtype=DType.uint32,
        rank=1,
        device=index_q.device,
    )
    k_dtype = index_kv_collection.kv_blocks.dtype
    if k_dtype not in (DType.bfloat16, DType.float8_e4m3fn):
        raise ValueError(
            "index_kv_collection.kv_blocks must be bfloat16 or "
            f"float8_e4m3fn, got {k_dtype}"
        )
    _validate_argument_tensor(
        "index_kv_collection.kv_blocks",
        index_kv_collection.kv_blocks,
        rank=6,
        device=index_q.device,
    )
    _validate_argument_tensor(
        "layer_idx", layer_idx, dtype=DType.uint32, device=DeviceRef.CPU()
    )
    _validate_argument_tensor(
        "score_scratch",
        score_scratch,
        dtype=DType.float32,
        rank=3,
        device=index_q.device,
    )
    scratch_heads = score_scratch.shape[0]
    if int(scratch_heads) != num_index_heads:
        raise ValueError(
            "score_scratch must carry one plane per index head: expected "
            f"shape[0] == {num_index_heads}, got {int(scratch_heads)}"
        )
    if topk <= 0:
        raise ValueError(f"topk must be greater than 0, got {topk}")

    num_rows = index_q.shape[0]

    parameters: dict[str, int | str | DType] = {
        "num_index_heads": num_index_heads,
        "idx_head_dim": idx_head_dim,
        "block_size": block_size,
        "topk": topk,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
    }

    if index_kv_collection.attention_dispatch_metadata is None:
        raise ValueError(
            "msa_sparse_indexer requires attention_dispatch_metadata on the"
            " index-K cache: it carries the kernel's msa_scalar_args input."
        )

    # The kernel's operand list ends at msa_scalar_args (fed from
    # attention_dispatch_metadata), so append the metadata explicitly rather
    # than using flatten(), which also emits optional fields the kernel does
    # not take (e.g. draft_attention_dispatch_metadata under speculative
    # decoding) and would shift every operand after them.
    values: list[Value[Any]] = [
        index_q,
        input_row_offsets,
        prefix_lens,
        *index_kv_collection.flatten_without_attention_dispatch_metadata(),
        index_kv_collection.attention_dispatch_metadata,
        layer_idx,
        score_scratch,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    return ops.inplace_custom(
        "mo.msa.indexer.ragged.paged",
        device=index_q.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=(num_index_heads, num_rows, topk),
                device=index_q.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def _kv_cache_row_offsets_ragged(
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
) -> TensorValue:
    """Builds cumulative valid-cache row offsets for a ragged prefill batch.

    Args:
        input_row_offsets: Ragged query offsets ``[batch + 1]`` uint32.
        kv_collection: Paged KV cache collection.

    Returns:
        ``uint32[batch + 1]`` cumulative offsets for the valid K/V rows read by
        sparse attention prefill.
    """
    _validate_argument_tensor(
        "input_row_offsets",
        input_row_offsets,
        dtype=DType.uint32,
        rank=1,
        device_type=DeviceKind.GPU,
    )

    return ops.custom(
        "mo.kv_cache.row_offsets.ragged.paged",
        device=input_row_offsets.device,
        values=[
            input_row_offsets,
            kv_collection.cache_lengths,
        ],
        out_types=[
            TensorType(
                dtype=DType.uint32,
                shape=input_row_offsets.shape,
                device=input_row_offsets.device,
            )
        ],
    )[0].tensor


def msa_sparse_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    cache_row_offsets: TensorValue,
    total_context_length: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    block_indices: TensorValue,
    *,
    group: int,
    topk: int,
    sparse_block_size: int,
    scale: float,
) -> TensorValue:
    """Computes MiniMax-M3 block-sparse attention over the main paged KV cache.

    Gathers ``topk`` 128-token KV blocks per (kv head, query token) using the
    block ids produced by :func:`msa_sparse_indexer`, then runs SM100
    block-sparse MHA. The main KV cache is BF16 or native FP8 e4m3; ``input``
    (the query) must match its dtype, and the output is always BF16.
    ``head_dim`` is 128. The op selects the prefill or decode kernel at runtime
    from the main KV cache's ``max_seq_length``, so the same call serves both
    paths.

    Args:
        kv_params: Key-value cache parameters for the main KV cache.
        input: Query tensor ``[total_q, n_heads, head_dim]`` (prefill) or
            ``[batch, n_heads, head_dim]`` (decode); dtype matches the KV cache
            (BF16 or FP8 e4m3).
        input_row_offsets: Ragged query offsets ``[batch + 1]`` uint32.
        cache_row_offsets: Ragged valid-cache offsets ``[batch + 1]`` uint32.
        total_context_length: Total padded cache length for the batch, CPU
            scalar ``[1]`` uint32.
        kv_collection: Main paged KV cache (BF16 or FP8 e4m3, no scales).
        layer_idx: Layer index, uint32, on CPU.
        block_indices: Selected block ids. Prefill: ``[n_kv_heads, total_q,
            topk]``; decode: ``[n_kv_heads, batch, topk]``. int32.
        group: Query heads per kv-head (``n_heads // n_kv_heads``).
        topk: Number of gathered KV blocks per token.
        sparse_block_size: KV block size in tokens; the model's
            ``sparse_attention_config.sparse_block_size``. Must equal the
            KV cache page size and the kernel's ``BN``.
        scale: QK scale.

    Returns:
        Output tensor ``[total_q, n_heads, head_dim]`` (prefill) or
        ``[batch, n_heads, head_dim]`` (decode), BF16.
    """
    values = _msa_sparse_attention_ragged_values(
        input=input,
        input_row_offsets=input_row_offsets,
        cache_row_offsets=cache_row_offsets,
        total_context_length=total_context_length,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        block_indices=block_indices,
        topk=topk,
        scale=scale,
    )

    return ops.inplace_custom(
        "mo.msa.attention.ragged.paged",
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=input.shape,
                device=input.device,
            )
        ],
        parameters={
            "group": group,
            "topk": topk,
            "sparse_block_size": sparse_block_size,
        },
    )[0].tensor


def msa_sparse_attention_ragged_mxfp8(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    cache_row_offsets: TensorValue,
    total_context_length: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    block_indices: TensorValue,
    *,
    group: int,
    topk: int,
    sparse_block_size: int,
    scale: float,
) -> tuple[TensorValue, TensorValue]:
    """Computes MiniMax-M3 block-sparse attention, emitting MXFP8 + scales.

    AMD (gfx950) variant of :func:`msa_sparse_attention_ragged` whose output
    is the o_proj-ready MXFP8 activation instead of BF16: quantized data
    ``[num_rows, n_heads, head_dim]`` in ``float8_e4m3fn`` plus E8M0 block
    scales ``[num_rows, n_heads * head_dim // 32]`` -- the same pair
    :func:`quantize_dynamic_block_scaled` produces from the BF16 output, so
    it feeds :func:`dynamic_block_scaled_matmul_amd` directly and the
    separate quantize dispatch is skipped. Bit-identical to that unfused
    pair; on split-K decode shapes the quantize fuses into the reduce and
    saves a dispatch.

    Args:
        kv_params: Key-value cache parameters for the main KV cache.
        input: Query tensor ``[total_q, n_heads, head_dim]`` (prefill) or
            ``[batch, n_heads, head_dim]`` (decode); dtype matches the KV
            cache (BF16 or FP8 e4m3).
        input_row_offsets: Ragged query offsets ``[batch + 1]`` uint32.
        cache_row_offsets: Ragged valid-cache offsets ``[batch + 1]`` uint32.
        total_context_length: Total padded cache length for the batch, CPU
            scalar ``[1]`` uint32.
        kv_collection: Main paged KV cache (BF16 or FP8 e4m3, no scales).
        layer_idx: Layer index, uint32, on CPU.
        block_indices: Selected block ids. Prefill: ``[n_kv_heads, total_q,
            topk]``; decode: ``[n_kv_heads, batch, topk]``. int32.
        group: Query heads per kv-head (``n_heads // n_kv_heads``).
        topk: Number of gathered KV blocks per token.
        sparse_block_size: KV block size in tokens; the model's
            ``sparse_attention_config.sparse_block_size``. Must equal the
            KV cache page size and the kernel's ``BN``.
        scale: QK scale.

    Returns:
        The quantized attention output ``[total_q, n_heads, head_dim]``
        ``float8_e4m3fn`` and its E8M0 block scales ``[total_q,
        n_heads * head_dim // 32]``.
    """
    values = _msa_sparse_attention_ragged_values(
        input=input,
        input_row_offsets=input_row_offsets,
        cache_row_offsets=cache_row_offsets,
        total_context_length=total_context_length,
        kv_collection=kv_collection,
        layer_idx=layer_idx,
        block_indices=block_indices,
        topk=topk,
        scale=scale,
    )

    # One E8M0 scale per block of the flattened [n_heads * head_dim] row,
    # matching `quantize_dynamic_block_scaled`'s rank-2 AMD layout.
    row_width = input.shape[1] * input.shape[2]
    if int(row_width) % _MX_SF_VECTOR_SIZE != 0:
        raise ValueError(
            "n_heads * head_dim must be a multiple of"
            f" {_MX_SF_VECTOR_SIZE}, got {row_width}"
        )

    results = ops.inplace_custom(
        "mo.msa.attention.ragged.paged.mxfp8",
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=DType.float8_e4m3fn,
                shape=input.shape,
                device=input.device,
            ),
            TensorType(
                dtype=DType.float8_e8m0fnu,
                shape=[input.shape[0], row_width // _MX_SF_VECTOR_SIZE],
                device=input.device,
            ),
        ],
        parameters={
            "group": group,
            "topk": topk,
            "sparse_block_size": sparse_block_size,
        },
    )
    return results[0].tensor, results[1].tensor


def _msa_sparse_attention_ragged_values(
    *,
    input: TensorValue,
    input_row_offsets: TensorValue,
    cache_row_offsets: TensorValue,
    total_context_length: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    block_indices: TensorValue,
    topk: int,
    scale: float,
) -> list[Value[Any]]:
    """Validates and builds the shared operand list for the MSA attention ops.

    ``mo.msa.attention.ragged.paged`` and its ``.mxfp8`` variant take the
    identical input operands in the identical order; only their outputs
    differ. Keeping the list in one place keeps them from drifting.
    """
    # The KV cache dtype selects the kernel's compute dtype. `input` (the query)
    # must match `kv_collection.kv_blocks`: BF16, or native FP8 e4m3 for an FP8
    # KV cache. The kernel infers `kv_type` from these operands.
    if input.dtype not in (DType.bfloat16, DType.float8_e4m3fn):
        raise ValueError(
            f"input must be bfloat16 or float8_e4m3fn, got {input.dtype}"
        )
    _validate_argument_tensor(
        "input",
        input,
        rank=3,
        device_type=DeviceKind.GPU,
    )
    _validate_argument_tensor(
        "input_row_offsets",
        input_row_offsets,
        dtype=DType.uint32,
        rank=1,
        device=input.device,
    )
    _validate_argument_tensor(
        "cache_row_offsets",
        cache_row_offsets,
        dtype=DType.uint32,
        rank=1,
        device=input.device,
    )
    _validate_argument_tensor(
        "total_context_length",
        total_context_length,
        dtype=DType.uint32,
        rank=1,
        device=DeviceRef.CPU(),
    )
    if kv_collection.kv_blocks.dtype != input.dtype:
        raise ValueError(
            "kv_collection.kv_blocks must have the same dtype as input"
            f" ({input.dtype}), got {kv_collection.kv_blocks.dtype}"
        )
    _validate_argument_tensor(
        "kv_collection.kv_blocks",
        kv_collection.kv_blocks,
        rank=6,
        device=input.device,
    )
    _validate_argument_tensor(
        "block_indices",
        block_indices,
        dtype=DType.int32,
        rank=3,
        device=input.device,
    )
    _validate_argument_tensor(
        "layer_idx", layer_idx, dtype=DType.uint32, device=DeviceRef.CPU()
    )
    if topk <= 0:
        raise ValueError(f"topk must be greater than 0, got {topk}")

    if kv_collection.attention_dispatch_metadata is None:
        raise ValueError(
            "msa_sparse_attention_ragged requires"
            " attention_dispatch_metadata on the KV cache: it carries the"
            " kernel's msa_scalar_args input."
        )

    # The kernel's operand list ends at msa_scalar_args (fed from
    # attention_dispatch_metadata), so append the metadata explicitly rather
    # than using flatten(), which also emits optional fields the kernel does
    # not take (e.g. draft_attention_dispatch_metadata under speculative
    # decoding) and would shift every operand after them.
    values: list[Value[Any]] = [
        input,
        input_row_offsets,
        cache_row_offsets,
        total_context_length,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        kv_collection.attention_dispatch_metadata,
        layer_idx,
        block_indices,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    return values


def flash_attention_gpu(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
    valid_length: TensorValue | None = None,
) -> TensorValue:
    """Computes flash attention using GPU-optimized kernel.

    Args:
        q: Query tensor of shape [batch, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, seq_len, num_heads, head_dim]
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention
        valid_length: Optional tensor of shape [batch] with dtype uint32.
            When provided, uses the padded kernel variant that respects
            the valid sequence lengths for each batch element.

    Returns:
        Output tensor of shape [batch, seq_len, num_heads, head_dim]
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

    expected_rank = 4
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got {tensor.rank}"
            )

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )

    # Validate valid_length if provided
    if valid_length is not None:
        if valid_length.dtype != DType.uint32:
            raise ValueError(
                f"valid_length must have dtype uint32, got {valid_length.dtype}"
            )

        if valid_length.rank != 1:
            raise ValueError(
                f"valid_length must be rank 1, got {valid_length.rank}"
            )

        if valid_length.shape[0] != q.shape[0]:
            raise ValueError(
                f"valid_length batch size ({valid_length.shape[0]}) must match "
                f"q batch size ({q.shape[0]})"
            )

    parameters = _mha_parameters(
        mask_variant, local_window_size=local_window_size
    )

    op_name = "mo.mha.no_cache"
    values = [q, k, v]
    if valid_length is not None:
        op_name = "mo.mha.padded.no_cache"
        values.append(valid_length)
    values.append(
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU())
    )

    return ops.custom(
        op_name,
        values=values,
        out_types=[TensorType(dtype=q.dtype, shape=q.shape, device=q.device)],
        parameters=parameters,
        device=q.device,
    )[0].tensor


def masked_flash_attention_gpu(
    q: TensorValue,
    k: TensorValue,
    v: TensorValue,
    mask: TensorValue,
    scale: float,
) -> TensorValue:
    """Computes flash attention using a materialized additive mask.

    Args:
        q: Query tensor of shape [batch, q_seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch, kv_seq_len, num_heads, head_dim]
        v: Value tensor of shape [batch, kv_seq_len, num_heads, head_dim]
        mask: Additive mask tensor. Rank 3 of shape
            [batch, q_seq_len, kv_seq_len] broadcasts across attention
            heads. Rank 4 of shape [batch, num_heads, q_seq_len,
            kv_seq_len] applies a per-head bias.
        scale: Scaling factor for attention scores.

    Returns:
        Output tensor of shape [batch, q_seq_len, num_heads, head_dim]
    """
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

    expected_rank = 4
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got {tensor.rank}"
            )

    if mask.rank not in (3, 4):
        raise ValueError(
            "mask must be rank 3 (broadcast across heads) or rank 4 "
            f"(per-head), got {mask.rank}"
        )

    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            "q, k, v batch sizes must match. Got "
            f"q: {q.shape[0]}, k: {k.shape[0]}, v: {v.shape[0]}"
        )

    if mask.shape[0] != q.shape[0]:
        raise ValueError(
            f"mask batch size ({mask.shape[0]}) must match q batch size"
            f" ({q.shape[0]})"
        )

    # Rank-4 masks are per-head: validate num_heads dim matches q.
    if mask.rank == 4:
        num_heads = q.shape[2]  # q is BSHD
        if mask.shape[1] != num_heads:
            raise ValueError(
                f"mask num_heads ({mask.shape[1]}) must match q num_heads "
                f"({num_heads})"
            )

    q_seq_idx = 2 if mask.rank == 4 else 1
    kv_seq_idx = 3 if mask.rank == 4 else 2

    if mask.shape[q_seq_idx] != q.shape[1]:
        raise ValueError(
            f"mask query length ({mask.shape[q_seq_idx]}) must match q "
            f"sequence length ({q.shape[1]})"
        )

    if mask.shape[kv_seq_idx] != k.shape[1]:
        raise ValueError(
            f"mask key length ({mask.shape[kv_seq_idx]}) must match k "
            f"sequence length ({k.shape[1]})"
        )

    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )

    _validate_argument_tensor("k", k, device=q.device)
    _validate_argument_tensor("v", v, device=q.device)
    _validate_argument_tensor("mask", mask, device=q.device)

    out_type = TensorType(dtype=q.dtype, shape=q.shape, device=q.device)
    scale_const = ops.constant(
        scale, dtype=DType.float32, device=DeviceRef.CPU()
    )
    # ``_add_op_generated`` passes operands straight to the generated op
    # constructor without coercing ``HasTensorValue`` inputs (unlike
    # ``ops.custom``). Under the experimental ``functional`` dispatch the
    # q/k/v/mask arrive as ``max.experimental.tensor.Tensor`` rather than
    # ``TensorValue``, so coerce them explicitly to match ``scale_const``.
    return Graph.current._add_op_generated(
        mo.CompositeMaskedFlashAttentionGpuOp,
        out_type,
        TensorValue(q),
        TensorValue(k),
        TensorValue(v),
        TensorValue(mask),
        scale_const,
    )[0].tensor


def flash_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
    sink_weights: TensorValue | None = None,
    rel_logits: TensorValue | None = None,
    output_dtype: DType | None = None,
) -> TensorValue:
    """Computes flash (self) attention provided the ``!mo.opaque`` KV Cache.

    Notably, this materializes the attention mask (dependent on
    :class:`MHAMaskVariant`) within the kernel. ``input`` and
    ``input_row_offsets`` are used together to implement the ragged tensor.
    ``input_row_offsets`` indicates where each batch starts and ends in
    ``input``.

    Note that this is self attention and the KV sequence length is assumed to
    be equal to the Q sequence length. For KV sequence length != Q sequence
    length, use :func:`cross_attention_ragged`.

    When ``rel_logits`` is set, the kernel gathers an additive
    relative-position bias by ``rel_dist = q_pos - k_pos`` from a
    ``(total_q_tokens, heads, extent)`` table and adds it on every visible
    position, selecting ``mo.mha.ragged.paged.rel_logits``. This path only
    supports ``mask_variant`` values ``CAUSAL_MASK`` (with
    ``local_window_size == -1``) and ``SLIDING_WINDOW_CAUSAL_MASK`` (with a
    positive ``local_window_size``); ``sink_weights`` and ``rel_logits`` are
    mutually exclusive.

    Args:
        kv_params: KVCacheParams object containing key-value cache parameters.
        input: TensorValue representing the input tensor with shape
            ``[total_seq_len, num_heads, head_dim]``.
        input_row_offsets: TensorValue indicating the start and end of each
            batch in the input tensor with shape ``[batch_size + 1]``.
        kv_collection: PagedCacheValues object for managing key-value cache.
        layer_idx: TensorValue representing the layer index, expected to have
            dtype uint32.
        mask_variant: MHAMaskVariant specifying the type of attention mask to
            use. With ``rel_logits``, only ``CAUSAL_MASK`` and
            ``SLIDING_WINDOW_CAUSAL_MASK`` are supported.
        scale: float value used to scale the attention scores.
        local_window_size: int specifying the size of the local attention
            window, default is -1 for no local window.
        sink_weights: Optional tensor of shape ``[num_heads]`` containing
            learnable sink weights for each attention head.
        rel_logits: Optional relative-position bias table with shape
            ``[total_seq_len, num_heads, extent]``; row ``r`` matches
            ``input``'s own ragged-flat row convention.
        output_dtype: Dtype for the attention output. Defaults to
            ``input.dtype``.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if input.dtype != kv_params.dtype:
        raise ValueError(
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    if sink_weights is not None and rel_logits is not None:
        raise ValueError(
            "sink_weights and rel_logits are mutually exclusive; pass at most one"
        )

    dispatch_metadata = kv_collection.attention_dispatch_metadata
    if dispatch_metadata is None:
        raise ValueError(
            "Expected attention_dispatch_metadata in kv_collection"
        )

    if sink_weights is not None:
        _check_rank(1, sink_weights=sink_weights)
        num_attention_heads = input.shape[1]
        if sink_weights.shape[0] != num_attention_heads:
            raise ValueError(
                f"expected sink_weights to have shape [{num_attention_heads}], "
                f"got {sink_weights.shape}"
            )

    if rel_logits is not None:
        if mask_variant == MHAMaskVariant.CAUSAL_MASK:
            if local_window_size != -1:
                raise ValueError(
                    "rel_logits with MHAMaskVariant.CAUSAL_MASK requires "
                    f"local_window_size == -1, got {local_window_size}"
                )
        elif mask_variant == MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK:
            if local_window_size <= 0:
                raise ValueError(
                    "rel_logits with MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK "
                    "requires a positive local_window_size, got "
                    f"{local_window_size}"
                )
        else:
            raise ValueError(
                "rel_logits requires mask_variant to be CAUSAL_MASK or "
                f"SLIDING_WINDOW_CAUSAL_MASK, got {mask_variant}"
            )
        _check_rank(3, rel_logits=rel_logits)
        _check_same_dtype(input=input, rel_logits=rel_logits)
        if rel_logits.shape[0] != input.shape[0]:
            raise ValueError(
                f"expected rel_logits to hold {input.shape[0]} rows, matching "
                f"input's ragged-flat rows, but got {rel_logits.shape[0]}"
            )
        if rel_logits.shape[1] != input.shape[1]:
            raise ValueError(
                f"expected rel_logits to hold {input.shape[1]} heads, matching "
                f"input's, but got {rel_logits.shape[1]}"
            )

        parameters: dict[str, int | str | DType] = (
            {"local_window_size": local_window_size}
            if local_window_size != -1
            else {}
        )
        op_name = "mo.mha.ragged.paged.rel_logits"
    else:
        parameters = _mha_parameters(
            mask_variant, local_window_size=local_window_size
        )
        op_name = "mo.mha.ragged.paged"
        if sink_weights is not None:
            op_name = "mo.mha.ragged.paged.sink_weights"

    values: MutableSequence[Value[Any]] = [
        input,
        input_row_offsets,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        # NOTE: The scale argument to flash attention is constrained to float32.
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    if rel_logits is not None:
        values.append(rel_logits)
    elif sink_weights is not None:
        values.append(sink_weights)
    values.append(dispatch_metadata.tensor)

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=values,
        out_types=[
            TensorType(
                dtype=output_dtype if output_dtype is not None else input.dtype,
                shape=input.shape,
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def flash_attention_ragged_gpu(
    q: TensorValueLike,
    k: TensorValueLike,
    v: TensorValueLike,
    input_row_offsets: TensorValueLike,
    max_seq_len: TensorValueLike,
    mask_variant: MHAMaskVariant,
    scale: float,
    local_window_size: int = -1,
) -> TensorValue:
    """Computes flash attention for ragged inputs using GPU-optimized kernel
    without a KV cache.

    Args:
        q: Query tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        k: Key tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        v: Value tensor of shape [total_seq_len, num_heads, head_dim] (ragged)
        input_row_offsets: Buffer of shape [batch_size + 1] with dtype uint32.
            Indicates where each sequence starts and ends in the ragged tensors.
            The values should be a prefix sum (cumulative sum) of sequence lengths.
        mask_variant: The mask variant to use for attention
        scale: Scaling factor for attention scores
        local_window_size: Local window size for sliding window attention

    Returns:
        Output tensor of shape [total_seq_len, num_heads, head_dim]
    """
    q = TensorValue(q)
    k = TensorValue(k)
    v = TensorValue(v)
    input_row_offsets = TensorValue(input_row_offsets)
    max_seq_len = TensorValue(max_seq_len)

    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            "q, k, v must have matching dtypes. Got "
            f"q.dtype={q.dtype}, k.dtype={k.dtype}, v.dtype={v.dtype}"
        )

    expected_rank = 3
    for name, tensor in [("q", q), ("k", k), ("v", v)]:
        if tensor.rank != expected_rank:
            raise ValueError(
                f"{name} must be rank {expected_rank}, got {tensor.rank}"
            )

    # Validate head dimension matches across all inputs
    head_dim = q.shape[-1]
    if k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        raise ValueError(
            "All inputs must have same head_dim. Got "
            f"q: {head_dim}, k: {k.shape[-1]}, v: {v.shape[-1]}"
        )

    # Validate total sequence lengths match
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            "q, k, v must have same total sequence length. Got "
            f"q: {q.shape[0]}, k: {k.shape[0]}, v: {v.shape[0]}"
        )

    # Validate num_heads match
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise ValueError(
            "q, k, v must have same num_heads. Got "
            f"q: {q.shape[1]}, k: {k.shape[1]}, v: {v.shape[1]}"
        )

    # Validate input_row_offsets
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "input_row_offsets must have dtype uint32, got"
            f" {input_row_offsets.dtype}"
        )

    if input_row_offsets.rank != 1:
        raise ValueError(
            f"input_row_offsets must be rank 1, got {input_row_offsets.rank}"
        )

    _validate_argument_tensor(
        "max_seq_len", max_seq_len, dtype=DType.uint32, device=DeviceRef.CPU()
    )

    parameters = _mha_parameters(
        mask_variant, local_window_size=local_window_size
    )

    op_name = "mo.mha.ragged.no_cache"
    values = [q, k, v, input_row_offsets, max_seq_len]
    values.append(
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU())
    )

    return ops.custom(
        op_name,
        values=values,
        out_types=[
            TensorType(
                dtype=q.dtype,
                shape=q.shape,
                device=q.device,
            )
        ],
        parameters=parameters,
        device=q.device,
    )[0].tensor


def flare_mla_decode_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    scalar_args: TensorValue,
    *,
    qk_rope_dim: int = 64,
) -> TensorValue:
    """Computes flash (self) attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MHAMaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    Note that this is self attention and the KV sequence length is
    assumed to be equal to the Q sequence length.
    For KV sequence length != Q sequence length, use `cross_attention_ragged`.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    # FP8 KVCache: Q can be bf16 (legacy) or fp8 (native FP8).
    # The underlying Mojo kernel handles both cases natively.
    # Output is always bfloat16 when Q is FP8 (native FP8 path).

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    if kv_collection.kv_blocks.shape[1] != 1:
        raise ValueError(
            "expected kv_collection.kv_blocks.shape[1] to be 1, got"
            f" {kv_collection.kv_blocks.shape[1]}"
        )

    assert kv_params.page_size is not None
    parameters = _mha_parameters(mask_variant)

    # Output dtype: always bfloat16 for FP8 Q (native FP8 path produces
    # bfloat16 output), same as input dtype otherwise.
    output_dtype = DType.bfloat16 if input.dtype.is_float8() else input.dtype

    input_values: MutableSequence[Value[Any]] = [
        input,
        input_row_offsets,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        # NOTE: The scale argument to flash attention is constrained to float32.
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    op_name = "mo.mla.decode.ragged.paged"
    input_values.append(scalar_args)

    return ops.inplace_custom(
        op_name,
        device=input.device,
        values=input_values,
        out_types=[
            TensorType(
                dtype=output_dtype,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    input.shape[2] - qk_rope_dim,
                ],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def flare_mla_decode_ragged_scaled(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    kv_scales: BufferValue,
    q_scales: TensorValue,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    scalar_args: TensorValue,
    qk_rope_dim: int = 64,
    per_token_scale_rope_aware: bool = False,
    quantization_granularity: int = 640,
) -> TensorValue:
    """MLA decode with explicit per-token KV and Q scale tensors.

    Like ``flare_mla_decode_ragged`` but accepts explicit scale tensors so the
    per-token-scale rope-aware kernel receives real (non-identity) scales.

    Args:
        kv_params: KV cache parameters.
        input: Query tensor [total_tokens, num_heads, head_dim].
        input_row_offsets: Ragged row offsets [batch_size + 1].
        kv_collection: Paged KV cache collection.
        kv_scales: Per-token KV scales buffer
            [num_blocks, 1, 1, page_size, 1, 1] float32.
        q_scales: Per-token Q scales tensor [total_tokens] float32.
        layer_idx: Layer index (uint32, on CPU).
        mask_variant: Attention mask variant.
        scale: Softmax scale (typically 1/sqrt(d_qk)).
        qk_rope_dim: Rope head dimension (default 64).
        per_token_scale_rope_aware: Use FP8+BF16 interleaved layout.
        quantization_granularity: Granularity for KV scale quantization.
            Should equal the KV cache head_dim (640 for rope-aware).

    Returns:
        Output tensor [total_tokens, num_heads, output_dim].
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    if kv_collection.kv_blocks.shape[1] != 1:
        raise ValueError(
            "expected kv_collection.kv_blocks.shape[1] to be 1, got"
            f" {kv_collection.kv_blocks.shape[1]}"
        )

    assert kv_params.page_size is not None
    parameters = _mha_parameters(mask_variant)
    if per_token_scale_rope_aware:
        parameters["per_token_scale_rope_aware"] = 1
    parameters["quantization_granularity"] = quantization_granularity

    output_dtype = DType.bfloat16 if input.dtype.is_float8() else input.dtype

    if per_token_scale_rope_aware:
        output_last_dim = input.shape[2] - qk_rope_dim * 2
    else:
        output_last_dim = input.shape[2] - qk_rope_dim

    return ops.inplace_custom(
        "mo.mla.decode.ragged.paged.scaled",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            kv_scales,
            kv_collection.scales_lookup_table or kv_collection.lookup_table,
            q_scales,
            layer_idx,
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
            scalar_args,
        ],
        out_types=[
            TensorType(
                dtype=output_dtype,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    output_last_dim,
                ],
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def flare_mla_prefill_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    k: TensorValue,
    v: TensorValue,
    input_row_offsets: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    qk_rope_dim: int = 64,
    output_dtype: DType | None = None,
) -> TensorValue:
    """Performs MLA prefill. In the MLA prefill, we need to decompress
    the KV tensors, as we store the latent representations in the KV cache.
    We will decompress the KV tensors into a fixed size buffer to avoid
    out-of-memory errors. In case the total cache length is greater than
    the buffer size, we will process the attention calculation in chunks.

    This MLA prefill kernel will return the output tensor for this iteration
    and the softmax info tensor for this iteration. Such tensors will be used
    by the next iteration of the MLA prefill kernel to continue the attention
    calculation.

    Args:
        kv_params: KVCacheParams
        input: Input tensor
        k: Key tensor
        v: Value tensor
        input_row_offsets: Indicates where each batch starts and ends in `input`
        buffer_row_offsets: Indicates where each batch starts and ends in the buffer
        cache_offsets: Indicates where each batch starts and ends in the KV cache
        kv_collection: KV collection
        layer_idx: Layer index tensor
        mask_variant: Mask variant
        scale: Scale
        qk_rope_dim: QK rope dimension
        output_dtype: Dtype for the attention output. Defaults to ``input.dtype``.
            FP8 inputs typically pass ``bfloat16`` since the MFMA pipeline
            accumulates in f32 and stores bf16.

    Returns:
        The output tensor for this iteration
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    assert kv_params.page_size is not None
    parameters = _mha_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        input,
        k,
        v,
        buffer_row_offsets,
        cache_offsets,
        input_row_offsets,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
    ]

    results = ops.inplace_custom(
        "mo.mla.prefill.ragged.paged",
        device=input.device,
        values=input_values,
        out_types=[
            TensorType(
                dtype=output_dtype if output_dtype is not None else input.dtype,
                shape=[
                    input.shape[0],
                    input.shape[1],
                    input.shape[2] - qk_rope_dim,
                ],
                device=input.device,
            )
        ],
        parameters=parameters,
    )

    return results[0].tensor


def flare_mla_prefill_plan(
    kv_params: KVCacheParams,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    buffer_size: int,
    max_chunks: int = 16,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """This kernel plans how to process a batch of sequences with
    varying lengths using a fixed-size buffer.

    Each sequence in the batch has some existing cached tokens and new input
    tokens. The kernel divides the total tokens into chunks of buffer_size.

    For each chunk (iteration), it calculates:
        1. Buffer offsets for each sequence in each chunk
        2. Cache offsets for each sequence in each chunk
        3. Total buffer lengths for each processing iteration
    """
    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    assert kv_params.page_size is not None

    buffer_size_tensor = ops.constant(
        buffer_size, DType.uint32, device=DeviceRef.CPU()
    )

    op_name = "mo.mla.prefill.ragged.plan"
    results = ops.inplace_custom(
        op_name,
        device=input_row_offsets.device,
        values=[
            input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            buffer_size_tensor,
        ],
        out_types=[
            TensorType(
                dtype=DType.uint32,
                shape=[max_chunks, input_row_offsets.shape[0]],
                device=input_row_offsets.device,
            ),  # buffer_row_offsets
            TensorType(
                dtype=DType.uint32,
                shape=[max_chunks, input_row_offsets.shape[0]],
                device=input_row_offsets.device,
            ),  # cache_offsets
            TensorType(
                dtype=DType.int32,
                shape=[max_chunks],
                device=input_row_offsets.device,
            ),  # buffer_lengths
        ],
    )

    return results[0].tensor, results[1].tensor, results[2].tensor


def _validate_mla_prefill_decode_graph_inputs(
    q: TensorValue,
    kv: TensorValue,
    input_row_offsets: TensorValue,
    kv_params: KVCacheParams,
    layer_idx: TensorValue,
    *,
    op_name: str,
    tensor_name: str = "q",
    expected_dtype: DType | None = None,
) -> None:
    input_rank_expected = 3
    if q.rank != input_rank_expected:
        raise ValueError(
            f"expected {tensor_name} of rank {input_rank_expected} but got"
            f" {q.rank}"
        )

    if kv.rank != 2:
        raise ValueError(f"expected kv of rank 2 but got {kv.rank}")

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    assert kv_params.page_size is not None


def _build_mla_prefill_decode_out_type(
    q: TensorValue,
    v_head_dim: int,
) -> TensorType:
    return TensorType(
        dtype=q.dtype,
        shape=[q.shape[0], q.shape[1], v_head_dim],
        device=q.device,
    )


def _fp8_mla_scale_params(
    quant_config: QuantConfig,
    override: int | None,
) -> dict[str, int]:
    """Returns the scale-granularity parameters the FP8 MLA kernel reads.

    When `override` is `None` the kernel uses the on-disk
    `weight_scale.block_size`. When the per-head row count straddles
    that block, callers pass an explicit override (e.g. 64 vs the
    on-disk 128); both N- and K-direction matmul granularities take the
    same value because the straddling sits along the M-disk axis.
    """
    assert quant_config.input_scale.block_size is not None
    assert quant_config.weight_scale.block_size is not None
    gran = (
        override
        if override is not None
        else quant_config.weight_scale.block_size[0]
    )
    return {
        "m_scale_granularity": quant_config.input_scale.block_size[0],
        "n_scale_granularity": gran,
        "k_scale_granularity": gran,
    }


def mla_prefill_graph(
    q: TensorValue,
    kv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    buffer_length: TensorValue,
    w_k: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    *,
    w_k_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    quant_config: QuantConfig | None = None,
    scale_granularity_override: int | None = None,
) -> TensorValue:
    """This is a manually fused kernel that performs the following operations:
    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Copy the KV latent values from PagedKVCache to a contiguous buffer.
    - Quantize the KV latent values to fp8.
    - Up-project the latent KV values to full K and V through two matmuls.
    - Perform MLA prefill.

    Args:
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        kv: KV latent tensor from the first projection. Shape:
            [num_tokens, cache_head_dim] where cache_head_dim = kv_lora_rank +
            qk_rope_head_dim.
        input_row_offsets: Indicates where each request starts and ends in
            `input`. This is a 1D tensor of shape [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_a_proj_layernorm: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        buffer_row_offsets: Indicates where each request's KV latent values
            should be stored in the contiguous buffer. This is a 1D tensor of
            shape [num_batches + 1].
        cache_offsets: Indicates the starting token position in the KV cache
            from which to copy KV latent values for each request. This is a 1D
            tensor of shape [num_batches + 1].
        buffer_length: The total number of tokens in the KV cache. Scalar.
        w_k: Weight matrix for up-projecting latent KV values to full K.
            Shape: [num_heads * qk_nope_head_dim, kv_latent_dim].
        w_uv: Weight tensor for up-projecting latent KV values to full V.
            Shape: [num_heads, v_head_dim, kv_latent_dim].
        kv_params: KVCacheParams
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        mask_variant: The attention mask variant controlling masking behavior.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        v_head_dim: Dimension of the V heads.
        w_k_scale: Optional FP8 scale tensor for `w_k`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        quant_config: Optional quantization config. When set, scales are required.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """
    _validate_mla_prefill_decode_graph_inputs(
        q,
        kv,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_prefill_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _mha_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        kv,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        buffer_row_offsets[0],  # one-shot prefill.
        cache_offsets[0],  # one-shot prefill.
        buffer_length[0],  # one-shot prefill.
        w_k,
        w_uv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    op_name = "mo.mla.graph.prefill.paged"

    if quant_config is not None:
        assert w_k_scale is not None and w_uv_scale is not None
        parameters.update(
            _fp8_mla_scale_params(quant_config, scale_granularity_override)
        )
        op_name += ".fp8"
        input_values += [w_k_scale, w_uv_scale]
    else:
        assert w_k_scale is None and w_uv_scale is None, (
            "w_k_scale and w_uv_scale must be None when quant_config is not set"
        )

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def compute_mla_dispatch_args_scalar(
    batch_size: TensorValue,
    max_cache_valid_length: TensorValue,
    q_max_seq_len: TensorValue,
    num_heads: int,
    device: DeviceRef,
    is_fp8_kv: bool = False,
) -> TensorValue:
    """Computes scalar dispatch arguments for the MLA decode kernel.

    Produces a CPU tensor of shape ``[3]`` containing pre-computed integer
    arguments used by the capturable MLA decode kernel variant to enable CUDA
    graph capture.

    Args:
        batch_size: Scalar tensor indicating the current batch size.
        max_cache_valid_length: Scalar tensor with the maximum valid cache
            sequence length across all requests in the batch.
        q_max_seq_len: Scalar tensor with the maximum query sequence length
            in the current batch.
        num_heads: Number of query attention heads.
        device: The :class:`~max.graph.DeviceRef` on which to run the op.

    Returns:
        A CPU :class:`~max.graph.TensorValue` of shape ``[3]`` and dtype
        ``int64`` containing the dispatch scalar arguments.
    """
    results = ops.custom(
        "mo.mla.compute_dispatch_args.scalar",
        device=device,
        values=[batch_size, max_cache_valid_length, q_max_seq_len],
        out_types=[
            TensorType(shape=[3], dtype=DType.int64, device=DeviceRef.CPU()),
        ],
        parameters={"num_heads": num_heads, "is_fp8_kv": is_fp8_kv},
    )
    return results[0].tensor


def compute_mha_decode_num_partitions(
    batch_size: TensorValue,
    max_cache_valid_length: TensorValue,
    n_kv_heads: int,
    device: DeviceRef,
) -> TensorValue:
    """Computes the MHA decode partition count inside a graph.

    Wraps the ``mo.mha.decode.get_num_partitions`` kernel as a graph op so
    that the partition heuristic can be evaluated dynamically during graph
    execution rather than only at graph-build time.

    Args:
        batch_size: Scalar int64 tensor with the current batch size.
        max_cache_valid_length: Scalar int64 tensor with the maximum valid
            cache length across all requests.
        n_kv_heads: Number of key-value attention heads per device
            (compile-time constant).
        device: The :class:`~max.graph.DeviceRef` whose hardware info
            determines the partition heuristic.

    Returns:
        A CPU :class:`~max.graph.TensorValue` of shape ``[1]`` and dtype
        ``int64`` containing the computed partition count.
    """
    request = ops.stack(
        [batch_size.reshape([]), max_cache_valid_length.reshape([])], axis=0
    )
    results = ops.custom(
        "mo.mha.decode.get_num_partitions",
        device=device,
        values=[request],
        out_types=[
            TensorType(shape=[1], dtype=DType.int64, device=DeviceRef.CPU()),
        ],
        parameters={"n_kv_heads": n_kv_heads},
    )
    return results[0].tensor


def mla_decode_graph(
    q: TensorValue,
    kv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    w_uk: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    scalar_args: TensorValue,
    num_partitions_scalar: TensorValue,
    *,
    w_uk_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    quant_config: QuantConfig | None = None,
    scale_granularity_override: int | None = None,
    sparse_indices: TensorValue | None = None,
    sparse_topk_lengths: TensorValue | None = None,
    sparse_attn_sink: TensorValue | None = None,
    sparse_indices_stride: int | None = None,
    index_share: bool = False,
) -> TensorValue:
    """This is a manually fused kernel that performs the following operations:

    - Apply RoPE to the query and the key cache (in-place).
    - Apply RMSNorm to the non-rope portion of the key cache (in-place).
    - Project q_nope to kv_latent_dim through a fp8 batched matmul:
      q_nope_proj = q_nope_t @ w_uk
    - Concatenate q_nope_proj and q_rope:
      q_full = concat(q_nope_proj, q_rope, axis=2)
    - Perform MLA decode
    - Project raw_output to v_head_dim through another fp8 batched matmul:
      output = raw_output_t @ w_uv

    Args:
        q: Combined query tensor containing both nope and rope parts. Shape:
            [tot_seq_len, num_heads, qk_nope_head_dim + qk_rope_head_dim].
        kv: KV latent tensor from the first projection. Shape:
            [num_tokens, cache_head_dim] where cache_head_dim = kv_lora_rank +
            qk_rope_head_dim.
        input_row_offsets: Indicates where each request starts and ends in
            `input`. This is a 1D tensor of shape [num_batches + 1].
        freqs_cis: Precomputed RoPE frequency values for rotary position
            embeddings. Shape: [max_seq_len, qk_rope_head_dim].
        kv_a_proj_layernorm: RMSNorm gamma weights for normalizing the KV cache.
            Shape: [kv_lora_rank].
        w_uk: Weight matrix for projecting q_nope to kv_latent_dim. Shape:
            [num_heads, kv_latent_dim, qk_nope_head_dim].
        w_uv: Weight matrix for projecting MLA decode output to v_head_dim.
            Shape: [num_heads, v_head_dim, kv_latent_dim].
        kv_params: KVCacheParams
        kv_collection: Paged KV Cache object.
        layer_idx: Layer index.
        mask_variant: The attention mask variant controlling masking behavior.
        scale: Scale for the attention calculation.
        epsilon: Small constant for numerical stability in RMSNorm.
        v_head_dim: Dimension of the V heads.
        scalar_args: Pre-computed dispatch scalar args (GPU buffer) for CUDA graph capture.
        w_uk_scale: Optional FP8 scale tensor for `w_uk`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        quant_config: Optional quantization config. When set, scales are required.
        sparse_indices: Optional ``int32`` tensor of shape ``[total_seq_len, max_topk]``
            with logical token indices into each sequence's KV; MOGG remaps them to
            physical ``block * page_size + offset`` rows before the kernel.
        sparse_topk_lengths: Per-batch valid top-k counts, ``int32`` rank-1.
        sparse_attn_sink: Per-batch attention sink weights, ``float32`` rank-1.
        sparse_indices_stride: Row stride in ``sparse_indices`` (max top-k across
            the batch). Required when ``sparse_indices`` is set.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """
    _validate_mla_prefill_decode_graph_inputs(
        q,
        kv,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_decode_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _mha_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        kv,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        w_uk,
        w_uv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    op_name = "mo.mla.graph.decode.paged"

    if quant_config is not None:
        assert w_uk_scale is not None and w_uv_scale is not None
        parameters.update(
            _fp8_mla_scale_params(quant_config, scale_granularity_override)
        )
        op_name += ".fp8"
        input_values += [w_uk_scale, w_uv_scale]

    input_values.append(scalar_args)

    if sparse_indices is not None:
        if (
            sparse_topk_lengths is None
            or sparse_attn_sink is None
            or sparse_indices_stride is None
        ):
            raise ValueError(
                "sparse_indices requires sparse_topk_lengths, sparse_attn_sink,"
                " and sparse_indices_stride."
            )
        if sparse_indices.dtype != DType.int32:
            raise ValueError(
                f"sparse_indices must be int32, got {sparse_indices.dtype}"
            )
        if sparse_topk_lengths.dtype != DType.int32:
            raise ValueError(
                "sparse_topk_lengths must be int32, got"
                f" {sparse_topk_lengths.dtype}"
            )
        if sparse_attn_sink.dtype != DType.float32:
            raise ValueError(
                "sparse_attn_sink must be float32, got"
                f" {sparse_attn_sink.dtype}"
            )
        parameters["indices_stride"] = sparse_indices_stride
        op_name += ".sparse"
        input_values += [
            sparse_indices,
            sparse_topk_lengths,
            sparse_attn_sink,
        ]
        if index_share:
            if quant_config is None:
                raise ValueError(
                    "index_share (read-once shared-KV MTP fold) is only"
                    " supported on the fp8 sparse MLA decode path."
                )
            # Read-once shared-index fold (KERN-3141): the folded q positions
            # share one identical top-k list, so the decode kernel gathers it
            # once. Only emitted when True -> the off path is byte-identical.
            parameters["index_share"] = True

    # Capturable-graph scalar is appended after the optional sparse
    # tensors so the input order matches the MoGG op signature
    # (see graph_compiler/builtin_kernels/attention.mojo).
    input_values += [num_partitions_scalar]

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def mla_prefill_decode_graph(
    q: TensorValue,
    kv: TensorValue,
    input_row_offsets: TensorValue,
    freqs_cis: TensorValue,
    kv_norm_gamma: TensorValue,
    buffer_row_offsets: TensorValue,
    cache_offsets: TensorValue,
    buffer_length: TensorValue,
    w_k: TensorValue,
    w_uk: TensorValue,
    w_uv: TensorValue,
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    scale: float,
    epsilon: float,
    v_head_dim: int,
    scalar_args: TensorValue,
    num_partitions_scalar: TensorValue,
    *,
    w_k_scale: TensorValue | None = None,
    w_uk_scale: TensorValue | None = None,
    w_uv_scale: TensorValue | None = None,
    quant_config: QuantConfig | None = None,
    scale_granularity_override: int | None = None,
    sparse_indices: TensorValue | None = None,
    sparse_topk_lengths: TensorValue | None = None,
    sparse_attn_sink: TensorValue | None = None,
    sparse_indices_stride: int | None = None,
    index_share: bool = False,
) -> TensorValue:
    """Fused MLA prefill/decode kernel for FP8.

    Switches between prefill and decode based on the maximum sequence length in
    the batch. See `mla_prefill_graph` and `mla_decode_graph` for the dedicated
    paths.

    Args:
        q: Combined query tensor with nope+rope parts.
        kv: KV latent tensor for current sequence.
        input_row_offsets: Row offsets for the batch.
        freqs_cis: RoPE frequencies tensor.
        kv_norm_gamma: RMSNorm gamma for KV cache.
        buffer_row_offsets: One-shot prefill buffer row offsets.
        cache_offsets: One-shot prefill cache offsets.
        buffer_length: One-shot prefill buffer length tensor.
        w_k: Prefill K up-projection weights.
        w_uk: Decode query-projection weights.
        w_uv: Decode output-projection / prefill V-projection weights.
        kv_params: KV cache parameters.
        kv_collection: Paged KV cache values.
        layer_idx: Layer index (uint32).
        mask_variant: Attention mask variant.
        scale: Attention scale.
        epsilon: RMSNorm epsilon.
        v_head_dim: Value head dimension for output tensor shape.
        scalar_args: Pre-computed dispatch scalar args (GPU buffer) for CUDA graph capture.
        w_k_scale: Optional FP8 scale tensor for `w_k`.
        w_uk_scale: Optional FP8 scale tensor for `w_uk`.
        w_uv_scale: Optional FP8 scale tensor for `w_uv`.
        quant_config: Optional quantization config. When set, scales are required.
        sparse_indices: Optional ``int32`` tensor for sparse decode (same semantics
            as :func:`mla_decode_graph`). Used only when the decode branch runs.
        sparse_topk_lengths: Per-batch valid top-k counts for sparse decode.
        sparse_attn_sink: Per-batch attention sink weights for sparse decode.
        sparse_indices_stride: Row stride in ``sparse_indices``. Required when
            ``sparse_indices`` is set.

    Returns:
        Tensor of shape [total_seq_len, num_heads, v_head_dim].
    """
    _validate_mla_prefill_decode_graph_inputs(
        q,
        kv,
        input_row_offsets,
        kv_params,
        layer_idx,
        op_name="mla_prefill_decode_graph",
        expected_dtype=kv_params.dtype,
    )
    parameters = _mha_parameters(mask_variant)

    input_values: MutableSequence[Value[Any]] = [
        q,
        kv,
        input_row_offsets,
        freqs_cis,
        kv_norm_gamma,
        buffer_row_offsets[0],  # one-shot prefill.
        cache_offsets[0],  # one-shot prefill.
        buffer_length[0],  # one-shot prefill.
        w_k,
        w_uk,
        w_uv,
        *kv_collection.flatten_without_attention_dispatch_metadata(),
        layer_idx,
        ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
    ]
    op_name = "mo.mla.graph.prefill.decode.paged"

    if quant_config is not None:
        assert (
            w_k_scale is not None
            and w_uk_scale is not None
            and w_uv_scale is not None
        )
        parameters.update(
            _fp8_mla_scale_params(quant_config, scale_granularity_override)
        )
        op_name += ".fp8"
        input_values += [w_k_scale, w_uk_scale, w_uv_scale]

    input_values.append(scalar_args)

    if sparse_indices is not None:
        if (
            sparse_topk_lengths is None
            or sparse_attn_sink is None
            or sparse_indices_stride is None
        ):
            raise ValueError(
                "sparse_indices requires sparse_topk_lengths, sparse_attn_sink,"
                " and sparse_indices_stride."
            )
        if sparse_indices.dtype != DType.int32:
            raise ValueError(
                f"sparse_indices must be int32, got {sparse_indices.dtype}"
            )
        if sparse_topk_lengths.dtype != DType.int32:
            raise ValueError(
                "sparse_topk_lengths must be int32, got"
                f" {sparse_topk_lengths.dtype}"
            )
        if sparse_attn_sink.dtype != DType.float32:
            raise ValueError(
                "sparse_attn_sink must be float32, got"
                f" {sparse_attn_sink.dtype}"
            )
        parameters["indices_stride"] = sparse_indices_stride
        op_name += ".sparse"
        input_values += [
            sparse_indices,
            sparse_topk_lengths,
            sparse_attn_sink,
        ]
        if index_share:
            # Read-once shared-index fold (KERN-3141); the sparse branch here
            # already requires fp8. Only emitted when True -> off is
            # byte-identical.
            parameters["index_share"] = True

    # Capturable-graph scalar appended last (see MoGG op signature).
    input_values += [num_partitions_scalar]

    return ops.inplace_custom(
        op_name,
        device=q.device,
        values=input_values,
        out_types=[_build_mla_prefill_decode_out_type(q, v_head_dim)],
        parameters=parameters,
    )[0].tensor


def flare_mla_decompress_k_cache(
    kv_params: KVCacheParams,
    buffer_row_offsets_1d: TensorValue,
    cache_offsets_1d: TensorValue,
    buffer_length: TensorValue,
    weight: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    buffer_size: int,
) -> TensorValue:
    """This kernel decompresses the key cache by up-projecting latent representations
    into the KV space using a weight matrix.

    The process involves:

    1. Copying buffer_length latent vectors from the key cache into a contiguous
        buffer (k_latent)
    2. Computing k = k_latent @ weight.T to obtain the decompressed keys

    Returns:
        A tensor of shape [buffer_size, weight.shape[0]] containing the decompressed
        keys. Note that only the first buffer_length tokens are valid.
    """
    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if cache_offsets_1d.dtype != DType.uint32:
        raise ValueError(
            f"expected uint32 cache_offsets but got {cache_offsets_1d.dtype}"
        )

    assert kv_params.page_size is not None

    results = ops.inplace_custom(
        "mo.mla.decompress.k.cache.ragged.paged",
        device=buffer_row_offsets_1d.device,
        values=[
            buffer_row_offsets_1d,
            cache_offsets_1d,
            buffer_length,
            weight,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
        ],
        out_types=[
            TensorType(
                dtype=kv_params.dtype,
                shape=[buffer_size, weight.shape[1]],
                device=buffer_row_offsets_1d.device,
            ),  # k_latent_buffer, only stores intermediate values
            TensorType(
                dtype=kv_params.dtype,
                shape=[buffer_size, weight.shape[0]],
                device=buffer_row_offsets_1d.device,
            ),  # k_buffer
        ],
    )

    return results[1].tensor


def cross_attention_ragged(
    kv_params: KVCacheParams,
    input: TensorValue,
    input_row_offsets: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    mask_variant: MHAMaskVariant,
    kv_input_row_offsets: TensorValue,
    q_max_seq_len: TensorValue,
    scale: float,
    local_window_size: int = -1,
    output_dtype: DType | None = None,
) -> TensorValue:
    """Computes cross attention provided the `!mo.opaque` KV Cache.

    Notably, this materializes the attention mask (dependent on MHAMaskVariant)
    within the kernel.
    `input` and `input_row_offsets` are used together to implement the ragged
    tensor.
    `input_row_offsets` indicates where each batch starts and ends in `input`

    attention, `kv_input_row_offsets` represents the KV sequence length.
    """
    input_rank_expected = 3
    if input.rank != input_rank_expected:
        raise ValueError(
            f"expected input of rank {input_rank_expected} but got {input.rank}"
        )

    if input.dtype != kv_params.dtype:
        raise ValueError(
            f"expected input to be dtype: {kv_params.dtype}, got {input.dtype}"
        )

    if layer_idx.dtype != DType.uint32:
        raise ValueError(f"expected uint32 layer_idx but got {layer_idx.dtype}")

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    _validate_argument_tensor(
        "q_max_seq_len",
        q_max_seq_len,
        dtype=DType.uint32,
        device=DeviceRef.CPU(),
    )

    parameters = _mha_parameters(
        mask_variant, local_window_size=local_window_size
    )

    return ops.inplace_custom(
        "mo.cross_attention.ragged.paged",
        device=input.device,
        values=[
            input,
            input_row_offsets,
            # Plumb in the query max sequence length for cross attention.
            # For self attention this is the same as the KV max seq len stored
            # on the kv_collection, but that isn't the case for cross attention.
            q_max_seq_len,
            kv_input_row_offsets,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            layer_idx,
            # NOTE: The scale argument to flash attention is constrained to float32.
            ops.constant(scale, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=output_dtype if output_dtype is not None else input.dtype,
                shape=input.shape,
                device=input.device,
            )
        ],
        parameters=parameters,
    )[0].tensor


def kv_cache_ragged_radd(
    kv_params: KVCacheParams,
    a: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
    batch_offset: TensorValue,
    layer_idx: int,
) -> None:
    """This function adds a tensor to a slice of the KVCache, sliced on the batch dimension.

    This expects that the requests which should be sliced out are contiguous and
    in the front of the tensor, and we're only adding to the last requests in the batch.

    Args:
        a: The tensor to add to the KVCache.
        kv_collection: The KVCache collection to add to.
        input_row_offsets: The offsets of the input tensor.
        batch_offset: The batch to start applying the r-add to.
        layer_idx: The layer index to add to.
    """
    _check_rank(2, a=a)
    _check_rank(1, input_row_offsets=input_row_offsets)

    if kv_params.page_size is None:
        raise ValueError("Expected kv_params.page_size to be set")

    # slice input_row_offsets to the batch offset
    input_row_offsets = ops.slice_tensor(
        input_row_offsets,
        [(slice(batch_offset, None), Dim("input_row_offsets_slice_len"))],
    )

    ops.inplace_custom(
        "mo.kv_cache.ragged.paged.radd",
        device=input_row_offsets.device,
        values=[
            a,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            input_row_offsets,
            batch_offset,
            ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
        ],
    )


def rms_norm_key_cache(
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    gamma: TensorValue,
    epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
    weight_offset: float | np.floating[Any],
    rms_norm_cols: int | None = None,
    multiply_before_cast: bool = True,
    per_head_norm: bool = True,
) -> None:
    """This function applies RMSNorm to the _new_ entries in the KVCache.

    When per_head_norm=True (default), RMSNorm is applied separately to each head.
    In this mode, gamma should have size [head_dim] and normalization occurs
    across the head_dim dimensions within each head.

    When per_head_norm=False, RMSNorm is applied per token across all heads.
    In this mode, gamma should have size [n_kv_heads * head_dim] and normalization
    occurs across all dimensions for each token.

    The size of the gamma tensor determines how many dimensions will be normalized.
    If gamma's size doesn't match the expected size based on per_head_norm setting,
    rms_norm_cols must be explicitly specified to confirm the intention to normalize
    only a subset of dimensions.

    Currently, the KVCacheT class itself isn't aware of the new cache entries
    until cache length increment, which happens after model forward.
    So use `input_row_offsets` to do this bookkeeping.
    """
    gamma_rank_expected = 1
    if gamma.rank != gamma_rank_expected:
        raise ValueError(
            f"expected gamma of rank {gamma_rank_expected} but got {gamma.rank}"
        )

    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )

    if gamma.shape[0] != kv_params.head_dim and per_head_norm:
        if rms_norm_cols is None:
            raise ValueError(
                "Size of gamma doesn't match head_dim. Please pass"
                " rms_norm_cols explicitly if you intend to apply RMSNorm to"
                " only a subset of head dimensions"
            )
        elif rms_norm_cols != gamma.shape[0]:
            raise ValueError(
                f"expected gamma of size {rms_norm_cols} but got"
                f" {gamma.shape[0]}"
            )

    # TODO: Remove this check once FP8 KVCache is supported (KERN-2394).
    if gamma.dtype != kv_params.dtype:
        raise TypeError(
            f"expected gamma dtype {gamma.dtype} to match KV dtype"
            f" {kv_params.dtype}"
        )

    parameters: dict[str, int | str | DType | bool] = {
        "multiply_before_cast": multiply_before_cast,
        "per_head_norm": per_head_norm,
    }
    assert kv_params.page_size is not None

    ops.inplace_custom(
        "mo.rms_norm_kv_cache.ragged.paged",
        device=input_row_offsets.device,
        values=[
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            gamma,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
            ops.cast(TensorValue(total_seq_len), DType.uint32),
            input_row_offsets,
            ops.constant(weight_offset, gamma.dtype, device=DeviceRef.CPU()),
        ],
        parameters=parameters,
    )


def rms_norm_value_cache(
    kv_params: KVCacheParams,
    kv_collection: PagedCacheValues,
    gamma: TensorValue,
    epsilon: float | np.floating[Any],
    layer_idx: TensorValue,
    total_seq_len: Dim,
    input_row_offsets: TensorValue,
    weight_offset: float | np.floating[Any],
    rms_norm_cols: int | None = None,
    multiply_before_cast: bool = True,
    per_head_norm: bool = True,
) -> None:
    """Applies RMSNorm in place to the _new_ entries in the value cache.
    Semantics match :func:`rms_norm_key_cache`, but updates the value tensor
    for the layer instead of the key tensor.
    """
    gamma_rank_expected = 1
    if gamma.rank != gamma_rank_expected:
        raise ValueError(
            f"expected gamma of rank {gamma_rank_expected} but got {gamma.rank}"
        )
    if input_row_offsets.dtype != DType.uint32:
        raise ValueError(
            "expected uint32 input_row_offsets but got"
            f" {input_row_offsets.dtype}"
        )
    if gamma.shape[0] != kv_params.head_dim and per_head_norm:
        if rms_norm_cols is None:
            raise ValueError(
                "Size of gamma doesn't match head_dim. Please pass"
                " rms_norm_cols explicitly if you intend to apply RMSNorm to"
                " only a subset of head dimensions"
            )
        elif rms_norm_cols != gamma.shape[0]:
            raise ValueError(
                f"expected gamma of size {rms_norm_cols} but got"
                f" {gamma.shape[0]}"
            )
    if gamma.dtype != kv_params.dtype:
        raise TypeError(
            f"expected gamma dtype {gamma.dtype} to match KV dtype"
            f" {kv_params.dtype}"
        )
    parameters: dict[str, int | str | DType | bool] = {
        "multiply_before_cast": multiply_before_cast,
        "per_head_norm": per_head_norm,
    }
    assert kv_params.page_size is not None
    ops.inplace_custom(
        "mo.rms_norm_value_cache.ragged.paged",
        device=input_row_offsets.device,
        values=[
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            gamma,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
            layer_idx,
            ops.cast(TensorValue(total_seq_len), DType.uint32),
            input_row_offsets,
            ops.constant(weight_offset, gamma.dtype, device=DeviceRef.CPU()),
        ],
        parameters=parameters,
    )


def moe_create_indices(
    topk_ids: TensorValue,
    num_local_experts: int,
    *,
    needs_scales_offset: bool = False,
    scales_alignment: int = 128,
) -> tuple[TensorValue, ...]:
    """Creates indices for the MoE layer.

    Args:
        topk_ids: The expert assignments for each token from the router.
        num_local_experts: The number of experts on this device.

    Returns:
        A tuple of five tensors:
        - token_expert_order: The reordered token indices, grouped by assigned expert.
        - expert_start_indices: The starting index for each expert's token group in
            the reordered sequence.
        - restore_token_order: The indices to restore original token ordering after
            expert computation.
        - expert_ids: ids of active experts selected for tokens
        - expert_usage_stats: The maximum number of tokens assigned to any expert,
            and the number of active experts.
    """

    op_name = "mo.moe.create.indices"
    if needs_scales_offset:
        op_name += ".with.scales.offset"

    out_types: list[Type[Any]] = [
        TensorType(
            dtype=DType.uint32,
            shape=[topk_ids.shape[0]],
            device=topk_ids.device,
        ),  # token_expert_order
        TensorType(
            dtype=DType.uint32,
            shape=[num_local_experts + 1],
            device=topk_ids.device,
        ),  # expert_start_indices
        TensorType(
            dtype=DType.uint32,
            shape=[topk_ids.shape[0]],
            device=topk_ids.device,
        ),  # restore_token_order
        TensorType(
            dtype=DType.int32,
            shape=[num_local_experts],
            device=topk_ids.device,
        ),  # expert_ids
        TensorType(
            dtype=DType.uint32, shape=[2], device=topk_ids.device
        ),  # expert_usage_stats
    ]

    if needs_scales_offset:
        out_types.append(
            TensorType(
                dtype=DType.uint32,
                shape=[num_local_experts],
                device=topk_ids.device,
            ),
        )

    results = ops.custom(
        op_name,
        device=topk_ids.device,
        values=[
            topk_ids,
        ],
        out_types=out_types,
    )

    return (
        results[0].tensor,
        results[1].tensor,
        results[2].tensor,
        results[3].tensor,
        results[4].tensor,
        *([results[5].tensor] if needs_scales_offset else []),
    )


def moe_router_group_limited(
    expert_scores: TensorValue,
    expert_bias: TensorValue,
    n_routed_experts: int,
    n_experts_per_tok: int,
    n_groups: int,
    topk_group: int,
    norm_weights: bool,
    routed_scaling_factor: float,
) -> tuple[TensorValue, TensorValue]:
    """Group limited MoE router.
    When `n_groups > 1`, selects up to `topk_group` expert groups, then
    picks ``n_experts_per_tok`` experts within those groups (DeepSeek-V3 style).
    When ``n_groups == 1``, there is only one group, so group selection is
    skipped and routing uses the dedicated GPU single-group path
    (``mo.moe.single.group.router``, implemented as ``single_group_router`` in
    Mojo). In that case ``topk_group`` is not used by the kernel.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566.

    Args:
        expert_scores: The scores for each expert for each token. Shape:
            [num_tokens, n_routed_experts].
        expert_bias: The bias for each expert. Shape: [n_routed_experts].
        n_routed_experts: The total number of experts. Must be divisible by
            n_groups.
        n_experts_per_tok: The number of experts to be selected per token.
        n_groups: The total number of expert groups. Must be divisible by
            n_routed_experts.
        topk_group: The maximum number of expert groups that a token will be
            routed to.
        norm_weights: Whether to normalize the selected expert weights when
            n_groups > 1. When n_groups == 1, normalization is currently
            always enabled (norm_weights is treated as True) so behavior
            matches the previous graph path that always divided weights by their
            sum per token.

    Returns:
        A tuple of two tensors:
        - expert_indices: The indices of the routed experts for each token.
            Shape: [num_tokens, n_experts_per_tok].
        - expert_weights: The weights of the routed experts for each token.
            Shape: [num_tokens, n_experts_per_tok].
    """

    if expert_bias.rank != 1:
        raise ValueError(
            f"expected expert_bias of rank 1 but got {expert_bias.rank}"
        )
    if expert_bias.shape[0] != expert_scores.shape[1]:
        raise ValueError(
            "expected expert_bias of shape [num_experts] but got"
            f" {expert_bias.shape}"
        )

    if n_groups == 1:
        parameters: dict[str, int | str | DType | bool] = {
            "n_routed_experts": n_routed_experts,
            "n_experts_per_tok": n_experts_per_tok,
            "norm_weights": norm_weights,
        }
        op_name = "mo.moe.single.group.router"
    else:
        parameters = {
            "n_routed_experts": n_routed_experts,
            "n_experts_per_tok": n_experts_per_tok,
            "n_groups": n_groups,
            "topk_group": topk_group,
            "norm_weights": norm_weights,
        }
        op_name = "mo.moe.router.group.limited"

    results = ops.custom(
        op_name,
        device=expert_scores.device,
        values=[
            expert_scores,
            expert_bias,
            ops.constant(
                routed_scaling_factor, DType.float32, device=DeviceRef.CPU()
            ),
        ],
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),  # expert_indices
            TensorType(
                dtype=expert_scores.dtype,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),  # expert_weights
        ],
        parameters=parameters,
    )

    return (results[0].tensor, results[1].tensor)


def moe_sink_gate_router(
    logits: TensorValue,
    expert_bias: TensorValue,
    global_scale: TensorValue,
    n_routed_experts: int,
    n_experts_per_tok: int,
    n_shared_experts: int,
    route_scale: float,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Fused sigmoid-gate MoE router with always-on sink experts.

    Sink experts are gated shared experts, not attention sinks.

    Selects the top ``n_experts_per_tok`` routed experts by
    ``sigmoid(logits) + expert_bias``, then softmax-normalizes the
    log-sigmoid of those experts' raw logits together with
    ``n_shared_experts`` always-selected sink logits, scaled by
    ``route_scale * global_scale``. This is the Inkling gate formula, run on
    the GPU as a single kernel (``mo.moe.sink.gate.router``, implemented as
    ``sink_gate_router`` in Mojo).

    Args:
        logits: Raw (pre-sigmoid) gate logits, routed experts followed by
            sink experts. Must be float32, which is the only dtype the
            kernel's joint softmax has been validated at. Shape:
            [num_tokens, n_routed_experts + n_shared_experts].
        expert_bias: Per-routed-expert selection bias. Shape: [n_routed_experts].
        global_scale: Scalar output-scaling weight. Shape: [1].
        n_routed_experts: Total number of routed experts. Must be a positive
            multiple of the target's warp width, no greater than 1024: the
            kernel runs one thread per routed expert.
        n_experts_per_tok: Number of routed experts selected per token.
            Bounded jointly with ``n_routed_experts`` by the top-k's
            surviving candidates having to fit one warp, so the ceiling
            falls as the expert count rises: at a warp width of 32, 256
            experts admit up to 10 per token and 512 up to 8, while at 64
            the same 256 experts admit up to 32.
        n_shared_experts: Number of always-selected sink experts.
            ``n_experts_per_tok + n_shared_experts`` must be a power of two
            no greater than the target's warp width, since the two are
            jointly normalized by a single warp-level reduction.
        route_scale: Scalar output-scaling factor.

    Returns:
        A tuple of three tensors:
        - expert_ids: Selected routed-expert indices. Shape:
            [num_tokens, n_experts_per_tok].
        - expert_weights: Routing weight per selected routed expert. Shape:
            [num_tokens, n_experts_per_tok].
        - sink_weights: Routing weight per sink expert. Shape:
            [num_tokens, n_shared_experts].

    Raises:
        ValueError: If the routing shape is one the kernel cannot compile.
    """
    _check_rank(2, logits=logits)
    _check_rank(1, expert_bias=expert_bias, global_scale=global_scale)
    _check_same_dtype(logits=logits, global_scale=global_scale)
    if logits.dtype != DType.float32:
        raise ValueError(f"expected float32 logits but got {logits.dtype}")

    # The kernel runs one thread per routed expert and normalizes the
    # selected-plus-sink weights with a single warp-level reduction, so both
    # counts are bounded by the target's warp width. Check here so an
    # unsupported checkpoint fails before the Mojo compiler sees it.
    warp_size = 64 if _is_amd_gpu() else 32
    if (
        n_routed_experts <= 0
        or n_routed_experts % warp_size
        or n_routed_experts > 1024
    ):
        raise ValueError(
            f"n_routed_experts must be a positive multiple of {warp_size} and"
            f" fit in one block (<= 1024) but got {n_routed_experts}"
        )
    k_total = n_experts_per_tok + n_shared_experts
    if k_total <= 0 or k_total & (k_total - 1) or k_total > warp_size:
        raise ValueError(
            "n_experts_per_tok + n_shared_experts must be a positive power of"
            f" two no greater than {warp_size} but got {n_experts_per_tok} +"
            f" {n_shared_experts} = {k_total}"
        )
    # The top-k reduces in phases, and warp 0 sorts the last round, so the
    # survivors of the round before it must fit one warp. Rules out shapes
    # like 1024/30/2 that clear the two bounds above.
    num_warps = n_routed_experts // warp_size
    phase2_warps = -(-(num_warps * n_experts_per_tok) // warp_size)
    if phase2_warps * n_experts_per_tok > warp_size:
        raise ValueError(
            f"{n_routed_experts} routed experts with n_experts_per_tok"
            f" {n_experts_per_tok} leaves"
            f" {phase2_warps * n_experts_per_tok} top-k survivors, which"
            f" exceeds the target's warp width of {warp_size}; lower"
            " n_experts_per_tok or n_routed_experts"
        )

    if logits.shape[1] != n_routed_experts + n_shared_experts:
        raise ValueError(
            "expected logits of shape [num_tokens, n_routed_experts +"
            f" n_shared_experts] but got {logits.shape}"
        )
    if expert_bias.shape[0] != n_routed_experts:
        raise ValueError(
            f"expected expert_bias of shape [{n_routed_experts}] but got"
            f" {expert_bias.shape}"
        )
    if global_scale.shape[0] != 1:
        raise ValueError(
            f"expected global_scale of shape [1] but got {global_scale.shape}"
        )

    results = ops.custom(
        "mo.moe.sink.gate.router",
        device=logits.device,
        values=[
            logits,
            expert_bias,
            global_scale,
            ops.constant(route_scale, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=[logits.shape[0], n_experts_per_tok],
                device=logits.device,
            ),  # expert_ids
            TensorType(
                dtype=logits.dtype,
                shape=[logits.shape[0], n_experts_per_tok],
                device=logits.device,
            ),  # expert_weights
            TensorType(
                dtype=logits.dtype,
                shape=[logits.shape[0], n_shared_experts],
                device=logits.device,
            ),  # sink_weights
        ],
        parameters={
            "n_routed_experts": n_routed_experts,
            "n_experts_per_tok": n_experts_per_tok,
            "n_shared_experts": n_shared_experts,
        },
    )

    return (results[0].tensor, results[1].tensor, results[2].tensor)


def _router_gate_mixed_gemv(
    hidden_states: TensorValue,
    gate_weight: TensorValue,
) -> TensorValue:
    """Computes mixed-input MiniMax router logits."""
    _check_rank(2, hidden_states=hidden_states, gate_weight=gate_weight)
    _check_dtype(DType.bfloat16, hidden_states=hidden_states)
    _check_dtype(DType.float32, gate_weight=gate_weight)
    _check_same_device(hidden_states=hidden_states, gate_weight=gate_weight)

    n_dim = gate_weight.shape[0]
    k_dim = gate_weight.shape[1]
    # N/K are inferred from the static gate-weight layout inside the op, so the
    # shape must be known here (the op also asserts this at compile time).
    if not isinstance(n_dim, StaticDim) or not isinstance(k_dim, StaticDim):
        raise ValueError(
            "_router_gate_mixed_gemv requires a static gate-weight shape, got"
            f" {gate_weight.shape}"
        )
    if hidden_states.shape[1] != k_dim:
        raise ValueError(
            "hidden_states K must match gate_weight K, got"
            f" {hidden_states.shape[1]} and {k_dim}"
        )

    return ops.custom(
        "mo.router.gate.mixed.gemv",
        device=hidden_states.device,
        values=[hidden_states, gate_weight],
        out_types=[
            TensorType(
                dtype=DType.float32,
                shape=[hidden_states.shape[0], n_dim],
                device=hidden_states.device,
            )
        ],
    )[0].tensor


def smallm_streaming_matmul(
    a: TensorValue,
    b_shuffled: TensorValue,
    b: TensorValue,
) -> TensorValue:
    """Computes bf16 ``a @ b^T`` over a smallm-preshuffled weight.

    ``b_shuffled`` must already be in the fragment-major layout produced by
    :func:`max.pipelines.weights.smallm_preshuffle.preshuffle_smallm_b` (a
    one-time CPU permutation at weight-load time); the layout is private to
    this op pair and a row-major weight here computes garbage. ``b`` is the
    same weight row-major: the streaming kernel serves ``M <= 32`` (its
    measured win band) and larger batches fall back to generic matmul
    dispatch over ``b`` at execute time, so the op is never worse than the
    dispatch it replaces. MiniMax-M3's MTP draft emits this op explicitly on
    MI355X for its decode-band vocab projections; nothing routes here through
    generic matmul dispatch.
    """
    _check_rank(2, a=a, b_shuffled=b_shuffled, b=b)
    _check_dtype(DType.bfloat16, a=a, b_shuffled=b_shuffled, b=b)
    _check_same_device(a=a, b_shuffled=b_shuffled, b=b)

    n_dim = b_shuffled.shape[0]
    k_dim = b_shuffled.shape[1]
    if not isinstance(n_dim, StaticDim) or not isinstance(k_dim, StaticDim):
        raise ValueError(
            "smallm_streaming_matmul requires a static weight shape, got"
            f" {b_shuffled.shape}"
        )
    if int(n_dim) % 16 != 0 or int(k_dim) % 256 != 0:
        raise ValueError(
            "smallm_streaming_matmul requires N % 16 == 0 and K % 256 == 0,"
            f" got [{n_dim}, {k_dim}]"
        )
    if a.shape[1] != k_dim:
        raise ValueError(
            f"activation K must match weight K, got {a.shape[1]} and {k_dim}"
        )
    if b.shape != b_shuffled.shape:
        raise ValueError(
            "the row-major weight must match the shuffled weight shape, got"
            f" {b.shape} and {b_shuffled.shape}"
        )

    # The second output is a graph-managed workspace for the op's activation
    # shuffle (32 rows covers every streaming band; m > 32 falls back and
    # ignores it). Graph memory keeps captured launch pointers valid across
    # device-graph replays, unlike an execute-time transient allocation.
    return ops.custom(
        "mo.smallm.streaming.matmul",
        device=a.device,
        values=[a, b_shuffled, b],
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=[a.shape[0], n_dim],
                device=a.device,
            ),
            TensorType(
                dtype=DType.bfloat16,
                shape=[32, k_dim],
                device=a.device,
            ),
        ],
    )[0].tensor


def moe_eplb_remap(
    router_idx: TensorValue,
    logcnt: TensorValue,
    log2phy: TensorValue,
    layer_idx: TensorValue,
    *,
    num_log: int,
    max_replicas: int,
    n_experts_per_tok: int,
    hash_decorrelate: bool = False,
) -> TensorValue:
    """Fused EPLB logical-to-physical id remap.
    single Mojo kernel that caches the per-layer slice of logcnt and
    log2phy in shared memory and writes physical ids in one launch.

    The replica picker is deterministic position-mod
    (``r = (n*K + k) % logcnt[layer, log]``), bit-identical to the legacy
    chain when ``hash_decorrelate=False``. With ``hash_decorrelate=True``
    the flat position is xor-hashed with a Knuth multiplicative hash of the
    logical id before the modulo, breaking structured position-vs-cnt
    alignment without warp primitives.

    Args:
        router_idx: [num_tokens, n_experts_per_tok] int32 logical
            expert ids from the gate.
        logcnt: [num_moe_layers, num_log] int32 replica count per
            (layer, logical id).
        log2phy: [num_moe_layers, num_log, max_replicas] int32
            physical-id table.
        layer_idx: Rank-0 or rank-1 [1] int32 scalar tensor on the same
            device as router_idx. Rank-0 is reshaped to [1] to match
            the kernel signature.
        num_log: Number of logical experts (comptime).
        max_replicas: Maximum replicas per logical expert (comptime).
        n_experts_per_tok: Top-K experts per token (comptime). Must be a
            power of two.
        hash_decorrelate: If True, xor-hash position before the modulo.
            Defaults to False to preserve legacy routing distribution.

    Returns:
        [num_tokens, n_experts_per_tok] int32 physical expert ids.
    """
    _check_dtype(
        DType.int32,
        router_idx=router_idx,
        logcnt=logcnt,
        log2phy=log2phy,
        layer_idx=layer_idx,
    )
    _check_rank(2, router_idx=router_idx, logcnt=logcnt)
    _check_rank(3, log2phy=log2phy)

    if layer_idx.rank not in (0, 1):
        raise ValueError(
            f"expected layer_idx of rank 0 or 1 but got {layer_idx.rank}"
        )

    # Kernel expects rank-1 [1] — reshape rank-0 scalar without copy.
    if layer_idx.rank == 0:
        layer_idx = ops.reshape(layer_idx, [1])

    if router_idx.shape[1] != n_experts_per_tok:
        raise ValueError(
            "expected router_idx of shape [num_tokens, n_experts_per_tok], "
            f"got shape {router_idx.shape} with n_experts_per_tok="
            f"{n_experts_per_tok}"
        )

    if logcnt.shape[1] != num_log:
        raise ValueError(
            "expected logcnt of shape [num_moe_layers, num_log], "
            f"got shape {logcnt.shape} with num_log={num_log}"
        )

    if log2phy.shape[1] != num_log or log2phy.shape[2] != max_replicas:
        raise ValueError(
            "expected log2phy of shape "
            "[num_moe_layers, num_log, max_replicas], "
            f"got shape {log2phy.shape} with num_log={num_log}, "
            f"max_replicas={max_replicas}"
        )

    parameters: dict[str, bool | int | str | DType] = {
        "num_log": num_log,
        "max_replicas": max_replicas,
        "K": n_experts_per_tok,
        "hash_decorrelate": hash_decorrelate,
    }

    return ops.custom(
        "mo.moe.eplb.remap",
        device=router_idx.device,
        values=[router_idx, logcnt, log2phy, layer_idx],
        out_types=[
            TensorType(
                dtype=DType.int32,
                shape=router_idx.shape,
                device=router_idx.device,
            ),
        ],
        parameters=parameters,
    )[0].tensor


def moe_router_single_group_eplb(
    expert_scores: TensorValue,
    expert_bias: TensorValue,
    logcnt: TensorValue,
    log2phy: TensorValue,
    layer_idx: TensorValue,
    *,
    n_routed_experts: int,
    n_experts_per_tok: int,
    norm_weights: bool,
    num_log: int,
    max_replicas: int,
    hash_decorrelate: bool,
    routed_scaling_factor: float,
) -> tuple[TensorValue, TensorValue, TensorValue]:
    """Fused single-group MoE router + EPLB log->phy remap.

    Replaces the chained ``moe_router_group_limited`` (n_groups==1) →
    ``moe_eplb_remap`` for the single-group path. Returns physical
    expert ids, logical expert ids (kept for the EPLB stats histogram),
    and routing weights in one launch.
    """
    if expert_bias.rank != 1:
        raise ValueError(
            f"expected expert_bias of rank 1 but got {expert_bias.rank}"
        )
    if expert_bias.shape[0] != expert_scores.shape[1]:
        raise ValueError(
            f"expected expert_bias of shape [num_experts] but got {expert_bias.shape}"
        )

    _check_dtype(
        DType.int32, logcnt=logcnt, log2phy=log2phy, layer_idx=layer_idx
    )
    _check_rank(2, logcnt=logcnt)
    _check_rank(3, log2phy=log2phy)
    if layer_idx.rank == 0:
        layer_idx = ops.reshape(layer_idx, [1])
    if logcnt.shape[1] != num_log:
        raise ValueError(
            f"expected logcnt of shape [L, num_log], got {logcnt.shape} num_log={num_log}"
        )
    if log2phy.shape[1] != num_log or log2phy.shape[2] != max_replicas:
        raise ValueError(
            f"expected log2phy of shape [L, num_log, max_replicas], got {log2phy.shape}"
        )

    parameters: dict[str, int | str | DType | bool] = {
        "n_routed_experts": n_routed_experts,
        "n_experts_per_tok": n_experts_per_tok,
        "norm_weights": norm_weights,
        "num_log": num_log,
        "max_replicas": max_replicas,
        "hash_decorrelate": hash_decorrelate,
    }

    results = ops.custom(
        "mo.moe.single.group.router.eplb",
        device=expert_scores.device,
        values=[
            expert_scores,
            expert_bias,
            logcnt,
            log2phy,
            layer_idx,
            ops.constant(
                routed_scaling_factor, DType.float32, device=DeviceRef.CPU()
            ),
        ],
        out_types=[
            TensorType(  # expert_indices_phy
                dtype=DType.int32,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),
            TensorType(  # expert_indices_log
                dtype=DType.int32,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),
            TensorType(  # expert_weights
                dtype=expert_scores.dtype,
                shape=[expert_scores.shape[0], n_experts_per_tok],
                device=expert_scores.device,
            ),
        ],
        parameters=parameters,
    )

    return (results[0].tensor, results[1].tensor, results[2].tensor)


def grouped_matmul_ragged(
    hidden_states: TensorValue,
    weight: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats: TensorValue,
) -> TensorValue:
    """Grouped matmul used in MoE layer.

    `hidden_states` and `expert_start_indices` are used together to implement
    the ragged tensor. `expert_start_indices` indicates where each group starts
    and ends in `hidden_states`

    `expert_ids` is the id of the expert for each group in `hidden_states`

    `expert_usage_stats` is a rank-1 ``uint32`` tensor laid out as
    ``[max_tokens_per_expert, num_active_experts]`` (the output of
    ``moe_create_indices``).
    """
    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError(
            "expected weight is of shape [num_experts, *,"
            f" {hidden_states.shape[1]}] but got {weight.shape}"
        )

    if expert_usage_stats.device != hidden_states.device:
        # This is a graph DeviceRef (``is_cpu()`` method) when building a Graph
        # and a driver Device (``is_host`` property) on the experimental-tensor
        # path.
        src_device = expert_usage_stats.device
        src_is_cpu = (
            src_device.is_cpu()
            if isinstance(src_device, DeviceRef)
            else src_device.is_host
        )
        if not src_is_cpu:
            raise ValueError(
                "grouped_matmul_ragged expected expert_usage_stats to be"
                " host-resident (CPU) or already on the compute device, but got"
                f" {src_device} with hidden_states on {hidden_states.device}; a"
                " device-to-device transfer of the expert-usage metadata is not"
                " supported."
            )
        expert_usage_stats = expert_usage_stats.to(hidden_states.device)

    output = ops.custom(
        "mo.grouped.matmul.ragged",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            expert_start_indices,
            expert_ids,
            expert_usage_stats,
        ],
        out_types=[
            TensorType(
                dtype=hidden_states.dtype,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
    )[0].tensor

    return output


def grouped_dynamic_block_scaled_matmul_amd(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    out_type: DType = DType.bfloat16,
    estimated_total_m: TensorValue | None = None,
    preshuffled_b: bool = False,
    a_scales_preshuffled: bool = False,
    a_scales_max_padded_m: int = 0,
    decode_grid_m_cap: int = 0,
    decode_grid_m_rows: int = 0,
) -> TensorValue:
    """Performs grouped NVFP4 matmul for MoE layers.

    Performs a grouped matmul with MXFP4 (4-bit) quantized inputs and weights.
    The inputs are packed as uint8 (2 MXFP4 values per byte) with float8_e8m0fnu
    scaling factors. MXFP4 uses fixed 1D block scaling with 32 elements per
    scale factor along the K dimension.

    ``hidden_states`` and ``expert_start_indices`` together implement the ragged
    tensor representation for variable-length expert inputs.

    Args:
        hidden_states: The input activations with shape
            ``[total_tokens, K/2]`` at MXFP4 or ``[total_tokens, K]`` at MXFP8,
            where K is the unpacked hidden dimension.
        weight: The expert weights, shaped ``[num_experts, N, K/2]`` at MXFP4
            or ``[num_experts, N, K]`` at MXFP8. Must share ``hidden_states``'
            dtype: uint8 (packed MXFP4) or float8_e4m3fn (MXFP8).
        a_scales: Scaling factors for inputs with shape
            ``[num_scale_rows, K/32]``. Dtype must be float8_e8m0fnu.
        b_scales: Scaling factors for weights with shape
            ``[num_experts, N, K/32]``. Dtype must be float8_e8m0fnu.
        expert_start_indices: Indices indicating where each expert's tokens
            start in ``hidden_states``.
        expert_ids: The expert ID for each group.
        expert_usage_stats_host: A tensor containing [max_tokens_per_expert,
            num_active_experts].
        out_type: Output dtype. Defaults to bfloat16.
        estimated_total_m: The estimated total number of tokens.
        decode_grid_m_cap: Decode-band gate on the AMD preb path; 0 disables.
            Selects the band; `decode_grid_m_rows` bounds the grid.
        decode_grid_m_rows: Rows grid.y must cover per expert on the decode
            bands. Ignored unless ``preshuffled_b``.

    Returns:
        The matmul result with shape ``[total_tokens, N]`` and dtype ``out_type``.
    """
    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    weight_k = weight.shape[2]
    hidden_k = hidden_states.shape[1]
    if weight_k != hidden_k or weight.shape[0] != expert_ids.shape[0]:
        raise ValueError(
            "expected weight is of shape [num_experts, *, "
            f"{hidden_k}] but got {weight.shape}"
        )

    # The kernel infers the packing from the shapes, so both operands need only
    # agree on one MX dtype: uint8 (MXFP4) or float8_e4m3fn (MXFP8).
    if hidden_states.dtype != weight.dtype or hidden_states.dtype not in (
        DType.uint8,
        DType.float8_e4m3fn,
    ):
        raise TypeError(
            "hidden_states and weight must share one MX dtype, either uint8 "
            "(MXFP4) or float8_e4m3fn (MXFP8), but got "
            f"{hidden_states.dtype}, {weight.dtype}"
        )

    if (a_scales.dtype != b_scales.dtype) or (
        a_scales.dtype != DType.float8_e8m0fnu
    ):
        raise TypeError(
            "a_scales and b_scales dtypes must be float8_e8m0fnu for MXFP4, "
            f"but got {a_scales.dtype}, {b_scales.dtype}"
        )

    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )

    if expert_ids.rank != 1:
        raise ValueError(
            f"expected expert_ids of rank 1 but got {expert_ids.rank}"
        )
    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            "expert_start_indices dtype must be uint32, but got"
            f" {expert_start_indices.dtype}"
        )
    if expert_start_indices.rank != 1:
        raise ValueError(
            "expected expert_start_indices of rank 1 but got"
            f" {expert_start_indices.rank}"
        )

    if a_scales.rank != 2 or b_scales.rank != 3:
        raise ValueError(
            "expected a_scales of rank 2 and b_scales of rank 3 but got"
            f" {a_scales.rank} and {b_scales.rank}"
        )

    MXFP4_SF_VECTOR_SIZE = 32

    # Shapes are in BYTES, so recover the element count before counting scale
    # groups: MXFP4 stores two elements per byte, MXFP8 one.
    elems_per_byte = 2 if hidden_states.dtype == DType.uint8 else 1

    a_scales_dim_1 = ceildiv(
        hidden_states.shape[1] * elems_per_byte, Dim(MXFP4_SF_VECTOR_SIZE)
    )
    if a_scales.shape[1] != a_scales_dim_1:
        raise ValueError(
            "a_scales shape must be "
            f"[*, {a_scales_dim_1}]"
            f" but got {a_scales.shape}"
        )

    b_scales_dim_2 = ceildiv(
        weight.shape[2] * elems_per_byte, Dim(MXFP4_SF_VECTOR_SIZE)
    )
    if (
        b_scales.shape[0] != weight.shape[0]
        or b_scales.shape[1] != weight.shape[1]
        or b_scales.shape[2] != b_scales_dim_2
    ):
        raise ValueError(
            f"b_scales shape must be [{weight.shape[0]}, {weight.shape[1]},"
            f" {b_scales_dim_2}] but got {b_scales.shape}"
        )

    # `estimated_total_m` defaults to 0 (unknown). When `preshuffled_b` is
    # True, the AMD preb kernel uses it to choose between persistent (small)
    # and direct 3D-grid (large) dispatch paths. Ignored on the dense path.
    if estimated_total_m is None:
        estimated_total_m_arg = ops.constant(
            0, dtype=DType.uint32, device=hidden_states.device
        )
    else:
        estimated_total_m_arg = estimated_total_m.cast(DType.uint32)

    # The preb kernel expects A-scales in `Shuffler.scale_4d_grouped_layout`
    # (i32 cells of 2x2 E8M0 bytes). Activations are quantized row-major by
    # `quantize_dynamic_block_scaled_mxfp4` upstream, so insert the per-step
    # preshuffle here. B-scales are static and preshuffled once at load.
    #
    # When `a_scales_preshuffled=True` (KS64 down-proj fusion), the
    # upstream `ep.fused_silu.mxfp4` kernel already wrote the scale directly
    # into the slot layout, so we skip the standalone preshuffle entirely.
    # Preshuffle must run exactly once: non-EP + up-proj keep `a_scales_preshuffled=False`.
    if preshuffled_b and not a_scales_preshuffled:
        a_scales = block_scaled_preshuffle_grouped_scale_4d(
            a_scales,
            expert_start_indices,
            expert_usage_stats_host[0].cast(DType.uint32),
            expert_usage_stats_host[1].cast(DType.uint32),
            num_experts=int(weight.shape[0]),
        )

    # The matmul derives the A-scale per-expert slot stride as
    # `align_up(max_num_tokens_per_expert, 32)`. On the non-fused path the
    # standalone preshuffle used the same runtime `expert_usage_stats[0]`, so
    # the writer and reader slot strides agree. On the fused path
    # (`a_scales_preshuffled`), the producer (`ep.fused_silu.mxfp4`) wrote the
    # slots with the *graph-build-time* stride `align_up(a_scales_max_padded_m,
    # 32)`; the matmul MUST read with that same constant — NOT the runtime max
    # — or, when the runtime max is below the build-time bound (the common
    # decode case), it reads the wrong expert's scale slot.
    if a_scales_preshuffled:
        if a_scales_max_padded_m <= 0:
            raise ValueError(
                "a_scales_max_padded_m must be > 0 when"
                " a_scales_preshuffled=True"
            )
        max_num_tokens_arg = ops.constant(
            a_scales_max_padded_m,
            dtype=expert_usage_stats_host.dtype,
            device=expert_usage_stats_host.device,
        )
    else:
        max_num_tokens_arg = expert_usage_stats_host[0]

    decode_grid_m_cap_arg = ops.constant(
        decode_grid_m_cap, dtype=DType.uint32, device=DeviceRef.CPU()
    )

    decode_grid_m_rows_arg = ops.constant(
        decode_grid_m_rows, dtype=DType.uint32, device=DeviceRef.CPU()
    )

    output = ops.custom(
        "mo.grouped.matmul.block.scaled.amd",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            max_num_tokens_arg,
            expert_usage_stats_host[1],
            estimated_total_m_arg,
            decode_grid_m_cap_arg,
            decode_grid_m_rows_arg,
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
        parameters={
            "preshuffled_b": preshuffled_b,
            # Both formats reach the kernel as raw bytes, so this is what tells
            # them apart: 16 bytes per lane at MXFP4, 32 at MXFP8.
            "lane_bytes": 32 // elems_per_byte,
        },
    )[0].tensor

    return output


def grouped_dynamic_scaled_mxfp6_matmul(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    fp6_format: str = "e2m3",
    out_type: DType = DType.bfloat16,
    estimated_total_m: TensorValue | None = None,
    decode_grid_m_cap: int = 0,
    decode_grid_m_rows: int = 0,
    a_scales_preshuffled: bool = False,
    a_scales_max_padded_m: int = 0,
) -> TensorValue:
    """Performs a grouped MXFP6 matmul for MoE layers.

    The FP6 sibling of :func:`grouped_dynamic_scaled_mxfp4_matmul`. Both
    operands are packed FP6 bytes (four codes per three bytes) with E8M0 scales
    over 32-element K blocks.

    Preshuffled-B only: an FP6 lane fragment is 24 bytes and reaches the MFMA
    plane-split, a layout the dense row-major grouped kernel has no path for.
    ``weight`` must therefore already carry the plane-split permutation from
    ``preshuffle_mxfp4_b_experts(..., lane_bytes=MXFP6_LANE_BYTES)``.

    Args:
        hidden_states: Packed activations ``[total_tokens, K * 3 // 4]``.
        weight: Plane-split preshuffled expert weights
            ``[num_experts, N, K * 3 // 4]``.
        a_scales: E8M0 activation scales ``[num_scale_rows, K // 32]``.
        b_scales: E8M0 weight scales ``[num_experts, N, K // 32]``.
        expert_start_indices: Where each expert's tokens start.
        expert_ids: The expert ID for each group.
        expert_usage_stats_host: ``[max_tokens_per_expert, num_active_experts]``.
        fp6_format: The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.
        out_type: Output dtype.
        estimated_total_m: Estimated total token count, used to pick the
            persistent vs direct kernel band.
        decode_grid_m_cap: Decode-band gate; 0 disables.
        decode_grid_m_rows: Rows grid.y must cover per expert on the decode
            bands.

    Returns:
        The matmul result, ``[total_tokens, N]``.
    """
    if not _is_amd_gpu():
        raise ValueError(
            "MXFP6 is supported on AMD CDNA4 (gfx950) only: the kernels issue"
            " through the f8f6f4 MFMA, which has no NVIDIA equivalent. Use"
            " float8_e4m3fn or float4_e2m1fnx2 on NVIDIA."
        )
    fp6_code = _fp6_format_code(fp6_format)

    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")
    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )
    if weight.shape[2] != hidden_states.shape[1]:
        raise ValueError(
            "expected weight of shape [num_experts, *, "
            f"{hidden_states.shape[1]}] but got {weight.shape}"
        )
    if hidden_states.dtype != DType.uint8 or weight.dtype != DType.uint8:
        raise TypeError(
            "MXFP6 operands are packed into uint8 (four codes per three "
            f"bytes), got {hidden_states.dtype}, {weight.dtype}"
        )
    if (
        a_scales.dtype != DType.float8_e8m0fnu
        or b_scales.dtype != DType.float8_e8m0fnu
    ):
        raise TypeError(
            "a_scales and b_scales dtypes must be float8_e8m0fnu, but got "
            f"{a_scales.dtype}, {b_scales.dtype}"
        )
    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )
    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            "expert_start_indices dtype must be uint32, but got"
            f" {expert_start_indices.dtype}"
        )
    if a_scales.rank != 2 or b_scales.rank != 3:
        raise ValueError(
            "expected a_scales of rank 2 and b_scales of rank 3 but got"
            f" {a_scales.rank} and {b_scales.rank}"
        )

    MX_SF_VECTOR_SIZE = 32
    a_scales_dim_1 = ceildiv(
        hidden_states.shape[1] * 4 // 3, Dim(MX_SF_VECTOR_SIZE)
    )
    if a_scales.shape[1] != a_scales_dim_1:
        raise ValueError(
            f"a_scales shape must be [*, {a_scales_dim_1}] but got "
            f"{a_scales.shape}"
        )
    b_scales_dim_2 = ceildiv(weight.shape[2] * 4 // 3, Dim(MX_SF_VECTOR_SIZE))
    if (
        b_scales.shape[0] != weight.shape[0]
        or b_scales.shape[1] != weight.shape[1]
        or b_scales.shape[2] != b_scales_dim_2
    ):
        raise ValueError(
            f"b_scales shape must be [{weight.shape[0]}, {weight.shape[1]},"
            f" {b_scales_dim_2}] but got {b_scales.shape}"
        )

    if estimated_total_m is None:
        estimated_total_m_arg = ops.constant(
            0, dtype=DType.uint32, device=hidden_states.device
        )
    else:
        estimated_total_m_arg = estimated_total_m.cast(DType.uint32)

    if a_scales_preshuffled:
        if a_scales_max_padded_m <= 0:
            raise ValueError(
                "a_scales_max_padded_m must be > 0 when"
                " a_scales_preshuffled=True"
            )
        max_num_tokens_arg = ops.constant(
            a_scales_max_padded_m,
            dtype=expert_usage_stats_host.dtype,
            device=expert_usage_stats_host.device,
        )
    else:
        a_scales = block_scaled_preshuffle_grouped_scale_4d(
            a_scales,
            expert_start_indices,
            expert_usage_stats_host[0].cast(DType.uint32),
            expert_usage_stats_host[1].cast(DType.uint32),
            num_experts=int(weight.shape[0]),
        )
        max_num_tokens_arg = expert_usage_stats_host[0]

    return ops.custom(
        "mo.grouped.matmul.block.scaled.mxfp6",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            max_num_tokens_arg,
            expert_usage_stats_host[1],
            estimated_total_m_arg,
            ops.constant(
                decode_grid_m_cap, dtype=DType.uint32, device=DeviceRef.CPU()
            ),
            ops.constant(
                decode_grid_m_rows, dtype=DType.uint32, device=DeviceRef.CPU()
            ),
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
        parameters={"FP6_FORMAT": fp6_code},
    )[0].tensor


def grouped_matmul_block_scaled(
    hidden_states: TensorValueLike,
    weight: TensorValueLike,
    a_scales: TensorValueLike,
    b_scales: TensorValueLike,
    expert_start_indices: TensorValueLike,
    a_scale_offsets: TensorValueLike,
    expert_ids: TensorValueLike,
    expert_scales: TensorValueLike,
    expert_usage_stats_host: TensorValueLike,
    out_type: DType = DType.bfloat16,
    estimated_total_m: TensorValueLike | None = None,
) -> TensorValue:
    """Performs a grouped block-scaled matmul for MoE layers.

    Supports four operand/scale combinations, which the op tells apart from the
    operand and scale dtypes alone. Every one uses fixed 1D block scaling along
    the K dimension:

    - NVFP4: both operands packed ``uint8``, ``float8_e4m3fn`` scales over
      16-element K groups.
    - MXFP4: both operands packed ``uint8``, ``float8_e8m0fnu`` scales over
      32-element K groups.
    - MXFP8: both operands ``float8_e4m3fn``, ``float8_e8m0fnu`` scales over
      32-element K groups.
    - W4A8: ``float8_e4m3fn`` activations against packed ``uint8`` weights,
      ``float8_e8m0fnu`` scales over 32-element K groups.

    Packed ``uint8`` carries 2 4-bit E2M1 values per byte, so a packed operand's
    row is ``K/2`` wide. W4A8 is the one combination whose two operands differ,
    and it pairs an unpacked activation row with a packed weight row. It also
    requires ``K`` to be a multiple of 128, which the padded FP4 TMA copy that
    feeds the weights into shared memory imposes.

    ``hidden_states`` and ``expert_start_indices`` together implement the ragged
    tensor representation for variable-length expert inputs.

    Args:
        hidden_states: The input activations with shape ``[total_tokens, K/2]``
            for packed ``uint8`` or ``[total_tokens, K]`` for
            ``float8_e4m3fn``, where K is the unpacked hidden dimension.
        weight: The expert weights with shape ``[num_experts, N, K/2]`` for
            packed ``uint8`` or ``[num_experts, N, K]`` for
            ``float8_e4m3fn``. Sized independently of ``hidden_states``, so
            W4A8 combines a ``[total_tokens, K]`` activation with a
            ``[num_experts, N, K/2]`` weight.
        a_scales: Scaling factors for inputs with shape
            ``[num_scale_rows, K_groups, 32, 4, 4]``, where ``K_groups`` is
            ``ceildiv(K, 4 * group_size)`` for the combination's K group size.
            Dtype must be float8_e4m3fn (NVFP4) or float8_e8m0fnu
            (MXFP4/MXFP8/W4A8).
        b_scales: Scaling factors for weights with shape
            ``[num_experts, N_groups, K_groups, 32, 4, 4]``. Dtype must match
            ``a_scales``.
        expert_start_indices: Indices indicating where each expert's tokens
            start in ``hidden_states``.
        a_scale_offsets: The offsets of the input scale tiles for each expert.
        expert_ids: The expert ID for each group.
        expert_scales: Per-expert scaling factors with shape ``[num_experts]``.
            Dtype must be float32. Multiplied with the matmul output in the
            epilogue.
        expert_usage_stats_host: A tensor containing [max_tokens_per_expert,
            num_active_experts].
        out_type: Output dtype. Defaults to bfloat16.
        estimated_total_m: The estimated total number of tokens.

    Returns:
        The matmul result with shape ``[total_tokens, N]`` and dtype ``out_type``.
    """
    hidden_states = TensorValue(hidden_states)
    weight = TensorValue(weight)
    a_scales = TensorValue(a_scales)
    b_scales = TensorValue(b_scales)
    expert_start_indices = TensorValue(expert_start_indices)
    a_scale_offsets = TensorValue(a_scale_offsets)
    expert_ids = TensorValue(expert_ids)
    expert_scales = TensorValue(expert_scales)
    expert_usage_stats_host = TensorValue(expert_usage_stats_host)
    if estimated_total_m:
        estimated_total_m = TensorValue(estimated_total_m)

    _check_rank(2, hidden_states=hidden_states)
    _check_rank(3, weight=weight)
    _check_rank(5, a_scales=a_scales)
    _check_rank(6, b_scales=b_scales)
    _check_rank(1, expert_start_indices=expert_start_indices)
    _check_rank(1, a_scale_offsets=a_scale_offsets)
    _check_rank(1, expert_ids=expert_ids)

    _check_dtype(DType.int32, expert_ids=expert_ids)
    _check_dtype(DType.uint32, expert_start_indices=expert_start_indices)

    # W4A8 is the one pair whose operands differ: E4M3 activations against
    # nibble-packed E2M1 weights, which the FP4 TMA copy pads into shared
    # memory. The scale dtype is part of the pair, not incidental: the kernel
    # only implements it on E8M0 group-32 scales, so admitting the mixed pair
    # under NVFP4 scales would trade this graph-build error for a Mojo
    # comptime failure partway through graph compilation.
    is_w4a8 = (
        hidden_states.dtype == DType.float8_e4m3fn
        and weight.dtype == DType.uint8
        and a_scales.dtype == DType.float8_e8m0fnu
    )
    if not is_w4a8:
        _check_same_dtype(hidden_states=hidden_states, weight=weight)
    _check_same_dtype(a_scales=a_scales, b_scales=b_scales)

    _check_same_device(
        hidden_states=hidden_states,
        weight=weight,
        a_scales=a_scales,
        b_scales=b_scales,
        expert_start_indices=expert_start_indices,
        a_scale_offsets=a_scale_offsets,
        expert_ids=expert_ids,
        expert_scales=expert_scales,
    )

    if hidden_states.dtype not in (DType.uint8, DType.float8_e4m3fn):
        raise TypeError(
            "hidden_states dtype must be uint8 (NVFP4/MXFP4) or "
            f"float8_e4m3fn (MXFP8), but got {hidden_states.dtype}"
        )

    if a_scales.dtype not in (DType.float8_e4m3fn, DType.float8_e8m0fnu):
        raise TypeError(
            "a_scales dtype must be float8_e4m3fn (NVFP4) or "
            f"float8_e8m0fnu (MXFP4/MXFP8), but got {a_scales.dtype}"
        )

    # Row lengths are in storage bytes, so a nibble-packed weight row is half
    # the activations'. Compare in elements.
    weight_k_factor = 2 if is_w4a8 else 1
    weight_k = weight.shape[2] * weight_k_factor
    hidden_k = hidden_states.shape[1]
    if weight_k != hidden_k or weight.shape[0] != expert_ids.shape[0]:
        raise ValueError(
            "expected weight is of shape [num_experts, *, "
            f"{hidden_k // weight_k_factor}] but got {weight.shape}"
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    # Infer SF_VECTOR_SIZE from scale dtype: NVFP4=16, MXFP4/MXFP8=32.
    SF_VECTOR_SIZE = 32 if a_scales.dtype == DType.float8_e8m0fnu else 16
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * SF_VECTOR_SIZE
    packed_k_factor = 2 if hidden_states.dtype == DType.uint8 else 1
    unpacked_hidden_k = hidden_states.shape[1] * packed_k_factor

    a_scales_dim_1 = ceildiv(unpacked_hidden_k, Dim(SF_K_GROUP_SIZE))
    if (
        a_scales.shape[1] != a_scales_dim_1
        or a_scales.shape[2] != SF_ATOM_M[0]
        or a_scales.shape[3] != SF_ATOM_M[1]
        or a_scales.shape[4] != SF_ATOM_K
    ):
        raise ValueError(
            f"a_scales shape must be [*, {a_scales_dim_1}, {SF_ATOM_M[0]},"
            f" {SF_ATOM_M[1]}, {SF_ATOM_K}] but got {a_scales.shape}"
        )

    b_scales_dim_1 = ceildiv(weight.shape[1], Dim(SF_MN_GROUP_SIZE))
    b_scales_dim_2 = ceildiv(
        weight.shape[2] * weight_k_factor * packed_k_factor,
        Dim(SF_K_GROUP_SIZE),
    )
    if (
        b_scales.shape[0] != weight.shape[0]
        or b_scales.shape[1] != b_scales_dim_1
        or b_scales.shape[2] != b_scales_dim_2
        or b_scales.shape[3] != SF_ATOM_M[0]
        or b_scales.shape[4] != SF_ATOM_M[1]
        or b_scales.shape[5] != SF_ATOM_K
    ):
        raise ValueError(
            f"b_scales shape must be [{weight.shape[0]}, {b_scales_dim_1},"
            f" {b_scales_dim_2}, {SF_ATOM_M[0]}, {SF_ATOM_M[1]}, {SF_ATOM_K}]"
            f" but got {b_scales.shape}"
        )

    output_type = TensorType(
        dtype=out_type,
        shape=[hidden_states.shape[0], weight.shape[1]],
        device=hidden_states.device,
    )
    # Emitted as a first-class composite op (lowers 1:1 to the
    # `mo.composite.grouped_matmul_block_scaled` kernel) so the MegaFFN fusion
    # can match a typed op rather than a string-keyed `mo.custom`.
    output = Graph.current._add_op_generated(
        mo.CompositeGroupedMatmulBlockScaledOp,
        output_type,
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        expert_ids,
        a_scale_offsets,
        expert_scales,
        estimated_total_m or expert_usage_stats_host[0],
        expert_usage_stats_host[1],
    )[0].tensor

    return output


def grouped_matmul_blocked_swiglu(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    a_scale_offsets: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    expert_scales: TensorValue | None = None,
    c_input_scales: TensorValue | None = None,
    estimated_total_m: TensorValue | None = None,
    clamp_activation: bool = False,
    swiglu_alpha: float = 0.0,
    swiglu_limit: float = 0.0,
) -> tuple[TensorValue, TensorValue]:
    """Performs fused grouped block-scaled matmul + SwiGLU for NVIDIA MoE.

    Replaces the two-step chain ``grouped_matmul_block_scaled`` (BF16) ->
    ``fused_silu_quantized`` with a single SM100 kernel whose epilogue produces
    NVFP4 or MXFP8 activations and a 5D scale tile directly.

    The caller must pre-permute ``weight`` and ``b_scales`` on the N axis
    with ``sigma(2i)=i, sigma(2i+1)=D+i`` (``D = moe_dim``, ``N = 2D``) so
    that adjacent matmul-output columns carry ``(gate, up)`` pairs. The
    fused output is byte-identical to the chained reference under the
    kernel's default ``match_bf16=True`` setting.

    Args:
        hidden_states: The input activations. Dtype must be ``uint8`` for packed
            NVFP4 or ``float8_e4m3fn`` for MXFP8.
        weight: The sigma-permuted expert weights with shape
            ``[num_experts, 2D, K]`` using the same storage dtype and packed K
            convention as ``hidden_states``.
        a_scales: Scaling factors for inputs with shape
            ``[num_scale_rows, K_groups, 32, 4, 4]``. Dtype must be
            ``float8_e4m3fn`` for NVFP4 or ``float8_e8m0fnu`` for MXFP8.
        b_scales: Scaling factors for weights with shape
            ``[num_experts, N_groups, K_groups, 32, 4, 4]``, with the
            matching sigma permutation already applied on the N axis.
        expert_start_indices: Per-expert token-prefix offsets (uint32, rank 1).
        a_scale_offsets: The offsets of the input scale tiles for each expert.
        expert_ids: The expert ID for each group.
        expert_scales: Per-expert scaling factors with shape ``[num_experts]``.
            Dtype must be float32. Multiplied with the matmul output in the
            epilogue.
        c_input_scales: Per-active-expert SiLU input scale used by the NVFP4
            quant epilogue. MXFP8 ignores this value.
        expert_usage_stats_host: A tensor containing [max_tokens_per_expert,
            num_active_experts].
        estimated_total_m: The estimated total number of tokens.

    Returns:
        Tuple ``(c_packed, c_swiglu_scales)`` where ``c_packed`` is packed
        NVFP4 ``uint8`` with shape ``[total_tokens, D/2]`` or MXFP8
        ``float8_e4m3fn`` with shape ``[total_tokens, D]``. The scale tile's
        first dim matches ``a_scales``'s first dim since the kernel re-uses
        ``a_scale_offsets`` as the per-expert SF offset for the output.
    """

    _check_rank(2, hidden_states=hidden_states)
    _check_rank(3, weight=weight)
    _check_rank(5, a_scales=a_scales)
    _check_rank(6, b_scales=b_scales)
    _check_rank(1, expert_start_indices=expert_start_indices)
    _check_rank(1, a_scale_offsets=a_scale_offsets)
    _check_rank(1, expert_ids=expert_ids)

    _check_dtype(DType.int32, expert_ids=expert_ids)
    _check_dtype(DType.uint32, expert_start_indices=expert_start_indices)

    _check_same_dtype(hidden_states=hidden_states, weight=weight)
    _check_same_dtype(a_scales=a_scales, b_scales=b_scales)

    dummy_scale = ops.broadcast_to(
        ops.constant(1.0, DType.float32, device=hidden_states.device),
        expert_ids.shape,
    )
    if expert_scales is None:
        expert_scales = dummy_scale
    if c_input_scales is None:
        c_input_scales = dummy_scale

    _check_same_device(
        hidden_states=hidden_states,
        weight=weight,
        a_scales=a_scales,
        b_scales=b_scales,
        expert_start_indices=expert_start_indices,
        a_scale_offsets=a_scale_offsets,
        expert_ids=expert_ids,
        expert_scales=expert_scales,
        c_input_scales=c_input_scales,
    )

    if hidden_states.dtype not in (DType.uint8, DType.float8_e4m3fn):
        raise TypeError(
            "hidden_states dtype must be uint8 (NVFP4/MXFP4) or "
            f"float8_e4m3fn (MXFP8), but got {hidden_states.dtype}"
        )

    if a_scales.dtype not in (DType.float8_e4m3fn, DType.float8_e8m0fnu):
        raise TypeError(
            "a_scales dtype must be float8_e4m3fn (NVFP4) or "
            f"float8_e8m0fnu (MXFP4/MXFP8), but got {a_scales.dtype}"
        )

    weight_k = weight.shape[2]
    hidden_k = hidden_states.shape[1]
    if weight_k != hidden_k or weight.shape[0] != expert_ids.shape[0]:
        raise ValueError(
            "expected weight is of shape [num_experts, *, "
            f"{hidden_k}] but got {weight.shape}"
        )

    if clamp_activation:
        if swiglu_alpha == 0.0 or swiglu_limit == 0.0:
            raise ValueError(
                "swiglu_alpha and swiglu_limit must be set when clamp_activation is True"
            )

    # N = 2D, so weight.shape[1] must be even and D = N // 2.
    n_dim = weight.shape[1]
    if isinstance(n_dim, StaticDim) and int(n_dim) % 2 != 0:
        raise ValueError(
            f"weight.shape[1] (= N = 2D) must be even, got {n_dim}"
        )
    d_dim = n_dim // 2

    is_mxfp8 = hidden_states.dtype == DType.float8_e4m3fn
    c_packed_type = TensorType(
        dtype=DType.float8_e4m3fn if is_mxfp8 else DType.uint8,
        shape=[hidden_states.shape[0], d_dim if is_mxfp8 else d_dim // 2],
        device=hidden_states.device,
    )
    # c_swiglu_scales shares its per-expert SF tile geometry with a_scales
    # (a_scale_offsets is re-used as c_swiglu_scales's per-expert offsets),
    # so its first dim matches a_scales' first dim. K-groups uses D as the
    # un-packed inner dim.
    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_VECTOR_SIZE = 32 if is_mxfp8 else 16
    SF_K_GROUP_SIZE = SF_ATOM_K * SF_VECTOR_SIZE
    c_swiglu_scales_type = TensorType(
        dtype=a_scales.dtype,
        shape=[
            a_scales.shape[0],
            ceildiv(d_dim, Dim(SF_K_GROUP_SIZE)),
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        ],
        device=hidden_states.device,
    )

    # Emitted as a first-class composite op (lowers 1:1 to the
    # `mo.composite.grouped_matmul_swiglu_nvfp4` kernel) so the MegaFFN fusion
    # can match a typed op rather than a string-keyed `mo.custom`. The SwiGLU
    # clamp params (`swiglu_alpha`/`swiglu_limit`) are host-scalar operands and
    # `clamp_activation` a comptime attribute, matching the kernel signature.
    results = Graph.current._add_op_generated(
        mo.CompositeGroupedMatmulSwigluNvfp4Op,
        c_packed_type,
        c_swiglu_scales_type,
        hidden_states,
        weight,
        a_scales,
        b_scales,
        expert_start_indices,
        expert_ids,
        a_scale_offsets,
        expert_scales,
        c_input_scales,
        estimated_total_m or expert_usage_stats_host[0],
        expert_usage_stats_host[1],
        ops.constant(swiglu_alpha, DType.float32, device=DeviceRef.CPU()),
        ops.constant(swiglu_limit, DType.float32, device=DeviceRef.CPU()),
        clamp_activation=builtin.BoolAttr(clamp_activation),
    )

    return results[0].tensor, results[1].tensor


def grouped_dynamic_scaled_fp8_matmul(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Grouped blockwise scaled matmul used in MoE layer.

    Perform a grouped blockwise scaled matmul of two tensors with scaling factors.
    `hidden_states` and `expert_start_indices` are used together to implement
    the ragged tensor.

    Args:
        hidden_states: The first tensor to multiply. (2D tensor)
        weight: The second tensor to multiply, must be transposed. (3D tensor)
        a_scales: The scaling factors for the first tensor. (2D tensor)
        b_scales: The scaling factors for the second tensor. (3D tensor)
        expert_start_indices: indicates where each group starts and ends in `hidden_states`.
        expert_ids: The id of the expert for each group in `hidden_states`.
        expert_usage_stats_host: The maximum number of tokens assigned to any expert, and the number of active experts.
        input_scale_spec: The scaling granularity for the input tensor.
        weight_scale_spec: The scaling granularity for the weight tensor.

    Returns:
        The result of the matmul operation.
    """
    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError(
            "expected weight is of shape [num_experts, *,"
            f" {hidden_states.shape[1]}] but got {weight.shape}"
        )

    if (hidden_states.dtype != weight.dtype) or (
        hidden_states.dtype != DType.float8_e4m3fn
    ):
        raise TypeError(
            "hidden_states and weight dtypes must be float8_e4m3fn, but got"
            f" {hidden_states.dtype}, {weight.dtype}"
        )

    if (a_scales.dtype != b_scales.dtype) or (
        a_scales.dtype not in (DType.float32, DType.bfloat16)
    ):
        raise TypeError(
            "a_scales and b_scales dtypes must be float32 or bfloat16 and"
            f" match, but got {a_scales.dtype}, {b_scales.dtype}"
        )

    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )

    if expert_ids.rank != 1:
        raise ValueError(
            f"expected expert_ids of rank 1 but got {expert_ids.rank}"
        )
    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            "expert_start_indices dtype must be uint32, but got"
            f" {expert_start_indices.dtype}"
        )
    if expert_start_indices.rank != 1:
        raise ValueError(
            "expected expert_start_indices of rank 1 but got"
            f" {expert_start_indices.rank}"
        )

    if a_scales.rank != 2 or b_scales.rank != 3:
        raise ValueError(
            "expected a_scales of rank 2 and b_scales of rank 3 but got"
            f" {a_scales.rank} and {b_scales.rank}"
        )

    if input_scale_spec.is_block and weight_scale_spec.is_block:
        # a_scale is of shape [ceildiv(K // BLOCK_SIZE), SeqLen-padded]
        # b_scale is of shape [num_of_experts, ceildiv(N // BLOCK_SIZE), ceildiv(K // BLOCK_SIZE)]
        if a_scales.rank != 2:
            raise ValueError(
                f"expected a_scales of rank 2 but got {a_scales.rank}"
            )
        if b_scales.rank != 3:
            raise ValueError(
                f"expected b_scales of rank 3 but got {b_scales.rank}"
            )

        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input block_size and weight block_size must be set for"
                " grouped blockwise scaling"
            )

        if (
            input_scale_spec.block_size[0] != 1
            or input_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "grouped blockwise scaling only supports (1,128) granularity"
                " for input"
            )
        if (
            weight_scale_spec.block_size[0] != 128
            or weight_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "grouped blockwise scaling only supports (128,128) granularity"
                " for weight"
            )
    else:
        raise ValueError("grouped FP8 matmul only supports blockwise scaling")

    output = ops.custom(
        "mo.grouped.matmul.dynamic.scaled.fp8",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host[0],
            expert_usage_stats_host[1],
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": input_scale_spec.block_size[0],
            "n_scale_granularity": weight_scale_spec.block_size[0],
            "k_scale_granularity": weight_scale_spec.block_size[1],
        },
    )[0].tensor

    return output


def _grouped_matmul_rowwise_dynamic_scaled_fp8(
    hidden_states: TensorValue,
    weight: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    expert_usage_stats_host: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Grouped (ragged MoE) FP8 matmul with rowwise weight + per-token scales.

    Drives ``mo.grouped.matmul.rowwise.dynamic.scaled.fp8`` (NVIDIA SM100 /
    B200). Computes, for each token ``t`` in expert group ``g`` and each output
    channel ``n``::

        out[t, n] = (sum_k a[t, k] * b[expert_ids[g], n, k])
                    * a_scale[t] * b_scale[expert_ids[g], n]

    This is the compressed-tensors FP8-dynamic layout (per-output-channel
    weight scale + per-token dynamic activation scale), e.g.
    ``RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic``. It is distinct
    from :func:`grouped_dynamic_scaled_fp8_matmul`, which handles the
    blockwise (1x128 act / 128x128 weight) layout.

    The kernel applies ``transpose_b=True``: ``weight`` must already be in
    ``[num_experts, N, K]`` orientation (K innermost), and the weight scale is
    per output channel ``N``.

    Args:
        hidden_states: The activations, ``float8_e4m3fn`` rank-2
            ``[total_tokens, K]``.
        weight: The expert weights, ``float8_e4m3fn`` rank-3
            ``[num_experts, N, K]`` (already transposed; K innermost).
        a_scales: Per-token activation scales, ``float32`` rank-2
            ``[total_tokens, 1]``.
        b_scales: Per-output-channel weight scales, ``float32`` rank-3
            ``[num_experts, N, 1]``.
        expert_start_indices: Where each group starts/ends in ``hidden_states``,
            ``uint32`` rank-1.
        expert_ids: The expert id for each group, ``int32`` rank-1.
        expert_usage_stats_host: ``[max_num_tokens_per_expert, num_active_experts]``
            on the host (CPU).
        out_type: The output dtype.

    Returns:
        The matmul output, ``[total_tokens, N]`` in ``out_type``.
    """
    if hidden_states.rank != 2:
        raise ValueError(
            f"expected hidden_states of rank 2 but got {hidden_states.rank}"
        )

    if weight.rank != 3:
        raise ValueError(f"expected weight of rank 3 but got {weight.rank}")

    # transpose_b=True: weight is [E, N, K], so its K dim (axis 2) must match
    # the activation K dim (axis 1).
    if (
        weight.shape[2] != hidden_states.shape[1]
        or weight.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError(
            "expected weight of shape [num_experts, N,"
            f" {hidden_states.shape[1]}] with num_experts ="
            f" {expert_ids.shape[0]} but got {weight.shape}"
        )

    if (hidden_states.dtype != weight.dtype) or (
        hidden_states.dtype != DType.float8_e4m3fn
    ):
        raise TypeError(
            "hidden_states and weight dtypes must be float8_e4m3fn, but got"
            f" {hidden_states.dtype}, {weight.dtype}"
        )

    if a_scales.rank != 2 or b_scales.rank != 3:
        raise ValueError(
            "expected a_scales of rank 2 and b_scales of rank 3 but got"
            f" {a_scales.rank} and {b_scales.rank}"
        )

    if a_scales.dtype != DType.float32 or b_scales.dtype != DType.float32:
        raise TypeError(
            "a_scales and b_scales dtypes must both be float32 for rowwise /"
            f" per-token granularity, but got {a_scales.dtype},"
            f" {b_scales.dtype}"
        )

    # Per-token activation scale: [total_tokens, 1]; per-channel weight scale:
    # [num_experts, N, 1].
    if a_scales.shape[1] != 1:
        raise ValueError(
            "expected per-token a_scales of shape [total_tokens, 1] but got"
            f" {a_scales.shape}"
        )
    if (
        b_scales.shape[2] != 1
        or b_scales.shape[0] != weight.shape[0]
        or b_scales.shape[1] != weight.shape[1]
    ):
        raise ValueError(
            "expected per-channel b_scales of shape [num_experts, N, 1]"
            f" matching weight [num_experts, N, K], but got b_scales"
            f" {b_scales.shape} and weight {weight.shape}"
        )

    if expert_ids.dtype != DType.int32:
        raise TypeError(
            f"expert_ids dtype must be int32, but got {expert_ids.dtype}"
        )

    if expert_ids.rank != 1:
        raise ValueError(
            f"expected expert_ids of rank 1 but got {expert_ids.rank}"
        )

    if expert_start_indices.dtype != DType.uint32:
        raise TypeError(
            "expert_start_indices dtype must be uint32, but got"
            f" {expert_start_indices.dtype}"
        )

    if expert_start_indices.rank != 1:
        raise ValueError(
            "expected expert_start_indices of rank 1 but got"
            f" {expert_start_indices.rank}"
        )

    return ops.custom(
        "mo.grouped.matmul.rowwise.dynamic.scaled.fp8",
        device=hidden_states.device,
        values=[
            hidden_states,
            weight,
            a_scales,
            b_scales,
            expert_start_indices,
            expert_ids,
            expert_usage_stats_host[0],
            expert_usage_stats_host[1],
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[hidden_states.shape[0], weight.shape[1]],
                device=hidden_states.device,
            ),
        ],
    )[0].tensor


def batched_dynamic_scaled_fp8_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Performs a batched blockwise scaled matmul of two tensors with scaling factors.

    Args:
        a: The first tensor to multiply (3D tensor).
        b: The second tensor to multiply, must be transposed (3D tensor).
        a_scales: The scaling factors for the first tensor (3D tensor).
        b_scales: The scaling factors for the second tensor (3D tensor).

    Returns:
        The result of the matmul operation.
    """
    if a.dtype != b.dtype:
        raise TypeError(
            f"a and b dtypes must match, but got {a.dtype}, {b.dtype}"
        )

    if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
        raise TypeError(
            "a_scales and b_scales dtypes must be float32, but got"
            f" {a_scales.dtype}, {b_scales.dtype}"
        )

    if a.rank != 3 or b.rank != 3:
        raise ValueError("A and B must be rank 3 tensors")

    if a_scales.rank != 3 or b_scales.rank != 3:
        raise ValueError("A_scales and B_scales must be rank 3 tensors")

    if a.shape[0] != b.shape[0]:
        raise ValueError(
            "The batch dimension of b must match the batch dimension of a"
        )

    if a.shape[2] != b.shape[2]:
        raise ValueError("A and B K dimension does not match")

    if a.dtype != b.dtype or a.dtype != DType.float8_e4m3fn:
        raise TypeError(
            f"a and b dtypes must be float8_e4m3fn, but got {a.dtype},"
            f" {b.dtype}"
        )

    if input_scale_spec.is_block and weight_scale_spec.is_block:
        # a_scale is of shape [batch_size, ceildiv(K, BLOCK_SIZE), M-padded]
        # b_scale is of shape [batch_size, ceildiv(N, BLOCK_SIZE), ceildiv(K, BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[0]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the"
                " batch dimension"
            )

        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input scale_granularity and weight scale_granularity must"
                " be set for batched blockwise scaling"
            )

        if (
            input_scale_spec.block_size[0] != 1
            or input_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "batched blockwise scaling only supports (1,128) granularity"
                " for input"
            )
        if (
            weight_scale_spec.block_size[0] != 128
            or weight_scale_spec.block_size[1] != 128
        ):
            raise ValueError(
                "batched blockwise scaling only supports (128,128) granularity"
                " for weight"
            )
    else:
        raise ValueError("unsupported FP8 scaling granularity")

    result = ops.custom(
        "mo.batched.matmul.dynamic.scaled.fp8",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[a.shape[0], a.shape[1], b.shape[1]],
                device=a.device,
            )
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": input_scale_spec.block_size[0],
            "n_scale_granularity": weight_scale_spec.block_size[0],
            "k_scale_granularity": weight_scale_spec.block_size[1],
        },
    )[0].tensor

    return result


def quantize_static_scaled_float8(
    x: TensorValue,
    scale: TensorValue,
    scale_is_inverted: bool = True,
    out_type: DType = DType.float8_e4m3fn,
) -> TensorValue:
    """Quantizes a rank-2 tensor to float8 using a static per-tensor scale.

    Args:
        x: Input tensor to quantize. Must be rank 2 with dtype ``float16``,
            ``bfloat16``, or ``float32``.
        scale: Scalar scale factor (shape ``[]`` or ``[1]``) residing on CPU.
        scale_is_inverted: When ``True`` (default), ``scale`` is interpreted
            as ``1 / max_val`` (inverted). When ``False``, it is the raw
            absolute-max scale.
        out_type: Output dtype. Defaults to ``DType.float8_e4m3fn``.

    Returns:
        A quantized :class:`~max.graph.TensorValue` with shape equal to ``x``
        and dtype ``out_type``.

    Raises:
        ValueError: If ``scale`` is not a scalar, ``x`` is not rank 2, ``x``
            dtype is unsupported, or ``scale`` is not on CPU.
    """
    if scale.shape not in [[], [1]]:
        raise ValueError(
            f"expected scale to be a scalar, but got shape of {scale.shape}"
        )

    if x.dtype not in [DType.float16, DType.bfloat16, DType.float32]:
        raise ValueError(
            "expected input dtype to be float16, bfloat16, or float32, but got"
            f" {x.dtype}"
        )

    if x.rank != 2:
        raise ValueError(f"expected input rank to be 2, but got {x.rank}")

    if scale.device != DeviceRef.CPU():
        raise ValueError(f"expected scale to be on CPU, but got {scale.device}")

    return ops.custom(
        "mo.quantize_static_scaled_float8",
        device=x.device,
        values=[x, scale.reshape([])],
        parameters={"scale_is_inverted": scale_is_inverted},
        out_types=[TensorType(dtype=out_type, shape=x.shape, device=x.device)],
    )[0].tensor


def quantize_tensor_dynamic_scaled_float8(
    input: TensorValue,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    scale_ub: float = 1200.0,
    group_size_or_per_token: int = -1,
    out_type: DType = DType.float8_e4m3fn,
    scales_type: DType = DType.bfloat16,
) -> tuple[TensorValue, TensorValue]:
    """Quantizes a rank-2 tensor to float8 using a dynamic per-tensor scale.

    Args:
        input: The input tensor to quantize.
        scale_ub: The upper bound of the scale factor.
        group_size_or_per_token: The group size for quantization. When set to -1,
            the quantization is column-wise.
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor and the scales.
    """
    if input.rank != 2:
        raise ValueError("input must be rank 2 tensor")

    if out_type != DType.float8_e4m3fn:
        raise ValueError("out_type must be float8_e4m3fn")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            "input.shape[1] must be a statically known dimension. Input shape"
            f" received: {input.shape}"
        )

    if not (input_scale_spec.is_tensor and weight_scale_spec.is_tensor):
        raise ValueError(
            "both input and weight must be tensor scaled for tensor scaling"
        )

    if group_size_or_per_token != -1:
        raise ValueError(
            "group_size_or_per_token should be -1 for dynamic tensor scaling so"
            " group_size == num_cols == input.shape[1]"
        )

    result = ops.custom(
        "mo.quantize_tensor_dynamic_scaled_float8",
        device=input.device,
        values=[
            input,
            ops.constant(scale_ub, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], input.shape[1]],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[1, input.shape[0]],
                device=input.device,
            ),
        ],
        parameters={
            "group_size_or_per_token": group_size_or_per_token,
        },
    )

    return result[0].tensor, result[1].tensor


def quantize_dynamic_scaled_float8(
    input: TensorValue,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    scale_ub: float = 1200.0,
    group_size_or_per_token: int = -1,
    out_type: DType = DType.float8_e4m3fn,
    scales_type: DType = DType.bfloat16,
) -> tuple[TensorValue, TensorValue]:
    """Dynamically quantize the input tensor to fp8.

    Args:
        input: The input tensor to quantize.
        scale_ub: The upper bound of the scale factor.
        group_size_or_per_token: The group size for quantization. When set to -1,
            the quantization is column-wise.
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor and the scales.
    """
    if input.rank != 2:
        raise ValueError("input must be rank 2 tensor")

    if out_type not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError("out_type must be float8_e4m3fn or float8_e4m3fnuz")

    if not isinstance(input.shape[1], StaticDim):
        raise ValueError(
            "input.shape[1] must be a statically known dimension. Input shape"
            f" received: {input.shape}"
        )

    if group_size_or_per_token == -1:
        if input_scale_spec.is_block or weight_scale_spec.is_block:
            assert input_scale_spec.block_size is not None
            group_size = input_scale_spec.block_size[1]
        else:
            group_size = int(input.shape[1])
    else:
        group_size = group_size_or_per_token

    a_scales_dim1 = input.shape[0]
    if input_scale_spec.is_block or weight_scale_spec.is_block:
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise"
                " scaling"
            )

        # For blockwise scaling pad the a_scales to 16 Bytes. This is required by NVIDIA SM90+ TMA instructions
        padding_size = 16 // scales_type.size_in_bytes
        a_scales_dim1 = (
            (input.shape[0] + padding_size - 1) // padding_size
        ) * padding_size

    result = ops.custom(
        "mo.quantize_dynamic_scaled_float8",
        device=input.device,
        values=[
            input,
            ops.constant(scale_ub, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], input.shape[1]],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[input.shape[1] // group_size, a_scales_dim1],
                device=input.device,
            ),
        ],
        parameters={
            "group_size_or_per_token": group_size,
        },
    )

    return result[0].tensor, result[1].tensor


def dynamic_scaled_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    input_scale_spec: InputScaleSpec,
    weight_scale_spec: WeightScaleSpec,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Performs a matmul of two tensors with scaling factors. Currently only
    supports channel-wise scaling for weights and per-token scaling for inputs.

    Args:
        a: The first tensor to multiply.
        b: The second tensor to multiply, must be transposed.
        a_scales: The scaling factors for the first tensor.
        b_scales: The scaling factors for the second tensor.

    Returns:
        The result of the matmul operation.
    """
    if a.rank != 2 or b.rank != 2 or a_scales.rank != 2 or b_scales.rank != 2:
        raise ValueError("All arguments must be rank 2 tensors")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "The second dimension of b must match the second dimension of a"
        )

    if input_scale_spec.is_tensor and weight_scale_spec.is_tensor:
        if input_scale_spec.origin.is_dynamic:
            if not (b_scales.shape[0] == b_scales.shape[1] == 1):
                raise ValueError(
                    "scaler weight tensors must be of shape [1, 1] for dynamic"
                    " tensor scaling"
                )
        else:
            if not (
                a_scales.shape[0]
                == a_scales.shape[1]
                == b_scales.shape[0]
                == b_scales.shape[1]
                == 1
            ):
                raise ValueError(
                    "scaler tensors must be of shape [1, 1] for tensor scaling"
                )

    elif input_scale_spec.is_colwise and weight_scale_spec.is_rowwise:
        if a_scales.shape[0] != 1:
            raise ValueError("only per-token scaling is supported for a")

        if b_scales.shape[1] != 1:
            raise ValueError("only channel-wise scaling is supported for b")

    elif input_scale_spec.is_block or weight_scale_spec.is_block:
        if (
            input_scale_spec.block_size is None
            or weight_scale_spec.block_size is None
        ):
            raise ValueError(
                "both input and weight block size must be set for blockwise"
                " scaling"
            )
        if not (input_scale_spec.is_block and weight_scale_spec.is_block):
            raise ValueError(
                "both input and weight must be blockwise scaled for blockwise"
                " scaling"
            )

        if a_scales.dtype != b_scales.dtype or a_scales.dtype != DType.float32:
            raise TypeError(
                "a_scales and b_scales dtypes must be float32, but got"
                f" {a_scales.dtype}, {b_scales.dtype}"
            )

        # a_scale is of shape [ceildiv(K, BLOCK_SIZE), M-padded]
        # b_scale is of shape [ceildiv(N, BLOCK_SIZE), ceildiv(K, BLOCK_SIZE)]
        if a_scales.shape[0] != b_scales.shape[1]:
            raise ValueError(
                "both a_scales and b_scales must have the same shape on the K"
                f" dimension. got a_scales.shape={a_scales.shape} and"
                f" b_scales.shape={b_scales.shape}"
            )

    else:
        raise ValueError("unsupported FP8 scaling granularity")

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

    result = ops.custom(
        "mo.matmul_dynamic_scaled_fp8",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={
            "input_scale_granularity": str(input_scale_spec.granularity),
            "weight_scale_granularity": str(weight_scale_spec.granularity),
            "m_scale_granularity": -1
            if input_scale_spec.block_size is None
            else input_scale_spec.block_size[0],
            "n_scale_granularity": -1
            if weight_scale_spec.block_size is None
            else weight_scale_spec.block_size[0],
            "k_scale_granularity": -1
            if weight_scale_spec.block_size is None
            else weight_scale_spec.block_size[1],
        },
    )[0].tensor

    return result


def dynamic_block_scaled_matmul(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    tensor_sf: TensorValue | float = 1.0,
    sf_vector_size: int = 16,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Performs a block-scaled matmul of two NVFP4 or MXFP8 tensors.

    Both formats drive the same SM100 tensor-core block-scaled MMA kernel
    (``UMMAKind.KIND_MXF8F6F4``) via the ``mo.matmul.dynamic.block.scaled`` op;
    the format is selected from ``a.dtype``:

    - NVFP4: ``a``/``b`` are ``uint8`` (two ``e2m1`` values packed per byte)
      with ``float8_e4m3fn`` scales and ``sf_vector_size=16``.
    - MXFP8: ``a``/``b`` are ``float8_e4m3fn`` (unpacked) with
      ``float8_e8m0fnu`` scales and ``sf_vector_size=32``.

    Args:
        a: The first tensor to multiply, rank 2 ``[M, K]``.
        b: The second tensor to multiply, rank 2 ``[N, K]`` (transposed).
        a_scales: The rank-5 SF-atom scaling factors for ``a``.
        b_scales: The rank-5 SF-atom scaling factors for ``b``.
        tensor_sf: Buffer-wise scaling factor applied to the output. For NVFP4
            this is ``weight_scale_2 * input_scale`` (non-inverted); MXFP8 uses
            pure block scaling, so it defaults to ``1.0`` (identity).
        sf_vector_size: K-block size for the scaling factors: 16 for NVFP4 or
            32 for MXFP8.
        out_type: The output dtype.

    Returns:
        The result of the matmul operation, shape ``[M, N]``.
    """
    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a_scales.rank != 5 or b_scales.rank != 5:
        raise ValueError("Both a_scales and b_scales must be rank 5 tensors")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "The second dimension of b must match the second dimension of a"
        )

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

    # Select the quantization format from the operand dtype.
    if a.dtype == DType.uint8:
        # NVFP4: two e2m1 values packed per uint8, so the logical K is doubled
        # relative to the stored K when sizing the scales.
        if a_scales.dtype != DType.float8_e4m3fn:
            raise ValueError("a_scales dtype must be float8_e4m3fn for NVFP4")
        if sf_vector_size != 16:
            raise ValueError("sf_vector_size must be 16 for NVFP4")
        k_packing = 2
    elif a.dtype == DType.float8_e4m3fn:
        # MXFP8: data is not packed, so the stored K is used directly.
        if a_scales.dtype != DType.float8_e8m0fnu:
            raise ValueError("a_scales dtype must be float8_e8m0fnu for MXFP8")
        if sf_vector_size != 32:
            raise ValueError("sf_vector_size must be 32 for MXFP8")
        k_packing = 1
    else:
        raise ValueError(
            "a dtype must be uint8 (NVFP4) or float8_e4m3fn (MXFP8), got"
            f" {a.dtype}"
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * sf_vector_size

    # scales tensor shape: [ceildiv(MN, SF_MN_GROUP_SIZE),
    #   ceildiv(K, SF_K_GROUP_SIZE), SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K].
    # For NVFP4 each uint8 packs two values, so the logical K is K * k_packing.
    a_scales_dim_1 = ceildiv(a.shape[1] * k_packing, Dim(SF_K_GROUP_SIZE))
    b_scales_dim_0 = ceildiv(b.shape[0], Dim(SF_MN_GROUP_SIZE))
    b_scales_dim_1 = ceildiv(b.shape[1] * k_packing, Dim(SF_K_GROUP_SIZE))
    scales_dim_2 = SF_ATOM_M[0]
    scales_dim_3 = SF_ATOM_M[1]
    scales_dim_4 = SF_ATOM_K

    if (
        a_scales.shape[1] != a_scales_dim_1
        or a_scales.shape[2] != scales_dim_2
        or a_scales.shape[3] != scales_dim_3
        or a_scales.shape[4] != scales_dim_4
    ):
        raise ValueError(
            "a_scales shape must be"
            f" {a_scales_dim_1, scales_dim_2, scales_dim_3, scales_dim_4}, but"
            f" got {a_scales.shape}"
        )

    if (
        b_scales.shape[0] != b_scales_dim_0
        or b_scales.shape[1] != b_scales_dim_1
        or b_scales.shape[2] != scales_dim_2
        or b_scales.shape[3] != scales_dim_3
        or b_scales.shape[4] != scales_dim_4
    ):
        raise ValueError(
            "b_scales shape must be"
            f" {b_scales_dim_0, b_scales_dim_1, scales_dim_2, scales_dim_3, scales_dim_4},"
            f" but got {b_scales.shape}"
        )

    if a_scales.shape[1] != b_scales.shape[1]:
        raise ValueError(
            "a_scales and b_scales must have the same shape on the K"
            f" dimension. got a_scales.shape={a_scales.shape} and"
            f" b_scales.shape={b_scales.shape}"
        )

    tensor_sf_value: TensorValue
    if isinstance(tensor_sf, float):
        tensor_sf_value = ops.constant(
            tensor_sf, DType.float32, device=DeviceRef.CPU()
        )
    else:
        tensor_sf_value = TensorValue(tensor_sf)

    result = ops.custom(
        "mo.matmul.dynamic.block.scaled",
        device=a.device,
        values=[
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf_value,
        ],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )[0].tensor

    return result


def _apple_weight_only_block_scaled_matmul(
    a: TensorValue,
    b: TensorValue,
    b_scales: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Apple M5 weight-only NVFP4 (W4A16) matmul: ``out = a @ dequant(b).T``.

    The Apple sibling of :func:`dynamic_block_scaled_matmul`. Unlike the NVIDIA
    SM100 path, the activation ``a`` stays in ``bfloat16`` (it is *not*
    dynamically quantized to FP4) and the weight block scales are plain rank-2
    ``[N, K // 16]`` (not the SM100 rank-5 TCGEN05 interleave). The FP4 weight
    is dequantized to bf16 in-register at the MMA loader seam; weights stay
    packed in DRAM.

    The NVFP4 per-tensor ``weight_scale_2`` scalar is *not* an argument here —
    the caller applies it as a post-matmul graph-level multiply.

    Args:
        a: The bf16 activation, shape ``[M, K]``.
        b: The packed FP4 weight, ``uint8`` shape ``[N, K // 2]`` (two ``e2m1``
            nibbles per byte, low nibble first).
        b_scales: The FP8-E4M3 block scales, ``float8_e4m3fn`` shape
            ``[N, K // 16]`` (block size 16 along K).
        out_type: The output dtype (``bfloat16``, ``float16``, or ``float32``).

    Returns:
        The matmul result, shape ``[M, N]``.
    """
    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if b_scales.rank != 2:
        raise ValueError("b_scales must be a rank 2 tensor")
    if a.dtype != DType.bfloat16:
        raise ValueError(f"activation a must be bfloat16, got {a.dtype}")
    if b.dtype != DType.uint8:
        raise ValueError(
            f"weight b must be uint8 (fp4-e2m1fnX2), got {b.dtype}"
        )
    if b_scales.dtype != DType.float8_e4m3fn:
        raise ValueError(
            f"b_scales must be float8_e4m3fn, got {b_scales.dtype}"
        )

    result = ops.custom(
        "mo.matmul.weight.only.block.scaled.apple",
        device=a.device,
        values=[a, b, b_scales],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
    )[0].tensor

    return result


def _apple_weight_only_scaled_float8_matmul(
    a: TensorValue,
    b: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Apple M5 weight-only FP8 (W8A16) matmul: ``out = a @ dequant(b).T``.

    The FP8 sibling of :func:`_apple_weight_only_block_scaled_matmul`. The
    activation ``a`` stays in ``bfloat16`` (it is *not* dynamically quantized to
    FP8) and the FP8-E4M3 weight ``b`` is widened to f32/bf16 at the point of
    consumption; weights stay ``float8_e4m3fn`` in DRAM. Unlike the NVFP4 sibling
    there is no per-block weight scale to pass -- modelopt static FP8 carries one
    per-tensor scalar ``weight_scale``, which the caller applies as a post-matmul
    graph-level multiply (the FP8 analog of NVFP4's ``weight_scale_2``). So this
    op takes neither a scale nor ``input_scale`` (``input_scale`` cancels for a
    bf16 activation).

    Args:
        a: The bf16 activation, shape ``[M, K]``.
        b: The FP8 weight, ``float8_e4m3fn`` shape ``[N, K]`` (``transpose_b``).
        out_type: The output dtype (``bfloat16``, ``float16``, or ``float32``).

    Returns:
        The raw (unscaled) matmul result, shape ``[M, N]``.
    """
    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a.dtype != DType.bfloat16:
        raise ValueError(f"activation a must be bfloat16, got {a.dtype}")
    if b.dtype != DType.float8_e4m3fn:
        raise ValueError(f"weight b must be float8_e4m3fn, got {b.dtype}")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must share the K dimension (a[M,K], b[N,K])")

    result = ops.custom(
        "mo.matmul.weight.only.scaled.float8.apple",
        device=a.device,
        values=[a, b],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
    )[0].tensor

    return result


def _apple_int8_w8a8_matmul(
    a: TensorValue,
    b: TensorValue,
    b_scale: TensorValue,
    bias: TensorValue | None = None,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Apple M5 int8 W8A8 matmul: ``out = dequant(quant(a) @ b^T)``.

    Fused single graph op wrapping ``int8_matmul.mojo``: the bf16 activation
    ``a`` is dynamically quantized to int8 (symmetric per-token absmax/127)
    *inside* the op, matmul'd against the pre-quantized int8 weight ``b`` on the
    int8 widening-MMA datapath (int32 accumulate), then dequantized by the
    per-token activation scale times the per-output-channel weight scale (with
    an optional bias added after dequant). Keeping the activation quant inside
    the op avoids materializing the int8 activation + its scales as separate
    graph values (and the extra per-Linear dispatches that entails).

    Args:
        a: The bf16 activation, shape ``[M, K]``.
        b: The int8 weight, shape ``[N, K]`` (``transpose_b``; RTN-quantized
            per output channel at load).
        b_scale: The fp32 per-output-channel weight scale, shape ``[N, 1]`` or
            ``[N]``.
        bias: Optional bias in ``out_type``, shape ``[N]``; added after dequant.
        out_type: The output dtype (``bfloat16``, ``float16``, or ``float32``).

    Returns:
        The matmul result, shape ``[M, N]``.
    """
    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a.dtype != DType.bfloat16:
        raise ValueError(f"activation a must be bfloat16, got {a.dtype}")
    if b.dtype != DType.int8:
        raise ValueError(f"weight b must be int8, got {b.dtype}")
    if a.shape[1] != b.shape[1]:
        raise ValueError("a and b must share the K dimension (a[M,K], b[N,K])")

    # The kernel narrows the B row stride (K) and the tile-count math (N) to
    # UInt16 (``MmaOpApple``); K or N > 65535 silently overflows and corrupts
    # the output. The in-kernel ``debug_assert`` is a release no-op, so gate it
    # here at graph build (always on). K and N are static for FLUX Linears.
    n_dim, k_dim = b.shape[0], b.shape[1]
    if isinstance(k_dim, StaticDim) and int(k_dim) > 65535:
        raise ValueError(
            "Apple int8 W8A8 matmul: K (b.shape[1]) must be <= 65535 "
            f"(UInt16 stride limit), got {int(k_dim)}"
        )
    if isinstance(n_dim, StaticDim) and int(n_dim) > 65535:
        raise ValueError(
            "Apple int8 W8A8 matmul: N (b.shape[0]) must be <= 65535 "
            f"(UInt16 stride limit), got {int(n_dim)}"
        )

    # The kernel reads a flat per-channel scale ``[N]``; accept the rowwise
    # ``[N, 1]`` weight-scale layout and squeeze it.
    b_scale = b_scale.to(a.device)
    if b_scale.rank == 2:
        b_scale = ops.reshape(b_scale, [b_scale.shape[0]])
    if b_scale.dtype != DType.float32:
        b_scale = ops.cast(b_scale, DType.float32)

    # Custom-op input arity is fixed per registration, so the optional bias is
    # a separate op name (`.bias`), the same idiom as the FP8 fused-QKV op --
    # not a comptime flag on one op.
    op_name = "mo.matmul.int8.w8a8.apple"
    values = [a, b, b_scale]
    if bias is not None:
        op_name += ".bias"
        values.append(ops.cast(bias.to(a.device), out_type))

    result = ops.custom(
        op_name,
        device=a.device,
        values=values,
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
    )[0].tensor

    return result


def dynamic_block_scaled_matmul_amd(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Performs a matmul of two FP4 tensors with 1D-block scaled scaling factors.

    Args:
        a: The first tensor to multiply.
        b: The second tensor to multiply, must be transposed.
        a_scales: The scaling factors for the first tensor.
        b_scales: The scaling factors for the second tensor.

    Returns:
        The result of the matmul operation.
    """
    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a_scales.rank != 2 or b_scales.rank != 2:
        raise ValueError("Both a_scales and b_scales must be rank 2 tensors")

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "The second dimension of b must match the second dimension of a"
        )

    if (a.dtype != b.dtype) or (a_scales.dtype != b_scales.dtype):
        raise TypeError(
            f"a and b dtypes {a.dtype}, {b.dtype} must match, "
            f"as do a and b scales dtypes {a_scales.dtype}, {b_scales.dtype}"
        )

    # The kernel reads raw bytes and takes the packing as a parameter.
    if a.dtype not in (DType.uint8, DType.float8_e4m3fn):
        raise ValueError(
            "A dtype must be uint8 (MXFP4) or float8_e4m3fn (MXFP8), got "
            f"{a.dtype}"
        )

    if a_scales.dtype != DType.float8_e8m0fnu:
        raise ValueError("a_scales dtype must be float8_e8m0fnu")

    elems_per_byte = 2 if a.dtype == DType.uint8 else 1

    # Validate the scale extents here rather than letting a mis-sized tensor
    # reach the kernel, where it reads out of bounds instead of raising.
    MX_SF_VECTOR_SIZE = 32
    expected_a_scales_k = ceildiv(
        a.shape[1] * elems_per_byte, Dim(MX_SF_VECTOR_SIZE)
    )
    if a_scales.shape[1] != expected_a_scales_k:
        raise ValueError(
            f"a_scales shape must be [*, {expected_a_scales_k}] but got "
            f"{a_scales.shape}"
        )
    expected_b_scales_k = ceildiv(
        b.shape[1] * elems_per_byte, Dim(MX_SF_VECTOR_SIZE)
    )
    if b_scales.shape[1] != expected_b_scales_k:
        raise ValueError(
            f"b_scales shape must be [*, {expected_b_scales_k}] but got "
            f"{b_scales.shape}"
        )

    result = ops.custom(
        "mo.matmul.dynamic.block.scaled.amd",
        device=a.device,
        values=[
            a,
            b,
            a_scales,
            b_scales,
        ],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={"lane_bytes": 32 // elems_per_byte},
    )[0].tensor

    return result


_FP6_FORMAT_CODE = {"e2m3": 0, "e3m2": 1}
"""Maps an FP6 element encoding name to the ``FP6_FORMAT`` op parameter."""


def _fp6_format_code(fp6_format: str) -> int:
    """Validates an FP6 encoding name and returns its op parameter value."""
    try:
        return _FP6_FORMAT_CODE[fp6_format]
    except KeyError:
        raise ValueError(
            f"fp6_format must be one of {sorted(_FP6_FORMAT_CODE)}, got "
            f"{fp6_format!r}"
        ) from None


def dynamic_block_scaled_matmul_mxfp6(
    a: TensorValue,
    b: TensorValue,
    a_scales: TensorValue,
    b_scales: TensorValue,
    fp6_format: str = "e2m3",
    out_type: DType = DType.bfloat16,
    preshuffled_b: bool = False,
) -> TensorValue:
    """Performs a matmul of two MXFP6 tensors with E8M0 block scales.

    The FP6 sibling of :func:`dynamic_block_scaled_matmul_amd`. It is a
    separate op rather than another ``lane_bytes`` value because both FP6
    encodings put 24 bytes in a lane, so the byte count cannot choose between
    them -- the encoding travels as its own parameter.

    AMD CDNA4 (gfx950) only.

    Args:
        a: The activations, packed FP6 bytes ``[M, K * 3 // 4]``.
        b: The weights, packed FP6 bytes ``[N, K * 3 // 4]`` (transposed).
        a_scales: E8M0 activation scales ``[M, K // 32]``.
        b_scales: E8M0 weight scales ``[N, K // 32]``.
        fp6_format: The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.
        out_type: The dtype of the result.
        preshuffled_b: When True, ``b`` and ``b_scales`` must already be in
            the plane-split / packed-scale layouts produced by
            ``preshuffle_block_scaled_b_dense`` (a one-time, load-time cost
            for the static weight). Ignored (falls back to the row-major
            kernel) for small ``M`` -- see ``mxfp6_block_scaled_matmul_amd``.

    Returns:
        The result of the matmul operation, ``[M, N]``.
    """
    if not _is_amd_gpu():
        raise ValueError(
            "MXFP6 is supported on AMD CDNA4 (gfx950) only: the kernels issue"
            " through the f8f6f4 MFMA, which has no NVIDIA equivalent. Use"
            " float8_e4m3fn or float4_e2m1fnx2 on NVIDIA."
        )
    fp6_code = _fp6_format_code(fp6_format)

    if a.rank != 2 or b.rank != 2:
        raise ValueError("Both a and b must be rank 2 tensors")
    if a_scales.rank != 2 or b_scales.rank != 2:
        raise ValueError("Both a_scales and b_scales must be rank 2 tensors")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "MXFP6 matmul operands disagree on packed K: "
            f"a={list(a.shape)} b={list(b.shape)} "
            f"a_scales={list(a_scales.shape)} b_scales={list(b_scales.shape)}. "
            "Both operands are byte-packed (four 6-bit codes per three "
            "bytes), so a logical K of n is n * 3 // 4 columns; a mismatch "
            "usually means one side was sized or sharded on the logical width."
        )
    if a.dtype != DType.uint8 or b.dtype != DType.uint8:
        raise ValueError(
            "MXFP6 operands are packed into uint8 (four codes per three "
            f"bytes), got a={a.dtype}, b={b.dtype}"
        )
    if (
        a_scales.dtype != DType.float8_e8m0fnu
        or b_scales.dtype != DType.float8_e8m0fnu
    ):
        raise ValueError("a_scales and b_scales dtypes must be float8_e8m0fnu")

    MX_SF_VECTOR_SIZE = 32
    expected_scales_k = ceildiv(a.shape[1] * 4 // 3, Dim(MX_SF_VECTOR_SIZE))
    if a_scales.shape[1] != expected_scales_k:
        raise ValueError(
            f"a_scales shape must be [*, {expected_scales_k}] but got "
            f"{a_scales.shape}"
        )
    if b_scales.shape[1] != expected_scales_k:
        raise ValueError(
            f"b_scales shape must be [*, {expected_scales_k}] but got "
            f"{b_scales.shape}"
        )

    return ops.custom(
        "mo.matmul.dynamic.block.scaled.mxfp6",
        device=a.device,
        values=[a, b, a_scales, b_scales],
        out_types=[
            TensorType(
                dtype=out_type, shape=[a.shape[0], b.shape[0]], device=a.device
            )
        ],
        parameters={"FP6_FORMAT": fp6_code, "preshuffled_b": preshuffled_b},
    )[0].tensor


def quantize_dynamic_block_scaled_mxfp6(
    input: TensorValue,
    fp6_format: str = "e2m3",
    scales_type: DType = DType.float8_e8m0fnu,
    out_type: DType = DType.uint8,
) -> tuple[TensorValue, TensorValue]:
    """Dynamically quantizes the input tensor to MXFP6.

    Args:
        input: The tensor to quantize, ``[seq_len, hidden_size]`` bf16.
        fp6_format: The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.
        scales_type: The dtype of the scales tensor.
        out_type: The dtype of the packed output.

    Returns:
        The packed tensor in ``[seq_len, hidden_size * 3 // 4]`` and the scales
        in ``[seq_len, hidden_size // 32]``.
    """
    if not _is_amd_gpu():
        raise ValueError(
            "MXFP6 is supported on AMD CDNA4 (gfx950) only: the kernels issue"
            " through the f8f6f4 MFMA, which has no NVIDIA equivalent. Use"
            " float8_e4m3fn or float4_e2m1fnx2 on NVIDIA."
        )
    fp6_code = _fp6_format_code(fp6_format)

    if input.rank != 2:
        raise ValueError("input tensor must be rank 2 tensor")
    if input.dtype != DType.bfloat16:
        raise ValueError("input tensor dtype must be bfloat16")
    if out_type != DType.uint8:
        raise ValueError("out_type must be uint8 (packed FP6)")
    if scales_type != DType.float8_e8m0fnu:
        raise ValueError("scales_type must be float8_e8m0fnu for MXFP6")

    MX_SF_VECTOR_SIZE = 32
    if int(input.shape[1]) % MX_SF_VECTOR_SIZE != 0:
        raise ValueError(
            "input.shape[1] must be a multiple of the 32-element MX block"
        )

    result = ops.custom(
        "mo.quantize.dynamic.block.scaled.mxfp6",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], input.shape[1] * 3 // 4],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[
                    input.shape[0],
                    ceildiv(input.shape[1], Dim(MX_SF_VECTOR_SIZE)),
                ],
                device=input.device,
            ),
        ],
        parameters={"FP6_FORMAT": fp6_code},
    )

    return result[0].tensor, result[1].tensor


def mxfp4_dequant(
    packed_weights: TensorValue,
    scales: TensorValue,
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Dequantizes MXFP4 packed weights to BF16 or FP8 on GPU.

    Supports rank 2 ``[N, K//2]`` and rank 3 ``[E, N, K//2]`` inputs.
    For rank 3, leading dims are flattened to 2D, dequantized, and reshaped back.

    Args:
        packed_weights: Packed weights in uint8 (2 FP4 values per byte).
            Shape ``[N, K//2]`` or ``[E, N, K//2]``.
        scales: Block scales in float8_e8m0fnu.
            Shape ``[N, K//32]`` or ``[E, N, K//32]``.
        out_type: Output dtype (bfloat16 or float8_e4m3fn).

    Returns:
        Dequantized tensor ``[N, K]`` or ``[E, N, K]`` in out_type.
    """
    if packed_weights.rank not in (2, 3):
        raise ValueError(
            f"packed_weights must be rank 2 or 3, got {packed_weights.rank}"
        )
    if scales.rank != packed_weights.rank:
        raise ValueError(
            f"scales rank ({scales.rank}) must match packed_weights rank"
            f" ({packed_weights.rank})"
        )
    if packed_weights.dtype != DType.uint8:
        raise ValueError(
            f"packed_weights must be uint8, got {packed_weights.dtype}"
        )

    # Flatten leading dims if rank 3
    is_batched_weights = packed_weights.rank == 3
    if is_batched_weights:
        e = packed_weights.shape[0]
        n = packed_weights.shape[1]
        k_packed = packed_weights.shape[2]
        packed_weights = ops.reshape(packed_weights, [e * n, k_packed])
        scales = ops.reshape(scales, [e * n, scales.shape[2]])

    rows = packed_weights.shape[0]
    k = packed_weights.shape[1] * 2  # Unpacked column count

    result = ops.custom(
        "mo.dequant.mxfp4",
        device=packed_weights.device,
        values=[packed_weights, scales],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[rows, k],
                device=packed_weights.device,
            )
        ],
    )[0].tensor

    # Reshape back if originally rank 3
    if is_batched_weights:
        result = ops.reshape(result, [e, n, k])

    return result


def mxfp6_dequant(
    packed_weights: TensorValue,
    scales: TensorValue,
    fp6_format: str = "e2m3",
    out_type: DType = DType.bfloat16,
) -> TensorValue:
    """Dequantizes MXFP6 packed weights to BF16 or FP8 on GPU.

    The FP6 sibling of :func:`mxfp4_dequant`. Supports rank 2
    ``[N, K * 3 // 4]`` and rank 3 ``[E, N, K * 3 // 4]`` inputs; for rank 3
    the leading dims are flattened, dequantized, and reshaped back.

    Args:
        packed_weights: Packed FP6 bytes, four codes per three bytes.
        scales: Block scales in ``float8_e8m0fnu``, ``[..., N, K // 32]``.
        fp6_format: The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.
        out_type: Output dtype (``bfloat16`` or ``float8_e4m3fn``).

    Returns:
        The dequantized tensor, ``[N, K]`` or ``[E, N, K]``.
    """
    if not _is_amd_gpu():
        raise ValueError(
            "MXFP6 is supported on AMD CDNA4 (gfx950) only: the kernels issue"
            " through the f8f6f4 MFMA, which has no NVIDIA equivalent. Use"
            " float8_e4m3fn or float4_e2m1fnx2 on NVIDIA."
        )
    fp6_code = _fp6_format_code(fp6_format)

    if packed_weights.rank not in (2, 3):
        raise ValueError(
            f"packed_weights must be rank 2 or 3, got {packed_weights.rank}"
        )
    if scales.rank != packed_weights.rank:
        raise ValueError(
            f"scales rank ({scales.rank}) must match packed_weights rank"
            f" ({packed_weights.rank})"
        )
    if packed_weights.dtype != DType.uint8:
        raise ValueError(
            f"packed_weights must be uint8, got {packed_weights.dtype}"
        )

    is_batched_weights = packed_weights.rank == 3
    if is_batched_weights:
        e = packed_weights.shape[0]
        n = packed_weights.shape[1]
        k_packed = packed_weights.shape[2]
        packed_weights = ops.reshape(packed_weights, [e * n, k_packed])
        scales = ops.reshape(scales, [e * n, scales.shape[2]])

    rows = packed_weights.shape[0]
    k = packed_weights.shape[1] * 4 // 3  # Unpacked column count

    result = ops.custom(
        "mo.dequant.mxfp6",
        device=packed_weights.device,
        values=[packed_weights, scales],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[rows, k],
                device=packed_weights.device,
            )
        ],
        parameters={"FP6_FORMAT": fp6_code},
    )[0].tensor

    if is_batched_weights:
        result = ops.reshape(result, [e, n, k])

    return result


def _is_sm10x_gpu() -> bool:
    """Checks if the current accelerator is NVIDIA SM100+ (Blackwell)."""
    try:
        return accelerator_architecture_name().startswith("sm_10")
    except Exception:
        return False


def _is_sm12x_gpu() -> bool:
    """Checks if the current accelerator is NVIDIA SM120/SM121 (consumer Blackwell)."""
    try:
        return accelerator_architecture_name().startswith("sm_12")
    except Exception:
        return False


def _is_apple_gpu() -> bool:
    """Checks if the current accelerator is an Apple (Metal) GPU."""
    try:
        return accelerator_api() == "metal"
    except Exception:
        return False


def _is_amd_gpu() -> bool:
    """Checks if the current accelerator is an AMD (HIP) GPU."""
    try:
        return accelerator_api() == "hip"
    except Exception:
        return False


def quantize_dynamic_block_scaled(
    input: TensorValue,
    tensor_sf: TensorValue | float = 1.0,
    sf_vector_size: int = 16,
    scales_type: DType = DType.float8_e4m3fn,
    out_type: DType = DType.uint8,  # fp4-e2m1fnX2
) -> tuple[TensorValue, TensorValue]:
    """Dynamically quantize a bf16 tensor to NVFP4 or MXFP8 with block scales.

    Both formats go through the ``mo.quantize.dynamic.block.scaled`` op; the
    format is selected from ``out_type``:

    - NVFP4 (``out_type=uint8``): two ``e2m1`` values packed per output byte,
      with ``float8_e4m3fn`` (NVFP4) or ``float8_e8m0fnu`` (MXFP4) scales and
      ``sf_vector_size`` 16 or 32.
    - MXFP8 (``out_type=float8_e4m3fn``): unpacked ``float8_e4m3fn`` data with
      ``float8_e8m0fnu`` scales and ``sf_vector_size=32``. MXFP8 uses pure
      block scaling, so ``tensor_sf`` defaults to ``1.0`` (identity).

    Args:
        input: The input tensor to quantize. Shape: [seq_len, hidden_size].
        tensor_sf: The tensor-wise scale factor (inverted as per the
            quantization kernel requirement for NVFP4; identity for MXFP8).
        sf_vector_size: The block size for the scaling factors.
            16 for NVFP4, 32 for MXFP4/MXFP8.
        scales_type: The type of the scales tensor.
            ``float8_e4m3fn`` for NVFP4, ``float8_e8m0fnu`` for MXFP4/MXFP8.
        out_type: The type of the output tensor. ``uint8`` for packed FP4 or
            ``float8_e4m3fn`` for MXFP8.

    Returns:
        The quantized tensor and scales. Scales layout depends on hardware:
        rank-5 interleaved on NVIDIA SM100, rank-2 ``[M, K // sf_vector_size]``
        otherwise.
    """
    if input.rank != 2:
        raise ValueError("input tensor must be rank 2 tensor")

    if input.dtype != DType.bfloat16:
        raise ValueError("input tensor dtype must be bfloat16")

    # Select the quantization format from the output dtype.
    if out_type == DType.uint8:
        # NVFP4 / MXFP4: two e2m1 values packed per output byte.
        if scales_type not in (DType.float8_e4m3fn, DType.float8_e8m0fnu):
            raise ValueError(
                "scales_type must be float8_e4m3fn (NVFP4) or float8_e8m0fnu"
                " (MXFP4)"
            )
        if sf_vector_size not in (16, 32):
            raise ValueError("sf_vector_size must be 16 (NVFP4) or 32 (MXFP4)")
        is_packed = True
    elif out_type == DType.float8_e4m3fn:
        # MXFP8: float8 data, not packed.
        if scales_type != DType.float8_e8m0fnu:
            raise ValueError("scales_type must be float8_e8m0fnu for MXFP8")
        if sf_vector_size != 32:
            raise ValueError("sf_vector_size must be 32 for MXFP8")
        is_packed = False
    else:
        raise ValueError(
            "out_type must be uint8 (FP4) or float8_e4m3fn (MXFP8), got"
            f" {out_type}"
        )

    # MXFP4/MXFP8 (sf_vector_size=32) requires K % 32 because the kernel's
    # 4-thread cooperative scale reduction operates on 32-element groups.
    # NVFP4 (sf_vector_size=16) only requires K % 8.
    k_alignment = (
        sf_vector_size if sf_vector_size == 32 else sf_vector_size // 2
    )
    if int(input.shape[1]) % k_alignment != 0:
        raise ValueError(f"input.shape[1] must be a multiple of {k_alignment}")

    if _is_sm10x_gpu() or _is_sm12x_gpu():
        # SM100 / SM120 TCGEN05: rank-5 interleaved scales layout.
        SF_ATOM_M = [32, 4]
        SF_ATOM_K = 4
        SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
        SF_K_GROUP_SIZE = SF_ATOM_K * sf_vector_size
        scales_shape: list[Dim | int] = [
            ceildiv(input.shape[0], Dim(SF_MN_GROUP_SIZE)),
            ceildiv(input.shape[1], Dim(SF_K_GROUP_SIZE)),
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        ]
    else:
        # Default: rank-2 scales [M, K // sf_vector_size].
        # TODO: 2D is a proxy for CDNA4. The optimized layout is likely
        # 6D (32x32 tiles) or 7D (16x16 tiles).
        scales_shape = [
            input.shape[0],
            ceildiv(input.shape[1], Dim(sf_vector_size)),
        ]

    # FP4 packs two values per output byte; MXFP8 stores one value per element.
    quantized_k = input.shape[1] // 2 if is_packed else input.shape[1]

    tensor_sf_value: TensorValue
    if isinstance(tensor_sf, float):
        tensor_sf_value = ops.constant(
            tensor_sf, DType.float32, device=DeviceRef.CPU()
        )
    else:
        tensor_sf_value = TensorValue(tensor_sf)

    result = ops.custom(
        "mo.quantize.dynamic.block.scaled",
        device=input.device,
        values=[input, tensor_sf_value],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[input.shape[0], quantized_k],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=scales_shape,
                device=input.device,
            ),
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )

    return result[0].tensor, result[1].tensor


def grouped_quantize_dynamic_block_scaled(
    input: TensorValue,
    row_offsets: TensorValue,
    scales_offsets: TensorValue,
    expert_ids: TensorValue,
    sf_tensor: TensorValue,
    sf_vector_size: int = 16,
    scales_type: DType = DType.float8_e4m3fn,
    out_type: DType = DType.uint8,
) -> tuple[TensorValue, TensorValue]:
    """Grouped dynamic NVFP4/MXFP4/MXFP8 quantization for MoE experts.

    Quantizes a concatenated token tensor where different row ranges belong
    to different experts, each with its own tensor-wise scale factor.

    Args:
        input: The concatenated input tensor. Shape: ``[total_tokens, K]``,
            dtype ``bfloat16``.
        row_offsets: Cumulative token offsets per expert.
            Shape: ``[num_experts + 1]``, dtype ``uint32``.
        scales_offsets: Per-expert scale tile offset corrections.
            Shape: ``[num_experts]``, dtype ``uint32``.
        expert_ids: Expert ID mapping (typically identity).
            Shape: ``[num_experts]``, dtype ``int32``.
        sf_tensor: Per-expert tensor-wise scale factors.
            Shape: ``[num_experts]``, dtype ``float32``.
        sf_vector_size: The block size for the scaling factors.
        scales_type: Scale factor dtype. ``float8_e4m3fn`` for NVFP4.
        out_type: Output dtype. ``uint8`` for packed FP4.

    Returns:
        The quantized tensor ``[total_tokens, K // 2]`` and scales in
        rank-5 interleaved layout
        ``[total_m_tiles, K_tiles, 32, 4, 4]``.
    """
    if input.rank != 2:
        raise ValueError("input tensor must be rank 2")

    if input.dtype != DType.bfloat16:
        raise ValueError("input tensor dtype must be bfloat16")

    if out_type not in (DType.uint8, DType.float8_e4m3fn):
        raise ValueError("out_type must be uint8 (FP4x2) or float8_e4m3fn")

    if not _is_sm10x_gpu():
        # route to the fallback kernel
        return quantize_dynamic_block_scaled(
            input, sf_tensor[0], sf_vector_size, scales_type, out_type
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    SF_K_GROUP_SIZE = SF_ATOM_K * sf_vector_size

    total_m_tiles = ceildiv(input.shape[0], Dim(SF_MN_GROUP_SIZE))
    # A row belongs to exactly one group, so the number of non-empty groups
    # is bounded by both the group count and the row count; padding one tile
    # per group beyond that bound is wasted work at low occupancy (e.g.
    # single-token decode routed across hundreds of experts).
    total_m_tiles += AlgebraicDim.apply(
        kgen.POC.min, expert_ids.shape[0], input.shape[0]
    )
    scales_shape: list[Dim | int] = [
        total_m_tiles,
        ceildiv(input.shape[1], Dim(SF_K_GROUP_SIZE)),
        SF_ATOM_M[0],
        SF_ATOM_M[1],
        SF_ATOM_K,
    ]

    result = ops.custom(
        "mo.grouped.quantize.dynamic.block.scaled",
        device=input.device,
        values=[input, row_offsets, scales_offsets, expert_ids, sf_tensor],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[
                    input.shape[0],
                    input.shape[1]
                    if out_type == DType.float8_e4m3fn
                    else input.shape[1] // 2,
                ],
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=scales_shape,
                device=input.device,
            ),
        ],
    )

    return result[0].tensor, result[1].tensor


def quantize_dynamic_block_scaled_mxfp4(
    input: TensorValue,
    scales_type: DType = DType.float8_e8m0fnu,
    out_type: DType = DType.uint8,  # fp4-e2m1fnX2
) -> tuple[TensorValue, TensorValue]:
    """Dynamically quantize the input tensor to fp4-e2m1fn.

    Args:
        input: The input tensor to quantize. Shape: [seq_len, hidden_size]
        out_type: The type of the output tensor.
        scales_type: The type of the scales tensor.

    Returns:
        The quantized tensor in [seq_len, hidden_size // 2] layout and the scales in
        [seq_len, hidden_size // 32] layout.
    """
    if input.rank != 2:
        raise ValueError("input tensor must be rank 2 tensor")

    if input.dtype != DType.bfloat16:
        raise ValueError("input tensor dtype must be bfloat16")

    if out_type != DType.uint8:
        raise ValueError("out_type must be uint8 (fp4-e2m1fnX2)")

    if scales_type != DType.float8_e8m0fnu:
        raise ValueError("scales_type must be float8_e8m0fnu for MXFP4")

    MXFP4_SF_VECTOR_SIZE = 32

    if int(input.shape[1]) % MXFP4_SF_VECTOR_SIZE != 0:
        raise ValueError(
            "input.shape[1] must be a multiple of MXFP4_SF_VECTOR_SIZE"
        )

    result = ops.custom(
        "mo.quantize.dynamic.block.scaled.mxfp4",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(
                dtype=out_type,
                shape=[
                    input.shape[0],
                    input.shape[1] // 2,
                ],  # each output element (uint8) is 2 fp4-e2m1fn values
                device=input.device,
            ),
            TensorType(
                dtype=scales_type,
                shape=[
                    input.shape[0],
                    ceildiv(input.shape[1], Dim(MXFP4_SF_VECTOR_SIZE)),
                ],
                device=input.device,
            ),
        ],
    )

    return result[0].tensor, result[1].tensor


def block_scales_interleave(
    scales: TensorValue,
    sf_vector_size: int = 16,
) -> TensorValue:
    """Interleaves rank-2 FP4 block scales into the rank-5 TCGEN layout.

    Args:
        scales: Rank-2 block scales in ``[M, K // sf_vector_size]`` layout.
            Supported dtypes are ``float8_e4m3fn`` for NVFP4 and
            ``float8_e8m0fnu`` for MXFP4.
        sf_vector_size: Scale-factor vector size: 16 for NVFP4 or 32 for MXFP4.

    Returns:
        The interleaved scales tensor in
        ``[ceildiv(M, 128), ceildiv(K // sf_vector_size, 4), 32, 4, 4]`` layout.
    """
    if scales.rank != 2:
        raise ValueError("scales must be a rank 2 tensor")

    if scales.dtype not in (DType.float8_e4m3fn, DType.float8_e8m0fnu):
        raise ValueError(
            "scales dtype must be float8_e4m3fn (NVFP4) or float8_e8m0fnu"
            " (MXFP4)"
        )

    expected_sf_vector_size = 32 if scales.dtype == DType.float8_e8m0fnu else 16
    if sf_vector_size != expected_sf_vector_size:
        raise ValueError(
            "sf_vector_size must match scales dtype:"
            " 16 for float8_e4m3fn (NVFP4),"
            " 32 for float8_e8m0fnu (MXFP4)"
        )

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128

    # Interleaved scales shape:
    # [ceildiv(M, 128), ceildiv(num_scale_cols, 4), 32, 4, 4].
    scales_dim_0 = ceildiv(scales.shape[0], Dim(SF_MN_GROUP_SIZE))
    scales_dim_1 = ceildiv(scales.shape[1], Dim(SF_ATOM_K))
    scales_dim_2 = SF_ATOM_M[0]
    scales_dim_3 = SF_ATOM_M[1]
    scales_dim_4 = SF_ATOM_K

    result = ops.custom(
        "mo.interleave.block.scales",
        device=scales.device,
        values=[scales],
        out_types=[
            TensorType(
                dtype=scales.dtype,
                shape=[
                    scales_dim_0,
                    scales_dim_1,
                    scales_dim_2,
                    scales_dim_3,
                    scales_dim_4,
                ],
                device=scales.device,
            ),
        ],
        parameters={
            "SF_VECTOR_SIZE": sf_vector_size,
        },
    )[0].tensor

    return result


def block_scaled_preshuffle_grouped_scale_4d(
    a_scales: TensorValue,
    expert_start_indices: TensorValue,
    max_num_tokens_per_expert: TensorValue,
    num_active_experts: TensorValue,
    num_experts: int,
) -> TensorValue:
    """Applies the per-step A-scale preshuffle for the AMD CDNA4 preb kernel.

    Takes row-major E8M0 A-scales ``[total_tokens, K_SCALES]`` and writes
    cell-packed scales into per-expert fixed-stride slots of stride
    ``align_up(max_num_tokens_per_expert, 32)``. Output slot ``e`` holds
    expert slot ``e``'s scales; the preb matmul reads from
    ``e * max_padded_M`` directly.

    Intended to be inserted before ``block_scaled_grouped_matmul_amd_preb`` when
    ``preshuffled_b=True`` so the matmul sees the cell layout it expects.

    Args:
        a_scales: Rank-2 ``float8_e8m0fnu`` tensor ``[total_tokens, K_SCALES]``
            from ``quantize_dynamic_block_scaled_mxfp4``. ``K_SCALES`` must
            be a multiple of 8.
        expert_start_indices: Rank-1 ``uint32`` cumulative token offsets,
            length ``num_active_experts + 1``.
        max_num_tokens_per_expert: Scalar ``uint32`` upper bound on
            per-expert token count this step.
        num_active_experts: Scalar ``uint32`` number of active expert slots.
        num_experts: Graph-build-time upper bound on ``num_active_experts``
            (e.g. ``weight.shape[0]``). Used to size the output buffer.

    Returns:
        Rank-2 ``float8_e8m0fnu`` tensor ``[num_experts * total_tokens,
        K_SCALES]``. The first ``num_active_experts * max_padded_M`` rows
        are written; the rest is left untouched but accessible.
    """
    if a_scales.rank != 2:
        raise ValueError(
            f"a_scales must be rank 2 [total_tokens, K_SCALES], got rank"
            f" {a_scales.rank}"
        )
    if a_scales.dtype != DType.float8_e8m0fnu:
        raise ValueError(
            f"a_scales must be float8_e8m0fnu, got {a_scales.dtype}"
        )
    if expert_start_indices.rank != 1:
        raise ValueError(
            "expert_start_indices must be rank 1, got rank"
            f" {expert_start_indices.rank}"
        )

    out_rows = num_experts * a_scales.shape[0]

    return ops.custom(
        "mo.block.scaled.preshuffle.scale.4d_per_expert",
        device=a_scales.device,
        values=[
            a_scales,
            expert_start_indices,
            max_num_tokens_per_expert,
            num_active_experts,
        ],
        out_types=[
            TensorType(
                dtype=DType.float8_e8m0fnu,
                shape=[out_rows, a_scales.shape[1]],
                device=a_scales.device,
            ),
        ],
    )[0].tensor


def block_scaled_preshuffle_b_5d(b: TensorValue) -> TensorValue:
    """Applies the AMD CDNA4 MXFP4 B 5D preshuffle to a rank-3 weight.

    Reorders the packed-FP4 bytes from ``[E, N, K_BYTES]`` row-major into the
    5D ``(E, N0, K0, KLane=4, NLane=16, KPack=16)`` byte layout expected by
    the ``block_scaled_grouped_matmul_amd_preb`` reader. Output is byte-identical to
    ``Shuffler[E].preshuffle_b_5d`` running on the same input.

    Intended for eager invocation from weight adapters (one-shot graph), not
    inside the main forward graph — the preb matmul kernel reads weights
    that are already in this layout.

    Args:
        b: Rank-3 ``uint8`` tensor ``[E, N, K_BYTES]`` of packed FP4 weights.
            ``N`` must be a multiple of 16 and ``K_BYTES`` a multiple of 64.

    Returns:
        Rank-3 ``uint8`` tensor with the same shape and total byte count as
        ``b``, with bytes reordered to the 5D layout.
    """
    if b.rank != 3:
        raise ValueError("b must be a rank 3 tensor [E, N, K_BYTES]")
    if b.dtype != DType.uint8:
        raise ValueError(f"b must be uint8 (packed MXFP4), got {b.dtype}")

    return ops.custom(
        "mo.block.scaled.preshuffle.b.5d",
        device=b.device,
        values=[b],
        out_types=[
            TensorType(
                dtype=DType.uint8,
                shape=b.shape,
                device=b.device,
            ),
        ],
    )[0].tensor


def matmul_static_scaled_float8(
    input: TensorValue,
    weight: TensorValue,
    input_scale: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Performs a static-scaled float8 matrix multiplication.

    Computes ``input @ weight.T`` where both tensors are float8, dequantized
    using the provided per-tensor CPU scalar scales before accumulation.
    The output is always ``bfloat16``.

    Args:
        input: Input tensor of rank 2 and dtype ``float8_e4m3fn`` or
            ``float8_e4m3fnuz``.
        weight: Weight tensor of rank 2 and matching float8 dtype, laid out
            so that the K dimension matches ``input.shape[1]``.
        input_scale: Scalar scale factor for ``input`` (shape ``[]`` or
            ``[1]``), must reside on CPU.
        weight_scale: Scalar scale factor for ``weight`` (shape ``[]`` or
            ``[1]``), must reside on CPU.

    Returns:
        A :class:`~max.graph.TensorValue` of shape
        ``[input.shape[0], weight.shape[0]]`` and dtype ``bfloat16``.

    Raises:
        ValueError: If scale shapes are not scalar, input or weight are not
            rank 2, K dimensions do not match, or scales are not on CPU.
    """
    if input_scale.shape not in [[], [1]]:
        raise ValueError(
            "expected input_scale to be a scalar, but got shape of"
            f" {input_scale.shape}"
        )
    if weight_scale.shape not in [[], [1]]:
        raise ValueError(
            "expected weight_scale to be a scalar, but got shape of"
            f" {weight_scale.shape}"
        )

    if input.dtype not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError(
            "expected input dtype to be float8_e4m3fn or float8_e4m3fnuz, but"
            f" got {input.dtype}"
        )
    if weight.dtype not in (DType.float8_e4m3fn, DType.float8_e4m3fnuz):
        raise ValueError(
            "expected weight dtype to be float8_e4m3fn or float8_e4m3fnuz, but"
            f" got {weight.dtype}"
        )

    if input.rank != 2:
        raise ValueError(f"expected input rank to be 2, but got {input.rank}")
    if weight.rank != 2:
        raise ValueError(f"expected weight rank to be 2, but got {weight.rank}")

    if input.shape[1] != weight.shape[1]:
        raise ValueError("K dimension does not match for matmul")

    if input_scale.device != DeviceRef.CPU():
        raise ValueError(
            f"expected input_scale to be on CPU, but got {input_scale.device}"
        )

    if weight_scale.device != DeviceRef.CPU():
        raise ValueError(
            f"expected weight_scale to be on CPU, but got {weight_scale.device}"
        )

    return ops.custom(
        "mo.matmul_static_scaled_float8",
        device=input.device,
        values=[
            input,
            weight,
            input_scale.reshape([]),
            weight_scale.reshape([]),
        ],
        out_types=[
            TensorType(
                dtype=DType.bfloat16,
                shape=[input.shape[0], weight.shape[0]],
                device=input.device,
            )
        ],
    )[0].tensor


def needs_fp8_fnuz_conversion() -> bool:
    """Checks if FP8 E4M3FN to FNUZ conversion is needed for AMD GPUs.

    Returns:
        ``True`` if running on AMD GPU with CDNA3 architecture, ``False`` otherwise.
    """
    try:
        return "gfx94" in accelerator_architecture_name()
    except Exception:
        return False


def normalize_e4m3fn_to_e4m3fnuz(
    weight: TensorValue,
    weight_scale: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Converts E4M3FN weights to E4M3FNUZ format for AMD GPUs.

    This conversion is necessary because AMD GPUs use the E4M3FNUZ format
    while NVIDIA GPUs use E4M3FN. The key differences are:
    1. The bit pattern 10000000 (-128) represents zero in E4M3FN but NaN in E4M3FNUZ
    2. For the same bit representation, E4M3FNUZ values are half of E4M3FN values

    Args:
        weight: The weight tensor in E4M3FN format.
        weight_scale: The weight scale factor.

    Returns:
        Tuple of (converted_weight, adjusted_weight_scale, adjusted_input_scale).
    """
    if weight.dtype != DType.float8_e4m3fn:
        raise ValueError(
            f"Expected weight dtype to be float8_e4m3fn, but got {weight.dtype}"
        )

    # Convert using custom op that takes float8_e4m3fn input and returns float8_e4m3fnuz
    # Then cast back to float8_e4m3fn to maintain dtype compatibility with kernels
    converted_weight_fnuz = ops.custom(
        "mo.convert_e4m3fn_to_e4m3fnuz",
        device=weight.device,
        values=[weight],
        out_types=[
            TensorType(
                dtype=DType.float8_e4m3fnuz,
                shape=weight.shape,
                device=weight.device,
            )
        ],
    )[0].tensor

    # Cast back to float8_e4m3fn to maintain kernel compatibility
    # The bit pattern has been converted, but we need FN dtype for the kernels
    # converted_weight = ops.cast(converted_weight_fnuz, DType.float8_e4m3fn)

    # For the same bits representation, e4m3fnuz value is half of
    # the e4m3fn value, so we should double the scaling factor to
    # get the same dequantized value.
    adjusted_weight_scale = weight_scale * ops.constant(
        2.0, weight_scale.dtype, device=weight_scale.device
    )

    return converted_weight_fnuz, adjusted_weight_scale


def convert_weights_to_fp8_fnuz_if_needed(
    weight: TensorValue,
    weight_scale: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Converts weights and scales to FP8 FNUZ format if needed for AMD GPUs.

    This utility function checks if FP8 FNUZ conversion is needed, currently onli AMD MI300 GPUs,
    and performs the conversion if required. This centralizes the conversion logic
    that was previously duplicated across multiple files.

    Args:
        weight: The weight tensor to potentially convert.
        weight_scale: The weight scale factor.

    Returns:
        Tuple of (weight, weight_scale) - converted if needed, original otherwise.
    """
    if needs_fp8_fnuz_conversion() and weight.dtype == DType.float8_e4m3fn:
        return normalize_e4m3fn_to_e4m3fnuz(weight, weight_scale)
    return weight, weight_scale


def merge_ragged_tensors(
    a: TensorValue,
    a_row_offsets: TensorValue,
    b: TensorValue,
    b_row_offsets: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Merges two ragged tensors into a single ragged tensor.

    Both ragged tensors must have the same batch size (same number of row
    offsets). This function interleaves the rows from each tensor based on
    their row offsets.

    Args:
        a: The first ragged tensor of shape [total_a_rows, ...].
        a_row_offsets: The row offsets of the first ragged tensor,indicating
            where each batch starts and ends in `a`.
        b: The second ragged tensor of shape [total_b_rows, ...].
        b_row_offsets: The row offsets of the second ragged tensor, indicating
            where each batch starts and ends in `b`.

    Returns:
        A tuple of two tensors:
            - The merged ragged tensor with shape
                [total_a_rows + total_b_rows, ...].
            - The merged row offsets with the same shape as input row offsets.

    The following example interleaves two ragged tensors that share a batch
    size of 2. Row ``a = [1, 2, 3, 4, 5, 6]`` with offsets ``[0, 2, 6]`` and
    row ``b = [7, 8, 9, 10]`` with offsets ``[0, 3, 4]`` merge into
    ``[1, 2, 7, 8, 9, 3, 4, 5, 6, 10]`` with offsets ``[0, 5, 10]``:

    .. code-block:: python

        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType
        from max.nn.kernels import merge_ragged_tensors

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        with Graph(
            "merge_ragged_tensors_example",
            input_types=(
                TensorType(DType.int32, ["a_seq_len"], device=device_ref),
                TensorType(DType.uint32, ["offsets_len"], device=device_ref),
                TensorType(DType.int32, ["b_seq_len"], device=device_ref),
                TensorType(DType.uint32, ["offsets_len"], device=device_ref),
            ),
        ) as graph:
            a, a_row_offsets, b, b_row_offsets = (v.tensor for v in graph.inputs)
            merged_tensor, merged_row_offsets = merge_ragged_tensors(
                a, a_row_offsets, b, b_row_offsets
            )
            graph.output(merged_tensor, merged_row_offsets)
    """
    if a.dtype != b.dtype:
        raise ValueError("a and b must have the same dtype")

    if a_row_offsets.shape[0] != b_row_offsets.shape[0]:
        raise ValueError(
            "a_row_offsets and b_row_offsets must have the same shape"
        )

    c_shape = [a.shape[0] + b.shape[0]] + a.shape[1:]

    results = ops.custom(
        "mo.merge_ragged_tensors",
        device=a.device,
        values=[a, a_row_offsets, b, b_row_offsets],
        out_types=[
            TensorType(dtype=a.dtype, shape=c_shape, device=a.device),
            TensorType(
                dtype=DType.uint32, shape=a_row_offsets.shape, device=a.device
            ),
        ],
    )

    return results[0].tensor, results[1].tensor


def mtp_eh_norm(
    embed: TensorValue,
    prev_hidden: TensorValue,
    enorm_weight: TensorValue,
    hnorm_weight: TensorValue,
    epsilon: float,
    block_threads: int = 256,
) -> TensorValue:
    """Normalizes both inputs of an MTP draft layer into one projection input.

    A multi-token-prediction draft predicts a token two ahead from the
    embedding of the token the target just produced and the target's hidden
    state. Each is RMS-normalized with its own weight, and the pair is
    projected back to one hidden width. This returns that projection's input
    in a single pass.

    Normalization matches :func:`~max.graph.ops.rms_norm` in its Llama-style
    configuration -- ``weight_offset=0`` and ``multiply_before_cast=False``.
    Do not use this with the Gemma-style configuration.

    Args:
        embed: Token embeddings ``[num_tokens, hidden_size]``.
        prev_hidden: Target hidden states, same shape and dtype as ``embed``.
        enorm_weight: Embedding norm weight ``[hidden_size]``.
        hnorm_weight: Hidden-state norm weight ``[hidden_size]``.
        epsilon: Added inside the square root.
        block_threads: Threads per block; must be a whole number of warps.

    Returns:
        ``[num_tokens, 2 * hidden_size]`` with the normalized embedding in the
        leading columns and the normalized hidden state in the trailing ones.

    Raises:
        ValueError: If the inputs disagree in shape, dtype or device, if the
            hidden size is not statically known, if the dtype is float8, or if
            ``block_threads`` is not a positive multiple of 32 no greater than
            1024.
    """
    if embed.shape != prev_hidden.shape:
        raise ValueError(
            "embed and prev_hidden must have the same shape, got"
            f" {embed.shape} and {prev_hidden.shape}"
        )
    if embed.dtype != prev_hidden.dtype:
        raise ValueError(
            "embed and prev_hidden must have the same dtype, got"
            f" {embed.dtype} and {prev_hidden.dtype}"
        )
    if embed.device != prev_hidden.device:
        raise ValueError(
            "embed and prev_hidden must be on the same device, got"
            f" {embed.device} and {prev_hidden.device}"
        )
    for name, w in (
        ("enorm_weight", enorm_weight),
        ("hnorm_weight", hnorm_weight),
    ):
        if w.dtype != embed.dtype:
            raise ValueError(
                f"{name} must match embed's dtype {embed.dtype}, got {w.dtype}"
            )
        if w.device != embed.device:
            raise ValueError(
                f"{name} must be on embed's device {embed.device}, got"
                f" {w.device}"
            )
        if w.shape[0] != embed.shape[1]:
            raise ValueError(
                f"{name} length {w.shape[0]} must equal the hidden size"
                f" {embed.shape[1]}"
            )
    if embed.dtype in (DType.float8_e4m3fn, DType.float8_e5m2):
        raise ValueError(
            f"mtp_eh_norm does not support {embed.dtype}: it accumulates in"
            " float32, where ops.rms_norm reduces float8 in float16"
        )
    if block_threads <= 0 or block_threads % 32 != 0:
        raise ValueError(
            "block_threads must be a positive multiple of 32, got"
            f" {block_threads}"
        )
    if block_threads > 1024:
        raise ValueError(
            "block_threads must not exceed 1024, the largest block any"
            f" supported GPU allows, got {block_threads}"
        )

    if not isinstance(embed.shape[1], StaticDim):
        raise ValueError(
            "the hidden size (embed.shape[1]) must be statically known, got"
            f" {embed.shape[1]}"
        )
    hidden_size = int(embed.shape[1])

    return ops.custom(
        "mo.mtp.eh_norm",
        device=embed.device,
        values=[
            embed,
            prev_hidden,
            enorm_weight,
            hnorm_weight,
            ops.constant(epsilon, dtype=DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=embed.dtype,
                shape=(embed.shape[0], 2 * hidden_size),
                device=embed.device,
            )
        ],
        parameters={
            "hidden_size": hidden_size,
            "block_threads": block_threads,
        },
    )[0].tensor


def eagle_prefill_shift_tokens(
    tokens: TensorValue,
    offsets: TensorValue,
    shift_next_tokens: TensorValue,
) -> TensorValue:
    """Shifts ragged tokens left by 1 per request, appending bonus tokens.

    Args:
        tokens: Flat ragged token sequence of shape ``[total_seq_len]``,
            dtype int64.
        offsets: Row offsets of shape ``[batch_size + 1]``, dtype uint32.
        shift_next_tokens: One token per request of shape ``[batch_size]``,
            dtype int64, to append after shifting.

    Returns:
        Shifted (or copied) tokens with the same shape as ``tokens``.
    """
    results = ops.custom(
        "mo.eagle_prefill_shift_tokens",
        device=tokens.device,
        values=[tokens, offsets, shift_next_tokens],
        out_types=[
            TensorType(
                dtype=tokens.dtype, shape=tokens.shape, device=tokens.device
            ),
        ],
    )
    return results[0].tensor


def apply_penalties_to_logits(
    logits_buffer: BufferValue,
    frequency_data: TensorValue,
    frequency_offsets: TensorValue,
    *,
    frequency_penalty: TensorValueLike = 0.0,
    presence_penalty: TensorValueLike = 0.0,
    repetition_penalty: TensorValueLike = 1.0,
) -> None:
    """Applies penalties to the logits.

    Args:
        logits_buffer: The buffer to apply penalties to.
        frequency_data: 2d tensor of shape [unique_tokens, 2], where
            the first column indicates the token id and the second column
            indicates the frequency of the token.
        frequency_offsets: 1d tensor of shape [batch_size + 1], indicating
            start of each sequence's data.
        frequency_penalty: The frequency penalty to apply to the model's output.
            A positive value will penalize new tokens based on their frequency
            in the generated text: tokens will receive a penalty proportional
            to the count of appearances.
        presence_penalty: The presence penalty to apply to the model's output
            A positive value will penalize new tokens that have already appeared
            in the generated text at least once by applying a constant penalty.
        repetition_penalty: The repetition penalty to apply to the model's
            output. Values > 1 will penalize new tokens that have already
            appeared in prompt and generated text at least once by dividing the
            logits by the repetition penalty.
    """
    if logits_buffer.rank != 2:
        raise ValueError("logits_buffer must be a 2d buffer")

    if frequency_data.rank != 2:
        raise ValueError("frequency_data must be a 2d tensor")

    if frequency_offsets.rank != 1:
        raise ValueError("frequency_offsets must be a 1d tensor")

    if isinstance(frequency_penalty, float):
        frequency_penalty_tensor = ops.broadcast_to(
            ops.constant(
                frequency_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        frequency_penalty_tensor = TensorValue(frequency_penalty)
        if frequency_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                "frequency_penalty tensor shape"
                f" {frequency_penalty_tensor.shape} does not match"
                f" logits_buffer shape {logits_buffer.shape}"
            )

    if isinstance(presence_penalty, float):
        presence_penalty_tensor = ops.broadcast_to(
            ops.constant(
                presence_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        presence_penalty_tensor = TensorValue(presence_penalty)
        if presence_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                "presence_penalty tensor shape"
                f" {presence_penalty_tensor.shape} does not match logits_buffer"
                f" shape {logits_buffer.shape}"
            )

    if isinstance(repetition_penalty, float):
        repetition_penalty_tensor = ops.broadcast_to(
            ops.constant(
                repetition_penalty,
                dtype=DType.float32,
                device=logits_buffer.device,
            ),
            [logits_buffer.shape[0]],
        )
    else:
        repetition_penalty_tensor = TensorValue(repetition_penalty)
        if repetition_penalty_tensor.shape[0] != logits_buffer.shape[0]:
            raise ValueError(
                "repetition_penalty tensor shape"
                f" {repetition_penalty_tensor.shape} does not match"
                f" logits_buffer shape {logits_buffer.shape}"
            )

    ops.inplace_custom(
        "sampler.apply_penalties",
        device=logits_buffer.device,
        values=[
            logits_buffer,
            frequency_data,
            frequency_offsets,
            frequency_penalty_tensor,
            presence_penalty_tensor,
            repetition_penalty_tensor,
        ],
    )


def update_frequency_data(
    frequency_data: BufferValue,
    frequency_offsets: TensorValue,
    tokens: TensorValue,
) -> None:
    """Updates the frequency data.

    Args:
        frequency_data: 2d tensor of shape [unique_tokens, 2], where
            the first column indicates the token id and the second column
            indicates the frequency of the token.
        frequency_offsets: 1d tensor of shape [batch_size + 1], indicating
            start of each sequence's data.
        tokens: The tokens to update the frequency data with.
    """
    if frequency_data.rank != 2:
        raise ValueError("frequency_data must be a 2d buffer")

    if frequency_offsets.rank != 1:
        raise ValueError("frequency_offsets must be a 1d tensor")

    if tokens.rank != 1:
        raise ValueError("tokens must be a 1d tensor")

    ops.inplace_custom(
        "sampler.update_frequency_data",
        device=frequency_data.device,
        values=[
            frequency_data,
            frequency_offsets,
            tokens,
        ],
    )


def scatter_set_constant(
    data: BufferValueLike,
    indices: TensorValueLike,
    fill_val: float,
) -> None:
    """Scatters values into a tensor at specified indices."""
    data = BufferValue(data)
    indices = TensorValue(indices)

    if data.rank != 2:
        raise ValueError(
            "scatter_set_constant currently only supports 2d tensors"
        )

    if indices.rank != 2:
        raise ValueError(
            "scatter_set_constant currently only supports 2d indices"
        )

    # Each indices row is a (row, col) coordinate into `data`. The kernel
    # reads indices[i, 1] unconditionally, so a statically-known inner
    # dimension other than 2 would read out of bounds.
    coords_per_index = indices.shape[1]
    if isinstance(coords_per_index, StaticDim) and int(coords_per_index) != 2:
        raise ValueError(
            "scatter_set_constant indices must have shape [num_indices, 2] "
            f"of (row, col) coordinates, got inner dimension {coords_per_index}"
        )

    ops.inplace_custom(
        "mo.scatter_set_constant",
        device=data.device,
        values=[
            data,
            indices,
            ops.constant(fill_val, data.dtype, device=DeviceRef.CPU()),
        ],
    )


def apply_packed_bitmask(
    logits: TensorValueLike,
    packed: TensorValueLike,
    fill_val: float = -10000.0,
) -> TensorValue:
    """Masks logits with a packed-int32 grammar bitmask in one fused GPU pass.

    Unpacks a packed bitmask (1 bit per token, 32 tokens per ``int32`` word) and
    applies it to ``logits`` without materializing an intermediate bool tensor:
    a token is kept when its bit is set, otherwise its logit is replaced with
    ``fill_val``.

    Args:
        logits: Logits tensor of shape ``[batch, vocab]`` or
            ``[batch, num_positions, vocab]``.
        packed: Packed ``int32`` bitmask of shape ``[..., ceil(vocab / 32)]``
            with leading dims matching ``logits``. A set bit means the token is
            grammar-valid. Trailing 32-bit alignment padding beyond ``vocab`` is
            never read.
        fill_val: Value written for masked-out (grammar-invalid) tokens.

    Returns:
        Masked logits, same shape and dtype as ``logits``.
    """
    logits = TensorValue(logits)
    packed = TensorValue(packed)

    if packed.dtype != DType.int32:
        raise ValueError(
            f"apply_packed_bitmask requires an int32 bitmask, got {packed.dtype}"
        )
    if logits.rank != packed.rank:
        raise ValueError(
            "apply_packed_bitmask requires logits and packed bitmask of equal "
            f"rank, got {logits.rank} and {packed.rank}"
        )
    if logits.rank not in (2, 3):
        raise ValueError(
            f"apply_packed_bitmask requires 2d or 3d logits, got {logits.rank}"
        )

    # The kernel is rank-2 ([rows, vocab]); collapse any leading dims into a
    # single row dimension so a [batch, num_positions, vocab] acceptance-sampler
    # tensor and a [batch, vocab] token-sampler tensor share one code path.
    orig_shape = logits.shape
    if logits.rank == 3:
        rows = logits.shape[0] * logits.shape[1]
        logits_2d = ops.reshape(logits, [rows, logits.shape[2]])
        packed_2d = ops.reshape(packed, [rows, packed.shape[2]])
    else:
        logits_2d = logits
        packed_2d = packed

    masked = ops.custom(
        "mo.apply_packed_bitmask",
        device=logits.device,
        values=[
            logits_2d,
            packed_2d,
            ops.constant(fill_val, logits.dtype, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=logits.dtype,
                shape=logits_2d.shape,
                device=logits.device,
            )
        ],
    )[0].tensor

    if logits.rank == 3:
        return ops.reshape(masked, orig_shape)
    return masked


def scatter_nd_skip_oob_indices(
    input: TensorValueLike,
    updates: TensorValueLike,
    indices: TensorValueLike,
) -> TensorValue:
    """Creates a new symbolic tensor where the updates are scattered into input at specified indices.

    This differs from scatter_nd in that it handles oob indices by skipping
    the update for that index. Oob indices are those which fall outside of
    the range [-dim, dim).

    Args:
        input: The input symbolic tensor to write elements to.
        updates: A symbolic tensor of elements to write to input.
        indices: A tensor of indices specifying where to write updates.
            Shape should be [num_updates, rank] for full indexing or
            [num_updates, k] for partial indexing where k < rank.

    Returns:
        A new symbolic tensor representing the result of the scatter_nd operation.
    """
    input = TensorValue(input)
    updates = TensorValue(updates)
    indices = TensorValue(indices)

    if input.dtype != updates.dtype:
        raise ValueError(
            f"The input dtype ({input.dtype}) and updates dtype"
            f" ({updates.dtype}) must match"
        )

    if indices.dtype not in (DType.int32, DType.int64):
        raise ValueError(
            f"Invalid indices dtype: '{indices.dtype}'. Indices must be of type"
            " int32 or int64."
        )

    assert_same_device(input=input, updates=updates, indices=indices)

    return ops.custom(
        "mo.scatter_nd.skip_neg_indices",
        device=input.device,
        values=[input, updates, indices],
        out_types=[TensorType(input.dtype, input.shape, device=input.device)],
    )[0].tensor


def topk_fused_sampling(
    logits: TensorValue,
    top_k: TensorValueLike,
    *,
    temperature: TensorValueLike = 1.0,
    max_k: TensorValueLike | None = None,
    min_top_p: TensorValueLike | None = None,
    top_p: TensorValueLike = 1.0,
    min_p: TensorValueLike | None = None,
    seed: TensorValueLike = 0,
) -> TensorValue:
    """Performs top-k sampling with temperature scaling.

    Args:
        logits: Input logits tensor of shape [batch_size, vocab_size].
        top_k: Number of top tokens to consider for sampling. Can be a scalar
            (which will be expanded to batch_size) or a tensor of shape
            [batch_size].
        temperature: Temperature for scaling logits before sampling.
        max_k: Maximum value of k across the batch. Required when top_k is a
            tensor.
        top_p: Top-p (nucleus) sampling threshold. Can be a scalar or tensor.
        min_p: Per-row min_p probability filtering threshold of shape
            [batch_size]. Tokens with probability below
            ``min_p * max_prob`` are zeroed before sampling.
        seed: Seed for the random number generator. Can be a scalar or tensor.

    Returns:
        Sampled tokens tensor of shape [batch_size, 1].

    Raises:
        ValueError: If input validation fails.
    """
    batch_size = logits.shape[0]
    device = logits.device
    max_k_tensor = max_k

    if isinstance(top_k, int):
        if top_k <= -1:
            raise ValueError(f"top_k must be greater than -1, got {top_k}")

        if top_k == 0:
            top_k = -1

        max_k_tensor = ops.constant(
            top_k, dtype=DType.int64, device=DeviceRef.CPU()
        )
        top_k_tensor = ops.broadcast_to(
            ops.constant(top_k, dtype=DType.int64, device=device), [batch_size]
        )
    else:
        top_k_tensor = TensorValue(top_k)
        top_k_tensor = ops.where(top_k_tensor == 0, -1, top_k_tensor)
        if max_k_tensor is None:
            raise ValueError(
                "max_k must be explicitly set when top_k is a tensor"
            )
        if top_k_tensor.shape[0] != batch_size:
            raise ValueError(
                f"top_k tensor shape {top_k_tensor.shape} does not match"
                f" batch_size {batch_size}"
            )
        max_k_tensor = TensorValue(max_k_tensor)

    if isinstance(temperature, float):
        temperature_tensor = ops.broadcast_to(
            ops.constant(temperature, dtype=DType.float32, device=device),
            [batch_size],
        )
    else:
        temperature_tensor = TensorValue(temperature)
        if temperature_tensor.shape[0] != batch_size:
            raise ValueError(
                f"temperature tensor shape {temperature_tensor.shape} does not"
                f" match batch_size {batch_size}"
            )

    # Handle top_p parameter - can be scalar or tensor
    min_top_p_tensor = min_top_p
    if isinstance(top_p, float | int):
        if top_p <= 0 or top_p > 1:
            raise ValueError(f"expected top_p to be in (0, 1], got {top_p}")
        top_p_tensor = ops.broadcast_to(
            ops.constant(top_p, dtype=DType.float32, device=device),
            [batch_size],
        )
        # Set min_top_p to the scalar value if provided, otherwise use top_p
        min_top_p_value = min_top_p if min_top_p is not None else top_p
        assert isinstance(min_top_p_value, float | int)
        min_top_p_tensor = ops.constant(
            min_top_p_value, dtype=DType.float32, device=DeviceRef.CPU()
        )
    else:
        top_p_tensor = TensorValue(top_p)
        if top_p_tensor.shape[0] != batch_size:
            raise ValueError(
                f"top_p tensor shape {top_p_tensor.shape} does not match"
                f" batch_size {batch_size}"
            )
        # When top_p is a tensor, min_top_p must be provided
        if min_top_p is None:
            raise ValueError(
                "min_top_p must be explicitly set when top_p is a tensor"
            )
        min_top_p_tensor = TensorValue(min_top_p)

    # Handle min_p parameter - per-row tensor
    if min_p is None:
        min_p_tensor = ops.broadcast_to(
            ops.constant(0.0, dtype=DType.float32, device=device),
            [batch_size],
        )
    else:
        min_p_tensor = TensorValue(min_p)
        if min_p_tensor.shape[0] != batch_size:
            raise ValueError(
                f"min_p tensor shape {min_p_tensor.shape} does not match"
                f" batch_size {batch_size}"
            )

    # Handle seed parameter - can be scalar or tensor
    if isinstance(seed, int):
        seed_tensor = ops.broadcast_to(
            ops.constant(seed, dtype=DType.uint64, device=device), [batch_size]
        )
    else:
        seed_tensor = TensorValue(seed)
        if seed_tensor.shape[0] != batch_size:
            raise ValueError(
                f"seed tensor shape {seed_tensor.shape} does not match"
                f" batch_size {batch_size}"
            )

    batch_shape = logits.shape[:-1]

    return ops.custom(
        "sampler.fused_token_sampling",
        device=logits.device,
        values=[
            top_k_tensor,
            max_k_tensor,
            temperature_tensor,
            top_p_tensor,
            min_top_p_tensor,
            min_p_tensor,
            seed_tensor,
            logits,
        ],
        out_types=[
            TensorType(
                dtype=DType.int64, shape=batch_shape + [1], device=device
            )
        ],
    )[0].tensor


def topk_fused_sampling_with_dist(
    logits: TensorValue,
    *,
    top_k: TensorValue,
    temperature: TensorValue,
    top_p: TensorValue,
    seed: TensorValue,
) -> tuple[TensorValue, TensorValue]:
    """Samples a token per row and returns the distribution it came from.

    Applies per-row temperature, top-k and top-p to ``logits``, samples one
    token per row, and also returns the masked, renormalized distribution the
    token was drawn from. Speculative decoding subtracts that distribution to
    build its rejection residual, and reads the sampled token's own
    probability out of it -- a value the sampler must agree with, so it comes
    from the sampling kernel rather than a separate softmax.

    This is the same kernel and the same code path as
    :func:`topk_fused_sampling`, with the distribution write enabled.
    GPU-only.

    Args:
        logits: Raw logits ``[rows, vocab_size]``. Softmax is fused in.
        top_k: Per-row top-k ``[rows]``, int64. ``-1`` disables top-k, which
            is the production sentinel (``SamplingParams`` normalizes ``0``
            to ``-1``).
        temperature: Per-row temperature ``[rows]``, float32. ``0`` is
            clamped, collapsing the row to its argmax.
        top_p: Per-row nucleus threshold ``[rows]``, float32.
        seed: Per-row RNG seed ``[rows]``, uint64.

    Returns:
        ``(token_ids, distribution)``: int64 ``[rows]`` and float32
        ``[rows, vocab_size]``.

    Raises:
        ValueError: If the logits are not rank-2 on GPU, or a per-row
            parameter does not carry exactly one entry per logits row.
    """
    if logits.rank != 2:
        raise ValueError(
            "topk_fused_sampling_with_dist requires rank-2 logits, got "
            f"{logits.rank}"
        )
    if logits.device == DeviceRef.CPU():
        raise ValueError("topk_fused_sampling_with_dist is GPU-only")

    rows = logits.shape[0]
    device = logits.device

    # The kernel indexes all four per row, so a short one is an out-of-bounds
    # device read rather than a graph-construction error.
    for name, param in (
        ("top_k", top_k),
        ("temperature", temperature),
        ("top_p", top_p),
        ("seed", seed),
    ):
        if param.rank != 1 or param.shape[0] != rows:
            raise ValueError(
                f"{name} must be rank-1 with shape [{rows}] to match the"
                f" logits rows, got {param.shape}"
            )

    # The kernel resolves a per-row ``-1`` against this default and then
    # clamps a non-positive k to the vocabulary, so forwarding the sentinel
    # keeps every token for the rows that asked for no top-k. Passing the
    # vocabulary size instead would work but requires a static dimension.
    max_k = ops.constant(-1, DType.int64, DeviceRef.CPU())

    results = ops.custom(
        "sampler.fused_token_sampling_with_dist",
        device=device,
        values=[top_k, max_k, temperature, top_p, seed, logits],
        out_types=[
            TensorType(dtype=DType.int64, shape=[rows], device=device),
            TensorType(dtype=DType.float32, shape=logits.shape, device=device),
        ],
    )
    return results[0].tensor, results[1].tensor


def topk_topp_masked_probs(
    logits: TensorValue,
    *,
    top_k: TensorValue,
    temperature: TensorValue,
    top_p: TensorValue,
) -> TensorValue:
    """Computes each row's top-k/top-p masked softmax, without sampling.

    A token survives the joint constraint iff
    ``e_i = exp((logit_i - row_max) / temperature)`` exceeds the row's
    cutoff, with masked probability ``e_i / kept_mass``; everything else is
    zero. Each output row is that masked renormalized distribution -- the
    same form :func:`topk_fused_sampling_with_dist` emits for the draft, so
    speculative verification reads target probabilities and builds its
    rejection residual from this one tensor with no in-graph rebuild.

    GPU-only, with the same top-k sentinel and tie handling as
    :func:`topk_fused_sampling_with_dist`.

    Args:
        logits: Raw logits ``[rows, vocab_size]``.
        top_k: Per-row top-k ``[rows]``, int64.
        temperature: Per-row temperature ``[rows]``, float32.
        top_p: Per-row nucleus threshold ``[rows]``, float32.

    Returns:
        The masked distribution, float32 ``[rows, vocab_size]``.

    Raises:
        ValueError: If the logits are not rank-2 on GPU.
    """
    if logits.rank != 2:
        raise ValueError(
            f"topk_topp_masked_probs requires rank-2 logits, got {logits.rank}"
        )
    if logits.device == DeviceRef.CPU():
        raise ValueError("topk_topp_masked_probs is GPU-only")

    device = logits.device
    return ops.custom(
        "sampler.topk_topp_masked_probs",
        device=device,
        values=[top_k, temperature, top_p, logits],
        out_types=[
            TensorType(
                dtype=DType.float32,
                shape=logits.shape,
                device=device,
            ),
        ],
    )[0].tensor


def gumbel_argmax_from_probs(
    probs: TensorValue, *, seed: TensorValue
) -> TensorValue:
    """Draws one token per row, proportionally to unnormalized probabilities.

    Gumbel-max over ``ln(p)``: a zero probability can never win while the row
    has any positive mass, and row normalization does not matter. The Gumbel
    noise is generated inside the kernel from the per-row ``seed``, so rows
    with equal seeds draw with equal noise -- speculative decoding passes one
    seed per request, repeated across its draft positions, to share one noise
    row per request. GPU-only, and not supported on Apple GPUs, where the
    kernel's block reduction cannot cover a full-sized block.

    Args:
        probs: Unnormalized probabilities ``[rows, vocab_size]``, float32.
        seed: Per-row RNG seed ``[rows]``, uint64.

    Returns:
        The drawn token id per row, ``int64 [rows]``.

    Raises:
        ValueError: If the probabilities are not rank-2 on GPU.
    """
    if probs.rank != 2:
        raise ValueError(
            f"gumbel_argmax_from_probs requires rank-2 probs, got {probs.rank}"
        )
    if probs.device == DeviceRef.CPU():
        raise ValueError("gumbel_argmax_from_probs is GPU-only")

    return ops.custom(
        "sampler.gumbel_argmax_from_probs",
        device=probs.device,
        values=[seed, probs],
        out_types=[
            TensorType(
                dtype=DType.int64, shape=[probs.shape[0]], device=probs.device
            )
        ],
    )[0].tensor


def sgmv_kernel(  # noqa: ANN201
    input: TensorValue,
    lora: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    input_row_offsets: TensorValue,
    max_lora_seq_len: int,
    lora_end_idx: TensorValue | None = None,
    bias: TensorValue | None = None,
):
    """Performs the SGMV kernel for LoRA. This is LoRA agnostic, meaning that
    we can perform LoRA A or B from this kernel call.

    Args:
        input: The input tensor.
        lora: The LoRA tensor.
        lora_ids: Ids of the LoRAs used for each sequence
        lora_ranks: The ranks of the LoRAs in the batch.
        input_row_offsets: The sequence offsets that use LoRA
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch
        bias: The LoRA bias

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)

    _check_rank(3, lora=lora)

    _check_same_dtype(input=input, lora=lora)

    _check_dtype(DType.uint32, input_row_offsets=input_row_offsets)

    M = input.shape[0] if not lora_end_idx else lora_end_idx.shape[0]

    out = ops.custom(
        "mo.lora_sgmv.ragged",
        device=input.device,
        values=[
            input,
            lora,
            input_row_offsets,
            lora_ids,
            ops.constant(
                max_lora_seq_len,
                DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[M, lora.shape[1]],
                device=input.device,
            ),
        ],
    )[0].tensor

    return out


def sgmv_lora_kernel(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    lora_ranks: TensorValue,
    grouped_row_offsets: TensorValue,
    lora_end_idx: TensorValue,
    max_lora_seq_len: int,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes the SGMV LoRA kernel for some number of LoRAs A and B given the input.

    out = Wx + xAB

    SGMV can be explained by two independent kernels:
        - shrink -> shrinks high-dimensional tensor to low-rank tensor
        - expand -> expands low-rank tensor to high-dimensional tensor

    where v = [0, ...] and y = (some output tensor)

    SGMV-shrink:
        v += xA

    SGMV-expand:
        y += vB

    Args:
        input: The input tensor
        lora_a: The LoRA tensor for A
        lora_b: The LoRA tensor for B
        lora_ids: Ids of the LoRAs used for each sequence
        lora_ranks: The ranks of the LoRAs in the batch.
        grouped_row_offsets: The grouped sequence offsets that use LoRA
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch
        bias: The LoRA bias

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)

    _check_rank(3, lora_a=lora_a, lora_b=lora_b)

    _check_same_dtype(input=input, lora_a=lora_a, lora_b=lora_b)

    _check_dtype(DType.uint32, grouped_row_offsets=grouped_row_offsets)

    v = sgmv_kernel(
        input,
        lora_a,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        lora_end_idx,
        bias,
    )

    output = sgmv_kernel(
        v,
        lora_b,
        lora_ids,
        lora_ranks,
        grouped_row_offsets,
        max_lora_seq_len,
        lora_end_idx,
        bias,
    )

    return output


def sgmv_lora_qkv_shrink(
    input: TensorValue,
    lora_a: TensorValue,
    lora_ids: TensorValue,
    lora_grouped_offsets: TensorValue,
    lora_end_idx: TensorValue,
    max_lora_seq_len: int,
    max_rank: int,
) -> TensorValue:
    """LoRA shrink grouped matmul with planar Q/K/V output.

    Performs the LoRA 'shrink' operation for routed tokens using SGMV (segmented
    grouped matrix-vector multiplication). Computes `[M, K] @ [G, 3*rank, K]^T`
    per active LoRA adapter, then permutes the flat `[M, 3*rank]` result into a
    planar layout `[3, M, rank]` representing separate Q, K, V projections.

    Args:
        input: Routed activation matrix with shape (M, K), where M is the total
            number of tokens and K is the hidden dimension.
        lora_a: Shrink weights for all LoRA adapters, shape (G, 3*rank, K) where
            G is the number of adapters and rank is the LoRA rank.
        lora_ids: Expert/adapter indices for each active group, shape (num_active,).
            Values in range [0, G). May use -1 to indicate inactive slots.
        lora_grouped_offsets: Inclusive prefix sums of tokens per active adapter,
            shape (num_active + 1,). Defines per-adapter [start, end) ranges in
            input. Must be non-decreasing with offsets[0] == 0.
        max_lora_seq_len: Upper bound on tokens for any active adapter. Used for
            kernel tuning and memory allocation.
        max_rank: The maximum LoRA rank, determines output shape.

    Returns:
        Output tensor with planar Q/K/V layout, shape (3, M, max_rank).

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)

    _check_rank(3, lora_a=lora_a)

    _check_same_dtype(input=input, lora_a=lora_a)

    _check_dtype(DType.uint32, lora_grouped_offsets=lora_grouped_offsets)

    return ops.custom(
        "mo.lora_sgmv.qkv_shrink.ragged",
        device=input.device,
        values=[
            input,
            lora_a,
            lora_grouped_offsets,
            lora_ids,
            ops.constant(
                max_lora_seq_len,
                DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[3, lora_end_idx.shape[0], max_rank],
                device=input.device,
            ),
        ],
    )[0].tensor


def _sgmv_qkv_expand(
    p: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    lora_grouped_offsets: TensorValue,
    q_dim: int,
    kv_dim: int,
    max_lora_seq_len: int,
) -> tuple[TensorValue, TensorValue]:
    """Boundary-aware single-launch LoRA-B expand for fused Q/K/V projections.

    Performs the LoRA 'expand' (up-projection) for routed tokens in ONE grouped
    matmul over a single fused weight, and is correct under grouped-query
    attention (``q_dim != kv_dim``). The kernel selects the Q/K/V boundary
    natively, so no separate K/V offset/id arrays are needed.

    Args:
        p: Planar shrink output, shape (3, M, R) where plane 0/1/2 hold the
            q/k/v low-rank projections.
        lora_b: Fused LoRA-B weight, shape (G, q_dim + 2*kv_dim, R), with output
            rows concatenated along the projection dimension as Q | K | V.
        lora_ids: Adapter indices for each active group, shape (num_active,).
            ``-1`` marks an inactive group.
        lora_grouped_offsets: Inclusive prefix sums of tokens per active adapter,
            shape (num_active + 1,). The same grouping used by the shrink.
        q_dim: Output dimension of the Q projection.
        kv_dim: Output dimension of the K (and V) projection.
        max_lora_seq_len: Upper bound on tokens for any active adapter.

    Returns:
        A tuple ``(q_out, kv_out)`` where ``q_out`` has shape (M, q_dim) and
        ``kv_out`` has shape (2*M, kv_dim) with K rows in [0, M) and V rows in
        [M, 2M) -- the layout consumed by ``kv_cache_ragged_2m_iadd``.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(3, p=p, lora_b=lora_b)

    _check_same_dtype(p=p, lora_b=lora_b)

    _check_dtype(DType.uint32, lora_grouped_offsets=lora_grouped_offsets)

    m_dim = p.shape[1]

    results = ops.custom(
        "mo.lora_sgmv.qkv_expand.ragged",
        device=p.device,
        values=[
            p,
            lora_b,
            lora_grouped_offsets,
            lora_ids,
            ops.constant(
                max_lora_seq_len,
                DType.uint32,
                device=DeviceRef.CPU(),
            ),
        ],
        out_types=[
            TensorType(
                dtype=p.dtype,
                shape=[m_dim, q_dim],
                device=p.device,
            ),
            TensorType(
                dtype=p.dtype,
                shape=[m_dim * 2, kv_dim],
                device=p.device,
            ),
        ],
    )

    return results[0].tensor, results[1].tensor


def sgmv_qkv_lora_kernel(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    input_row_offsets: TensorValue,
    lora_grouped_offsets: TensorValue,
    lora_end_idx: TensorValue,
    batch_seq_len: TensorValue,
    kv_collection: PagedCacheValues,
    kv_params: KVCacheParams,
    layer_idx: TensorValue,
    q_dim: int,
    kv_dim: int,
    max_lora_seq_len: int,
    max_rank: int,
) -> TensorValue:
    """Computes the SGMV QKV LoRA kernel for Q, K, V projections with LoRA.

    The LoRA 'expand' (up-projection) is a single boundary-aware grouped matmul
    over one fused LoRA-B weight, correct under grouped-query attention
    (``q_dim != kv_dim``). The K/V projections are written directly into the
    paged KV cache; the Q projection is returned.

    Args:
        input: The input tensor.
        lora_a: The fused LoRA A tensor, shape (G, 3*R, K) (q/k/v stacked on
            the rank dim).
        lora_b: The fused LoRA B tensor, shape (G, q_dim + 2*kv_dim, R), with
            output rows concatenated along the projection dimension as Q | K | V.
        lora_ids: IDs of the LoRAs used for each sequence.
        input_row_offsets: The sequence offsets that use LoRA.
        lora_grouped_offsets: Grouped offsets for LoRA sequences.
        lora_end_idx: End index of LoRA tokens in the batch.
        batch_seq_len: Total sequence length of the batch.
        kv_collection: The KV cache.
        kv_params: The key-value cache configuration parameters.
        layer_idx: The layer index to retrieve the KV cache.
        q_dim: Output dimension of the Q projection.
        kv_dim: Output dimension of the K (and V) projection.
        max_lora_seq_len: The maximum sequence length of any given LoRA in the batch.
        max_rank: The maximum rank for the LoRAs.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)

    _check_rank(3, lora_a=lora_a, lora_b=lora_b)

    _check_same_dtype(input=input, lora_a=lora_a, lora_b=lora_b)

    _check_dtype(
        DType.uint32,
        input_row_offsets=input_row_offsets,
        lora_grouped_offsets=lora_grouped_offsets,
        layer_idx=layer_idx,
    )

    # shrink GMM: [M, K] @ [G, 3*R, K] -> planar [3, M, R].
    p = sgmv_lora_qkv_shrink(
        input=input,
        lora_a=lora_a,
        lora_ids=lora_ids,
        lora_grouped_offsets=lora_grouped_offsets,
        lora_end_idx=lora_end_idx,
        max_lora_seq_len=max_lora_seq_len,
        max_rank=max_rank,
    )

    # expand GMM: single boundary-aware launch over the fused LoRA-B weight.
    # q_out: [M, q_dim]; kv_out: [2M, kv_dim] (K rows [0, M), V rows [M, 2M)).
    q_out, kv_out = _sgmv_qkv_expand(
        p=p,
        lora_b=lora_b,
        lora_ids=lora_ids,
        lora_grouped_offsets=lora_grouped_offsets,
        q_dim=q_dim,
        kv_dim=kv_dim,
        max_lora_seq_len=max_lora_seq_len,
    )

    # write [2M, KVdim] directly to the paged cache.
    kv_cache_ragged_2m_iadd(
        kv_params=kv_params,
        a=kv_out,
        kv_collection=kv_collection,
        input_row_offsets=input_row_offsets,
        lora_end_idx=lora_end_idx,
        batch_seq_len=batch_seq_len,
        layer_idx=layer_idx,
    )

    return q_out


def sgmv_qkv_lora_fused(
    input: TensorValue,
    lora_a: TensorValue,
    lora_b: TensorValue,
    lora_ids: TensorValue,
    lora_grouped_offsets: TensorValue,
    lora_end_idx: TensorValue,
    q_dim: int,
    kv_dim: int,
    max_lora_seq_len: int,
    max_rank: int,
) -> TensorValue:
    """Returns the fused ``[q|k|v]`` LoRA contribution for a QKV projection.

    Same shrink + single boundary-aware expand as :func:`sgmv_qkv_lora_kernel`,
    but returns the full ``[q|k|v]`` output of shape ``[M, q_dim + 2*kv_dim]``
    (for the ``M`` LoRA tokens) instead of writing K/V into the cache. The
    caller adds it to the ``qkv`` projection before the fused rope + KV store.

    Args:
        input: The input tensor.
        lora_a: The fused LoRA A tensor, shape (G, 3*R, K) (q/k/v stacked on
            the rank dim).
        lora_b: The fused LoRA B tensor, shape (G, q_dim + 2*kv_dim, R), with
            output rows concatenated along the projection dimension as Q | K | V.
        lora_ids: IDs of the LoRAs used for each sequence.
        lora_grouped_offsets: Grouped offsets for LoRA sequences.
        lora_end_idx: End index of LoRA tokens in the batch.
        q_dim: Output dimension of the Q projection.
        kv_dim: Output dimension of the K (and V) projection.
        max_lora_seq_len: The maximum sequence length of any LoRA in the batch.
        max_rank: The maximum rank for the LoRAs.

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)
    _check_rank(3, lora_a=lora_a, lora_b=lora_b)
    _check_same_dtype(input=input, lora_a=lora_a, lora_b=lora_b)
    _check_dtype(
        DType.uint32,
        lora_grouped_offsets=lora_grouped_offsets,
    )

    m = lora_end_idx.shape[0]

    # shrink GMM: [M, K] @ [G, 3*R, K] -> planar [3, M, R].
    p = sgmv_lora_qkv_shrink(
        input=input,
        lora_a=lora_a,
        lora_ids=lora_ids,
        lora_grouped_offsets=lora_grouped_offsets,
        lora_end_idx=lora_end_idx,
        max_lora_seq_len=max_lora_seq_len,
        max_rank=max_rank,
    )

    # expand GMM: single boundary-aware launch over the fused LoRA-B weight.
    # q_out: [M, q_dim]; kv_out: [2M, kv_dim] (K rows [0, M), V rows [M, 2M)).
    q_out, kv_out = _sgmv_qkv_expand(
        p=p,
        lora_b=lora_b,
        lora_ids=lora_ids,
        lora_grouped_offsets=lora_grouped_offsets,
        q_dim=q_dim,
        kv_dim=kv_dim,
        max_lora_seq_len=max_lora_seq_len,
    )

    # kv_out is [2M, kv_dim] (K then V on dim 0); fold into [M, q_dim+2*kv_dim].
    return ops.concat([q_out, kv_out[:m, :], kv_out[m:, :]], axis=-1)


def kv_cache_ragged_2m_iadd(
    kv_params: KVCacheParams,
    a: TensorValue,
    kv_collection: PagedCacheValues,
    input_row_offsets: TensorValue,
    lora_end_idx: TensorValue,
    batch_seq_len: TensorValue,
    layer_idx: TensorValue,
) -> None:
    """In-place add to paged KV cache with interleaved K/V layout.

    Performs an in-place addition of new key-value projections to paged KV cache.
    The input tensor `a` uses a "2M" layout where keys and values are interleaved:
    rows [0, m) contain keys and rows [m, 2m) contain values, where m is the number
    of tokens.

    Args:
        kv_params: KV cache configuration parameters.
        a: Input tensor with interleaved K/V data, shape (2*m, hidden_size) where
            m is the number of tokens. Rows [0, m) are keys, rows [m, 2m) are values.
        kv_collection: The paged KV cache collection containing cache blocks,
            cache lengths, lookup tables, and max lengths tensors.
        input_row_offsets: Ragged tensor offsets indicating where each batch starts and ends
        lora_end_idx: End index of LoRA token portion. Marks the boundary between
            LoRA sequences and base model sequences in the batch.
        batch_seq_len: Total sequence length in the batch. Used for indexing
            into the value portion of `a`.
        layer_idx: The transformer layer index to update in the KV cache.

    Raises:
        ValueError: If `a` does not have rank 2.
        ValueError: If `input_row_offsets` does not have rank 1.
    """
    _check_rank(2, a=a)
    _check_rank(1, input_row_offsets=input_row_offsets)

    ops.inplace_custom(
        "mo.kv_cache.ragged.paged.2m_iadd",
        device=input_row_offsets.device,
        values=[
            a,
            *kv_collection.flatten_without_attention_dispatch_metadata(),
            input_row_offsets,
            lora_end_idx,
            batch_seq_len,
            layer_idx,
        ],
    )


def spatial_merge(
    input: TensorValue,
    grid_thw: TensorValue,
    hidden_size: int,
    merge_size: int,
) -> TensorValue:
    """Performs spatial merge operation on ragged input tensors.

    This operation merges spatial dimensions of input patches according to
    the grid dimensions specified in grid_thw.

    Args:
        input: Input tensor of shape [total_patches_in_grid, hidden_size]
        grid_thw: Grid dimensions tensor of shape [batch_size, 3] containing
            [t, h, w] for each batch item, where:
            - t: temporal/frame dimension
            - h: height dimension
            - w: width dimension
        hidden_size: Hidden dimension size
        merge_size: Size of spatial merge blocks (typically 2)

    Returns:
        Output tensor of shape [total_patches_in_grid, hidden_size]

    Raises:
        ValueError: on input shapes/dtypes that are invalid for the kernel.
    """
    _check_rank(2, input=input)

    _check_dtype(DType.int64, grid_thw=grid_thw)
    _check_rank(2, grid_thw=grid_thw)
    if grid_thw.shape[1] != 3:
        raise ValueError(
            f"expected grid_thw.shape[1] to be 3, got {grid_thw.shape[1]}"
        )

    if input.shape[1] != hidden_size:
        raise ValueError(
            f"expected input.shape[1] to match hidden_size ({hidden_size}), "
            f"got {input.shape[1]}"
        )

    return ops.custom(
        "mo.spatial_merge",
        device=input.device,
        values=[
            input,
            grid_thw,
            ops.constant(
                hidden_size, dtype=DType.int32, device=DeviceRef.CPU()
            ),
            ops.constant(merge_size, dtype=DType.int32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=[input.shape[0], hidden_size],
                device=input.device,
            )
        ],
    )[0].tensor


def learnable_2d_interp_pos_emb(
    x: TensorValueLike,
    weight: TensorValueLike,
    grid_thws: TensorValueLike,
    time_weight: TensorValueLike,
) -> TensorValue:
    """Applies learnable 2D interpolated position embedding (Kimi K2.5).

    For each video described by ``grid_thws``, bicubic-interpolates ``weight``
    from (H, W) to (h, w), optionally adds temporal sincos embedding when
    ``t > 1``, and adds the result element-wise to ``x``.

    Args:
        x: Patch embeddings of shape ``(L, dim)``.
        weight: Learnable 2D grid of shape ``(H, W, dim)``.
        grid_thws: Per-video ``(t, h, w)`` of shape ``(N, 3)``, dtype int64.
        time_weight: 1D sincos temporal embedding of shape
            ``(num_frames, dim)``, dtype float32.

    Returns:
        Tensor of shape ``(L, dim)`` with position embeddings added.

    Raises:
        ValueError: On invalid input shapes or dtypes.
    """
    x = TensorValue(x)
    weight = TensorValue(weight)
    grid_thws = TensorValue(grid_thws)
    time_weight = TensorValue(time_weight)

    _check_rank(2, x=x)
    _check_rank(3, weight=weight)
    if grid_thws.rank != 2 or grid_thws.shape[1] != 3:
        raise ValueError(
            "expected grid_thws of shape (N, 3), got rank="
            f"{grid_thws.rank} shape[1]={grid_thws.shape[1]}"
        )
    if grid_thws.dtype != DType.int64:
        raise ValueError(
            f"expected grid_thws dtype int64, got {grid_thws.dtype}"
        )
    _check_rank(2, time_weight=time_weight)

    return ops.custom(
        "learnable_2d_interp_pos_emb",
        device=x.device,
        values=[x, weight, grid_thws, time_weight],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=x.shape,
                device=x.device,
            )
        ],
    )[0].tensor


def sliced_add(
    x: TensorValue,
    y: TensorValue,
    lora_end_idx: TensorValue,
) -> TensorValue:
    """Adds tensors x and y element-wise for rows < lora_end_idx, otherwise copies x.

    This is used for LoRA where only some sequences have LoRA applied.
    For rows in [0, lora_end_idx): c = x + y
    For rows in [lora_end_idx, batch_seq_len): c = x

    Args:
        x: First input tensor.
        y: Second input tensor.
        lora_end_idx: End index of LoRA token portion (rows to apply add).
    """
    return ops.custom(
        "mo.sliced.add.ragged",
        device=x.device,
        values=[
            x,
            y,
            lora_end_idx,
        ],
        out_types=[
            TensorType(
                dtype=x.dtype,
                shape=x.shape,
                device=x.device,
            )
        ],
    )[0].tensor


def kv_cache_copy_pages_d2h(
    device_kv_collection: PagedCacheValues,
    device_page_ids: TensorValue,
    host_kv_blocks: BufferValue,
    host_page_ids: TensorValue,
    layer_idx: int,
    device_ref: DeviceRef,
) -> None:
    """Copy KV cache pages from GPU to CPU for a single layer.

    Performs async GPU->CPU copy of specified pages for layer-wise KV cache
    offloading.

    Args:
        device_kv_collection: Source KV cache on GPU.
        device_page_ids: Source page IDs to read from GPU.
        host_kv_collection: Destination KV cache on CPU.
        host_page_ids: Destination page IDs to write to CPU.
            Must have same length as device_page_ids.
        layer_idx: Which layer to copy.
        device_ref: Device for the GPU context.
    """
    ops.inplace_custom(
        name="mo.kv_cache.copy_pages_d2h",
        device=device_ref,
        values=[
            device_kv_collection.kv_blocks,
            host_kv_blocks,
            device_page_ids,
            host_page_ids,
            ops.constant(layer_idx, DType.uint32, device=DeviceRef.CPU()),
        ],
    )


def inplace_memcpy(dst: BufferValue, src: TensorValue) -> None:
    """Copies `src` into `dst` in place.

    Wraps the `mo.inplace_memcpy` custom op. Semantically equivalent to
    ``Buffer.inplace_copy_from``, but usable from within a compiled MAX
    graph so the copy can be scheduled alongside other graph work.

    Both operands must have the same dtype and shape. The op supports
    the four combinations expressible with a single `DeviceContext`:
    GPU-to-GPU on the same device, GPU-to-CPU, CPU-to-GPU, and
    CPU-to-CPU. Cross-GPU memcpy (different GPU ids) is rejected; use
    an explicit cross-device transfer for that case.
    The compute device is inferred from the operands: if either lives
    on a GPU the op is scheduled on that GPU, otherwise on CPU.
    Args:
        dst: Destination buffer mutated in place.
        src: Source tensor whose contents are copied into `dst`.
    """
    _check_same_dtype(dst=dst, src=src)
    if dst.shape != src.shape:
        raise ValueError(
            "Expected dst and src to have the same shape, but got "
            f"dst={dst.shape} and src={src.shape}"
        )
    if dst.device.is_gpu() and src.device.is_gpu() and dst.device != src.device:
        raise ValueError(
            "Cross-GPU memcpy is not supported; dst and src must be on "
            f"the same GPU, but got dst={dst.device} and src={src.device}"
        )

    if dst.device.is_gpu():
        compute_device = dst.device
    elif src.device.is_gpu():
        compute_device = src.device
    else:
        compute_device = dst.device

    ops.inplace_custom(
        "mo.inplace_memcpy",
        device=compute_device,
        values=[dst, src],
        out_types=[],
        parameters={
            "DstDevice": dst.device.device_type.value,
            "SrcDevice": src.device.device_type.value,
        },
    )


def launch_host_func(payload: BufferValue, device: DeviceRef) -> None:
    """Enqueues a Python callback on the device stream.

    Wraps the ``mo.launch_host_func`` custom op. The callback runs on a
    driver thread once the stream reaches this point, after all preceding
    work has completed.

    The payload buffer must be a CPU-resident int64[2] containing
    ``(trampoline_ptr, user_data_ptr)`` as returned by
    ``driver.__unsafe_pack_py_host_func``.

    Only supported on CUDA devices.

    Args:
        payload: CPU buffer of shape [2] and dtype int64 holding the
            packed callback pointers.
        device: GPU device on whose stream to enqueue the callback.
    """
    if payload.dtype != DType.int64:
        raise ValueError(f"Expected payload dtype int64, got {payload.dtype}")
    if payload.shape != [2]:
        raise ValueError(f"Expected payload shape [2], got {payload.shape}")
    if not device.is_gpu():
        raise ValueError("launch_host_func is only supported on GPU devices")

    ops.inplace_custom(
        "mo.launch_host_func",
        device=device,
        values=[payload],
        out_types=[],
    )


def wait_host_value_with_dep(
    payload: BufferValue,
    dep: BufferValue,
    device: DeviceRef,
) -> None:
    """Variant of ``wait_host_value`` with a fake mutable dependency.

    Wraps ``mo.wait_host_value_with_dep``. Behaves identically to
    :func:`wait_host_value` at runtime, but threads ``dep`` through the
    op as a mutated operand so any downstream op that mutates ``dep``
    must chain after the wait completes.

    Use this in place of :func:`wait_host_value` when the next op is an
    :func:`inplace_memcpy` whose dst is the buffer that needs to
    receive host-produced data. Without a shared operand the two
    ``inplace_custom`` ops carry no data dependency, and the graph
    compiler / cuGraph capture is free to parallelise them -- so the
    in-graph H2D can complete before the host callback signals the
    flag, producing one-iter-stale data at the consumer.

    Args:
        payload: CPU buffer of shape ``[2]`` and dtype ``int64`` holding
            ``[CompletionFlag._unsafe_ptr, expected_value]``. Same as
            :func:`wait_host_value`'s ``payload``.
        dep: The buffer the downstream op mutates. Threaded through as
            a fake mutable operand here to register a data dependency;
            not otherwise touched by this op.
        device: GPU device on whose stream to insert the wait node.
    """
    if payload.dtype != DType.int64:
        raise ValueError(f"Expected payload dtype int64, got {payload.dtype}")
    if payload.shape != [2]:
        raise ValueError(f"Expected payload shape [2], got {payload.shape}")
    if not device.is_gpu():
        raise ValueError(
            "wait_host_value_with_dep is only supported on GPU devices"
        )

    ops.inplace_custom(
        "mo.wait_host_value_with_dep",
        device=device,
        values=[payload, dep],
        out_types=[],
    )


def wait_host_value(payload: BufferValue, device: DeviceRef) -> None:
    """Stalls the device stream until a host-visible flag reaches a value.

    Wraps the ``mo.wait_host_value`` custom op, which lowers to
    ``cuStreamWaitValue64`` / ``hipStreamWaitValue64`` via
    ``DeviceQueue.wait_for_host_value``. Records into a captured device
    graph as a wait-value (batch-mem-op) node, so it can sit inside a
    captured forward graph to gate a downstream consumer kernel on
    CPU-produced data while the rest of the forward body runs
    concurrently.

    The payload buffer must be a CPU-resident ``int64[2]``:

    - ``payload[0]``: raw address of an ``M::Driver::CompletionFlag``
      (as ``u64``), typically obtained from
      ``max.driver.CompletionFlag._unsafe_ptr``. The C++ object must
      outlive any graph execution that references it.
    - ``payload[1]``: the 64-bit value to wait for (the ``int64``
      element is reinterpreted as a ``u64``).

    The payload shape mirrors ``mo.launch_host_func``'s
    ``[trampoline_ptr, user_data_ptr]`` pair; both ops carry their
    runtime pointers through a single ``int64[2]`` buffer rather than
    a typed graph operand.

    Typically paired with ``launch_host_func`` (or
    ``Device.__unsafe_enqueue_async_py_host_func``) placed earlier in
    the graph: the host callback dispatches CPU work that eventually
    signals the flag, and this op gates the consumer kernel on that
    signal.

    Only supported on CUDA and HIP devices.

    Args:
        payload: CPU buffer of shape ``[2]`` and dtype ``int64`` holding
            ``[CompletionFlag._unsafe_ptr, expected_value]``.
        device: GPU device on whose stream to insert the wait node.
    """
    if payload.dtype != DType.int64:
        raise ValueError(f"Expected payload dtype int64, got {payload.dtype}")
    if payload.shape != [2]:
        raise ValueError(f"Expected payload shape [2], got {payload.shape}")
    if not device.is_gpu():
        raise ValueError("wait_host_value is only supported on GPU devices")

    ops.inplace_custom(
        "mo.wait_host_value",
        device=device,
        values=[payload],
        out_types=[],
    )


def sleep(duration_sec: BufferValue, device_ref: DeviceRef) -> None:
    """Sleep for the given duration in seconds.

    This kernel is supported on CPUs and GPUs. However, the timing may be completely
    inaccurate on AMD GPUs due to limitation of current time.sleep(...) impl.

    Args:
        duration_sec: The duration to sleep in seconds.
    """
    # FIXME(GEX-3080): Convert duration_sec to a 0-d scalar instead of 1-d buffer.
    # We currently use 1-d buffer to prevent sleep op from being DCE'd away.
    if duration_sec.shape.static_dims != [1]:
        raise ValueError(
            "Expected duration_sec to have shape [1] but got"
            f" {duration_sec.shape.static_dims}"
        )
    if duration_sec.dtype != DType.float64:
        raise ValueError(
            "Expected duration_sec to have DType.float64 but got"
            f" {duration_sec.dtype}"
        )
    if duration_sec.device != DeviceRef.CPU():
        raise ValueError(
            f"Expected duration_sec to be on cpu but got {duration_sec.device}"
        )

    ops.inplace_custom(
        "mo.sleep",
        device=device_ref,
        values=[duration_sec],
        out_types=[],
    )


def tpool_patch_merger(
    input: TensorValueLike,
    grid_thws: TensorValueLike,
    kH: int,
    kW: int,
    max_h: int | TensorValueLike,
    max_w: int | TensorValueLike,
) -> TensorValue:
    """Performs temporal pooling patch merger on ragged video tokens.

    For each video in the batch, averages the input across the temporal (T)
    dimension and rearranges the result according to the spatial merge kernel
    (kH, kW).  Each video's T*H*W input tokens are reduced to H*W output
    tokens.  All videos are concatenated contiguously in the output.

    Args:
        input: Input tensor of shape ``[total_input_tokens, D]`` where
            ``total_input_tokens = sum(T_i * H_i * W_i)`` over all videos.
        grid_thws: Grid dimensions tensor of shape ``[n_videos, 3]`` with
            ``(T, H, W)`` per video.  Must have dtype ``int64``.
        kH: Merge kernel height.
        kW: Merge kernel width.
        max_h: Maximum ``H`` across all videos in the batch (for grid sizing).
            May be a Python int (baked as a graph constant) or a
            ``TensorValue`` computed at runtime (e.g. via ``ops.max``).
        max_w: Maximum ``W`` across all videos in the batch (for grid sizing).
            May be a Python int or a ``TensorValue``.
    Returns:
        Output tensor of shape ``[sum(H_i * W_i), D]``.

    Raises:
        ValueError: On invalid input shapes or dtypes.
    """
    input = TensorValue(input)
    grid_thws = TensorValue(grid_thws)

    _check_rank(2, input=input)

    _check_dtype(DType.int64, grid_thws=grid_thws)
    _check_rank(2, grid_thws=grid_thws)

    if grid_thws.shape[1] != 3:
        raise ValueError(
            f"expected grid_thws.shape[1] to be 3, got {grid_thws.shape[1]}"
        )

    D = input.shape[-1]

    max_h_val = (
        ops.constant(max_h, dtype=DType.int32, device=DeviceRef.CPU())
        if isinstance(max_h, int)
        else TensorValue(max_h)
    )
    max_w_val = (
        ops.constant(max_w, dtype=DType.int32, device=DeviceRef.CPU())
        if isinstance(max_w, int)
        else TensorValue(max_w)
    )
    # Compute exact merged row count dynamically and feed it to the custom-op
    # shape function as an integer scalar tensor.
    total_output_patches = ops.reshape(
        ops.sum(
            grid_thws[:, 1].cast(DType.int32)
            * grid_thws[:, 2].cast(DType.int32),
            axis=0,
        ),
        [],
    ).to(DeviceRef.CPU())

    return ops.custom(
        "tpool_patch_merger",
        device=input.device,
        values=[
            input,
            grid_thws,
            ops.constant(kH, dtype=DType.int32, device=DeviceRef.CPU()),
            ops.constant(kW, dtype=DType.int32, device=DeviceRef.CPU()),
            max_h_val,
            max_w_val,
            total_output_patches,
        ],
        out_types=[
            TensorType(
                dtype=input.dtype,
                shape=["total_output_patches", D],
                device=input.device,
            )
        ],
    )[0].tensor


def row_mean_of_squares(x: TensorValue) -> TensorValue:
    """Computes the per-row mean of squares over the last axis.

    For an input ``x`` flattened to ``[M, N]`` over its last axis, computes
    ``out[m] = sum_n(float32(x[m, n]) ** 2) / N``. The square and accumulation
    always run in ``float32`` regardless of the input dtype, and the result is
    always ``float32``. The output preserves the leading axes with a trailing
    size-1 reduction axis, matching ``ops.mean(x * x, axis=-1)``.

    This is a fused, single-pass replacement for ``ops.mean(x * x, axis=-1)``
    used in QK-RMSNorm-style variance computations. The generic reduce path
    over-provisions the grid for small ``M`` (decode); this op launches exactly
    one block per row.

    Args:
        x: The input tensor. Reduction runs over the last axis. Accepts
            ``bfloat16`` or ``float32`` (any rank >= 1).

    Returns:
        A ``float32`` :class:`~max.graph.TensorValue` whose shape matches
        ``x`` with the last axis replaced by ``1``.

    Raises:
        ValueError: If ``x`` dtype is not ``bfloat16`` or ``float32``.
    """
    if x.dtype not in (DType.bfloat16, DType.float32):
        raise ValueError(
            f"row_mean_of_squares expects bfloat16 or float32 input, got "
            f"{x.dtype}"
        )

    # Flatten leading axes to a single rows dimension so the kernel sees [M, N].
    rows = x.shape[:-1]
    cols = x.shape[-1]
    x_2d = x.reshape([-1, cols]) if x.rank != 2 else x

    out_2d = ops.custom(
        "mo.reduce.row_mean_of_squares",
        device=x.device,
        values=[x_2d],
        out_types=[
            TensorType(
                dtype=DType.float32,
                shape=[x_2d.shape[0], 1],
                device=x.device,
            )
        ],
    )[0].tensor

    # Restore the leading axes with a trailing size-1 reduction axis.
    return out_2d.reshape([*rows, 1])


def row_mean_of_squares_qk(q: TensorValue, k: TensorValue) -> TensorValue:
    """Fused per-row mean of squares for two operands ``q`` and ``k``.

    For ``q`` of shape ``[M, Nq]`` and ``k`` of shape ``[M, Nk]`` (sharing the
    leading rows dimension but with possibly different column counts), computes
    a ``[M, 2]`` ``float32`` result where column 0 is ``mean_n(q[m, n] ** 2)``
    and column 1 is ``mean_n(k[m, n] ** 2)``. The square and accumulation always
    run in ``float32`` regardless of input dtype.

    This is a single-kernel fusion of two :func:`row_mean_of_squares` calls plus
    a concat, equivalent to
    ``ops.concat([row_mean_of_squares(q), row_mean_of_squares(k)], axis=-1)``
    but with one launch instead of two plus the concat. It is used for the
    cross-head QK-RMSNorm variance statistics in tensor-parallel attention.

    Args:
        q: The Q projection, ``[M, Nq]``. Reduction runs over the last axis.
        k: The K projection, ``[M, Nk]``. Must share ``q``'s rows dim and dtype.

    Returns:
        A ``float32`` :class:`~max.graph.TensorValue` of shape ``[M, 2]``.

    Raises:
        ValueError: If ``q``/``k`` are not rank 2, have mismatched dtypes or
            rows, or use a dtype other than ``bfloat16`` or ``float32``.
    """
    if q.dtype not in (DType.bfloat16, DType.float32):
        raise ValueError(
            f"row_mean_of_squares_qk expects bfloat16 or float32 input, got "
            f"{q.dtype}"
        )
    if q.dtype != k.dtype:
        raise ValueError(
            f"row_mean_of_squares_qk requires matching dtypes, got {q.dtype} "
            f"(q) and {k.dtype} (k)"
        )
    if q.rank != 2 or k.rank != 2:
        raise ValueError(
            f"row_mean_of_squares_qk requires rank-2 inputs, got ranks "
            f"{q.rank} (q) and {k.rank} (k)"
        )

    return ops.custom(
        "mo.reduce.row_mean_of_squares_qk",
        device=q.device,
        values=[q, k],
        out_types=[
            TensorType(
                dtype=DType.float32,
                shape=[q.shape[0], 2],
                device=q.device,
            )
        ],
    )[0].tensor


def apply_qk_rms_norm(
    q: TensorValue,
    k: TensorValue,
    qk_var: TensorValue,
    gamma_q: TensorValue,
    gamma_k: TensorValue,
    epsilon: float | np.floating[Any],
) -> tuple[TensorValue, TensorValue]:
    """Applies QK-RMSNorm to ``q`` and ``k`` from precomputed variance.

    Given the already cross-rank-reduced per-row statistics ``qk_var`` of shape
    ``[M, 2]`` (column 0 = ``mean_n(q[m, n] ** 2)``, column 1 =
    ``mean_n(k[m, n] ** 2)``, ``float32``) and per-column ``float32`` scales
    ``gamma_q``/``gamma_k``, computes in a single kernel launch::

        q_out[m, c] = cast((cast(q[m, c], f32) * rsqrt(qk_var[m, 0] + eps))
                           * gamma_q[c], q.dtype)
        k_out[m, c] = cast((cast(k[m, c], f32) * rsqrt(qk_var[m, 1] + eps))
                           * gamma_k[c], k.dtype)

    This is a single-kernel fusion of the QK-RMSNorm apply chain
    (``rsqrt`` scale, ``gamma`` multiply, and downcast for both Q and K),
    equivalent to::

        qf = ops.cast(q, DType.float32) * ops.rsqrt(qk_var[:, 0:1] + epsilon)
        kf = ops.cast(k, DType.float32) * ops.rsqrt(qk_var[:, 1:2] + epsilon)
        q_out = ops.cast(qf * gamma_q, q.dtype)
        k_out = ops.cast(kf * gamma_k, k.dtype)

    but with one launch instead of ~7 elementwise/slice kernels. The float
    grouping ``((x * rs) * gamma)`` then cast matches the unfused form above. It
    is used for the cross-head QK-RMSNorm apply in tensor-parallel attention,
    where the variance is all-reduced across ranks between its computation
    (:func:`row_mean_of_squares_qk`) and this apply.

    Args:
        q: The Q projection, ``[M, Nq]``. ``bfloat16`` or ``float32``.
        k: The K projection, ``[M, Nk]``. Must share ``q``'s rows dim and dtype.
        qk_var: The reduced variance statistics, ``[M, 2]`` ``float32``.
        gamma_q: The Q norm scale, ``[Nq]`` ``float32``.
        gamma_k: The K norm scale, ``[Nk]`` ``float32``.
        epsilon: The RMSNorm epsilon added to the variance before ``rsqrt``.

    Returns:
        A tuple ``(q_out, k_out)`` of normalized tensors matching the shapes and
        dtype of ``q`` and ``k`` respectively.

    Raises:
        ValueError: If ``q``/``k`` are not rank 2, have mismatched dtypes or
            rows, use a dtype other than ``bfloat16`` or ``float32``, or if the
            ``qk_var``/``gamma_q``/``gamma_k`` shapes or dtypes do not match.
    """
    if q.dtype not in (DType.bfloat16, DType.float32):
        raise ValueError(
            f"apply_qk_rms_norm expects bfloat16 or float32 input, got {q.dtype}"
        )
    if q.dtype != k.dtype:
        raise ValueError(
            f"apply_qk_rms_norm requires matching q/k dtypes, got {q.dtype} "
            f"(q) and {k.dtype} (k)"
        )
    if q.rank != 2 or k.rank != 2:
        raise ValueError(
            f"apply_qk_rms_norm requires rank-2 q/k, got ranks {q.rank} (q) "
            f"and {k.rank} (k)"
        )
    if qk_var.dtype != DType.float32 or qk_var.rank != 2:
        raise ValueError(
            f"apply_qk_rms_norm requires a rank-2 float32 qk_var, got dtype "
            f"{qk_var.dtype} rank {qk_var.rank}"
        )
    if gamma_q.dtype != DType.float32 or gamma_k.dtype != DType.float32:
        raise ValueError(
            f"apply_qk_rms_norm requires float32 gamma, got {gamma_q.dtype} "
            f"(gamma_q) and {gamma_k.dtype} (gamma_k)"
        )
    if gamma_q.rank != 1 or gamma_k.rank != 1:
        raise ValueError(
            f"apply_qk_rms_norm requires rank-1 gamma, got ranks "
            f"{gamma_q.rank} (gamma_q) and {gamma_k.rank} (gamma_k)"
        )

    q_out, k_out = ops.custom(
        "mo.norm.apply_qk_rms_norm",
        device=q.device,
        values=[
            q,
            k,
            qk_var,
            gamma_q,
            gamma_k,
            ops.constant(epsilon, DType.float32, device=DeviceRef.CPU()),
        ],
        out_types=[
            TensorType(dtype=q.dtype, shape=q.shape, device=q.device),
            TensorType(dtype=k.dtype, shape=k.shape, device=k.device),
        ],
    )
    return q_out.tensor, k_out.tensor
