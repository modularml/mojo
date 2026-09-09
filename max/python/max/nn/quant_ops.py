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

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops

from .kernels import (
    _apple_int8_w8a8_matmul,
    _apple_weight_only_block_scaled_matmul,
    _apple_weight_only_scaled_float8_matmul,
    _fused_qkv_index_ragged_matmul_scaled_mxfp8,
    _fused_qkv_ragged_matmul_scaled_float4,
    _fused_qkv_ragged_matmul_scaled_float8,
    _fused_qkv_ragged_matmul_scaled_mxfp8,
    _grouped_matmul_rowwise_dynamic_scaled_fp8,
    _is_amd_gpu,
    _is_apple_gpu,
    block_scales_interleave,
    convert_weights_to_fp8_fnuz_if_needed,
    dynamic_block_scaled_matmul,
    dynamic_block_scaled_matmul_amd,
    dynamic_block_scaled_matmul_mxfp6,
    dynamic_scaled_matmul,
    grouped_dynamic_scaled_fp8_matmul,
    grouped_matmul_ragged,
    matmul_static_scaled_float8,
    mxfp4_dequant,
    mxfp6_dequant,
    quantize_dynamic_block_scaled,
    quantize_dynamic_block_scaled_mxfp4,
    quantize_dynamic_block_scaled_mxfp6,
    quantize_dynamic_scaled_float8,
    quantize_static_scaled_float8,
    quantize_tensor_dynamic_scaled_float8,
)
from .kv_cache import KVCacheParams, PagedCacheValues
from .quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    WeightScaleSpec,
)


def _reshape_pre_interleaved_scales(
    weight_scale: TensorValue,
) -> TensorValue:
    """Reshape pre-interleaved FP4 scales from 2D to 5D TCGEN layout.

    Checkpoints with pre-interleaved scales store them in 5D TCGEN order
    but flattened to `[M, K//16]`.  The matmul kernel requires rank-5
    input: `(M//128, K//64, 32, 4, 4)`.
    """
    M = weight_scale.shape[0]
    n_blocks = weight_scale.shape[1]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = 128
    SF_ATOM_M0 = 32
    return weight_scale.reshape(
        [
            M // SF_MN_GROUP_SIZE,
            n_blocks // SF_ATOM_K,
            SF_ATOM_M0,
            SF_MN_GROUP_SIZE // SF_ATOM_M0,
            SF_ATOM_K,
        ]
    )


def _matmul_float4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue,
    weight_scale_2: TensorValue,
    scales_pre_interleaved: bool = False,
) -> TensorValue:
    """Computes x @ weight.T with modelopt NVFP4 quantization.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in uint8 (float4-e2m1x2).
        weight_scale: The weight scale tensor in f8e4m3fn.
        input_scale: The input scale factor in f32 (used with vLLM convention by kernel).
        weight_scale_2: Additional weight scale factor in f32.
        scales_pre_interleaved: If True, weight_scale is already in 5D
            TCGEN interleaved layout and `block_scales_interleave` is
            skipped.

    Returns:
        The output tensor in bf16.
    """
    if _is_apple_gpu():
        # Apple M5 weight-only (W4A16) path: keep the activation in bf16 (do
        # NOT dynamically quantize it to FP4) and feed the weight's PLAIN
        # rank-2 ``[N, K // 16]`` block scales straight to the kernel (no
        # rank-5 TCGEN05 interleave). The FP4 weight is dequantized to bf16
        # in-register at the MMA loader seam. The kernel applies only the
        # per-16-element block scale, so the NVFP4 per-tensor ``weight_scale_2``
        # is folded in here as a post-matmul scalar multiply. (``input_scale``
        # cancels: the SM100 path scales x by ``1/input_scale`` then folds
        # ``input_scale`` back into the epilogue alpha; with bf16 activations
        # neither step happens, so the only surviving global factor is
        # ``weight_scale_2``.)
        weight_scale = weight_scale.to(x.device)
        if scales_pre_interleaved:
            # Pre-interleaved checkpoints store rank-2 scales flattened from
            # the SM100 5D layout, which Apple's rank-2 [N, K//16] consumer
            # cannot read directly. The FLUX.2 adapter deinterleaves to true
            # rank-2 at load (scales_pre_interleaved=False), so this is not hit
            # on the supported path.
            raise NotImplementedError(
                "Apple W4A16 path requires deinterleaved rank-2 weight scales "
                "(scales_pre_interleaved=False)"
            )
        res = _apple_weight_only_block_scaled_matmul(
            x,
            weight,
            weight_scale,
            out_type=DType.bfloat16,
        )
        # Fold the NVFP4 per-tensor scale (the kernel applies block scales
        # only). Do the multiply in f32 for precision, then cast the product to
        # bf16 -- folding a bf16-rounded scale would lose mantissa bits before
        # the multiply.
        return (res.cast(DType.float32) * weight_scale_2.to(res.device)).cast(
            DType.bfloat16
        )

    x, x_scales = quantize_dynamic_block_scaled(
        x,
        tensor_sf=1.0 / input_scale,
        scales_type=DType.float8_e4m3fn,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )

    weight_scale = weight_scale.to(x.device)
    if scales_pre_interleaved:
        weight_scale = _reshape_pre_interleaved_scales(weight_scale)
    else:
        weight_scale = block_scales_interleave(weight_scale)

    res = dynamic_block_scaled_matmul(
        x,
        weight,
        x_scales,
        weight_scale,
        tensor_sf=weight_scale_2 * input_scale,
        out_type=DType.bfloat16,
    )
    return res


def _matmul_float4_mxfp4(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Computes x @ weight.T with MXFP4 quantization.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in uint8 (float4-e2m1x2).
        weight_scale: The weight scale tensor in float8_e8m0fnu.

    Returns:
        The output tensor in bf16.
    """
    x, x_scales = quantize_dynamic_block_scaled_mxfp4(
        x,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.uint8,  # fp4-e2m1fnX2
    )

    weight_scale = weight_scale.to(x.device)

    res = dynamic_block_scaled_matmul_amd(
        x,
        weight,
        x_scales,
        weight_scale,
        out_type=DType.bfloat16,
    )
    return res


def _matmul_float6_mxfp6(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    fp6_format: str,
) -> TensorValue:
    """Computes ``x @ weight.T`` with MXFP6 quantization (A6W6).

    MXFP6 sibling of :func:`_matmul_float4_mxfp4`: the activation is
    dynamically quantized to the same FP6 encoding as the weights, so both
    operands drive the CDNA4 ``f8f6f4`` MFMA at 24 bytes per lane.

    Args:
        x: The input tensor in bf16.
        weight: The packed FP6 weights, uint8 ``[N, K * 3 // 4]``.
        weight_scale: The E8M0 weight scales, ``[N, K // 32]``.
        fp6_format: The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.

    Returns:
        The output tensor in bf16.
    """
    if not _is_amd_gpu():
        raise ValueError(
            "MXFP6 requires the AMD CDNA4 block-scaled MFMA (gfx950); no other "
            "target implements a 6-bit block-scaled matmul"
        )

    x_fp6, x_scales = quantize_dynamic_block_scaled_mxfp6(
        x,
        fp6_format=fp6_format,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.uint8,
    )
    return dynamic_block_scaled_matmul_mxfp6(
        x_fp6,
        weight,
        x_scales,
        weight_scale.to(x.device),
        fp6_format=fp6_format,
        out_type=DType.bfloat16,
    )


def _matmul_float8_mxfp8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Computes ``x @ weight.T`` with MXFP8 quantization.

    MXFP8 sibling of :func:`_matmul_float4_mxfp4`: the activation is
    dynamically quantized to ``float8_e4m3fn`` with E8M0 block scales.

    On cuda the ``float8_e8m0fnu`` weight scales are then interleaved into the
    rank-5 SF-atom layout and the SM100 block-scaled tensor-core MMA
    (``UMMAKind.KIND_MXF8F6F4``) is used, avoiding the naive CUDA-core
    blockwise-FP8 fallback. AMD returns earlier: it consumes rank-2 E8M0 scales
    and raw byte operands directly, with no interleave.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in ``float8_e4m3fn``, shape ``[N, K]``.
        weight_scale: The E8M0 (``float8_e8m0fnu``) weight scales, rank-2
            ``[N, K // 32]`` as loaded from the checkpoint.

    Returns:
        The output tensor in bf16.
    """
    if _is_amd_gpu():
        # AMD reads rank-2 E8M0 scales and raw byte operands; no rank-5 SF-atom
        # interleave, and `lane_bytes=32` is derived from the operand dtype in
        # the wrapper. Mirrors `_matmul_float4_mxfp4`, one byte per element.
        x_fp8, x_scales = quantize_dynamic_block_scaled(
            x,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
        return _matmul_float8_mxfp8_prequantized(
            x_fp8, x_scales, weight, weight_scale
        )

    x, x_scales = quantize_dynamic_block_scaled(
        x,
        sf_vector_size=32,
        scales_type=DType.float8_e8m0fnu,
        out_type=DType.float8_e4m3fn,
    )

    weight_scale = weight_scale.to(x.device)
    # Lift the rank-2 [N, K/32] checkpoint scales into the rank-5 tcgen05
    # SF-atom layout the block-scaled MMA consumes.
    weight_scale = block_scales_interleave(weight_scale, sf_vector_size=32)

    return dynamic_block_scaled_matmul(
        x,
        weight,
        x_scales,
        weight_scale,
        sf_vector_size=32,
        out_type=DType.bfloat16,
    )


def _matmul_float8_mxfp8_prequantized(
    x_fp8: TensorValue,
    x_scales: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
) -> TensorValue:
    """Computes the AMD MXFP8 matmul from an already-quantized activation.

    The AMD arm of :func:`_matmul_float8_mxfp8`, minus the dynamic quantize:
    single owner for the operand order, scale layout (rank-2 E8M0, no SF-atom
    interleave on CDNA), and output dtype. Callers that produce the
    ``(data, scales)`` pair themselves -- e.g. the MSA attention op that fuses
    the quantize into its split-K combine -- must go through here so a scale
    layout change lands in one place.

    Args:
        x_fp8: The activation in ``float8_e4m3fn``, shape ``[M, K]``.
        x_scales: The activation's E8M0 block scales, rank-2 ``[M, K // 32]``.
        weight: The weight tensor in ``float8_e4m3fn``, shape ``[N, K]``.
        weight_scale: The E8M0 weight scales, rank-2 ``[N, K // 32]`` as loaded
            from the checkpoint.

    Returns:
        The output tensor in bf16, shape ``[M, N]``.
    """
    return dynamic_block_scaled_matmul_amd(
        x_fp8,
        weight,
        x_scales,
        weight_scale.to(x_fp8.device),
        out_type=DType.bfloat16,
    )


def _matmul_float8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    quant_config: QuantConfig,
) -> TensorValue:
    """Computes x @ weight.T with float8 quantization.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (only required for static
            fp8 quantization).
        quant_config: The quantization configuration.

    Returns:
        The output tensor.
    """
    if _is_apple_gpu():
        # Apple M5 weight-only (W8A16) path: keep the activation in bf16 (do NOT
        # quantize it to FP8) and feed the FP8-E4M3 weight straight to the kernel,
        # which widens it to f32/bf16 at the point of consumption (register-
        # resident GEMV at M=1; a transient bf16 buffer for the M>1 interim). The
        # kernel produces the RAW `x @ W_fp8^T`, so the modelopt per-tensor scalar
        # `weight_scale` is folded here as a post-matmul multiply -- the FP8 analog
        # of the NVFP4 `weight_scale_2` fold in `_matmul_float4`. (`input_scale`
        # cancels: the static path scales x by `1/input_scale` then folds
        # `input_scale` back into the epilogue; with a bf16 activation neither step
        # happens, so `weight_scale` is the only surviving global factor. Because
        # it is a per-tensor scalar it factors out of the sum, so the post-matmul
        # fold is exact.) Placed before the fnuz conversion (AMD-only) so the
        # weight stays `float8_e4m3fn`, which Metal reads natively. The gate is
        # additive: NVIDIA/AMD FP8 below is untouched.
        res = _apple_weight_only_scaled_float8_matmul(
            x, weight, out_type=DType.float32
        )
        # Fold the per-tensor scale in f32, then a single cast to bf16:
        # fp32-accum -> f32 out -> `* weight_scale` (f32) -> one bf16 round.
        # Requesting f32 out (not bf16) avoids a premature bf16 round of the raw
        # matmul BEFORE the scale multiply (folding a bf16-rounded product would
        # lose mantissa bits). Mirrors the NVFP4 `weight_scale_2` fold.
        return (res * weight_scale.to(res.device)).cast(DType.bfloat16)

    weight, weight_scale = convert_weights_to_fp8_fnuz_if_needed(
        weight, weight_scale
    )

    if input_scale is not None:
        # Static quantization: input scale was pre-computed.
        x = quantize_static_scaled_float8(x, input_scale, out_type=weight.dtype)
        if quant_config.weight_scale.is_tensor and weight_scale.rank >= 2:
            # Fused QKV: per-projection weight scales were broadcast to
            # rowwise [N, 1] in qkv_weight_scale. Route through
            # dynamic_scaled_matmul (colwise/rowwise) because
            # matmul_static_scaled_float8 requires scalar scales.
            weight_scale = weight_scale.to(x.device)
            # Broadcast scalar input_scale to [1, M] to match the
            # per-token layout that the colwise/rowwise kernel expects.
            a_scales = ops.broadcast_to(
                input_scale.reshape([1, 1]).to(x.device), [1, x.shape[0]]
            )
            colwise_input_spec = InputScaleSpec(
                granularity=ScaleGranularity.COLWISE,
                origin=quant_config.input_scale.origin,
                dtype=weight_scale.dtype,
            )
            rowwise_weight_spec = WeightScaleSpec(
                granularity=ScaleGranularity.ROWWISE,
                dtype=weight_scale.dtype,
            )
            return dynamic_scaled_matmul(
                x,
                weight,
                a_scales,
                weight_scale,
                colwise_input_spec,
                rowwise_weight_spec,
                out_type=DType.bfloat16,
            )
        return matmul_static_scaled_float8(x, weight, input_scale, weight_scale)
    elif (
        quant_config.input_scale.is_tensor
        and quant_config.weight_scale.is_tensor
    ):
        # GEX-3496 workaround for dynamic tensor-wise FP8: force
        # tensor-scale into row-scale format so we hit the
        # rowwise/colwise kernel path.
        n_out = weight.shape[0]
        if quant_config.weight_scale.is_tensor and weight_scale.rank < 2:
            weight_scale = ops.broadcast_to(
                weight_scale.reshape([1, 1]), [n_out, 1]
            )
        elif weight_scale.rank < 2:
            weight_scale = weight_scale.reshape([1, 1])
        weight_scale = weight_scale.to(x.device)

        colwise_input_spec = InputScaleSpec(
            granularity=ScaleGranularity.COLWISE,
            origin=quant_config.input_scale.origin,
            dtype=weight_scale.dtype,
        )
        rowwise_weight_spec = WeightScaleSpec(
            granularity=ScaleGranularity.ROWWISE,
            dtype=weight_scale.dtype,
        )

        x, x_scales = quantize_dynamic_scaled_float8(
            x,
            colwise_input_spec,
            rowwise_weight_spec,
            scales_type=weight_scale.dtype,
            out_type=weight.dtype,
        )

        # activation scales are [1xm] which but in the dynamic scaled matmul we expect just use x_scale[0,0]
        return dynamic_scaled_matmul(
            x,
            weight,
            x_scales,
            weight_scale,
            colwise_input_spec,
            rowwise_weight_spec,
            out_type=DType.bfloat16,
        )
    else:
        # Dynamic non-per-tensor (per-row/block).
        x, x_scale = quantize_dynamic_scaled_float8(
            x,
            quant_config.input_scale,
            quant_config.weight_scale,
            scales_type=weight_scale.dtype,
            out_type=weight.dtype,
        )
        weight_scale = weight_scale.to(x.device)

    return dynamic_scaled_matmul(
        x,
        weight,
        x_scale,
        weight_scale,
        quant_config.input_scale,
        quant_config.weight_scale,
        out_type=DType.bfloat16,
    )


def _matmul_int8(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Computes x @ weight.T (+bias) with symmetric int8 W8A8 quant (Apple M5).

    The activation ``x`` (bf16) is dynamically quantized to int8 per token and
    the pre-quantized int8 ``weight`` (per-output-channel ``weight_scale``) is
    fed to the int8 widening-MMA GEMM. Apple M5 only -- there is no
    NVIDIA/AMD int8 W8A8 dense path here.

    Args:
        x: The bf16 input activation, shape ``[M, K]``.
        weight: The int8 weight, shape ``[N, K]`` (RTN-quantized at load).
        weight_scale: The fp32 per-output-channel weight scale, ``[N, 1]``.
        bias: Optional per-output-channel bias, shape ``[N]``; fused into the
            dequant epilogue (added after dequant) instead of a separate add.

    Returns:
        The output tensor in bf16, shape ``[M, N]``.
    """
    if not _is_apple_gpu():
        raise NotImplementedError(
            "int8 W8A8 dense matmul is only implemented for Apple M5 GPUs"
        )
    return _apple_int8_w8a8_matmul(
        x, weight, weight_scale, bias=bias, out_type=DType.bfloat16
    )


def quantized_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    input_scale: TensorValue | None,
    quant_config: QuantConfig,
    weight_scale_2: TensorValue | None = None,
    bias: TensorValue | None = None,
) -> TensorValue:
    """Single entry point for all quantized dense matmuls.

    Dispatches to the appropriate kernel based on the quantization format
    in ``quant_config``.

    Args:
        x: The input tensor.
        weight: The weight tensor.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor (required for NVFP4 and
            static FP8).
        quant_config: The quantization configuration.
        weight_scale_2: Additional weight scale factor (NVFP4 only).
        bias: Optional bias tensor. Only the int8 W8A8 (Apple M5) path fuses it
            into the matmul epilogue; other formats leave the caller to add it.

    Returns:
        The output tensor.
    """
    match quant_config.format:
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert weight_scale_2 is not None
            return _matmul_float4(
                x,
                weight,
                weight_scale,
                input_scale,
                weight_scale_2,
                scales_pre_interleaved=quant_config.scales_pre_interleaved,
            )
        case QuantFormat.MXFP4:
            return _matmul_float4_mxfp4(
                x,
                weight,
                weight_scale,
            )
        case QuantFormat.MXFP6:
            return _matmul_float6_mxfp6(
                x,
                weight,
                weight_scale,
                quant_config.mxfp6_format,
            )
        case QuantFormat.MXFP8:
            return _matmul_float8_mxfp8(
                x,
                weight,
                weight_scale,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            return _matmul_float8(
                x,
                weight,
                weight_scale,
                input_scale,
                quant_config,
            )
        case QuantFormat.INT8_W8A8:
            return _matmul_int8(x, weight, weight_scale, bias=bias)
        case _:
            raise ValueError(
                f"Unsupported quantization format for dense matmul: {quant_config.format}"
            )


def quantized_fused_qkv_matmul(
    kv_params: KVCacheParams,
    x: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    quant_config: QuantConfig,
    weight_scale: TensorValue,
    input_scale: TensorValue | None = None,
    weight_scale_2: TensorValue | None = None,
    bias: TensorValue | None = None,
    _output_dim: int | None = None,
) -> TensorValue:
    """Single entry point for quantized fused QKV matmuls.

    Dispatches to the NVFP4 or FP8 fused QKV kernel based on the
    quantization format in ``quant_config``.

    Args:
        kv_params: KV cache parameters.
        x: The input tensor of shape ``[total_seq_len, hidden_dim]``.
        wqkv: The concatenated QKV weight tensor.
        kv_collection: The paged KV cache.
        layer_idx: The current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of attention heads.
        quant_config: The quantization configuration.
        weight_scale: The weight scale tensor.
        input_scale: The input scale tensor.
        weight_scale_2: Additional weight scale factor (NVFP4 only).
        bias: Optional bias tensor (FP8 only).
        _output_dim: Optional output dimension override for the FP8
            kernel. If not provided, defaults to
            ``n_heads * head_dim``.

    Returns:
        The query projection output tensor.
    """
    match quant_config.format:
        case QuantFormat.MXFP8:
            if bias is not None:
                raise NotImplementedError(
                    "bias is not supported by the fused MXFP8 QKV kernel"
                )
            x_fp8, x_scales = quantize_dynamic_block_scaled(
                x,
                sf_vector_size=32,
                scales_type=DType.float8_e8m0fnu,
                out_type=DType.float8_e4m3fn,
            )
            # Only SM100 needs the rank-5 SF-atom interleave; CDNA4 consumes the
            # checkpoint's rank-2 E8M0 scales directly.
            weight_scale = weight_scale.to(x.device)
            if not _is_amd_gpu():
                weight_scale = block_scales_interleave(
                    weight_scale, sf_vector_size=32
                )
            return _fused_qkv_ragged_matmul_scaled_mxfp8(
                kv_params,
                input=x_fp8,
                input_row_offsets=input_row_offsets,
                wqkv=wqkv,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale,
                _output_dim=_output_dim,
            )
        case QuantFormat.NVFP4:
            assert input_scale is not None
            assert weight_scale_2 is not None

            x, x_scales = quantize_dynamic_block_scaled(
                x,
                tensor_sf=1.0 / input_scale,
                scales_type=DType.float8_e4m3fn,
                out_type=DType.uint8,
            )

            weight_scale = weight_scale.to(x.device)
            if quant_config.scales_pre_interleaved:
                weight_scale = _reshape_pre_interleaved_scales(weight_scale)
            else:
                weight_scale = block_scales_interleave(weight_scale)

            return _fused_qkv_ragged_matmul_scaled_float4(
                kv_params,
                input=x,
                input_row_offsets=input_row_offsets,
                wqkv=wqkv,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale,
                tensor_sf=input_scale * weight_scale_2,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            # FP8 path (static or dynamic)
            if quant_config.is_static:
                assert input_scale is not None
                x = quantize_static_scaled_float8(
                    x, input_scale.to(DeviceRef.CPU())
                )
                x_scales = input_scale
            elif (
                quant_config.input_scale.is_tensor
                and quant_config.weight_scale.is_tensor
            ):
                # Workaround for GEX-3496: force tensor-scale into
                # row-scale format. Input: per-tensor dynamic quantization.
                # Weight: rowwise [total_dim, 1] from qkv_weight_scale.
                x, x_scales = quantize_tensor_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    out_type=wqkv.dtype,
                    scales_type=weight_scale.dtype,
                )
                # x_scales = x_scales.reshape([1, 1])
                # Don't pass quant_config so the kernel infers
                # per-channel (1,1,-1) from [1,1] + [N,1] shapes.
                return _fused_qkv_ragged_matmul_scaled_float8(
                    kv_params,
                    input=x,
                    wqkv=wqkv,
                    bias=bias,
                    input_row_offsets=input_row_offsets,
                    kv_collection=kv_collection,
                    layer_idx=layer_idx,
                    n_heads=n_heads,
                    input_scale=x_scales.to(x.device),
                    weight_scale=weight_scale.to(x.device),
                    quant_config=None,
                    _output_dim=_output_dim,
                )
            else:
                x, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    scales_type=weight_scale.dtype,
                    out_type=wqkv.dtype,
                )

            return _fused_qkv_ragged_matmul_scaled_float8(
                kv_params,
                input=x,
                wqkv=wqkv,
                bias=bias,
                input_row_offsets=input_row_offsets,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=n_heads,
                input_scale=x_scales.to(x.device),
                weight_scale=weight_scale.to(x.device),
                quant_config=quant_config,
                _output_dim=_output_dim,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for fused QKV matmul: {quant_config.format}"
            )


def quantized_fused_qkv_index_matmul(
    kv_params: KVCacheParams,
    index_kv_params: KVCacheParams,
    x: TensorValue,
    wqkv: TensorValue,
    kv_collection: PagedCacheValues,
    index_kv_collection: PagedCacheValues,
    layer_idx: TensorValue,
    input_row_offsets: TensorValue,
    n_heads: int,
    num_index_heads: int,
    idx_head_dim: int,
    quant_config: QuantConfig,
    weight_scale: TensorValue,
    prequantized: tuple[TensorValue, TensorValue] | None = None,
) -> tuple[TensorValue, TensorValue]:
    """Fuses MiniMax-M3's QKV and index-QK projections into one MXFP8 matmul.

    All five projections (``Q``, ``K``, ``V``, ``IndexQ``, ``IndexK``) read the
    same hidden state ``x``. This quantizes ``x`` once and runs a single
    block-scaled GEMM over the concatenated weights ``[Wq | Wk | Wv | Wiq |
    Wik]``, scattering ``K`` / ``V`` into ``kv_collection`` and ``IndexK`` into
    ``index_kv_collection`` while returning ``Q`` and ``IndexQ`` as two separate
    output tensors.

    The scatter rides the GEMM's own epilogue on both vendors; only the weight
    scale layout differs, so the interleave below is SM100-only.

    Only the MXFP8 dynamic-activation-quant format is supported; callers must
    gate on it (other formats keep the separate QKV + IndexQK matmuls).

    Args:
        kv_params: KVCacheParams for the MAIN (K, V) cache.
        index_kv_params: KVCacheParams for the INDEX (IndexK) cache.
        x: The input tensor of shape ``[total_seq_len, hidden_dim]``.
        wqkv: The concatenated ``[Wq | Wk | Wv | Wiq | Wik]`` weight tensor.
        kv_collection: The MAIN paged KV cache.
        index_kv_collection: The INDEX paged KV cache.
        layer_idx: The current layer index.
        input_row_offsets: Batch boundary offsets.
        n_heads: Number of main attention heads.
        num_index_heads: Number of index Q heads.
        idx_head_dim: Index head dimension (also the IndexK width).
        quant_config: The quantization configuration; must be MXFP8.
        weight_scale: The concatenated E8M0 weight scale tensor (pre-interleave).
        prequantized: ``(x_fp8, x_scales)`` for ``x``, skipping the quantize.
            Scales are the plain rank-2 ``[rows, hidden / 32]`` E8M0 layout,
            NOT the interleaved weight-scale one.

    Returns:
        A tuple ``(q, index_q)`` of bf16 tensors: ``q`` is
        ``[total_seq_len, q_dim]`` and ``index_q`` is
        ``[total_seq_len, iq_dim]``.
    """
    if quant_config.format != QuantFormat.MXFP8:
        raise ValueError(
            "quantized_fused_qkv_index_matmul only supports MXFP8, got"
            f" {quant_config.format}"
        )
    if prequantized is not None:
        x_fp8, x_scales = prequantized
        if not _is_amd_gpu():
            raise ValueError(
                "prequantized carries the plain rank-2 [rows, hidden/32] E8M0"
                " scale the CDNA4 kernel takes; the SM100 op wants the"
                " interleaved SF-atom layout, so it must quantize its own"
                " activation."
            )
        if (
            x_fp8.dtype != DType.float8_e4m3fn
            or x_scales.dtype != DType.float8_e8m0fnu
        ):
            raise ValueError(
                "prequantized must be (float8_e4m3fn, float8_e8m0fnu), got"
                f" ({x_fp8.dtype}, {x_scales.dtype})"
            )
        if x_fp8.shape != x.shape:
            raise ValueError(
                f"prequantized payload shape {x_fp8.shape} != x {x.shape}"
            )
        if x_scales.rank != 2 or x_scales.shape[0] != x.shape[0]:
            raise ValueError(
                "prequantized scales must be rank-2 [rows, hidden/32]; got"
                f" {x_scales.shape} for x {x.shape}"
            )
    else:
        x_fp8, x_scales = quantize_dynamic_block_scaled(
            x,
            sf_vector_size=32,
            scales_type=DType.float8_e8m0fnu,
            out_type=DType.float8_e4m3fn,
        )
    if not _is_amd_gpu():
        weight_scale = block_scales_interleave(
            weight_scale.to(x.device), sf_vector_size=32
        )
    return _fused_qkv_index_ragged_matmul_scaled_mxfp8(
        kv_params=kv_params,
        index_kv_params=index_kv_params,
        input=x_fp8,
        input_row_offsets=input_row_offsets,
        wqkv=wqkv,
        kv_collection=kv_collection,
        index_kv_collection=index_kv_collection,
        layer_idx=layer_idx,
        n_heads=n_heads,
        num_index_heads=num_index_heads,
        idx_head_dim=idx_head_dim,
        input_scale=x_scales.to(x.device),
        weight_scale=weight_scale,
    )


def quantized_grouped_matmul(
    x: TensorValue,
    weight: TensorValue,
    weight_scale: TensorValue,
    expert_start_indices: TensorValue,
    expert_ids: TensorValue,
    usage_stats: TensorValue,
    quant_config: QuantConfig,
) -> TensorValue:
    """Single entry point for quantized grouped matmuls (MoE).

    Dispatches to the appropriate kernel based on the quantization format.
    Handles weight dequant (MXFP4) or input quantization + transpose (FP8)
    internally.

    Args:
        x: The input tensor in bf16.
        weight: The weight tensor in storage layout
            (MXFP4: ``[E, out, in//2]``, FP8: ``[E, in, out]``).
        weight_scale: The weight scale tensor in storage layout.
        expert_start_indices: Starting index of each expert's token group.
        expert_ids: Expert identifier for each token group.
        usage_stats: Per-expert usage statistics. The MXFP4 path passes it
            straight to ``grouped_matmul_ragged``, currently the FP8 path
            copies it to CPU.
        quant_config: The quantization configuration.

    Returns:
        The grouped matmul output tensor in bf16.
    """
    match quant_config.format:
        case QuantFormat.MXFP4:
            dequanted = mxfp4_dequant(
                weight, weight_scale, out_type=DType.bfloat16
            )
            return grouped_matmul_ragged(
                x,
                dequanted,
                expert_start_indices,
                expert_ids,
                usage_stats,
            )
        case QuantFormat.MXFP6:
            dequanted = mxfp6_dequant(
                weight,
                weight_scale,
                fp6_format=quant_config.mxfp6_format,
                out_type=DType.bfloat16,
            )
            return grouped_matmul_ragged(
                x,
                dequanted,
                expert_start_indices,
                expert_ids,
                usage_stats,
            )
        case (
            QuantFormat.COMPRESSED_TENSORS_FP8
            | QuantFormat.FBGEMM_FP8
            | QuantFormat.BLOCKSCALED_FP8
        ):
            # Weight is stored [E, in, out] = [E, K, N]; transpose to the
            # [E, N, K] orientation both grouped FP8 kernels expect (K
            # innermost, transpose_b=True).
            weight_t = weight.transpose(1, 2)

            if (
                quant_config.weight_scale.is_rowwise
                and quant_config.input_scale.block_size is None
            ):
                # Rowwise (per-output-channel) weight scale + per-token dynamic
                # activation scale -- the compressed-tensors FP8-dynamic layout
                # (e.g. RedHatAI Llama-4-Scout FP8-dynamic). No block_size.
                #
                # Orientation:
                #   * weight_scale arrives [E, N, 1] (per output channel) from
                #     StackedMLP._init_weights, which is exactly what the kernel
                #     wants as b_scales -- so it is NOT transposed (unlike the
                #     block path, whose 2D-per-expert scale needs transposing).
                #   * per-token activation quant returns [1, total_tokens]; the
                #     kernel wants a_scales [total_tokens, 1], so transpose it.
                x_fp8, x_scales = quantize_dynamic_scaled_float8(
                    x,
                    quant_config.input_scale,
                    quant_config.weight_scale,
                    out_type=weight.dtype,
                    scales_type=DType.float32,
                )

                return _grouped_matmul_rowwise_dynamic_scaled_fp8(
                    x_fp8,
                    weight_t,
                    x_scales.transpose(0, 1),
                    weight_scale,
                    expert_start_indices,
                    expert_ids,
                    usage_stats.to(DeviceRef.CPU()),
                )

            assert quant_config.input_scale.block_size is not None
            input_block_size = quant_config.input_scale.block_size[1]

            scale_t = weight_scale.transpose(1, 2)

            x_fp8, x_scales = quantize_dynamic_scaled_float8(
                x,
                quant_config.input_scale,
                quant_config.weight_scale,
                group_size_or_per_token=input_block_size,
                out_type=weight.dtype,
                scales_type=quant_config.weight_scale.dtype,
            )

            return grouped_dynamic_scaled_fp8_matmul(
                x_fp8,
                weight_t,
                x_scales,
                scale_t,
                expert_start_indices,
                expert_ids,
                usage_stats.to(DeviceRef.CPU()),
                quant_config.input_scale,
                quant_config.weight_scale,
            )
        case _:
            raise ValueError(
                f"Unsupported quantization format for grouped matmul: {quant_config.format}"
            )
