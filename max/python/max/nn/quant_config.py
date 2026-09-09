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

"""Scaled quantization configuration data structures for models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from max.driver import accelerator_api
from max.dtype import DType
from max.graph import DeviceRef, Dim, DimLike, Shape, TensorType


class ScaleGranularity(Enum):
    """Specifies the granularity of the quantization scale factor.

    Determines whether a scale factor applies per-tensor, per-row (often for
    weights), per-column, or per-block within a tensor.
    """

    TENSOR = "tensor"
    """Per-tensor scaling."""

    ROWWISE = "rowwise"
    """Per-row scaling."""

    COLWISE = "colwise"
    """Per-column scaling."""

    BLOCK = "block"
    """Per-block scaling."""

    def __str__(self):
        return self.value


class ScaleOrigin(Enum):
    """Specifies whether the quantization scale is determined statically or dynamically."""

    STATIC = "static"
    """Scales are pre-computed and loaded with the model weights."""

    DYNAMIC = "dynamic"
    """Scales are computed at runtime based on the input data."""

    @property
    def is_dynamic(self) -> bool:
        """Whether the scale origin is dynamic."""
        return self == ScaleOrigin.DYNAMIC

    @property
    def is_static(self) -> bool:
        """Whether the scale origin is static."""
        return self == ScaleOrigin.STATIC


class QuantFormat(Enum):
    """Identifies the quantization format of a model checkpoint."""

    COMPRESSED_TENSORS_FP8 = "compressed-tensors-fp8"
    """FP8 quantization using the compressed-tensors format."""

    FBGEMM_FP8 = "fbgemm-fp8"
    """FP8 quantization using the FBGEMM format."""

    BLOCKSCALED_FP8 = "blockscaled-fp8"
    """FP8 quantization with block-level scaling."""

    MXFP8 = "mxfp8"
    """Microscaling FP8 (MX) quantization: ``float8_e4m3fn`` data with E8M0
    block scales at a 32-element K granularity. Uses the SM100 block-scaled
    tensor-core MMA (``KIND_MXF8F6F4``) rather than the 128-granularity
    blockwise-FP8 path."""

    NVFP4 = "nvfp4"
    """NVIDIA FP4 quantization format."""

    MXFP4 = "mxfp4"
    """Microscaling FP4 (MX) quantization format."""

    MXFP6 = "mxfp6"
    """Microscaling FP6 (MX) quantization: six-bit elements (E2M3 or E3M2)
    packed four-to-three-bytes, with E8M0 block scales at a 32-element K
    granularity. AMD CDNA4 only: routes to the ``f8f6f4`` block-scaled MFMA,
    which selects the operand format per operand. Chosen over MXFP4 where
    accuracy matters -- E2M3 carries the same 3 mantissa bits as FP8 E4M3, so
    A6W6 lands within ~1.3 dB of A8W8 where A4W4 loses ~13 dB."""

    INT8_W8A8 = "int8-w8a8"
    """Symmetric int8 W8A8: per-output-channel (rowwise) int8 weight scales and
    per-token (dynamic rowwise) int8 activation scales, both symmetric
    absmax/127. Weights are RTN-quantized at load (no pre-quantized checkpoint).
    Apple M5 only: routes to the int8 widening-MMA GEMM
    (``int8_matmul.mojo``)."""


@dataclass
class WeightScaleSpec:
    """Specifies how weights are scaled for scaled quantization."""

    granularity: ScaleGranularity
    """The :class:`~max.nn.quant_config.ScaleGranularity` of the weight scale factor application."""

    dtype: DType
    """The :class:`~max.dtype.DType` of the weight scale factor(s)."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the weight scale granularity is per-tensor."""
        return self.granularity == ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the weight scale granularity is row-wise."""
        return self.granularity == ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the weight scale granularity is column-wise."""
        return self.granularity == ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the weight scale granularity is block-wise."""
        return self.granularity == ScaleGranularity.BLOCK


@dataclass
class InputScaleSpec:
    """Specifies how input activations are scaled for scaled quantization."""

    granularity: ScaleGranularity
    """The :class:`~max.nn.quant_config.ScaleGranularity` of the input scale factor application."""

    origin: ScaleOrigin
    """The :class:`~max.nn.quant_config.ScaleOrigin` (static or dynamic) of the input scale factor."""

    dtype: DType
    """The :class:`~max.dtype.DType` of the input scale factor(s)."""

    activation_scale_ub: float | None = None
    """An optional upper bound for dynamic activation scaling."""

    block_size: tuple[int, int] | None = None
    """The :obj:`tuple[int, int]` of the block size for block-wise scaling."""

    def __post_init__(self):
        if self.granularity == ScaleGranularity.BLOCK:
            if self.block_size is None:
                raise ValueError(
                    "block_size must be specified for block-wise scaling"
                )
            if len(self.block_size) != 2:
                raise ValueError("block_size must be a tuple of two integers")

    @property
    def is_tensor(self) -> bool:
        """Whether the input scale granularity is per-tensor."""
        return self.granularity == ScaleGranularity.TENSOR

    @property
    def is_rowwise(self) -> bool:
        """Whether the input scale granularity is row-wise."""
        return self.granularity == ScaleGranularity.ROWWISE

    @property
    def is_colwise(self) -> bool:
        """Whether the input scale granularity is column-wise."""
        return self.granularity == ScaleGranularity.COLWISE

    @property
    def is_block(self) -> bool:
        """Whether the input scale granularity is block-wise."""
        return self.granularity == ScaleGranularity.BLOCK


@dataclass
class QuantConfig:
    """Configures scaled quantization settings for a layer or model section.

    For example, to configure NVFP4 block-scaled quantization for all layers
    in a 19-layer model:

    .. code-block:: python

        from max.dtype import DType
        from max.nn import QuantConfig, QuantFormat
        from max.nn.quant_config import (
            InputScaleSpec,
            ScaleGranularity,
            ScaleOrigin,
            WeightScaleSpec,
        )

        all_layers = set(range(19))

        input_spec = InputScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
            block_size=(1, 16),
        )
        weight_spec = WeightScaleSpec(
            granularity=ScaleGranularity.BLOCK,
            dtype=DType.float8_e4m3fn,
            block_size=(1, 8),
        )
        config = QuantConfig(
            input_scale=input_spec,
            weight_scale=weight_spec,
            mlp_quantized_layers=all_layers,
            attn_quantized_layers=all_layers,
            format=QuantFormat.NVFP4,
        )
    """

    input_scale: InputScaleSpec
    """:class:`~max.nn.quant_config.InputScaleSpec` for input activation scaling."""

    weight_scale: WeightScaleSpec
    """:class:`~max.nn.quant_config.WeightScaleSpec` for weight scaling."""

    mlp_quantized_layers: set[int]
    """Set of layer indices with quantized MLPs.

    MLPs are quantized on an all-or-nothing basis per layer: either all of
    ``gate_proj``, ``down_proj``, and ``up_proj`` are quantized, or all three
    remain in ``bfloat16``.
    """

    attn_quantized_layers: set[int]
    """Set of layer indices with quantized attention projections.

    Attention projections are quantized on an all-or-nothing basis per layer:
    either all of ``q_proj``, ``k_proj``, ``v_proj``, and ``o_proj`` are
    quantized, or all four remain in ``bfloat16``.
    """

    format: QuantFormat
    """The :class:`~max.nn.quant_config.QuantFormat` identifying the quantization format."""

    embedding_output_dtype: DType | None = None
    """The :class:`~max.dtype.DType` of the output from the embedding layer."""

    shared_experts_weight_dtype: DType | None = None
    """Weight storage dtype for MoE shared-expert MLPs when they differ from routed experts.

    When ``None``, shared experts use the same dtype and quantization as routed experts.
    When set (e.g. :class:`~max.dtype.DType.bfloat16` for mixed Kimi K2.6 NVFP4
    checkpoints), shared-expert linears omit ``quant_config`` while routed experts
    remain quantized.
    """

    bias_dtype: DType | None = None
    """The :class:`~max.dtype.DType` of bias weights."""

    can_use_fused_mlp: bool = False
    """Whether the quantization scales can be used with fused MLP operations."""

    can_use_fused_swiglu: bool = False
    """Whether to use the fused grouped matmul + SwiGLU + NVFP4/MXFP4/MXFP8 quant
    SM100 kernel for the MoE gate/up projection. When ``True``, the MoE layer
    pre-permutes ``gate_up_proj`` and its scales on the N axis
    (``sigma(2i)=i, sigma(2i+1)=D+i``) and dispatches the internal
    ``grouped_matmul_blocked_swiglu`` kernel wrapper. Defaults to ``False``
    so the chained (matmul -> BF16 -> SwiGLU+quant) path is unchanged."""

    scales_pre_interleaved: bool = False
    """Whether weight scales in the checkpoint are already stored in the 5D
    TCGEN-interleaved layout expected by the FP4 matmul kernel (NVFP4 only).
    Note that scales in the 5D TCGEN-interleaved layout are typically flattened
    to 2D ``[M, K//16]`` in the checkpoint."""

    _mxfp6_element_format: str = "e2m3"
    """Which OCP FP6 encoding an MXFP6 checkpoint holds. Read through
    :attr:`mxfp6_format`, which rejects the question for non-MXFP6 configs."""

    block_scaled_preshuffled_b: bool = False
    """Whether MXFP4 weight ``B`` is preshuffled into the 5D layout that the
    AMD preb kernel reads (produced by ``Shuffler.preshuffle_b_5d``). When
    True, ``MoEQuantized`` dispatches the grouped matmul to the
    ``block_scaled_grouped_matmul_amd_preb`` kernel variant; when False (default)
    it dispatches to the dense row-major ``block_scaled_grouped_matmul_amd``
    kernel. Must be set in lockstep with the weight loader actually
    applying the preshuffle (e.g. Kimi K2.5's
    ``weight_adapters.py:_shuffle_group``)."""

    @property
    def scales_granularity_mnk(self) -> tuple[int, int, int]:
        """The weight and input scale granularities on the M, N, and K axes."""
        m_input_granularity: int
        k_input_granularity: int
        if self.input_scale.is_block:
            input_block_size = self.input_scale.block_size
            assert input_block_size is not None
            m_input_granularity = input_block_size[0]
            k_input_granularity = input_block_size[1]
        elif self.input_scale.is_colwise:
            m_input_granularity = 1
            k_input_granularity = -1  # one scale shared by one token
        elif self.input_scale.is_tensor:
            m_input_granularity = -1
            k_input_granularity = -1
        else:
            raise ValueError("unsupported input scale granularity")

        n_weight_granularity: int
        k_weight_granularity: int
        if self.weight_scale.is_block:
            weight_block_size = self.weight_scale.block_size
            assert weight_block_size is not None
            n_weight_granularity = weight_block_size[0]
            k_weight_granularity = weight_block_size[1]
        elif self.weight_scale.is_rowwise:
            n_weight_granularity = 1
            k_weight_granularity = -1  # one scale shared by one row
        elif self.weight_scale.is_tensor:
            n_weight_granularity = -1
            k_weight_granularity = -1
        else:
            raise ValueError("unsupported weight scale granularity")

        assert k_input_granularity == k_weight_granularity, (
            "k_input_granularity and k_weight_granularity must be the same"
        )

        return (m_input_granularity, n_weight_granularity, k_input_granularity)

    @property
    def is_static(self) -> bool:
        """``True`` if this input scale is static."""
        return self.input_scale.origin == ScaleOrigin.STATIC

    @property
    def is_dynamic(self) -> bool:
        """``True`` if this input scale is dynamic."""
        return self.input_scale.origin == ScaleOrigin.DYNAMIC

    @property
    def is_nvfp4(self) -> bool:
        """``True`` if this config represents modelopt NVFP4."""
        return self.format == QuantFormat.NVFP4

    @property
    def is_mxfp4(self) -> bool:
        """Returns ``True`` if this config represents MXFP4 quantization."""
        return self.format == QuantFormat.MXFP4

    @property
    def is_mxfp6(self) -> bool:
        """Returns ``True`` if this config represents MXFP6 quantization."""
        return self.format == QuantFormat.MXFP6

    @property
    def is_mxfp8(self) -> bool:
        """Returns ``True`` if this config represents MXFP8 quantization."""
        return self.format == QuantFormat.MXFP8

    @property
    def is_fp4(self) -> bool:
        """``True`` if this config represents any FP4 variant (NVFP4 or MXFP4)."""
        return self.is_nvfp4 or self.is_mxfp4

    @property
    def mxfp6_format(self) -> str:
        """The FP6 element encoding, ``"e2m3"`` or ``"e3m2"``.

        E2M3 is the default: with 3 mantissa bits against E3M2's 2, it is the
        better choice for weights, whose dynamic range a per-32 block scale
        already absorbs.
        """
        if not self.is_mxfp6:
            raise ValueError(f"not an MXFP6 config: {self.format}")
        return self._mxfp6_element_format

    @property
    def is_mx(self) -> bool:
        """``True`` for any OCP microscaling format (MXFP4, MXFP6 or MXFP8).

        These share the E8M0 per-32-element block scale; NVFP4 is excluded, as
        it scales per 16 elements with an FP8 scale.
        """
        return self.is_mxfp4 or self.is_mxfp6 or self.is_mxfp8

    @property
    def mx_element_dtype(self) -> DType:
        """The dtype of one quantized element, for MX formats.

        This is the *logical* element type, not the storage type: sub-byte MX
        weights are stored as packed :obj:`~max.dtype.DType.uint8` bytes, and
        that spelling cannot say which format the bytes hold -- MXFP4 and MXFP6
        share it. Code that needs to branch on the format should branch on this
        instead of pairing ``dtype == DType.uint8`` with an ``is_mxfp6`` flag.

        Raises:
            ValueError: If this config is not an MX format.
        """
        if self.is_mxfp6:
            return (
                DType.float6_e3m2fn
                if self._mxfp6_element_format == "e3m2"
                else DType.float6_e2m3fn
            )
        if self.is_mxfp4:
            return DType.float4_e2m1fn
        if self.is_mxfp8:
            return DType.float8_e4m3fn
        raise ValueError(f"not an MX config: {self.format}")

    @property
    def is_int8_w8a8(self) -> bool:
        """``True`` if this config represents symmetric int8 W8A8."""
        return self.format == QuantFormat.INT8_W8A8

    def shared_experts_dtype(self, routed_weight_dtype: DType) -> DType:
        """Resolve weight dtype for MoE shared-expert MLPs."""
        if self.shared_experts_weight_dtype is not None:
            return self.shared_experts_weight_dtype
        return routed_weight_dtype

    def shared_experts_use_quant(self, routed_weight_dtype: DType) -> bool:
        """Whether shared experts use the same quantized weights as routed experts."""
        return (
            self.shared_experts_dtype(routed_weight_dtype)
            == routed_weight_dtype
        )

    def quantized_scales_type(
        self, quantized_shape: Shape, device_ref: DeviceRef
    ) -> TensorType:
        """The :class:`~max.graph.TensorType` of the scales tensor after dynamic quantization."""
        if self.is_nvfp4:
            return _nvmxf4f8_scales_type(
                quantized_shape, device_ref, DType.float8_e4m3fn, 16
            )
        elif self.is_mxfp4:
            return _mxfp4_scales_type(quantized_shape, device_ref)
        elif self.is_mxfp6:
            return _mxfp6_scales_type(quantized_shape, device_ref)
        elif self.is_mxfp8:
            return _mxfp8_scales_type(quantized_shape, device_ref)
        elif (
            self.input_scale.block_size is not None
            and self.input_scale.block_size == (1, 128)
        ):
            return _blockwise_fp8_scales_type(quantized_shape, device_ref)
        else:
            raise ValueError("Can not determine the quantized scales type")


def fp4_packed_k(in_dim: int, quant_config: QuantConfig | None) -> int:
    """Returns the packed K dimension for sub-byte MX weights, else ``in_dim``.

    FP4 packs two codes per byte and FP6 packs four codes per three bytes, so
    the on-disk K of a quantized weight is measured in bytes and differs from
    its logical K. Formats whose elements are a whole byte pass through.
    """
    if quant_config is None:
        return in_dim
    if quant_config.is_fp4:
        return in_dim // 2
    if quant_config.is_mxfp6:
        return in_dim * 3 // 4
    return in_dim


def ceildiv(n: DimLike, d: DimLike) -> Dim:
    """Returns the ceiling division of ``n`` by ``d``.

    Args:
        n: The numerator as a ``DimLike`` value.
        d: The denominator as a ``DimLike`` value.

    Returns:
        A :class:`~max.graph.Dim` equal to ceil(n / d).
    """
    return (Dim(n) + Dim(d) - Dim(1)) // Dim(d)


def _blockwise_fp8_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the blockwise FP8 scales tensor."""
    # Blockwise FP8 quantization uses a transposed layout for the scales tensor.
    return TensorType(
        dtype=DType.float32,
        shape=(ceildiv(quantized_shape[1], 128), quantized_shape[0]),
        device=device_ref,
    )


def _nvmxf4f8_scales_type(
    quantized_shape: Shape,
    device_ref: DeviceRef,
    scales_dtype: DType,
    sf_vector_size: int,
) -> TensorType:
    """Returns the TensorType of the NVFP4/MXFP4/MXFP8 scales tensor on NVIDIA
    GPUs."""
    # Nvidia NVFP4/MXFP4/MXFP8 format requires the scales tensor to be in a
    # 128x4 tiled layout. The follow constant needs to be in sync with those
    # defined in `max/kernels/src/linalg/fp4_utils.mojo`.
    #
    # References:
    # - https://docs.nvidia.com/cuda/cublas/#d-block-scaling-factors-layout

    SF_ATOM_M = [32, 4]
    SF_ATOM_K = 4
    SF_MN_GROUP_SIZE = SF_ATOM_M[0] * SF_ATOM_M[1]  # 128
    scales_dim_0 = ceildiv(quantized_shape[0], SF_MN_GROUP_SIZE)
    scales_dim_1 = ceildiv(quantized_shape[1], sf_vector_size * SF_ATOM_K)
    return TensorType(
        dtype=scales_dtype,
        shape=(
            scales_dim_0,
            scales_dim_1,
            SF_ATOM_M[0],
            SF_ATOM_M[1],
            SF_ATOM_K,
        ),
        device=device_ref,
    )


def _mxfp4_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the MXFP4 scales tensor."""
    api_name = accelerator_api()
    if api_name == "cuda":
        return _nvmxf4f8_scales_type(
            quantized_shape, device_ref, DType.float8_e8m0fnu, 32
        )
    elif api_name == "hip":
        return TensorType(
            dtype=DType.float8_e8m0fnu,
            shape=(quantized_shape[0], ceildiv(quantized_shape[1], 32)),
            device=device_ref,
        )
    else:
        raise ValueError(f"Unsupported accelerator API: {api_name}")


def _mxfp6_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the MXFP6 scales tensor.

    Identical to MXFP4's: the scale layout depends only on the 32-element block
    granularity, not on the width of the elements inside the block.
    """
    if accelerator_api() != "hip":
        raise ValueError("MXFP6 is only supported on AMD CDNA4 (gfx950)")
    return TensorType(
        dtype=DType.float8_e8m0fnu,
        shape=(quantized_shape[0], ceildiv(quantized_shape[1], 32)),
        device=device_ref,
    )


def _mxfp8_scales_type(
    quantized_shape: Shape, device_ref: DeviceRef
) -> TensorType:
    """Returns the TensorType of the MXFP8 scales tensor."""
    api_name = accelerator_api()
    if api_name == "cuda":
        return _nvmxf4f8_scales_type(
            quantized_shape, device_ref, DType.float8_e8m0fnu, 32
        )
    elif api_name == "hip":
        # Same plain rank-2 layout as MXFP4 on AMD: both formats scale groups of
        # 32 elements, and the difference (packed nibbles vs one byte) is in the
        # quantized operand, not its scales.
        return TensorType(
            dtype=DType.float8_e8m0fnu,
            shape=(quantized_shape[0], ceildiv(quantized_shape[1], 32)),
            device=device_ref,
        )
    else:
        raise ValueError(f"Unsupported accelerator API: {api_name}")
