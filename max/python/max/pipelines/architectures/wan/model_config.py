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

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from pydantic import Field


def build_wan_fp8_quant_config() -> QuantConfig:
    """Static per-tensor W8A8 FP8 config for Comfy-Org ``fp8_scaled`` Wan DiT.

    The Comfy-Org checkpoints store every linear weight as ``float8_e4m3fn``
    with a scalar ``scale_weight`` and a scalar ``scale_input`` (static input
    quantization). This maps onto MAX's ``matmul_static_scaled_float8`` path
    via :class:`~max.nn.quant_config.QuantConfig` with per-tensor scales.

    Biases, norms, modulation tables, and the patch-embedding Conv3d stay in
    ``bfloat16`` — only the linear projections are quantized.
    """
    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            origin=ScaleOrigin.STATIC,
            dtype=DType.float32,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            dtype=DType.float32,
        ),
        # Wan applies the quant config directly to each Linear rather than
        # gating by layer index, so these sets are unused; pass empty.
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
        bias_dtype=DType.bfloat16,
    )


class WanConfigBase(MAXModelConfigBase):
    # Defaults mirror diffusers WanTransformer3DModel constructor defaults.
    patch_size: tuple[int, int, int] = (1, 2, 2)
    num_attention_heads: int = 40
    attention_head_dim: int = 128
    in_channels: int = 16
    out_channels: int = 16
    text_dim: int = 4096
    freq_dim: int = 256
    ffn_dim: int = 13824
    num_layers: int = 40
    cross_attn_norm: bool = True
    qk_norm: str | None = "rms_norm_across_heads"
    eps: float = 1e-6
    image_dim: int | None = None
    added_kv_proj_dim: int | None = None
    rope_max_seq_len: int = 1024
    pos_embed_seq_len: int | None = None
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)
    quant_config: QuantConfig | None = None
    """Static per-tensor FP8 quantization config, populated when the
    transformer encoding is ``float8_e4m3fn``. ``None`` for bfloat16."""


class WanConfig(WanConfigBase):
    @staticmethod
    def generate(
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> "WanConfig":
        init_dict = {
            key: value
            for key, value in config_dict.items()
            if key in WanConfigBase.__annotations__
        }
        # The DiT working dtype (activations, norms, modulation) stays
        # bfloat16 even under FP8 — only linear weights are quantized — so
        # don't let the float8 encoding flip the model dtype.
        quant_config: QuantConfig | None = None
        if encoding == "float8_e4m3fn":
            model_dtype = DType.bfloat16
            quant_config = build_wan_fp8_quant_config()
        else:
            model_dtype = supported_encoding_dtype(encoding)
        init_dict.update(
            {
                "dtype": model_dtype,
                "device": DeviceRef.from_device(devices[0]),
                "quant_config": quant_config,
            }
        )
        return WanConfig(**init_dict)
