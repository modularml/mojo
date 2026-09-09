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
"""Weight adapters for Qwen3.5 models."""

from __future__ import annotations

from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.graph.weights.weights import Shape
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_dtype_cast,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model_config import Qwen3_5Config

# -----------------------------------------------------------------------
# Vision encoder weight mapping
# -----------------------------------------------------------------------
# Checkpoint key prefix for the vision encoder.
_VISION_CHECKPOINT_PREFIX = "model.visual."
# MAX module prefix used when loading vision weights into VisionTransformer.
_VISION_MAX_PREFIX = "vision_encoder."

# -----------------------------------------------------------------------
# Language-model weight prefix map (longest-match wins).
# Order matters: the multimodal prefix must come before the plain one.
# -----------------------------------------------------------------------
_LM_PREFIX_MAP = {
    "model.language_model.": "",  # multimodal checkpoint
    "model.": "",  # text-only checkpoint
}

# Weight prefixes to skip entirely.
_SKIP_PREFIXES = ("mtp.",)


def _convert_patch_embed_weight(weight_data: WeightData) -> WeightData:
    """Reshape Conv3D patch embed weight → Linear weight.

    HuggingFace stores the patch projection as a 5-D Conv3D tensor
    `(out_channels, in_channels, kT, kH, kW)`. The MAX VisionPatchEmbed
    uses an equivalent Linear layer whose weight is 2-D
    `(out_channels, in_channels * kT * kH * kW)`.
    """
    if len(weight_data.shape) != 5:
        return weight_data
    out_c, in_c, kt, kh, kw = weight_data.shape
    flat = in_c * kt * kh * kw
    buf = Buffer.from_dlpack(weight_data.data).view(
        dtype=weight_data.dtype, shape=(int(out_c), int(flat))
    )
    return WeightData(
        data=buf,
        name=weight_data.name,
        dtype=weight_data.dtype,
        shape=Shape([out_c, flat]),
        quantization_encoding=weight_data.quantization_encoding,
    )


def convert_qwen3_5_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs: object,
) -> dict[str, WeightData]:
    """Convert Qwen3.5 checkpoint weights to MAX format.

    Handles both text-only and multimodal checkpoints:
    - Vision encoder weights (`model.visual.*`) are remapped to
      `vision_encoder.*` and the patch-embed projection is reshaped
      from Conv3D to Linear format.
    - Language-model weights are stripped of their HuggingFace prefix
      (`model.language_model.` or `model.`).
    - Multi-token prediction (`mtp.*`) weights are skipped.

    Args:
        state_dict: The raw checkpoint weights.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: Pipeline configuration.

    Returns:
        The transformed weights for the MAX Qwen3.5 model.
    """
    new_state_dict: dict[str, WeightData] = {}

    for safetensor_name, value in state_dict.items():
        # ── Skip unwanted weights ──────────────────────────────────────
        if any(safetensor_name.startswith(p) for p in _SKIP_PREFIXES):
            continue

        weight_data = value.data()

        # ── Vision encoder ─────────────────────────────────────────────
        if safetensor_name.startswith(_VISION_CHECKPOINT_PREFIX):
            max_name = (
                _VISION_MAX_PREFIX
                + safetensor_name[len(_VISION_CHECKPOINT_PREFIX) :]
            )
            # Remap flat pos_embed → nested BilinearInterpolationPositionEmbedding
            max_name = max_name.replace(
                "vision_encoder.pos_embed.",
                "vision_encoder.pos_embed.embedding.",
                1,
            )
            # Remap stacked QKV in vision attention to StackedLinear namespace.
            max_name = max_name.replace("attn.qkv.", "attn.qkv_proj.")
            # Conv3D → Linear for the patch projection
            if max_name == "vision_encoder.patch_embed.proj.weight":
                weight_data = _convert_patch_embed_weight(weight_data)
            # NOTE: If vision logits diverge from HuggingFace, verify that
            # attention output projection bias (attn.proj.bias) is loading
            # correctly. HuggingFace Qwen3_5VisionAttention uses
            # nn.Linear(dim, dim) which defaults to bias=True; our
            # DistributedVisionWindowAttention sets has_bias=True to match,
            # so the checkpoint key vision_encoder.blocks.X.attn.proj.bias
            # should be present and load cleanly via this prefix remap.
            new_state_dict[max_name] = weight_data
            continue

        # ── Language model ─────────────────────────────────────────────
        max_name = safetensor_name
        for before, after in _LM_PREFIX_MAP.items():
            if max_name.startswith(before):
                max_name = after + max_name[len(before) :]
                break

        # Cast linear attention scalar weights and gated RMSNorm weight to
        # float32.  dt_bias is stored as bfloat16 but model expects float32.
        # A_log is already stored as float32; the cast is a no-op but kept
        # for safety in case older checkpoints stored it as bfloat16.
        # linear_attn.norm.weight backs a float32 RMSNorm inside
        # GatedDeltaNet, but the global float32→bfloat16 cast applied below
        # would otherwise downcast it.
        if max_name.endswith(
            (".dt_bias", ".A_log", ".linear_attn.norm.weight")
        ):
            weight_data = weight_data.astype(DType.float32)

        # Reshape conv1d weight from [conv_dim, kernel_size] to
        # [conv_dim, 1, kernel_size] for depthwise conv
        if max_name.endswith(".conv1d.weight") and len(weight_data.shape) == 2:
            d0, d1 = weight_data.shape
            buf = Buffer.from_dlpack(weight_data.data).view(
                dtype=weight_data.dtype, shape=(int(d0), 1, int(d1))
            )
            weight_data = WeightData(
                data=buf,
                name=weight_data.name,
                dtype=weight_data.dtype,
                shape=Shape([d0, 1, d1]),
                quantization_encoding=weight_data.quantization_encoding,
            )

        new_state_dict[max_name] = weight_data

    # Apply dtype casting when the pipeline resolved a cast between float32
    # and bfloat16 (e.g. due to mixed-dtype checkpoint files). This mirrors
    # the same logic in llama3/weight_adapters.py.
    #
    # Qwen3.5 checkpoints have a small number of intentionally-float32
    # tensors (A_log, norm.weight). Those are already in float32 so they
    # are unaffected by either direction of casting.
    # TODO(MXF-517): this should be resolved by the ArchConfig, not the adapter.
    cast_from, cast_to = _select_dtype_cast(
        pipeline_config.model, Qwen3_5Config.DEFAULT_ENCODING
    )
    if cast_from:
        assert cast_to, (
            "Invalid configuration: cast_to is not set but "
            "cast_from is set. This should not happen."
        )
        cast_from_dtype = supported_encoding_dtype(cast_from)
        cast_to_dtype = supported_encoding_dtype(cast_to)
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == cast_from_dtype and not key.startswith(
                _VISION_MAX_PREFIX
            ):
                new_state_dict[key] = weight_data.astype(cast_to_dtype)

    return new_state_dict
