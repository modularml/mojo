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
"""Config for Nemotron-H hybrid Mamba-2 + attention + MLP models."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.kv_cache import cache_dtype_for_encoding
from max.pipelines.lib import KVCacheConfig, MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces import (
    ArchConfigWithKVCache,
    ArchConfigWithStoredKVParams,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers.models.auto.configuration_auto import AutoConfig

logger = logging.getLogger("max.pipelines")


def build_fp8_quant_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
) -> QuantConfig | None:
    """Build a per-tensor static FP8 ``QuantConfig`` for a modelopt checkpoint.

    Nemotron-H FP8 (modelopt ``quant_algo=FP8``) stores, per quantized Linear,
    an ``F8_E4M3`` weight plus a scalar ``weight_scale`` and ``input_scale``
    (both fp32) — i.e. per-tensor static activation+weight scaling. Returns
    ``None`` if the checkpoint is not FP8 (no ``weight_scale`` tensors).

    The generic ``mlp_quantized_layers`` / ``attn_quantized_layers`` machinery
    is bypassed (it hard-codes ``self_attn`` / ``mlp.{gate,up,down}`` names that
    don't fit Nemotron's ``backbone.layers.{i}.mixer.*``); FP8 is wired
    per-module in the nn.Module via the config's FP8 layer sets.
    """
    has_fp8 = any(v.dtype == DType.float8_e4m3fn for v in state_dict.values())
    if not has_fp8:
        return None

    # The weight_scale / input_scale tensors are scalar fp32 (per-tensor static
    # scaling); their *storage* dtype (fp32) is what the scale-spec dtype must
    # report — NOT the fp8 dtype the activations/weights are quantized to.
    scale_dtype = DType.float32
    for name, v in state_dict.items():
        if name.endswith("weight_scale"):
            scale_dtype = v.dtype
            break
    input_scale_dtype = DType.float32
    for name, v in state_dict.items():
        if name.endswith("input_scale"):
            input_scale_dtype = v.dtype
            break

    return QuantConfig(
        input_scale=InputScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            origin=ScaleOrigin.STATIC,
            dtype=input_scale_dtype,
        ),
        weight_scale=WeightScaleSpec(
            granularity=ScaleGranularity.TENSOR,
            dtype=scale_dtype,
        ),
        mlp_quantized_layers=set(),
        attn_quantized_layers=set(),
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )


def resolve_attention_head_dim(huggingface_config: AutoConfig) -> int:
    """Attention head dim, matching HF ``NemotronHAttention``.

    HF uses ``getattr(config, "head_dim", hidden_size // num_attention_heads)``.
    The real Nemotron checkpoint sets both ``head_dim`` and
    ``attention_head_dim`` to the same value; prefer ``head_dim`` to match the
    reference exactly, falling back to ``attention_head_dim`` then the derived
    default.
    """
    head_dim = getattr(huggingface_config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    attn_head_dim = getattr(huggingface_config, "attention_head_dim", None)
    if attn_head_dim is not None:
        return attn_head_dim
    return (
        huggingface_config.hidden_size // huggingface_config.num_attention_heads
    )


def parse_hybrid_pattern(pattern: str) -> list[str]:
    """Map a Nemotron-H ``hybrid_override_pattern`` to per-layer kinds.

    ``M`` -> ``"mamba"``, ``*`` -> ``"attention"``, ``-`` -> ``"mlp"``,
    ``E`` -> ``"moe"`` (the Nemotron-3 MoE hybrids, e.g. 30B-A3B).
    """
    kinds: list[str] = []
    for ch in pattern:
        if ch == "M":
            kinds.append("mamba")
        elif ch == "*":
            kinds.append("attention")
        elif ch == "-":
            kinds.append("mlp")
        elif ch == "E":
            kinds.append("moe")
        else:
            raise ValueError(
                f"invalid hybrid_override_pattern character {ch!r}; "
                "expected 'M', '*', '-', or 'E'"
            )
    return kinds


@dataclass(kw_only=True)
class NemotronHConfig(ArchConfigWithStoredKVParams, ArchConfigWithKVCache):
    """Configuration for a Nemotron-H (``nemotron_h``) hybrid decoder.

    Nemotron-H interleaves Mamba-2 mixers, NoPE GQA attention, and relu2
    (non-gated) MLP blocks per ``hybrid_override_pattern``. There is no
    rotary embedding (attention is NoPE; position information flows through
    the SSM). FP8 (modelopt per-tensor static) is applied per-module to the
    Mamba in/out projections and the MLP up/down projections, honoring the
    checkpoint's ``exclude_modules`` list; attention, conv1d, norms, and
    lm_head stay bf16.
    """

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float8_e4m3fn",
    }

    # Core dims
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    layer_norm_epsilon: float
    max_seq_len: int
    dtype: DType
    devices: list[DeviceRef]
    tie_word_embeddings: bool = False

    # Per-layer kinds parsed from hybrid_override_pattern.
    layer_kinds: list[str] = field(default_factory=list)

    # Attention (NoPE GQA)
    num_attention_heads: int
    num_key_value_heads: int
    attention_head_dim: int
    attention_bias: bool = False

    # MLP
    intermediate_size: int
    mlp_hidden_act: str = "relu2"
    mlp_bias: bool = False

    # MoE (Nemotron-3 hybrids with ``E`` layers, e.g. 30B-A3B). These defaults
    # leave the dense 4B/8B variants (no ``moe`` layers) unaffected.
    num_experts: int = 0
    # Mirrors the HF config key name; feeds the MAX-side
    # ``num_experts_per_token`` MoE param. The ``_tok``/``_token`` split is
    # deliberate.
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    moe_shared_expert_intermediate_size: int = 0
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True

    # Mamba-2 mixer
    mamba_num_heads: int
    mamba_head_dim: int
    n_groups: int
    ssm_state_size: int
    conv_kernel: int
    chunk_size: int
    use_conv_bias: bool = True
    mamba_proj_bias: bool = False
    time_step_limit: tuple[float, float] = (0.0, float("inf"))

    # KV cache (full-attention layers only)
    kv_params: KVCacheParams

    # FP8: set of (kind, layer_idx) modules quantized to FP8 per-tensor static.
    # ``fp8_mamba_layers``: mamba layers whose in/out_proj are FP8.
    # ``fp8_mlp_layers``: MLP layers whose up/down_proj are FP8.
    # ``fp8_moe_layers``: MoE layers whose routed + shared expert up/down_proj
    # are FP8 (the 30B-A3B hybrid). The routed-expert grouped matmul consumes
    # the FP8 weight stack directly (weight-only W8A16); the shared expert runs
    # the dense FP8 Linear path.
    fp8_mamba_layers: set[int] = field(default_factory=set)
    fp8_mlp_layers: set[int] = field(default_factory=set)
    fp8_moe_layers: set[int] = field(default_factory=set)
    is_fp8: bool = False

    quantization_encoding: SupportedEncoding | None = None

    @property
    def mamba_intermediate_size(self) -> int:
        return self.mamba_num_heads * self.mamba_head_dim

    @property
    def conv_dim(self) -> int:
        return (
            self.mamba_intermediate_size
            + 2 * self.n_groups * self.ssm_state_size
        )

    @property
    def mamba_in_proj_out(self) -> int:
        # [gate=intermediate, hidden_states_B_C=conv_dim, dt=nheads]
        return (
            self.mamba_intermediate_size + self.conv_dim + self.mamba_num_heads
        )

    @property
    def attention_q_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def attention_kv_dim(self) -> int:
        return self.num_key_value_heads * self.attention_head_dim

    @property
    def mamba_layer_indices(self) -> list[int]:
        return [i for i, k in enumerate(self.layer_kinds) if k == "mamba"]

    @property
    def attention_layer_indices(self) -> list[int]:
        return [i for i, k in enumerate(self.layer_kinds) if k == "attention"]

    def populate_fp8_layers(self, state_dict: Mapping[str, WeightData]) -> None:
        """Record which mamba/MLP layers are FP8 from the checkpoint.

        A Linear is FP8 iff its ``weight_scale`` is present in the checkpoint
        (the exact inverse of the modelopt ``exclude_modules`` list). Names are
        the post-adapter MAX names: ``blocks.{i}.mixer.{in_proj,out_proj,
        up_proj,down_proj}.weight_scale``.
        """
        fp8_mamba: set[int] = set()
        fp8_mlp: set[int] = set()
        fp8_moe: set[int] = set()
        for name in state_dict:
            if not name.endswith("weight_scale"):
                continue
            parts = name.split(".")
            # blocks.{i}.mixer.{proj}.weight_scale
            if len(parts) >= 5 and parts[0] == "blocks":
                li = int(parts[1])
                proj = parts[3]
                # in_proj is split into in_proj_{gate,hidden_BC,dt}.
                if proj.startswith("in_proj") or proj == "out_proj":
                    fp8_mamba.add(li)
                elif proj in ("up_proj", "down_proj"):
                    fp8_mlp.add(li)
                # MoE routed/shared experts nest one level deeper:
                # blocks.{i}.mixer.experts.{j}.{up,down}_proj.weight_scale and
                # blocks.{i}.mixer.shared_experts.{up,down}_proj.weight_scale.
                elif proj in ("experts", "shared_experts"):
                    fp8_moe.add(li)
        self.fp8_mamba_layers = fp8_mamba
        self.fp8_mlp_layers = fp8_mlp
        self.fp8_moe_layers = fp8_moe
        self.is_fp8 = bool(fp8_mamba or fp8_mlp or fp8_moe)
        logger.info(
            f"Nemotron-H FP8: {len(fp8_mamba)} mamba layers,"
            f" {len(fp8_mlp)} MLP layers, {len(fp8_moe)} MoE layers quantized"
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        return huggingface_config.num_hidden_layers

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        max_seq_len = model_config.max_length
        if max_seq_len:
            return max_seq_len
        return huggingface_config.max_position_embeddings

    @staticmethod
    def construct_kv_params(
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
        *,
        allow_kv_head_replication: bool = False,
    ) -> KVCacheParams:
        """Allocate KV cache only for the (4) full-attention layers.

        The forward pass maps each attention layer to a sequential KV cache
        index (0, 1, 2, ...), independent of the absolute layer index.

        The reference configuration uses an FP8 KV cache for this checkpoint.
        Select the same default while preserving explicit cache-format
        overrides and non-FP8 model behavior.
        """
        resolved_encoding = _select_quantization_encoding(
            pipeline_config.model, NemotronHConfig.DEFAULT_ENCODING
        )
        if (
            resolved_encoding == "float8_e4m3fn"
            and kv_cache_config.kv_cache_format is None
        ):
            cache_dtype = DType.float8_e4m3fn
        kinds = parse_hybrid_pattern(huggingface_config.hybrid_override_pattern)
        num_attention_layers = sum(1 for k in kinds if k == "attention")
        return kv_cache_config.to_params(
            allow_kv_head_replication=allow_kv_head_replication,
            dtype=cache_dtype,
            n_kv_heads=huggingface_config.num_key_value_heads,
            head_dim=resolve_attention_head_dim(huggingface_config),
            num_layers=num_attention_layers,
            devices=devices,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> NemotronHConfig:
        """``ArchConfig`` protocol entry point.

        Derives dtype / devices / KV params from ``model_config`` and delegates
        to :meth:`from_hf`. ``model.py`` calls :meth:`from_hf` directly during
        graph build (it already has the resolved KV params); this method exists
        so the pipeline config-resolution / memory-estimation path can build the
        arch config from a ``PipelineConfig`` alone.
        """
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                "HuggingFace config is required for Nemotron-H but could not "
                "be loaded; ensure config.json is present."
            )
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs
        ]
        kv_params = cls.construct_kv_params(
            huggingface_config=huggingface_config,
            pipeline_config=pipeline_config,
            devices=device_refs,
            kv_cache_config=model_config.kv_cache,
            cache_dtype=cache_dtype_for_encoding(
                quantization_encoding,
                model_config.kv_cache.kv_cache_format,
            ),
        )
        return cls.from_hf(
            pipeline_config,
            huggingface_config,
            dtype,
            kv_params,
            device_refs,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def from_hf(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        dtype: DType,
        kv_params: KVCacheParams,
        devices: list[DeviceRef],
        *,
        max_seq_len: int,
    ) -> NemotronHConfig:
        # The model (activation / embedding / norm / attention) dtype is always
        # bf16; FP8 is applied PER-MODULE to specific Linears via quant_config,
        # NOT model-wide. So ignore an fp8 resolved encoding for the model dtype
        # — otherwise the embedding and every non-quantized weight would be
        # declared fp8 and fail to load the bf16 checkpoint tensors.
        if dtype == DType.float8_e4m3fn:
            dtype = DType.bfloat16
        kinds = parse_hybrid_pattern(huggingface_config.hybrid_override_pattern)
        quantization_encoding = _select_quantization_encoding(
            pipeline_config.model, cls.DEFAULT_ENCODING
        )
        return cls(
            hidden_size=huggingface_config.hidden_size,
            vocab_size=huggingface_config.vocab_size,
            num_hidden_layers=huggingface_config.num_hidden_layers,
            layer_norm_epsilon=huggingface_config.layer_norm_epsilon,
            max_seq_len=max_seq_len,
            dtype=dtype,
            devices=devices,
            tie_word_embeddings=getattr(
                huggingface_config, "tie_word_embeddings", False
            ),
            layer_kinds=kinds,
            num_attention_heads=huggingface_config.num_attention_heads,
            num_key_value_heads=huggingface_config.num_key_value_heads,
            attention_head_dim=resolve_attention_head_dim(huggingface_config),
            attention_bias=getattr(huggingface_config, "attention_bias", False),
            intermediate_size=huggingface_config.intermediate_size,
            mlp_hidden_act=getattr(
                huggingface_config, "mlp_hidden_act", "relu2"
            ),
            mlp_bias=getattr(huggingface_config, "mlp_bias", False),
            # MoE fields (guarded: dense 4B/8B configs lack them and keep the
            # dataclass defaults, so their code paths are unaffected).
            num_experts=getattr(huggingface_config, "n_routed_experts", 0),
            num_experts_per_tok=getattr(
                huggingface_config, "num_experts_per_tok", 0
            ),
            moe_intermediate_size=getattr(
                huggingface_config, "moe_intermediate_size", 0
            ),
            moe_shared_expert_intermediate_size=getattr(
                huggingface_config, "moe_shared_expert_intermediate_size", 0
            ),
            routed_scaling_factor=getattr(
                huggingface_config, "routed_scaling_factor", 1.0
            ),
            norm_topk_prob=getattr(huggingface_config, "norm_topk_prob", True),
            mamba_num_heads=huggingface_config.mamba_num_heads,
            mamba_head_dim=huggingface_config.mamba_head_dim,
            n_groups=huggingface_config.n_groups,
            ssm_state_size=huggingface_config.ssm_state_size,
            conv_kernel=huggingface_config.conv_kernel,
            chunk_size=huggingface_config.chunk_size,
            use_conv_bias=getattr(huggingface_config, "use_conv_bias", True),
            mamba_proj_bias=getattr(
                huggingface_config, "mamba_proj_bias", False
            ),
            kv_params=kv_params,
            quantization_encoding=quantization_encoding,
        )
