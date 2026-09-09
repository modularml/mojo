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
"""Config for Mamba models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights import WeightData
from max.nn.kv_cache import KVCacheParams, MHAKVCacheParams
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnLogits
from max.pipelines.lib import (
    MAXModelConfig,
    PipelineConfig,
    parse_quant_config,
    upper_bounded_default,
)
from max.pipelines.lib.config.model_config import _select_quantization_encoding
from max.pipelines.lib.interfaces import ArchConfigWithKVCache
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override

logger = logging.getLogger("max.pipelines")


@dataclass
class SSMStateCacheParams:
    """Parameters for SSM state cache memory estimation.

    SSM cache is fixed-size per batch element (no sequence-length scaling):
      conv_state: num_layers * intermediate_size * conv_kernel * dtype_bytes
      ssm_state:  num_layers * intermediate_size * d_state * dtype_bytes
    """

    num_layers: int
    intermediate_size: int
    d_state: int
    conv_kernel: int
    dtype: DType

    @property
    def per_element_bytes(self) -> int:
        return (
            self.num_layers
            * self.intermediate_size
            * (self.conv_kernel + self.d_state)
            * self.dtype.size_in_bytes
        )

    def estimated_memory_size(
        self,
        available_cache_memory: int,
        max_batch_size: int,
        max_seq_len: int,
    ) -> int:
        """Total SSM cache bytes. Independent of sequence length."""
        return self.per_element_bytes * max_batch_size

    def compute_max_seq_len_fitting_in_cache(
        self, cache_memory: int
    ) -> int | None:
        """SSM cache doesn't scale with seq length — no constraint."""
        return None


@dataclass(kw_only=True)
class MambaConfig(ArchConfigWithKVCache):
    """Model configuration for Mamba graph construction/execution."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "float32"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    # Core architecture fields
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    max_seq_len: int
    dtype: DType
    devices: list[DeviceRef]

    # SSM-specific fields
    d_state: int
    dt_rank: int | str | None = None
    conv_kernel: int = 4
    x_proj_dim: int | None = None

    # Normalization
    rms_norm_eps: float | None = None

    # Toggles
    use_bias: bool = False
    use_conv_bias: bool = True
    residual_in_fp32: bool = True
    tie_word_embeddings: bool = True

    # Output configuration
    return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN
    quant_config: QuantConfig | None = None
    use_subgraphs: bool = True
    data_parallel_degree: int = 1

    # Expand factor (needed for intermediate_size derivation)
    expand: int = 2

    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_kv_params(self) -> KVCacheParams:
        """Return minimal dummy KV cache params.

        Mamba uses SSM state caching, not attention KV cache, but the
        pipeline interfaces require KVCacheParams for memory estimation
        and PagedKVCacheManager initialization.  The tiny dimensions
        (1 head, 1 dim, 1 layer) make the allocation negligible.
        """
        return MHAKVCacheParams(
            dtype=self.dtype,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=self.devices,
            page_size=128,
        )

    def get_ssm_cache_params(self) -> SSMStateCacheParams:
        return SSMStateCacheParams(
            num_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            d_state=self.d_state,
            conv_kernel=self.conv_kernel,
            dtype=self.dtype,
        )

    @staticmethod
    def get_num_layers(huggingface_config: AutoConfig) -> int:
        num_layers = getattr(
            huggingface_config, "num_hidden_layers", None
        ) or getattr(huggingface_config, "n_layer", 64)
        assert num_layers is not None
        return int(num_layers)

    @staticmethod
    def calculate_max_seq_len(
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        try:
            return upper_bounded_default(
                upper_bound=getattr(
                    huggingface_config, "max_position_embeddings", 2048
                ),
                default=model_config.max_length,
            )
        except ValueError as e:
            raise ValueError(
                "Unable to infer max_length for Mamba, the provided "
                f"max_length ({model_config.max_length}) exceeds the "
                f"model's max_position_embeddings "
                f"({getattr(huggingface_config, 'max_position_embeddings', 2048)})."
            ) from e

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded."
            )
        return cls.initialize_from_config(
            pipeline_config,
            huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )

    @classmethod
    def initialize_from_config(
        cls,
        pipeline_config: PipelineConfig,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        dtype = supported_encoding_dtype(quantization_encoding)
        n_devices = len(pipeline_config.model.device_specs)

        device_refs = [
            DeviceRef(spec.device_type, spec.id)
            for spec in model_config.device_specs[:n_devices]
        ]

        # Map model architecture fields.
        # hidden_size can come from d_model or hidden_size.
        _hidden = getattr(huggingface_config, "hidden_size", None) or getattr(
            huggingface_config, "d_model", 2560
        )
        hidden_size: int = int(_hidden) if _hidden is not None else 2560

        expand: int = int(getattr(huggingface_config, "expand", 2))

        # intermediate_size can come from d_inner, d_intermediate, or
        # intermediate_size.  Falls back to expand * hidden_size.
        intermediate_size_raw = (
            getattr(huggingface_config, "intermediate_size", None)
            or getattr(huggingface_config, "d_inner", None)
            or getattr(huggingface_config, "d_intermediate", None)
        )
        intermediate_size: int = (
            int(intermediate_size_raw)
            if intermediate_size_raw
            else expand * hidden_size
        )

        num_hidden_layers = cls.get_num_layers(huggingface_config)

        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=huggingface_config.vocab_size,
            max_seq_len=max_seq_len,
            dtype=dtype,
            devices=device_refs,
            # SSM-specific
            d_state=getattr(huggingface_config, "state_size", 16),
            dt_rank=getattr(huggingface_config, "time_step_rank", None),
            conv_kernel=getattr(huggingface_config, "conv_kernel", 4),
            x_proj_dim=getattr(huggingface_config, "x_proj_dim", None),
            # Normalization
            rms_norm_eps=getattr(huggingface_config, "layer_norm_epsilon", None)
            or getattr(huggingface_config, "rms_norm_eps", None),
            # Toggles
            use_bias=getattr(huggingface_config, "use_bias", False),
            use_conv_bias=getattr(huggingface_config, "use_conv_bias", True),
            residual_in_fp32=getattr(
                huggingface_config, "residual_in_fp32", True
            ),
            tie_word_embeddings=getattr(
                huggingface_config, "tie_embeddings", True
            ),
            expand=expand,
            use_subgraphs=pipeline_config.model.use_subgraphs,
            data_parallel_degree=pipeline_config.model.data_parallel_degree,
            quantization_encoding=quantization_encoding,
        )

    def finalize(
        self,
        huggingface_config: AutoConfig,
        state_dict: dict[str, WeightData],
        return_logits: ReturnLogits,
    ) -> None:
        """Set parameters that require introspecting the state dict."""
        quant_config = parse_quant_config(
            huggingface_config, state_dict, self.dtype
        )
        self.quant_config = quant_config

        # Detect tie_word_embeddings from weights if lm_head is absent.
        if "tie_word_embeddings" in huggingface_config:
            self.tie_word_embeddings = huggingface_config.tie_word_embeddings
        else:
            self.tie_word_embeddings = (
                getattr(huggingface_config, "tie_word_embeddings", False)
                or "lm_head.weight" not in state_dict
            )

        self.return_logits = return_logits

    @staticmethod
    def help() -> dict[str, str]:
        return {}
