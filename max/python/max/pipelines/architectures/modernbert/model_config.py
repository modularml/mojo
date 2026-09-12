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
"""Config for ModernBERT models."""

from __future__ import annotations

from dataclasses import dataclass

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class ModernBertConfig(ArchConfigWithBoundedMaxSeqLen, ArchConfig):
    """Configuration for ModernBERT models."""

    dtype: DType
    device: DeviceRef
    pool_embeddings: bool
    huggingface_config: AutoConfig
    max_seq_len: int

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        quantization_encoding = model_config.quantization_encoding
        if quantization_encoding is None:
            raise ValueError("quantization_encoding must not be None")
        if len(model_config.device_specs) != 1:
            raise ValueError(
                "ModernBERT model is only supported on a single device"
            )
        device_spec = model_config.device_specs[0]
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"HuggingFace config is required for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )
        return cls(
            dtype=supported_encoding_dtype(quantization_encoding),
            device=DeviceRef(
                device_type=device_spec.device_type, id=device_spec.id
            ),
            pool_embeddings=model_config.pool_embeddings,
            huggingface_config=huggingface_config,
            max_seq_len=cls.calculate_max_seq_len(
                pipeline_config, huggingface_config, model_config
            ),
        )
