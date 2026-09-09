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
"""Configuration for Bert models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
    ArchConfigWithBoundedMaxSeqLen,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from transformers import AutoConfig
from typing_extensions import Self, override


@dataclass(kw_only=True)
class BertModelConfig(ArchConfigWithBoundedMaxSeqLen, ArchConfig):
    """Configuration for Bert models."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    dtype: DType
    device: DeviceRef
    pool_embeddings: bool
    huggingface_config: AutoConfig
    max_seq_len: int
    quantization_encoding: SupportedEncoding | None = None

    @override
    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        """Initializes a BertModelConfig instance from pipeline configuration.

        Args:
            pipeline_config: The MAX Engine pipeline configuration.

        Returns:
            An initialized BertModelConfig instance.
        """
        model_config = model_config or pipeline_config.model
        quantization_encoding = _select_quantization_encoding(
            model_config, cls.DEFAULT_ENCODING
        )
        if len(model_config.device_specs) != 1:
            raise ValueError("BERT model is only supported on a single device")
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
            max_seq_len=max_seq_len,
            quantization_encoding=quantization_encoding,
        )
