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

from dataclasses import dataclass
from typing import ClassVar

from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfig,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self


@dataclass(kw_only=True)
class Qwen3EmbeddingConfig(ArchConfig):
    """Qwen3 embedding model configuration."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "float32",
        "bfloat16",
    }

    pipeline_config: PipelineConfig
    max_seq_len: int
    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        # The configured max_length, bounded by max_position_embeddings.
        model_max = getattr(
            huggingface_config, "max_position_embeddings", 32768
        )
        configured_max = model_config.max_length or 8192

        if configured_max > model_max:
            raise ValueError(
                f"Configured max_length ({configured_max}) exceeds model's "
                f"max_position_embeddings ({model_max})"
            )

        return configured_max

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        return cls(pipeline_config=pipeline_config, max_seq_len=max_seq_len)
