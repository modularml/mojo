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
"""MAX LoRA configuration."""

from __future__ import annotations

import logging

from max.config import ConfigFileModel
from pydantic import ConfigDict, Field

_logger = logging.getLogger("max.pipelines")


class LoRAConfig(ConfigFileModel):
    """Configuration for LoRA (Low-Rank Adaptation) inference."""

    model_config = ConfigDict(frozen=True)

    enable_lora: bool = Field(
        default=False, description="Enables LoRA on the server."
    )
    """Whether LoRA adapters are enabled on the server."""

    lora_paths: list[str] = Field(
        default_factory=list,
        description="List of statically defined LoRA paths.",
    )
    """The list of statically defined LoRA adapter paths."""

    max_lora_rank: int = Field(
        default=16, description="Maximum rank of all possible LoRAs."
    )
    """The maximum rank of all LoRA adapters."""

    max_num_loras: int = Field(
        default=1,
        description=(
            "The maximum number of active LoRAs in a batch. This controls how "
            "many LoRA adapters can be active simultaneously during inference. "
            "Lower values reduce memory usage but limit concurrent adapter "
            "usage."
        ),
    )
    """The maximum number of active LoRA adapters in a batch."""

    _config_file_section_name: str = "lora_config"
    """The section name to use when loading this config from a MAXConfig file.
    This is used to differentiate between different config sections in a single
    MAXConfig file."""
