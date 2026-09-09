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
"""
Model configuration for adaptive fuzz test sizing.

Fetches model parameters from HuggingFace config.json (using the --model value
as a HF repo ID) and provides derived helper properties that scenarios use
instead of hardcoded values.

Priority: CLI overrides > HuggingFace fetch > defaults
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    max_position_embeddings: int = 4096
    max_num_tokens: int = 4096
    source: str = "defaults"

    # -- derived properties used by scenarios --

    @property
    def context_boundary_probe_sizes(self) -> list[int]:
        """Context sizes to probe, scaled to max_position_embeddings."""
        max_ctx = self.max_position_embeddings
        return [int(max_ctx * f) for f in [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]]

    @property
    def context_near_boundary_sizes(self) -> list[int]:
        """Sizes near the boundary for precise probing (90%, 97%, 99%)."""
        max_ctx = self.max_position_embeddings
        return [int(max_ctx * f) for f in [0.90, 0.97, 0.99]]

    @property
    def context_over_limit_size(self) -> int:
        """Size slightly exceeding max_position_embeddings (102%)."""
        return int(self.max_position_embeddings * 1.02)

    @property
    def large_input_tokens(self) -> int:
        """A 'large but fits' input size for pressure tests."""
        return max(min(self.max_position_embeddings // 4, 8000), 100)

    @property
    def medium_input_tokens(self) -> int:
        """Medium input size for mixed workload tests."""
        return max(min(self.max_position_embeddings // 8, 4000), 50)

    @property
    def prefill_flood_size(self) -> int:
        """Input size for prefill-heavy flood tests."""
        return max(min(self.max_position_embeddings // 4, 16000), 100)

    @property
    def decode_heavy_max_tokens(self) -> int:
        """max_tokens for decode-heavy tests."""
        return min(self.max_num_tokens, 16384)

    @property
    def large_generation_max_tokens(self) -> int:
        """max_tokens for long-generation KV growth tests."""
        return self.max_num_tokens

    @property
    def mixed_prefill_sizes(self) -> list[int]:
        """Mixed sizes for chunked prefill scheduler stress tests."""
        max_ctx = self.max_position_embeddings
        return [
            max(int(max_ctx * 0.01), 10),
            max(int(max_ctx * 0.01), 10),
            max(int(max_ctx * 0.06), 50),
            max(int(max_ctx * 0.06), 50),
            max(int(max_ctx * 0.30), 200),
            max(int(max_ctx * 0.30), 200),
        ]


@dataclass
class ModelProfile:
    """Model-specific test configuration for --model-profile."""

    name: str
    default_port: int
    tags: list[str]
    description: str = ""


MODEL_PROFILES: dict[str, ModelProfile] = {
    "kimi-k2.5": ModelProfile(
        name="kimi-k2.5",
        default_port=8200,
        tags=["model:kimi-k2.5"],
        description="Kimi K2.5 model-specific tests",
    ),
    "glm-5.1": ModelProfile(
        name="glm-5.1",
        default_port=8100,
        tags=["model:glm-5.1"],
        description="GLM-5.1 model-specific tests",
    ),
    "gemma4": ModelProfile(
        name="gemma4",
        default_port=8300,
        tags=["model:gemma4"],
        description="Gemma 4 model-specific tests",
    ),
    "minimax-m3": ModelProfile(
        name="minimax-m3",
        default_port=8400,
        tags=["model:minimax-m3"],
        description="MiniMax M3 model-specific tests",
    ),
}


def get_model_profile(name: str) -> ModelProfile | None:
    """Get a model profile by name, or None if not found."""
    return MODEL_PROFILES.get(name)


def fetch_hf_config(model: str, timeout: float = 10.0) -> dict[str, Any]:
    """Fetch config.json from HuggingFace Hub using only stdlib.

    Returns the parsed config dict, or empty dict on any failure.
    """
    url = f"https://huggingface.co/{model}/raw/main/config.json"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "llm-fuzz/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        return {}
    except urllib.error.URLError:
        return {}
    except (json.JSONDecodeError, Exception):
        return {}


def _extract_max_position_embeddings(hf_data: dict[str, Any]) -> int | None:
    """Extract max_position_embeddings from HF config, checking common locations."""
    for key in (
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
    ):
        if key in hf_data and isinstance(hf_data[key], int):
            return hf_data[key]
    # Check nested text_config (multimodal models)
    text_config = hf_data.get("text_config", {})
    if isinstance(text_config, dict):
        for key in (
            "max_position_embeddings",
            "n_positions",
            "max_sequence_length",
        ):
            if key in text_config and isinstance(text_config[key], int):
                return text_config[key]
    return None


def build_model_config(
    model: str = "",
    *,
    no_hf_fetch: bool = False,
    max_context_length: int | None = None,
    max_num_tokens: int | None = None,
) -> ModelConfig:
    """Build ModelConfig with priority: CLI overrides > HF fetch > defaults."""
    config = ModelConfig()

    # Step 1: Try HuggingFace fetch using --model as repo ID
    if model and not no_hf_fetch:
        hf_data = fetch_hf_config(model)
        if hf_data:
            max_pos = _extract_max_position_embeddings(hf_data)
            if max_pos is not None:
                config.max_position_embeddings = max_pos
                config.source = "huggingface"
                print(
                    f"  Fetched model config from HuggingFace: max_position_embeddings={max_pos:,}",
                    file=sys.stderr,
                )

    # Step 2: Apply manual overrides (highest priority)
    has_override = max_context_length is not None or max_num_tokens is not None
    if max_context_length is not None:
        config.max_position_embeddings = max_context_length
    if max_num_tokens is not None:
        config.max_num_tokens = max_num_tokens
    if has_override:
        config.source = (
            "manual"
            if config.source == "defaults"
            else f"{config.source}+manual"
        )

    return config
