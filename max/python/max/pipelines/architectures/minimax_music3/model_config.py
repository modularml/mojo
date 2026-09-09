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
"""Per-component configuration for MiniMax Music 3.

The checkpoint ships five independent ``config.json`` files under their own
subfolders. Each dataclass here mirrors one of them, defaulting to the released
checkpoint's values so a test can construct one without a download.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


def _from_dict(cls: type[_T], raw: Mapping[str, Any]) -> _T:
    """Build a config dataclass from a parsed diffusers-style ``config.json``.

    Unknown keys are dropped rather than raising: these files carry bookkeeping
    such as ``_class_name`` and ``_diffusers_version`` that is not configuration.
    """
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in raw.items() if k in known})


def _from_json(cls: type[_T], path: Path) -> _T:
    """Build a config dataclass from a diffusers-style ``config.json`` file."""
    return _from_dict(cls, json.loads(path.read_text()))


@dataclass
class VocoderConfig:
    """Configuration of the DAC-style Flow-VAE waveform decoder."""

    latent_channels: int = 128
    decoder_input_dim: int = 1024
    decoder_hidden_dim: int = 1536
    upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2)
    sampling_rate: int = 44100

    def __post_init__(self) -> None:
        self.upsampling_ratios = tuple(self.upsampling_ratios)

    @property
    def hop_length(self) -> int:
        """Waveform samples produced per latent frame."""
        hop = 1
        for ratio in self.upsampling_ratios:
            hop *= ratio
        return hop

    @staticmethod
    def from_pretrained(root: Path) -> VocoderConfig:
        """Load from ``{root}/vocoder/config.json``."""
        return _from_json(VocoderConfig, root / "vocoder" / "config.json")


@dataclass
class ConditionEncoderConfig:
    """Configuration of the layer-mixing conditioning encoder."""

    condition_hidden_dim: int = 4096
    num_condition_layers: int = 8
    out_dim: int = 2048
    input_sampling_rate: int = 24000
    input_hop_length: int = 960
    output_sampling_rate: int = 44100
    output_hop_length: int = 512

    @property
    def frame_rate(self) -> float:
        """Autoregressive frames per second (25.0 for the released checkpoint)."""
        return self.input_sampling_rate / self.input_hop_length

    def latent_length(self, num_frames: int) -> int:
        """Latent frames the encoder resamples ``num_frames`` onto.

        Truncating integer arithmetic, matching the reference exactly: at the
        released rates one autoregressive frame becomes 3.4453125 latent frames,
        so 200 frames give 689 rather than 690.
        """
        return max(
            1,
            int(
                num_frames
                * self.output_sampling_rate
                / self.input_sampling_rate
                * self.input_hop_length
                / self.output_hop_length
            ),
        )

    @staticmethod
    def from_pretrained(root: Path) -> ConditionEncoderConfig:
        """Load from ``{root}/condition_encoder/config.json``."""
        return _from_json(
            ConditionEncoderConfig, root / "condition_encoder" / "config.json"
        )


@dataclass
class TransformerConfig:
    """Configuration of the flow-matching DiT."""

    in_channels: int = 128
    condition_dim: int = 2048
    num_layers: int = 36
    num_attention_heads: int = 32
    attention_head_dim: int = 64
    ff_inner_dim: int = 8192
    fourier_embedding_dim: int = 256
    rotary_dim: int = 32
    # Not a checkpoint field: the reference hardcodes it in the rotary module's
    # default argument, so config.json never mentions it.
    rotary_theta: float = 10000.0

    @property
    def hidden_size(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @staticmethod
    def from_pretrained(root: Path) -> TransformerConfig:
        """Load from ``{root}/transformer/config.json``."""
        return _from_json(
            TransformerConfig, root / "transformer" / "config.json"
        )


@dataclass
class DepthDecoderConfig:
    """Configuration of the RVQ depth decoder."""

    hidden_size: int = 4096
    num_layers: int = 4
    num_attention_heads: int = 16
    intermediate_size: int = 6144
    num_codebooks: int = 8
    audio_vocab_size: int = 1024
    max_position_embeddings: int = 16

    @staticmethod
    def from_pretrained(root: Path) -> DepthDecoderConfig:
        """Load from ``{root}/rvq_depth_decoder/config.json``."""
        return _from_json(
            DepthDecoderConfig, root / "rvq_depth_decoder" / "config.json"
        )


@dataclass
class LanguageModelConfig:
    """Configuration of the global autoregressive model, a stock Qwen3.

    The vocabulary is where this stops being a plain text model. Of 200000 rows,
    generation can only ever emit 16385: one per semantic code, plus the token
    that ends the audio. Everything else is masked to negative infinity before
    sampling, so the port carries a 16385-row head rather than the full one and
    saves 1.5 GiB of a 22 GiB device.
    """

    hidden_size: int = 4096
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 12288
    rms_norm_eps: float = 1e-6
    vocab_size: int = 200000
    max_position_embeddings: int = 10240

    # Token-id contract, fixed by the checkpoint rather than by its config.json:
    # the reference hardcodes these in its pipeline.
    audio_code_offset: int = 151675
    semantic_vocab_size: int = 16384
    audio_end_token_id: int = 151670
    audio_cfg_token_id: int = 151654

    @property
    def rope_theta(self) -> float:
        return 1_000_000.0

    @property
    def head_vocab_size(self) -> int:
        """Rows of the sliced head: every semantic code, then the end token."""
        return self.semantic_vocab_size + 1

    @property
    def end_of_audio_row(self) -> int:
        """Where the end token lands in the sliced head, which is last."""
        return self.semantic_vocab_size

    @staticmethod
    def from_dict(raw: Mapping[str, Any]) -> LanguageModelConfig:
        """Build from a parsed ``config.json``, checking its rope period.

        Rope is the one field the port hardcodes rather than reads, so the
        checkpoint gets to contradict it. A model rotated at a different period
        would still run and still produce plausible frames, which is exactly the
        failure worth raising on.

        Raises:
            ValueError: If the checkpoint states a period this port was not
                written against.
        """
        config = _from_dict(LanguageModelConfig, raw)
        theta = (raw.get("rope_parameters") or {}).get("rope_theta")
        if theta is not None and float(theta) != config.rope_theta:
            raise ValueError(
                f"checkpoint rope_theta {theta} is not the {config.rope_theta} "
                "this port was written against"
            )
        return config

    @staticmethod
    def from_pretrained(root: Path) -> LanguageModelConfig:
        """Load from ``{root}/language_model/config.json``."""
        path = root / "language_model" / "config.json"
        return LanguageModelConfig.from_dict(json.loads(path.read_text()))


@dataclass
class SamplingConfig:
    """How a code is chosen, at both of the stages that choose one.

    A property of neither checkpoint: the reference hardcodes these in its
    pipeline, and the global model and the depth decoder draw with the same scale
    and the same width. Where they differ is that only the global model narrows
    the candidate set to the conditional row's best before guiding, which is why
    ``cfg_top_k`` is passed to one and not the other -- see :mod:`.sampling`.
    """

    cfg_scale: float = 1.5
    cfg_top_k: int = 50
    sampling_top_k: int = 50


@dataclass
class MiniMaxMusic3Config:
    """The five component configurations plus the pipeline-level constants."""

    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    condition_encoder: ConditionEncoderConfig = field(
        default_factory=ConditionEncoderConfig
    )
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    depth_decoder: DepthDecoderConfig = field(
        default_factory=DepthDecoderConfig
    )
    language_model: LanguageModelConfig = field(
        default_factory=LanguageModelConfig
    )
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @property
    def sampling_rate(self) -> int:
        """The output audio sample rate, in Hz."""
        return self.vocoder.sampling_rate

    @property
    def frame_rate(self) -> float:
        """The autoregressive frame rate, in frames per second."""
        return self.condition_encoder.frame_rate

    @property
    def latent_hop_length(self) -> int:
        """The latent hop length, in samples."""
        return self.condition_encoder.output_hop_length

    @staticmethod
    def from_pretrained(root: Path | str) -> MiniMaxMusic3Config:
        """Loads every component configuration from a checkpoint directory."""
        root = Path(root)
        return MiniMaxMusic3Config(
            vocoder=VocoderConfig.from_pretrained(root),
            condition_encoder=ConditionEncoderConfig.from_pretrained(root),
            transformer=TransformerConfig.from_pretrained(root),
            depth_decoder=DepthDecoderConfig.from_pretrained(root),
            language_model=LanguageModelConfig.from_pretrained(root),
        )

    @staticmethod
    def from_dicts(
        configs: Mapping[str, Mapping[str, Any]],
    ) -> MiniMaxMusic3Config:
        """Builds a configuration from each component's parsed ``config.json``.

        Args:
            configs: The parsed ``config.json`` contents, keyed by the
                checkpoint's role names. Those names are the subfolders listed
                in the component index.

        Raises:
            KeyError: If a component is missing.
        """
        return MiniMaxMusic3Config(
            vocoder=_from_dict(VocoderConfig, configs["vocoder"]),
            condition_encoder=_from_dict(
                ConditionEncoderConfig, configs["condition_encoder"]
            ),
            transformer=_from_dict(TransformerConfig, configs["transformer"]),
            depth_decoder=_from_dict(
                DepthDecoderConfig, configs["rvq_depth_decoder"]
            ),
            language_model=LanguageModelConfig.from_dict(
                configs["language_model"]
            ),
        )
