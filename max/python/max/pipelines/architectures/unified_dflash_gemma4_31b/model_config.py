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
"""Config for the unified DFlash Gemma4 31B pipeline.

Binds a Gemma4 target to a z-lab DFlash drafter (``z-lab/gemma-4-31B-it-
DFlash``). The draft HF config is the same native ``DFlashDraftModel`` shape
the Llama3 pipeline parses, so the parsing helpers are reused; only the
target binding and the draft's own KV geometry are Gemma4-specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import ClassVar

from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.lib.interfaces.arch_config import (
    ArchConfigWithKVCache,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.speculative._dflash import (
    DflashDraftHFConfig,
    parse_dflash_draft_hf_config,
)
from transformers import AutoConfig
from typing_extensions import Self

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..llama3.model_config import Llama3Config
from ..unified_dflash_llama3.model_config import (
    resolve_dflash_num_speculative_tokens,
)

__all__ = [
    "DflashDraftHFConfig",
    "UnifiedDflashGemma4_31BConfig",
    "parse_dflash_draft_hf_config",
    "resolve_dflash_num_speculative_tokens",
]


def _with_num_draft_tokens(
    params: MultiKVCacheParams, num_draft_tokens: int
) -> MultiKVCacheParams:
    """Rebakes ``num_draft_tokens`` onto every leaf of the target's KV tree."""
    children: dict[str, KVCacheParams] = {}
    for name, leaf in params.children.items():
        assert isinstance(leaf, KVCacheParams)
        children[name] = replace(leaf, num_draft_tokens=num_draft_tokens)
    return MultiKVCacheParams.from_params(children)


def construct_dflash_draft_kv_params(
    pipeline_config: PipelineConfig,
    draft_config: Llama3Config,
    devices: list[DeviceRef],
    *,
    num_draft_tokens: int,
) -> KVCacheParams:
    """Builds the draft KV leaf from the DFlash drafter's own geometry.

    The draft's head_dim / KV head count differ from the Gemma4 target's, so
    the leaf cannot be derived by copying the target's params. Every layer
    gets a full-length cache even where the layer's mask is windowed: DFlash
    writes the whole context K/V before drafting, so windowed layers must
    still be able to address old blocks (the vLLM ``DFlashAttention``
    full-attention KV spec).
    """
    assert pipeline_config.speculative is not None
    return pipeline_config.model.kv_cache.to_params(
        dtype=draft_config.kv_params.dtype,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.kv_params.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=pipeline_config.speculative.speculative_method,
        num_draft_tokens=num_draft_tokens,
    )


@dataclass(kw_only=True)
class UnifiedDflashGemma4_31BConfig(ArchConfigWithKVCache):
    # Mirrors the `SupportedArchitecture` registration; generic consumers
    # (`PipelineModel._resolved_encoding`, `ArchConfig.initialize`) resolve a
    # config with no explicit `quantization_encoding` through these. The
    # target may be an NVFP4 checkpoint (auto-detected from the repo's packed
    # uint8 tensors); the DFlash drafter stays bfloat16 either way, since it
    # is loaded from its own checkpoint under its own encoding.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    target: Gemma4ForConditionalGenerationConfig
    draft: Llama3Config
    draft_kv_params: KVCacheParams
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    layer_types: list[str] = field(default_factory=list)
    mask_token_id: int = 0
    block_size: int = 0
    resolved_num_speculative_tokens: int | None = None
    """Per-step draft count: explicit value if set, else the trained width."""

    def __post_init__(self) -> None:
        self.target.text_config.return_logits = ReturnLogits.VARIABLE
        self.target.text_config.return_hidden_states = (
            ReturnHiddenStates.SELECTED_LAYERS
        )
        self.target.text_config.target_layer_ids = list(self.target_layer_ids)
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != 1:
            raise ValueError(
                "DFlash currently supports a single device only. Got"
                f" {len(self.target.devices)} devices."
            )

    def validate_dflash_fields(self) -> None:
        """Strict validation run once the DFlash-specific fields have been
        populated from the draft HF config.

        Unlike the Llama3 and Kimi DFlash pipelines this does NOT require one
        target tap per draft layer: the Gemma4 drafter fuses six taps into
        five layers, and the tap count is fixed by ``fc``'s in-features, not
        by the layer count.
        """
        n_target_layers = self.target.text_config.num_hidden_layers
        if not self.target_layer_ids:
            raise ValueError(
                "DFlash requires non-empty target_layer_ids (one per fc"
                " context feature)."
            )
        if any(not 0 <= i < n_target_layers for i in self.target_layer_ids):
            raise ValueError(
                "DFlash target_layer_ids must index the target's layers,"
                f" [0, {n_target_layers}). Got {self.target_layer_ids}."
            )
        if self.draft.hidden_size != self.target.text_config.hidden_size:
            raise ValueError(
                "DFlash draft hidden_size must match the target's (the fc /"
                " block-embedding / lm_head contract). Got"
                f" draft={self.draft.hidden_size}"
                f" target={self.target.text_config.hidden_size}."
            )
        if self.draft.vocab_size != self.target.text_config.vocab_size:
            raise ValueError(
                "DFlash draft vocab must match the target's: the draft has no"
                " head of its own and borrows the target's tied embedding."
                f" Got draft={self.draft.vocab_size}"
                f" target={self.target.text_config.vocab_size}."
            )
        if not 0 <= self.mask_token_id < self.target.text_config.vocab_size:
            raise ValueError(
                "DFlash mask_token_id must be in [0, target vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" vocab_size={self.target.text_config.vocab_size}."
            )
        if self.layer_types and len(self.layer_types) != (
            self.draft.num_hidden_layers
        ):
            raise ValueError(
                "DFlash layer_types must have one entry per draft layer. Got"
                f" {len(self.layer_types)} entries for"
                f" {self.draft.num_hidden_layers} layers."
            )

    @property
    def effective_block_size(self) -> int:
        """Anchor slot plus the drafted tokens per step.

        The DFlash block is trained at a fixed width; the anchor slot never
        predicts, so the drafted count is ``block_size - 1``. The trained
        width is resolved as a plain int by
        :func:`resolve_dflash_num_speculative_tokens` and threaded by the
        model; the caller's pipeline config is never rewritten.
        """
        if self.block_size > 0:
            return self.block_size
        num_spec = (
            self.resolved_num_speculative_tokens
            if self.resolved_num_speculative_tokens is not None
            else self.speculative_config.num_speculative_tokens
        )
        if num_spec is None:
            raise ValueError(
                "The DFlash draft checkpoint declares no block_size; set"
                " --num-speculative-tokens explicitly."
            )
        return num_spec + 1

    @property
    def devices(self) -> list[DeviceRef]:
        """Devices the unified model runs on (the memory-planner protocol)."""
        return list(self.target.devices)

    def get_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.target.get_kv_params(),
                "draft": self.draft_kv_params,
            }
        )

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        assert model_config.huggingface_config is not None
        assert pipeline_config.draft_model is not None
        draft_hf_config = pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None
        assert pipeline_config.speculative is not None

        speculative_config = pipeline_config.speculative
        explicit = speculative_config.num_speculative_tokens
        if explicit is None:
            # Unset resolves to the drafter's trained width.
            resolved = resolve_dflash_num_speculative_tokens(pipeline_config)
        else:
            resolved = explicit

        dflash_hf = parse_dflash_draft_hf_config(draft_hf_config)
        target_config = (
            Gemma4ForConditionalGenerationConfig.initialize_from_config(
                pipeline_config,
                model_config.huggingface_config,
                max_seq_len=max_seq_len,
            )
        )
        # Resolved at construction by PipelineConfig._resolve_max_length.
        draft_max_length = pipeline_config.draft_model.max_length
        assert draft_max_length is not None
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config,
            draft_hf_config,
            pipeline_config.draft_model,
            max_seq_len=draft_max_length,
        )
        if explicit is None:
            # ``initialize_from_config`` derives KV from the raw pipeline
            # config, where the unset width would bake num_draft_tokens=0.
            target_config.kv_params = _with_num_draft_tokens(
                target_config.kv_params, resolved
            )
        # ``initialize_from_config`` defaults the draft to gpu:0; pin it to
        # the target's device so the weights co-locate on a non-zero GPU.
        draft_config.devices = list(target_config.devices)
        draft_config.sliding_window = getattr(
            draft_hf_config, "sliding_window", None
        )

        # One leaf, shared: the draft module reads its geometry off
        # ``draft.kv_params`` while the KV manager reads ``draft_kv_params``,
        # and the two silently diverging is a wrong-shaped cache.
        draft_config.kv_params = construct_dflash_draft_kv_params(
            pipeline_config,
            draft_config,
            list(target_config.devices),
            num_draft_tokens=(resolved or 0),
        )

        return cls(
            target=target_config,
            draft=draft_config,
            draft_kv_params=draft_config.kv_params,
            speculative_config=speculative_config,
            resolved_num_speculative_tokens=resolved,
            target_layer_ids=list(dflash_hf.target_layer_ids),
            layer_types=list(
                getattr(draft_hf_config, "layer_types", None) or []
            ),
            mask_token_id=int(dflash_hf.mask_token_id),
            block_size=int(dflash_hf.block_size or 0),
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return Gemma4ForConditionalGenerationConfig.calculate_max_seq_len(
            huggingface_config, model_config
        )
