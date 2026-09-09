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
"""Config parsing for vLLM speculators-format DSpark draft checkpoints.

A speculators DSpark checkpoint (e.g.
``RedHatAI/gemma-4-31B-it-speculator.dspark``) has no top-level
``model_type`` (MAX's raw-JSON config fallback wraps it in a
``PretrainedConfig``), DSpark fields at the top level, and the llama-style
draft block geometry nested under ``transformer_layer_config``. Nothing here
is model-specific — all geometry parses from the checkpoint config, so every
unified speculators-DSpark architecture shares this module.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import KVCacheParams
from max.pipelines.lib.config import (
    MAXModelConfig,
    PipelineConfig,
    SpeculativeConfig,
)
from transformers import AutoConfig

logger = logging.getLogger("max.pipelines")

_SLIDING = "sliding_attention"
_FULL = "full_attention"


def _get(obj: Any, name: str, default: Any = None) -> Any:
    """Reads *name* from an HF config object or a plain (nested) dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def construct_draft_kv_params(
    pipeline_config: PipelineConfig,
    draft_config: DSparkSpeculatorsDraftConfig,
    devices: list[DeviceRef],
    *,
    num_draft_tokens: int,
) -> KVCacheParams:
    """Builds the draft KV leaf from the DSpark drafter's own geometry."""
    assert pipeline_config.speculative is not None
    # The dtype is deliberately pinned rather than derived from
    # ``kv_cache_format``: the draft block runs generic ``AttentionWithRope``,
    # which feeds a bfloat16 Q into ``flash_attention_ragged`` (no fp8-Q cast
    # like the gemma4 target attention's ``q_out_dtype``), and the drafter is
    # trained against bfloat16. An fp8 target tree pairs with a bfloat16
    # draft leaf; mixed-dtype trees are valid (dtype is per-leaf).
    return pipeline_config.model.kv_cache.to_params(
        dtype=DType.bfloat16,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=devices,
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        speculative_method=pipeline_config.speculative.speculative_method,
        num_draft_tokens=num_draft_tokens,
    )


@dataclass(frozen=True)
class DSparkSpeculatorsDraftConfig:
    """DSpark drafter hyperparameters from a speculators-format checkpoint.

    Follows the vLLM/speculators ``DSparkSpeculatorConfig`` schema: DSpark
    fields at the top level and a llama-style ``transformer_layer_config``
    for the draft block. Parsing is strict — unsupported variants (mixed
    layer causality, non-vanilla markov heads) fail loudly at load.
    """

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    """Separate K/V GQA head count (NOT the 12B k_eq_v convention)."""
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    """Full target vocabulary (embedding / markov_w1 side)."""
    draft_vocab_size: int
    """Pruned draft vocabulary (lm_head / markov_w2 side, ``d2t``-mapped)."""
    hidden_activation: str
    rope_theta: float
    max_seq_len: int
    """From the NESTED ``transformer_layer_config.max_position_embeddings``."""
    sliding_window: int
    causal: bool
    """Per-layer causality, homogeneous across layers (guarded at parse).

    Causal iff ``layer_types[i] == "sliding_attention"``, honoring a
    top-level ``causal`` override when present (the vLLM
    ``_dflash_layer_causal`` rule).
    """
    block_size: int
    """Draft block width INCLUDING the anchor slot when
    ``sample_from_anchor`` is false (the RedHat convention: 8 = anchor + 7
    drafts)."""
    sample_from_anchor: bool
    mask_token_id: int
    aux_hidden_state_layer_ids: tuple[int, ...]
    """Target taps in the vLLM eagle convention: aux id ``j`` is the raw
    residual stream at the INPUT of target layer ``j``."""
    markov_rank: int
    markov_head_type: str

    @property
    def num_speculative_tokens(self) -> int:
        """Drafted tokens per step: the anchor slot never predicts unless
        ``sample_from_anchor`` is set."""
        return self.block_size - (0 if self.sample_from_anchor else 1)

    def checkpoint_draft_width(
        self, speculative_config: SpeculativeConfig
    ) -> int:
        """Resolves the per-step draft count against the trained block.

        An explicitly configured ``num_speculative_tokens`` K is always
        honored. At or below the trained draft width the truncation is
        prefix-stable: the draft block is CAUSAL, so the K drafted
        positions come out identical to the trained-width block's first
        K. Beyond the trained width the block is K-generic but the extra
        positions run as extrapolation (no trained mask slot backs them),
        so acceptance is expected to degrade there and a warning is
        logged. When the field is ``None`` (unset), the trained width
        applies.

        Args:
            speculative_config: The pipeline's speculative config; only
                its ``num_speculative_tokens`` field is consulted.

        Returns:
            The effective per-step draft count.

        Raises:
            ValueError: If the configured value is not positive.
        """
        trained = self.num_speculative_tokens
        requested = speculative_config.num_speculative_tokens
        if requested is not None and requested < 1:
            raise ValueError(
                f"num_speculative_tokens={requested} must be at least 1."
            )
        if requested is not None and requested > trained:
            logger.warning(
                "This DSpark drafter was trained at block_size=%d"
                " (sample_from_anchor=%s), i.e. %d drafted tokens per step;"
                " honoring num_speculative_tokens=%d, but positions beyond"
                " the trained width are extrapolation and acceptance is"
                " expected to degrade there.",
                self.block_size,
                self.sample_from_anchor,
                trained,
                requested,
            )
        resolved = trained if requested is None else requested
        return resolved

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        """Aux ids shifted to MAX's capture convention.

        MAX's SELECTED_LAYERS capture appends AFTER layer ``k`` runs, so
        MAX id ``k`` is the output of layer ``k`` — the input of layer
        ``j`` is MAX id ``j - 1``.
        """
        return tuple(j - 1 for j in self.aux_hidden_state_layer_ids)

    @property
    def num_context_features(self) -> int:
        return len(self.aux_hidden_state_layer_ids)

    @classmethod
    def from_huggingface_config(
        cls,
        huggingface_config: Any,
        *,
        max_seq_len: int | None = None,
    ) -> DSparkSpeculatorsDraftConfig:
        """Parses a speculators-format DSpark config (object or dict)."""
        cfg = huggingface_config

        speculators_model_type = _get(cfg, "speculators_model_type")
        if speculators_model_type != "dspark":
            raise ValueError(
                "Expected a speculators checkpoint with"
                " speculators_model_type='dspark', got"
                f" {speculators_model_type!r}."
            )

        layer_cfg = _get(cfg, "transformer_layer_config")
        if layer_cfg is None:
            raise ValueError(
                "DSpark speculators config is missing transformer_layer_config."
            )

        vocab_size = int(_get(layer_cfg, "vocab_size"))
        num_hidden_layers = int(_get(layer_cfg, "num_hidden_layers"))

        draft_vocab_size = int(_get(cfg, "draft_vocab_size", 0))
        if not 0 < draft_vocab_size <= vocab_size:
            raise ValueError(
                "DSpark draft_vocab_size must be in (0, vocab_size]. Got"
                f" draft_vocab_size={draft_vocab_size}"
                f" vocab_size={vocab_size}."
            )

        block_size = int(_get(cfg, "block_size", 0))
        if block_size < 2:
            raise ValueError(
                f"DSpark block_size must be >= 2, got {block_size}."
            )

        mask_token_id = int(_get(cfg, "mask_token_id", -1))
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                "DSpark mask_token_id must be in [0, vocab_size). Got"
                f" mask_token_id={mask_token_id} vocab_size={vocab_size}."
            )

        aux_ids = tuple(
            int(j) for j in (_get(cfg, "aux_hidden_state_layer_ids") or ())
        )
        if not aux_ids:
            raise ValueError(
                "DSpark speculators config is missing"
                " aux_hidden_state_layer_ids."
            )
        if any(j < 1 for j in aux_ids):
            raise ValueError(
                "DSpark aux_hidden_state_layer_ids use the vLLM eagle"
                " convention (input of target layer j) and must all be"
                f" >= 1. Got {list(aux_ids)}."
            )
        if any(a >= b for a, b in itertools.pairwise(aux_ids)):
            raise ValueError(
                "DSpark aux_hidden_state_layer_ids must be strictly"
                f" increasing. Got {list(aux_ids)}."
            )

        layer_types = list(_get(layer_cfg, "layer_types") or [])
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                "DSpark draft layer_types must list every layer. Got"
                f" {len(layer_types)} entries for {num_hidden_layers}"
                " layers."
            )
        unknown = sorted(set(layer_types) - {_SLIDING, _FULL})
        if unknown:
            raise ValueError(
                f"DSpark draft has unsupported layer_types {unknown};"
                f" expected only {_SLIDING!r} or {_FULL!r}."
            )
        if len(set(layer_types)) != 1:
            # Mixed sliding/full drafts need per-layer masks and multiple
            # draft KV groups; fail loudly until supported.
            raise ValueError(
                "DSpark draft with mixed layer_types is not supported yet."
                f" Got {layer_types}."
            )
        # The vLLM _dflash_layer_causal rule: an explicit `causal` field
        # overrides, else causal iff sliding_attention.
        causal_override = _get(cfg, "causal", None)
        causal = (
            bool(causal_override)
            if causal_override is not None
            else layer_types[0] == _SLIDING
        )
        if bool(_get(cfg, "sliding_window_non_causal", False)):
            raise ValueError(
                "DSpark sliding_window_non_causal=true is not supported yet."
            )

        sliding_window = int(_get(layer_cfg, "sliding_window") or 0)
        if layer_types[0] == _SLIDING and sliding_window <= 0:
            raise ValueError(
                "DSpark sliding_attention layers require sliding_window > 0."
                f" Got {sliding_window}."
            )

        sample_from_anchor = bool(_get(cfg, "sample_from_anchor", False))
        expected_spec_tokens = block_size - (0 if sample_from_anchor else 1)
        speculators_cfg = _get(cfg, "speculators_config") or {}
        proposal_methods = _get(speculators_cfg, "proposal_methods") or []
        if proposal_methods:
            proposal_spec_tokens = _get(
                proposal_methods[0], "speculative_tokens"
            )
            if (
                proposal_spec_tokens is not None
                and int(proposal_spec_tokens) != expected_spec_tokens
            ):
                raise ValueError(
                    "DSpark speculators proposal declares"
                    f" speculative_tokens={int(proposal_spec_tokens)} but"
                    f" block_size={block_size} with"
                    f" sample_from_anchor={sample_from_anchor} implies"
                    f" {expected_spec_tokens}."
                )

        markov_rank = int(_get(cfg, "markov_rank", 0))
        markov_head_type = str(_get(cfg, "markov_head_type", "vanilla"))
        if markov_head_type != "vanilla":
            raise ValueError(
                "Only the vanilla markov head is supported, got"
                f" markov_head_type={markov_head_type!r}."
            )
        if markov_rank <= 0:
            raise ValueError(
                f"DSpark markov_rank must be > 0, got {markov_rank}."
            )

        rope_parameters = _get(layer_cfg, "rope_parameters") or {}
        rope_theta = _get(rope_parameters, "rope_theta")
        if rope_theta is None:
            raise ValueError(
                "DSpark draft config is missing"
                " transformer_layer_config.rope_parameters.rope_theta."
            )
        rope_type = _get(rope_parameters, "rope_type", "default")
        if rope_type != "default":
            raise ValueError(
                "DSpark draft supports only the default rope, got"
                f" rope_type={rope_type!r}."
            )

        max_position_embeddings = _get(layer_cfg, "max_position_embeddings")
        if max_seq_len is None and max_position_embeddings is None:
            raise ValueError(
                "DSpark draft config is missing"
                " transformer_layer_config.max_position_embeddings."
            )

        return cls(
            hidden_size=int(_get(layer_cfg, "hidden_size")),
            intermediate_size=int(_get(layer_cfg, "intermediate_size")),
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=int(_get(layer_cfg, "num_attention_heads")),
            num_key_value_heads=int(_get(layer_cfg, "num_key_value_heads")),
            head_dim=int(_get(layer_cfg, "head_dim")),
            rms_norm_eps=float(_get(layer_cfg, "rms_norm_eps")),
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_activation=str(_get(layer_cfg, "hidden_act")),
            rope_theta=float(rope_theta),
            max_seq_len=(
                int(max_seq_len)
                if max_seq_len is not None
                else int(max_position_embeddings)
            ),
            sliding_window=sliding_window,
            causal=causal,
            block_size=block_size,
            sample_from_anchor=sample_from_anchor,
            mask_token_id=mask_token_id,
            aux_hidden_state_layer_ids=aux_ids,
            markov_rank=markov_rank,
            markov_head_type=markov_head_type,
        )


@dataclass
class DSparkSpeculatorsDraftArchConfig:
    """Thin ArchConfig for the standalone ``DSparkDraftModel`` registration.

    A speculators DSpark checkpoint is only ever served as ``--draft-model``
    next to its target (the registry rewrites the pair to the target's
    unified architecture), so the registry needs just the draft-side
    max-sequence-length clamp from this config. ``max_position_embeddings``
    is nested under ``transformer_layer_config``.
    """

    max_position_embeddings: int = 262144

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> DSparkSpeculatorsDraftArchConfig:
        del pipeline_config
        assert model_config is not None
        huggingface_config = model_config.huggingface_config
        assert huggingface_config is not None
        layer_cfg = _get(huggingface_config, "transformer_layer_config")
        max_pos = (
            _get(layer_cfg, "max_position_embeddings")
            if layer_cfg is not None
            else None
        )
        if max_pos is None:
            raise ValueError(
                "DSpark speculators draft config is missing"
                " transformer_layer_config.max_position_embeddings."
            )
        return cls(max_position_embeddings=int(max_pos))

    def get_max_seq_len(self) -> int:
        return self.max_position_embeddings

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del model_config
        layer_cfg = _get(huggingface_config, "transformer_layer_config")
        max_pos = (
            _get(layer_cfg, "max_position_embeddings")
            if layer_cfg is not None
            else None
        )
        if max_pos is None:
            raise ValueError(
                "DSpark speculators draft config is missing"
                " transformer_layer_config.max_position_embeddings."
            )
        return int(max_pos)


def speculators_dspark_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the user's width if set, else the draft's trained width."""
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DSpark requires a draft model.")
    draft = DSparkSpeculatorsDraftConfig.from_huggingface_config(
        draft_huggingface_config
    )
    return draft.checkpoint_draft_width(speculative)
