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
"""Config for the fused DFlash2 Qwen3.5 pipeline.

Binds a Qwen3.5 target to a ``z-lab/Qwen3.8-27B-DFlash2`` drafter. The draft
HF config is the v1 ``dflash_config`` shape plus four keys v1 never had
(``conv_kernel_size``, ``conv_group_size``, ``selector_rank``,
``selector_top_k``), so the v1 parser is reused for the shared keys and
extended here rather than widened in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from max.dtype import DType
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
from max.pipelines.lib.interfaces.arch_config import ArchConfigWithKVCache
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig
from typing_extensions import Self

from ..llama3.model_config import Llama3Config
from ..qwen3_5.model_config import Qwen3_5Config
from ..unified_dflash_llama3.model_config import parse_dflash_draft_hf_config

__all__ = [
    "Dflash2DraftHFConfig",
    "UnifiedDflash2Qwen3_5Config",
    "construct_dflash2_draft_kv_params",
    "dflash2_draft_width",
    "parse_dflash2_draft_hf_config",
    "resolve_dflash2_num_speculative_tokens",
]

DRAFT_SLIDING_WINDOW = 2048
"""Positions the drafter can attend to, from the checkpoint's config.

The draft KV leaf is bounded at this rather than at the target's context: the
drafter's mask is windowed, so nothing older is ever read, and the leaf's cost
becomes a per-slot constant instead of scaling with ``--max-length``.
"""


@dataclass(frozen=True)
class Dflash2DraftHFConfig:
    """The ``dflash_config`` block of a DFlash2 draft checkpoint."""

    mask_token_id: int
    target_layer_ids: list[int]
    block_size: int
    num_target_layers: int | None
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int

    def draft_width(self, speculative: SpeculativeConfig) -> int:
        """Returns ``block_size - 1``, refusing a flag that disagrees.

        The v1 helper warns and overrides. DFlash2 hard-errors instead
        (design D3): the width is not recoverable from the MEF, so the export
        records it alongside and the engine checks its own
        ``--num-speculative-tokens`` against that at boot. An overridden width
        would compile a block-8 graph and declare it was built for something
        else, and the engine would then feed the graph at the width it was
        told.

        Raises:
            ValueError: If ``num_speculative_tokens`` is set and is not
                ``block_size - 1``.
        """
        expected = self.block_size - 1
        requested = speculative.num_speculative_tokens
        if requested is not None and requested != expected:
            raise ValueError(
                f"DFlash2 was trained at block_size={self.block_size}, so it"
                f" drafts exactly {expected} tokens per step, but"
                f" --num-speculative-tokens is {requested}. The block width is"
                " structural (the drafter's convolution zeroes its t-1 tap at"
                " block-local position 0 and the selector walks a fixed"
                " tensor), so this is a hard error rather than an override."
            )
        return expected


def _dflash_config_of(huggingface_config: Any) -> Any:
    cfg = (
        huggingface_config.get("dflash_config")
        if isinstance(huggingface_config, dict)
        else getattr(huggingface_config, "dflash_config", None)
    )
    if cfg is None:
        raise ValueError(
            "DFlash2 draft HF config is missing ``dflash_config``."
        )
    return cfg


def _required_int(dflash_config: Any, name: str) -> int:
    value = (
        dflash_config.get(name)
        if isinstance(dflash_config, dict)
        else getattr(dflash_config, name, None)
    )
    if value is None:
        raise ValueError(
            f"DFlash2 draft HF config is missing ``dflash_config.{name}``."
        )
    return int(value)


def parse_dflash2_draft_hf_config(
    huggingface_config: Any,
) -> Dflash2DraftHFConfig:
    """Reads the DFlash2 drafter's structural constants off its HF config.

    Raises:
        ValueError: If a required key is missing, if the checkpoint declares no
            ``block_size``, or if it asks for an embedding/logit rescale this
            graph does not implement.
    """
    shared = parse_dflash_draft_hf_config(huggingface_config)
    if shared.block_size is None:
        raise ValueError(
            "DFlash2 draft HF config declares no ``dflash_config.block_size``."
            " The block width is structural — the drafter's convolution zeroes"
            " its t-1 tap at block-local position 0 — so it cannot be supplied"
            " by a flag."
        )
    dflash_config = _dflash_config_of(huggingface_config)

    # The reference multiplies the borrowed embedding and the borrowed head's
    # logits by these. Both are 1.0 on the shipped checkpoint and this graph
    # applies neither, so a checkpoint that sets them would draft from a
    # differently scaled space with no symptom but lost acceptance.
    for name in ("input_embedding_scale", "output_multiplier"):
        scale = (
            dflash_config.get(name, 1.0)
            if isinstance(dflash_config, dict)
            else getattr(dflash_config, name, 1.0)
        )
        if float(scale) != 1.0:
            raise ValueError(
                f"DFlash2 draft declares dflash_config.{name}={scale}; this"
                " graph borrows the target's embedding and head unscaled and"
                " implements neither rescale."
            )

    return Dflash2DraftHFConfig(
        mask_token_id=shared.mask_token_id,
        target_layer_ids=list(shared.target_layer_ids),
        block_size=shared.block_size,
        num_target_layers=shared.num_target_layers,
        conv_kernel_size=_required_int(dflash_config, "conv_kernel_size"),
        conv_group_size=_required_int(dflash_config, "conv_group_size"),
        selector_rank=_required_int(dflash_config, "selector_rank"),
        selector_top_k=_required_int(dflash_config, "selector_top_k"),
    )


def dflash2_draft_width(
    speculative: SpeculativeConfig,
    target_huggingface_config: Any,
    draft_huggingface_config: Any,
) -> int:
    """Returns the width the DFlash2 draft checkpoint was trained for.

    Registered as the architecture's ``checkpoint_draft_width`` so that
    ``PipelineConfig.from_args`` resolves an omitted
    ``--num-speculative-tokens``. Without it the pipeline config keeps
    ``None`` while the graph is built at the checkpoint's width, and whatever
    reads the config instead of the graph -- the scheduler's draft-token
    handoff among them -- sees zero.
    """
    del target_huggingface_config
    if draft_huggingface_config is None:
        raise ValueError("DFlash2 requires a draft model.")
    return parse_dflash2_draft_hf_config(draft_huggingface_config).draft_width(
        speculative
    )


def resolve_dflash2_num_speculative_tokens(
    pipeline_config: PipelineConfig,
) -> int:
    """Returns the draft width for ``pipeline_config``'s drafter.

    Raises:
        ValueError: If ``num_speculative_tokens`` is set and is not
            ``block_size - 1``.
    """
    assert pipeline_config.speculative is not None
    assert pipeline_config.draft_model is not None
    return parse_dflash2_draft_hf_config(
        pipeline_config.draft_model.huggingface_config
    ).draft_width(pipeline_config.speculative)


def construct_dflash2_draft_kv_params(
    pipeline_config: PipelineConfig,
    draft_config: Llama3Config,
    target_kv_params: KVCacheParams,
) -> KVCacheParams:
    """Builds the bounded draft KV leaf from the drafter's own geometry.

    Three things differ from the target's leaf and none can be inherited: the
    drafter's head geometry (8 KV heads of 128 against the target's 4 of 256),
    its dtype (its K/V projections are bf16 whatever the target's cache is),
    and its window. The window is what makes this leaf a per-slot constant:
    the drafter's mask never reaches further back than
    :data:`DRAFT_SLIDING_WINDOW`, and the materialize is trimmed to the same
    bound, so a page older than the window is neither read nor written.

    ``page_size`` follows the target's, which Qwen3.5 bumps to its head_dim;
    :class:`MultiKVCacheParams` requires one page size across the tree.
    """
    return pipeline_config.model.kv_cache.to_params(
        # Pinned rather than following --kv-cache-dtype: the drafter fills
        # this cache with its own unquantized projections. Mach's registry
        # pins the matching group and the MEF sidecar's draft_kv_cache_dtype
        # makes a disagreement a startup error.
        dtype=DType.bfloat16,
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.kv_params.head_dim,
        num_layers=draft_config.num_hidden_layers,
        devices=list(draft_config.devices),
        data_parallel_degree=pipeline_config.model.data_parallel_degree,
        page_size=target_kv_params.page_size,
        window_size=DRAFT_SLIDING_WINDOW,
    )


@dataclass(kw_only=True)
class UnifiedDflash2Qwen3_5Config(ArchConfigWithKVCache):
    """Target, drafter and the structural constants that bind them."""

    # The target may be NVFP4; the drafter is loaded from its own bfloat16
    # checkpoint under its own encoding either way.
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = (
        Qwen3_5Config.DEFAULT_ENCODING
    )
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = (
        Qwen3_5Config.SUPPORTED_ENCODINGS
    )

    target: Qwen3_5Config
    draft: Llama3Config
    draft_kv_params: KVCacheParams
    speculative_config: SpeculativeConfig
    target_layer_ids: list[int] = field(default_factory=list)
    layer_types: list[str] = field(default_factory=list)
    num_target_layers: int | None = None
    """Target depth the drafter was trained against; ``None`` when the
    checkpoint omits the optional field."""
    mask_token_id: int = 0
    block_size: int = 0
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0

    def __post_init__(self) -> None:
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.target.target_layer_ids = list(self.target_layer_ids)
        # The rollback reads each linear layer's state-kernel inputs out of the
        # verify pass, and those captures cannot cross a subgraph boundary.
        self.target.use_subgraphs = False
        # The fused graph is text-only; a vision encoder would compile and
        # never be called.
        self.target.vision_config = None
        self.draft.return_hidden_states = ReturnHiddenStates.LAST

        if len(self.target.devices) != 1:
            raise ValueError(
                "DFlash2 supports a single device only: the drafter's body is"
                " unsharded and borrows the target's collective embedding and"
                f" head. Got {len(self.target.devices)} devices."
            )

    def validate_dflash2_fields(self) -> None:
        """Checks the drafter against the target it was trained to draft for.

        Every one of these is a silent-wrong-answer if it is wrong: a bad tap
        id reads a different layer, a hidden or vocab mismatch misreads the
        borrowed ``fc`` / ``lm_head`` contract, and a bad mask id drafts from
        an embedding row the drafter never saw.
        """
        n_target_layers = self.target.num_hidden_layers
        if not self.target_layer_ids:
            raise ValueError(
                "DFlash2 requires non-empty target_layer_ids (one per fc"
                " context feature)."
            )
        if len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError(
                "DFlash2 target_layer_ids must be distinct: the capture fires"
                " once per tapped layer, so a duplicate yields one fewer"
                f" column than fc expects. Got {self.target_layer_ids}."
            )
        if any(not 0 <= i < n_target_layers for i in self.target_layer_ids):
            raise ValueError(
                "DFlash2 target_layer_ids must index the target's layers,"
                f" [0, {n_target_layers}). Got {self.target_layer_ids}."
            )
        if (
            self.num_target_layers is not None
            and self.num_target_layers != n_target_layers
        ):
            raise ValueError(
                "DFlash2 draft declares it was trained against a target of"
                f" {self.num_target_layers} layers, but this target has"
                f" {n_target_layers}. The tap ids alone cannot catch this:"
                " they stay in range whenever the target is deep enough, so a"
                " drafter trained on a different depth would read the same"
                " layer indices off a different residual stream."
            )
        if self.draft.hidden_size != self.target.hidden_size:
            raise ValueError(
                "DFlash2 draft hidden_size must match the target's (the fc /"
                " block-embedding / lm_head contract). Got"
                f" draft={self.draft.hidden_size}"
                f" target={self.target.hidden_size}."
            )
        if self.draft.vocab_size != self.target.vocab_size:
            raise ValueError(
                "DFlash2 draft vocab must match the target's: the draft has no"
                " head of its own and borrows the target's."
                f" Got draft={self.draft.vocab_size}"
                f" target={self.target.vocab_size}."
            )
        if not 0 <= self.mask_token_id < self.target.vocab_size:
            raise ValueError(
                "DFlash2 mask_token_id must be in [0, target vocab_size)."
                f" Got mask_token_id={self.mask_token_id}"
                f" vocab_size={self.target.vocab_size}."
            )
        if self.block_size < 2:
            raise ValueError(
                "DFlash2 block_size must be at least 2 (an anchor slot plus at"
                f" least one mask slot). Got {self.block_size}."
            )
        if self.layer_types and len(self.layer_types) != (
            self.draft.num_hidden_layers
        ):
            raise ValueError(
                "DFlash2 layer_types must have one entry per draft layer. Got"
                f" {len(self.layer_types)} entries for"
                f" {self.draft.num_hidden_layers} layers."
            )

    @property
    def num_speculative_tokens(self) -> int:
        """Mask slots per block: the anchor slot never predicts."""
        return self.block_size - 1

    @property
    def devices(self) -> list[DeviceRef]:
        return list(self.target.devices)

    def get_kv_params(self) -> KVCacheParamInterface:
        return MultiKVCacheParams.from_params(
            {
                "target": self.target.get_kv_params(),
                "draft": self.draft_kv_params,
            }
        )

    def get_max_seq_len(self) -> int:
        return self.target.get_max_seq_len()

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        return Qwen3_5Config.calculate_max_seq_len(
            huggingface_config, model_config
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
        assert pipeline_config.speculative is not None
        draft_hf_config = pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None

        dflash_hf = parse_dflash2_draft_hf_config(draft_hf_config)
        speculative_config = pipeline_config.speculative
        resolved = resolve_dflash2_num_speculative_tokens(pipeline_config)
        if speculative_config.num_speculative_tokens is None:
            speculative_config = speculative_config.model_copy(
                update={"num_speculative_tokens": resolved}
            )

        target_config = Qwen3_5Config.initialize_from_config(
            pipeline_config,
            model_config.huggingface_config,
            model_config,
            max_seq_len=max_seq_len,
        )
        draft_max_length = pipeline_config.draft_model.max_length
        assert draft_max_length is not None
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config,
            draft_hf_config,
            pipeline_config.draft_model,
            max_seq_len=draft_max_length,
        )
        # ``initialize_from_config`` defaults the draft to gpu:0; pin it to the
        # target's device so the weights co-locate on a non-zero GPU.
        draft_config.devices = list(target_config.devices)
        draft_config.sliding_window = getattr(
            draft_hf_config, "sliding_window", None
        )
        # One leaf, shared: the draft module reads its geometry off
        # ``draft.kv_params`` while the KV manager reads ``draft_kv_params``,
        # and the two silently diverging is a wrong-shaped cache.
        draft_config.kv_params = construct_dflash2_draft_kv_params(
            pipeline_config, draft_config, target_config.kv_params
        )

        return cls(
            target=target_config,
            draft=draft_config,
            draft_kv_params=draft_config.kv_params,
            speculative_config=speculative_config,
            target_layer_ids=list(dflash_hf.target_layer_ids),
            layer_types=list(
                getattr(draft_hf_config, "layer_types", None) or []
            ),
            num_target_layers=dflash_hf.num_target_layers,
            mask_token_id=dflash_hf.mask_token_id,
            block_size=dflash_hf.block_size,
            conv_kernel_size=dflash_hf.conv_kernel_size,
            conv_group_size=dflash_hf.conv_group_size,
            selector_rank=dflash_hf.selector_rank,
            selector_top_k=dflash_hf.selector_top_k,
        )
