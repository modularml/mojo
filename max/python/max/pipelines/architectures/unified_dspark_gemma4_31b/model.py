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
"""Unified speculators-DSpark Gemma4 PipelineModel: target + draft in one
graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import KVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import TextContext
from max.pipelines.kv_cache.config import cache_dtype_for_encoding
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    PipelineConfig,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib._hf_config import PretrainedConfig
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.interfaces.pipeline_model import (
    AlwaysSignalBuffersMixin,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..speculators_common.draft_config import construct_draft_kv_params
from ..speculators_common.weight_adapters import (
    merge_unified_state_dict,
    validate_draft_checkpoint_weights,
)
from .batch_processor import UnifiedDSparkGemma4_31BBatchProcessor
from .model_config import UnifiedDSparkGemma4_31BConfig
from .unified_dspark_gemma4_31b import (
    UnifiedDSparkGemma4_31B as UnifiedDSparkGemma4_31BModule,
)

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedDSparkGemma4_31BInputs(UnifiedSpecDecodeInputs):
    """Inputs for the unified speculators-DSpark Gemma4 graph.

    The spec-decode fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`; ``tokens`` / ``input_row_offsets`` /
    ``return_n_logits`` / ``signal_buffers`` plus the KV cache tree form this
    single-device graph's prefix. Signal buffers are bound because Gemma4's
    embedding and lm_head layers use collectives even on one device. The
    tail binds ``in_thinking_phase`` unconditionally and the structured-output
    bitmask triple when ``structured_output`` is set — the flags here must
    match the module's ``SpecDecodeInputTypeSpec``.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    signal_buffers: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        buffers = (
            self.tokens,
            self.input_row_offsets,
            self.return_n_logits,
            *self.signal_buffers,
            *(self.kv_cache_inputs.flatten() if self.kv_cache_inputs else ()),
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedDSparkGemma4_31BModel(
    _UnifiedSpecDecodeModelMixin,
    AlwaysSignalBuffersMixin,
    GraphPipelineModelWithKVCache[TextContext],
):
    """Unified speculators-DSpark Gemma4: target + draft in one compiled
    graph."""

    model_config_cls: ClassVar[type[Any]] = UnifiedDSparkGemma4_31BConfig
    batch_processor_cls: ClassVar[
        type[UnifiedDSparkGemma4_31BBatchProcessor]
    ] = UnifiedDSparkGemma4_31BBatchProcessor

    model: Model
    _draft_state_dict: dict[str, Any]

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        *,
        memory_plan: MemoryPlan,
        adapter: WeightsAdapter | None = None,
        return_logits: ReturnLogits = ReturnLogits.LAST_TOKEN,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.SELECTED_LAYERS,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.model = self.load_model(session)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: PretrainedConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> MultiKVCacheParams:
        return Gemma4ForConditionalGenerationConfig.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def _load_state_dict(self) -> dict[str, Any]:
        target_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        # Speculators checkpoint keys carry no ``model.`` prefix and match
        # the draft module's names 1:1; no conversion beyond materialization.
        self._draft_state_dict = {
            name: weight.data() for name, weight in draft_weights.items()
        }

        return target_state_dict

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> UnifiedDSparkGemma4_31BConfig:
        unified_config = UnifiedDSparkGemma4_31BConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        target_hf_config = self.huggingface_config
        assert target_hf_config is not None
        unified_config.target.finalize(
            huggingface_config=target_hf_config,
            state_dict=state_dict,
            return_logits=ReturnLogits.VARIABLE,
        )

        # ``kv_params.num_draft_tokens`` was baked from the CLI
        # num_speculative_tokens at ``PipelineModelWithKVCache.__init__``,
        # BEFORE the block-size override above. Re-derive every leaf (both
        # target leaves and the draft leaf) so the KV manager reserves the
        # true worst-case per-step write.
        corrected_target_kv = (
            Gemma4ForConditionalGenerationConfig.construct_kv_params(
                target_hf_config,
                self.pipeline_config,
                self.device_refs,
                self.kv_cache_config,
                # The config carries only the raw user value (None unless
                # set explicitly); resolve against the arch default like
                # Gemma4TextConfig.initialize_from_config does, or the cache
                # leaf silently becomes float32 against a bfloat16 graph.
                cache_dtype_for_encoding(
                    _select_quantization_encoding(
                        self.pipeline_config.model,
                        UnifiedDSparkGemma4_31BConfig.DEFAULT_ENCODING,
                    ),
                    self.pipeline_config.model.kv_cache.kv_cache_format,
                ),
            )
        )
        # The draft leaf writes the block forward's effective_block_size
        # slots past the post-commit length in addition to the materialized
        # verify block, so its per-step headroom is num_speculative_tokens
        # + 1. MultiKVCacheParams requires one num_draft_tokens across the
        # tree, and the target verify itself writes the same count per
        # step, so every leaf carries effective_block_size.
        corrected_target_children: dict[str, KVCacheParams] = {}
        for name, leaf in corrected_target_kv.children.items():
            assert isinstance(leaf, KVCacheParams)
            corrected_target_children[name] = replace(
                leaf, num_draft_tokens=unified_config.effective_block_size
            )
        corrected_target_kv = MultiKVCacheParams.from_params(
            corrected_target_children
        )
        draft_kv_params = construct_draft_kv_params(
            self.pipeline_config,
            unified_config.draft,
            list(unified_config.target.devices),
            num_draft_tokens=unified_config.effective_block_size,
        )
        unified_config.target.kv_params = corrected_target_kv
        unified_config.draft_kv_params = draft_kv_params
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": corrected_target_kv, "draft": draft_kv_params}
        )

        return unified_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, UnifiedDSparkGemma4_31BConfig)

        nn_model = UnifiedDSparkGemma4_31BModule(
            model_config,
            enable_structured_output=(
                self.pipeline_config.needs_bitmask_constraints
            ),
        )

        validate_draft_checkpoint_weights(
            self._draft_state_dict, model_config.draft
        )

        # Unlike the dense-DSpark pipeline there is no module aliasing: the
        # draft owns its full-vocab embedding (raw, unscaled rows) and its
        # pruned-vocab lm_head, both loaded verbatim from the checkpoint's
        # frozen copies.
        unified_state_dict = merge_unified_state_dict(
            state_dict, self._draft_state_dict
        )

        # strict=False: a tied-embedding target has no target.lm_head.*
        # checkpoint key. load_state_dict(strict=False) reports nothing, so
        # audit name coverage explicitly below.
        nn_model.load_state_dict(
            unified_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        self._audit_state_dict_names(
            nn_model,
            unified_state_dict,
            tied_target_lm_head=model_config.target.tie_word_embeddings,
        )
        weights_registry = nn_model.state_dict()

        with Graph(
            "unified_dspark_gemma4_31b",
            input_types=nn_model.input_types(),
        ) as graph:
            inputs = nn_model._unflatten_graph_inputs(graph.inputs)
            outputs = nn_model(inputs)
            graph.output(*outputs)

        return graph, weights_registry

    def _audit_state_dict_names(
        self,
        nn_model: UnifiedDSparkGemma4_31BModule,
        unified_state_dict: dict[str, Any],
        *,
        tied_target_lm_head: bool,
    ) -> None:
        """Set-difference audit of provided vs expected weight names.

        ``load_state_dict(strict=False)`` is silent about mismatches; the
        only names allowed to differ are, for tied-embedding targets, the
        target lm_head that shares the embedding weight.
        """
        expected = set(nn_model.raw_state_dict().keys())
        provided = set(unified_state_dict.keys())
        allowed_missing_prefixes: tuple[str, ...] = ()
        if tied_target_lm_head:
            allowed_missing_prefixes += ("target.lm_head.",)
        missing = {
            name
            for name in expected - provided
            if not name.startswith(allowed_missing_prefixes)
        }
        extra = provided - expected
        if missing:
            raise ValueError(
                "Unified speculators-DSpark model has unloaded weights:"
                f" {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                "Unified speculators-DSpark state dict has unused keys:"
                f" {sorted(extra)}"
            )
