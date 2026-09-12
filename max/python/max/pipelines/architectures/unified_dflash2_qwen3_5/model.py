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
"""Qwen3.5-with-DFlash2 PipelineModel: target, block drafter, state rollback."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferValue, DeviceRef, Graph, TensorValue
from max.graph.weights import Weights, WeightsAdapter, load_weights
from max.nn.kv_cache import (
    KVCacheInputs,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.qwen3vl_moe.context import (
    Qwen3VLTextAndVisionContext,
)
from max.pipelines.lib import (
    GraphPipelineModelWithKVCache,
    KVCacheConfig,
    PipelineConfig,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib._hf_config import PretrainedConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    AlwaysSignalBuffersMixin,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights

from ..llama3.model_config import Llama3Config
from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from ..qwen3_5.model import _SCALE_SUFFIXES
from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.state_cache import attn_cache
from .batch_processor import UnifiedDflash2Qwen3_5BatchProcessor
from .model_config import (
    UnifiedDflash2Qwen3_5Config,
    construct_dflash2_draft_kv_params,
)
from .unified_dflash2_qwen3_5 import UnifiedDflash2Qwen3_5

logger = logging.getLogger("max.pipelines")

GRAPH_NAME = "qwen3_5_with_dflash2_graph"
"""Exported submodel name; the Mach spec-step executor selects it by name."""

_DRAFT_PREFIX = "draft."
_TARGET_PREFIX = "target."


@dataclass
class UnifiedDflash2Qwen3_5Inputs(UnifiedSpecDecodeInputs):
    """Inputs for the fused Qwen3.5 DFlash2 graph.

    Identical to the Qwen3.5 MTP graph's packing: the canonical spec-decode
    prefix and tail, then this target's state-pool tail. Only the draft KV
    leaf's shapes differ between the two graphs.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]
    live_conv_pools: list[Buffer]
    live_recurrent_pools: list[Buffer]
    live_conv_row_ids: list[Buffer]
    live_recurrent_row_ids: list[Buffer]
    shadow_conv_pools: list[Buffer]
    shadow_recurrent_pools: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.batch_context_lengths,
        )
        return (
            prefix
            + self._spec_decode_tail_buffers(include_in_thinking_phase=True)
            + (
                *self.live_conv_pools,
                *self.live_recurrent_pools,
                *self.live_conv_row_ids,
                *self.live_recurrent_row_ids,
                *self.shadow_conv_pools,
                *self.shadow_recurrent_pools,
            )
        )


class UnifiedDflash2Qwen3_5Model(
    _UnifiedSpecDecodeModelMixin,
    AlwaysSignalBuffersMixin,
    GraphPipelineModelWithKVCache[Qwen3VLTextAndVisionContext],
):
    """Qwen3.5 with a DFlash2 block drafter, in one compiled graph."""

    model_config_cls: ClassVar[type[Any]] = UnifiedDflash2Qwen3_5Config
    batch_processor_cls: ClassVar[type[UnifiedDflash2Qwen3_5BatchProcessor]] = (
        UnifiedDflash2Qwen3_5BatchProcessor
    )

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
            adapter,
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
        """The target's full-attention leaf plus the drafter's windowed one.

        Called during memory planning, before ``_create_model_config``; both
        must agree, so both go through the same two constructors.
        """
        assert pipeline_config.draft_model is not None
        draft_hf_config = pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None
        target_kv = Qwen3_5Config.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        draft_max_length = pipeline_config.draft_model.max_length
        assert draft_max_length is not None
        draft_config = Llama3Config.initialize_from_config(
            pipeline_config,
            draft_hf_config,
            pipeline_config.draft_model,
            max_seq_len=draft_max_length,
        )
        draft_config.devices = list(devices)
        return MultiKVCacheParams.from_params(
            {
                "target": target_kv,
                "draft": construct_dflash2_draft_kv_params(
                    pipeline_config, draft_config, attn_cache(target_kv)
                ),
            }
        )

    def _load_state_dict(self) -> dict[str, Any]:
        """Loads the target through its adapter and the drafter beside it."""
        target_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_weights = load_weights(draft_model_config.resolved_weight_paths())
        draft_hf_config = draft_model_config.huggingface_config
        assert draft_hf_config is not None
        # The drafter's 81 checkpoint names already match its module tree; this
        # only resolves the draft's own encoding (bf16 under an NVFP4 target).
        self._draft_state_dict = _convert_safetensor_with_model_config(
            dict(draft_weights.items()),
            draft_hf_config,
            draft_model_config,
        )
        return target_state_dict

    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> UnifiedDflash2Qwen3_5Config:
        unified_config = UnifiedDflash2Qwen3_5Config.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        unified_config.validate_dflash2_fields()

        target_hf_config = self.huggingface_config
        assert target_hf_config is not None
        unified_config.target.finalize(
            huggingface_config=Qwen3_5Config._get_text_config(target_hf_config),
            state_dict=state_dict,
            return_logits=ReturnLogits.VARIABLE,
        )
        unified_config.target.tie_word_embeddings = getattr(
            target_hf_config, "tie_word_embeddings", False
        )
        assert self.pipeline_config.draft_model is not None
        draft_hf_config = self.pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None
        unified_config.draft.finalize(
            huggingface_config=draft_hf_config,
            state_dict=self._draft_state_dict,
            return_logits=ReturnLogits.LAST_TOKEN,
            return_hidden_states=ReturnHiddenStates.LAST,
        )

        self.kv_params = unified_config.get_kv_params()
        return unified_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, UnifiedDflash2Qwen3_5Config)
        if not self.pipeline_config.needs_bitmask_constraints:
            raise ValueError(
                "Qwen3.5 DFlash2 needs the constrained-decoding bitmask input:"
                " this checkpoint's lm_head has live padding rows past"
                " sampleable_vocab_size and the in-graph acceptance sampler"
                " excludes them only through that mask. Exporting a MEF with"
                " mach/tools/gen-mef turns it on by default and"
                " --no-sampler-grammar turns it off; elsewhere it follows"
                " --enable-structured-output (or a tool parser that implies"
                " it)."
            )

        nn_model = UnifiedDflash2Qwen3_5(
            model_config,
            enable_structured_output=(
                self.pipeline_config.needs_bitmask_constraints
            ),
        )

        full_state_dict = _merge_state_dicts(state_dict, self._draft_state_dict)
        _check_weights_match(
            expected=set(nn_model.raw_state_dict().keys()),
            provided=set(full_state_dict.keys()),
            tie_word_embeddings=model_config.target.tie_word_embeddings,
        )
        nn_model.load_state_dict(
            full_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = nn_model.state_dict()
        self.state_dict = weights_registry

        kv_params = self.kv_params
        assert isinstance(kv_params, MultiKVCacheParams)
        num_devices = len(self.devices)

        with Graph(
            GRAPH_NAME, input_types=nn_model.input_types(kv_params)
        ) as graph:
            (
                tokens,
                input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *rest,
            ) = graph.inputs
            it = iter(rest)
            signal_buffers = [next(it).buffer for _ in range(num_devices)]

            kv_tree = kv_params.unflatten_kv_inputs(it)
            assert isinstance(kv_tree, MultiKVCacheInputs)
            target_leaf = kv_tree.children["target"]
            draft_leaf = kv_tree.children["draft"]
            assert isinstance(target_leaf, KVCacheInputs)
            assert isinstance(draft_leaf, KVCacheInputs)

            # Consumed by the canonical signature but unused: Qwen3.5 has no
            # sparse-attention budget to bound.
            for _ in range(num_devices):
                next(it)

            draft_tokens = next(it).tensor
            seed = next(it).tensor
            temperature = next(it).tensor
            top_k = next(it).tensor
            max_k = next(it).tensor
            top_p = next(it).tensor
            min_top_p = next(it).tensor
            in_thinking_phase = next(it).tensor

            pinned_bitmask: TensorValue | None = None
            wait_payload: BufferValue | None = None
            device_bitmask_scratch: BufferValue | None = None
            if nn_model.enable_structured_output:
                pinned_bitmask = next(it).tensor
                wait_payload = next(it).buffer
                device_bitmask_scratch = next(it).buffer

            def per_device_buffers() -> list[BufferValue]:
                return [next(it).buffer for _ in range(num_devices)]

            def per_device_tensors() -> list[TensorValue]:
                return [next(it).tensor for _ in range(num_devices)]

            live_conv_pools = per_device_buffers()
            live_recurrent_pools = per_device_buffers()
            live_conv_row_ids = per_device_tensors()
            live_recurrent_row_ids = per_device_tensors()
            shadow_conv_pools = per_device_buffers()
            shadow_recurrent_pools = per_device_buffers()
            sentinel = object()
            assert next(it, sentinel) is sentinel, (
                "input_types() and the graph unflatten disagree: unconsumed"
                " graph inputs remain"
            )

            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=input_row_offsets.tensor,
                draft_tokens=draft_tokens,
                signal_buffers=signal_buffers,
                target_kv=list(target_leaf.inputs),
                draft_kv=list(draft_leaf.inputs),
                return_n_logits=return_n_logits.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                data_parallel_splits=data_parallel_splits.tensor,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                live_conv_pools=live_conv_pools,
                live_recurrent_pools=live_recurrent_pools,
                live_conv_row_ids=live_conv_row_ids,
                live_recurrent_row_ids=live_recurrent_row_ids,
                shadow_conv_pools=shadow_conv_pools,
                shadow_recurrent_pools=shadow_recurrent_pools,
                pinned_bitmask=pinned_bitmask,
                wait_payload=wait_payload,
                device_bitmask_scratch=device_bitmask_scratch,
            )
            graph.output(*outputs)

        return graph, weights_registry


def _merge_state_dicts(
    target: Mapping[str, Any], draft: Mapping[str, Any]
) -> dict[str, Any]:
    """Prefixes both checkpoints into the one flat namespace the graph declares.

    Both halves name their decoder stack ``layers.N.``, so the module path is
    the only thing that tells them apart. The drafter ships no
    ``embed_tokens`` / ``lm_head`` -- it calls the target's -- so nothing is
    skipped here; a draft checkpoint that shipped one would be reported as an
    unconsumed key rather than silently loaded into a second copy.
    """
    merged: dict[str, Any] = {
        f"{_TARGET_PREFIX}{name}": value for name, value in target.items()
    }
    merged.update(
        {f"{_DRAFT_PREFIX}{name}": value for name, value in draft.items()}
    )
    return merged


def _check_weights_match(
    expected: set[str], provided: set[str], *, tie_word_embeddings: bool
) -> None:
    """Fails the load rather than letting ``strict=False`` drop a mismatch.

    A mis-named draft tensor loads as nothing and shows up only as collapsed
    acceptance, so the draft side is checked in both directions: every
    ``draft.*`` name the graph declares must be supplied, and every one
    supplied must be consumed. Unconsumed target-side tensors keep the base
    architecture's treatment -- a hard failure for quantization scales, whose
    silent loss would leave a quantized layer reading garbage, and a warning
    for the rest (a Qwen3.8 checkpoint also ships the MTP head this graph does
    not use).
    """
    allowed_missing: tuple[str, ...] = ()
    if tie_word_embeddings:
        allowed_missing += (f"{_TARGET_PREFIX}lm_head.",)
    missing = sorted(
        name
        for name in expected - provided
        if not name.startswith(allowed_missing)
    )
    if missing:
        raise ValueError(
            f"Qwen3.5 DFlash2 graph is missing {len(missing)} weight(s): "
            f"{missing[:20]}"
        )

    unused = provided - expected
    unused_draft = sorted(k for k in unused if k.startswith(_DRAFT_PREFIX))
    if unused_draft:
        raise ValueError(
            f"DFlash2 draft checkpoint supplies {len(unused_draft)} tensor(s) "
            f"the fused graph does not consume, so this drafter is not the one "
            f"it implements: {unused_draft[:20]}"
        )
    unused_scales = sorted(k for k in unused if k.endswith(_SCALE_SUFFIXES))
    if unused_scales:
        raise ValueError(
            f"Qwen3.5 DFlash2 checkpoint supplies {len(unused_scales)} "
            f"quantization scale tensor(s) that no layer consumes: "
            f"{unused_scales[:20]}"
        )
    if remaining := sorted(unused - set(unused_scales)):
        logger.warning(
            "Qwen3.5 DFlash2 load_state_dict: %d unused checkpoint keys: %s",
            len(remaining),
            remaining[:20],
        )
