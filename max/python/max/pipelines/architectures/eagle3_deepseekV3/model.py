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
"""Eagle3 + DeepseekV3 PipelineModel: target + draft in one graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferValue, Graph, TensorValue, Value
from max.graph.weights import WeightData, load_weights
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates
from max.pipelines.lib.interfaces import (
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from transformers import AutoConfig
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from ..deepseekV3.model_config import DeepseekV3Config
from .batch_processor import Eagle3DeepseekV3BatchProcessor
from .unified_eagle import Eagle3DeepseekV3Unified
from .weight_adapters import convert_eagle3_draft_state_dict

logger = logging.getLogger("max.pipelines")


def extract_eagle_aux_layer_ids(
    hf_config: AutoConfig,
) -> list[int] | None:
    """Extract ``eagle_aux_hidden_state_layer_ids`` from a HuggingFace config.

    The IDs live inside an ``eagle_config`` sub-dict/object that is present on
    the *draft* checkpoint's config.
    """
    eagle_config = getattr(hf_config, "eagle_config", None)
    if eagle_config is None:
        return None
    raw = (
        eagle_config.get("eagle_aux_hidden_state_layer_ids", [])
        if isinstance(eagle_config, dict)
        else getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", [])
    )
    raw_list = list(raw)
    if not raw_list:
        return None
    if any(i <= 0 for i in raw_list):
        raise ValueError(
            "eagle_aux_hidden_state_layer_ids must contain positive ids "
            "(capturing layer-0's input = raw token embeddings is not yet "
            f"wired in MAX). Got {raw_list}."
        )
    return [i - 1 for i in raw_list]


@dataclass
class Eagle3DeepseekV3Inputs(UnifiedSpecDecodeInputs, DeepseekV3Inputs):
    """Inputs for the Eagle3 + DeepseekV3 unified model.

    Target-prefix fields come from :class:`DeepseekV3Inputs`; the spec-decode
    fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`. The eagle3_deepseekV3 graph does not bind
    ``in_thinking_phase``.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return super().buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False
        )


class Eagle3DeepseekV3Model(_UnifiedSpecDecodeModelMixin, DeepseekV3Model):
    """Eagle3 + DeepseekV3: target + draft in one compiled graph.

    Loads target weights from a DeepseekV3-shaped main checkpoint and draft
    weights from a separate Eagle3 checkpoint
    (``pipeline_config.draft_model``).
    """

    batch_processor_cls: ClassVar[type[Eagle3DeepseekV3BatchProcessor]] = (
        Eagle3DeepseekV3BatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _draft_config: DeepseekV3Config

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        target_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        self._draft_state_dict = convert_eagle3_draft_state_dict(
            dict(draft_weights.items()),
        )

        return target_state_dict

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV3Config:
        config = DeepseekV3Model._create_model_config(self, state_dict)

        # The target HF config doesn't carry eagle_config; propagate from draft.
        if config.eagle_aux_hidden_state_layer_ids is None:
            assert self.pipeline_config.draft_model is not None
            draft_hf = self.pipeline_config.draft_model.huggingface_config
            ids = extract_eagle_aux_layer_ids(draft_hf)
            if ids is None:
                raise ValueError(
                    "eagle_aux_hidden_state_layer_ids must be present in the "
                    "draft model's eagle_config for EAGLE3 hidden-state "
                    "capture, but was not found in the draft HF config."
                )
            config.eagle_aux_hidden_state_layer_ids = ids

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        draft_config = self._create_draft_config(config, self._draft_state_dict)
        if draft_config.ep_config is not None and config.ep_config is not None:
            draft_config.ep_config.node_id = config.ep_config.node_id

        assert isinstance(self.kv_params, KVCacheParams)
        target_kv_params = self.kv_params
        self._draft_kv_params = replace(target_kv_params, num_layers=1)
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv_params, "draft": self._draft_kv_params}
        )

        draft_config.return_hidden_states = ReturnHiddenStates.LAST
        self._draft_config = draft_config
        return config

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, DeepseekV3Config)
        assert self.pipeline_config.speculative is not None

        nn_model = Eagle3DeepseekV3Unified(
            model_config,
            self._draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        # Share embed_tokens before loading so the graph sees a single
        # Weight object for the shared embedding.  norm and lm_head are
        # loaded independently from the draft checkpoint.
        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens

        nn_model.target.load_state_dict(
            state_dict, weight_alignment=1, strict=True
        )
        nn_model.draft.load_state_dict(
            self._draft_state_dict, weight_alignment=1, strict=False
        )

        draft_expected = set(nn_model.draft.raw_state_dict().keys())
        draft_provided = set(self._draft_state_dict.keys())
        shared_prefixes = ("embed_tokens.",)
        missing = {
            k
            for k in draft_expected - draft_provided
            if not k.startswith(shared_prefixes)
        }
        extra = draft_provided - draft_expected
        if missing:
            raise ValueError(
                "Draft model has unloaded non-shared weights:"
                f" {sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        weights_registry = self._unified_weights_registry(nn_model)

        assert isinstance(self.kv_params, MultiKVCacheParams)
        kv_params = self.kv_params

        with Graph(
            "eagle3_deepseekV3_graph",
            input_types=nn_model.input_types(kv_params),
        ) as graph:
            (
                tokens,
                devices_input_row_offsets,
                host_input_row_offsets,
                return_n_logits,
                data_parallel_splits,
                *variadic_args,
            ) = graph.inputs

            variadic_args_iter = iter(variadic_args)
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            kv_collections, draft_kv_collections = (
                kv_params.unflatten_basic_kv_tree(variadic_args_iter)
            )

            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            target_ep_inputs: list[Value[Any]] | None = None
            if nn_model.target.ep_manager is not None:
                n_target_ep = len(nn_model.target.ep_manager.input_types())
                target_ep_inputs = [
                    next(variadic_args_iter) for _ in range(n_target_ep)
                ]

            draft_tokens = next(variadic_args_iter).tensor

            seed = next(variadic_args_iter).tensor
            temperature = next(variadic_args_iter).tensor
            top_k = next(variadic_args_iter).tensor
            max_k = next(variadic_args_iter).tensor
            top_p = next(variadic_args_iter).tensor
            min_top_p = next(variadic_args_iter).tensor

            # Optional bitmask triple — present only when
            # structured output is enabled (matches the
            # conditional in input_types()).
            pinned_bitmask_graph: TensorValue | None = None
            wait_payload_graph: BufferValue | None = None
            device_bitmask_scratch_graph: BufferValue | None = None
            if nn_model.enable_structured_output:
                pinned_bitmask_graph = next(variadic_args_iter).tensor
                wait_payload_graph = next(variadic_args_iter).buffer
                device_bitmask_scratch_graph = next(variadic_args_iter).buffer

            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=devices_input_row_offsets.tensor,
                draft_tokens=draft_tokens.tensor,
                signal_buffers=signal_buffers,
                kv_collections=kv_collections,
                return_n_logits=return_n_logits.tensor,
                host_input_row_offsets=host_input_row_offsets.tensor,
                data_parallel_splits=data_parallel_splits.tensor,
                batch_context_lengths=batch_context_lengths,
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                ep_inputs=target_ep_inputs,
                draft_kv_collections=draft_kv_collections,
                pinned_bitmask=pinned_bitmask_graph,
                wait_payload=wait_payload_graph,
                device_bitmask_scratch=device_bitmask_scratch_graph,
            )
            graph.output(*outputs)

        return graph, weights_registry

    def _unified_weights_registry(
        self, nn_model: Eagle3DeepseekV3Unified
    ) -> dict[str, Any]:
        """Build the combined registry with ``draft.*`` prefixes on draft-only weights."""
        assert nn_model.draft is not None
        # Capture concrete draft weights before renaming; ``state_dict()``
        # resets weight.name back to the module-path key.
        draft_weights_registry = nn_model.draft.state_dict()

        # Rename non-shared draft Weights so graph-level names are unique
        # (e.g. "draft.norm.weight" vs "norm.weight" from target).
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith("embed_tokens."):
                continue
            weight.name = f"draft.{name}"

        weights_registry = dict(nn_model.target.state_dict())
        for k, v in draft_weights_registry.items():
            if k.startswith("embed_tokens."):
                continue
            weights_registry[f"draft.{k}"] = v
        return weights_registry

    def _create_draft_config(
        self,
        target_config: DeepseekV3Config,
        draft_state_dict: dict[str, WeightData],
    ) -> DeepseekV3Config:
        """Create config for the Eagle3 draft model.

        Uses the target config as base but overrides rope_scaling from the
        draft's HF config and dtype/quant based on the draft checkpoint.
        """
        draft_config = DeepseekV3Config(
            **{
                f.name: getattr(target_config, f.name)
                for f in fields(target_config)
                if f.name in {ff.name for ff in fields(DeepseekV3Config)}
            }
        )

        # The draft may use different YarnRoPE parameters (e.g.
        # beta_fast=1.0 vs target's 32.0).
        assert self.pipeline_config.draft_model is not None
        draft_hf_config = self.pipeline_config.draft_model.huggingface_config
        if draft_hf_config is not None:
            draft_rope = getattr(draft_hf_config, "rope_scaling", None)
            if draft_rope is not None:
                draft_config.rope_scaling = draft_rope

        # Avoid mutating the target's ep_config (shallow-copied from target).
        if draft_config.ep_config is not None:
            draft_config.ep_config = replace(draft_config.ep_config)

        # Eagle3 draft has BF16 dense MLP (not quantized, not MoE)
        if (
            draft_config.quant_config is not None
            and draft_config.quant_config.is_fp4
            and not any("weight_scale" in key for key in draft_state_dict)
        ):
            logger.info(
                "Eagle3 draft weights are BF16 (no weight_scale found); "
                "disabling FP4 config for draft."
            )
            draft_config.quant_config = None
            draft_config.dtype = DType.bfloat16
            if draft_config.ep_config is not None:
                draft_config.ep_config.dispatch_dtype = DType.bfloat16
                draft_config.ep_config.dispatch_quant_config = None

        return draft_config
