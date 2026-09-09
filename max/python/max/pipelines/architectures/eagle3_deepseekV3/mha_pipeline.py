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
"""Eagle3 MHA-draft + DeepseekV3 (MLA target) PipelineModel.

Sibling of :class:`Eagle3DeepseekV3Model` for the case where the draft is a
Llama-style MHA Eagle3 head (``LlamaForCausalLMEagle3``) over a bare
``DeepseekV3ForCausalLM`` target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferValue, Graph, TensorValue, Value
from max.graph.weights import load_weights
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.lib.interfaces import UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs, DeepseekV3Model
from ..deepseekV3.model_config import DeepseekV3Config
from ..eagle_common.eagle_mha_draft import Eagle3MHADraftConfig
from ..kimik2_5.unified_eagle_mha_model import Eagle3MHAKimiK25Unified
from ..kimik2_5.unified_eagle_mha_pipeline_model import (
    _build_mha_draft_config,
    _infer_fc_input_multiplier,
)
from ..kimik2_5.weight_adapters import convert_llama_eagle3_draft_state_dict
from .batch_processor import Eagle3MHADeepseekV3BatchProcessor
from .model import extract_eagle_aux_layer_ids

logger = logging.getLogger("max.pipelines")


@dataclass
class Eagle3MHADeepseekV3Inputs(UnifiedSpecDecodeInputs, DeepseekV3Inputs):
    """Inputs for the Eagle3 MHA-draft + DeepseekV3 unified model.

    The draft owns its own MHA ``KVCacheInputs`` (separate from the target's
    MLA cache) as the ``"draft"`` leaf of the unified ``{"target", "draft"}``
    KV tree, so MHA dispatch metadata is plumbed at both prefill
    (q_max_seq_len = ``1 + num_speculative_tokens``) and decode
    (q_max_seq_len = 1) widths. The spec-decode fields and trailing buffer
    packing come from :class:`UnifiedSpecDecodeInputs`; the graph binds the
    per-row ``in_thinking_phase`` flag and the structured-output bitmask
    triple.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return super().buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class Eagle3MHADeepseekV3Model(_UnifiedSpecDecodeModelMixin, DeepseekV3Model):
    """Eagle3 MHA-draft + DeepseekV3: target + draft in one compiled graph."""

    batch_processor_cls: ClassVar[type[Eagle3MHADeepseekV3BatchProcessor]] = (
        Eagle3MHADeepseekV3BatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _draft_config: Eagle3MHADraftConfig

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        target_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_hf = draft_model_config.huggingface_config
        assert draft_hf is not None

        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        self._draft_state_dict = convert_llama_eagle3_draft_state_dict(
            dict(draft_weights.items())
        )

        return target_state_dict

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV3Config:
        config = DeepseekV3Model._create_model_config(self, state_dict)

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_hf = draft_model_config.huggingface_config
        assert draft_hf is not None

        fc_input_multiplier = _infer_fc_input_multiplier(
            self._draft_state_dict, hidden_size=int(draft_hf.hidden_size)
        )

        if config.eagle_aux_hidden_state_layer_ids is None:
            ids = extract_eagle_aux_layer_ids(draft_hf)
            if ids is None:
                target_layers = int(config.num_hidden_layers)
                ids = _default_aux_layer_ids(target_layers, fc_input_multiplier)
                logger.warning(
                    "Draft HF config has no "
                    "'eagle_config.eagle_aux_hidden_state_layer_ids'. "
                    "Falling back to evenly-spaced defaults for "
                    f"{fc_input_multiplier}-way fusion over {target_layers} "
                    f"target layers: {ids}. Override by setting "
                    "``eagle_config.eagle_aux_hidden_state_layer_ids`` on "
                    "the draft HF config if the training-time IDs are "
                    "known."
                )
            config.eagle_aux_hidden_state_layer_ids = ids

        n_target_capture = len(config.eagle_aux_hidden_state_layer_ids)
        if n_target_capture != fc_input_multiplier:
            raise ValueError(
                f"Draft fc fuses {fc_input_multiplier} target hidden states "
                f"but the target is configured to capture {n_target_capture} "
                f"(eagle_aux_hidden_state_layer_ids="
                f"{config.eagle_aux_hidden_state_layer_ids})."
            )

        assert isinstance(self.kv_params, KVCacheParams)
        target_kv = self.kv_params
        if target_kv.dtype != DType.bfloat16:
            logger.warning(
                "Draft KV cache dtype forced to bfloat16 (target uses "
                f"{target_kv.dtype}). MHA flash attention requires q/k/v "
                "and KV cache dtypes to match; the draft module emits bf16."
            )
        self._draft_kv_params = self.pipeline_config.model.kv_cache.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=int(draft_hf.num_key_value_heads),
            head_dim=int(draft_hf.head_dim),
            num_layers=1,
            devices=target_kv.devices,
            data_parallel_degree=target_kv.data_parallel_degree,
            is_mla=False,
            num_q_heads=int(draft_hf.num_attention_heads),
            speculative_method=target_kv.speculative_method,
            num_draft_tokens=target_kv.num_draft_tokens,
        )
        # The model owns the unified ``{"target", "draft"}`` tree
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv, "draft": self._draft_kv_params}
        )

        self._draft_config = _build_mha_draft_config(
            draft_hf,
            target_config=config,
            devices=config.devices,
            data_parallel_degree=config.data_parallel_degree,
            fc_input_multiplier=fc_input_multiplier,
            kv_params=self._draft_kv_params,
            sliding_window_override=draft_model_config.sliding_window,
        )
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

        nn_model = Eagle3MHAKimiK25Unified(
            model_config,
            self._draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

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
                f"Draft model has unloaded non-shared weights: {sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        draft_weights_registry = nn_model.draft.state_dict()
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith("embed_tokens."):
                continue
            weight.name = f"draft.{name}"

        weights_registry = dict(nn_model.target.state_dict())
        for k, v in draft_weights_registry.items():
            if k.startswith("embed_tokens."):
                continue
            weights_registry[f"draft.{k}"] = v

        assert isinstance(self.kv_params, MultiKVCacheParams)
        kv_params = self.kv_params

        with Graph(
            "eagle3_mha_deepseekV3_graph",
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

            # Unflatten the unified KV-cache tree ``{"target", "draft"}``
            kv_caches_per_dev, draft_kv_collections = (
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
            in_thinking_phase = next(variadic_args_iter).tensor

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
                kv_collections=kv_caches_per_dev,
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
                in_thinking_phase=in_thinking_phase,
                ep_inputs=target_ep_inputs,
                draft_kv_collections=draft_kv_collections,
                pinned_bitmask=pinned_bitmask_graph,
                wait_payload=wait_payload_graph,
                device_bitmask_scratch=device_bitmask_scratch_graph,
            )
            graph.output(*outputs)

        return graph, weights_registry


def _default_aux_layer_ids(
    num_target_layers: int, fc_input_multiplier: int
) -> list[int]:
    """Picks ``fc_input_multiplier`` aux capture layer IDs spread across
    a target with ``num_target_layers`` layers.

    Fallback used when the draft HF config doesn't declare its
    training-time aux IDs. Shifted one layer below vLLM's
    ``SupportsEagle3.get_eagle3_default_aux_hidden_state_layers`` default
    (``(1, num_layers // 2 - 1, num_layers - 4)`` for 3-way) — empirically
    a better match for the ``modularai/kimi-k2.5-eagle3`` checkpoint.
    """
    if num_target_layers <= 4:
        raise ValueError(
            f"Default aux layer IDs require >4 target layers, got "
            f"{num_target_layers}."
        )
    early = 1
    late = num_target_layers - 4
    if fc_input_multiplier == 2:
        return [early, late]
    if fc_input_multiplier == 3:
        return [early, num_target_layers // 2 - 1, late]
    raise ValueError(
        f"Unsupported fc_input_multiplier={fc_input_multiplier}; "
        "expected 2 or 3."
    )
