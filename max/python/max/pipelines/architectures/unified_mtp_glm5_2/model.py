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
"""GLM-5.2 (DeepSeek-V3.2 sparse) with MTP PipelineModel: target + draft graph."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferValue, Graph, TensorValue, Value
from max.graph.weights import WeightData
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheParams,
    MultiKVCacheInputs,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.lib import UnifiedSpecDecodeInputs
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from typing_extensions import override

from ..deepseekV3.model import DeepseekV3Inputs
from ..deepseekV3_2.model import DeepseekV3_2Model
from ..deepseekV3_2.model_config import DeepseekV3_2Config
from ..deepseekV3_2_nextn.model_config import DeepseekV3_2NextNConfig
from ..glm5_1.model import Glm5_1Model
from .batch_processor import UnifiedMTPGlm5_2BatchProcessor
from .unified_mtp_glm5_2 import UnifiedMTPGlm5_2

logger = logging.getLogger("max.pipelines")


def _subtree_quantized(
    draft_state_dict: dict[str, WeightData], subtree: str
) -> bool:
    """Whether the draft's ``subtree`` ships a quantized ``weight_scale``.

    A projection is quantized iff the checkpoint provides a ``weight_scale``
    companion for it. Used to detect the MTP layer's per-subtree quantization
    (e.g. NVFP4 leaves the whole MTP layer in bf16, with no scales).
    """
    return any(
        subtree in key and key.endswith(".weight_scale")
        for key in draft_state_dict
    )


@dataclass
class UnifiedMTPGlm5_2Inputs(UnifiedSpecDecodeInputs, DeepseekV3Inputs):
    """Inputs for the UnifiedMTPGlm5_2 model.

    Target-prefix fields come from :class:`DeepseekV3Inputs`; the spec-decode
    fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`. The MTP graph binds the per-row
    ``in_thinking_phase`` flag (consumed by relaxed acceptance).
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return super().buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedMTPGlm5_2Model(_UnifiedSpecDecodeModelMixin, Glm5_1Model):
    """GLM-5.2 with MTP: merge + V3.2 target + rejection + sparse draft."""

    batch_processor_cls: ClassVar[type[UnifiedMTPGlm5_2BatchProcessor]] = (
        UnifiedMTPGlm5_2BatchProcessor
    )

    _draft_state_dict: dict[str, Any]
    _draft_config: DeepseekV3_2NextNConfig

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.ALL_NORMALIZED
        super().__init__(*args, **kwargs)

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        if self.adapter:
            raw_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            raw_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        self._draft_state_dict = {
            k[len("draft.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("draft.")
        }
        if (
            "shared_head_norm.weight" not in self._draft_state_dict
            and "target.norm.weight" in raw_state_dict
        ):
            # Some checkpoints share shared_head_norm with the base model's
            # final norm and don't emit it as a draft weight.
            self._draft_state_dict["shared_head_norm.weight"] = raw_state_dict[
                "target.norm.weight"
            ]

        return {
            k[len("target.") :]: v
            for k, v in raw_state_dict.items()
            if k.startswith("target.")
        }

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> DeepseekV3_2Config:
        config = DeepseekV3_2Model._create_model_config(self, state_dict)

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        # Build the nested {target: {mla, indexer}, draft: {mla, indexer}}
        # KV tree. The draft caches store a single layer.
        assert isinstance(self.kv_params, MultiKVCacheParams)
        target_kv = self.kv_params
        target_mla_params = target_kv.children["mla"]
        target_indexer_params = target_kv.children["indexer"]
        assert isinstance(target_mla_params, KVCacheParams)
        assert isinstance(target_indexer_params, KVCacheParams)
        draft_kv = MultiKVCacheParams.from_params(
            {
                "mla": replace(target_mla_params, num_layers=1),
                "indexer": replace(target_indexer_params, num_layers=1),
            }
        )

        draft_config = self._create_draft_config(
            self._draft_state_dict, draft_kv
        )

        if draft_config.ep_config is not None and config.ep_config is not None:
            draft_config.ep_config.node_id = config.ep_config.node_id

        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv, "draft": draft_kv}
        )

        draft_config.return_hidden_states = ReturnHiddenStates.LAST
        self._draft_config = draft_config
        return config

    @override
    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        assert isinstance(model_config, DeepseekV3_2Config)
        self.ep_comm_initializer = None
        self.draft_ep_comm_initializer = None

        if model_config.ep_config is None or is_virtual_device_mode():
            return

        # NVFP4 leaves the MTP layer's routed experts in bf16 (no
        # ``.weight_scale``), so the draft dispatches through EP in bf16 —
        # wider than an NVFP4 target dispatch, so the shared EP buffers must
        # be sized for the bf16 case.
        draft_moe_dispatches_bf16 = not _subtree_quantized(
            self._draft_state_dict, ".mlp.experts."
        )

        ep_alloc_config = model_config.ep_config
        if draft_moe_dispatches_bf16:
            ep_alloc_config = replace(
                model_config.ep_config,
                dispatch_dtype=DType.bfloat16,
                dispatch_quant_config=None,
                fused_shared_expert=model_config.n_shared_experts == 1,
            )
            logger.info("Upsizing shared EP buffers for bf16 draft dispatch.")

        self.ep_comm_initializer = EPCommInitializer(ep_alloc_config)
        self.ep_comm_initializer.ep_init(session)
        model_config.ep_config.node_id = ep_alloc_config.node_id
        if model_config.ep_config.node_id == -1:
            raise ValueError(
                "EP node ID is not set. Please check if the EP "
                "initialization is successful."
            )
        self.draft_ep_comm_initializer = self.ep_comm_initializer

    @override
    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        del session
        assert isinstance(model_config, DeepseekV3_2Config)
        assert self.pipeline_config.speculative is not None

        nn_model = UnifiedMTPGlm5_2(
            model_config,
            self._draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        # Share embed_tokens and lm_head BEFORE loading so state_dict()
        # deduplicates them — the adapter only emits target.* copies.
        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens
        nn_model.draft.lm_head = nn_model.target.lm_head

        nn_model.target.load_state_dict(
            state_dict, weight_alignment=1, strict=True
        )
        # strict=False because shared weights (embed_tokens, lm_head) are
        # aliased to target's and won't have keys in draft_state_dict.
        nn_model.draft.load_state_dict(
            self._draft_state_dict, weight_alignment=1, strict=False
        )

        draft_expected = set(nn_model.draft.raw_state_dict().keys())
        draft_provided = set(self._draft_state_dict.keys())
        shared_prefixes = ("embed_tokens.", "lm_head.")
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

        weights_registry = {
            **nn_model.draft.state_dict(),
            **nn_model.target.state_dict(),
        }

        assert isinstance(self.kv_params, MultiKVCacheParams)
        kv_params = self.kv_params

        with Graph(
            "glm5_2_with_mtp_graph",
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

            kv_tree = kv_params.unflatten_kv_inputs(variadic_args_iter)
            assert isinstance(kv_tree, MultiKVCacheInputs)
            target_tree = kv_tree.children["target"]
            draft_tree = kv_tree.children["draft"]
            assert isinstance(target_tree, MultiKVCacheInputs)
            assert isinstance(draft_tree, MultiKVCacheInputs)
            target_mla = target_tree.children["mla"]
            target_indexer = target_tree.children["indexer"]
            draft_mla = draft_tree.children["mla"]
            draft_indexer = draft_tree.children["indexer"]
            assert isinstance(target_mla, KVCacheInputs)
            assert isinstance(target_indexer, KVCacheInputs)
            assert isinstance(draft_mla, KVCacheInputs)
            assert isinstance(draft_indexer, KVCacheInputs)
            target_mla_kv = list(target_mla.inputs)
            target_indexer_kv = list(target_indexer.inputs)
            draft_mla_kv = list(draft_mla.inputs)
            draft_indexer_kv = list(draft_indexer.inputs)

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
                draft_tokens=draft_tokens,
                signal_buffers=signal_buffers,
                target_mla_kv=target_mla_kv,
                target_indexer_kv=target_indexer_kv,
                draft_mla_kv=draft_mla_kv,
                draft_indexer_kv=draft_indexer_kv,
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
                pinned_bitmask=pinned_bitmask_graph,
                wait_payload=wait_payload_graph,
                device_bitmask_scratch=device_bitmask_scratch_graph,
            )

            graph.output(*outputs)

        return graph, weights_registry

    def _create_draft_config(
        self,
        draft_state_dict: dict[str, WeightData],
        draft_kv: MultiKVCacheParams,
    ) -> DeepseekV3_2NextNConfig:
        """Create the NextN draft config for the GLM-5.2 MTP layer."""
        nextn_key = "decoder_layer.self_attn.kv_a_layernorm.weight"
        base_key = "layers.0.self_attn.kv_a_layernorm.weight"

        if nextn_key not in draft_state_dict:
            raise KeyError(
                f"Expected NextN norm key '{nextn_key}' not found in "
                f"draft state_dict. Available keys: "
                f"{list(draft_state_dict.keys())[:10]}..."
            )

        draft_state_dict[base_key] = draft_state_dict[nextn_key]
        base_config = DeepseekV3_2Model._create_model_config(
            self, draft_state_dict
        )
        if base_key in draft_state_dict and nextn_key in draft_state_dict:
            del draft_state_dict[base_key]

        draft_config = DeepseekV3_2NextNConfig(
            **{
                f.name: getattr(base_config, f.name)
                for f in fields(base_config)
            }
        )
        # Empty schedule keeps the single MTP layer's indexer full (it computes
        # its own top-k at step 0) and avoids indexing the target's 78-element
        # schedule at the MTP layer index.
        draft_config.indexer_types = []
        # The draft owns a single-layer {mla, indexer} cache.
        draft_config.kv_params = draft_kv

        if draft_config.quant_config is not None:
            nextn_layer_idx = max(
                draft_config.num_hidden_layers,
                draft_config.first_k_dense_replace,
            )

            # Correct draft model quantization config for NVFP4 MTP layer.
            if _subtree_quantized(draft_state_dict, ".mlp.experts."):
                draft_config.quant_config.mlp_quantized_layers.add(
                    nextn_layer_idx
                )
            if _subtree_quantized(draft_state_dict, ".self_attn."):
                draft_config.quant_config.attn_quantized_layers.add(
                    nextn_layer_idx
                )

        # Correct draft model EP config for NVFP4 MTP layer.
        if draft_config.ep_config is not None and not _subtree_quantized(
            draft_state_dict, ".mlp.experts."
        ):
            draft_config.ep_config = replace(
                draft_config.ep_config,
                dispatch_dtype=DType.bfloat16,
                dispatch_quant_config=None,
                fused_shared_expert=draft_config.n_shared_experts == 1,
            )
        return draft_config
