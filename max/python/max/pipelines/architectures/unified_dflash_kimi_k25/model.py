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
"""Unified DFlash Kimi K2.5 PipelineModel.

Inherits from :class:`KimiK2_5Model` and replaces ``load_model`` to wire a
DFlash draft (multi-layer non-causal block transformer with external-KV
materialization) against the Kimi K2.5 (MLA) target. Per-step buffers
mirror :class:`Eagle3MHAKimiK25Inputs` to keep the spec-decode driver
unchanged.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferValue, DeviceRef, Graph, Module, TensorValue, Value
from max.graph.weights import WeightData, load_weights
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheInputsInterface,
    KVCacheParams,
    spec_decode_cache_slack,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.lib import CompilationTimer, KVCacheConfig, PipelineConfig
from max.pipelines.lib.interfaces import (
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from transformers import AutoConfig
from typing_extensions import override

from ..dflash_kimi_k25 import DFlashKimiK25DraftConfig
from ..kimik2_5.model import KimiK2_5Model, KimiK2_5ModelInputs
from ..kimik2_5.model_config import KimiK2_5TextConfig
from ..llama3.weight_adapters import _convert_safetensor_with_model_config
from .batch_processor import UnifiedDflashKimiK25BatchProcessor
from .model_config import (
    MultiKVCacheParams,
    UnifiedDflashKimiK25Config,
    parse_dflash_draft_hf_config,
)
from .unified_dflash_kimi_k25 import UnifiedDflashKimiK25

logger = logging.getLogger("max.pipelines")


@dataclass
class UnifiedDflashKimiK25Inputs(UnifiedSpecDecodeInputs, KimiK2_5ModelInputs):
    """Inputs for the unified DFlash Kimi K2.5 graph.

    Same as :class:`KimiK2_5ModelInputs` -- including the per-device
    vision-merge inputs (base ``vision_embeddings`` /
    ``vision_scatter_indices``, set by the pipeline's vision seam) that the
    unified graph scatters into the merged token embedding before the target
    forward -- plus the spec-decode fields and trailing buffer packing from
    :class:`UnifiedSpecDecodeInputs`. The draft owns its own MHA
    :class:`KVCacheInputs` so its dispatch metadata is independent of the
    target's MLA cache. The DFlash graph does not bind ``in_thinking_phase``
    (it is only consumed by the relaxed-acceptance-for-thinking sampler rule,
    which DFlash does not configure); the structured-output bitmask triple is
    packed whenever the graph was compiled with it.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert len(self.vision_embeddings) == len(
            self.vision_scatter_indices
        ), (
            "vision_embeddings and vision_scatter_indices must have the "
            "same length"
        )
        buffers = (
            self.tokens,
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *(
                self.kv_cache_inputs.flatten()
                if self.kv_cache_inputs is not None
                else ()
            ),
            *self.batch_context_lengths,
            *self.ep_inputs,
        )
        return buffers + self._spec_decode_tail_buffers(
            include_in_thinking_phase=False
        )


class UnifiedDflashKimiK25Model(_UnifiedSpecDecodeModelMixin, KimiK2_5Model):
    """Unified DFlash Kimi K2.5 pipeline model.

    Routed here when target HF arch is
    ``KimiK25ForConditionalGeneration`` and
    ``SpeculativeConfig.is_dflash()`` is true.
    """

    batch_processor_cls = UnifiedDflashKimiK25BatchProcessor

    def __init__(
        self, pipeline_config: PipelineConfig, *args: Any, **kwargs: Any
    ) -> None:
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.SELECTED_LAYERS
        assert pipeline_config.speculative is not None
        self.resolved_num_speculative_tokens = (
            pipeline_config.speculative.draft_width
        )
        super().__init__(pipeline_config, *args, **kwargs)

    @override
    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParams:
        # The parent reads the raw speculative section, where the unset
        # width would bake num_draft_tokens=0; rebake at the trained width.
        params = super().get_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )
        assert pipeline_config.speculative is not None
        assert isinstance(params, KVCacheParams)
        return replace(
            params,
            num_draft_tokens=pipeline_config.speculative.draft_width,
        )

    @override
    def _create_model_config(
        self, state_dict: dict[str, WeightData]
    ) -> KimiK2_5TextConfig:
        model_config = super()._create_model_config(state_dict)
        # ``KimiK2_5TextConfig.initialize`` derives its KV params, and the
        # rope table's spec-decode slack, from the raw speculative section,
        # where the unset width reads as 0; re-derive both at the drafter's
        # trained width.
        resolved_spec = self.resolved_num_speculative_tokens
        assert resolved_spec is not None
        assert isinstance(model_config.kv_params, KVCacheParams)
        stale_slack = spec_decode_cache_slack(model_config.kv_params)
        model_config.kv_params = replace(
            model_config.kv_params, num_draft_tokens=resolved_spec
        )
        model_config.max_position_embeddings += (
            spec_decode_cache_slack(model_config.kv_params) - stale_slack
        )
        return model_config

    @override
    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
        max_batch_size = self.max_batch_size
        assert max_batch_size, "Expected max_batch_size to be set"

        dp_size = self.pipeline_config.model.data_parallel_degree
        max_batch_size *= dp_size

        self._host_input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        )
        self._device_input_row_offsets_prealloc = (
            self._host_input_row_offsets_prealloc.to(self.devices[0])
        )
        self._batch_context_lengths_prealloc_cpu = [
            Buffer.zeros(shape=[1], dtype=DType.int32)
            for _ in range(len(self.devices))
        ]

        if self.adapter:
            target_state_dict = self.adapter(
                dict(self.weights.items()),
                huggingface_config=self.huggingface_config,
                pipeline_config=self.pipeline_config,
            )
        else:
            target_state_dict = {
                key: value.data() for key, value in self.weights.items()
            }

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in target_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith(("language_model.", "language_")):
                llm_state_dict[key] = value

        target_config = self._create_model_config(target_state_dict)

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        self.ep_comm_initializer = None
        if target_config.ep_config is not None and not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(
                target_config.ep_config
            )
            self.ep_comm_initializer.ep_init(session)
            target_config.ep_config.node_id = (
                self.ep_comm_initializer.config.node_id
            )
            if target_config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP "
                    "initialization is successful."
                )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_hf = draft_model_config.huggingface_config
        assert draft_hf is not None

        dflash_hf = parse_dflash_draft_hf_config(draft_hf)
        shifted_target_layer_ids = [i - 1 for i in dflash_hf.target_layer_ids]
        if any(i < 0 for i in shifted_target_layer_ids):
            raise NotImplementedError(
                "DFlash dflash_config.target_layer_ids must not contain 0 "
                "for the current MAX impl — capturing layer-0's input "
                "(= raw token embeddings) is not yet wired. Got "
                f"target_layer_ids={dflash_hf.target_layer_ids}."
            )
        target_config.eagle_aux_hidden_state_layer_ids = (
            shifted_target_layer_ids
        )

        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        draft_state_dict = _convert_safetensor_with_model_config(
            dict(draft_weights.items()),
            draft_hf,
            draft_model_config,
        )

        assert isinstance(self.kv_params, KVCacheParams)
        target_kv = self.kv_params
        draft_num_hidden_layers = int(draft_hf.num_hidden_layers)

        assert self.pipeline_config.speculative is not None
        dflash_block_size = int(dflash_hf.block_size or 0)
        if dflash_block_size <= 0:
            num_spec = self.pipeline_config.speculative.num_speculative_tokens
            if num_spec is None:
                raise ValueError(
                    "The DFlash draft checkpoint declares no block_size; set"
                    " --num-speculative-tokens explicitly."
                )
            dflash_block_size = num_spec + 1

        target_kv.num_draft_tokens = dflash_block_size
        logger.info(
            "DFlash Kimi: target draft_attention_dispatch_metadata sized "
            "for q_max_seq_len=%d (block_size).",
            dflash_block_size,
        )
        self._draft_kv_params = self.pipeline_config.model.kv_cache.to_params(
            dtype=DType.bfloat16,
            n_kv_heads=int(draft_hf.num_key_value_heads),
            head_dim=int(draft_hf.head_dim),
            num_layers=draft_num_hidden_layers,
            devices=target_kv.devices,
            data_parallel_degree=target_kv.data_parallel_degree,
            is_mla=False,
            num_q_heads=int(draft_hf.num_attention_heads),
            speculative_method=target_kv.speculative_method,
            num_draft_tokens=dflash_block_size,
        )
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv, "draft": self._draft_kv_params}
        )

        draft_config = _build_dflash_draft_config(
            draft_hf,
            target_config=target_config,
            devices=list(target_config.devices),
            data_parallel_degree=target_config.data_parallel_degree,
            target_layer_ids=list(dflash_hf.target_layer_ids),
            kv_params=self._draft_kv_params,
            sliding_window_override=draft_model_config.sliding_window,
        )

        assert self.pipeline_config.speculative is not None
        unified_config = UnifiedDflashKimiK25Config(
            target=target_config,
            draft=draft_config,
            speculative_config=self.pipeline_config.speculative,
            target_layer_ids=list(dflash_hf.target_layer_ids),
            mask_token_id=int(dflash_hf.mask_token_id),
            block_size=int(dflash_hf.block_size or 0),
        )
        unified_config.validate_dflash_fields()

        nn_model = UnifiedDflashKimiK25(
            unified_config,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
        )

        nn_model.draft.embed_tokens = nn_model.target.embed_tokens
        nn_model.draft.lm_head = nn_model.target.lm_head

        target_llm_sd = {
            k[len("language_model.") :]: v
            for k, v in llm_state_dict.items()
            if k.startswith("language_model.")
        }
        nn_model.target.load_state_dict(
            target_llm_sd, weight_alignment=1, strict=True
        )
        nn_model.draft.load_state_dict(
            draft_state_dict, weight_alignment=1, strict=False
        )

        draft_expected = set(nn_model.draft.raw_state_dict().keys())
        draft_provided = set(draft_state_dict.keys())
        shared_prefixes = ("embed_tokens.", "lm_head.")
        missing = {
            k
            for k in draft_expected - draft_provided
            if not k.startswith(shared_prefixes)
        }
        extra = draft_provided - draft_expected
        if missing:
            raise ValueError(
                f"Draft model has unloaded non-shared weights: "
                f"{sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        draft_weights_registry = nn_model.draft.state_dict()

        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith(shared_prefixes):
                continue
            weight.name = f"draft.{name}"

        from ..kimik2_5.kimik2_5 import KimiK2_5
        from ..kimik2_5.model_config import KimiK2_5Config

        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=target_config,
            max_seq_len=self.max_seq_len,
        )
        self.model_config = kimik2_5_config
        self.nn_model = KimiK2_5(kimik2_5_config)
        self.nn_model.load_state_dict(
            target_state_dict, weight_alignment=1, strict=False
        )
        self.state_dict = dict(self.nn_model.state_dict())
        self.state_dict.update(nn_model.target.state_dict())
        for k, v in draft_weights_registry.items():
            if k.startswith(shared_prefixes):
                continue
            self.state_dict[f"draft.{k}"] = v

        with CompilationTimer("vision + unified_dflash_kimi_k25 model") as t:
            graph_module = Module()
            vision_graph, _ = self._build_vision_graph(
                kimik2_5_config, vision_state_dict, module=graph_module
            )
            with Graph(
                "unified_dflash_kimi_k25_graph",
                input_types=nn_model.input_types(self.kv_params),
                module=graph_module,
            ) as graph:
                tokens, *rest_inputs = graph.inputs
                variadic_args_iter = iter(rest_inputs)

                image_embeddings_in = [
                    next(variadic_args_iter).tensor for _ in range(n_devices)
                ]
                image_token_indices_in = [
                    next(variadic_args_iter).tensor for _ in range(n_devices)
                ]
                devices_input_row_offsets = next(variadic_args_iter)
                host_input_row_offsets = next(variadic_args_iter)
                return_n_logits = next(variadic_args_iter)
                data_parallel_splits = next(variadic_args_iter)

                signal_buffers = [
                    next(variadic_args_iter).buffer
                    for _ in range(len(self.devices))
                ]

                kv_collections, draft_kv_collections = (
                    self.kv_params.unflatten_basic_kv_tree(variadic_args_iter)
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

                # Optional constrained-decoding bitmask triple — present
                # only when structured output is enabled (matches the
                # conditional in input_types()). Bound at runtime by the
                # OverlapTextGenerationPipeline from
                # StructuredOutputOverlapState.
                pinned_bitmask_graph: TensorValue | None = None
                wait_payload_graph: BufferValue | None = None
                device_bitmask_scratch_graph: BufferValue | None = None
                if nn_model.enable_structured_output:
                    pinned_bitmask_graph = next(variadic_args_iter).tensor
                    wait_payload_graph = next(variadic_args_iter).buffer
                    device_bitmask_scratch_graph = next(
                        variadic_args_iter
                    ).buffer

                outputs = nn_model(
                    tokens=tokens.tensor,
                    input_row_offsets=devices_input_row_offsets.tensor,
                    draft_tokens=draft_tokens,
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
                    image_embeddings=image_embeddings_in,
                    image_token_indices=image_token_indices_in,
                    ep_inputs=target_ep_inputs,
                    draft_kv_collections=draft_kv_collections,
                    pinned_bitmask=pinned_bitmask_graph,
                    wait_payload=wait_payload_graph,
                    device_bitmask_scratch=device_bitmask_scratch_graph,
                )
                graph.output(*outputs)

            t.mark_build_complete()
            models = session.load_all(
                graph_module, weights_registry=self.state_dict
            )
            vision_model = models[vision_graph.name]
            language_model = models[graph.name]

        return vision_model, language_model

    @property
    def _spec_decode_model(self) -> Model:
        return self.language_model

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[KimiK2_5TextAndVisionContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
        draft_tokens: Buffer | None = None,
        **kwargs: Any,
    ) -> UnifiedDflashKimiK25Inputs:
        if self._batch_processor is not None:
            assert isinstance(
                self._batch_processor, UnifiedDflashKimiK25BatchProcessor
            )
            return self._batch_processor.prepare_initial_token_inputs(
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
                draft_tokens=draft_tokens,
                **kwargs,
            )
        raise RuntimeError(
            "No batch processor configured for UnifiedDflashKimiK25Model"
        )


def _build_dflash_draft_config(
    draft_hf: Any,
    *,
    target_config: Any,
    devices: list[Any],
    data_parallel_degree: int,
    target_layer_ids: list[int],
    kv_params: KVCacheParams,
    sliding_window_override: int | None = None,
) -> DFlashKimiK25DraftConfig:
    """Build :class:`DFlashKimiK25DraftConfig` from the draft HF config."""
    rope_scaling = getattr(draft_hf, "rope_scaling", None)
    if (
        rope_scaling is None
        or rope_scaling.get("rope_type", rope_scaling.get("type")) != "yarn"
    ):
        raise ValueError(
            "Draft HF config must declare a Deepseek-yarn rope_scaling block."
        )

    use_override = sliding_window_override is not None
    raw = (
        sliding_window_override
        if use_override
        else getattr(draft_hf, "sliding_window", None)
    )
    sliding_window = int(raw) if raw is not None and int(raw) > 0 else None
    logger.info(
        "DFlash Kimi draft: sliding_window=%s (from %s)",
        sliding_window,
        "MAXModelConfig override" if use_override else "draft HF config",
    )

    return DFlashKimiK25DraftConfig(
        hidden_size=int(draft_hf.hidden_size),
        num_attention_heads=int(draft_hf.num_attention_heads),
        num_key_value_heads=int(draft_hf.num_key_value_heads),
        head_dim=int(draft_hf.head_dim),
        intermediate_size=int(draft_hf.intermediate_size),
        num_hidden_layers=int(draft_hf.num_hidden_layers),
        vocab_size=int(draft_hf.vocab_size),
        rms_norm_eps=float(draft_hf.rms_norm_eps),
        rope_theta=float(get_rope_theta(draft_hf)),
        max_position_embeddings=int(draft_hf.max_position_embeddings),
        devices=list(devices),
        data_parallel_degree=data_parallel_degree,
        dtype=DType.bfloat16,
        norm_dtype=target_config.norm_dtype,
        kv_params=kv_params,
        rope_scaling=dict(rope_scaling),
        target_layer_ids=list(target_layer_ids),
        sliding_window=sliding_window,
    )
