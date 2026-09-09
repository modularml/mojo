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
"""Gemma4 with MTP PipelineModel: target + draft in one graph."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from max.driver import Buffer, Device, DLPackArray
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, Module
from max.graph.weights import (
    WeightData,
    Weights,
    WeightsAdapter,
    load_weights,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import ImageMetadata
from max.pipelines.lib import (
    AlwaysSignalBuffersMixin,
    KVCacheConfig,
    ModelInputs,
    MultiGraphPipelineModelWithKVCache,
    PipelineConfig,
    UnifiedEagleOutputs,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.lib.vision_encoder_cache import VisionEncodeResult
from transformers import AutoConfig
from typing_extensions import override

from ..gemma4.batch_vision_inputs import (
    VisionRawInputs,
    create_empty_embeddings,
    pack_uncached_images,
)
from ..gemma4.context import Gemma4Context
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4.vision_model.vision_model import Gemma4VisionModel
from ..gemma4.weight_adapters import (
    convert_safetensor_vision_state_dict,
    fuse_gemma4_projection_weights,
    gemma4_uses_fused_projections,
)
from ..gemma4_assistant.gemma4_assistant import Gemma4Assistant
from ..gemma4_assistant.model_config import Gemma4AssistantConfig
from .batch_processor import UnifiedMTPGemma4BatchProcessor
from .unified_mtp_gemma4 import UnifiedMTPGemma4
from .weight_adapters import convert_unified_safetensor_state_dict


@dataclass
class UnifiedMTPGemma4Inputs(UnifiedSpecDecodeInputs):
    """Inputs for the UnifiedMTPGemma4 model.

    The spec-decode fields and trailing buffer packing come from
    :class:`UnifiedSpecDecodeInputs`; the fields below plus the KV cache form
    this distributed MTP graph's prefix. The graph binds the per-row
    ``in_thinking_phase`` flag and, when structured output is enabled, the
    constrained-decoding bitmask triple.
    """

    tokens: Buffer
    input_row_offsets: Buffer
    host_input_row_offsets: Buffer
    return_n_logits: Buffer
    data_parallel_splits: Buffer
    signal_buffers: list[Buffer]
    batch_context_lengths: list[Buffer]

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        assert self.kv_cache_inputs is not None
        prefix = (
            self.tokens,
            # Vision embeds + scatter indices follow tokens, matching
            # build_spec_decode_input_types(enable_vision=True).
            *self.vision_embeddings,
            *self.vision_scatter_indices,
            self.input_row_offsets,
            self.host_input_row_offsets,
            self.return_n_logits,
            self.data_parallel_splits,
            *self.signal_buffers,
            *self.kv_cache_inputs.flatten(),
            *self.batch_context_lengths,
        )
        return prefix + self._spec_decode_tail_buffers(
            include_in_thinking_phase=True
        )


class UnifiedMTPGemma4Model(
    _UnifiedSpecDecodeModelMixin,
    AlwaysSignalBuffersMixin,
    MultiGraphPipelineModelWithKVCache[Gemma4Context],
):
    """Gemma4 with MTP: merge + target + rejection + shift in one graph."""

    model_config_cls: ClassVar[type[Any]] = Gemma4ForConditionalGenerationConfig
    batch_processor_cls: ClassVar[type[UnifiedMTPGemma4BatchProcessor]] = (
        UnifiedMTPGemma4BatchProcessor
    )

    model: Model
    """The compiled unified MTP graph (target + draft + rejection). Re-executed
    each step; device graph capture is disabled for this arch (see ``arch.py``:
    the eager prefill vision encoder produces variable-shape embeddings)."""

    vision_model: Model | None
    """The compiled vision encoder graph, or None for text-only checkpoints.
    Runs eagerly during prefill (outside the captured graph)."""

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
        self._max_batch_size = max_batch_size
        super().__init__(
            pipeline_config,
            session,
            devices,
            kv_cache_config,
            weights,
            adapter=adapter,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
            memory_plan=memory_plan,
        )

        # Force signal buffer initialization.
        _ = self.signal_buffers

        self.vision_model, self.model = self.load_model(session)

        if self._batch_processor is not None:
            assert isinstance(
                self._batch_processor, UnifiedMTPGemma4BatchProcessor
            )
            self._batch_processor.bind_model_state(config=self.config)

    _draft_state_dict: dict[str, Any]
    _target_state_dict: dict[str, Any]
    _draft_config: Gemma4AssistantConfig

    @override
    def _load_state_dict(self) -> dict[str, Any]:
        assert self._max_batch_size, "Expected max_batch_size to be set"

        weights_dict = dict(self.weights.items())
        self._vision_weights_dict = convert_safetensor_vision_state_dict(
            weights_dict
        )
        self._language_weights_dict = {}

        target_state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        self._draft_state_dict = self._convert_draft_weights(
            dict(draft_weights.items())
        )

        return target_state_dict

    @override
    def _create_model_config(
        self, state_dict: dict[str, Any]
    ) -> Gemma4ForConditionalGenerationConfig:
        config = Gemma4ForConditionalGenerationConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=ReturnLogits.VARIABLE,
        )
        self.config = config

        self._target_state_dict = state_dict
        # DISTINF-194: pre-fuse the target's gate/up and qkv/qk projections
        # when configured (before target.*/draft.* prefixing below), so the
        # fused keys match the target submodel's FusedMLP / stacked qkv
        # layers. The draft submodel is not fused.
        if gemma4_uses_fused_projections(config):
            self._target_state_dict = fuse_gemma4_projection_weights(
                self._target_state_dict
            )

        assert self.pipeline_config.draft_model is not None
        draft_hf_config = self.pipeline_config.draft_model.huggingface_config
        assert draft_hf_config is not None
        self._draft_config = self._create_draft_config(
            draft_hf_config, config.devices
        )

        assert isinstance(self.kv_params, MultiKVCacheParams)
        self._target_sliding_kv_params = self.kv_params.children[
            "sliding_attention"
        ]
        self._target_global_kv_params = self.kv_params.children[
            "full_attention"
        ]
        self._target_layer_types = config.text_config.layer_types

        # The draft is Q-only cross-attention into the target's KV caches
        # (no K/V projections), so it allocates no cache of its own. None
        # signals SpecDecodeState to skip the draft manager and the graph to
        # declare no draft KV inputs.
        self._draft_kv_params = None

        return config

    def _include_vision_graph(
        self, model_config: Gemma4ForConditionalGenerationConfig
    ) -> bool:
        return model_config.vision_config is not None

    @override
    def _build_language_graph(
        self,
        model_config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        del state_dict
        spec_cfg = self.pipeline_config.speculative
        assert spec_cfg is not None

        nn_model = UnifiedMTPGemma4(
            model_config,
            self._draft_config,
            speculative_config=spec_cfg,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            use_greedy_acceptance=spec_cfg.use_greedy_acceptance,
        )

        nn_model.target.return_logits = ReturnLogits.VARIABLE
        nn_model.target.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED

        assert isinstance(self._target_sliding_kv_params, KVCacheParams)
        assert isinstance(self._target_global_kv_params, KVCacheParams)
        nn_model.draft = Gemma4Assistant(
            self._draft_config,
            target_layer_types=self._target_layer_types,
            target_sliding_kv_params=self._target_sliding_kv_params,
            target_global_kv_params=self._target_global_kv_params,
        )
        # Share the target's embed_tokens for the concat(embed, hidden)
        # input step.  The assistant's own 1024-dim draft_embed_tokens
        # and tied lm_head are loaded from the assistant checkpoint.
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens

        unified_state_dict = convert_unified_safetensor_state_dict(
            self._target_state_dict, self._draft_state_dict
        )
        # strict=False: shared weights (embed_tokens, lm_head) are aliased
        # to target's and won't have draft.* copies.
        nn_model.load_state_dict(
            unified_state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=False,
        )
        weights_registry = nn_model.state_dict()

        n_devs = len(self.devices)
        with Graph(
            "gemma4_with_mtp_graph",
            input_types=nn_model.input_types(self.kv_params),
            module=module,
        ) as graph:
            graph_inputs = iter(graph.inputs)
            tokens = next(graph_inputs)
            # Vision embeds + scatter indices follow tokens, matching
            # build_spec_decode_input_types(enable_vision=True).
            image_embeddings = [
                next(graph_inputs).tensor for _ in range(n_devs)
            ]
            image_token_indices = [
                next(graph_inputs).tensor for _ in range(n_devs)
            ]
            device_input_row_offsets = next(graph_inputs)
            host_input_row_offsets = next(graph_inputs)
            return_n_logits = next(graph_inputs)
            data_parallel_splits = next(graph_inputs)
            variadic_args = list(graph_inputs)

            variadic_args_iter = iter(variadic_args)
            signal_buffers = [
                next(variadic_args_iter).buffer
                for _ in range(len(self.devices))
            ]

            # Unflatten the hybrid {sliding, global} KV tree.
            sliding_kv_collections, global_kv_collections = (
                self.kv_params.unflatten_basic_kv_tree(variadic_args_iter)
            )

            batch_context_lengths = [
                next(variadic_args_iter).tensor
                for _ in range(len(self.devices))
            ]

            draft_tokens = next(variadic_args_iter).tensor

            seed = next(variadic_args_iter).tensor
            temperature = next(variadic_args_iter).tensor
            top_k = next(variadic_args_iter).tensor
            max_k = next(variadic_args_iter).tensor
            top_p = next(variadic_args_iter).tensor
            min_top_p = next(variadic_args_iter).tensor
            in_thinking_phase = next(variadic_args_iter).tensor

            pinned_bitmask_graph = None
            wait_payload_graph = None
            device_bitmask_scratch_graph = None
            if nn_model.enable_structured_output:
                pinned_bitmask_graph = next(variadic_args_iter).tensor
                wait_payload_graph = next(variadic_args_iter).buffer
                device_bitmask_scratch_graph = next(variadic_args_iter).buffer

            outputs = nn_model(
                tokens=tokens.tensor,
                input_row_offsets=device_input_row_offsets.tensor,
                image_embeddings=image_embeddings,
                image_token_indices=image_token_indices,
                draft_tokens=draft_tokens,
                signal_buffers=signal_buffers,
                sliding_kv_collections=sliding_kv_collections,
                global_kv_collections=global_kv_collections,
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
                pinned_bitmask=pinned_bitmask_graph,
                wait_payload=wait_payload_graph,
                device_bitmask_scratch=device_bitmask_scratch_graph,
            )

            graph.output(*outputs)

        return graph, weights_registry

    def pack_vision_inputs(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
    ) -> VisionRawInputs | None:
        """Pack the pipeline-selected uncached image pixels to device.

        Runs in the pipeline's prep-ahead window (pinned host-to-device copy)
        so it overlaps the prior batch, delegating to the shared
        :func:`pack_uncached_images`.
        """
        assert self.config.vision_config is not None
        return pack_uncached_images(
            selection,
            devices,
            self.config.vision_config.pooling_kernel_size,
            self.config.unquantized_dtype,
        )

    def vision_execute(
        self,
        selection: Sequence[tuple[Gemma4Context, Sequence[ImageMetadata]]],
        devices: list[Device],
        packed: VisionRawInputs | None,
    ) -> VisionEncodeResult:
        """Run the vision encoder on the pixels ``pack_vision_inputs`` packed.

        Returns embeddings only; the cache derives per-image counts from its
        selection. When there were no packable patches (``packed is None``),
        returns an empty result so the cache assembles from resident entries.
        """
        if packed is None:
            return VisionEncodeResult(
                embeddings=self.empty_vision_embeddings(self.devices)
            )
        return VisionEncodeResult(embeddings=self._run_vision_encoder(packed))

    def empty_vision_embeddings(self, devices: list[Device]) -> list[Buffer]:
        """Per-device zero-row image embeddings for cached / text-only batches.

        Cached: hit on every text-only / decode step, so it must not allocate
        per call.
        """
        if not hasattr(self, "_cached_empty_embeddings"):
            self._cached_empty_embeddings = create_empty_embeddings(
                devices,
                self.huggingface_config.text_config.hidden_size,
                self.config.unquantized_dtype,
            )
        return self._cached_empty_embeddings

    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> UnifiedEagleOutputs:
        """Execute and return all 3 graph outputs for speculative decoding.

        Images appear only during prefill (``draft_tokens`` is ``[batch,
        0]``); decode steps carry the empty vision-merge inputs, so the
        graph's scatter is a no-op there.
        """
        assert isinstance(model_inputs, UnifiedMTPGemma4Inputs)
        model_outputs = self.model.execute(*model_inputs.buffers)
        assert len(model_outputs) == 3, (
            f"Expected 3 outputs, got {len(model_outputs)}"
        )

        return UnifiedEagleOutputs(
            num_accepted_draft_tokens=model_outputs[0],
            next_tokens=model_outputs[1],
            next_draft_tokens=model_outputs[2],
        )

    def _build_vision_graph(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        state_dict: dict[str, WeightData],
        module: Module | None = None,
    ) -> tuple[Graph, dict[str, DLPackArray]]:
        """Build the vision model with our input types and graph"""
        vision_model = Gemma4VisionModel(
            config,
            device=DeviceRef.from_device(self.devices[0]),
        )
        vision_model.load_state_dict(
            state_dict=state_dict,
            override_quantization_encoding=True,
            weight_alignment=1,
            strict=True,
        )
        with Graph(
            "gemma4_vision",
            input_types=vision_model.input_types(),
            module=module,
        ) as vision_graph:
            all_inputs = vision_graph.inputs
            n_devices = len(self.devices)

            patches_flat_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            pixel_position_ids_list = [
                inp.tensor for inp in all_inputs[:n_devices]
            ]
            all_inputs = all_inputs[n_devices:]

            cu_seqlens_list = [inp.tensor for inp in all_inputs[:n_devices]]
            all_inputs = all_inputs[n_devices:]

            pool_gather_index_list = [
                inp.tensor for inp in all_inputs[:n_devices]
            ]
            all_inputs = all_inputs[n_devices:]

            max_seq_len = all_inputs[0].tensor

            outputs = vision_model(
                patches_flat_list,
                pixel_position_ids_list,
                cu_seqlens_list,
                pool_gather_index_list,
                max_seq_len,
            )
            vision_graph.output(*outputs)

        return vision_graph, vision_model.state_dict()

    def _run_vision_encoder(self, raw: VisionRawInputs) -> list[Buffer]:
        if self.vision_model is None:
            raise ValueError(
                "This checkpoint is served text-only (no vision encoder"
                " is loaded); image and video inputs are not supported."
            )
        return self.vision_model(
            *raw.patches_flat,
            *raw.pixel_position_ids,
            *raw.cu_seqlens,
            *raw.pool_gather_index,
            raw.max_seq_len,
        )

    def _convert_draft_weights(
        self,
        draft_weights_dict: dict[str, Weights],
    ) -> dict[str, WeightData]:
        """Convert HuggingFace assistant checkpoint keys to MAX format.

        The HF assistant checkpoint has keys like:
        - ``model.layers.0.self_attn.q_proj.weight`` -> ``layers.0.self_attn.q_proj.weight``
        - ``model.norm.weight`` -> ``norm.weight``
        - ``pre_projection.weight`` -> ``pre_projection.weight`` (at top level)
        - ``post_projection.weight`` -> ``post_projection.weight``
        - ``model.embed_tokens.weight`` -> kept (assistant's own 1024-dim embedding)
        """
        new_state_dict: dict[str, WeightData] = {}

        for name, value in draft_weights_dict.items():
            data = value.data()

            # Strip "model." prefix for keys under model.*
            if name.startswith("model."):
                max_name = name[len("model.") :]
            else:
                # Top-level keys like pre_projection, post_projection
                max_name = name

            new_state_dict[max_name] = data

        return new_state_dict

    def _create_draft_config(
        self,
        draft_hf_config: AutoConfig,
        devices: list[DeviceRef],
    ) -> Gemma4AssistantConfig:
        """Create Gemma4AssistantConfig from the draft HF config."""
        from ..gemma3.model_config import _HIDDEN_ACTIVATION_MAP
        from ..gemma4.layers.rotary_embedding import ProportionalScalingParams

        raw_text_config = draft_hf_config
        if hasattr(draft_hf_config, "text_config"):
            raw_text_config = draft_hf_config.text_config

        # Normalize to dict so we can use .get() uniformly whether
        # the HF shim stored it as a dict or a sub-config object.
        tc: dict[str, Any] = (
            raw_text_config
            if isinstance(raw_text_config, dict)
            else raw_text_config.__dict__
        )

        # Extract global rope scaling if available.
        global_rope_scaling = None
        rope_parameters = tc.get("rope_parameters")
        if rope_parameters is not None and "full_attention" in rope_parameters:
            full_attn_params = rope_parameters["full_attention"]
            partial_rotary_factor = full_attn_params.get(
                "partial_rotary_factor"
            )
            if partial_rotary_factor is not None:
                global_rope_scaling = ProportionalScalingParams(
                    partial_rotary_factor=partial_rotary_factor,
                )

        # Get backbone hidden size from the target HF config.
        target_text_config = self.huggingface_config.text_config
        backbone_hidden_size = target_text_config.hidden_size

        num_hidden_layers = tc["num_hidden_layers"]
        return Gemma4AssistantConfig(
            devices=devices,
            backbone_hidden_size=backbone_hidden_size,
            hidden_size=tc["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            num_global_key_value_heads=tc.get("num_global_key_value_heads", 4),
            head_dim=tc["head_dim"],
            global_head_dim=tc.get("global_head_dim", 512),
            intermediate_size=tc["intermediate_size"],
            vocab_size=tc["vocab_size"],
            rms_norm_eps=tc["rms_norm_eps"],
            hidden_activation=_HIDDEN_ACTIVATION_MAP.get(
                tc.get("hidden_activation", "gelu_pytorch_tanh"),
                tc.get("hidden_activation", "gelu_pytorch_tanh"),
            ),
            layer_types=tc.get(
                "layer_types",
                ["sliding_attention"] * (num_hidden_layers - 1)
                + ["full_attention"],
            ),
            sliding_window=tc.get("sliding_window", 1024),
            sliding_window_rope_theta=tc.get(
                "sliding_window_rope_theta", 10000.0
            ),
            global_rope_theta=tc.get("global_rope_theta", 1000000.0),
            global_rope_scaling=global_rope_scaling,
            attention_k_eq_v=tc.get("attention_k_eq_v", True),
            num_kv_shared_layers=tc.get("num_kv_shared_layers", 4),
            max_position_embeddings=tc.get("max_position_embeddings", 262144),
        )
