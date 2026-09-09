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
"""Eagle3 MHA-draft + Kimi K2.5 (MLA target) PipelineModel.

Sibling of :class:`Eagle3KimiK25Model` for the case where the draft uses
Llama-style MHA (e.g. ``LlamaForCausalLMEagle3`` checkpoints) over an MLA
Kimi K2.5 target. Target and draft no longer share attention parameters; a
separate MHA :class:`KVCacheParams` is constructed for the draft so the
spec-decode pipeline allocates its KV cache for MHA shapes.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max._core.driver import is_virtual_device_mode
from max.driver import Buffer
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import BufferValue, Graph, Module, TensorValue, Value
from max.graph.weights import WeightData, load_weights
from max.nn.comm.ep import EPCommInitializer
from max.nn.kv_cache import (
    KVCacheInputsInterface,
    KVCacheParams,
    MultiKVCacheParams,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.kimik2_5.context import (
    KimiK2_5TextAndVisionContext,
)
from max.pipelines.lib import CompilationTimer
from max.pipelines.lib.interfaces import (
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lib.pipeline_variants.unified_spec_decode_model import (
    _UnifiedSpecDecodeModelMixin,
)
from max.pipelines.lib.pipeline_variants.utils import get_rope_theta
from typing_extensions import override

from ..deepseekV3.model_config import DeepseekV3Config
from ..eagle_common.eagle_mha_draft import Eagle3MHADraftConfig
from .model import KimiK2_5Model, KimiK2_5ModelInputs
from .model_config import _extract_eagle_aux_layer_ids
from .unified_eagle_mha_model import Eagle3MHAKimiK25Unified
from .weight_adapters import convert_llama_eagle3_draft_state_dict

logger = logging.getLogger("max.pipelines")


@dataclass
class Eagle3MHAKimiK25Inputs(UnifiedSpecDecodeInputs, KimiK2_5ModelInputs):
    """Inputs for the Eagle3 MHA-draft + Kimi K2.5 model.

    Same as :class:`KimiK2_5ModelInputs` plus the spec-decode fields and
    trailing buffer packing from :class:`UnifiedSpecDecodeInputs`. The draft
    owns a full :class:`KVCacheInputs` (separate from the target's MLA cache)
    so its MHA dispatch metadata can be plumbed at both prefill and decode
    q_max_seq_len without colliding with the target's MLA slots. The graph
    binds the per-row ``in_thinking_phase`` flag.
    """

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        # Ordering must match ``Eagle3MHAKimiK25Unified.input_types``:
        # tokens, per-device image_embeddings, per-device
        # image_token_indices, then the rest of the inputs.
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
            include_in_thinking_phase=True
        )


class Eagle3MHAKimiK25Model(_UnifiedSpecDecodeModelMixin, KimiK2_5Model):
    """Eagle3 MHA-draft + Kimi K2.5 (MLA target) pipeline model.

    The pipeline routes here when the target HF arch is
    ``KimiK25ForConditionalGeneration`` and the draft HF arch is
    ``LlamaForCausalLMEagle3``.
    """

    def __init__(self, *args, **kwargs):
        kwargs["return_logits"] = ReturnLogits.VARIABLE
        kwargs["return_hidden_states"] = ReturnHiddenStates.SELECTED_LAYERS
        super().__init__(*args, **kwargs)

    @override
    def load_model(self, session: InferenceSession) -> tuple[Model, Model]:
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

        # ``_create_model_config`` may mutate the state dictionary.
        config = self._create_model_config(target_state_dict)

        vision_state_dict: dict[str, WeightData] = {}
        llm_state_dict: dict[str, WeightData] = {}
        for key, value in target_state_dict.items():
            if key.startswith("vision_encoder."):
                vision_state_dict[key] = value
            elif key.startswith(("language_model.", "language_")):
                llm_state_dict[key] = value

        n_devices = len(self.devices)
        if n_devices > 1 and self.pipeline_config.runtime.ep_size != n_devices:
            raise ValueError("Only the EP strategy is supported.")

        self.ep_comm_initializer = None
        if config.ep_config is not None and not is_virtual_device_mode():
            self.ep_comm_initializer = EPCommInitializer(config.ep_config)
            self.ep_comm_initializer.ep_init(session)
            config.ep_config.node_id = self.ep_comm_initializer.config.node_id
            if config.ep_config.node_id == -1:
                raise ValueError(
                    "EP node ID is not set. Please check if the EP "
                    "initialization is successful."
                )

        assert self.pipeline_config.draft_model is not None
        draft_model_config = self.pipeline_config.draft_model
        draft_hf = draft_model_config.huggingface_config
        assert draft_hf is not None

        draft_weight_paths = draft_model_config.resolved_weight_paths()
        draft_weights = load_weights(draft_weight_paths)
        draft_state_dict = convert_llama_eagle3_draft_state_dict(
            dict(draft_weights.items())
        )

        fc_input_multiplier = _infer_fc_input_multiplier(
            draft_state_dict, hidden_size=int(draft_hf.hidden_size)
        )

        if config.eagle_aux_hidden_state_layer_ids is None:
            ids = _extract_eagle_aux_layer_ids(draft_hf)
            if ids is None:
                target_layers = int(config.num_hidden_layers)
                ids = _default_aux_layer_ids(target_layers, fc_input_multiplier)
                logger.warning(
                    "Draft HF config has no "
                    "'eagle_config.eagle_aux_hidden_state_layer_ids'. "
                    "Falling back to evenly-spaced defaults for "
                    f"{fc_input_multiplier}-way fusion over {target_layers} "
                    f"target layers: {ids}."
                )
            config.eagle_aux_hidden_state_layer_ids = ids

        n_target_capture = len(config.eagle_aux_hidden_state_layer_ids)
        if n_target_capture != fc_input_multiplier:
            raise ValueError(
                f"Draft fc fuses {fc_input_multiplier} target hidden states "
                f"but the target is configured to capture {n_target_capture} "
                f"(eagle_aux_hidden_state_layer_ids="
                f"{config.eagle_aux_hidden_state_layer_ids}). The draft "
                f"checkpoint's fc.weight shape and the draft HF "
                f"eagle_config.eagle_aux_hidden_state_layer_ids must agree."
            )

        assert isinstance(self.kv_params, KVCacheParams)
        target_kv = self.kv_params
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
        self.kv_params = MultiKVCacheParams.from_params(
            {"target": target_kv, "draft": self._draft_kv_params}
        )

        draft_config = _build_mha_draft_config(
            draft_hf,
            target_config=config,
            devices=config.devices,
            data_parallel_degree=config.data_parallel_degree,
            fc_input_multiplier=fc_input_multiplier,
            kv_params=self._draft_kv_params,
            sliding_window_override=draft_model_config.sliding_window,
        )

        assert self.pipeline_config.speculative is not None
        nn_model = Eagle3MHAKimiK25Unified(
            config,
            draft_config,
            speculative_config=self.pipeline_config.speculative,
            enable_structured_output=self.pipeline_config.needs_bitmask_constraints,
            enable_vision=True,
        )

        assert nn_model.draft is not None
        nn_model.draft.embed_tokens = nn_model.target.embed_tokens
        if "lm_head.weight" not in draft_state_dict:
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
                f"Draft model has unloaded non-shared weights: {sorted(missing)}"
            )
        if extra:
            logger.warning(f"Draft state_dict has unused keys: {sorted(extra)}")

        draft_weights_registry = nn_model.draft.state_dict()

        draft_lm_head_shared = "lm_head.weight" not in draft_state_dict

        # Rename non-shared draft Weights so graph-level names are unique.
        for name, weight in nn_model.draft.raw_state_dict().items():
            if name.startswith("embed_tokens."):
                continue
            if draft_lm_head_shared and name.startswith("lm_head."):
                continue
            weight.name = f"draft.{name}"

        from .kimik2_5 import KimiK2_5
        from .model_config import KimiK2_5Config

        kimik2_5_config = KimiK2_5Config.initialize_from_config(
            pipeline_config=self.pipeline_config,
            huggingface_config=self.huggingface_config,
            llm_config=config,
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
            if k.startswith("embed_tokens."):
                continue
            if draft_lm_head_shared and k.startswith("lm_head."):
                continue
            self.state_dict[f"draft.{k}"] = v

        with CompilationTimer("vision + eagle3 mha language model") as timer:
            graph_module = Module()
            assert self.model_config is not None
            vision_graph, _ = self._build_vision_graph(
                self.model_config, vision_state_dict, module=graph_module
            )
            with Graph(
                "eagle3_mha_kimik25_graph",
                input_types=nn_model.input_types(self.kv_params),
                module=graph_module,
            ) as graph:
                (
                    tokens,
                    *rest_inputs,
                ) = graph.inputs

                rest_iter = iter(rest_inputs)
                n_devices = len(self.devices)
                # Image embeddings + scatter indices (per device) appear
                # right after ``tokens`` in the graph's input_types. Match
                # the same ordering when destructuring inputs here.
                image_embeddings_in = [
                    next(rest_iter).tensor for _ in range(n_devices)
                ]
                image_token_indices_in = [
                    next(rest_iter).tensor for _ in range(n_devices)
                ]
                devices_input_row_offsets = next(rest_iter)
                host_input_row_offsets = next(rest_iter)
                return_n_logits = next(rest_iter)
                data_parallel_splits = next(rest_iter)
                variadic_args = list(rest_iter)

                variadic_args_iter = iter(variadic_args)
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
                in_thinking_phase = next(variadic_args_iter).tensor

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
                    in_thinking_phase=in_thinking_phase,
                    image_embeddings=image_embeddings_in,
                    image_token_indices=image_token_indices_in,
                    ep_inputs=target_ep_inputs,
                    draft_kv_collections=draft_kv_collections,
                    pinned_bitmask=pinned_bitmask_graph,
                    wait_payload=wait_payload_graph,
                    device_bitmask_scratch=device_bitmask_scratch_graph,
                )
                graph.output(*outputs)

            timer.mark_build_complete()
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
        draft_kv_cache_buffers: list[Buffer] | None = None,
        **kwargs,
    ) -> Eagle3MHAKimiK25Inputs:
        base = KimiK2_5Model.prepare_initial_token_inputs(
            self,
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
            return_n_logits=return_n_logits,
        )
        return Eagle3MHAKimiK25Inputs(
            tokens=base.tokens,
            input_row_offsets=base.input_row_offsets,
            host_input_row_offsets=base.host_input_row_offsets,
            batch_context_lengths=base.batch_context_lengths,
            signal_buffers=base.signal_buffers,
            kv_cache_inputs=base.kv_cache_inputs,
            return_n_logits=base.return_n_logits,
            data_parallel_splits=base.data_parallel_splits,
            ep_inputs=base.ep_inputs,
            # Vision inputs computed by the base call's host-side encoder
            # run (or empty placeholders when no images are present).
            image_token_indices=base.image_token_indices,
            precomputed_image_embeddings=base.precomputed_image_embeddings,
            pixel_values=base.pixel_values,
            grid_thws=base.grid_thws,
            cu_seqlens=base.cu_seqlens,
            max_seqlen=base.max_seqlen,
            vision_position_ids=base.vision_position_ids,
            draft_tokens=draft_tokens,
            structured_output=self.pipeline_config.needs_bitmask_constraints,
        )


def _infer_fc_input_multiplier(
    draft_state_dict: dict[str, WeightData], hidden_size: int
) -> int:
    """Returns 2 or 3 from the ``fc.weight`` shape in the draft checkpoint.

    Llama-style EAGLE3 drafts fuse 2 captured target hidden states; the
    Kimi MLA draft variant fuses 3. The fc projection's input dim is
    ``hidden_size * multiplier``.
    """
    if "fc.weight" not in draft_state_dict:
        raise ValueError(
            "Draft checkpoint is missing 'fc.weight' — required to infer "
            "the number of fused target hidden states."
        )
    fc_weight = draft_state_dict["fc.weight"]
    # Linear weights are stored on CPU as [out_dim, in_dim].
    in_dim = int(fc_weight.shape[1])
    if in_dim % hidden_size != 0:
        raise ValueError(
            f"fc.weight in_dim={in_dim} is not a multiple of hidden_size="
            f"{hidden_size}; cannot infer the fusion width."
        )
    multiplier = in_dim // hidden_size
    if multiplier not in (2, 3):
        raise ValueError(
            f"Unexpected fc fusion width: in_dim={in_dim}, hidden_size="
            f"{hidden_size}, multiplier={multiplier}. Expected 2 or 3."
        )
    return multiplier


def _build_mha_draft_config(
    draft_hf: Any,
    *,
    target_config: DeepseekV3Config,
    devices: list[Any],
    data_parallel_degree: int,
    fc_input_multiplier: int,
    kv_params: KVCacheParams,
    sliding_window_override: int | None = None,
) -> Eagle3MHADraftConfig:
    """Build :class:`Eagle3MHADraftConfig` from the draft HF config.

    ``sliding_window_override`` (typically threaded from
    ``MAXModelConfig.sliding_window`` on the draft model — settable via
    ``--model-override draft.sliding_window=N``) takes precedence over the
    ``sliding_window`` field on the draft's HF config. Passing a non-positive
    value disables sliding window even if the HF config asks for one.
    """
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
        "Eagle3 MHA draft: sliding_window=%s (from %s)",
        sliding_window,
        "MAXModelConfig override" if use_override else "draft HF config",
    )

    return Eagle3MHADraftConfig(
        hidden_size=int(draft_hf.hidden_size),
        num_attention_heads=int(draft_hf.num_attention_heads),
        num_key_value_heads=int(draft_hf.num_key_value_heads),
        head_dim=int(draft_hf.head_dim),
        intermediate_size=int(draft_hf.intermediate_size),
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
        fc_input_multiplier=fc_input_multiplier,
        sliding_window=sliding_window,
    )


def _default_aux_layer_ids(
    num_target_layers: int, fc_input_multiplier: int
) -> list[int]:
    """Picks ``fc_input_multiplier`` aux capture layer IDs spread across
    a target with ``num_target_layers`` layers.

    Fallback used when the draft HF config doesn't declare its
    training-time aux IDs.
    """
    if num_target_layers <= 3:
        raise ValueError(
            f"Default aux layer IDs require >3 target layers, got "
            f"{num_target_layers}."
        )
    early = 2
    late = num_target_layers - 3
    if fc_input_multiplier == 2:
        return [early, late]
    if fc_input_multiplier == 3:
        return [early, num_target_layers // 2, late]
    raise ValueError(
        f"Unsupported fc_input_multiplier={fc_input_multiplier}; "
        "expected 2 or 3."
    )
