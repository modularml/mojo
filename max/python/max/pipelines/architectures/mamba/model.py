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

from __future__ import annotations

import logging
from typing import Any, ClassVar, cast

import numpy as np
from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import LogProbabilities, TextContext
from max.pipelines.lib import (
    KVCacheConfig,
    ModelInputs,
    ModelOutputs,
    PipelineConfig,
    PipelineModelWithKVCache,
)
from max.pipelines.lib.log_probabilities import (
    compute_log_probabilities_ragged,
    log_probabilities_ragged_graph,
)
from max.pipelines.lib.memory_estimation import MemoryPlan
from max.pipelines.lib.utils import parse_state_dict_from_weights
from max.pipelines.modeling.types import RequestID
from max.profiler import traced
from transformers import AutoConfig

from .batch_processor import MambaBatchProcessor
from .functional_ops import _get_state_space_paths
from .model_config import MambaConfig
from .ssm_cache import SSMStateCache

logger = logging.getLogger("max.pipelines")


class MambaModelInputs(ModelInputs):
    """Inputs for the Mamba pipeline model."""

    tokens: Buffer
    input_row_offsets: Buffer
    return_n_logits: Buffer
    is_prefill: bool
    layer_states: list[Buffer]
    request_ids: list[RequestID]

    def __init__(
        self,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        is_prefill: bool = True,
        layer_states: list[Buffer] | None = None,
        request_ids: list[RequestID] | None = None,
    ) -> None:
        self.tokens = tokens
        self.input_row_offsets = input_row_offsets
        self.return_n_logits = return_n_logits
        self.is_prefill = is_prefill
        self.layer_states = layer_states or []
        self.request_ids = request_ids or []


class MambaModel(PipelineModelWithKVCache[TextContext]):
    """Mamba pipeline model with incremental SSM state caching.

    Uses separate compiled prefill and step models. The prefill model
    processes the full prompt and extracts per-layer conv/ssm states.
    The step model processes single new tokens using cached states.
    """

    model_config_cls: ClassVar[type[Any]] = MambaConfig
    batch_processor_cls: ClassVar[type[MambaBatchProcessor]] = (
        MambaBatchProcessor
    )

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
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self._prefill_model, self._step_model = self._load_models(session)
        self._ssm_cache = self._create_ssm_cache()
        if self._batch_processor is not None:
            bind = getattr(self._batch_processor, "bind_ssm_cache", None)
            if bind is not None:
                bind(self._ssm_cache)
        self.logprobs_device = devices[0]
        self.logprobs_model = self._load_logprobs_model(session)

    def _create_ssm_cache(self) -> SSMStateCache:
        """Create the SSM state cache with pre-allocated slot buffers."""
        cfg = self._model_config
        max_slots = self.max_batch_size
        return SSMStateCache(
            num_layers=cfg.num_hidden_layers,
            intermediate_size=cfg.intermediate_size,
            d_state=cfg.d_state,
            conv_kernel=cfg.conv_kernel,
            dtype=self.dtype,
            max_slots=max_slots,
            device=self.devices[0],
        )

    @classmethod
    def get_num_layers(cls, huggingface_config: AutoConfig) -> int:
        return MambaConfig.get_num_layers(huggingface_config)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Return minimal dummy KV cache params.

        Mamba uses SSM state caching internally, not attention KV cache.
        These dummy params satisfy the PipelineModelWithKVCache interface
        with negligible memory overhead.
        """
        return MHAKVCacheParams(
            dtype=cache_dtype or DType.float32,
            n_kv_heads=1,
            head_dim=1,
            num_layers=1,
            devices=devices,
            page_size=128,
        )

    def _load_logprobs_model(self, session: InferenceSession) -> Model:
        graph = log_probabilities_ragged_graph(
            DeviceRef.from_device(self.logprobs_device), levels=3
        )
        return session.load(graph)

    @traced
    def _load_models(self, session: InferenceSession) -> tuple[Any, Any]:
        from max.experimental import functional as F
        from max.graph import TensorType
        from max.pipelines.lib import CompilationTimer

        from .mamba_module import MambaPrefill, MambaStep

        assert self.max_batch_size, "Expected max_batch_size to be set"
        self._input_row_offsets_prealloc = Buffer.from_numpy(
            np.arange(
                self.max_batch_size + 1,
                dtype=np.uint32,
            )
        ).to(self.devices[0])

        # Parse state dict using the standard utility.
        state_dict = parse_state_dict_from_weights(
            self.pipeline_config, self.weights, self.adapter
        )

        # Build config: initialize from HF config, then finalize with weights.
        model_config = MambaConfig.initialize(
            self.pipeline_config, max_seq_len=self.max_seq_len
        )
        model_config.finalize(
            huggingface_config=self.huggingface_config,
            state_dict=state_dict,
            return_logits=self.return_logits,
        )

        device0 = self.devices[0]
        device_ref = DeviceRef(device0.label, device0.id)

        # Store config values for state tensor types.
        self._model_config = model_config
        num_layers = model_config.num_hidden_layers
        intermediate = model_config.intermediate_size
        d_state = model_config.d_state
        conv_width = model_config.conv_kernel

        # Get kernel library paths for custom extensions.
        kernel_paths = list(_get_state_space_paths())

        # --- Compile prefill model ---
        tokens_type = TensorType(
            DType.int64, shape=["total_seq_len"], device=device_ref
        )
        row_offsets_type = TensorType(
            DType.uint32, shape=["row_offsets_len"], device=device_ref
        )

        from max.experimental.tensor import default_dtype

        with CompilationTimer("prefill") as timer:
            with F.lazy(), default_dtype(self.dtype):
                prefill_module = MambaPrefill(model_config)
                prefill_module.to(device0)

            timer.mark_build_complete()
            prefill_model = prefill_module.compile(
                tokens_type,
                row_offsets_type,
                weights=state_dict,
                custom_extensions=kernel_paths,
            )

        # --- Compile step model ---
        timer = CompilationTimer("step")
        step_tokens_type = TensorType(
            DType.int64, shape=["batch"], device=device_ref
        )
        layer_state_types: list[TensorType] = []
        for _ in range(num_layers):
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", intermediate, conv_width],
                    device=device_ref,
                )
            )
            layer_state_types.append(
                TensorType(
                    self.dtype,
                    shape=["batch", intermediate, d_state],
                    device=device_ref,
                )
            )

        with CompilationTimer("step") as timer:
            with F.lazy(), default_dtype(self.dtype):
                step_module = MambaStep(model_config)
                step_module.to(device0)

            timer.mark_build_complete()
            step_model = step_module.compile(
                step_tokens_type,
                *layer_state_types,
                weights=state_dict,
                custom_extensions=kernel_paths,
            )

        return prefill_model, step_model

    def execute(self, model_inputs: ModelInputs) -> ModelOutputs:
        assert isinstance(model_inputs, MambaModelInputs)

        if model_inputs.is_prefill:
            outputs = self._prefill_model(
                model_inputs.tokens,
                model_inputs.input_row_offsets,
            )
        else:
            outputs = self._step_model(
                model_inputs.tokens,
                *model_inputs.layer_states,
            )

        # First output is logits, rest are layer states.
        logits = cast(Buffer, outputs[0].driver_tensor)

        # Store updated states back into the SSM cache slots.
        new_states = [s.driver_tensor for s in outputs[1:]]
        if model_inputs.request_ids:
            self._ssm_cache.update_states(model_inputs.request_ids, new_states)

        return ModelOutputs(
            logits=logits,
            next_token_logits=logits,
        )

    def release(self, request_id: RequestID) -> None:
        """Release SSM cache slot when a request completes."""
        self._ssm_cache.release(request_id)

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        logits = model_outputs.logits
        assert model_outputs.next_token_logits is not None
        next_token_logits = model_outputs.next_token_logits

        assert isinstance(model_inputs, MambaModelInputs)

        sampled_tokens = next_tokens.to_numpy()
        tokens = model_inputs.tokens.to_numpy()
        input_row_offsets = model_inputs.input_row_offsets.to_numpy()

        return compute_log_probabilities_ragged(
            self.logprobs_device,
            self.logprobs_model,
            input_row_offsets=input_row_offsets,
            logits=logits,
            next_token_logits=next_token_logits,
            tokens=tokens,
            sampled_tokens=sampled_tokens,
            batch_top_n=batch_top_n,
            batch_echo=batch_echo,
        )
