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
"""MAX pipeline model base classes for model execution."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

from max.driver import Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.experimental.tensor import default_dtype
from max.graph import DeviceRef, Graph, Module, Value
from max.graph.weights import Weights, WeightsAdapter
from max.nn.kv_cache import (
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheParamInterface,
    PagedCacheValues,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContextType, LogProbabilities
from max.pipelines.kv_cache.config import (
    KVCacheConfig,
    cache_dtype_for_encoding,
)
from max.pipelines.lib.config.model_config import (
    _select_quantization_encoding,
)
from max.pipelines.lib.utils import (
    CompilationTimer,
    parse_state_dict_from_weights,
)
from max.pipelines.lora import (
    LoRAManagerV3,
    LoRATargetModule,
)
from max.pipelines.modeling.config_enums import (
    SupportedEncoding,
    supported_encoding_dtype,
)
from max.profiler import traced
from transformers import AutoConfig

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

    # TODO(MXF-517): Remove the interfaces/memory_estimation import cycle.
    from max.pipelines.lib.memory_estimation import MemoryPlan

    from .batch_processor import BatchProcessor

logger = logging.getLogger("max.pipelines")

ArchConfigT = TypeVar("ArchConfigT")


class AlwaysSignalBuffersMixin:
    """Mixin for models that always require signal buffers.

    Use this for models that use VocabParallelEmbedding or other distributed
    components that always perform allreduce, even on single-device setups.

    Models using this mixin build graphs that always include signal buffer
    inputs, regardless of device count. This is typically because they use
    distributed embedding layers or other components that call allreduce
    operations unconditionally.
    """

    devices: list[Device]
    """Device list that must be provided by the model class."""

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        """Override to always create signal buffers.

        Models using this mixin have distributed components that always
        perform allreduce, even for single-device setups. Therefore,
        signal buffers are always required to match the graph inputs.

        In compile-only mode (virtual device mode), returns an empty list
        to avoid GPU memory allocation which is not supported.

        Returns:
            List of signal buffer tensors, one per device, or empty list
            in compile-only mode.
        """
        # Import here to avoid circular dependency
        from max.nn.comm import Signals

        # Signals.allocate initializes the signal buffers and enables p2p access
        return Signals.allocate(self.devices)


@dataclass
class ModelOutputs:
    """Pipeline model outputs.

    Shape conventions below are for text-generation pipelines:

    - ``B``: batch size
    - ``V``: vocabulary size
    - ``H``: hidden-state width
    - ``T``: number of returned logit rows (depends on return mode)

    The shape depends on the value of the :class:`ReturnLogits` and :class:`ReturnHiddenStates`
    enums. Unless we are running with spec decoding, we use ``ReturnLogits.LAST_TOKEN``
    and ``ReturnHiddenStates.NONE``.
    """

    logits: Buffer
    """Primary logits buffer.

    For text generation this has shape ``[T, V]`` where:
    - last-token mode: ``T = B`` (default)
    - all-token mode: ``T = total_input_tokens``
    - variable mode: ``T = logit_offsets[-1]`` (typically ``B * return_n_logits``)
    """

    next_token_logits: Buffer | None = None
    """Next-token logits for text generation, shape ``[B, V]`` when present."""

    logit_offsets: Buffer | None = None
    """Cumulative row offsets into ``logits`` for text generation.

    Shape is ``[B + 1]``. Per-sequence logits are:
    ``logits[logit_offsets[i]:logit_offsets[i + 1], :]``.
    """

    hidden_states: Buffer | None = None
    """Optional hidden states for text generation.

    Single-device shape is ``[T_h, H]`` where:
    - none mode: NONE (default)
    - last-token mode: ``T_h = B``
    - all-token mode: ``T_h = total_input_tokens``

    For data parallel models, the hs will be on the first gpu since it is replicated.
    """


@dataclass(kw_only=True)
class ModelInputs:
    """Base class for model inputs.

    Use this class to encapsulate inputs for your model; you may store any
    number of dataclass fields.

    The following example demonstrates how to create a custom inputs class:

    .. code-block:: python

        from dataclasses import dataclass

        from max.driver import Buffer
        from max.dtype import DType
        from max.pipelines.lib.interfaces.pipeline_model import ModelInputs

        @dataclass
        class ReplitInputs(ModelInputs):
            tokens: Buffer
            input_row_offsets: Buffer

        # Create tensors
        tokens = Buffer.zeros((1, 2, 3), DType.int64)
        input_row_offsets = Buffer.zeros((1, 1, 1), DType.int64)

        # Initialize inputs
        inputs = ReplitInputs(tokens=tokens, input_row_offsets=input_row_offsets)

        # Access tensors
        assert inputs.tokens is tokens
        assert inputs.input_row_offsets is input_row_offsets
    """

    kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None
    """KV cache graph inputs holding every (DP replica x TP shard) device's
    inputs: a ``KVCacheInputs`` leaf, or a ``MultiKVCacheInputs`` tree for
    multi-cache models. ``flatten()`` yields the full positional input list."""

    lora_buffers: tuple[Buffer, ...] = ()
    """ModuleV3 LoRA graph inputs (routing triple + per-slot adapter stacks)
    in ``LoRAManagerV3.symbolic_inputs`` order, set by the ModuleV3 batch
    processor; empty when ModuleV3 LoRA is off. The arch's ``buffers``
    property splices these onto the positional ABI tail."""

    vision_embeddings: list[Buffer] = field(default_factory=list)
    """Per-device vision-merge embedding inputs for the language graph, set
    by the pipeline's vision seam (``finalize_vision_inputs``) on every
    prepared batch of a vision-capable model: the assembled embeddings when
    this step encoded images, the model's cached zero-row empties otherwise.
    Stays empty for text-only architectures."""

    vision_scatter_indices: list[Buffer] = field(default_factory=list)
    """Per-device merge (scatter) indices for :attr:`vision_embeddings`,
    with the same lifecycle."""

    hidden_states: Buffer | list[Buffer] | None = None
    """Hidden states for a variable number of tokens per sequence.

    For data parallel models, this can be a list of Buffers where each Buffer
    contains hidden states for the sequences assigned to that device.
    """

    def update(self, **kwargs) -> None:
        """Updates attributes from keyword arguments (only existing, non-None)."""
        key: str
        value: Any
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

    @property
    def buffers(self) -> tuple[Buffer, ...]:
        """Returns positional Buffer inputs for model ABI calls."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define model ABI buffers."
        )


@dataclass(kw_only=True)
class UnifiedEagleOutputs(ModelOutputs):
    """Outputs from a unified EAGLE graph execution."""

    num_accepted_draft_tokens: Buffer
    next_tokens: Buffer
    next_draft_tokens: Buffer
    next_draft_probs_full: Buffer | None = None
    """The distribution each next-step draft token was sampled from,
    ``[batch_size, num_speculative_tokens, vocab_size]``. Only populated when
    the graph was built with ``draft_proposal="sampled"``."""

    # HACK: These are required to inherit from ModelOutputs but are unused
    # for UnifiedEagleOutputs!
    logits: Buffer | None = None  # type: ignore[assignment]
    next_token_logits: None = None
    logit_offsets: None = None
    hidden_states: None = None


@dataclass(kw_only=True)
class UnifiedSpecDecodeInputs(ModelInputs):
    """Shared spec-decode fields + buffer-tail packing for unified ``*Inputs``.

    Each arch composes the tail via :meth:`_spec_decode_tail_buffers`, which
    mirrors ``build_spec_decode_input_types`` and must stay in lockstep with it.
    """

    draft_tokens: Buffer | None = None
    draft_probs_full: Buffer | None = None
    """The distribution each ``draft_tokens`` entry was sampled from,
    ``[batch_size, num_speculative_tokens, vocab_size]``. Only set when
    ``draft_proposal="sampled"``."""
    seed: Buffer | None = None
    temperature: Buffer | None = None
    top_k: Buffer | None = None
    max_k: Buffer | None = None
    top_p: Buffer | None = None
    min_top_p: Buffer | None = None
    in_thinking_phase: Buffer | None = None
    pinned_bitmask: Buffer | None = None
    wait_payload: Buffer | None = None
    device_bitmask_scratch: Buffer | None = None

    structured_output: bool = False
    """Whether this graph was compiled with constrained-decoding bitmask
    inputs. Mirrors ``pipeline_config.needs_bitmask_constraints`` -- the same
    value that gates the bitmask triple in ``build_spec_decode_input_types`` --
    so the buffer tail and the graph signature derive the decision from one
    place. Set by each capable module's ``prepare_initial_token_inputs``."""

    sampled_draft_proposal: bool = False
    """Whether this graph was compiled with ``draft_proposal="sampled"``,
    which gates the ``draft_probs_full`` buffer in the tail. Only
    ``UnifiedEagleLlama3`` and ``Eagle3MHAMiniMaxM3Unified`` set this today."""

    def _spec_decode_tail_buffers(
        self,
        *,
        include_in_thinking_phase: bool,
        supports_structured_output: bool = True,
        include_draft_probs_full: bool = False,
    ) -> tuple[Buffer, ...]:
        # draft_tokens, seed, and the five sampling params are unconditional in
        # build_spec_decode_input_types; assert them so a missing one is a loud
        # error, not a silently shortened ABI tuple. (Draft KV lives in the
        # {"target", "draft"} tree, packed by super().buffers.)
        assert self.draft_tokens is not None
        tail: tuple[Buffer, ...] = (self.draft_tokens,)
        if include_draft_probs_full:
            assert self.draft_probs_full is not None
            tail += (self.draft_probs_full,)
        assert self.seed is not None
        tail += (self.seed,)
        assert self.temperature is not None
        assert self.top_k is not None
        assert self.max_k is not None
        assert self.top_p is not None
        assert self.min_top_p is not None
        tail += (
            self.temperature,
            self.top_k,
            self.max_k,
            self.top_p,
            self.min_top_p,
        )
        if include_in_thinking_phase:
            assert self.in_thinking_phase is not None
            tail += (self.in_thinking_phase,)
        # Gate the bitmask triple on two compile-time flags, not a runtime
        # pinned_bitmask is not None check: supports_structured_output is False
        # for the dflash Llama3 graph, which still declares no bitmask graph
        # inputs even though the pipeline may set pinned_bitmask;
        # structured_output mirrors needs_bitmask_constraints, the same value
        # that gates the triple in build_spec_decode_input_types.
        if supports_structured_output and self.structured_output:
            assert self.pinned_bitmask is not None
            assert self.wait_payload is not None
            assert self.device_bitmask_scratch is not None
            tail += (
                self.pinned_bitmask,
                self.wait_payload,
                self.device_bitmask_scratch,
            )
        return tail


class PipelineModel(ABC, Generic[BaseContextType]):
    """A pipeline model with setup, input preparation and execution methods."""

    #: Optional batch processor class for input/output handling.
    batch_processor_cls: ClassVar[type[BatchProcessor[Any, Any]] | None] = None
    #: Config class used to build the arch config and delegate KV params.
    model_config_cls: ClassVar[type[Any] | None] = None
    #: Whether this arch serves LoRA via the ModuleV3 adapters-as-inputs path
    #: (``LoRAManagerV3``). Non-ModuleV3 archs cannot serve LoRA.
    lora_modulev3: ClassVar[bool] = False
    #: The ModuleV3 LoRA target projections this arch wraps. Read by the base
    #: to construct ``LoRAManagerV3``; empty for non-ModuleV3-LoRA archs.
    lora_targets: ClassVar[tuple[LoRATargetModule, ...]] = ()

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        *,
        memory_plan: MemoryPlan,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        self.pipeline_config = pipeline_config
        self.memory_plan = memory_plan
        self.max_batch_size = max_batch_size
        self.devices = devices
        self.device_refs = [DeviceRef.from_device(d) for d in devices]
        self.kv_cache_config = kv_cache_config
        self.weights = weights
        self.adapter = adapter
        self.return_logits = return_logits
        self.return_hidden_states = return_hidden_states

        # Graph modules read the length off this config.
        model_config_cls = type(self).model_config_cls
        self.arch_config: Any = (
            model_config_cls.initialize(
                pipeline_config, max_seq_len=self.max_seq_len
            )
            if model_config_cls is not None
            else None
        )

        if pipeline_config.lora and kv_cache_config.enable_prefix_caching:
            raise ValueError(
                "LoRA is incompatible with prefix caching; serve with "
                "prefix caching disabled (`--no-enable-prefix-caching`)."
            )

        self._lora_manager: LoRAManagerV3 | None
        if not pipeline_config.lora:
            self._lora_manager = None
        else:
            if not type(self).lora_modulev3:
                raise ValueError(
                    f"{type(self).__qualname__} does not support LoRA serving. "
                    "LoRA requires a ModuleV3 architecture; relaunch the "
                    "ModuleV3 variant of this model (e.g. `--prefer-module-v3`) "
                    "or serve without `--lora-paths`."
                )
            common_args = (
                pipeline_config.lora,
                pipeline_config.model.model_name,
                self.dtype,
                self.huggingface_config.num_attention_heads,
                self.huggingface_config.num_key_value_heads,
                self.huggingface_config.head_dim,
                self.max_seq_len * max_batch_size,
            )
            self._lora_manager = LoRAManagerV3(
                *common_args, targets=type(self).lora_targets
            )

        if isinstance(self._lora_manager, LoRAManagerV3):
            assert self.adapter is not None, (
                "ModuleV3 LoRA requires a base weight adapter to wrap"
            )
            self.adapter = self._lora_manager.lora_weight_adapter(self.adapter)

        self._batch_processor: BatchProcessor[Any, Any] | None = None
        batch_processor_cls = type(self).batch_processor_cls
        if batch_processor_cls is not None:
            from .batch_processor import BatchProcessorRuntime

            if self.arch_config is None:
                raise ValueError(
                    f"{type(self).__qualname__} sets batch_processor_cls but "
                    "does not define model_config_cls."
                )
            pad_token_id = getattr(self.huggingface_config, "pad_token_id", 0)
            self._batch_processor = batch_processor_cls(
                self.arch_config,
                BatchProcessorRuntime(
                    pipeline_config=pipeline_config,
                    devices=devices,
                    return_logits=return_logits,
                    return_hidden_states=return_hidden_states,
                    signal_buffers=self.signal_buffers,
                    lora_manager=self._lora_manager,
                    pad_token_id=pad_token_id or 0,
                    max_batch_size=self.max_batch_size,
                ),
            )

    @property
    def batch_processor(self) -> BatchProcessor[Any, Any] | None:
        """Returns the batch processor when configured."""
        return self._batch_processor

    def arch_config_as(self, cls: type[ArchConfigT]) -> ArchConfigT:
        """Returns the built-once arch config, narrowed to ``cls``."""
        assert isinstance(self.arch_config, cls), (
            f"{type(self).__qualname__} built "
            f"{type(self.arch_config).__name__}, expected {cls.__name__}"
        )
        return self.arch_config

    @property
    def max_seq_len(self) -> int:
        """The effective maximum sequence length, read from the memory plan.

        A view of the plan's ``planned_max_length`` — the model stores no
        copy of it.
        """
        max_length = self.memory_plan.planned_max_length
        assert max_length is not None, (
            f"{type(self).__qualname__} requires a memory plan with a max "
            "length; only multi-component plans carry none"
        )
        return max_length

    @property
    def planned_max_batch_total_tokens(self) -> int | None:
        """The plan's batch token budget."""
        return self.memory_plan.planned_max_batch_total_tokens

    def _maybe_release_host_weights(self, *models: Any) -> None:
        """Releases the host copies of the weights after the model is loaded.

        Gated on ``MODULAR_MAX_RELEASE_HOST_WEIGHTS=1``. Once the compiled
        model holds its device copy, drops the references that pin host
        weight memory: the engine registry (:meth:`Model.release_weights`),
        the retained state dicts, and the weight loader's file mappings.

        GPU deployments only: a CPU-resident weight is read in place on
        every execution, and releasing it is undefined behavior. ModuleV3
        compiled callables do not support the release.
        """
        if os.environ.get("MODULAR_MAX_RELEASE_HOST_WEIGHTS") != "1":
            return
        for model in models:
            release_weights = getattr(model, "release_weights", None)
            if release_weights is not None:
                release_weights()
        for attr in (
            "state_dict",
            "_vision_weights_dict",
            "_language_weights_dict",
        ):
            if hasattr(self, attr):
                setattr(self, attr, {})
        close = getattr(self.weights, "close", None)
        if close is not None:
            close()
        logger.info("Released host weight memory after model load.")

    @property
    def huggingface_config(self) -> AutoConfig:
        """Returns the HuggingFace config from pipeline config.

        For multimodal models (e.g., Pixtral, Gemma3 multimodal), this
        returns the top-level config which contains both text_config and
        vision_config. Models should explicitly access .text_config or
        .vision_config as needed.

        Returns:
            The HuggingFace AutoConfig for this model.

        Raises:
            ValueError: If HuggingFace config could not be loaded.
        """
        config = self.pipeline_config.model.huggingface_config
        if config is None:
            raise ValueError(
                f"HuggingFace config is required but could not be loaded for "
                f"model '{self.pipeline_config.model.model_path}'. "
                "Ensure the model repository contains a valid config.json."
            )
        return config

    @property
    def lora_manager(self) -> LoRAManagerV3 | None:
        """Returns the LoRA manager if LoRA is enabled, otherwise None."""
        return self._lora_manager

    @cached_property
    def signal_buffers(self) -> list[Buffer]:
        """Lazily initialize signal buffers for multi-GPU communication collectives.

        Signal buffers are only needed during model execution, not during compilation.
        By deferring their allocation, we avoid memory allocation in compile-only mode.

        Returns:
            List of signal buffer tensors, one per device for multi-device setups,
            or an empty list for single-device setups or compile-only mode.
        """
        if len(self.devices) <= 1:
            return []

        # Import here to avoid circular dependency
        from max.nn.comm import Signals

        # Signals.allocate initializes the signal buffers and enables p2p access
        return Signals.allocate(self.devices)

    @property
    def _resolved_encoding(self) -> SupportedEncoding:
        """The effective quantization encoding for this model.

        The config holds only the raw user value (possibly ``None``); encoding
        resolution lives in the consumer. Resolve here against the
        architecture's ``DEFAULT_ENCODING`` (the same value
        ``ArchConfig.initialize`` uses), for the generic consumers that hold
        only the config class rather than a built ``ArchConfig``.
        """
        model_config = self.pipeline_config.model
        default = getattr(
            getattr(type(self), "model_config_cls", None),
            "DEFAULT_ENCODING",
            None,
        )
        if default is not None:
            return _select_quantization_encoding(model_config, default)
        encoding = model_config.quantization_encoding
        if encoding is None:
            raise ValueError(
                "quantization_encoding could not be resolved for "
                f"'{model_config.model_path}'."
            )
        return encoding

    @property
    def dtype(self) -> DType:
        """Returns the model data type."""
        return supported_encoding_dtype(self._resolved_encoding)

    @property
    def sampler_custom_extensions(self) -> Sequence[Path]:
        """Custom-op extension paths to compile the sampler graph with."""
        return ()

    @abstractmethod
    def execute(
        self,
        model_inputs: ModelInputs,
    ) -> ModelOutputs:
        """Executes the graph with the given inputs.

        Args:
            model_inputs: The model inputs to execute, containing tensors and any other
                required data for model execution.

        Returns:
            ModelOutputs containing the pipeline's output tensors.

        This is an abstract method that must be implemented by concrete PipelineModels
        to define their specific execution logic.
        """

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[BaseContextType]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        """Prepares the initial inputs to be passed to ``execute()``.

        The inputs and functionality can vary per model. For example, model
        inputs could include encoded tensors, unique IDs per tensor when using
        a KV cache manager, and ``kv_cache_inputs`` (or None if the model does
        not use KV cache). This method typically batches encoded tensors,
        claims a KV cache slot if needed, and returns the inputs and caches.

        When :attr:`batch_processor_cls` is set, delegates to the batch processor.
        """
        if self._batch_processor is not None:
            return self._batch_processor.prepare_initial_token_inputs(
                replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )
        return self._prepare_initial_token_inputs(
            replica_batches, kv_cache_inputs, return_n_logits
        )

    def _prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[BaseContextType]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> ModelInputs:
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement prepare_initial_token_inputs "
            "or set batch_processor_cls."
        )

    def compute_log_probabilities(
        self,
        session: InferenceSession,
        model_inputs: ModelInputs,
        model_outputs: ModelOutputs,
        next_tokens: Buffer,
        batch_top_n: list[int],
        batch_echo: list[bool],
    ) -> list[LogProbabilities | None]:
        """Optional method that can be overridden to compute log probabilities.

        Args:
            session: Inference session to compute log probabilities within.
            model_inputs: Inputs to the model returned by
                ``prepare_initial_token_inputs()``.
            model_outputs: Outputs returned by `execute()`.
            next_tokens: Sampled tokens. Should have shape=[batch size]
            batch_top_n: Number of top log probabilities to return per input in
                the batch. For any element where `top_n == 0`, the
                LogProbabilities is skipped.
            batch_echo: Whether to include input tokens in the returned log
                probabilities.

        Returns:
            List of log probabilities.
        """
        raise NotImplementedError(
            f"Log probabilities not implemented for {type(self)}."
        )


class GraphPipelineModel(PipelineModel[BaseContextType]):
    """Graph-API pipeline model without KV cache.

    Subclasses implement :meth:`_build_graph_for_compile` and optionally
    :meth:`_create_model_config` and :meth:`_wire_batch_processor`.
    """

    state_dict: dict[str, Any]

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Load weights, build the graph, compile, and wire the batch processor."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)

        with CompilationTimer("model") as timer:
            graph, weights_registry = self._build_graph_for_compile(
                session, state_dict, model_config
            )
            timer.mark_build_complete()
            self.state_dict = weights_registry
            model = session.load(graph, weights_registry=weights_registry)

        self._maybe_release_host_weights(model)
        self._wire_batch_processor(model, model_config)
        return model

    def _load_state_dict(self) -> dict[str, Any]:
        """Load and optionally adapt weights from the configured source."""
        if self.adapter:
            return self.adapter(dict(self.weights.items()))
        return {key: value.data() for key, value in self.weights.items()}

    def _hf_config_for_weights(self) -> AutoConfig | None:
        """Optional HuggingFace config override for weight loading."""
        return None

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Optional hook; returns ``None`` when no arch config object is needed."""
        del state_dict
        return None

    def _wire_batch_processor(
        self, model: Any = None, model_config: Any = None
    ) -> None:
        """Optional hook to construct ``self.batch_processor`` after compile."""
        del model, model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        """Build the graph and return ``(graph, weights_registry)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_graph_for_compile"
        )


class ModuleV3PipelineModel(PipelineModel[BaseContextType]):
    """ModuleV3 eager pipeline model without KV cache.

    Subclasses implement :meth:`_instantiate_module` and optionally
    :meth:`_create_model_config`, :meth:`_prepare_state_dict`, and
    :meth:`_module_default_dtype`.
    """

    @traced
    def load_model(self) -> Callable[..., Any]:
        """Build and compile the ModuleV3 callable."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        state_dict = self._prepare_state_dict(state_dict, model_config)

        with CompilationTimer("model") as timer:
            module_default_dtype = self._module_default_dtype(
                state_dict, model_config
            )
            with F.lazy(), default_dtype(module_default_dtype):
                nn_model = self._instantiate_module(model_config)
            compile_input_types = self._get_compile_input_types(model_config)
            timer.mark_build_complete()
            return nn_model.compile(*compile_input_types, weights=state_dict)

    def _load_state_dict(self) -> dict[str, Any]:
        """Load and optionally adapt weights from the configured source."""
        if self.adapter:
            return self.adapter(dict(self.weights.items()))
        return {key: value.data() for key, value in self.weights.items()}

    def _hf_config_for_weights(self) -> AutoConfig | None:
        """Optional HuggingFace config override for weight loading."""
        return None

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Builds model config from ``state_dict``."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_create_model_config`."
        )

    def _prepare_state_dict(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> dict[str, Any]:
        """Optional hook to cast or rewrite weights before ``nn.compile``."""
        del model_config
        return state_dict

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        """Default dtype for the eager module build context."""
        del state_dict, model_config
        return self.dtype

    def _instantiate_module(self, model_config: Any) -> Any:
        """Constructs and places the nn module under ``F.lazy()``."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_instantiate_module`."
        )

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        """Symbolic inputs passed to ``nn_model.compile``."""
        del model_config
        batch_processor = self.batch_processor
        assert batch_processor is not None
        return tuple(
            batch_processor.get_symbolic_inputs(
                kv_params=cast(KVCacheParamInterface, None),
                device_refs=self.device_refs,
            )
        )


class PipelineModelWithKVCache(PipelineModel[BaseContextType]):
    """A pipeline model that supports KV cache."""

    kv_params: KVCacheParamInterface

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        session: InferenceSession,
        devices: list[Device],
        kv_cache_config: KVCacheConfig,
        weights: Weights,
        adapter: WeightsAdapter | None,
        return_logits: ReturnLogits,
        *,
        memory_plan: MemoryPlan,
        return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE,
        max_batch_size: int = 1,
    ) -> None:
        super().__init__(
            pipeline_config=pipeline_config,
            session=session,
            devices=devices,
            kv_cache_config=kv_cache_config,
            weights=weights,
            adapter=adapter,
            return_logits=return_logits,
            return_hidden_states=return_hidden_states,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )
        self.kv_params = type(self).get_kv_params(
            huggingface_config=self.huggingface_config,
            pipeline_config=self.pipeline_config,
            devices=self.device_refs,
            kv_cache_config=self.kv_cache_config,
            cache_dtype=cache_dtype_for_encoding(
                self._resolved_encoding,
                self.pipeline_config.model.kv_cache.kv_cache_format,
            ),
        )

    def _unflatten_kv_inputs(
        self, kv_inputs_flat: Sequence[Value[Any]]
    ) -> list[PagedCacheValues]:
        # This helper supports single-cache (leaf) models; multi-cache trees
        # are unflattened by the architecture itself.
        kv_inputs = self.kv_params.unflatten_kv_inputs(iter(kv_inputs_flat))
        assert isinstance(kv_inputs, KVCacheInputs)
        return list(kv_inputs.inputs)

    @classmethod
    def get_kv_params(
        cls,
        huggingface_config: AutoConfig,
        pipeline_config: PipelineConfig,
        devices: list[DeviceRef],
        kv_cache_config: KVCacheConfig,
        cache_dtype: DType,
    ) -> KVCacheParamInterface:
        """Returns the KV cache params for the pipeline model.

        Delegates to ``model_config_cls.construct_kv_params(...)``.
        Subclasses with custom KV behavior should override this method.
        """
        model_config_cls = cls.model_config_cls
        if model_config_cls is None:
            raise NotImplementedError(
                f"{cls.__qualname__} must set `model_config_cls` "
                "or override `get_kv_params()`."
            )
        return model_config_cls.construct_kv_params(
            huggingface_config,
            pipeline_config,
            devices,
            kv_cache_config,
            cache_dtype,
        )

    def _load_state_dict(self) -> dict[str, Any]:
        """Loads weights via :func:`~max.pipelines.lib.utils.parse_state_dict_from_weights`."""
        return parse_state_dict_from_weights(
            self.pipeline_config,
            self.weights,
            self.adapter,
            hf_config=self._hf_config_for_weights(),
        )

    def _hf_config_for_weights(self) -> AutoConfig | None:
        """HuggingFace config passed to the weight adapter, if any."""
        return None

    def _wire_batch_processor(
        self,
        model: Any = None,
        model_config: Any = None,
    ) -> None:
        """Post-compile wiring into the batch processor (EP bind, vision, etc.)."""
        del model, model_config
        batch_processor = self.batch_processor
        if batch_processor is None:
            return
        bind_ep = getattr(batch_processor, "bind_ep_comm_initializer", None)
        if bind_ep is not None:
            bind_ep(getattr(self, "ep_comm_initializer", None))


class GraphPipelineModelWithKVCache(PipelineModelWithKVCache[BaseContextType]):
    """Graph-API pipeline model with shared compile-and-load template.

    Subclasses override :meth:`_build_graph_for_compile` (and optionally
    :meth:`_create_model_config`, :meth:`_init_distributed_runtime`) rather than
    duplicating weight loading, timing, and EP batch-processor wiring.

    ModuleV3 (eager) models and multi-graph VLMs should inherit
    :class:`MultiGraphPipelineModelWithKVCache` instead.
    """

    @traced
    def load_model(self, session: InferenceSession) -> Model:
        """Build, compile, and load the model graph into ``session``."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(session, model_config)

        with CompilationTimer("model") as timer:
            graph, weights_registry = self._build_graph_for_compile(
                session,
                state_dict,
                model_config,
            )
            timer.mark_build_complete()
            self.state_dict = weights_registry
            model = session.load(graph, weights_registry=weights_registry)

        self._maybe_release_host_weights(model)
        self._wire_batch_processor(model, model_config)
        return model

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Builds model config from ``state_dict``.

        Subclasses implement ``initialize`` / ``finalize`` (or heavier setup)
        here. There is no separate finalize hook.
        """
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_create_model_config`."
        )

    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        """Initializes EP/NVSHMEM or other distributed runtime (no-op by default)."""
        del session, model_config

    def _build_graph_for_compile(
        self,
        session: InferenceSession,
        state_dict: dict[str, Any],
        model_config: Any,
    ) -> tuple[Graph, dict[str, Any]]:
        """Instantiates the nn module, captures the graph, returns the registry."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_build_graph_for_compile`."
        )


class MultiGraphPipelineModelWithKVCache(
    PipelineModelWithKVCache[BaseContextType]
):
    """Graph-API VLM with unified :meth:`load_model` and per-tower hooks.

    :meth:`_create_model_config` should return the full VLM config (with
    ``.vision_config`` and ``.text_config`` / ``.llm_config`` subconfigs) and
    assign :attr:`model_config`. Both :meth:`_build_*` hooks receive that same
    ``model_config``.

    Override :meth:`load_model` when graph capture or weight loading does not
    fit this flow (e.g. Qwen2.5VL, Kimi-K2.5).
    """

    _vision_weights_dict: dict[str, Any]
    _language_weights_dict: dict[str, Any]

    @traced
    def load_model(
        self, session: InferenceSession
    ) -> tuple[Model | None, Model]:
        """Build, compile, and load vision and language graphs into ``session``."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(session, model_config)

        with CompilationTimer("vision + language model") as timer:
            graph_module = Module()

            vision_graph: Graph | None = None
            vision_registry: dict[str, Any] = {}
            if self._include_vision_graph(model_config):
                vision_graph, vision_registry = self._build_vision_graph(
                    model_config,
                    self._vision_weights_dict,
                    module=graph_module,
                )

            language_graph, language_registry = self._build_language_graph(
                model_config,
                self._language_weights_dict,
                module=graph_module,
            )
            timer.mark_build_complete()

            models = session.load_all(
                graph_module,
                weights_registry={**vision_registry, **language_registry},
            )

        self._maybe_release_host_weights(*models.values())
        vision_model = (
            models[vision_graph.name] if vision_graph is not None else None
        )
        language_model = models[language_graph.name]
        self._wire_batch_processor(vision_model, model_config)
        return vision_model, language_model

    def _include_vision_graph(self, model_config: Any) -> bool:
        """Whether to capture and load a vision graph (override for text-only VLMs)."""
        del model_config
        return True

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Builds the full VLM config from ``state_dict``.

        Should assign :attr:`model_config` and return the same object.
        """
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_create_model_config`."
        )

    def _init_distributed_runtime(
        self,
        session: InferenceSession,
        model_config: Any,
    ) -> None:
        """Initializes EP/NVSHMEM or other distributed runtime (no-op by default)."""
        del session, model_config

    def _build_vision_graph(
        self,
        model_config: Any,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        """Captures the vision tower graph and its ``nn.state_dict()`` registry."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_build_vision_graph`."
        )

    def _build_language_graph(
        self,
        model_config: Any,
        state_dict: dict[str, Any],
        module: Module,
    ) -> tuple[Graph, dict[str, Any]]:
        """Captures the language tower graph and its ``nn.state_dict()`` registry."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_build_language_graph`."
        )


class ModuleV3PipelineModelWithKVCache(
    PipelineModelWithKVCache[BaseContextType]
):
    """ModuleV3 (eager) pipeline model with shared compile template.

    Subclasses override :meth:`_instantiate_module` (and optionally
    :meth:`_create_model_config`, :meth:`_init_distributed_runtime`,
    :meth:`_module_default_dtype`, :meth:`_get_compile_input_types`) rather than
    duplicating weight loading, timing, and ``nn.compile`` wiring.

    Graph-API models should inherit :class:`GraphPipelineModelWithKVCache`
    instead. Encoder models without KV cache should inherit
    :class:`ModuleV3PipelineModel` instead. Multi-graph VLMs should inherit
    :class:`MultiGraphPipelineModelWithKVCache` (graph API) or
    :class:`ModuleV3MultiGraphPipelineModelWithKVCache` (ModuleV3).
    ``ComponentModel`` types and unified spec-decode pipelines should override
    :meth:`load_model` entirely.
    """

    _modulev3_extra_input_types: list[Any]

    @traced
    def load_model(self) -> Callable[..., Any]:
        """Build and compile the ModuleV3 callable."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(model_config)
        module_default_dtype = self._module_default_dtype(
            state_dict, model_config
        )
        with F.lazy(), default_dtype(module_default_dtype):
            nn_model = self._instantiate_module(model_config)
            if isinstance(self._lora_manager, LoRAManagerV3):
                nn_model = self._lora_manager.wrap(nn_model)
                self._modulev3_extra_input_types = (
                    self._lora_manager.symbolic_inputs(self.device_refs[0])
                )
        compile_input_types = self._get_compile_input_types(model_config)
        self._wire_batch_processor(nn_model, model_config)
        return nn_model.compile(*compile_input_types, weights=state_dict)

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Builds model config from ``state_dict``."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_create_model_config`."
        )

    def _init_distributed_runtime(self, model_config: Any) -> None:
        """Initializes EP/NVSHMEM or other distributed runtime (no-op by default)."""
        del model_config
        self._modulev3_extra_input_types = []

    def _module_default_dtype(
        self, state_dict: dict[str, Any], model_config: Any
    ) -> DType:
        """Default dtype for the eager module build context."""
        del state_dict
        return model_config.dtype

    def _instantiate_module(self, model_config: Any) -> Any:
        """Constructs and places the nn module under ``F.lazy()``."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_instantiate_module`."
        )

    def _get_compile_input_types(self, model_config: Any) -> tuple[Any, ...]:
        """Symbolic inputs passed to ``nn_model.compile``."""
        del model_config
        batch_processor = self.batch_processor
        assert batch_processor is not None
        input_types = list(
            batch_processor.get_symbolic_inputs(
                kv_params=self.kv_params,
                device_refs=self.device_refs,
            )
        )
        input_types.extend(self._modulev3_extra_input_types)
        return tuple(input_types)


class ModuleV3MultiGraphPipelineModelWithKVCache(
    PipelineModelWithKVCache[BaseContextType]
):
    """ModuleV3 VLM with separate vision and language compiled callables.

    Subclasses implement :meth:`_load_state_dict` (tower weight prep),
    :meth:`_create_model_config`, and :meth:`_compile_vision_model` /
    :meth:`_compile_language_model`. The base :meth:`load_model` passes each
    tower's ``WeightData`` dict into the matching compile hook (the vision or
    language slice of :attr:`_vision_weights_dict` / :attr:`_language_weights_dict`,
    not the raw checkpoint returned from :meth:`_load_state_dict`).

    Graph-API VLMs should inherit :class:`MultiGraphPipelineModelWithKVCache`
    instead.
    """

    _vision_weights_dict: dict[str, Any]
    _language_weights_dict: dict[str, Any]

    @traced
    def load_model(
        self,
    ) -> tuple[Callable[..., Any] | None, Callable[..., Any]]:
        """Build and compile vision and language ModuleV3 callables."""
        state_dict = self._load_state_dict()
        model_config = self._create_model_config(state_dict)
        self._init_distributed_runtime(model_config)

        vision_model = self._compile_vision_model(
            model_config, self._vision_weights_dict
        )
        language_model = self._compile_language_model(
            model_config, self._language_weights_dict
        )

        self._wire_batch_processor(vision_model, model_config)
        return vision_model, language_model

    def _create_model_config(self, state_dict: dict[str, Any]) -> Any:
        """Builds model config from ``state_dict``."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_create_model_config`."
        )

    def _init_distributed_runtime(self, model_config: Any) -> None:
        """Initializes EP/NVSHMEM or other distributed runtime (no-op by default)."""
        del model_config

    def _compile_vision_model(
        self, model_config: Any, state_dict: dict[str, Any]
    ) -> Callable[..., Any]:
        """Builds and compiles the vision tower."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_compile_vision_model`."
        )

    def _compile_language_model(
        self, model_config: Any, state_dict: dict[str, Any]
    ) -> Callable[..., Any]:
        """Builds and compiles the language tower."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement `_compile_language_model`."
        )
