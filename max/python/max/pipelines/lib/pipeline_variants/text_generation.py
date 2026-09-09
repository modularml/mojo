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
"""MAX pipeline for model inference and generation (Text Generation variant)."""

from __future__ import annotations

import dataclasses
import json
import logging
from abc import abstractmethod
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic

import numpy as np
import numpy.typing as npt
from max.driver import (
    CPU,
    Buffer,
    Device,
    DevicePinnedBuffer,
    Usage,
    is_virtual_device_mode,
    load_devices,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn import ReturnLogits
from max.pipelines.context import (
    BatchLogitsProcessor,
    LogProbabilities,
    TextAndVisionContext,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.context.exceptions import (  # noqa: F401 (for docstring)
    InputError,
)
from max.pipelines.kv_cache import PagedKVCacheManagerInterface, load_kv_manager
from max.pipelines.lib.vision_encoder_cache import (
    SupportsPooledVisionMetrics,
    SupportsVisionEncoding,
    VisionEncoderCache,
    as_vision_context_batches,
)
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineOutputsDict,
    PipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.profiler import Tracer, traced
from max.support.algorithm import flatten2d

from .utils import StructuredOutputHelper, update_context_and_prepare_responses

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

from max.pipelines.sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

from ..interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
)
from ..interfaces.generate import GenerateMixin
from ..memory_estimation import MemoryPlan
from ..utils import CompilationTimer
from ..vision_encoder_cache import VideoEncoderMetrics, VisionEncoderMetrics

logger = logging.getLogger("max.pipelines")


@dataclasses.dataclass
class BatchInfo:
    """Information about a batch of requests passed to the pipeline."""

    past_seq_lens: list[int]
    """Coordinated list of past sequence lengths (i.e. context lengths)"""

    seq_lens: list[int]
    """Coordinated list of sequence lengths, i.e. prompt_len or 1"""


class TextGenerationPipelineInterface(
    Pipeline[
        TextGenerationInputs[TextGenerationContextType], TextGenerationOutput
    ],
    GenerateMixin[TextGenerationContextType, TextGenerationRequest],
    Generic[TextGenerationContextType],
):
    """Interface for text generation pipelines."""

    # TODO: Get rid of these fields
    _devices: list[Device]
    _pipeline_model: PipelineModelWithKVCache[TextGenerationContextType]

    @property
    @abstractmethod
    def kv_manager(self) -> PagedKVCacheManagerInterface:
        """Returns the KV cache managers for this pipeline."""
        ...


class TextGenerationPipeline(
    TextGenerationPipelineInterface[TextGenerationContextType],
    Generic[TextGenerationContextType],
):
    """Generalized token generator pipeline."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[TextGenerationContextType]],
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextGenerationContextType,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
        memory_plan: MemoryPlan,
    ) -> None:
        """Initialize a text generation pipeline instance.

        This sets up devices, the inference session, tokenizer, KV-cache manager,
        sampling kernel, and loads model weights and adapters.

        Args:
            pipeline_config: Configuration for the pipeline and runtime behavior.
            pipeline_model: Concrete model implementation to use for execution.
            weight_adapters: Mapping from weights format to adapter implementation.
            tokenizer: Tokenizer implementation used to build contexts and decode.
            memory_plan: Memory plan from the registry containing max_batch_size
                and other resolved memory parameters.

        Raises:
            ValueError: If ``quantization_encoding`` is not configured in
                ``pipeline_config.model`` or if structured output is
                requested without a valid tokenizer delegate.
        """
        self._pipeline_config = pipeline_config
        self._max_batch_size = memory_plan.planned_max_batch_size
        max_batch_size = memory_plan.planned_max_batch_size
        model_config: MAXModelConfig = pipeline_config.model
        huggingface_config = model_config.huggingface_config
        if huggingface_config is None:
            raise ValueError(
                f"Text generation pipeline requires a HuggingFace config for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        self._devices = load_devices(list(memory_plan.require_device_specs()))
        self._tokenizer = tokenizer

        self.batch_info_output_fname = environ.get(
            "MAX_BATCH_INFO_FILENAME", None
        )
        self.batch_infos: list[BatchInfo] = []

        # Initialize structured output helper for constrained decoding.
        # The helper's ``enable_response_format_schema`` mirrors the user
        # flag and gates user-supplied JSON schemas; the bitmask-in-the-graph
        # decisions below are gated separately on
        # ``pipeline_config.needs_bitmask_constraints``.
        # structured_output_backend is None only on an unresolved config;
        # from_tokenizer falls back to "xgrammar" in that case.
        self._structured_output = StructuredOutputHelper.from_tokenizer(
            self.tokenizer,
            pipeline_config.sampling.enable_structured_output,
            pipeline_config.runtime.tool_parser,
            pipeline_config.sampling.structured_output_backend,
            pipeline_config.sampling.structured_output_any_whitespace,
        )
        self.vocab_size = self._structured_output.vocab_size

        # Initialize Session.
        session = InferenceSession(
            devices=[*self._devices],
            precompiled_mefs=pipeline_config.runtime.precompiled_mefs,
            export_mefs=pipeline_config.runtime.export_mefs,
        )
        self.session = session

        # Configure session with pipeline settings.
        self._pipeline_config.configure_session(session)

        # Load model.
        # Retrieve the weights repo id (falls back to model_path when unset).
        weight_paths: list[Path] = model_config.resolved_weight_paths()

        if not issubclass(pipeline_model, PipelineModelWithKVCache):
            raise ValueError(
                f"TextGenerationPipeline requires a model with KV cache support, found {pipeline_model.__name__}"
            )
        self._pipeline_model = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            kv_cache_config=model_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=ReturnLogits.ALL
            if self._pipeline_config.model.enable_echo
            else ReturnLogits.LAST_TOKEN,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        available_cache_memory = memory_plan.available_cache_memory
        kv_params = self._pipeline_model.kv_params
        self._kv_manager = load_kv_manager(
            params=kv_params,
            max_batch_size=max_batch_size,
            max_seq_len=self._pipeline_model.max_seq_len,
            session=session,
            available_cache_memory=available_cache_memory,
            is_di_enabled=pipeline_config.runtime.is_disaggregated,
            model_name=pipeline_config.model.model_name,
        )

        self._encoder_cache: VisionEncoderCache[TextAndVisionContext] | None = (
            None
        )
        if isinstance(self._pipeline_model, SupportsVisionEncoding):
            self._encoder_cache = VisionEncoderCache[TextAndVisionContext](
                plan=memory_plan.vision_cache_plan,
                devices=self._devices,
            )

        # Device the sampler runs on. ``sample_on_host`` routes sampling to the
        # host CPU.
        self._sampler_device: Device = (
            CPU()
            if pipeline_config.sampling.sample_on_host
            else self._devices[0]
        )
        sampler_device_ref = DeviceRef.from_device(self._sampler_device)

        # Load sampler. The bitmask-aware sampler is loaded when constrained
        # decoding could fire (see ``needs_bitmask_constraints``).
        self._sampler_with_bitmask: Model | None = None
        self._sampler_without_bitmask: Model | None = None
        with_bitmask_graph = None
        with CompilationTimer("sampler") as sampler_timer:
            if pipeline_config.needs_bitmask_constraints:
                with_bitmask_graph = token_sampler(
                    pipeline_config.sampling,
                    device=sampler_device_ref,
                    needs_bitmask_input=True,
                )
            without_bitmask_graph = token_sampler(
                pipeline_config.sampling,
                device=sampler_device_ref,
                needs_bitmask_input=False,
            )
            sampler_timer.mark_build_complete()
            if with_bitmask_graph is not None:
                self._sampler_with_bitmask = session.load(with_bitmask_graph)
            self._sampler_without_bitmask = session.load(without_bitmask_graph)

        # Pre-allocate pinned buffer for D2H token copies only when the
        # bitmask path is wired in. This buffer is used for async token
        # transfers in the guided decoding path. Allocated once and reused
        # across batches. Skip in virtual device mode (compile-only) since
        # VirtualDevice does not support memory allocation.
        self._pinned_new_tokens: Buffer | None = None
        if (
            pipeline_config.needs_bitmask_constraints
            and not self._sampler_device.is_host
            and not is_virtual_device_mode()
        ):
            self._pinned_new_tokens = DevicePinnedBuffer(
                shape=(max_batch_size,),
                dtype=DType.int64,
                device=self._sampler_device,
            )

        self._identity_logit_offsets = (
            FusedSamplingProcessor.allocate_identity_logit_offsets(
                pipeline_config, self._sampler_device, max_batch_size
            )
        )

    @property
    def max_batch_size(self) -> int:
        """Maximum number of requests that can be processed in a single batch."""
        return self._max_batch_size

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Return the tokenizer used for building contexts and decoding."""
        return self._tokenizer

    def update_for_structured_output(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Update context and logits bitmask for structured output.

        If a ``json_schema`` is present and no matcher is set, this compiles a
        grammar matcher and installs it on the context. It may also jump ahead in
        generation and fills the per-request token bitmask used to constrain the
        next-token distribution.

        Args:
            context: Request context to update.
            bitmask: Optional preallocated bitmask buffer; updated in-place.
            index: Global position into the bitmask for this request.

        Raises:
            InputError: If a JSON schema is provided but structured output is not
                enabled via sampling configuration.
        """
        self._structured_output.update_context(context, bitmask, index)

    def initialize_bitmask(
        self, batch: list[TextGenerationContextType]
    ) -> npt.NDArray[np.int32] | None:
        """Allocates a per-request token bitmask for structured decoding.

        Args:
            batch: The generation contexts for the batch.

        Returns:
            A bitmask array of shape [batch_size, vocab_size] if structured
            output is enabled; otherwise ``None``.
        """
        if not self._structured_output.enabled:
            return None

        if all(
            context.json_schema is None and context.grammar is None
            for context in batch
        ):
            return None

        return self._structured_output.allocate_bitmask(len(batch))

    @traced
    def prepare_batch(
        self,
        batches: list[list[TextGenerationContextType]],
    ) -> tuple[
        Any,
        npt.NDArray[np.int32] | None,
        list[TextGenerationContextType],
    ]:
        """Prepare model inputs and ancillary state for execution.

        This flattens replica batches, optionally initializes constrained
        decoding bitmasks, ensures KV-cache reservations, and builds
        initial model inputs.

        Args:
            batches: Per-replica list of contexts.

        Returns:
            A tuple of:
                - ModelInputs: Prepared inputs for the step.
                - Optional[np.ndarray]: The structured decoding bitmask or None.
                - list[TextGenerationContextType]: The flattened context batch.
        """
        replica_batches: list[list[TextGenerationContextType]] = [
            self._maybe_sort_loras(batch) for batch in batches
        ]
        flat_batch = flatten2d(replica_batches)

        # Initialize a bitmask for structured output.
        bitmask = self.initialize_bitmask(flat_batch)

        # Keep a global index for bitmask indexing.
        for i, context in enumerate(flat_batch):
            # Update state for structured output. Initialize a matcher if needed, this includes:
            # - Initializing a matcher if needed [once per request]
            # - Jumping ahead in generation if possible
            # - Updating the bitmask for the context.
            if bitmask is not None:
                self.update_for_structured_output(context, bitmask, i)

        # Retrieve the KV Cache Inputs.
        kv_cache_inputs = self._kv_manager.runtime_inputs(replica_batches)

        # Log batch details
        if self.batch_info_output_fname is not None:
            self._record_batch_info(flat_batch)

        model_inputs = self._pipeline_model.prepare_initial_token_inputs(
            replica_batches=replica_batches,
            kv_cache_inputs=kv_cache_inputs,
        )

        if self._encoder_cache is not None:
            assert isinstance(self._pipeline_model, SupportsVisionEncoding)
            vision_result = self._encoder_cache.run_vision_encode(
                self._pipeline_model,
                as_vision_context_batches(replica_batches),
                self._devices,
            )
            self._encoder_cache.finalize_vision_inputs(
                self._pipeline_model,
                model_inputs,
                self._devices,
                vision_result,
            )

        return (model_inputs, bitmask, flat_batch)

    @traced
    def _maybe_sort_loras(
        self, batch: list[TextGenerationContextType]
    ) -> list[TextGenerationContextType]:
        """Optionally sorts the batch by LoRA IDs.

        Requests that use the same LoRA are placed adjacent to each other.
        """
        if self._pipeline_model._lora_manager is None:
            return batch

        return self._pipeline_model._lora_manager.sort_lora_batch(batch)

    def _record_batch_info(self, contexts: Any) -> None:
        """Record per-step batch statistics for diagnostics.

        Args:
            contexts: Contexts in the step, providing ``start_idx`` and
                ``active_length``.

        Side Effects:
            Appends a ``BatchInfo`` entry to ``self.batch_infos``.
        """
        self.batch_infos.append(
            BatchInfo(
                past_seq_lens=[x.tokens.processed_length for x in contexts],
                seq_lens=[x.tokens.active_length for x in contexts],
            )
        )

    def __del__(self) -> None:
        """Flush recorded batch information to disk if configured.

        When ``MAX_BATCH_INFO_FILENAME`` is set, this writes a JSON file
        containing per-step batch statistics collected during execution.
        """
        if (
            hasattr(self, "batch_info_output_fname")
            and self.batch_info_output_fname is not None
        ):
            output = {
                "batch_data": [dataclasses.asdict(x) for x in self.batch_infos]
            }
            with open(self.batch_info_output_fname, "w") as f:
                json.dump(output, f, indent=2)
                f.flush()  # Refer to MAXSERV-893

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Processes the batch and returns decoded tokens.

        Executes the graph for a single decode step, samples the next token,
        then decodes and returns the generated tokens.
        """
        # Prepare the batch.
        model_inputs, bitmask, flat_batch = self.prepare_batch(inputs.batches)

        batch_processors: list[BatchLogitsProcessor] = []
        if len(flat_batch) > 0:
            # If structured output is present in the batch, use the sampler with bitmask.
            sampler: Model
            if bitmask is not None:
                assert self._sampler_with_bitmask is not None, (
                    "Sampler must be built with bitmask sampling"
                )
                sampler = self._sampler_with_bitmask
            else:
                assert self._sampler_without_bitmask is not None
                sampler = self._sampler_without_bitmask

            with Tracer("FusedSamplingProcessor"):
                sampling_processor = FusedSamplingProcessor(
                    sampler=sampler,
                    pipeline_config=self._pipeline_config,
                    context_batch=flat_batch,
                    device=self._sampler_device,
                    pinned_new_tokens=self._pinned_new_tokens,
                    identity_logit_offsets=self._identity_logit_offsets,
                    bitmask=bitmask,
                    vocab_size=self.vocab_size,
                )

            batch_processors.append(sampling_processor)

        curr_step_inputs = model_inputs
        batch_log_probabilities: list[list[LogProbabilities | None]] = []
        # Launch the forward pass.
        model_outputs = self._launch_forward_pass(
            curr_step_inputs, flat_batch, step=0
        )

        # Validate output. This is more of an internal check that the model
        # is implemented correctly.
        if (
            self._pipeline_config.sampling.enable_variable_logits
            and model_outputs.logit_offsets is None
        ):
            raise ValueError(
                "Model must return logit_offsets when enable_variable_logits is True."
            )

        # Execute the single step if the batch is not empty.
        if len(flat_batch) > 0:
            # Sample next token.
            with Tracer("sample_next_token"):
                sample_logits, sample_offsets = (
                    sampling_processor.logits_for_sampling(
                        logits=model_outputs.logits,
                        next_token_logits=model_outputs.next_token_logits,
                        logit_offsets=model_outputs.logit_offsets,
                    )
                )
                apply_logits_processors(
                    context_batch=flat_batch,
                    batch_logits=sample_logits,
                    batch_logit_offsets=sample_offsets,
                    batch_processors=batch_processors,
                )
                new_tokens = sampling_processor.new_tokens
                assert new_tokens is not None

            if inputs.enable_log_probs:
                with Tracer("compute_log_probabilities"):
                    try:
                        batch_log_probabilities.append(
                            self._pipeline_model.compute_log_probabilities(
                                self.session,
                                curr_step_inputs,
                                model_outputs,
                                new_tokens,
                                inputs.batch_top_log_probs,
                                inputs.batch_echo,
                            )
                        )
                    except NotImplementedError:
                        logger.warning(
                            "Unable to compute log probabilities for"
                            f" {self._pipeline_config.model.model_path}"
                        )
                        batch_log_probabilities.append(
                            [None for _ in flat_batch]
                        )

        # Return early if the batch is empty.
        if len(flat_batch) == 0:
            return {}

        # Do the copy to host for each token generated. The sampler output
        # lives on the sampler device (the model device, or the host CPU when
        # ``sample_on_host`` is set), so stage the D2H copy from there.
        sampler_device = self._sampler_device
        with Tracer("d2h_generated_tokens"):
            generated_tokens_device = sampling_processor.generated_tokens
            # Staging memory keeps the d2h transfer async on accelerators.
            # Note that we do not want to use `DevicePinnedBuffer` here.
            generated_tokens_host = Buffer(
                shape=generated_tokens_device.shape,
                dtype=generated_tokens_device.dtype,
                device=sampler_device,
                usage=Usage.STAGING,
            )
            generated_tokens_host.inplace_copy_from(generated_tokens_device)
            # Staging buffers do not synchronize on host reads, so the copy
            # above is still in flight here.
            sampler_device.synchronize()
            generated_tokens_np = generated_tokens_host.to_numpy()

        res = update_context_and_prepare_responses(
            generated_tokens_np,
            flat_batch,
            batch_log_probabilities=batch_log_probabilities,
            enable_log_probs=inputs.enable_log_probs,
        )

        # Update the cache lengths in our kv_cache manager.
        # This should be done after the contexts are updated.
        for ctx in inputs.flat_batch:
            self._kv_manager.step(ctx)

        return res

    def _launch_forward_pass(
        self,
        curr_step_inputs: ModelInputs,
        flat_batch: list[TextGenerationContextType],
        step: int,
    ) -> ModelOutputs:
        with Tracer(f"forward_pass_step_{step}"):
            try:
                model_outputs = self._pipeline_model.execute(
                    model_inputs=curr_step_inputs
                )
                return model_outputs
            except Exception:
                batch_size = len(flat_batch)
                cache_tokens = sum(
                    ctx.tokens.processed_length for ctx in flat_batch
                )
                input_tokens = sum(
                    ctx.tokens.active_length for ctx in flat_batch
                )
                logger.error(
                    "Encountered an exception while executing batch: "
                    f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}"
                )
                raise  # re-raise the original exception

    def release(self, request_id: RequestID) -> None:
        """Release model-specific resources for a completed request.

        Primary and extra KV cache lifecycle is managed by the batch
        constructor.  This method drops the request's vision-encoder-cache
        references and any model-specific state.
        """
        if self._encoder_cache is not None:
            self._encoder_cache.release_request(request_id)
        if hasattr(self._pipeline_model, "release"):
            self._pipeline_model.release(request_id)

    @property
    def kv_manager(self) -> PagedKVCacheManagerInterface:
        """Returns the KV cache manager for this pipeline."""
        return self._kv_manager

    def batch_vision_metrics(self) -> VisionEncoderMetrics | None:
        """Returns vision encoder metrics for the most recent batch.

        Returns ``None`` for text-only models and for batches that did no
        vision encoding (e.g. decode steps). The metrics come from the
        pipeline-owned :class:`VisionEncoderCache`, if this pipeline has one;
        otherwise, for a model that owns its encoder cache internally, from
        :class:`SupportsPooledVisionMetrics`.
        """
        if self._encoder_cache is not None:
            return self._encoder_cache.pop_metrics()
        if isinstance(self._pipeline_model, SupportsPooledVisionMetrics):
            return self._pipeline_model.pop_vision_metrics()
        return None

    def batch_video_metrics(self) -> VideoEncoderMetrics | None:
        """Returns video encoder metrics for the most recent batch.

        Returns ``None`` for models with no video support and for batches
        that did no video encoding. Video encoding has no pipeline-owned
        cache equivalent to :class:`VisionEncoderCache`, so this only ever
        comes from a model implementing :class:`SupportsPooledVisionMetrics`.
        """
        if isinstance(self._pipeline_model, SupportsPooledVisionMetrics):
            return self._pipeline_model.pop_video_metrics()
        return None
