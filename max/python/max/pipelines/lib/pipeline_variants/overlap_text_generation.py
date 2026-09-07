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
"""MAX pipeline for model inference and generation (Overlap Text Generation Variant).

This pipeline supports overlap scheduling where GPU execution is overlapped with
python host logic.

Note that this pipeline only supports num_steps=1.

Here is the CPU and GPU timeline for overlap scheduling:

   I3: Input processing for batch 3
   O3: Output processing for batch 3
   K3: GPU kernel execution for batch 3

    CPU: [I1][I2]          [O1][I3]      [O2][I4]      [O3][I5]      ...
    GPU:     [     K1     ][     K2     ][     K3     ][     K4     ][ ...

During I3, we have to prepare the model inputs for batch3. However, K2 may
still be in flight. If batch 2 and 3 share the same requests, then we rely on the
RealizeFutureTokenProcessor to prepare the ragged_input_tokens for batch 3 on
the GPU. This essentially scatters the generated tokens from the output of batch 2
on the slots corresponding to placeholder future tokens in batch 3's inputs.

For example:

  Batch 2 has reqA, reqB, reqC.
  Batch 3 has reqB, reqC, reqD.

    reqA = [I, dream, of, FUTURE_TOKEN]
    reqB = [I, like, to, go, FUTURE_TOKEN]
    reqC = [I, like, to, eat, FUTURE_TOKEN]
    reqD = [I, like, to, read]

  The ragged_input_tokens for Batch 3 would be:
                      idx=4                           idx=9
    [I, like, to, go, FUTURE_TOKEN, I, like, to, eat, FUTURE_TOKEN, I, like, to, read]

  RealizeFutureTokenProcessor would scatter the outputs of batch2 to the right slots:
    scatter_nd(
       inputs=ragged_input_tokens,
       indices=[[-99999], [4], [9]],
       updates=[sheep, fishing, cake]
    )

  Note that reqA is part of Batch 2 but not present in Batch 3. As such the update
  "sheep" corresponding to reqA is skipped since its idx=-99999 is out of bounds.
"""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    final,
    runtime_checkable,
)

import numpy as np
import numpy.typing as npt
from max.driver import (
    CPU,
    Buffer,
    Device,
    DeviceEvent,
    DevicePinnedBuffer,
    is_virtual_device_mode,
    load_devices,
)
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import (
    BufferType,
    DeviceRef,
    Dim,
    Graph,
    SymbolicDim,
    TensorType,
    TensorValue,
    ops,
)
from max.graph.weights import (
    WeightsAdapter,
    WeightsFormat,
    load_weights,
    weights_format,
)
from max.nn import kernels
from max.nn.kv_cache import (
    BatchCharacteristics,
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheInputsPerDevice,
    MultiKVCacheInputs,
    spec_decode_cache_slack,
)
from max.nn.transformer import ReturnLogits
from max.pipelines.context import (
    EOSTracker,
    SpecDecodingState,
    TextAndVisionContext,
    TextContext,
    TextGenerationContextType,
    TextGenerationOutput,
)
from max.pipelines.context.exceptions import (  # noqa: F401 (for docstring)
    InputError,
)
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.kv_cache import PagedKVCacheManagerInterface
from max.pipelines.kv_cache.paged_kv_cache.cache_manager import (
    _contiguous_prefix_2d,
    cache_valid_length_for_context,
    prompt_tokens_for_context,
)
from max.pipelines.lib.vision_encoder_cache import (
    SupportsPooledVisionMetrics,
    SupportsVisionEncoding,
    VisionEncoderCache,
    as_vision_context_batches,
)
from max.pipelines.modeling.types import (
    BatchType,
    CompletedBatchStats,
    PipelineOutputsDict,
    PipelineTokenizer,
    ReasoningPipelineTokenizer,
    RequestID,
    TextGenerationInputs,
    TextGenerationRequest,
)
from max.pipelines.speculative.config import (
    MAGIC_DRAFT_TOKEN_ID,
    SpeculativeConfig,
)
from max.pipelines.speculative.depth_schedule import build_depth_lookup
from max.pipelines.speculative.ragged_token_merger import _shape_to_scalar
from max.pipelines.speculative.utils import _SpeculativeDecodingMetrics
from max.profiler import Tracer, traced

from ..memory_estimation import MemoryPlan
from .structured_output_overlap import StructuredOutputOverlapState
from .text_generation import TextGenerationPipelineInterface, load_kv_manager
from .utils import (
    CommittedSpanSnapshot,
    StructuredOutputHelper,
    update_context_and_prepare_responses,
    update_spec_decode_context_and_prepare_responses,
)

if TYPE_CHECKING:
    from ..config import MAXModelConfig, PipelineConfig

from dataclasses import dataclass

from max.pipelines.sampling import (
    FusedSamplingProcessor,
    apply_logits_processors,
    token_sampler,
)

from ..graph_capture import ServeGraphCaptureRunner
from ..interfaces import (
    ModelInputs,
    ModelOutputs,
    PipelineModel,
    PipelineModelWithKVCache,
    UnifiedEagleOutputs,
)
from ..synthesis_bucket_aligner import SynthesisBucketAligner
from ..utils import CompilationTimer
from ..vision_encoder_cache import VideoEncoderMetrics, VisionEncoderMetrics

logger = logging.getLogger("max.pipelines")

_MAX_GRAPH_CAPTURE_BATCH_SIZE = 128
_OOB_IDX = np.iinfo(np.int32).min


def _verify_width_lookup(
    spec_config: SpeculativeConfig | None,
    num_speculative_tokens: int,
    max_batch_size: int,
) -> list[int] | None:
    """Dense ``batch_size -> drafts to verify``; ``None`` when unscheduled.

    A step always drafts ``num_speculative_tokens`` proposals. The schedule
    decides how many of them the target verifies.

    Index 0 is unused so a runtime batch size indexes the list directly.

    Args:
        spec_config: The pipeline's speculative config, which carries the
            schedule.
        num_speculative_tokens: The configured draft depth, which caps every
            scheduled count. For block drafters this is the checkpoint's
            fixed block width; the schedule narrows only how much of that
            block the target verifies, not how much the draft produces.
        max_batch_size: Largest decode batch size the lookup must cover.

    Returns:
        The dense lookup, or ``None`` when no schedule applies.
    """
    if spec_config is None or num_speculative_tokens <= 0:
        return None
    schedule = spec_config.verify_width_schedule
    if schedule is None:
        return None
    return build_depth_lookup(
        schedule,
        max_batch_size=max(1, max_batch_size),
        max_depth=num_speculative_tokens,
    )


def _reachable_verify_widths(
    spec_config: SpeculativeConfig | None,
    num_speculative_tokens: int,
    max_batch_size: int,
) -> list[int]:
    """Every number of carried drafts a step could verify."""
    lookup = _verify_width_lookup(
        spec_config, num_speculative_tokens, max_batch_size
    )
    if lookup is None:
        return [num_speculative_tokens]
    return sorted(set(lookup[1:]))


def _contiguous_prefix_3d(
    buffer: Buffer, rows: int, cols: int, depth: int
) -> Buffer:
    """Returns a contiguous 3D prefix view of ``buffer``, aliasing it."""
    num_elements = rows * cols * depth
    if num_elements > buffer.num_elements:
        raise ValueError(
            "Requested contiguous prefix exceeds backing buffer capacity: "
            f"{num_elements} > {buffer.num_elements}."
        )
    flat = buffer.view(buffer.dtype, (buffer.num_elements,))
    return flat[:num_elements].view(buffer.dtype, (rows, cols, depth))


@runtime_checkable
class _UnifiedSpecDecodeInputs(Protocol):
    tokens: Buffer
    input_row_offsets: Buffer
    kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer]

    draft_tokens: Buffer | None
    draft_probs_full: Buffer | None

    seed: Buffer | None

    temperature: Buffer | None
    top_k: Buffer | None
    max_k: Buffer | None
    top_p: Buffer | None
    min_top_p: Buffer | None
    in_thinking_phase: Buffer | None


@dataclass
class _SpecDecodeSamplingBuffers:
    """Per-batch sampling parameter buffers for stochastic target acceptance.

    Per-batch tensors (``temperature``, ``top_k``, ``top_p``, ``seed``) are
    prefix views of persistent device buffers, providing stable base
    pointers required by graph capture. The 0-d scalars (``max_k``,
    ``min_top_p``) live on the host and are fresh each execute.

    """

    temperature: Buffer
    top_k: Buffer
    max_k: Buffer
    top_p: Buffer
    min_top_p: Buffer
    in_thinking_phase: Buffer
    seed: Buffer


def _host_mirror_realized_drafts(
    draft_tokens_np: npt.NDArray[np.int64],
    prev_to_curr_map: npt.NDArray[np.int64],
    prev_next_draft_tokens: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Reconstruct the post-realize device draft buffer on the host.

    The device draft buffer for an EAGLE verify step is built as the
    H2D copy of ``draft_tokens_np`` (its pre-scatter state), then
    overwritten for rows present in the previous batch by the realize scatter
    (``RealizeFutureTokenProcessor``) using ``prev_to_curr_map``. This mirrors
    that mapping exactly so the constrained-decoding synchronous-fill builds the bitmask from
    the drafts the GPU actually verifies.
    Note that:

      1. An unmapped current row keeps its ``draft_tokens_np`` value -- the MAGIC
        placeholder, or a preempted/resumed request's real saved drafts.
      2. A mapped current row takes the previous batch's ``next_draft_tokens``.

    Using ``prev_to_curr_map`` (the same scatter map the device graph uses)
    keeps this a line-for-line mirror of the device scatter.

    The previous batch always drafted the configured depth, but this step may
    verify fewer, so the source row is trimmed to the verify width exactly as
    the device graph trims ``prev_generated_draft_tokens``. Dropping the tail
    here rather than raising is what keeps the two mirrors in step.

    Args:
        draft_tokens_np: ``[curr_batch, verify_width]`` pre-scatter device state
            (the host array copied to the device draft buffer before the
            realize scatter).
        prev_to_curr_map: ``[prev_batch]`` mapping prev-batch row -> current
            row, with ``_OOB_IDX`` for prev rows absent from the current batch.
        prev_next_draft_tokens: ``[prev_batch, K]`` previous batch's next
            drafts, at the configured depth.

    Returns:
        ``[curr_batch, verify_width]`` host mirror of the device draft buffer.
    """
    realized = draft_tokens_np.copy()
    curr_batch_size, verify_width = realized.shape
    for prev_i, curr_i in enumerate(prev_to_curr_map):
        if 0 <= curr_i < curr_batch_size:
            realized[curr_i] = prev_next_draft_tokens[prev_i, :verify_width]
    return realized


def _resolve_thinking_token_ids(
    tokenizer: ReasoningPipelineTokenizer[Any, Any, Any],
) -> tuple[int, int]:
    """Resolve reasoning-delimiter token ids from a reasoning-aware tokenizer.

    Architecture-specific tokenizers that drive a reasoning parser declare
    their delimiter ids by implementing
    :class:`~max.pipelines.modeling.types.ReasoningPipelineTokenizer` (Gemma 4's
    ``<|channel>``/``<channel|>``, Kimi K2.5's and MiniMax M2's
    ``<think>``/``</think>``, etc.) and resolving the ids once at
    construction.
    """
    return (
        tokenizer.reasoning_start_token_id,
        tokenizer.reasoning_end_token_id,
    )


@dataclass
class SpecDecodeState:
    """Pipeline for unified EAGLE: single fused graph handles target + draft.

    Unlike EAGLESpeculativeDecodingPipeline which manages two separate models,
    this pipeline uses a single model that runs both target forward and draft
    generation in one compiled graph call. Rejection sampling also happens
    in-graph (greedy acceptance).

    Orchestration:
    Prefill: model(draft_tokens=[?,0]) -> commit bonus, save new_token.
    Decode:  model(draft_tokens=[?,K]) -> verify drafts, commit tokens,
            save new_token for next iteration.
    """

    num_speculative_tokens: int
    """The number of speculative tokens to generate."""

    kv_manager: PagedKVCacheManagerInterface
    """The KVCache manager for model."""

    persistent_draft_tokens: Buffer
    """Persistent input buffer for draft tokens.

    A stable buffer must be used for inputs to device graphs."""

    persistent_temperature: Buffer
    """Persistent per-batch temperature for stochastic target acceptance."""

    persistent_top_k: Buffer
    """Persistent per-batch top_k values."""

    persistent_top_p: Buffer
    """Persistent per-batch top_p values."""

    persistent_in_thinking_phase: Buffer
    """Persistent per-batch boolean for in-thinking-phase state."""

    persistent_seed: Buffer
    """Persistent ``[total_max_batch]`` uint64 seed values, one per request,
    derived from ``sampling_params.seed + len(tokens)``."""

    batch_metrics: _SpeculativeDecodingMetrics | None = None
    """Per-batch metrics for the most recently completed batch."""

    persistent_bonus_tokens_pinned: DevicePinnedBuffer | None = None
    """Pinned memory for async callback: bonus tokens (next_tokens) per request.

    Shape: [total_max_batch]. DType int64. Reused across iterations; the
    callback captures a DLPack view that is valid when the callback body runs
    (D2H completes before the callback fires on the same CUDA stream).
    None when structured output is disabled globally.
    """

    persistent_num_accepted_pinned: DevicePinnedBuffer | None = None
    """Pinned memory for async callback: accepted draft token counts.

    Shape: [total_max_batch]. DType int64 (from ops.argmax).
    None when structured output is disabled globally.
    """

    persistent_next_draft_tokens_pinned: DevicePinnedBuffer | None = None
    """Pinned memory for async callback: next-batch draft tokens.

    Shape: [total_max_batch, num_speculative_tokens]. DType int64.
    None when structured output is disabled globally.
    """

    accepted_token_pinned: DevicePinnedBuffer | None = None
    """Pinned mirror of the accepted draft tokens.

    The GPU acceptance sampler writes the verified accepted tokens back into
    the draft_tokens device buffer after the forward pass. These are D2H'd
    into this persistent buffer (on the same CUDA stream, before the callback
    is enqueued) so the callback can capture a DLPack view that is valid when
    it fires. This avoids the race where a per-batch .copy() on the main
    thread captures stale MAGIC tokens before the D2H completes.

    Shape: [total_max_batch, num_speculative_tokens]. DType int64. Sized for
    the deepest draft a step can carry; a step verifying fewer takes a
    contiguous prefix view, so one allocation serves every verify width.
    None when structured output is disabled globally.
    """

    has_precomputed_bitmask: bool = False
    """True when a CUDA host callback has computed a bitmask for the next batch."""

    overlap_state: StructuredOutputOverlapState | None = None
    """Flag + pinned/device bitmask buffers for constrained-decoding
    overlap. ``None`` when structured output is globally off (no vocab
    size). Owned for the lifetime of :class:`SpecDecodeState`.
    """

    draft_probs_full_zero_row: Buffer | None = None
    """Device-resident zeros of shape ``[K, vocab_size]``, the source of the
    stale-row clears in ``_prepare_draft_probs_full``. Allocated once so the
    clear stays a device-to-device copy."""

    persistent_draft_probs_full: Buffer | None = None
    """Persistent input buffer for the draft's whole proposal distribution, ``[max_batch, K, vocab_size]``, paired with
    :attr:`persistent_draft_tokens`. ``None`` unless
    ``draft_proposal == "sampled"``. Device-only:
    unlike the scalar probabilities it is never mirrored to the host, since
    only the acceptance graph reads it."""

    @classmethod
    def load(
        cls,
        session: InferenceSession,
        model: PipelineModelWithKVCache[Any],
        pipeline_config: PipelineConfig,
        max_batch_size: int,
        available_cache_memory: int | None = None,
        vocab_size: int | None = None,
    ) -> SpecDecodeState:
        """Load the spec decode state.

        Note: if vocab_size is None then structured output bitmask is not allocated.
        """
        if pipeline_config.speculative is None:
            raise ValueError(
                "Speculative decoding is not enabled in the pipeline config."
            )

        # In compile-only mode (virtual device mode), put the persistent buffers
        # below on the host: they are runtime-only spec-decode state, and
        # VirtualDevice does not support memory allocation.
        buffer_device: Device = (
            CPU() if is_virtual_device_mode() else model.devices[0]
        )

        kv_manager = load_kv_manager(
            params=model.kv_params,
            max_batch_size=max_batch_size,
            max_seq_len=model.max_seq_len,
            session=session,
            available_cache_memory=available_cache_memory,
            is_di_enabled=pipeline_config.runtime.is_disaggregated,
            model_name=pipeline_config.model.model_name,
        )

        assert pipeline_config.speculative is not None
        num_speculative_tokens = (
            pipeline_config.speculative.num_speculative_tokens
        )
        assert num_speculative_tokens is not None
        total_max_batch = (
            max_batch_size * pipeline_config.model.data_parallel_degree
        )
        verify_widths = _reachable_verify_widths(
            pipeline_config.speculative,
            num_speculative_tokens,
            max_batch_size,
        )

        persistent_draft_tokens = Buffer(
            dtype=DType.int64,
            shape=(total_max_batch, num_speculative_tokens),
            device=buffer_device,
        )
        persistent_temperature = Buffer(
            dtype=DType.float32,
            shape=(total_max_batch,),
            device=buffer_device,
        )
        persistent_top_k = Buffer(
            dtype=DType.int64,
            shape=(total_max_batch,),
            device=buffer_device,
        )
        persistent_top_p = Buffer(
            dtype=DType.float32,
            shape=(total_max_batch,),
            device=buffer_device,
        )
        persistent_seed = Buffer(
            dtype=DType.uint64,
            shape=(total_max_batch,),
            device=buffer_device,
        )
        persistent_draft_probs_full: Buffer | None = None
        draft_probs_full_zero_row: Buffer | None = None
        if pipeline_config.speculative.draft_proposal == "sampled":
            draft_vocab_size = getattr(
                model.huggingface_config, "vocab_size", None
            )
            if draft_vocab_size is not None:
                persistent_draft_probs_full = Buffer(
                    dtype=DType.float32,
                    shape=(
                        total_max_batch,
                        num_speculative_tokens,
                        draft_vocab_size,
                    ),
                    device=buffer_device,
                )
                zero_shape = (num_speculative_tokens, draft_vocab_size)
                draft_probs_full_zero_row = Buffer.from_numpy(
                    np.zeros(zero_shape, dtype=np.float32)
                ).to(buffer_device)

        # The packed-int32 bitmask the async FSM callback fills lives in
        # :class:`StructuredOutputOverlapState`'s ``pinned_bitmask`` (allocated
        # below). The callback writes the packed bitmask there directly and the
        # GPU acceptance sampler unpacks and applies it, so no separate staging
        # buffer is needed.
        persistent_bonus_tokens_pinned: DevicePinnedBuffer | None = None
        persistent_num_accepted_pinned: DevicePinnedBuffer | None = None
        persistent_next_draft_tokens_pinned: DevicePinnedBuffer | None = None
        accepted_token_pinned: DevicePinnedBuffer | None = None
        if vocab_size is not None and not is_virtual_device_mode():
            persistent_bonus_tokens_pinned = DevicePinnedBuffer(
                dtype=DType.int64,
                shape=(total_max_batch,),
                device=model.devices[0],
            )
            persistent_num_accepted_pinned = DevicePinnedBuffer(
                dtype=DType.int64,
                shape=(total_max_batch,),
                device=model.devices[0],
            )
            persistent_next_draft_tokens_pinned = DevicePinnedBuffer(
                dtype=DType.int64,
                shape=(total_max_batch, num_speculative_tokens),
                device=model.devices[0],
            )
            accepted_token_pinned = DevicePinnedBuffer(
                dtype=DType.int64,
                shape=(total_max_batch, num_speculative_tokens),
                device=model.devices[0],
            )
        persistent_in_thinking_phase = Buffer(
            dtype=DType.bool,
            shape=(total_max_batch,),
            device=buffer_device,
        )

        overlap_state: StructuredOutputOverlapState | None = None
        if vocab_size is not None and not is_virtual_device_mode():
            # A step binds ``verified drafts + 1`` bitmask positions. Width 1
            # is always included for the prefill->decode boundary, where
            # nothing was drafted yet.
            overlap_state = StructuredOutputOverlapState(
                device=model.devices[0],
                cpu=CPU(),
                max_batch_size=total_max_batch,
                num_positions=sorted({1} | {w + 1 for w in verify_widths}),
                vocab_size=vocab_size,
            )

        return SpecDecodeState(
            num_speculative_tokens=num_speculative_tokens,
            kv_manager=kv_manager,
            persistent_draft_tokens=persistent_draft_tokens,
            persistent_temperature=persistent_temperature,
            persistent_top_k=persistent_top_k,
            persistent_top_p=persistent_top_p,
            persistent_bonus_tokens_pinned=persistent_bonus_tokens_pinned,
            persistent_num_accepted_pinned=persistent_num_accepted_pinned,
            persistent_next_draft_tokens_pinned=persistent_next_draft_tokens_pinned,
            accepted_token_pinned=accepted_token_pinned,
            persistent_in_thinking_phase=persistent_in_thinking_phase,
            persistent_seed=persistent_seed,
            overlap_state=overlap_state,
            persistent_draft_probs_full=persistent_draft_probs_full,
            draft_probs_full_zero_row=draft_probs_full_zero_row,
        )


@runtime_checkable
class _HasRaggedTokens(Protocol):
    tokens: Buffer
    input_row_offsets: Buffer


@runtime_checkable
class _SupportsModelCapture(Protocol):
    model: Model


@runtime_checkable
class SupportsSSMStateWarmup(Protocol):
    """Protocol for pipeline models with SSM/conv state pools.

    Models (e.g. Nemotron-H, Qwen3.5, LFM2) that allocate per-request state
    pool slots outside the KV cache must implement
    :meth:`release_warmup_state` so that graph-capture warmup can release
    those slots after each ``(batch_size, cache_length)`` probe.  Without it,
    warmup would exhaust the state pool before reaching steady serving.

    The overlap pipeline's ``_warmup_model_inputs`` context manager calls
    ``release_warmup_state`` after each probe's capture completes, with the
    same request IDs that were passed to ``prepare_initial_token_inputs``
    during that probe.
    """

    def release_warmup_state(self, request_ids: list[RequestID]) -> None:
        """Release state pool slots claimed for the given warmup request IDs.

        Called once per ``_warmup_model_inputs`` probe, after graph capture
        for that probe completes.  ``request_ids`` matches the list of request
        IDs from the warmup batch constructed inside ``_warmup_model_inputs``.
        Implementations should release slots without zeroing the underlying
        pool memory (zeroing happens at claim time, not release time).
        """
        ...


@dataclass
class _AsyncBatchOutput:
    output_dict: PipelineOutputsDict[TextGenerationOutput]
    spec_decode_metrics: _SpeculativeDecodingMetrics | None = None


@dataclass
class AsyncSpecDecodeBatch:
    """Extra outputs specific for speculative decoding async batch."""

    draft_tokens_to_verify_device: Buffer
    """The draft tokens to verify for the batch on gpu.

    The shape of the buffer is (batch_size, num_draft_tokens_to_verify).
    """

    draft_tokens_to_verify_host: Buffer
    """The draft tokens to verify for the batch.

    The shape of the array is (batch_size, num_draft_tokens_to_verify).
    """

    next_draft_tokens_device: Buffer
    """The next draft tokens for the batch on gpu.

    The shape of the buffer is (batch_size, num_speculative_tokens).
    """

    next_draft_tokens_host: Buffer
    """The next draft tokens for the batch on pinned gpu memory.

    The shape of the buffer is (batch_size, num_speculative_tokens).
    """

    num_accepted_draft_tokens_device: Buffer
    """The number of accepted draft tokens for the batch on gpu.

    The shape of the buffer is (batch_size,).
    """

    num_accepted_draft_tokens_host: Buffer
    """The number of accepted draft tokens for the batch on pinned gpu memory.

    The shape of the buffer is (batch_size,).
    """

    max_seq_len: int
    """The maximum sequence length for the pipeline model."""

    fsm_advanced_by_callback: bool = False
    """True when a CUDA host callback already advanced the FSM for this batch."""

    next_draft_probs_full_device: Buffer | None = None
    """The distribution each next-step draft token was sampled from, retained
    on device for the realize-future-tokens scatter. ``None`` unless
    ``draft_proposal="sampled"``. It has no host mirror: only the acceptance
    graph reads it, so it never leaves the GPU.

    The shape of the buffer is
    (batch_size, num_speculative_tokens, vocab_size).
    """

    @property
    def num_draft_tokens_to_verify(self) -> int:
        """The number of draft tokens to verify during this batch."""
        return self.draft_tokens_to_verify_host.shape[1]


@dataclass
class AsyncBatch(Generic[TextGenerationContextType]):
    """A batch that is being asynchronously executed on the GPU."""

    inputs: TextGenerationInputs[TextGenerationContextType]
    """The inputs for the batch.
    """

    generated_tokens_device: Buffer
    """The generated tokens for the batch on the gpu.

    The shape of the buffer is (batch_size,). The ordering of the generated tokens
    should be the same as the ordering of the requests in the input batch.
    """

    generated_tokens_host: Buffer
    """The generated tokens for the batch on the cpu.

    It is backed by pinned memory which makes d2h transfers asynchronous.
    This buffer is not ready to read until the batch has completed executing.

    This buffers has the same contents as `generated_tokens_device`.
    """

    copy_event: DeviceEvent
    """Event that tracks completion of the d2h copy."""

    enqueue_monotonic: float = 0.0
    """``time.monotonic()`` timestamp taken when this batch's ``execute()``
    call began, i.e. an upper bound on when its GPU work could have started.
    Used together with the previous batch's sync timestamp to estimate this
    batch's execution time when its outputs are synchronized."""

    _is_processed: bool = False
    """Whether the outputs have been already been processed."""

    spec_decode: AsyncSpecDecodeBatch | None = None
    """Extra outputs specific for speculative decoding async batch."""

    overwrite_future: bool = True
    """Whether to overwrite future tokens when processing outputs.

    For normal overlap scheduling, this is True since we use placeholder future
    tokens. For structured output, this is False since matchers track exact token
    sequences and cannot use placeholder tokens.
    """

    structured_output: StructuredOutputHelper | None = None
    """Helper for structured output operations (filling bitmasks)."""

    think_start_token_id: int | None = None
    think_end_token_id: int | None = None
    """``None`` disables ``in_reasoning_phase`` tracking on commit."""

    @traced
    def sync_and_process_outputs(
        self,
        curr_flat_batch: list[TextGenerationContextType] | None = None,
        bitmask: npt.NDArray[np.int32] | None = None,
        sampling_processor: FusedSamplingProcessor | None = None,
    ) -> _AsyncBatchOutput:
        """Syncs on completion of this batch and processes the outputs.

        Replaces the placeholder future tokens in the TextContext CPU numpy
        token buffers with the real token values. When structured output is
        enabled, advances FSM state and updates the bitmask for continuing
        requests.

        Args:
            curr_flat_batch: The current batch's flat_batch, used to map
                request IDs to bitmask indices. Required when bitmask is provided.
            bitmask: The bitmask to update for structured output. If None,
                structured output updates are skipped.
            sampling_processor: The sampling processor to update with the new
                bitmask. Required when bitmask is provided.
        """
        if self._is_processed:
            raise ValueError("Outputs have already been processed.")
        self._is_processed = True

        # Synchronize on the copy event to ensure the async d2h transfer is done.
        self.copy_event.synchronize()
        generated_tokens_np = self.generated_tokens_host.to_numpy()

        # Now that we have synced, it is safe to read the contents of the
        # generated_tokens_np on the host.

        # Advance FSM state for previous batch contexts with their realized tokens.
        # Note: Spec decode handles FSM advancement in update_spec_decode_context_and_prepare_responses.
        if (
            self.spec_decode is None
            and self.structured_output is not None
            and self.structured_output.enabled
        ):
            for idx, ctx in enumerate(self.inputs.flat_batch):
                # Gate on generated_length, not actively_chunked: that flag gets
                # mutated by the current-batch rebuild before this sync runs, so it
                # can advance the FSM on an intermediate chunk's artifact token and
                # drop the grammar's opening `{` (structured-output runaway).
                if ctx.matcher is not None and ctx.tokens.generated_length:
                    token = int(generated_tokens_np[idx])
                    # advance_fsm handles enforcement state internally
                    ctx.advance_fsm(token)

        # Update bitmask for requests continuing from previous batch to current batch.
        if bitmask is not None and curr_flat_batch is not None:
            assert sampling_processor is not None, (
                "sampling_processor required when bitmask is provided"
            )
            # Build mapping from request_id to current batch index
            curr_request_to_idx = {
                ctx.request_id: idx for idx, ctx in enumerate(curr_flat_batch)
            }

            with Tracer("fill_bitmask_for_continuing"):
                for ctx in self.inputs.flat_batch:
                    # Only update bitmask for requests that:
                    # 1. Have a matcher (structured output enabled)
                    # 2. Are continuing in the current batch
                    if (
                        ctx.matcher is not None
                        and ctx.request_id in curr_request_to_idx
                    ):
                        curr_idx = curr_request_to_idx[ctx.request_id]
                        assert self.structured_output is not None
                        self.structured_output.fill_bitmask(
                            ctx, bitmask, curr_idx
                        )

            with Tracer("update_bitmask_h2d"):
                # H2D transfer on default stream - will complete before sampling
                sampling_processor.update_bitmask(bitmask)

        if self.spec_decode is None:
            # Update the context object, realizing the placeholder future tokens.
            # Overlap scheduler only supports num_steps=1.
            batch_size = len(self.inputs.flat_batch)
            assert generated_tokens_np.shape == (batch_size,)
            outputs = update_context_and_prepare_responses(
                generated_tokens_np.reshape((batch_size, 1)),
                self.inputs.flat_batch,
                overwrite_future=self.overwrite_future,
            )
            wrapped_outputs = _AsyncBatchOutput(output_dict=outputs)
        else:
            spec_decode_batch = self.spec_decode
            draft_tokens_np = (
                spec_decode_batch.draft_tokens_to_verify_host.to_numpy()
            )
            num_accepted_draft_tokens = (
                spec_decode_batch.num_accepted_draft_tokens_host.to_numpy()
            )
            next_draft_tokens = (
                self.spec_decode.next_draft_tokens_host.to_numpy()
            )
            max_seq_len = spec_decode_batch.max_seq_len
            batch_size = len(self.inputs.flat_batch)
            is_dummy_draft_tokens: list[bool] = [
                all(draft_tokens_np[i, :] == MAGIC_DRAFT_TOKEN_ID)
                for i in range(batch_size)
            ]
            outputs = update_spec_decode_context_and_prepare_responses(
                draft_tokens=draft_tokens_np,
                next_draft_tokens=next_draft_tokens,
                num_accepted_draft_tokens=num_accepted_draft_tokens,
                next_tokens=generated_tokens_np,
                context_batch=self.inputs.flat_batch,
                max_seq_len=max_seq_len,
                think_start_token_id=self.think_start_token_id,
                think_end_token_id=self.think_end_token_id,
                skip_fsm_advance=spec_decode_batch.fsm_advanced_by_callback,
            )

            num_draft_tokens_to_verify = draft_tokens_np.shape[1]

            # Compute per-position acceptance counts.
            # For each position i, count how many requests accepted at least i+1 tokens.
            accepted_per_position = [0] * num_draft_tokens_to_verify
            for dummy_batch, accepted_count in zip(
                is_dummy_draft_tokens,
                num_accepted_draft_tokens,
                strict=True,
            ):
                if dummy_batch:
                    continue
                for pos in range(int(accepted_count)):
                    if pos < num_draft_tokens_to_verify:
                        accepted_per_position[pos] += 1

            # Count verifications where there are real draft tokens to verify.
            # Otherwise we'd dilute the per-position acceptance rate.
            num_verifications = 0
            if num_draft_tokens_to_verify > 0:
                num_verifications = batch_size - sum(is_dummy_draft_tokens)

            metrics = _SpeculativeDecodingMetrics(
                num_speculative_tokens=num_draft_tokens_to_verify,
                accepted_per_position=accepted_per_position,
                num_verifications=num_verifications,
            )

            wrapped_outputs = _AsyncBatchOutput(
                output_dict=outputs,
                spec_decode_metrics=metrics,
            )

        return wrapped_outputs


_Tensor = TypeVar("_Tensor")
_Buffer = TypeVar("_Buffer")


@dataclass
class _RealizeFutureTokenSpecDecodeInputs(Generic[_Tensor, _Buffer]):
    curr_draft_tokens: _Tensor
    data_parallel_splits: _Tensor | None
    curr_cache_lengths: Sequence[_Tensor]
    signal_buffers: Sequence[_Buffer] | None
    prev_generated_draft_tokens: _Tensor
    prev_draft_tokens: _Tensor
    prev_num_accepted_draft_tokens: _Tensor
    curr_draft_probs_full: _Tensor | None = None
    """Draft proposal distributions paired with ``curr_draft_tokens``. Set
    iff ``draft_proposal="sampled"``."""
    prev_generated_draft_probs_full: _Tensor | None = None
    """Previous batch's ``next_draft_probs_full`` device tensor, paired with
    ``prev_generated_draft_tokens``."""

    def flatten(self) -> list[_Tensor | _Buffer]:
        return [
            self.curr_draft_tokens,
            *(
                (self.data_parallel_splits,)
                if self.data_parallel_splits is not None
                else ()
            ),
            *self.curr_cache_lengths,
            *(self.signal_buffers if self.signal_buffers is not None else ()),
            self.prev_generated_draft_tokens,
            self.prev_draft_tokens,
            self.prev_num_accepted_draft_tokens,
            *(
                (self.curr_draft_probs_full,)
                if self.curr_draft_probs_full is not None
                else ()
            ),
            *(
                (self.prev_generated_draft_probs_full,)
                if self.prev_generated_draft_probs_full is not None
                else ()
            ),
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> _RealizeFutureTokenSpecDecodeInputs[Any, Any]:
        return _RealizeFutureTokenSpecDecodeInputs(
            curr_draft_tokens=next(it),
            data_parallel_splits=next(it)
            if self.data_parallel_splits is not None
            else None,
            curr_cache_lengths=[
                next(it) for _ in range(len(self.curr_cache_lengths))
            ],
            signal_buffers=[next(it) for _ in range(len(self.signal_buffers))]
            if self.signal_buffers is not None
            else None,
            prev_generated_draft_tokens=next(it),
            prev_draft_tokens=next(it),
            prev_num_accepted_draft_tokens=next(it),
            curr_draft_probs_full=next(it)
            if self.curr_draft_probs_full is not None
            else None,
            prev_generated_draft_probs_full=next(it)
            if self.prev_generated_draft_probs_full is not None
            else None,
        )


@dataclass
class _RealizeFutureTokenInputs(Generic[_Tensor, _Buffer]):
    prev_to_curr_map: _Tensor
    curr_to_prev_map: _Tensor
    curr_tokens: _Tensor
    curr_input_row_offsets: _Tensor
    prev_generated_tokens: _Tensor

    spec_decode: (
        _RealizeFutureTokenSpecDecodeInputs[_Tensor, _Buffer] | None
    ) = None

    def flatten(self) -> list[_Tensor | _Buffer]:
        return [
            self.prev_to_curr_map,
            self.curr_to_prev_map,
            self.curr_tokens,
            self.curr_input_row_offsets,
            self.prev_generated_tokens,
            *(
                self.spec_decode.flatten()
                if self.spec_decode is not None
                else ()
            ),
        ]

    def unflatten(
        self, it: Iterator[Any]
    ) -> _RealizeFutureTokenInputs[Any, Any]:
        return _RealizeFutureTokenInputs(
            prev_to_curr_map=next(it),
            curr_to_prev_map=next(it),
            curr_tokens=next(it),
            curr_input_row_offsets=next(it),
            prev_generated_tokens=next(it),
            spec_decode=self.spec_decode.unflatten(it)
            if self.spec_decode is not None
            else None,
        )


def build_realize_future_token_graph(
    *,
    devices: Sequence[DeviceRef],
    enable_dp: int,
    num_speculative_tokens: int,
    data_parallel_degree: int = 1,
    sampled_draft_vocab_size: int | None = None,
) -> Graph:
    """Builds a graph that prepares the input for the next batch.

    Args:
        devices: All devices spanned by the model, in replica-major order
            (``[r0_tp0, r0_tp1, ..., r1_tp0, ...]``).
        enable_dp: Whether data parallelism is enabled (``dp_degree > 1``).
        num_speculative_tokens: Number of speculative tokens per step.
        data_parallel_degree: Number of data-parallel replicas. Devices are
            replica-major, so ``tp_degree = len(devices) // data_parallel_degree``
            consecutive devices form one DP replica and share the same request
            slice; this maps device ``i`` to replica ``i // tp_degree`` when
            indexing ``data_parallel_splits`` (which is indexed by replica, not
            device). Defaults to ``1`` (pure DP), which reproduces the
            historical device-index-equals-replica behavior.
        sampled_draft_vocab_size: When set (the
            ``draft_proposal="sampled"`` path), also realize the fp32
            ``[batch, K, vocab_size]`` ``draft_probs_full`` paired with
            ``draft_tokens``, on the same slot indices so a distribution
            cannot drift away from the token it proposed.
    """
    device0 = devices[0]
    if num_speculative_tokens > 0:
        spec_decode_input_types = _RealizeFutureTokenSpecDecodeInputs[
            TensorType, BufferType
        ](
            curr_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("curr_batch_size"),
                    SymbolicDim("num_draft_tokens"),
                ],
                device=device0,
            ),
            data_parallel_splits=TensorType(
                DType.int64,
                shape=[SymbolicDim("num_replicas_plus_one")],
                device=DeviceRef.CPU(),
            )
            if enable_dp
            else None,
            curr_cache_lengths=[
                TensorType(
                    DType.uint32,
                    shape=[SymbolicDim(f"curr_batch_size_gpu_{i}")],
                    device=device,
                )
                for i, device in enumerate(devices)
            ],
            signal_buffers=[
                BufferType(
                    DType.uint8,
                    shape=[SymbolicDim(f"signal_buffer_gpu_{i}")],
                    device=device,
                )
                for i, device in enumerate(devices)
            ]
            if len(devices) > 1
            else None,
            prev_generated_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("prev_batch_size"),
                    SymbolicDim("gen_num_draft_tokens"),
                ],
                device=device0,
            ),
            prev_draft_tokens=TensorType(
                DType.int64,
                shape=[
                    SymbolicDim("prev_batch_size"),
                    SymbolicDim("prev_num_draft_tokens"),
                ],
                device=device0,
            ),
            prev_num_accepted_draft_tokens=TensorType(
                DType.int64,
                shape=[SymbolicDim("prev_batch_size")],
                device=device0,
            ),
            curr_draft_probs_full=TensorType(
                DType.float32,
                shape=[
                    SymbolicDim("curr_batch_size"),
                    SymbolicDim("num_draft_tokens"),
                    sampled_draft_vocab_size,
                ],
                device=device0,
            )
            if sampled_draft_vocab_size is not None
            else None,
            prev_generated_draft_probs_full=TensorType(
                DType.float32,
                shape=[
                    SymbolicDim("prev_batch_size"),
                    SymbolicDim("gen_num_draft_tokens"),
                    sampled_draft_vocab_size,
                ],
                device=device0,
            )
            if sampled_draft_vocab_size is not None
            else None,
        )
    else:
        spec_decode_input_types = None
    input_types = _RealizeFutureTokenInputs[TensorType, BufferType](
        prev_to_curr_map=TensorType(
            DType.int64,
            shape=[SymbolicDim("prev_batch_size")],
            device=device0,
        ),
        curr_to_prev_map=TensorType(
            DType.int64,
            shape=[SymbolicDim("curr_batch_size")],
            device=device0,
        ),
        curr_tokens=TensorType(
            DType.int64,
            shape=[SymbolicDim("seq_len")],
            device=device0,
        ),
        curr_input_row_offsets=TensorType(
            DType.uint32,
            shape=[SymbolicDim("curr_batch_size_plus_one")],
            device=device0,
        ),
        prev_generated_tokens=TensorType(
            DType.int64,
            shape=[SymbolicDim("prev_batch_size")],
            device=device0,
        ),
        spec_decode=spec_decode_input_types,
    )
    with Graph(
        "realize_future_token_graph",
        input_types=input_types.flatten(),
    ) as graph:
        it = iter(graph.inputs)
        input_values = input_types.unflatten(it)

        curr_to_prev_map = ops.unsqueeze(input_values.curr_to_prev_map, axis=-1)
        prev_to_curr_map = ops.unsqueeze(input_values.prev_to_curr_map, axis=-1)

        curr_input_row_offsets = ops.rebind(
            input_values.curr_input_row_offsets, [Dim("curr_batch_size") + 1]
        )
        possible_future_token_indices = ops.rebind(
            curr_input_row_offsets[1:] - 1, ["curr_batch_size"]
        )

        oob_idx = ops.constant(_OOB_IDX, dtype=DType.int64, device=device0)

        # scatter the prev generated tokens into the curr tokens
        prev_to_curr_token_indices = oob_idx.broadcast_to(("prev_batch_size",))
        prev_to_curr_token_indices = kernels.scatter_nd_skip_oob_indices(
            input=prev_to_curr_token_indices,
            updates=possible_future_token_indices.cast(DType.int64),
            indices=curr_to_prev_map,
        )
        realized_tokens = kernels.scatter_nd_skip_oob_indices(
            input=input_values.curr_tokens,
            updates=input_values.prev_generated_tokens,
            indices=ops.unsqueeze(prev_to_curr_token_indices, axis=-1),
        )

        if input_values.spec_decode is None:
            graph.output(realized_tokens)
        else:
            spec_decode = input_values.spec_decode
            num_draft_tokens_dim = spec_decode.curr_draft_tokens.shape[1]
            num_draft_tokens = _shape_to_scalar(
                num_draft_tokens_dim, device0, dtype=DType.uint32
            )
            prev_generated_draft_tokens = (
                spec_decode.prev_generated_draft_tokens[
                    :, :num_draft_tokens_dim
                ]
            )

            # 0...verify_width
            draft_col_range = ops.range(
                start=0,
                stop=num_draft_tokens_dim,
                out_dim="num_draft_tokens",
                device=device0,
                dtype=DType.uint32,
            )

            draft_slot_indices = (
                prev_to_curr_map * num_draft_tokens
            ).broadcast_to(
                shape=["prev_batch_size", "num_draft_tokens"]
            ) + draft_col_range

            total_curr_draft_elems = SymbolicDim(
                "curr_batch_size"
            ) * SymbolicDim("num_draft_tokens")
            total_prev_draft_elems = SymbolicDim(
                "prev_batch_size"
            ) * SymbolicDim("num_draft_tokens")
            flat_draft_slot_indices = draft_slot_indices.reshape(
                [total_prev_draft_elems, 1]
            )
            realized_draft_tokens = kernels.scatter_nd_skip_oob_indices(
                input=spec_decode.curr_draft_tokens.reshape(
                    [total_curr_draft_elems]
                ),
                updates=prev_generated_draft_tokens.reshape(
                    [total_prev_draft_elems]
                ),
                indices=flat_draft_slot_indices,
            ).reshape(["curr_batch_size", "num_draft_tokens"])

            # Same slot indices as the tokens, just widened by the vocabulary
            # axis, so a distribution cannot drift away from the token it
            # proposed. Rows not in the previous batch keep the zeros
            # `_prepare_draft_probs_full` cleared them to.
            realized_draft_probs_full: TensorValue | None = None
            if spec_decode.curr_draft_probs_full is not None:
                assert spec_decode.prev_generated_draft_probs_full is not None
                vocab_dim = spec_decode.curr_draft_probs_full.shape[2]
                prev_generated_probs = (
                    spec_decode.prev_generated_draft_probs_full[
                        :, :num_draft_tokens_dim, :
                    ]
                )
                realized_draft_probs_full = kernels.scatter_nd_skip_oob_indices(
                    input=spec_decode.curr_draft_probs_full.reshape(
                        [total_curr_draft_elems, vocab_dim]
                    ),
                    updates=prev_generated_probs.reshape(
                        [total_prev_draft_elems, vocab_dim]
                    ),
                    indices=flat_draft_slot_indices,
                ).reshape(["curr_batch_size", "num_draft_tokens", vocab_dim])

            # Per-sequence increments in int64 (may be negative), then fold into
            # uint32 cache lengths to match :class:`~max.nn.kv_cache.KVCacheParams`
            # paged-cache ``cache_lengths`` dtype.
            batch_increments_i64 = ops.broadcast_to(
                ops.constant(0, dtype=DType.int64, device=device0),
                ["curr_batch_size"],
            )

            # The curr cache lengths optimistically assume every draft the
            # PREVIOUS step verified was accepted, so roll back by that step's
            # own verify width ``prev_draft_tokens``, not this step's
            # draft-column count, which may be 0 or a different width entirely.
            # It is also the width ``maybe_accepted_draft_tokens`` was filled
            # to, which is what the host added to the cache length.
            prev_num_draft_tokens = _shape_to_scalar(
                spec_decode.prev_draft_tokens.shape[1],
                device0,
                dtype=DType.int64,
            )
            delta = (
                spec_decode.prev_num_accepted_draft_tokens
                - ops.broadcast_to(prev_num_draft_tokens, ["prev_batch_size"])
            )
            batch_increments_i64 = kernels.scatter_nd_skip_oob_indices(
                input=batch_increments_i64,
                updates=delta,
                indices=prev_to_curr_map,
            )

            curr_cache_lengths = spec_decode.curr_cache_lengths

            # realize the cache lengths
            realized_cache_lengths: list[TensorValue] = []
            if len(devices) == 1:
                cache_length_u32 = ops.rebind(
                    curr_cache_lengths[0], ["curr_batch_size"]
                )
                cache_length_adjusted = (
                    cache_length_u32.cast(DType.int64) + batch_increments_i64
                )
                realized_cache_lengths.append(
                    cache_length_adjusted.cast(DType.uint32)
                )
            elif enable_dp:
                # DP > 1 (possibly mixed with TP). ``data_parallel_splits`` is
                # indexed by DP replica (length ``dp_degree + 1``), NOT by
                # device. Under mixed TP+DP every ``tp_degree`` consecutive
                # devices form one replica and share the same request slice, so
                # map device ``i`` to replica ``i // tp_degree`` before slicing.
                # Indexing by the raw device index (the historical pure-DP
                # assumption, valid only when ``tp_degree == 1``) walks off the
                # end of the ``dp_degree + 1`` splits tensor for ``tp_degree >
                # 1``, producing out-of-bounds / empty slices that corrupt
                # ``cache_lengths`` on the non-leader TP ranks of each replica.
                assert spec_decode.signal_buffers is not None
                assert spec_decode.data_parallel_splits is not None

                batch_increments_distributed = ops.distributed_broadcast(
                    batch_increments_i64, spec_decode.signal_buffers
                )

                tp_degree = len(devices) // data_parallel_degree
                for i in range(len(devices)):
                    replica = i // tp_degree
                    start_offset = spec_decode.data_parallel_splits[replica]
                    end_offset = spec_decode.data_parallel_splits[replica + 1]

                    batch_increments_local = ops.slice_tensor(
                        batch_increments_distributed[i],
                        [
                            (
                                slice(
                                    start_offset,
                                    end_offset,
                                ),
                                f"curr_batch_size_gpu_{i}",
                            )
                        ],
                    )
                    replica_cache_length = (
                        curr_cache_lengths[i].cast(DType.int64)
                        + batch_increments_local
                    )
                    realized_cache_lengths.append(
                        replica_cache_length.cast(DType.uint32)
                    )
            else:
                # TP > 1
                assert spec_decode.signal_buffers is not None
                cache_length_u32 = ops.rebind(
                    curr_cache_lengths[0], ["curr_batch_size"]
                )
                cache_length_adjusted = (
                    cache_length_u32.cast(DType.int64) + batch_increments_i64
                ).cast(DType.uint32)

                realized_cache_lengths.extend(
                    ops.distributed_broadcast(
                        cache_length_adjusted, spec_decode.signal_buffers
                    )
                )

            # The realized probabilities must precede the variadic
            # cache-lengths tail: the call site unpacks
            # ``(tokens, draft_tokens, [draft_probs, draft_probs_full,]
            # *cache_lengths)``.
            graph.output(
                realized_tokens,
                realized_draft_tokens,
                *(
                    (realized_draft_probs_full,)
                    if realized_draft_probs_full is not None
                    else ()
                ),
                *realized_cache_lengths,
            )

    return graph


class RealizeFutureTokenProcessor:
    """Processor for realizing placeholder future tokens in ragged input on the GPU.

    We scatter the generated tokens from the previous batch into the slots
    containing placeholder future tokens in the current batch. This all occurs
    efficiently on the gpu. We use a variant of scatter_nd that skips out of
    bound indices in cases where the current batch does not contain a request
    present in the previous batch.
    """

    def __init__(
        self,
        session: InferenceSession,
        devices: Sequence[DeviceRef],
        num_speculative_tokens: int = 0,
        enable_dp: bool = False,
        data_parallel_degree: int = 1,
        sampled_draft_vocab_size: int | None = None,
    ) -> None:
        with CompilationTimer("realize_future_token") as timer:
            graph = build_realize_future_token_graph(
                devices=devices,
                num_speculative_tokens=num_speculative_tokens,
                enable_dp=enable_dp,
                data_parallel_degree=data_parallel_degree,
                sampled_draft_vocab_size=sampled_draft_vocab_size,
            )
            timer.mark_build_complete()
            self._graph = session.load(graph)
        self._enable_dp = enable_dp
        self._num_speculative_tokens = num_speculative_tokens
        self._num_devices = len(devices)
        self._sampled_draft_proposal = sampled_draft_vocab_size is not None

    def _compute_mappings(
        self,
        prev_batch: AsyncBatch[TextGenerationContextType],
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> tuple[DevicePinnedBuffer, DevicePinnedBuffer] | None:
        """Computes scatter indices mapping previous-batch tokens to current slots.

        Returns None if all indices are out-of-bounds (no overlap between
        the previous and current batch), indicating the scatter can be skipped.
        """
        prev_generated_tokens = prev_batch.generated_tokens_device
        device = prev_generated_tokens.device
        if device.is_host:
            raise ValueError(
                "Realize future tokens processor must be on the gpu."
            )

        # Prepare the scatter indices.
        prev_batch_size = prev_generated_tokens.shape[0]
        prev_to_curr_map_host = DevicePinnedBuffer(
            shape=(prev_batch_size,),
            dtype=DType.int64,
            device=device,
        )
        prev_to_curr_map = prev_to_curr_map_host.to_numpy()

        curr_batch_size = len(inputs.flat_batch)
        curr_to_prev_map_host = DevicePinnedBuffer(
            shape=(curr_batch_size,),
            dtype=DType.int64,
            device=device,
        )
        curr_to_prev_map = curr_to_prev_map_host.to_numpy()

        # Initialize the scatter indices with an oob_idx. These updates will be
        # skipped by the scatter_nd kernel.
        prev_to_curr_map.fill(_OOB_IDX)
        curr_to_prev_map.fill(_OOB_IDX)

        # If a request is present in both the previous and current batch,
        # we record the mapping from the prev to curr batch idx.
        req_id_to_curr_batch_idx = {
            context.request_id: curr_batch_idx
            for curr_batch_idx, context in enumerate(inputs.flat_batch)
        }

        prev_flat_batch = prev_batch.inputs.flat_batch
        for prev_idx, context in enumerate(prev_flat_batch):
            req_id = context.request_id
            # If generated_length is still 0, then there is no placeholder future
            # token. This is possible due to chunked prefill.
            if (
                req_id in req_id_to_curr_batch_idx
                and context.tokens.generated_length
            ):
                prev_to_curr_map[prev_idx] = req_id_to_curr_batch_idx[req_id]
                curr_to_prev_map[req_id_to_curr_batch_idx[req_id]] = prev_idx

        if np.all(prev_to_curr_map == _OOB_IDX):
            return None
        else:
            return prev_to_curr_map_host, curr_to_prev_map_host

    @traced
    def realize_future_tokens(
        self,
        prev_batch: AsyncBatch[TextGenerationContextType],
        inputs: TextGenerationInputs[TextGenerationContextType],
        model_inputs: ModelInputs,
        draft_tokens_np: npt.NDArray[np.int64] | None = None,
    ) -> npt.NDArray[np.int64] | None:
        """Scatters generated tokens from the previous batch into placeholder slots.

        Fills placeholder future tokens in the current batch on the GPU.
        Returns ragged_input_tokens unchanged if there is no overlap between
        the previous and current batch.

        Returns:
            Host mirror of the device draft scatter ``[curr_batch, K]`` for the
            synchronous-fill bitmask, or ``None`` (see ``realized_draft_tokens_host``).
        """
        assert isinstance(model_inputs, _HasRaggedTokens)
        assert not prev_batch._is_processed, (
            "Cannot realize device inputs from an already host-processed batch"
        )
        realized_draft_tokens_host: npt.NDArray[np.int64] | None = None
        mappings = self._compute_mappings(prev_batch, inputs)

        if mappings is None:
            return None

        assert self._graph is not None, (
            "RealizeFutureTokenProcessor is None but there are tokens to scatter."
        )

        # Traverse the KV tree and collect the KV cache inputs per device.
        def _recurse_kv_tree(
            kv: KVCacheInputsInterface[Any, Any],
            kv_collections: list[KVCacheInputsPerDevice[Buffer, Buffer]],
        ) -> None:
            if isinstance(kv, KVCacheInputs):
                kv_collections.extend(kv.inputs)
            elif isinstance(kv, MultiKVCacheInputs):
                for child in kv.children.values():
                    _recurse_kv_tree(child, kv_collections)
            else:
                raise ValueError(f"Unexpected KV cache input type: {type(kv)}")

        kv_collections: list[KVCacheInputsPerDevice[Buffer, Buffer]] = []

        if self._num_speculative_tokens > 0:
            assert isinstance(model_inputs, _UnifiedSpecDecodeInputs)
            assert prev_batch.spec_decode is not None
            assert model_inputs.kv_cache_inputs is not None
            _recurse_kv_tree(model_inputs.kv_cache_inputs, kv_collections)

            cache_lengths = [
                kv.cache_lengths for kv in kv_collections[: self._num_devices]
            ]
            assert model_inputs.draft_tokens is not None
            num_draft_tokens_to_verify = model_inputs.draft_tokens.shape[1]
            num_accepted_draft_tokens = (
                prev_batch.spec_decode.num_accepted_draft_tokens_device
            )
            prev_generated_draft_tokens = (
                prev_batch.spec_decode.next_draft_tokens_device
            )
            prev_draft_tokens = (
                prev_batch.spec_decode.draft_tokens_to_verify_device
            )
            signal_buffers = (
                getattr(model_inputs, "signal_buffers", None)
                if self._num_devices > 1
                else None
            )

            if self._enable_dp:
                data_parallel_splits = model_inputs.data_parallel_splits
            else:
                data_parallel_splits = None
            curr_draft_probs_full: Buffer | None = None
            prev_generated_draft_probs_full: Buffer | None = None
            if self._sampled_draft_proposal:
                assert model_inputs.draft_probs_full is not None, (
                    "draft_proposal='sampled' requires model_inputs to carry "
                    "draft_probs_full"
                )
                assert (
                    prev_batch.spec_decode.next_draft_probs_full_device
                    is not None
                ), (
                    "draft_proposal='sampled' requires the architecture to "
                    "emit next_draft_probs_full (the 4th graph output)"
                )
                curr_draft_probs_full = model_inputs.draft_probs_full
                prev_generated_draft_probs_full = (
                    prev_batch.spec_decode.next_draft_probs_full_device
                )

            # The generated array binds at its own width and the graph
            # trims it to this step's verify width, so a zero-verify step needs
            # no substitute empty buffer here.
            spec_decode: (
                _RealizeFutureTokenSpecDecodeInputs[Buffer, Buffer] | None
            ) = _RealizeFutureTokenSpecDecodeInputs(
                curr_draft_tokens=model_inputs.draft_tokens,
                data_parallel_splits=data_parallel_splits,
                curr_cache_lengths=cache_lengths,
                signal_buffers=signal_buffers,
                prev_generated_draft_tokens=prev_generated_draft_tokens,
                prev_draft_tokens=prev_draft_tokens,
                prev_num_accepted_draft_tokens=num_accepted_draft_tokens,
                curr_draft_probs_full=curr_draft_probs_full,
                prev_generated_draft_probs_full=prev_generated_draft_probs_full,
            )
        else:
            spec_decode = None

        prev_to_curr_map, curr_to_prev_map = mappings
        device = prev_to_curr_map.device

        # TODO: This is a hotfix for MODELS-1350. Do the cleaner thing in followup.
        input_row_offsets: list[Buffer] | Buffer = (
            model_inputs.input_row_offsets
        )
        if isinstance(input_row_offsets, list):
            input_row_offsets = input_row_offsets[0]

        my_inputs = _RealizeFutureTokenInputs[Buffer, Buffer](
            prev_to_curr_map=prev_to_curr_map.to(device),
            curr_to_prev_map=curr_to_prev_map.to(device),
            curr_tokens=model_inputs.tokens,
            curr_input_row_offsets=input_row_offsets,
            prev_generated_tokens=prev_batch.generated_tokens_device,
            spec_decode=spec_decode,
        )

        out = self._graph.execute(*my_inputs.flatten())

        # Execute the realize_future_tokens kernel.
        if my_inputs.spec_decode is not None:
            assert isinstance(model_inputs, _UnifiedSpecDecodeInputs)
            draft_probs_full = None
            if my_inputs.spec_decode.curr_draft_probs_full is not None:
                (
                    tokens,
                    draft_tokens,
                    draft_probs_full,
                    *cache_lengths,
                ) = out
            else:
                (tokens, draft_tokens, *cache_lengths) = out
            model_inputs.tokens = tokens
            # This is pretty subtle. We copy the realized tokens into the original
            # draft tokens buffer so that when we read from draft_tokens later on
            # we get the real values...
            model_inputs.draft_tokens.inplace_copy_from(draft_tokens)
            if draft_probs_full is not None:
                assert model_inputs.draft_probs_full is not None
                model_inputs.draft_probs_full.inplace_copy_from(
                    draft_probs_full
                )
            # Overwrite the cache_lengths with the realized cache_lengths.
            for i, kv in enumerate(kv_collections):
                cl = cache_lengths[i % len(cache_lengths)]
                assert kv.cache_lengths.device == cl.device
                assert kv.cache_lengths.shape == cl.shape
                assert kv.cache_lengths.dtype == cl.dtype
                kv.cache_lengths = cache_lengths[i % len(cache_lengths)]
            # Host mirror of the device draft buffer for the constrained-
            # decoding synchronous-fill (see _host_mirror_realized_drafts). Only on the
            # synchronous-fill case -- prev did not advance the FSM via callback -- where
            # the early-sync guard has already made next_draft_tokens_host
            # complete.
            if (
                num_draft_tokens_to_verify > 0
                and not prev_batch.spec_decode.fsm_advanced_by_callback
                and draft_tokens_np is not None
            ):
                realized_draft_tokens_host = _host_mirror_realized_drafts(
                    draft_tokens_np,
                    prev_to_curr_map.to_numpy(),
                    prev_batch.spec_decode.next_draft_tokens_host.to_numpy(),
                )
        else:
            (new_ragged_input_tokens,) = out
            model_inputs.tokens = new_ragged_input_tokens

        # Update the model inputs with the new ragged input tokens.

        return realized_draft_tokens_host


@dataclass
class _AsyncSpecDecodeHostBuffers:
    """Fresh per-batch pinned buffers for spec-decode D2H outputs.

    Populated by D2H from spec-decode model outputs and read by the sync
    path on the next iteration.

    These are separate from the persistent pinned buffers on `SpecDecodeState`:
    the persistent buffers are reused across iterations (overwritten by each
    new D2H) and read by the async callback before the next D2H runs, while
    these per-batch buffers live as long as their owning AsyncSpecDecodeBatch
    and are safe to read on a later iteration's sync path.
    """

    num_accepted_draft_tokens_host: DevicePinnedBuffer
    next_tokens_host: DevicePinnedBuffer
    next_draft_tokens_host: DevicePinnedBuffer


@final
class OverlapTextGenerationPipeline(
    TextGenerationPipelineInterface[TextGenerationContextType],
    Generic[TextGenerationContextType],
):
    """Overlap text generation pipeline."""

    _pipeline_model: PipelineModelWithKVCache[Any]

    _width_lookup: list[int] | None = None
    """``batch_size -> drafts to verify``, set only under speculative decoding
    with a schedule configured. ``None`` verifies every carried draft."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineModel[Any]],
        weight_adapters: dict[WeightsFormat, WeightsAdapter],
        tokenizer: PipelineTokenizer[
            TextGenerationContextType,
            npt.NDArray[np.integer[Any]],
            TextGenerationRequest,
        ],
        memory_plan: MemoryPlan,
        disable_overlap: bool = False,
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
            disable_overlap: When this flag is set, the overlap scheduler will
                immediately synchronize after model execution. This removes any
                potential cpu / gpu overlap.

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
                f"Overlap text generation pipeline requires a HuggingFace config for '{model_config.model_path}', "
                "but config could not be loaded. "
                "Please ensure the model repository contains a valid config.json file."
            )

        self._devices = load_devices(list(memory_plan.require_device_specs()))
        if self._devices[0].is_host:
            raise ValueError(
                "OverlapTextGenerationPipeline does not support CPU models."
            )
        self._tokenizer = tokenizer

        # -1 sentinel disables in_reasoning_phase tracking.
        self._think_start_token_id: int = -1
        self._think_end_token_id: int = -1
        if pipeline_config.runtime.reasoning_parser is not None:
            if not isinstance(tokenizer, ReasoningPipelineTokenizer):
                raise ValueError(
                    f"reasoning_parser={pipeline_config.runtime.reasoning_parser!r} "
                    f"requires the architecture's tokenizer to implement "
                    f"ReasoningPipelineTokenizer, but "
                    f"{type(tokenizer).__name__} does not. "
                    f"Implement reasoning_start_token_id and "
                    f"reasoning_end_token_id on the tokenizer."
                )
            self._think_start_token_id, self._think_end_token_id = (
                _resolve_thinking_token_ids(tokenizer)
            )

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
                f"OverlapTextGenerationPipeline requires a model with KV cache support, found {pipeline_model.__name__}"
            )

        enable_echo = self._pipeline_config.model.enable_echo
        is_spec_decode = self._pipeline_config.speculative is not None
        if is_spec_decode and enable_echo:
            raise ValueError(
                "Enable Echo is not supported for speculative decoding. Please disable echo."
            )
        if is_spec_decode:
            return_logits = ReturnLogits.VARIABLE
        else:
            return_logits = ReturnLogits.LAST_TOKEN
        self._pipeline_model: PipelineModelWithKVCache[Any] = pipeline_model(
            pipeline_config=self._pipeline_config,
            session=session,
            devices=self._devices,
            kv_cache_config=model_config.kv_cache,
            weights=load_weights(weight_paths),
            adapter=weight_adapters.get(weights_format(weight_paths)),
            return_logits=return_logits,
            max_batch_size=max_batch_size,
            memory_plan=memory_plan,
        )

        available_cache_memory = memory_plan.available_cache_memory
        kv_params = self._pipeline_model.kv_params

        # Load the KVCache manager.  For models with multiple KV caches
        # (e.g. sliding-window + global attention), a single manager
        # handles all caches natively via MultiKVCacheParams.
        self._spec_decode_state: SpecDecodeState | None = None
        if not is_spec_decode:
            self._kv_manager = load_kv_manager(
                params=kv_params,
                max_batch_size=max_batch_size,
                max_seq_len=self._pipeline_model.max_seq_len,
                session=session,
                available_cache_memory=available_cache_memory,
                is_di_enabled=self._pipeline_config.runtime.is_disaggregated,
                model_name=pipeline_config.model.model_name,
            )
        else:
            # vocab_size gates bitmask buffer + overlap_state
            # allocation. Pass None when constrained decoding can't
            # fire so we don't allocate persistent pinned buffers /
            # an overlap state that the (bitmask-less) model graph
            # has no inputs for.
            self._spec_decode_state = SpecDecodeState.load(
                session=session,
                model=self._pipeline_model,
                pipeline_config=self._pipeline_config,
                max_batch_size=max_batch_size,
                available_cache_memory=available_cache_memory,
                vocab_size=(
                    self.vocab_size
                    if pipeline_config.needs_bitmask_constraints
                    else None
                ),
            )
            self._kv_manager = self._spec_decode_state.kv_manager
            # ``batch_size -> drafts to verify``.
            self._width_lookup = _verify_width_lookup(
                self._pipeline_config.speculative,
                self._spec_decode_state.num_speculative_tokens,
                self._max_batch_size,
            )
            if self._width_lookup is not None:
                logger.info(
                    "Verifying %s of %d drafted tokens by decode batch size.",
                    sorted(set(self._width_lookup[1:])),
                    self._spec_decode_state.num_speculative_tokens,
                )
            if (
                self._pipeline_config.speculative is not None
                and self._pipeline_config.speculative.synthetic_acceptance_rate
                is not None
            ):
                logger.info(
                    "Synthetic acceptance rate is enabled (rate=%.2f). "
                    "Actual model acceptance will be overridden. "
                    "Results are for benchmarking only.",
                    self._pipeline_config.speculative.synthetic_acceptance_rate,
                )

        self._encoder_cache: VisionEncoderCache[TextAndVisionContext] | None = (
            None
        )
        if isinstance(self._pipeline_model, SupportsVisionEncoding):
            self._encoder_cache = VisionEncoderCache[TextAndVisionContext](
                plan=memory_plan.vision_cache_plan,
                devices=self._devices,
            )

        # Load sampler(s) for the non-spec-decode path. The bitmask-aware
        # sampler is loaded when constrained decoding could fire (see
        # ``needs_bitmask_constraints``). The bitmask-free sampler is
        # always loaded so requests that don't engage structured output
        # (even with ``--enable-structured-output`` set server-wide) can
        # still be sampled.
        # Device the sampler runs on. ``sample_on_host`` routes sampling to the
        # host CPU.
        self._sampler_device: Device = (
            CPU()
            if pipeline_config.sampling.sample_on_host
            else self._devices[0]
        )
        sampler_device_ref = DeviceRef.from_device(self._sampler_device)

        sampler_extensions = (
            ()
            if self._sampler_device.is_host
            else self._pipeline_model.sampler_custom_extensions
        )

        self._sampler_with_bitmask: Model | None = None
        self._sampler_without_bitmask: Model | None = None
        if not is_spec_decode:
            with_bitmask_graph = None
            with CompilationTimer("sampler") as sampler_timer:
                if pipeline_config.needs_bitmask_constraints:
                    with_bitmask_graph = token_sampler(
                        pipeline_config.sampling,
                        device=sampler_device_ref,
                        needs_bitmask_input=True,
                        custom_extensions=sampler_extensions,
                    )
                without_bitmask_graph = token_sampler(
                    pipeline_config.sampling,
                    device=sampler_device_ref,
                    needs_bitmask_input=False,
                    custom_extensions=sampler_extensions,
                )
                sampler_timer.mark_build_complete()
                if with_bitmask_graph is not None:
                    self._sampler_with_bitmask = session.load(
                        with_bitmask_graph
                    )
                self._sampler_without_bitmask = session.load(
                    without_bitmask_graph
                )

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

        # Persistent pinned host buffer for the per-step generated-token D2H in
        # `_sample_logits`. Allocated once and reused (sliced to the batch size)
        # so the decode critical path does not pay a page-locking pinned-host
        # allocation on every step. Lazily created on first use to match the
        # sampler's token dtype; the host / virtual-device paths skip it.
        self._pinned_generated_tokens_host: Buffer | None = None

        self._identity_logit_offsets = (
            FusedSamplingProcessor.allocate_identity_logit_offsets(
                pipeline_config, self._sampler_device, max_batch_size
            )
        )

        # Overlap scheduling specific initialization.

        # Load the realize future tokens graph — not needed on prefill-only
        # workers (no decode phase, so no future tokens to scatter).
        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )
        self._realize_future_token_processor: (
            RealizeFutureTokenProcessor | None
        ) = (
            RealizeFutureTokenProcessor(
                session=session,
                devices=[
                    DeviceRef.from_device(device) for device in self._devices
                ],
                num_speculative_tokens=num_speculative_tokens,
                enable_dp=model_config.data_parallel_degree > 1,
                data_parallel_degree=model_config.data_parallel_degree,
                # Derived from the one existing source of truth: the
                # persistent probs buffer exists iff
                # speculative.draft_proposal == "sampled".
                sampled_draft_vocab_size=(
                    self._spec_decode_state.persistent_draft_probs_full.shape[2]
                    if self._spec_decode_state is not None
                    and self._spec_decode_state.persistent_draft_probs_full
                    is not None
                    else None
                ),
            )
            if self._pipeline_config.runtime.pipeline_role
            in ("prefill_and_decode", "decode_only")
            else None
        )
        # Set previous asynchronously executing batch to None.
        self._prev_batch: AsyncBatch[TextGenerationContextType] | None = None
        # Timing state for attributing execution time to completed batches.
        # ``_last_sync_monotonic`` is when the previously synced batch's
        # outputs were observed on the host; ``_completed_batch_stats`` holds
        # stats for the most recently synced batch until the scheduler
        # collects them via ``take_completed_batch_stats()``.
        self._last_sync_monotonic: float | None = None
        self._completed_batch_stats: CompletedBatchStats | None = None
        self._graph_capture_runner: ServeGraphCaptureRunner | None = None
        # set a default graph capture size, 128
        self._max_graph_capture_batch_size: int = _MAX_GRAPH_CAPTURE_BATCH_SIZE
        # Cache-length bucket aligner for the device-graph-synthesis pathway.
        self._synthesis_aligner: SynthesisBucketAligner | None = None

        self._disable_overlap = disable_overlap

    @property
    def max_batch_size(self) -> int:
        """Maximum number of requests that can be processed in a single batch."""
        return self._max_batch_size

    @property
    def _effective_max_cache_length(self) -> int:
        """Capture-time upper bound on a request's runtime cache length.

        ``PagedKVCacheManager.runtime_inputs`` rejects any batch whose
        ``_compute_seq_len`` exceeds this bound, so it folds in the worst-case
        speculative-decode slack a context-boundary request adds beyond
        ``max_seq_len``. Capped to pool capacity, which already carries that
        headroom, so capture never reserves more pages than were allocated.
        """
        params = self._kv_manager.params
        max_seq_len = (
            self._pipeline_model.max_seq_len + spec_decode_cache_slack(params)
        )
        kv_max_seq_len = self._kv_manager.effective_max_seq_length
        if kv_max_seq_len is None:
            return max_seq_len
        else:
            return min(max_seq_len, kv_max_seq_len)

    @property
    def overlap_active(self) -> bool:
        """Whether CPU/GPU overlap is actually in effect.

        When overlap is active, ``execute()`` defers synchronization of the
        current batch until the next call, so wall-clock time measured around
        ``execute()`` reflects the previous batch's execution, not the
        current one.
        """
        return not self._disable_overlap

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Returns the pipeline configuration."""
        return self._pipeline_config

    @property
    def tokenizer(
        self,
    ) -> PipelineTokenizer[
        TextGenerationContextType,
        npt.NDArray[np.integer[Any]],
        TextGenerationRequest,
    ]:
        """Returns the tokenizer used for building contexts and decoding."""
        return self._tokenizer

    def update_for_structured_output(
        self,
        context: TextGenerationContextType,
        bitmask: npt.NDArray[np.int32],
        index: int,
    ) -> None:
        """Update context and logits bitmask for structured output.

        If a ``json_schema`` is present and no matcher is set, this compiles a
        grammar matcher and installs it on the context, then fills the per-request
        token bitmask used to constrain the next-token distribution.

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

    def has_pending_outputs(self) -> bool:
        """Returns True if there are pending outputs for the previous batch.

        If this is True, the caller should call ``execute()`` even with empty
        inputs to retrieve the outputs for the previous batch.
        """
        return self._prev_batch is not None

    def take_completed_batch_stats(self) -> CompletedBatchStats | None:
        """Returns and clears stats for the most recently completed batch.

        When overlap is active, a batch's outputs are synchronized one
        ``execute()`` call after it was enqueued, so the scheduler cannot
        attribute execution time to the batch it just submitted. After each
        ``execute()`` call that synchronized a batch, this returns that
        batch's composition and timing; the scheduler should publish
        execution-time and throughput telemetry from this record instead of
        from its own wall-clock measurement. Returns ``None`` when no batch
        completed since the last call.
        """
        stats = self._completed_batch_stats
        self._completed_batch_stats = None
        return stats

    def _record_completed_batch_stats(
        self,
        batch: AsyncBatch[TextGenerationContextType],
        spec_decode_metrics: _SpeculativeDecodingMetrics | None,
        sync_monotonic: float,
        early_sync_duration_s: float | None = None,
    ) -> None:
        """Records stats for a batch whose outputs were just synchronized.

        The execution time is estimated host-side as the interval from when
        the batch could have started executing — the later of its enqueue
        timestamp and the previous batch's sync timestamp (batches execute
        back-to-back on the same stream) — until its outputs were observed on
        the host. On a saturated GPU this converges to the batch's GPU
        duration; it overestimates by any GPU idle gap before the batch.

        ``early_sync_duration_s`` carries the ``_should_early_sync_prev_batch``
        guard's own blocking duration, when it fired for this batch.
        """
        start_bound = batch.enqueue_monotonic
        if self._last_sync_monotonic is not None:
            start_bound = max(start_bound, self._last_sync_monotonic)
        inputs = batch.inputs
        stats = CompletedBatchStats(
            batch_type=inputs.batch_type,
            batch_size=inputs.batch_size,
            num_input_tokens=inputs.input_tokens,
            num_context_tokens=inputs.context_tokens,
            execution_time_s=max(sync_monotonic - start_bound, 0.0),
            early_sync_duration_s=early_sync_duration_s,
        )
        if spec_decode_metrics is not None:
            stats.num_output_tokens = spec_decode_metrics.output_tokens
            stats.draft_tokens_generated = (
                spec_decode_metrics.draft_tokens_generated
            )
            stats.draft_tokens_accepted = (
                spec_decode_metrics.draft_tokens_accepted
            )
            stats.avg_acceptance_length = (
                spec_decode_metrics.avg_acceptance_length
            )
            stats.max_acceptance_length = (
                spec_decode_metrics.num_speculative_tokens
            )
            stats.acceptance_rate_per_position = (
                spec_decode_metrics.acceptance_rate_per_position
            )
        self._completed_batch_stats = stats
        self._last_sync_monotonic = sync_monotonic

    # Warmup inputs use runtime construction with explicit max-cache-length LUT
    # sizing, so eager warmup and capture both see replay-stable buffer shapes.
    @contextmanager
    def _warmup_model_inputs(
        self, batch_size: int, batch_characteristics: BatchCharacteristics
    ) -> Iterator[ModelInputs]:
        dp_size = self._pipeline_config.model.data_parallel_degree
        replica_batches: list[list[TextContext]] = []

        num_speculative_tokens = (
            batch_characteristics.max_prompt_length - 1
            if self._spec_decode_state is not None
            else 0
        )

        # For unified Eagle/MTP models, the graph merges prompt tokens with
        # draft tokens internally. Each request contributes 1 decode token
        # as input; the q_max_seq_len only affects KV cache dispatch metadata.
        num_decode_tokens = 1
        for _replica_idx in range(dp_size):
            replica_batches.append(
                [
                    TextContext(
                        max_length=self._pipeline_model.max_seq_len,
                        tokens=TokenBuffer(
                            np.zeros(num_decode_tokens, dtype=np.int64)
                        ),
                        eos_tracker=EOSTracker(),
                        model_name=self._pipeline_config.model.model_name,
                        _spec_decoding_state=SpecDecodingState(
                            draft_tokens_to_verify=[0] * num_speculative_tokens,
                        ),
                    )
                    for idx in range(batch_size)
                ]
            )
        for replica_idx, contexts in enumerate(replica_batches):
            for context in contexts:
                self._kv_manager.claim(context, replica_idx=replica_idx)
                self._kv_manager.alloc(context)

        max_cache_length = self._effective_max_cache_length
        # Prepare dispatch metadata for the probed characteristics so the
        # captured graph matches what replay produces for the same aligned
        # cache length.
        kv_cache_inputs = self._kv_manager.runtime_inputs(
            replica_batches,
            max_cache_length=max_cache_length,
            batch_characteristics=batch_characteristics,
        )

        return_n_logits = (
            num_speculative_tokens + 1
            if self._spec_decode_state is not None
            else 0
        )

        with Tracer("prepare_initial_token_inputs"):
            model_inputs = self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=replica_batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )

        # Warmup packs ``.buffers`` without going through the prep phase's
        # vision drive (or ``execute()``), so the vision-merge inputs the
        # compiled graph unconditionally declares must be finalized here
        # with the model's empties.
        if self._encoder_cache is not None:
            assert isinstance(self._pipeline_model, SupportsVisionEncoding)
            self._encoder_cache.finalize_vision_inputs(
                self._pipeline_model, model_inputs, self._devices, None
            )

        if self._spec_decode_state is not None:
            assert isinstance(model_inputs, _UnifiedSpecDecodeInputs)
            draft_tokens = Buffer.from_numpy(
                np.zeros(
                    (batch_size * dp_size, num_speculative_tokens),
                    dtype=np.int64,
                )
            )
            persistent_draft_tokens = (
                self._spec_decode_state.persistent_draft_tokens
            )
            persistent_draft_tokens = _contiguous_prefix_2d(
                persistent_draft_tokens,
                batch_size * dp_size,
                num_speculative_tokens,
            )
            persistent_draft_tokens.inplace_copy_from(draft_tokens)
            model_inputs.draft_tokens = persistent_draft_tokens

            if self._spec_decode_state.persistent_draft_probs_full is not None:
                model_inputs.draft_probs_full = _contiguous_prefix_3d(
                    self._spec_decode_state.persistent_draft_probs_full,
                    batch_size * dp_size,
                    num_speculative_tokens,
                    self._spec_decode_state.persistent_draft_probs_full.shape[
                        2
                    ],
                )

            warmup_flat_batch = [
                ctx for replica in replica_batches for ctx in replica
            ]
            sampling_buffers = self._build_spec_decode_sampling_buffers(
                warmup_flat_batch
            )
            model_inputs.temperature = sampling_buffers.temperature
            model_inputs.top_k = sampling_buffers.top_k
            model_inputs.max_k = sampling_buffers.max_k
            model_inputs.top_p = sampling_buffers.top_p
            model_inputs.min_top_p = sampling_buffers.min_top_p
            model_inputs.seed = sampling_buffers.seed
            model_inputs.in_thinking_phase = sampling_buffers.in_thinking_phase

            # Set all-valid packed bitmask for warmup (unconstrained).
            # Shape: [batch_size, num_speculative_tokens + 1,
            # packed_vocab_size]; -1 = all bits set = all tokens valid.
            # The overlap path binds a (pinned, wait_payload, scratch)
            # triple and primes the completion flag so each warmup
            # replay's in-graph wait passes immediately.
            overlap_state = self._spec_decode_state.overlap_state
            total_batch = batch_size * dp_size
            if overlap_state is not None:
                # The width being probed, not the deepest one, so the captured
                # bitmask buffers are the ones this width binds at replay.
                num_positions = num_speculative_tokens + 1
                packed_vocab_dim = overlap_state.packed_vocab_size
                prime_np = np.full(
                    (total_batch, num_positions, packed_vocab_dim),
                    -1,
                    dtype=np.int32,
                )
                overlap_state.prime(prime_np)
                # Bind via the cached-view helper: the same Buffer view
                # objects are returned at capture and every replay, so
                # GraphCaptureRunner.replay's preface inplace_copy_from
                # short-circuits via the self-is-src identity check.
                pinned_view, scratch_view = overlap_state.get_input_views(
                    total_batch, num_positions
                )
                model_inputs.pinned_bitmask = pinned_view
                model_inputs.wait_payload = overlap_state.wait_payload
                model_inputs.device_bitmask_scratch = scratch_view

        # Collect the warmup request IDs before yielding so they are
        # available for state-pool release after the probe completes.
        warmup_request_ids: list[RequestID] = [
            ctx.request_id for replica in replica_batches for ctx in replica
        ]

        try:
            yield model_inputs
        finally:
            # Models that maintain per-request SSM / conv state pools
            # outside the KV cache (e.g. Nemotron-H, Qwen3.5, LFM2) must
            # release their warmup slots here; otherwise the pool is
            # exhausted before serving begins.
            if isinstance(self._pipeline_model, SupportsSSMStateWarmup):
                self._pipeline_model.release_warmup_state(warmup_request_ids)
            for replica in replica_batches:
                for context in replica:
                    self._kv_manager.release(context)

    def warmup_graph_capture(self) -> None:
        """Initializes and runs overlap device graph capture warmup."""
        if not isinstance(self._pipeline_model, _SupportsModelCapture):
            raise RuntimeError(
                "Device graph capture is enabled but pipeline model does not "
                "expose a compiled model for capture/replay."
            )
        max_capture_batch_size = min(
            self._max_batch_size,
            _MAX_GRAPH_CAPTURE_BATCH_SIZE,
        )
        if max_capture_batch_size < self._max_batch_size:
            logger.warning(
                "Capping graph capture batch size to %d "
                "(max_batch_size=%d). Decode batches above %d will fall "
                "back to eager execution.",
                max_capture_batch_size,
                self._max_batch_size,
                max_capture_batch_size,
            )

        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )

        width_lookup = self._width_lookup
        verify_widths = (
            sorted(set(width_lookup[1 : max_capture_batch_size + 1]))
            if width_lookup is not None
            else [num_speculative_tokens]
        )

        graph_capture_runner = ServeGraphCaptureRunner(
            model=self._pipeline_model.model,
            kv_params=self._kv_manager.params,
            warmup_model_inputs=self._warmup_model_inputs,
            max_cache_length_upper_bound=self._effective_max_cache_length,
            max_batch_size=max_capture_batch_size,
            num_speculative_tokens=num_speculative_tokens,
            verify_widths=verify_widths,
            width_lookup=width_lookup,
        )
        self._graph_capture_runner = graph_capture_runner
        self._max_graph_capture_batch_size = max_capture_batch_size
        logger.info("Starting serve device graph capture warmup.")
        graph_capture_runner.warmup_pre_ready()
        logger.info("Completed serve device graph capture warmup.")

    def prepare_graph_synthesis_buckets(self) -> None:
        """Builds the device-graph-synthesis cache-length bucket aligner.

        Synthesis records into a device graph lazily inside ``execute`` (no
        pre-capture warmup like graph capture), so this only fixes the
        cache-length boundaries the runtime cache length snaps up to. The
        boundaries reuse the KV-cache probe-length policy from graph capture,
        keeping dispatch-metadata buffer contents stable across consecutive
        decode steps so the driver ``DeviceGraphCache`` hits instead of
        rebuilding each step (a ~3.3x decode cost without this).
        """
        num_speculative_tokens = (
            self._spec_decode_state.num_speculative_tokens
            if self._spec_decode_state is not None
            else 0
        )
        self._synthesis_aligner = SynthesisBucketAligner(
            kv_params=self._kv_manager.params,
            max_cache_length_upper_bound=self._effective_max_cache_length,
            num_speculative_tokens=num_speculative_tokens,
        )
        logger.info(
            "Prepared device-graph-synthesis cache-length buckets: %s",
            self._synthesis_aligner.recorded_cache_lengths,
        )

    def _build_spec_decode_sampling_buffers(
        self,
        context_batch: list[TextGenerationContextType],
    ) -> _SpecDecodeSamplingBuffers:
        """Fill persistent sampling buffers for the current batch."""
        assert self._spec_decode_state is not None
        batch_size = len(context_batch)
        device0 = self._devices[0]

        # Implicit reasoning pre-fill: Kimi/MiniMax chat templates start
        # the assistant turn already inside <think>, so the model never
        # emits the open token. Seed in_reasoning_phase=True at first
        # decode step so the </think> toggle has something to flip.
        if self._think_start_token_id >= 0 and self._think_end_token_id >= 0:
            for ctx in context_batch:
                if (
                    ctx.tokens.generated_length == 0
                    and not ctx.in_reasoning_phase
                ):
                    ctx.in_reasoning_phase = True

        # Pick thinking_temperature for rows whose host-side
        # ``in_reasoning_phase`` is True.
        temperature_np = np.fromiter(
            (
                ctx.sampling_params.thinking_temperature
                if (
                    ctx.in_reasoning_phase
                    and ctx.sampling_params.thinking_temperature is not None
                )
                else ctx.sampling_params.temperature
                for ctx in context_batch
            ),
            dtype=np.float32,
            count=batch_size,
        )
        top_k_np = np.fromiter(
            (ctx.sampling_params.top_k for ctx in context_batch),
            dtype=np.int64,
            count=batch_size,
        )
        top_p_np = np.fromiter(
            (ctx.sampling_params.top_p for ctx in context_batch),
            dtype=np.float32,
            count=batch_size,
        )

        in_thinking_phase_np = np.fromiter(
            (ctx.in_reasoning_phase for ctx in context_batch),
            dtype=np.bool_,
            count=batch_size,
        )

        temperature_pinned = DevicePinnedBuffer(
            shape=(batch_size,), dtype=DType.float32, device=device0
        )
        temperature_pinned.to_numpy()[:] = temperature_np
        top_k_pinned = DevicePinnedBuffer(
            shape=(batch_size,), dtype=DType.int64, device=device0
        )
        top_k_pinned.to_numpy()[:] = top_k_np
        top_p_pinned = DevicePinnedBuffer(
            shape=(batch_size,), dtype=DType.float32, device=device0
        )
        top_p_pinned.to_numpy()[:] = top_p_np
        in_thinking_phase_pinned = DevicePinnedBuffer(
            shape=(batch_size,), dtype=DType.bool, device=device0
        )
        in_thinking_phase_pinned.to_numpy()[:] = in_thinking_phase_np

        temperature_view = self._spec_decode_state.persistent_temperature[
            :batch_size
        ]
        temperature_view.inplace_copy_from(temperature_pinned)

        top_k_view = self._spec_decode_state.persistent_top_k[:batch_size]
        top_k_view.inplace_copy_from(top_k_pinned)

        top_p_view = self._spec_decode_state.persistent_top_p[:batch_size]
        top_p_view.inplace_copy_from(top_p_pinned)

        in_thinking_phase_view = (
            self._spec_decode_state.persistent_in_thinking_phase[:batch_size]
        )
        in_thinking_phase_view.inplace_copy_from(in_thinking_phase_pinned)

        max_k = Buffer.from_numpy(np.array(int(top_k_np.max()), dtype=np.int64))
        min_top_p = Buffer.from_numpy(
            np.array(float(top_p_np.min()), dtype=np.float32)
        )

        # Per-request seed mirrors the production sampler at
        # `SamplerInputs.create` (sampling_logits_processor.py:610-619):
        # seed[i] = sampling_params.seed + len(context.tokens). Adding the
        # current token count gives a fresh effective Philox seed per
        # decoding step without needing in-graph seed mutation.
        seed_np = np.fromiter(
            (
                ctx.sampling_params.seed + len(ctx.tokens)
                for ctx in context_batch
            ),
            dtype=np.uint64,
            count=batch_size,
        )
        seed_pinned = DevicePinnedBuffer(
            shape=(batch_size,), dtype=DType.uint64, device=device0
        )
        seed_pinned.to_numpy()[:] = seed_np
        seed_view = self._spec_decode_state.persistent_seed[:batch_size]
        seed_view.inplace_copy_from(seed_pinned)

        return _SpecDecodeSamplingBuffers(
            temperature=temperature_view,
            top_k=top_k_view,
            max_k=max_k,
            top_p=top_p_view,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase_view,
            seed=seed_view,
        )

    def _prepare_draft_probs_full(
        self,
        batch_size: int,
        num_draft_tokens_to_verify: int,
    ) -> Buffer | None:
        """Binds this batch's draft-distribution view, cleared to zero.

        These distributions live only on device, indexed by batch slot. The
        realize scatter refreshes the rows carried over from the previous
        batch; every other row still holds whatever request last occupied that
        slot. Clearing the prefix first makes the scatter the only source of
        truth, so a request that was preempted, resumed, or simply moved slots
        gets zeros rather than a stranger's proposal.

        Zero is what the acceptance path reads as "no distribution for this
        row": it falls back to typical acceptance and the argmax residual,
        which cannot re-emit the token the target just rejected.
        """
        assert self._spec_decode_state is not None
        persistent = self._spec_decode_state.persistent_draft_probs_full
        if persistent is None:
            return None

        vocab_size = persistent.shape[2]
        view = _contiguous_prefix_3d(
            persistent, batch_size, num_draft_tokens_to_verify, vocab_size
        )
        if num_draft_tokens_to_verify == 0 or batch_size == 0:
            return view

        zeros = self._spec_decode_state.draft_probs_full_zero_row
        assert zeros is not None
        row_elems = num_draft_tokens_to_verify * vocab_size
        flat = view.view(view.dtype, (view.num_elements,))
        zero_row = zeros.view(zeros.dtype, (zeros.num_elements,))[:row_elems]
        for i in range(batch_size):
            flat[i * row_elems : (i + 1) * row_elems].inplace_copy_from(
                zero_row
            )
        return view

    def _verify_width(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> int:
        """How many of the carried drafts this step verifies; 0 for none.

        A step always drafts the configured number of proposals; this is how
        many of them the target checks.

        This is the single source of the width for a step. Both the draft
        array and the constrained-decoding bitmask are shaped from it.

        """
        assert self._spec_decode_state is not None
        if not all(
            ctx.tokens.generated_length > 0 for ctx in inputs.flat_batch
        ):
            return 0
        if self._width_lookup is None:
            return self._spec_decode_state.num_speculative_tokens
        batch_size = max((len(b) for b in inputs.batches), default=0)
        return self._width_lookup[
            min(max(batch_size, 1), len(self._width_lookup) - 1)
        ]

    def _replay_batch_characteristics(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> BatchCharacteristics:
        """Computes the real (upper-bound) batch characteristics for replay.

        Mirrors the per-request cache / prompt-length computation in
        ``PagedKVCacheManager`` so the result is an upper bound on what
        ``runtime_inputs`` sees. For data parallelism the per-replica maxima are
        folded into one uniform shape, since every replica must replay the
        identical captured graph.
        """
        num_draft_tokens = self._kv_manager.params.num_draft_tokens
        batch_size = max((len(b) for b in inputs.batches), default=0)
        max_prompt_length = 0
        max_cache_valid_length = 0
        for ctx in inputs.flat_batch:
            max_prompt_length = max(
                max_prompt_length, prompt_tokens_for_context(ctx)
            )
            max_cache_valid_length = max(
                max_cache_valid_length,
                cache_valid_length_for_context(ctx, num_draft_tokens),
            )
        return BatchCharacteristics(
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            max_cache_valid_length=max_cache_valid_length,
        )

    def _run_forward(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
        draft_tokens: Buffer | None = None,
        sampling_buffers: _SpecDecodeSamplingBuffers | None = None,
        draft_tokens_np: npt.NDArray[np.int64] | None = None,
        draft_probs_full: Buffer | None = None,
    ) -> ModelOutputs:
        """Runs the forward pass for the provided inputs and returns the ModelOutputs.

        This handles both the non spec-decode and spec-decode paths. When running
        with spec-decode, you must provide the draft tokens even when there are
        no draft tokens to verify. In which case the shape is (batch_size, 0).

        Args:
            inputs: The text generation inputs.
            draft_tokens: Optional draft tokens for speculative decoding.
            sampling_buffers: Optional persistent sampling buffers for
                speculative decoding. Required when ``draft_tokens`` is set.
            draft_tokens_np: CPU view of the draft token buffer, used to
                compute the speculative bitmask for structured output just
                before graph replay. Required when ``draft_tokens`` is set
                and structured output is enabled.
            draft_probs_full: Distributions the draft sampled ``draft_tokens``
                from. Set iff ``draft_proposal="sampled"``.

        Returns:
            The model outputs containing logits and other inference results.
        """
        if draft_tokens is not None:
            assert self._spec_decode_state is not None
            num_draft_tokens_to_verify = draft_tokens.shape[1]
        else:
            assert self._spec_decode_state is None
            num_draft_tokens_to_verify = 0

        runner = self._graph_capture_runner
        batch_per_rank = max((len(b) for b in inputs.batches), default=0)
        use_graph_capture_replay = (
            runner is not None
            and bool(inputs)
            and inputs.batch_type == BatchType.TG
            and batch_per_rank <= self._max_graph_capture_batch_size
            and (draft_tokens is None or num_draft_tokens_to_verify > 0)
        )
        # The device graph synthesis version of `use_graph_capture_replay`.
        use_graph_synthesis = (
            self._synthesis_aligner is not None
            and bool(inputs)
            and inputs.batch_type == BatchType.TG
            and (draft_tokens is None or num_draft_tokens_to_verify > 0)
        )
        debug_verify_replay_enabled = (
            use_graph_capture_replay
            and self._pipeline_config.debug_verify_replay
        )
        debug_verify_model_inputs: ModelInputs | None = None

        # Prepare the batch.
        # Replay uses LUT buffers sized by max cache length so copied inputs
        # match captured graph buffer shapes.
        aligned_characteristics: BatchCharacteristics | None = None
        if use_graph_capture_replay:
            assert self._graph_capture_runner is not None
            # Align the batch's real (upper-bound) shape to a captured graph,
            # then prepare dispatch metadata once for the aligned cache length.
            real_characteristics = self._replay_batch_characteristics(inputs)
            aligned_characteristics = self._graph_capture_runner.align(
                real_characteristics
            )
            kv_cache_inputs = self._kv_manager.runtime_inputs(
                inputs.batches,
                max_cache_length=self._graph_capture_runner._max_cache_length_upper_bound,
                batch_characteristics=aligned_characteristics,
            )
        elif use_graph_synthesis:
            assert self._synthesis_aligner is not None
            # Mirror the use_graph_capture_replay path, substituting the
            # ServeGraphCaptureRunner for the SynthesisBucketAligner.

            # Align the batch's real (upper-bound) shape to a captured graph,
            # then prepare dispatch metadata once for the aligned cache length.
            # The caching infrastructure in the graph synthesis workflow
            # automatically re-uses the correct device graph.
            real_characteristics = self._replay_batch_characteristics(inputs)
            aligned_characteristics = self._synthesis_aligner.align(
                real_characteristics
            )
            kv_cache_inputs = self._kv_manager.runtime_inputs(
                inputs.batches,
                max_cache_length=self._synthesis_aligner.max_cache_length_upper_bound,
                batch_characteristics=aligned_characteristics,
            )
        else:
            kv_cache_inputs = self._kv_manager.runtime_inputs(inputs.batches)

        return_n_logits = (
            num_draft_tokens_to_verify + 1 if draft_tokens is not None else 0
        )

        with Tracer("prepare_initial_token_inputs"):
            model_inputs = self._pipeline_model.prepare_initial_token_inputs(
                replica_batches=inputs.batches,
                kv_cache_inputs=kv_cache_inputs,
                return_n_logits=return_n_logits,
            )

        if self._encoder_cache is not None:
            assert isinstance(self._pipeline_model, SupportsVisionEncoding)
            vision_result = self._encoder_cache.run_vision_encode(
                self._pipeline_model,
                as_vision_context_batches(inputs.batches),
                self._devices,
            )
            self._encoder_cache.finalize_vision_inputs(
                self._pipeline_model,
                model_inputs,
                self._devices,
                vision_result,
            )

        if debug_verify_replay_enabled:
            # Reuse non-KV buffers from replay inputs and only swap the
            # runtime-shaped KV inputs used for debug verification.
            debug_verify_model_inputs = copy.copy(model_inputs)
            debug_verify_model_inputs.update(
                kv_cache_inputs=self._kv_manager.runtime_inputs(inputs.batches)
            )

        if not isinstance(model_inputs, _HasRaggedTokens):
            raise RuntimeError(
                "OverlapTextGenerationPipeline requires model inputs with a "
                "Buffer `tokens` field."
            )
        if debug_verify_model_inputs is not None and not isinstance(
            debug_verify_model_inputs, _HasRaggedTokens
        ):
            raise RuntimeError(
                "OverlapTextGenerationPipeline requires debug-verify model "
                "inputs with a Buffer `tokens` field."
            )

        # Wrap the model inputs when speculative decoding is enabled.
        if draft_tokens is not None:
            assert self._spec_decode_state is not None
            assert isinstance(model_inputs, _UnifiedSpecDecodeInputs)
            model_inputs.draft_tokens = draft_tokens
            if draft_probs_full is not None:
                model_inputs.draft_probs_full = draft_probs_full
            assert sampling_buffers is not None
            model_inputs.temperature = sampling_buffers.temperature
            model_inputs.top_k = sampling_buffers.top_k
            model_inputs.max_k = sampling_buffers.max_k
            model_inputs.top_p = sampling_buffers.top_p
            model_inputs.min_top_p = sampling_buffers.min_top_p
            model_inputs.seed = sampling_buffers.seed
            model_inputs.in_thinking_phase = sampling_buffers.in_thinking_phase
        realized_draft_tokens_host: npt.NDArray[np.int64] | None = None
        if (
            self._prev_batch is not None
            and self._realize_future_token_processor is not None
            and not self._prev_batch._is_processed
        ):
            realized_draft_tokens_host = (
                self._realize_future_token_processor.realize_future_tokens(
                    prev_batch=self._prev_batch,
                    inputs=inputs,
                    model_inputs=model_inputs,
                    draft_tokens_np=draft_tokens_np,
                )
            )
            if debug_verify_model_inputs is not None:
                debug_verify_model_inputs.tokens = model_inputs.tokens
        # Compute speculative bitmasks here, as late as possible before graph
        # replay, so that all model-input preparation above can overlap with
        # the CUDA host callback computing the bitmask on the driver thread.
        # No CPU synchronize is needed: same-stream ordering guarantees the
        # callback's bool data is valid when the H2D copy DMA starts.
        #
        # Gate on ``overlap_state is not None`` (equivalently
        # ``pipeline_config.needs_bitmask_constraints``) rather than
        # ``structured_output.enabled``: the helper is enabled
        # whenever the tokenizer can build an FSM, but the bitmask
        # graph inputs / persistent buffers are only allocated when
        # constrained decoding can actually fire (user enabled the
        # feature flag OR a tool parser is configured). With
        # ``enabled=True`` but no overlap_state, the graph has no
        # bitmask inputs to bind and ``_assign_bitmask_inputs`` would
        # assert.
        if (
            draft_tokens is not None
            and self._structured_output.enabled
            and self._spec_decode_state is not None
            and self._spec_decode_state.overlap_state is not None
        ):
            assert draft_tokens_np is not None, (
                "draft_tokens_np must be provided when structured output is enabled"
            )
            self._assign_bitmask_inputs(
                model_inputs=model_inputs,
                context_batch=inputs.flat_batch,
                draft_tokens_np=draft_tokens_np,
                num_draft_tokens_to_verify=num_draft_tokens_to_verify,
                realized_draft_tokens_host=realized_draft_tokens_host,
            )

        # Execute the model and get next tokens.
        try:
            with Tracer("pipeline_model.execute"):
                if use_graph_capture_replay:
                    assert runner is not None
                    assert aligned_characteristics is not None
                    return runner.replay(
                        model_inputs=model_inputs,
                        batch_characteristics=aligned_characteristics,
                        debug_verify_replay=debug_verify_replay_enabled,
                        debug_verify_model_inputs=debug_verify_model_inputs,
                    )

                return self._pipeline_model.execute(model_inputs=model_inputs)
        except Exception:
            batch_size = len(inputs.flat_batch)
            cache_tokens = sum(
                ctx.tokens.processed_length for ctx in inputs.flat_batch
            )
            input_tokens = sum(
                ctx.tokens.active_length for ctx in inputs.flat_batch
            )
            logger.error(
                "Encountered an exception while executing batch: "
                f"{batch_size=:}, {cache_tokens=:}, {input_tokens=:}"
            )
            raise  # re-raise the original exception

    def _create_sampling_processor(
        self,
        flat_batch: list[TextGenerationContextType],
    ) -> tuple[FusedSamplingProcessor, npt.NDArray[np.int32] | None]:
        """Creates a sampling processor for the given batch.

        Args:
            flat_batch: The flattened batch of generation contexts.

        Returns:
            A tuple of (sampling_processor, bitmask). The bitmask is None when
            structured output is not enabled for any request in the batch.
        """
        device0 = self._devices[0]
        assert not device0.is_host

        # Check for structured output - use appropriate sampler
        bitmask = self.initialize_bitmask(flat_batch)
        has_structured_output = bitmask is not None

        if has_structured_output:
            assert bitmask is not None  # for linter
            # Initialize per-context bitmask state for structured output
            for i, ctx in enumerate(flat_batch):
                self.update_for_structured_output(ctx, bitmask, i)

            assert self._sampler_with_bitmask is not None
            with Tracer("fused_sampling_processor_w_bitmask"):
                sampling_processor = FusedSamplingProcessor(
                    sampler=self._sampler_with_bitmask,
                    pipeline_config=self._pipeline_config,
                    context_batch=flat_batch,
                    device=self._sampler_device,
                    pinned_new_tokens=self._pinned_new_tokens,
                    identity_logit_offsets=self._identity_logit_offsets,
                    bitmask=bitmask,
                    vocab_size=self.vocab_size,
                )
        else:
            assert self._sampler_without_bitmask is not None
            with Tracer("fused_sampling_processor"):
                sampling_processor = FusedSamplingProcessor(
                    sampler=self._sampler_without_bitmask,
                    pipeline_config=self._pipeline_config,
                    context_batch=flat_batch,
                    device=self._sampler_device,
                    pinned_new_tokens=self._pinned_new_tokens,
                    identity_logit_offsets=self._identity_logit_offsets,
                )

        return sampling_processor, bitmask

    def _sample_logits(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
        model_outputs: ModelOutputs,
        sampling_processor: FusedSamplingProcessor,
    ) -> AsyncBatch[TextGenerationContextType]:
        """Applies logits processors, samples tokens, and returns an AsyncBatch.

        Args:
            inputs: The text generation inputs.
            model_outputs: The model outputs containing logits.
            sampling_processor: The sampling processor to use.

        Returns:
            An AsyncBatch with the sampled tokens and copy event.
        """
        device0 = self._devices[0]
        flat_batch = inputs.flat_batch

        if model_outputs.logit_offsets is None:
            batch_size = len(flat_batch)
            logits_batch = int(model_outputs.logits.shape[0])
            if logits_batch != batch_size:
                raise AssertionError(
                    "Model returned LAST_TOKEN logits with a leading dimension "
                    f"that does not match request batch size: logits.shape[0]={logits_batch}, "
                    f"batch_size={batch_size}, input_tokens={sum(ctx.tokens.active_length for ctx in flat_batch)}, "
                    f"active_lengths={[ctx.tokens.active_length for ctx in flat_batch]}, "
                    f"generated_lengths={[ctx.tokens.generated_length for ctx in flat_batch]}."
                )

        with Tracer("apply_logits_processors"):
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
                batch_processors=[sampling_processor],
            )
        generated_tokens = sampling_processor.generated_tokens
        # [B, 1] -> [B]
        generated_tokens = generated_tokens.view(
            dtype=generated_tokens.dtype,
            shape=(generated_tokens.shape[0],),
        )

        with Tracer("D2H generated_tokens"):
            if self._sampler_device.is_host:
                generated_tokens_host = generated_tokens
                generated_tokens_device = generated_tokens.to(device0)
            else:
                generated_tokens_device = generated_tokens
                # Reuse a persistent pinned host buffer for the async D2H instead
                # of page-locking a fresh one every decode step (this allocation
                # was exposed host time on the per-step critical path). The
                # overlap scheduler reads the previous batch's copy in
                # sync_and_process_outputs strictly before _sample_logits writes
                # the next batch's copy, so a single reused buffer is race-free.
                d2h_batch = int(generated_tokens_device.shape[0])
                pinned = self._pinned_generated_tokens_host
                if (
                    pinned is None
                    or int(pinned.shape[0]) < d2h_batch
                    or pinned.dtype != generated_tokens_device.dtype
                ):
                    pinned = DevicePinnedBuffer(
                        shape=(max(self._max_batch_size, d2h_batch),),
                        dtype=generated_tokens_device.dtype,
                        device=device0,
                    )
                    self._pinned_generated_tokens_host = pinned
                generated_tokens_host = pinned[:d2h_batch]
                generated_tokens_host.inplace_copy_from(generated_tokens_device)
            # Record an event to track the completion of the copy. This ensures
            # the subsequent synchronize() call blocks until the copy is
            # complete, and no more.
            copy_event = device0.default_queue.record_event()

        # Make a deep copy of the input object in case the caller modifies it!
        cloned_inputs = TextGenerationInputs(
            batches=[
                [ctx for ctx in replica_batch]
                for replica_batch in inputs.batches
            ],
        )

        return AsyncBatch(
            inputs=cloned_inputs,
            generated_tokens_device=generated_tokens_device,
            generated_tokens_host=generated_tokens_host,
            copy_event=copy_event,
            # Always use realize_future_token() to overwrite placeholder tokens.
            # For structured output, the FSM is advanced separately in
            # sync_and_process_outputs() with the real token.
            overwrite_future=True,
            structured_output=self._structured_output,
        )

    def _assign_bitmask_inputs(
        self,
        model_inputs: Any,
        context_batch: list[TextGenerationContextType],
        draft_tokens_np: npt.NDArray[np.int64],
        num_draft_tokens_to_verify: int,
        realized_draft_tokens_host: npt.NDArray[np.int64] | None = None,
    ) -> None:
        """Populate the structured-output bitmask graph inputs.

        Binds the (pinned, wait_payload, device_scratch) triple on
        ``model_inputs`` from :class:`StructuredOutputOverlapState`.

        Steady state (a callback ran): the async callback enqueued at the head
        of :meth:`execute` is the **sole writer** of the ``[0, batch)`` bitmask
        rectangle. It advanced the producing batch's FSM and wrote every
        consumer row directly in this batch's row order -- resetting any row it
        cannot attribute to -1 -- then signalled the completion flag. This
        method therefore performs no synchronous bitmask fill; it only binds the
        graph-input views. With a single writer there is no main-thread write to
        race the in-flight callback, and the model graph consumes the bitmask in
        place with no device gather and no host wait.

        Cold start (no callback ran -- prefill->first-decode, or the first
        iteration after a non-verify batch): every row is computed
        synchronously via :meth:`StructuredOutputOverlapState.prime`, which
        writes rows ``[0, batch)`` and signals the flag so the first replay's
        wait passes immediately. This is the only path that fills newly-admitted
        rows; the scheduler routes every fresh or resumed request through it
        (such a request has ``generated_length == 0``, so the batch does not
        verify drafts and the callback is left unsent).
        """
        assert self._spec_decode_state is not None
        overlap_state = self._spec_decode_state.overlap_state
        assert overlap_state is not None, (
            "_assign_bitmask_inputs requires structured output to be enabled"
        )

        spec_state = self._spec_decode_state
        batch_size = len(context_batch)
        # One position per verified draft plus the bonus slot. This must be the
        # width this step actually verifies.
        # When num_draft_tokens_to_verify == 0 (prefill->decode boundary),
        # compute_speculative_bitmasks writes only slot 0 and leaves
        # trailing slots unconstrained (all bits set, i.e. -1 in the
        # packed int32 bitmask).
        num_positions = num_draft_tokens_to_verify + 1

        callback_available = spec_state.has_precomputed_bitmask
        spec_state.has_precomputed_bitmask = False

        if not callback_available:
            # Cold start: no callback advanced the FSM (prefill->first-decode,
            # or first iter after a non-verify batch). Compute every row
            # synchronously in this batch's order; prime writes rows
            # [0, batch) and signals the flag. Build the bitmask from real
            # drafts -- the realized host mirror of draft tokens, not MAGIC
            # placeholders. ``compute_speculative_bitmasks`` also initialises
            # ctx.matcher for any fresh constrained row, so this is the path
            # that admits new and resumed requests.
            drafts = (
                realized_draft_tokens_host
                if realized_draft_tokens_host is not None
                else draft_tokens_np
            )
            bitmask_np = self._structured_output.compute_speculative_bitmasks(
                context_batch=context_batch,
                draft_tokens=drafts,
                num_positions=num_positions,
            )
            overlap_state.prime(bitmask_np)
        else:
            # Steady state: the head-of-execute callback already wrote every
            # row -- it is the sole writer of the [0, batch) rectangle (advanced
            # the producing batch's FSM, wrote each consumer row in this batch's
            # order, signalled the flag). Nothing to fill here; the binding
            # below is all that remains. With no second writer the pinned buffer
            # the in-graph H2D reads is never raced.
            pass

        # Bind the graph inputs via the cached-view helper so the same Buffer
        # objects are passed at warmup capture and every replay —
        # GraphCaptureRunner.replay's preface inplace_copy_from short-circuits
        # via the self-is-src identity check, avoiding a real device memcpy.
        pinned_view, scratch_view = overlap_state.get_input_views(
            batch_size, num_positions
        )
        model_inputs.pinned_bitmask = pinned_view
        model_inputs.wait_payload = overlap_state.wait_payload
        model_inputs.device_bitmask_scratch = scratch_view

    def _build_bitmask_callback(
        self,
        context_batch: list[TextGenerationContextType],
        output_context_batch: list[TextGenerationContextType],
        bonus_tokens_np: npt.NDArray[np.int64],
        num_accepted_np: npt.NDArray[np.int64],
        accepted_draft_tokens_np: npt.NDArray[np.int64],
        next_draft_tokens_np: npt.NDArray[np.int64],
        overlap_pinned_np: npt.NDArray[np.int32],
    ) -> Callable[[], None]:
        """Build a callback closure that advances FSM then computes bitmasks.

        Snapshots each producing row's token history before returning, while
        still on the main thread. The worker must not read ctx.tokens itself:
        the enqueuing execute() goes on to realize this batch's committed span
        into that buffer while the worker runs. Capturing here rather than at
        the call site is what keeps the snapshots paired with context_batch.

        All numpy arrays must be views into persistent pinned buffers owned by
        SpecDecodeState / StructuredOutputOverlapState (or plain copies for
        CPU-only data). Views are safe because the owning state objects
        outlive every callback invocation, so freeing a DLPack view inside
        the CUDA host callback never drives the owning DevicePinnedBuffer's
        refcount to zero, and cuMemFreeHost is never called from within the
        callback.

        Args:
            context_batch: Generation contexts of the producing batch (the one
                whose FSM is advanced). Indexes the token arrays below.
            output_context_batch: Generation contexts of the consuming batch,
                in its logits row order. The bitmask is written in this order,
                so the model graph consumes it without a device gather. Equals
                ``context_batch`` only when the batch did not change.
            bonus_tokens_np: Bonus tokens array, shape [batch].
            num_accepted_np: Accepted draft token counts, shape [batch].
            accepted_draft_tokens_np: Draft tokens verified, shape [batch, K].
            next_draft_tokens_np: Draft tokens for next batch, shape [batch, K].
            overlap_pinned_np: Packed int32 bitmask view aliasing the leading
                rows of :attr:`StructuredOutputOverlapState.pinned_bitmask`,
                shape [out_batch, K+1, packed_vocab]. The callback writes the
                packed FSM bitmask here directly in the consuming batch's row
                order; the next iter's in-graph H2D copies it to device, where
                the GPU acceptance sampler unpacks and applies it in one fused
                pass. ``advance_fsm_and_compute_bitmasks`` owns the whole
                rectangle: it resets every row to -1 (all valid) before filling
                the continuing rows, so no row is ever left stale and there is
                no second writer on the main thread.

        Returns:
            A zero-argument callable for use with
            ``Device.__unsafe_enqueue_async_py_host_func``.
        """
        structured_output = self._structured_output

        with Tracer("capture_committed_span_snapshots"):
            committed_span_snapshots = CommittedSpanSnapshot.capture(
                context_batch
            )

        def callback() -> None:
            try:
                # Write the packed int32 FSM bitmask straight into the pinned
                # buffer the next iter's in-graph H2D reads, in the consuming
                # batch's row order. The GPU acceptance sampler unpacks and
                # applies it (apply_packed_bitmask), so the callback no longer
                # unpacks on the CPU -- this removes the (benchmarked)
                # ~600-800us per-step unpack that previously ran here.
                structured_output.advance_fsm_and_compute_bitmasks(
                    context_batch=context_batch,
                    accepted_draft_tokens=accepted_draft_tokens_np,
                    num_accepted=num_accepted_np,
                    bonus_tokens=bonus_tokens_np,
                    next_draft_tokens=next_draft_tokens_np,
                    bitmask_out=overlap_pinned_np,
                    output_context_batch=output_context_batch,
                    committed_span_snapshots=committed_span_snapshots,
                )
            except Exception as e:
                logger.error(
                    "Async bitmask callback failed: %s", e, exc_info=True
                )
                # Trampoline auto-signals the flag on exception, but the
                # pinned buffer could be partially written. All-valid
                # (-1 = all bits set = unconstrained) is the safest
                # fallback: the model still produces a token, generation
                # makes forward progress, and the grammar will re-converge
                # on the next iter.
                #
                # The callback is the sole writer of this [:curr_batch_size]
                # rectangle -- the synchronous new-admission fill was removed
                # when bitmask preparation was consolidated here -- so resetting
                # the whole view races no main-thread write. (``overlap_pinned_np``
                # already aliases exactly the [:curr_batch_size] consumer rows.)
                try:
                    overlap_pinned_np[:] = -1
                except Exception:
                    pass

        return callback

    @traced
    def _enqueue_prev_bitmask_callback(
        self,
        curr_context_batch: list[TextGenerationContextType],
        curr_verify_width: int,
    ) -> bool:
        """Enqueue the previous batch's FSM-advance + in-order bitmask callback.

        Runs at the head of :meth:`execute`, once this iteration's batch (and
        therefore the consumer logits-row order, ``curr_context_batch``) is
        known. The async callback advances the producing (previous) batch's
        FSM through its committed tokens and writes the packed bitmask for
        every row **directly in ``curr_context_batch`` order**, so the model
        graph consumes it without a device gather. The callback is the sole
        writer of the bitmask rectangle on this path: it is only enqueued when
        the whole current batch verifies drafts (so every row continues from
        the previous batch), and :meth:`_assign_bitmask_inputs` performs no
        synchronous fill when it runs.

        The producing batch's committed/draft tokens are read from the
        persistent pinned buffers, which still hold its D2H output (this
        iteration's D2H runs later, in ``_execute_spec_decode``). The kickoff
        trampoline lands on the device default stream after that D2H, so the
        worker observes complete data; it signals the completion flag the next
        iter's in-graph ``mo.wait_host_value_with_dep`` gates the H2D on. The
        FSM advance + bitmask compute run on a separate AsyncRT worker, so they
        overlap this iteration's target forward.

        On success the producing batch's ``fsm_advanced_by_callback`` is set
        (its later sync skips the now-redundant FSM advance) and
        ``has_precomputed_bitmask`` is primed for
        :meth:`_assign_bitmask_inputs`.

        Returns early without enqueuing when there is no previous batch, when
        the previous batch did not verify drafts (a prefill / mixed batch has
        no committed tokens to advance through -- its successor cold-starts via
        prime instead), or when structured output is not configured.

        Args:
            curr_context_batch: This iteration's contexts, in logits row order.
            curr_verify_width: How many drafts this iteration will verify. The
                bitmask carries one position per verified draft plus the bonus
                slot, and this iteration is the consumer.

        Returns:
            True if the callback was enqueued.
        """
        if not self._pipeline_config.needs_bitmask_constraints:
            return False

        spec_state = self._spec_decode_state
        if (
            spec_state is None
            or spec_state.persistent_bonus_tokens_pinned is None
            or spec_state.persistent_num_accepted_pinned is None
            or spec_state.accepted_token_pinned is None
            or spec_state.persistent_next_draft_tokens_pinned is None
        ):
            return False

        prev_batch = self._prev_batch
        if prev_batch is None or prev_batch.spec_decode is None:
            return False

        # Only a verify (decode) batch produced committed tokens to advance the
        # FSM through. After a prefill / mixed batch (no draft verification),
        # there is nothing to advance; its successor cold-starts the bitmask
        # via prime() in _assign_bitmask_inputs, and the prefill batch's own
        # FSM advance happens on its (early-)sync path.
        if not self._prev_batch_verified_drafts():
            return False
        prev_num_draft_tokens_to_verify = (
            prev_batch.spec_decode.num_draft_tokens_to_verify
        )

        # Only enqueue when THIS iteration will also verify drafts -- the
        # steady decode path that consumes the callback's bitmask in place.
        # If the current batch does not verify (a fresh prefill joined, making
        # it a mixed / prefill batch), ``_execute_spec_decode`` clears
        # ``has_precomputed_bitmask`` and ``_assign_bitmask_inputs`` cold-starts
        # via ``prime`` -- which writes the pinned buffer and signals the flag.
        # Enqueuing here too would double-write the buffer and double-signal the
        # flag against this enqueue's trampoline reset. Instead leave the
        # callback unsent: ``fsm_advanced_by_callback`` stays False, the
        # early-sync guard advances the previous batch's FSM, and the cold-start
        # prime owns the bitmask. (Matches the predicate in
        # ``_should_early_sync_prev_batch``.)
        #
        # Cost of this fallback: when a mixed batch recurs, the cold-start
        # prime recomputes every row's bitmask synchronously, including the
        # continuing constrained rows the callback could have produced
        # off-thread. The scheduler does not emit mixed batches today for
        # aggregated mode, and on a disaggregated decode-only engine a
        # KV-transferred row arrives with generated_length > 0 yet was never
        # in this engine's producing batch -- so the generated_length proxy is
        # insufficient. Once mixed batches are benchmarkable, preserving
        # overlap here means enqueuing the callback for just the continuing
        # subset and cold-starting only the genuinely new rows -- deferred
        # until then to avoid splitting the prime/callback bitmask ownership
        # (and the flag signalling) on a path that cannot yet be exercised or
        # benchmarked.
        prev_context_batch = prev_batch.inputs.flat_batch
        prev_rids = {ctx.request_id for ctx in prev_context_batch}
        all_continuing = all(
            (
                (ctx.request_id in prev_rids and not ctx.is_initial_prompt)
                or ctx._is_padding_ctx  # Padding contexts shouldn't affect continuation
            )
            for ctx in curr_context_batch
        )
        if not all_continuing:
            return False

        overlap_state = spec_state.overlap_state
        assert overlap_state is not None, (
            "Async bitmask callback requires structured output to be enabled"
        )

        prev_batch_size = len(prev_context_batch)
        next_draft_k = prev_batch.spec_decode.next_draft_tokens_host.shape[1]
        # The bitmask is for the consuming step, so its width is how many
        # drafts that step will verify.
        num_positions = curr_verify_width + 1
        curr_batch_size = len(curr_context_batch)

        # Capture BEFORE enqueue: capture numpy views into the persistent pinned
        # buffers so the closure binds live data. Use DevicePinnedBuffer.to_numpy()
        # not Buffer.to_numpy() — the latter may synchronize on a view/slice.
        with Tracer("convert_buffers_to_np_views"):
            assert spec_state.persistent_bonus_tokens_pinned is not None
            assert spec_state.persistent_num_accepted_pinned is not None
            assert spec_state.accepted_token_pinned is not None
            assert spec_state.persistent_next_draft_tokens_pinned is not None
            bonus_tokens_np = (
                spec_state.persistent_bonus_tokens_pinned.to_numpy()[
                    :prev_batch_size
                ]
            )
            num_accepted_np = (
                spec_state.persistent_num_accepted_pinned.to_numpy()[
                    :prev_batch_size
                ]
            )
            # Reshape in numpy, not in buffer views: the D2H packed rows
            # tightly at the step's verify width, which is narrower than the
            # buffer's own stride. Taking that prefix with Buffer.view/slice
            # would drop back to a plain Buffer, whose to_numpy() can trigger
            # the stream sync this callback exists to avoid.
            accepted_draft_tokens_np = (
                spec_state.accepted_token_pinned.to_numpy()
                .reshape(-1)[
                    : prev_batch_size * prev_num_draft_tokens_to_verify
                ]
                .reshape(prev_batch_size, prev_num_draft_tokens_to_verify)
            )
            next_draft_tokens_np = (
                spec_state.persistent_next_draft_tokens_pinned.to_numpy()[
                    :prev_batch_size, :next_draft_k
                ]
            )

        # View the leading consumer rows of the persistent pinned bitmask.
        # The worker is the sole writer of this [:curr_batch_size] rectangle: it
        # writes every row in curr order (every row continues from the producing
        # batch on this path). The captured graph reads the whole rectangle,
        # gated on the worker's release-store of the flag. Same lifetime
        # guarantees as the other captured views: the underlying
        # DevicePinnedBuffer outlives every callback invocation.
        overlap_pinned_np = overlap_state.pinned_for(num_positions).to_numpy()[
            :curr_batch_size, :num_positions, :
        ]

        with Tracer("build_bitmask_callback"):
            callback = self._build_bitmask_callback(
                context_batch=prev_context_batch,
                output_context_batch=curr_context_batch,
                bonus_tokens_np=bonus_tokens_np,
                num_accepted_np=num_accepted_np,
                accepted_draft_tokens_np=accepted_draft_tokens_np,
                next_draft_tokens_np=next_draft_tokens_np,
                overlap_pinned_np=overlap_pinned_np,
            )

        # The trampoline + worker dispatch goes on the device default stream.
        # The trampoline's flag.reset() is therefore naturally ordered against
        # this iter's captured-graph wait (same stream), so the wait cannot
        # observe a stale 1 from a prior prime / worker signal. The trampoline
        # body is microseconds (atomic store + heap alloc + MLRT::addTask +
        # return); the slow FSM advance + bitmask compute run inside fn on an
        # AsyncRT worker off-stream and signal the flag on completion, so the
        # overlap with the target forward is preserved.
        overlap_state.enqueue_async_callback(callback)

        # The callback advances the producing batch's FSM, so its later sync
        # must skip the now-redundant advance.
        prev_batch.spec_decode.fsm_advanced_by_callback = True
        spec_state.has_precomputed_bitmask = True
        return True

    def _d2h_spec_decode_outputs(
        self,
        outputs: UnifiedEagleOutputs,
        draft_tokens_device: Buffer,
        batch_size: int,
        num_draft_tokens_to_verify: int,
        next_draft_k: int,
    ) -> _AsyncSpecDecodeHostBuffers:
        """D2H copy spec-decode model outputs to host.

        Two D2H destinations are populated when structured output is active:

        1. Persistent pinned buffers on SpecDecodeState (read by the async
           bitmask callback). Views into persistent memory are safe to release
           on the CUDA driver thread because the owning DevicePinnedBuffers
           live for the pipeline's lifetime, so DLPack teardown never calls
           `cuMemFreeHost` from a host callback.

        2. Fresh per-batch DevicePinnedBuffers (read by the sync path on the
           next iteration). The persistent buffers cannot be reused for the
           sync path because `_execute_spec_decode(N+1)` queues N+1's D2H
           into them BEFORE this iteration's sync path runs — by the time
           the sync path reads, the persistent buffers contain N+1's data,
           not N's.

        Both D2H ops are enqueued on the same CUDA stream before `copy_event`
        is recorded, so `copy_event.synchronize()` in the sync path
        guarantees both copies are complete before any read.

        When structured output is disabled, only the per-batch buffers are
        populated (the persistent buffers may be None).

        Returns the fresh per-batch buffers for use in AsyncSpecDecodeBatch.
        """
        device0 = self._devices[0]
        num_accepted_draft_tokens_device = outputs.num_accepted_draft_tokens
        next_tokens_device = outputs.next_tokens
        next_draft_tokens_device = outputs.next_draft_tokens

        spec_state = self._spec_decode_state
        if (
            spec_state is not None
            and spec_state.persistent_bonus_tokens_pinned is not None
            and spec_state.persistent_num_accepted_pinned is not None
            and spec_state.persistent_next_draft_tokens_pinned is not None
            and spec_state.accepted_token_pinned is not None
        ):
            # D2H into persistent pinned buffers. The callback reads numpy
            # views from these directly (via DevicePinnedBuffer.to_numpy()),
            # which avoids the stream sync that Buffer.to_numpy() on a
            # view/slice can trigger.
            _contiguous_prefix_2d(
                spec_state.persistent_num_accepted_pinned,
                batch_size,
                1,
            ).view(DType.int64, (batch_size,)).inplace_copy_from(
                num_accepted_draft_tokens_device
            )
            _contiguous_prefix_2d(
                spec_state.persistent_bonus_tokens_pinned,
                batch_size,
                1,
            ).view(DType.int64, (batch_size,)).inplace_copy_from(
                next_tokens_device
            )
            _contiguous_prefix_2d(
                spec_state.persistent_next_draft_tokens_pinned,
                batch_size,
                next_draft_k,
            ).inplace_copy_from(next_draft_tokens_device)
            # A prefill->decode boundary step verifies nothing, so there is
            # no accepted array to mirror.
            if num_draft_tokens_to_verify > 0:
                _contiguous_prefix_2d(
                    spec_state.accepted_token_pinned,
                    batch_size,
                    num_draft_tokens_to_verify,
                ).inplace_copy_from(draft_tokens_device)

        # Fresh per-batch allocations for the sync path — immune to the next
        # batch's writes into the persistent buffers above.
        num_accepted_draft_tokens_host = DevicePinnedBuffer(
            shape=num_accepted_draft_tokens_device.shape,
            dtype=num_accepted_draft_tokens_device.dtype,
            device=device0,
        )
        num_accepted_draft_tokens_host.inplace_copy_from(
            num_accepted_draft_tokens_device
        )
        next_tokens_host = DevicePinnedBuffer(
            shape=next_tokens_device.shape,
            dtype=next_tokens_device.dtype,
            device=device0,
        )
        next_tokens_host.inplace_copy_from(next_tokens_device)
        next_draft_tokens_host = DevicePinnedBuffer(
            shape=next_draft_tokens_device.shape,
            dtype=next_draft_tokens_device.dtype,
            device=device0,
        )
        next_draft_tokens_host.inplace_copy_from(next_draft_tokens_device)

        return _AsyncSpecDecodeHostBuffers(
            num_accepted_draft_tokens_host=num_accepted_draft_tokens_host,
            next_tokens_host=next_tokens_host,
            next_draft_tokens_host=next_draft_tokens_host,
        )

    @traced
    def _execute_spec_decode(
        self, inputs: TextGenerationInputs[TextGenerationContextType]
    ) -> AsyncBatch[TextGenerationContextType]:
        """Executes unified EAGLE speculative decoding.

        Single graph call handles: merge, target forward, greedy rejection,
        shift, and draft forward.
        """
        assert self._spec_decode_state is not None

        context_batch = inputs.flat_batch
        num_draft_tokens_to_verify = self._verify_width(inputs)
        verify_draft_tokens = num_draft_tokens_to_verify > 0

        # The bitmask callback is only ever enqueued for decode batches
        # (verify_draft_tokens=True). Any has_precomputed_bitmask=True
        # seen here when verify_draft_tokens=False was set by a
        # previous request's decode callback and is stale for this
        # batch. Clearing it forces ``_assign_bitmask_inputs`` onto
        # the sync-prime path, which initialises ctx.matcher for new
        # contexts and computes the bitmask from the correct initial
        # FSM state.
        # Delete the saved draft tokens if we are not verifying them.
        if not verify_draft_tokens:
            self._spec_decode_state.has_precomputed_bitmask = False
            for ctx in context_batch:
                if len(ctx.spec_decoding_state.draft_tokens_to_verify):
                    ctx.spec_decoding_state.draft_tokens_to_verify = []

        # Load or create draft tokens.
        draft_tokens_pinned = DevicePinnedBuffer(
            shape=(len(context_batch), num_draft_tokens_to_verify),
            dtype=DType.int64,
            device=self._devices[0],
        )
        draft_tokens_np = draft_tokens_pinned.to_numpy()
        if num_draft_tokens_to_verify:
            for i, ctx in enumerate(context_batch):
                # If there are no draft_tokens to verify, populate it with a
                # arbitrary token value. This is to trigger token verification
                # more often. When we do not verify tokens, we cannot replay cuda
                # graph which hurts perf.
                if not ctx.spec_decoding_state.draft_tokens_to_verify:
                    ctx.spec_decoding_state.draft_tokens_to_verify = [
                        MAGIC_DRAFT_TOKEN_ID
                    ] * num_draft_tokens_to_verify
                tokens = ctx.spec_decoding_state.draft_tokens_to_verify
                if len(tokens) > num_draft_tokens_to_verify:
                    tokens = tokens[:num_draft_tokens_to_verify]
                    ctx.spec_decoding_state.draft_tokens_to_verify = tokens
                assert len(tokens) == num_draft_tokens_to_verify
                draft_tokens_np[i, :] = tokens

        draft_tokens_device = self._spec_decode_state.persistent_draft_tokens
        draft_tokens_device = _contiguous_prefix_2d(
            draft_tokens_device, len(context_batch), num_draft_tokens_to_verify
        )
        draft_tokens_device.inplace_copy_from(draft_tokens_pinned)

        draft_probs_full_device = self._prepare_draft_probs_full(
            len(context_batch), num_draft_tokens_to_verify
        )

        sampling_buffers = self._build_spec_decode_sampling_buffers(
            context_batch
        )

        outputs = self._run_forward(
            inputs,
            draft_tokens=draft_tokens_device,
            sampling_buffers=sampling_buffers,
            draft_tokens_np=draft_tokens_np,
            draft_probs_full=draft_probs_full_device,
        )
        assert isinstance(outputs, UnifiedEagleOutputs)

        draft_tokens_pinned.inplace_copy_from(draft_tokens_device)

        # Do the copy to host for each model output using pinned memory.
        with Tracer("D2H generated_tokens"):
            device0 = self._devices[0]
            batch_size = len(context_batch)
            num_accepted_draft_tokens_device = outputs.num_accepted_draft_tokens
            next_tokens_device = outputs.next_tokens
            next_draft_tokens_device = outputs.next_draft_tokens
            next_draft_k = next_draft_tokens_device.shape[1]

            host_buffers = self._d2h_spec_decode_outputs(
                outputs=outputs,
                draft_tokens_device=draft_tokens_device,
                batch_size=batch_size,
                num_draft_tokens_to_verify=num_draft_tokens_to_verify,
                next_draft_k=next_draft_k,
            )
            num_accepted_draft_tokens_host = (
                host_buffers.num_accepted_draft_tokens_host
            )
            next_tokens_host = host_buffers.next_tokens_host
            next_draft_tokens_host = host_buffers.next_draft_tokens_host

            # Record an event to track the completion of the d2h copies.
            # This will ensure that the subsequent synchronize() call will
            # block until the d2h copy is complete, and no more.
            copy_event = device0.default_queue.record_event()

            # The FSM-advance + in-order bitmask callback for THIS batch is not
            # enqueued here. It is enqueued at the head of the NEXT execute()
            # call (``_enqueue_prev_bitmask_callback``), once that iteration's
            # row order is known, so the bitmask is written in the consuming
            # batch's order and the model graph needs no device gather. The
            # D2H above lands in the persistent pinned buffers the callback
            # reads; ``fsm_advanced_by_callback`` starts False and is flipped
            # to True by that next-iter enqueue.

            async_batch = AsyncBatch(
                inputs=inputs,
                generated_tokens_device=next_tokens_device,
                generated_tokens_host=next_tokens_host,
                copy_event=copy_event,
                spec_decode=AsyncSpecDecodeBatch(
                    draft_tokens_to_verify_device=draft_tokens_device,
                    draft_tokens_to_verify_host=draft_tokens_pinned,
                    next_draft_tokens_device=next_draft_tokens_device,
                    next_draft_tokens_host=next_draft_tokens_host,
                    num_accepted_draft_tokens_device=num_accepted_draft_tokens_device,
                    num_accepted_draft_tokens_host=num_accepted_draft_tokens_host,
                    max_seq_len=self._pipeline_model.max_seq_len,
                    fsm_advanced_by_callback=False,
                    next_draft_probs_full_device=outputs.next_draft_probs_full,
                ),
                think_start_token_id=(
                    self._think_start_token_id
                    if self._think_start_token_id >= 0
                    else None
                ),
                think_end_token_id=(
                    self._think_end_token_id
                    if self._think_end_token_id >= 0
                    else None
                ),
            )

        return async_batch

    def _prev_batch_verified_drafts(self) -> bool:
        """Return True iff the previous batch ran a verify (decode) step.

        A verify batch has ``spec_decode.num_draft_tokens_to_verify > 0``.
        Prefill and mixed batches have zero and are excluded.
        """
        return (
            self._prev_batch is not None
            and self._prev_batch.spec_decode is not None
            and self._prev_batch.spec_decode.num_draft_tokens_to_verify > 0
        )

    def _should_early_sync_prev_batch(self) -> bool:
        """Return True iff the previous batch must be early-synced.

        Checked at the head of `execute`, just after
        `_enqueue_prev_bitmask_callback` has had its chance to advance the
        previous batch's FSM via the async callback.

        Fires whenever no async callback advanced the previous batch's FSM,
        i.e. `fsm_advanced_by_callback` is still False after
        `_enqueue_prev_bitmask_callback` ran. With structured output enabled
        and a previous batch present, that is the case when:

          * the previous batch did not verify drafts (a prefill / mixed
            previous batch has no committed tokens to advance through), or
          * the current batch does not verify (a fresh prefill joined, making
            it a mixed batch): the callback is left unsent so it cannot
            double-write the cold-start prime path, or
          * the persistent pinned spec-decode buffers are not yet allocated.

        In all of these the previous batch's FSM is still un-advanced. Syncing
        it here advances its FSM before this iteration's bitmask compute
        (`_assign_bitmask_inputs`, cold-start prime path) reads the matchers,
        and before any new callback could touch them. Concurrent unsynchronized
        matcher access produces "doesn't satisfy the grammar" errors and state
        corruption.

        When a callback did advance the previous batch (the steady
        decode→decode path, both batches verifying), it has
        `fsm_advanced_by_callback=True` (set by
        `_enqueue_prev_bitmask_callback` just above), so its sync path never
        advances `ctx.matcher` and full overlap is preserved — the guard does
        not fire for it.

        Gated on `needs_bitmask_constraints`: when structured output is off, no
        callback is ever enqueued, so `fsm_advanced_by_callback` is always
        False. Without this gate the guard would fire every decode step.

        IMPORTANT: even when this returns True, `_prev_batch` is NOT cleared
        by the caller. `_run_forward` needs it so `realize_future_tokens` can
        scatter the previous batch's GPU-side EAGLE draft tokens into the
        current batch's `draft_tokens` input. Clearing `_prev_batch` would
        leave MAGIC placeholder tokens (42) in `model_inputs.draft_tokens`,
        which the EAGLE model would treat as real context — its next-draft
        predictions become garbage, propagating through every subsequent
        `realize_future_tokens` call and keeping EAGLE acceptance near zero
        for the entire generation (apparent hang). The early-sync result is
        instead saved so the normal sync path below can skip re-syncing the
        same batch.
        """
        return (
            self._pipeline_config.needs_bitmask_constraints
            and self._prev_batch is not None
            and self._prev_batch.spec_decode is not None
            and not self._prev_batch.spec_decode.fsm_advanced_by_callback
        )

    @traced
    def execute(
        self,
        inputs: TextGenerationInputs[TextGenerationContextType],
    ) -> PipelineOutputsDict[TextGenerationOutput]:
        """Executes a batch of requests asynchronously on the GPU.

        This method returns before the outputs for the current batch are
        ready, so the outputs it returns belong to the *previous* batch. To
        drain the outputs for the final batch, call ``execute()`` again with an
        empty batch.

        The batch of requests is a
        :class:`~max.pipelines.modeling.types.TextGenerationInputs`, which wraps
        one or more :class:`~max.pipelines.context.TextContext` objects. Build a
        batch on CPU like this:

        .. code-block:: python

            import numpy as np
            from max.pipelines.context import TextContext, TokenBuffer
            from max.pipelines.modeling.types import (
                RequestID,
                TextGenerationInputs,
            )

            contexts = [
                TextContext(
                    request_id=RequestID(),
                    max_length=32,
                    tokens=TokenBuffer(np.arange(8, dtype=np.int64)),
                )
                for _ in range(4)
            ]
            inputs = TextGenerationInputs(batches=[contexts])
            empty_inputs = TextGenerationInputs(batches=[[]])
            assert len(inputs.flat_batch) == 4
            assert len(empty_inputs.flat_batch) == 0

        Given a loaded ``pipeline``, the first ``execute`` returns no outputs
        (they belong to a not-yet-submitted previous batch); a second, empty
        call drains the first batch's outputs:

        .. code-block:: text

            output_a = pipeline.execute(inputs)
            assert len(output_a) == 0

            output_b = pipeline.execute(empty_inputs)
            assert len(output_b) == len(inputs.flat_batch)

        Args:
            inputs: The inputs for the batch.

        Returns:
            A dictionary of request IDs to outputs. The outputs do not correspond
            to the requests in the input batch. Instead they are from the previous batch.
        """
        if inputs.enable_log_probs:
            raise ValueError(
                "Log probabilities are not supported with overlap pipeline"
            )

        execute_start_monotonic = time.monotonic()

        # Initialize variables that may be set conditionally below.
        curr_batch: AsyncBatch[TextGenerationContextType] | None = None
        sampling_processor: FusedSamplingProcessor | None = None
        bitmask: npt.NDArray[np.int32] | None = None
        model_outputs: ModelOutputs | None = None
        outputs: PipelineOutputsDict[TextGenerationOutput] = {}
        # Set when the early-sync guard fires; reused in the normal sync path
        # below to prevent double-calling sync_and_process_outputs.
        _early_sync_outputs: _AsyncBatchOutput | None = None
        _early_sync_monotonic: float | None = None
        _early_sync_duration_s: float | None = None

        if self._spec_decode_state is not None:
            self._spec_decode_state.batch_metrics = None

        if inputs:
            # Spec-decode handles sampling internally.
            # Remove the condition below when SERVOPT-992 is resolved.
            if self._spec_decode_state is not None:
                # Now that this iteration's batch order is known, enqueue the
                # previous batch's FSM-advance + bitmask callback so it writes
                # the bitmask directly in THIS batch's row order. This must run
                # before the early-sync guard below: it sets the previous
                # batch's ``fsm_advanced_by_callback``, which the guard reads to
                # decide whether the previous batch still needs a synchronous
                # FSM advance (it does only when no callback advanced it, e.g.
                # the prefill->decode boundary).
                self._enqueue_prev_bitmask_callback(
                    curr_context_batch=inputs.flat_batch,
                    curr_verify_width=self._verify_width(inputs),
                )

                if self._should_early_sync_prev_batch():
                    assert self._prev_batch is not None
                    _early_sync_start_monotonic = time.monotonic()
                    _early_sync_outputs = (
                        self._prev_batch.sync_and_process_outputs()
                    )
                    _early_sync_monotonic = time.monotonic()
                    _early_sync_duration_s = (
                        _early_sync_monotonic - _early_sync_start_monotonic
                    )

                # FSM is advanced asynchronously by the host callback enqueued
                # just above. The captured graph's ``mo.wait_host_value_with_dep``
                # blocks the in-graph H2D until the worker signals, so the FSM
                # advancement is observed before the sampler runs.
                curr_batch = self._execute_spec_decode(inputs)
            else:
                # Run the entire forward pass and output processing if the
                # batch has at least one request.
                #
                # Launch the forward pass FIRST, then build the sampling
                # processor. ``_run_forward`` enqueues the decode kernels
                # asynchronously and returns immediately, so constructing the
                # sampling processor (host-side param gather + small H2D copies
                # in ``SamplerInputs.create``) overlaps with the GPU forward
                # instead of running as exposed host time before it. The two
                # are independent: ``_run_forward`` neither reads the sampling
                # processor / bitmask nor mutates any state
                # ``_create_sampling_processor`` consumes (``sampling_params``
                # and host token lengths are unchanged by the forward launch),
                # and both still precede the previous-batch sync / FSM advance
                # below, so ordering and results are identical.
                model_outputs = self._run_forward(inputs)
                sampling_processor, bitmask = self._create_sampling_processor(
                    inputs.flat_batch
                )

        elif self.pipeline_config.runtime.execute_empty_batches:
            # If the batch is empty and execute_empty_batches is True, we will
            # only run the forward pass to ensure that the barrier point is reached
            # for EP + DP. We skip all output processing.
            self._run_forward(inputs)

        if self._prev_batch is not None:
            assert not self._disable_overlap, (
                "Cannot have a previous batch when overlap is disabled"
            )
            if _early_sync_outputs is not None:
                # Early-sync guard already called sync_and_process_outputs;
                # reuse the result to avoid syncing the same batch twice.
                wrapped_outputs = _early_sync_outputs
            else:
                # Normal path: sync previous batch, advancing FSM and updating
                # bitmask for requests continuing into the current batch.
                wrapped_outputs = self._prev_batch.sync_and_process_outputs(
                    curr_flat_batch=inputs.flat_batch if inputs else None,
                    bitmask=bitmask,
                    sampling_processor=sampling_processor,
                )

            if self._spec_decode_state is not None:
                assert wrapped_outputs.spec_decode_metrics is not None
                self._spec_decode_state.batch_metrics = (
                    wrapped_outputs.spec_decode_metrics
                )
            self._record_completed_batch_stats(
                self._prev_batch,
                wrapped_outputs.spec_decode_metrics,
                sync_monotonic=_early_sync_monotonic
                if _early_sync_monotonic is not None
                else time.monotonic(),
                early_sync_duration_s=_early_sync_duration_s,
            )
            outputs = wrapped_outputs.output_dict
            self._prev_batch = None

        # Sample logits for current batch (if we have inputs).
        if (
            inputs
            and model_outputs is not None
            and sampling_processor is not None
        ):
            curr_batch = self._sample_logits(
                inputs, model_outputs, sampling_processor
            )

        if curr_batch is not None:
            for context in inputs.flat_batch:
                context.update_with_future_token()
                # TODO: these two fields should not both be named spec_decode_state...
                if self._spec_decode_state is not None:
                    assert curr_batch.spec_decode is not None
                    num_draft_tokens_to_verify = (
                        curr_batch.spec_decode.num_draft_tokens_to_verify
                    )
                    context.spec_decoding_state.maybe_accepted_draft_tokens = [
                        _OOB_IDX
                    ] * num_draft_tokens_to_verify
                    if (
                        context.tokens.generated_length
                        and not self._disable_overlap
                    ):
                        # In overlap mode, _execute_spec_decode (step A) always
                        # runs BEFORE sync_and_process_outputs (step B) in each
                        # execute() call.  Step B is what writes real draft
                        # tokens into draft_tokens_to_verify, so those tokens
                        # are never available in time for step A.  Without this
                        # reset, step A would read tokens written by step B two
                        # iterations ago — stale by one context-advance and
                        # therefore wrong to verify.  Resetting to [] causes
                        # _execute_spec_decode to use MAGIC_DRAFT_TOKEN_ID as a
                        # placeholder (see fallback there), which keeps the
                        # draft-token tensor at shape K>0 so CUDA graph replay
                        # remains active.  Previously [_OOB_IDX] * K was used
                        # here, but that list is non-empty so the MAGIC fallback
                        # never triggered and the OOB indices were sent directly
                        # to the GPU acceptance sampler.
                        context.spec_decoding_state.draft_tokens_to_verify = []

        # Commit the new KV blocks into the prefix cache, ignoring the trailing
        # placeholder future token.
        for ctx in inputs.flat_batch:
            self._kv_manager.step(ctx)

        if curr_batch is not None:
            if self._disable_overlap:
                # Immediately synchronize after gpu execution and return the
                # results of the current batch.
                wrapped_outputs = curr_batch.sync_and_process_outputs()
                if self._spec_decode_state is not None:
                    assert wrapped_outputs.spec_decode_metrics is not None
                    self._spec_decode_state.batch_metrics = (
                        wrapped_outputs.spec_decode_metrics
                    )
                # Merge current batch outputs with any previous batch outputs
                outputs.update(wrapped_outputs.output_dict)
            else:
                # Delay the synchronization until the next step.
                # For structured output, the bitmask is updated in
                # sync_and_process_outputs() after the FSM is advanced,
                # so overlap scheduling still works correctly.
                curr_batch.enqueue_monotonic = execute_start_monotonic
                self._prev_batch = curr_batch

        return outputs

    def release(self, request_id: RequestID) -> None:
        """Mark the context as complete, releasing the cache slot from the KV manager.

        Note: Primary KV cache lifecycle is managed by the scheduler. This method
        handles extra KV caches managed by the pipeline model (e.g., indexer cache
        for DeepSeekV3.2).
        """
        # Primary KV cache release is handled by the scheduler via batch_constructor.
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

    def batch_spec_decode_metrics(
        self,
    ) -> _SpeculativeDecodingMetrics | None:
        """Returns the per-batch draft token acceptance metrics for the most recent batch."""
        if self._spec_decode_state is None:
            return None
        return self._spec_decode_state.batch_metrics
