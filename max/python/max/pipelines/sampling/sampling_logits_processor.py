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

"""Fused sampling logits processor used to apply sampling parameters to logits."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from max.driver import (
    Buffer,
    Device,
    DeviceEvent,
    DevicePinnedBuffer,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.engine import Model
from max.pipelines.context import (
    BatchProcessorInputs,
    TextContext,
    TextGenerationContextType,
)
from max.profiler import Tracer, traced
from typing_extensions import Self

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig

_logger = logging.getLogger("max.pipelines")


@dataclass(frozen=True)
class FrequencyData:
    """Container for token frequency data in CSR format."""

    data: Buffer
    """data[:, 0]: 1D array of the column indices of the
        non-zero elements in the matrix.
    data[:, 1]: 1D array of the non-zero elements in the
        matrix."""

    offsets: Buffer
    """Row offsets: shape [batch_size + 1] indicating start of each
    sequence's data."""


def _to_pinned_host_buffer(
    values: Sequence[Any], device: Device, dtype: DType = DType.float32
) -> Buffer:
    """Copies a sequence of values into a buffer, pinned when device is not host.

    Args:
        values: Values to copy into the buffer.
        device: Target device (buffer is pinned if this is a non-host device).
        dtype: Buffer dtype. Defaults to float32.

    Returns:
        A buffer containing the values, with pinning if applicable.
    """
    if device.is_host:
        buffer = Buffer(
            shape=(len(values),),
            dtype=dtype,
            device=device,
        )
    else:
        buffer = DevicePinnedBuffer(
            shape=(len(values),),
            dtype=dtype,
            device=device,
        )

    buffer_np = buffer.to_numpy()
    buffer_np[:] = values
    return buffer


class FusedSamplingProcessor:
    """Applies sampling parameters to logits and stores the chosen tokens."""

    new_tokens: Buffer | None = None
    """The new tokens that were sampled."""

    generated_tokens: Buffer
    """The generated tokens that have been sampled so far."""

    @staticmethod
    def allocate_identity_logit_offsets(
        pipeline_config: PipelineConfig, device: Device, max_batch_size: int
    ) -> Buffer | None:
        """Returns a preallocated ``[0, 1, ..., max_batch_size]`` index buffer.

        Used by `logits_for_sampling` when sampling from ``next_token_logits``
        with a variable-logit sampler. Returns ``None`` when the buffer is not
        needed (variable-logit sampling disabled, or running in virtual-device
        mode).
        """
        if (
            not pipeline_config.sampling.enable_variable_logits
            or is_virtual_device_mode()
        ):
            return None
        return Buffer.from_numpy(
            np.arange(max_batch_size + 1, dtype=np.uint32)
        ).to(device)

    @traced
    def __init__(
        self,
        sampler: Model,
        pipeline_config: PipelineConfig,
        context_batch: list[Any],
        device: Device,
        bitmask: npt.NDArray[np.int32] | None = None,
        vocab_size: int | None = None,
        pinned_new_tokens: Buffer | None = None,
        identity_logit_offsets: Buffer | None = None,
    ):
        self.sampler = sampler
        self.batch_size = len(context_batch)
        self.device = device
        self.bitmask = bitmask
        self.vocab_size = vocab_size
        self._identity_logit_offsets = identity_logit_offsets

        # If a structured decoding bitmask was provided, stage the packed int32
        # bitmask in pinned memory for efficient per-step H2D updates. The
        # sampler graph unpacks and applies it on the GPU (apply_packed_bitmask),
        # so no CPU-side unpack to bool happens here.
        self.tensor_bitmask: Buffer | None = None
        self._pinned_bitmask: Buffer | None = None

        # Event to track D2H token copy completion. All transfers happen on the
        # default stream; we use events for fine-grained synchronization.
        self._d2h_copy_event: DeviceEvent | None = None

        # Pre-allocated pinned buffer for D2H token copy (optional).
        self._pinned_new_tokens = pinned_new_tokens

        if (
            self.bitmask is not None
            and self.vocab_size is not None
            and self.bitmask.shape[1] != self.vocab_size
        ):
            # ``self.bitmask`` is a packed int32 bitmask
            # [batch, ceil(vocab/32)]. Stage it as-is; the sampler graph
            # unpacks and applies it on the GPU.
            if device.is_host:
                # Host device case - no pinned memory needed
                self._pinned_bitmask = Buffer(
                    shape=self.bitmask.shape,
                    dtype=DType.int32,
                    device=device,
                )
            else:
                # GPU - allocate pinned memory for faster H2D transfer
                self._pinned_bitmask = DevicePinnedBuffer(
                    shape=self.bitmask.shape,
                    dtype=DType.int32,
                    device=device,
                )
            self._pinned_bitmask.to_numpy()[:] = self.bitmask
            self.tensor_bitmask = self._pinned_bitmask.to(self.device)

        batch_size = len(context_batch)

        # Allocate input and output tensors on pinned memory for faster async
        # h2d and d2h transfer speeds. If model is on host, then fall back to
        # normal pageable memory.

        # generated_tokens is a tensor of shape (batch_size, 0)
        if device.is_host:
            self.generated_tokens = Buffer(
                shape=(batch_size, 0),
                dtype=DType.int64,
                device=device,
            )
        else:
            self.generated_tokens = DevicePinnedBuffer(
                shape=(batch_size, 0),
                dtype=DType.int64,
                device=device,
            )

        self.sampler_inputs = SamplerInputs.create(context_batch, device)

        self.frequency_data: list[FrequencyData] | None = None
        self.frequency_penalty: Buffer | None = None
        self.presence_penalty: Buffer | None = None
        self.repetition_penalty: Buffer | None = None

        self.penalty_inputs: PenaltyInputs | None = None
        if pipeline_config.sampling.enable_penalties:
            self.penalty_inputs = PenaltyInputs.create(context_batch, device)
        else:
            needs_penalties = any(
                context.sampling_params.needs_penalties
                for context in context_batch
            )
            if needs_penalties:
                _logger.warning(
                    "Penalties are provided in the request, but the model was not configured with enable_penalties=True, ignoring"
                )

        self.min_tokens_masks = _build_min_tokens_masks(
            context_batch,
            1,
            device,
            pipeline_config.sampling.enable_min_tokens,
        )

        self.step_counter = 0

    def logits_for_sampling(
        self,
        *,
        logits: Buffer,
        next_token_logits: Buffer | None,
        logit_offsets: Buffer | None,
    ) -> tuple[Buffer, Buffer | None]:
        """Returns the logits and offsets to pass to logits processors."""
        if next_token_logits is None:
            return logits, logit_offsets

        if logit_offsets is None:
            return next_token_logits, None

        offsets_len = self.batch_size + 1
        if self._identity_logit_offsets is None:
            raise AssertionError(
                "identity_logit_offsets is required when sampling from "
                "next_token_logits with variable-logit sampler offsets."
            )
        if int(self._identity_logit_offsets.shape[0]) < offsets_len:
            raise AssertionError(
                "identity_logit_offsets is smaller than the current batch: "
                f"offsets_len={self._identity_logit_offsets.shape[0]}, "
                f"required={offsets_len}."
            )

        return next_token_logits, self._identity_logit_offsets[:offsets_len]

    def __call__(self, inputs: BatchProcessorInputs) -> None:
        """Processes the batch logits and updates generated tokens and seed."""
        logits = inputs.logits
        logit_offsets = inputs.logit_offsets

        if logits.device != self.device:
            logits = logits.to(self.device)
            if logit_offsets is not None:
                logit_offsets = logit_offsets.to(self.device)

        new_tokens, new_generated_tokens, new_seed = _sample_logits(
            self.sampler,
            logits,
            self.generated_tokens,
            self.sampler_inputs,
            logit_offsets=logit_offsets,
            bitmask=self.tensor_bitmask,
            min_tokens_mask=self.min_tokens_masks[self.step_counter]
            if self.min_tokens_masks
            else None,
            penalty_inputs=self.penalty_inputs,
        )

        assert isinstance(new_tokens, Buffer)
        assert isinstance(new_generated_tokens, Buffer)
        assert isinstance(new_seed, Buffer)

        self.generated_tokens = new_generated_tokens
        self.seed = new_seed
        self.new_tokens = new_tokens

        self.step_counter += 1

    def update_bitmask(self, packed_bitmask: npt.NDArray[np.int32]) -> None:
        """Update the GPU bitmask with new FSM state for multi-step execution.

        Copies the packed int32 bitmask from llguidance into the pinned host
        buffer and transfers it to the GPU, keeping the bitmask synchronized
        with the FSM state after each token is sampled. Unpacking is done on the
        GPU by the sampler graph (apply_packed_bitmask), not here.

        Args:
            packed_bitmask: Packed int32 bitmask from
                llguidance.numpy.allocate_token_bitmask. Shape is
                [batch_size, ceil(vocab_size/32)].
        """
        if (
            self.tensor_bitmask is None
            or self._pinned_bitmask is None
            or self.vocab_size is None
        ):
            return

        # Copy the packed int32 bitmask straight into the pinned buffer.
        self._pinned_bitmask.to_numpy()[:] = packed_bitmask

        # Transfer to GPU on the default stream. The H2D copy will complete
        # before subsequent GPU operations (like sampling) that use the bitmask.
        self.tensor_bitmask.inplace_copy_from(self._pinned_bitmask)

    def start_async_token_copy(self) -> None:
        """Start D2H copy of new_tokens to pinned buffer on the default stream.

        The copy happens on the default stream after sampling completes. We record
        an event after the copy so get_new_tokens_numpy() can wait for just the
        copy without waiting for subsequent GPU operations (like the next forward
        pass).
        """
        if self.new_tokens is None or self._pinned_new_tokens is None:
            return

        # Copy tokens to pinned buffer on default stream (after sampling)
        self._pinned_new_tokens.inplace_copy_from(self.new_tokens)

        # Record event after copy so we can wait for it specifically
        self._d2h_copy_event = self.device.default_queue.record_event()

    def get_new_tokens_numpy(self) -> npt.NDArray[np.int64]:
        """Wait for D2H copy and return the new tokens as numpy array.

        If async copy was started via start_async_token_copy(), this waits for
        the copy event. Otherwise, falls back to synchronous copy.

        Returns:
            Numpy array of the new tokens with shape (batch_size,).
        """
        assert self.new_tokens is not None

        if self._d2h_copy_event is not None:
            # Wait for the D2H copy to complete (not the entire stream)
            self._d2h_copy_event.synchronize()
            self._d2h_copy_event = None
            assert self._pinned_new_tokens is not None
            assert self._pinned_new_tokens.pinned
            # Slice numpy result to batch_size (pinned buffer may be larger)
            with Tracer("pinned_new_tokens_to_numpy"):
                return self._pinned_new_tokens.to_numpy()[: self.batch_size]

        # Fallback: synchronous copy (host device or no async copy started)
        return self.new_tokens.to_numpy()


@traced
def _build_token_frequency_csr(
    batch: list[TextGenerationContextType],
    padding_size: int,
    device: Device,
    include_prompt: bool = False,
) -> FrequencyData:
    """Builds a CSR matrix of token frequency in the batch.

    The original matrix is (batch_size, vocab_size), where each element is
    the number of times a token appears in the batch.

    Returns:
        FrequencyData containing the CSR representation with:
        - data: 2D array where each row is [token_id, count]
        - row_offsets: 1D array of the starting index of each sequence's data
    """
    tracer: Tracer = Tracer("build_token_frequency_csr")

    PADDING_TOKEN = -1
    batch_size = len(batch)

    # Pre-allocate row offsets
    frequency_row_offsets = np.zeros(batch_size + 1, dtype=np.uint32)

    # Calculate max size needed for token frequency pairs (upper bound)
    if include_prompt:
        total_tokens = sum(
            len(context.tokens) + padding_size for context in batch
        )
    else:
        total_tokens = sum(
            len(context.tokens.generated) + padding_size for context in batch
        )
    token_frequency_pairs = np.empty((total_tokens, 2), dtype=np.int32)

    tracer.next("build_token_frequency_csr_loop")

    current_offset = 0
    for i, context in enumerate(batch):
        tokens = (
            context.tokens.all if include_prompt else context.tokens.generated
        )
        unique_tokens, counts = np.unique(tokens, return_counts=True)

        num_unique = len(unique_tokens)
        padded_size = num_unique + padding_size

        # Record start offset for this context
        frequency_row_offsets[i] = current_offset

        # Write unique tokens and their counts directly (no np.pad)
        end_unique = current_offset + num_unique
        token_frequency_pairs[current_offset:end_unique, 0] = unique_tokens
        token_frequency_pairs[current_offset:end_unique, 1] = counts

        # Write padding directly
        end_padded = current_offset + padded_size
        token_frequency_pairs[end_unique:end_padded, 0] = PADDING_TOKEN
        token_frequency_pairs[end_unique:end_padded, 1] = 0

        current_offset = end_padded

    # Record final offset
    frequency_row_offsets[batch_size] = current_offset

    # Trim to actual size used
    token_frequency_pairs = token_frequency_pairs[:current_offset]

    return FrequencyData(
        data=Buffer.from_dlpack(token_frequency_pairs).to(device),
        offsets=Buffer.from_dlpack(frequency_row_offsets).to(device),
    )


@traced
def _build_min_tokens_masks(
    batch: list[TextGenerationContextType],
    num_steps: int,
    device: Device,
    enable_min_tokens: bool,
) -> list[Buffer] | None:
    """Build a mask of the min tokens for the batch."""
    if not enable_min_tokens:
        for context in batch:
            if context.min_tokens > 0:
                _logger.warning(
                    "min_tokens is provided in the request, but the model was not configured with enable_min_tokens=True, ignoring"
                )
        return None

    min_tokens_masks: list[npt.NDArray[np.int32]] = []
    min_tokens_masks = batch[0].get_min_token_logit_mask(num_steps)

    for bs in range(1, len(batch)):
        new_min_tokens_masks = batch[bs].get_min_token_logit_mask(num_steps)
        for i in range(num_steps):
            new_min_tokens_masks[i][:, 0] += bs
            min_tokens_masks[i] = np.concatenate(
                (min_tokens_masks[i], new_min_tokens_masks[i])
            )

    min_tokens_masks_max = [
        Buffer.from_dlpack(mask).to(device) for mask in min_tokens_masks
    ]
    return min_tokens_masks_max


@traced
def _sample_logits(
    sampler: Model,
    logits: Buffer,
    prev_tokens: Buffer,
    sampler_inputs: SamplerInputs,
    *,
    logit_offsets: Buffer | None = None,
    bitmask: Buffer | None = None,
    min_tokens_mask: Buffer | None = None,
    penalty_inputs: PenaltyInputs | None = None,
) -> tuple[Buffer, Buffer, Buffer]:
    opt_inputs = [logit_offsets, bitmask]

    base_inputs = [
        logits,
        prev_tokens,
        *sampler_inputs.as_list(),
    ]

    # Add frequency data if provided
    if penalty_inputs:
        opt_inputs.extend(penalty_inputs.as_list())

    if min_tokens_mask:
        opt_inputs.append(min_tokens_mask)

    graph_inputs = base_inputs + [
        tensor for tensor in opt_inputs if tensor is not None
    ]

    sampler_output = sampler(*graph_inputs)
    tokens, generated_tokens = sampler_output[:2]
    new_seed = sampler_output[-1]
    assert isinstance(tokens, Buffer)
    assert isinstance(generated_tokens, Buffer)
    assert isinstance(new_seed, Buffer)
    return (tokens, generated_tokens, new_seed)


@dataclass
class PenaltyInputs:
    """Container for penalty inputs."""

    frequency_data: list[FrequencyData]
    frequency_penalty: Buffer
    presence_penalty: Buffer
    repetition_penalty: Buffer

    def as_list(self) -> list[Buffer]:
        """Returns the penalty inputs as a list of buffers."""
        buffers = []
        for fd in self.frequency_data:
            buffers.extend([fd.data, fd.offsets])
        buffers.extend(
            [
                self.frequency_penalty,
                self.presence_penalty,
                self.repetition_penalty,
            ]
        )
        return buffers

    @classmethod
    def create(
        cls,
        batch: list[TextContext],
        device: Device,
        num_steps: int = 1,
    ) -> Self:
        """Create penalty input tensors from context batch.

        Args:
            batch: List of context objects containing sampling parameters.
            device: Device to place the tensors on.
            num_steps: Number of generation steps for frequency CSR padding.

        Returns:
            PenaltyInputs containing the penalty input tensors
        """
        frequency_data = [
            _build_token_frequency_csr(batch, num_steps, device),
            _build_token_frequency_csr(
                batch, num_steps, device, include_prompt=True
            ),
        ]

        frequency_penalty = Buffer.from_numpy(
            np.array(
                [
                    context.sampling_params.frequency_penalty
                    for context in batch
                ],
                dtype=np.float32,
            )
        ).to(device)
        presence_penalty = Buffer.from_numpy(
            np.array(
                [context.sampling_params.presence_penalty for context in batch],
                dtype=np.float32,
            )
        ).to(device)
        repetition_penalty = Buffer.from_numpy(
            np.array(
                [
                    context.sampling_params.repetition_penalty
                    for context in batch
                ],
                dtype=np.float32,
            )
        ).to(device)

        return cls(
            frequency_data=frequency_data,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
        )


@dataclass
class SamplerInputs:
    """Container for sampler inputs."""

    top_k: Buffer
    max_k: Buffer
    temperature: Buffer
    top_p: Buffer
    min_top_p: Buffer
    min_p: Buffer
    seed: Buffer

    def as_list(self) -> list[Buffer]:
        """Returns the sampler inputs as a list of buffers."""
        return [
            self.top_k,
            self.max_k,
            self.temperature,
            self.top_p,
            self.min_top_p,
            self.min_p,
            self.seed,
        ]

    @classmethod
    def create(
        cls,
        batch: list[TextContext],
        device: Device,
    ) -> Self:
        """Create sampling parameter tensors from context batch.

        Args:
            batch: List of context objects containing sampling parameters.
            device: Device to place the tensors on.

        Returns:
            SamplerInputs containing the sampling parameter tensors.
        """
        temperature_host = _to_pinned_host_buffer(
            [context.sampling_params.temperature for context in batch],
            device=device,
        )
        temperature = temperature_host.to(device)

        # top_k is a tensor of shape (batch_size,)
        top_k_host = _to_pinned_host_buffer(
            [context.sampling_params.top_k for context in batch],
            device=device,
            dtype=DType.int64,
        )
        top_k = top_k_host.to(device)

        # max_k is a scalar 0-d tensor. It does not need to be pinned since it
        # is not copied to the device.
        max_k_np = np.array(np.max(top_k_host.to_numpy()), dtype=np.int64)
        max_k = Buffer.from_numpy(max_k_np)

        # top_p is a tensor of shape (batch_size,)
        top_p_host = _to_pinned_host_buffer(
            [context.sampling_params.top_p for context in batch],
            device=device,
        )
        top_p = top_p_host.to(device)

        # min_top_p is a scalar 0-d tensor. It does not need to be pinned since it
        # is not copied to the device.
        min_top_p_np = np.array(np.min(top_p_host.to_numpy()), dtype=np.float32)
        min_top_p = Buffer.from_numpy(min_top_p_np)

        # min_p is a tensor of shape (batch_size,)
        min_p_host = _to_pinned_host_buffer(
            [context.sampling_params.min_p for context in batch],
            device=device,
        )
        min_p = min_p_host.to(device)

        # seed is a tensor of shape (batch_size,)
        seed_host = _to_pinned_host_buffer(
            [
                context.sampling_params.seed + len(context.tokens)
                for context in batch
            ],
            device=device,
            dtype=DType.uint64,
        )
        seed = seed_host.to(device)

        return cls(
            top_k=top_k,
            max_k=max_k,
            temperature=temperature,
            top_p=top_p,
            min_top_p=min_top_p,
            min_p=min_p,
            seed=seed,
        )
