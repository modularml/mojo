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
"""Batch processor base types and shared helpers for MAX pipeline models."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

import numpy as np
from max.driver import (
    Buffer,
    Device,
    DevicePinnedBuffer,
    is_virtual_device_mode,
)
from max.dtype import DType
from max.graph import BufferType, DeviceRef, TensorType
from max.nn.comm import Signals
from max.nn.kv_cache import KVCacheInputsInterface
from max.nn.kv_cache.cache_params import KVCacheParamInterface
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.context import BaseContext, TextContext
from max.pipelines.context.tokens import TokenBuffer
from max.pipelines.lib.interfaces.arch_config import ArchConfig
from max.pipelines.lib.interfaces.pipeline_model import (
    ModelInputs,
    ModelOutputs,
    UnifiedSpecDecodeInputs,
)
from max.pipelines.lora import LoRAManagerV3
from max.pipelines.modeling.dataprocessing import collate_batch

if TYPE_CHECKING:
    from max.pipelines.lib import PipelineConfig

logger = logging.getLogger("max.pipelines")

ContextT = TypeVar("ContextT", bound=BaseContext)
InputsT = TypeVar("InputsT", bound=ModelInputs)
SpecDecodeInputsT = TypeVar("SpecDecodeInputsT", bound=UnifiedSpecDecodeInputs)


@runtime_checkable
class RaggableContext(Protocol):
    """Context protocol for ragged token batching helpers."""

    tokens: TokenBuffer


@dataclass
class BatchProcessorRuntime:
    """Runtime dependencies shared by batch processors."""

    pipeline_config: PipelineConfig
    devices: list[Device]
    return_logits: ReturnLogits
    return_hidden_states: ReturnHiddenStates = ReturnHiddenStates.NONE
    signal_buffers: Sequence[Buffer] = ()
    lora_manager: LoRAManagerV3 | None = None
    pad_token_id: int = 0
    max_batch_size: int | None = None


class ModelInputCachingAllocator:
    """Caches reusable, non-pinned device input buffers across steps.

    Small requests reuse a fixed-capacity backing buffer.
    Views with the same ``(name, device)`` alias storage and can race if used
    concurrently.

    Requests larger than the fixed capacity fall back to a dedicated owning
    buffer cached by the exact ``(name, device, dtype, shape)`` key, and log a
    warning on each new allocation.

    Reuse keeps a captured graph replaying in place (the replay copy is a
    no-op when the model inputs already are the captured buffers). Callers own
    copying host data into the returned buffer.

    Never cache pinned host staging buffers here: those must be allocated fresh
    every step so the next overlap step's host writes can't clobber an
    in-flight H2D copy.
    """

    FIXED_CAPACITY_BYTES: ClassVar[int] = 8 * 1024 * 1024

    def __init__(self) -> None:
        self._fixed_capacity_buffers: dict[tuple[str, int], Buffer] = {}
        # Graph replay skips input copies only when capture and replay receive
        # the same Buffer object; recreating an aliased view still copies.
        self._fixed_capacity_views: dict[
            tuple[str, int, DType, tuple[int, ...]], Buffer
        ] = {}
        self._oversized_buffers: dict[
            tuple[str, int, DType, tuple[int, ...]], Buffer
        ] = {}

    def alloc(
        self,
        *,
        name: str,
        dtype: DType,
        shape: tuple[int, ...],
        device: Device,
    ) -> Buffer:
        """Returns a reusable buffer of ``shape`` and ``dtype`` on ``device``.

        Up to :attr:`FIXED_CAPACITY_BYTES` the result is a view aliasing the
        single owning backing buffer held for ``(name, device)``. A later
        allocation under the same name and device may overwrite its contents,
        while an exact dtype-and-shape request returns the same view object for
        graph replay. Beyond that capacity the result is a dedicated owning
        buffer cached by the exact ``(name, device, dtype, shape)`` key.
        Distinct ``name``s never alias, and each device gets its own backing
        under a given name.

        Args:
            name: Stable identifier for this logical buffer.
            dtype: Element type.
            shape: Buffer shape (ragged inputs change shape per step).
            device: Target device for the buffer.
        """
        shape = tuple(shape)
        num_bytes = math.prod(shape) * dtype.size_in_bytes
        capacity = self.FIXED_CAPACITY_BYTES

        if num_bytes <= capacity:
            view_key = (name, id(device), dtype, shape)
            view = self._fixed_capacity_views.get(view_key)
            if view is not None:
                return view

            backing_key = (name, id(device))
            backing = self._fixed_capacity_buffers.get(backing_key)
            if backing is None:
                backing = Buffer(
                    shape=(capacity,), dtype=DType.uint8, device=device
                )
                self._fixed_capacity_buffers[backing_key] = backing
            view = backing[:num_bytes].view(dtype, shape)
            self._fixed_capacity_views[view_key] = view
            return view

        oversized_key = (name, id(device), dtype, shape)
        buffer = self._oversized_buffers.get(oversized_key)
        if buffer is None:
            logger.warning(
                "Model input %r needs %d bytes (shape=%s, dtype=%s, "
                "device=%s), exceeding the %d-byte reusable input capacity; "
                "allocating a dedicated buffer cached per exact shape.",
                name,
                num_bytes,
                shape,
                dtype,
                device,
                capacity,
            )
            buffer = Buffer(shape=shape, dtype=dtype, device=device)
            self._oversized_buffers[oversized_key] = buffer
        return buffer


class BatchProcessor(ABC, Generic[ContextT, InputsT]):
    """Batches pipeline contexts into model inputs and parses execution outputs."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        self.config = config
        self.runtime = runtime
        # Caches reusable non-pinned device input buffers so captured graphs
        # replay in place. Pinned host staging is never cached here.
        self._device_input_allocator = ModelInputCachingAllocator()

    @abstractmethod
    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns non-KV graph input types in execution order."""

    @abstractmethod
    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares inputs for the first execution step of a batch."""

    @abstractmethod
    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw ``Model.execute`` buffers to :class:`ModelOutputs`."""


class RaggedBatchProcessor(BatchProcessor[ContextT, InputsT]):
    """Base for ragged KV text batching."""

    def __init__(
        self,
        config: ArchConfig,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        # Pre-allocate row offsets for multistep decode to avoid materializing
        # and copying a buffer on each step. Skip in virtual device mode
        # (warm-cache/cross-compilation) since VirtualDeviceContext does not
        # support memAlloc.
        assert runtime.max_batch_size, "Expected max_batch_size to be set"
        self._input_row_offsets_prealloc: Buffer | None = None
        if not is_virtual_device_mode() and runtime.devices:
            self._input_row_offsets_prealloc = Buffer.from_numpy(
                np.arange(runtime.max_batch_size + 1, dtype=np.uint32),
            ).to(runtime.devices[0])


def single_replica_context_batch(
    replica_batches: Sequence[Sequence[ContextT]],
    *,
    processor_name: str,
) -> Sequence[ContextT]:
    """Returns the sole replica batch or raises when DP is unsupported."""
    if len(replica_batches) > 1:
        raise ValueError(f"{processor_name} does not support DP>1")
    return replica_batches[0]


def build_single_replica_ragged_token_arrays(
    context_batch: Sequence[RaggableContext],
) -> tuple[np.ndarray, np.ndarray]:
    """Builds concatenated token and row-offset arrays for a ragged batch."""
    input_row_offsets = np.cumsum(
        [0] + [ctx.tokens.active_length for ctx in context_batch],
        dtype=np.uint32,
    )
    tokens = np.concatenate([ctx.tokens.active for ctx in context_batch])
    return tokens, input_row_offsets


class SingleReplicaRaggedBatchProcessor(
    RaggedBatchProcessor[ContextT, InputsT]
):
    """Single-replica ragged KV batching for Graph-path models (no DP / LoRA).

    Subclasses implement :meth:`_make_inputs` to construct their architecture-
    specific :class:`~max.pipelines.lib.interfaces.pipeline_model.ModelInputs`
    type. Override :attr:`_include_signal_buffers` or :meth:`get_symbolic_inputs`
    when signal-buffer wiring differs.
    """

    _include_signal_buffers: ClassVar[bool] = False

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for single-replica ragged KV models."""
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=self._include_signal_buffers,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica Graph-path batch."""
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device0 = self.runtime.devices[0]
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        return self._make_inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
            signal_buffers=list(self.runtime.signal_buffers),
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        signal_buffers: list[Buffer],
    ) -> InputsT:
        """Constructs architecture-specific model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


class ModuleV3SingleReplicaBatchProcessor(BatchProcessor[ContextT, InputsT]):
    """Single-replica ragged KV batching for ModuleV3 compile-path models."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for single-replica ModuleV3 models."""
        return modulev3_ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[ContextT]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica ModuleV3 batch.

        Appends the per-call ModuleV3 LoRA buffers when the runtime's manager
        is a :class:`LoRAManagerV3`, reusing the row offsets already built here.
        """
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        assert kv_cache_inputs is not None
        tokens_np, offsets_np = build_single_replica_ragged_token_arrays(
            cast(Sequence[RaggableContext], context_batch)
        )
        device0 = self.runtime.devices[0]
        inputs = self._make_inputs(
            tokens=Buffer.from_numpy(tokens_np).to(device0),
            input_row_offsets=Buffer.from_numpy(offsets_np).to(device0),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.int64)
            ),
            kv_cache_inputs=kv_cache_inputs,
        )
        if isinstance(self.runtime.lora_manager, LoRAManagerV3):
            inputs.lora_buffers = self.runtime.lora_manager.input_buffers(
                context_batch, offsets_np, device0
            )
        return inputs

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer],
    ) -> InputsT:
        """Constructs architecture-specific model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        return process_ragged_kv_outputs(
            outputs,
            return_logits=self.runtime.return_logits,
            return_hidden_states=self.runtime.return_hidden_states,
        )


def process_ragged_kv_outputs(
    outputs: Sequence[Buffer | object],
    *,
    return_logits: ReturnLogits,
    return_hidden_states: ReturnHiddenStates,
) -> ModelOutputs:
    """Maps standard ragged+KV logits buffers to :class:`ModelOutputs`."""
    has_offsets = return_logits in (ReturnLogits.VARIABLE, ReturnLogits.ALL)
    has_hidden_states = return_hidden_states != ReturnHiddenStates.NONE

    assert isinstance(outputs[0], Buffer)
    if has_offsets and has_hidden_states:
        assert len(outputs) == 4
        assert isinstance(outputs[1], Buffer)
        assert isinstance(outputs[2], Buffer)
        assert isinstance(outputs[3], Buffer)
        return ModelOutputs(
            logits=outputs[1],
            next_token_logits=outputs[0],
            logit_offsets=outputs[2],
            hidden_states=outputs[3],
        )
    if has_offsets:
        assert len(outputs) == 3
        assert isinstance(outputs[1], Buffer)
        assert isinstance(outputs[2], Buffer)
        return ModelOutputs(
            logits=outputs[1],
            next_token_logits=outputs[0],
            logit_offsets=outputs[2],
        )
    if has_hidden_states:
        assert len(outputs) == 2
        assert isinstance(outputs[1], Buffer)
        return ModelOutputs(
            logits=outputs[0],
            next_token_logits=outputs[0],
            hidden_states=outputs[1],
        )
    assert len(outputs) == 1
    return ModelOutputs(
        logits=outputs[0],
        next_token_logits=outputs[0],
    )


def ragged_kv_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
    include_signal_buffers: bool,
) -> list[TensorType | BufferType]:
    """Returns symbolic graph inputs for a standard ragged KV text model."""
    device_ref = device_refs[0]
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    if include_signal_buffers:
        signals = Signals(devices=device_refs)
        return [
            tokens_type,
            input_row_offsets_type,
            return_n_logits_type,
            *signals.input_types(),
            *kv_inputs,
        ]
    return [
        tokens_type,
        input_row_offsets_type,
        return_n_logits_type,
        *kv_inputs,
    ]


def modulev3_ragged_kv_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for ModuleV3 ragged KV models.

    ModuleV3 ``forward`` expects ``(tokens, return_n_logits, input_row_offsets,
    *kv)`` — a different argument order than the Graph-path
    :func:`ragged_kv_symbolic_inputs`.
    """
    device_ref = device_refs[0]
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    return [
        tokens_type,
        return_n_logits_type,
        input_row_offsets_type,
        *kv_inputs,
    ]


@dataclass
class PersistentInputBuffers:
    """Pinned-host buffers reused across unified spec-decode batch steps."""

    tokens: Buffer
    input_row_offsets: Buffer

    @classmethod
    def alloc(
        cls,
        max_batch_size: int,
        max_batch_input_tokens: int,
        device: Device,
    ) -> PersistentInputBuffers:
        """Allocates persistent token and row-offset buffers for spec-decode batching."""
        max_batch_input_tokens = max(max_batch_input_tokens, max_batch_size)
        tokens = Buffer(
            shape=(max_batch_input_tokens,), dtype=DType.int64, device=device
        )
        input_row_offsets = Buffer(
            shape=(max_batch_size + 1,), dtype=DType.uint32, device=device
        )
        return cls(tokens, input_row_offsets)


class UnifiedSpecDecodeBatchProcessor(
    BatchProcessor[TextContext, SpecDecodeInputsT], Generic[SpecDecodeInputsT]
):
    """Ragged batching with persistent buffers and seed for unified spec-decode graphs."""

    def __init__(
        self,
        config: Any,
        runtime: BatchProcessorRuntime,
    ) -> None:
        super().__init__(config, runtime)
        assert runtime.max_batch_size is not None
        max_batch_input_tokens = (
            runtime.pipeline_config.runtime.max_batch_input_tokens
        )
        self._persistent_input_buffers: PersistentInputBuffers | None = None
        if not is_virtual_device_mode():
            self._persistent_input_buffers = PersistentInputBuffers.alloc(
                max_batch_size=runtime.max_batch_size,
                max_batch_input_tokens=max_batch_input_tokens,
                device=runtime.devices[0],
            )
        self._seed_counter = 0

    def _next_seed(self, device0: Device) -> Buffer:
        self._seed_counter += 1
        return Buffer.from_numpy(
            np.array([self._seed_counter], dtype=np.uint64)
        ).to(device0)

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for unified spec-decode ragged KV models."""
        return ragged_kv_symbolic_inputs(
            kv_params=kv_params,
            device_refs=device_refs,
            include_signal_buffers=False,
        )

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> SpecDecodeInputsT:
        """Prepares ragged token inputs with persistent buffers and a per-step seed."""
        context_batch = [ctx for batch in replica_batches for ctx in batch]
        device0 = self.runtime.devices[0]
        buffer_type = Buffer if device0.is_host else DevicePinnedBuffer

        total_seq_len = sum(ctx.tokens.active_length for ctx in context_batch)
        batch_size = len(context_batch)

        assert self._persistent_input_buffers is not None
        persistent_tokens = self._persistent_input_buffers.tokens
        persistent_tokens = persistent_tokens[:total_seq_len]
        persistent_input_row_offsets = (
            self._persistent_input_buffers.input_row_offsets
        )
        persistent_input_row_offsets = persistent_input_row_offsets[
            : batch_size + 1
        ]

        tokens_host = buffer_type(
            dtype=DType.int64,
            shape=(total_seq_len,),
            device=device0,
        )
        offsets_host = buffer_type(
            dtype=DType.uint32,
            shape=(batch_size + 1,),
            device=device0,
        )

        np.concatenate(
            [ctx.tokens.active for ctx in context_batch],
            out=tokens_host.to_numpy(),
        )
        persistent_tokens.inplace_copy_from(tokens_host)
        np.cumsum(
            [0] + [ctx.tokens.active_length for ctx in context_batch],
            dtype=np.uint32,
            out=offsets_host.to_numpy(),
        )
        persistent_input_row_offsets.inplace_copy_from(offsets_host)

        return_n_logits_buf = Buffer.from_numpy(
            np.array([return_n_logits], dtype=np.int64)
        )

        return self._make_inputs(
            tokens=persistent_tokens,
            input_row_offsets=persistent_input_row_offsets,
            return_n_logits=return_n_logits_buf,
            kv_cache_inputs=kv_cache_inputs,
            seed=self._next_seed(device0),
            structured_output=self.runtime.pipeline_config.needs_bitmask_constraints,
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None,
        seed: Buffer,
        structured_output: bool,
    ) -> SpecDecodeInputsT:
        """Constructs architecture-specific unified spec-decode model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def embedding_ragged_symbolic_inputs(
    *,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for single-replica ragged embedding models."""
    device_ref = device_refs[0]
    return [
        TensorType(DType.int64, shape=["total_seq_len"], device=device_ref),
        TensorType(
            DType.uint32,
            shape=["input_row_offsets_len"],
            device=device_ref,
        ),
        TensorType(
            DType.uint32,
            shape=["return_n_logits"],
            device=DeviceRef.CPU(),
        ),
    ]


class SingleReplicaEmbeddingBatchProcessor(
    BatchProcessor[TextContext, InputsT], Generic[InputsT]
):
    """Single-replica ragged batching for encoder embedding models (no KV cache)."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for ragged embedding models."""
        del kv_params
        return embedding_ragged_symbolic_inputs(device_refs=device_refs)

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares ragged token inputs for a single-replica embedding batch."""
        del kv_cache_inputs
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device = self.runtime.devices[0]

        all_tokens: list[int] = []
        row_offsets = [0]
        for ctx in context_batch:
            all_tokens.extend(ctx.tokens.active)
            row_offsets.append(len(all_tokens))

        return self._make_inputs(
            tokens=Buffer.from_numpy(np.array(all_tokens, dtype=np.uint32)).to(
                device
            ),
            input_row_offsets=Buffer.from_numpy(
                np.array(row_offsets, dtype=np.uint32)
            ),
            return_n_logits=Buffer.from_numpy(
                np.array([return_n_logits], dtype=np.uint32)
            ),
        )

    def _make_inputs(
        self,
        *,
        tokens: Buffer,
        input_row_offsets: Buffer,
        return_n_logits: Buffer,
    ) -> InputsT:
        """Constructs architecture-specific embedding model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def padded_encoder_symbolic_inputs(
    *,
    device_refs: list[DeviceRef],
) -> list[TensorType | BufferType]:
    """Symbolic compile inputs for fixed-shape padded encoder models."""
    device_ref = device_refs[0]
    return [
        TensorType(
            DType.int64,
            shape=["batch_size", "seq_len"],
            device=device_ref,
        ),
        TensorType(
            DType.float32,
            shape=["batch_size", "seq_len"],
            device=device_ref,
        ),
    ]


class PaddedEncoderBatchProcessor(
    BatchProcessor[TextContext, InputsT], Generic[InputsT]
):
    """Fixed-shape padded batching for encoder-only BERT-style models."""

    def get_symbolic_inputs(
        self,
        *,
        kv_params: KVCacheParamInterface,
        device_refs: list[DeviceRef],
    ) -> list[TensorType | BufferType]:
        """Returns symbolic graph inputs for padded encoder models."""
        del kv_params
        return padded_encoder_symbolic_inputs(device_refs=device_refs)

    def _pad_token_id(self) -> int:
        """Returns the pad token id used for ``collate_batch``."""
        return self.runtime.pad_token_id

    def prepare_initial_token_inputs(
        self,
        replica_batches: Sequence[Sequence[TextContext]],
        kv_cache_inputs: KVCacheInputsInterface[Buffer, Buffer] | None = None,
        return_n_logits: int = 1,
    ) -> InputsT:
        """Prepares padded token and attention-mask inputs for one replica."""
        del kv_cache_inputs, return_n_logits
        context_batch = single_replica_context_batch(
            replica_batches,
            processor_name=type(self).__qualname__,
        )
        device0 = self.runtime.devices[0]
        tokens = [ctx.tokens.active for ctx in context_batch]
        pad_value = self._pad_token_id()
        next_tokens_batch, _ = collate_batch(
            tokens,
            pad_value=pad_value,
            batch_size=len(tokens),
        )
        attention_mask = (next_tokens_batch != pad_value).astype(np.float32)
        return self._make_inputs(
            next_tokens_batch=Buffer.from_numpy(next_tokens_batch).to(device0),
            attention_mask=Buffer.from_numpy(attention_mask).to(device0),
        )

    def _make_inputs(
        self,
        *,
        next_tokens_batch: Buffer,
        attention_mask: Buffer,
    ) -> InputsT:
        """Constructs architecture-specific padded encoder model inputs."""
        raise NotImplementedError(
            f"{type(self).__qualname__} must implement _make_inputs"
        )

    def process_outputs(
        self, outputs: Sequence[Buffer | object]
    ) -> ModelOutputs:
        """Maps raw execution buffers to :class:`ModelOutputs`."""
        assert isinstance(outputs[0], Buffer)
        return ModelOutputs(logits=outputs[0])


def modulev3_gemma_multimodal_language_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    hidden_size: int,
    embedding_dtype: DType = DType.bfloat16,
) -> list[TensorType | BufferType]:
    """Symbolic language-model inputs for Gemma3 multimodal ModuleV3 compile."""
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    image_embeddings_type = TensorType(
        embedding_dtype,
        shape=["num_image_tokens", hidden_size],
        device=device_ref,
    )
    image_token_indices_type = TensorType(
        DType.int32,
        shape=["total_image_tokens"],
        device=device_ref,
    )
    input_row_offsets_type = TensorType(
        DType.uint32,
        shape=["input_row_offsets_len"],
        device=device_ref,
    )
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    return [
        tokens_type,
        return_n_logits_type,
        input_row_offsets_type,
        image_embeddings_type,
        image_token_indices_type,
        *kv_inputs,
    ]


def modulev3_idefics3_language_symbolic_inputs(
    *,
    kv_params: KVCacheParamInterface,
    device_ref: DeviceRef,
    hidden_size: int,
    embedding_dtype: DType,
) -> list[TensorType | BufferType]:
    """Symbolic language-model inputs for Idefics3 ModuleV3 compile."""
    tokens_type = TensorType(
        DType.int64, shape=["total_seq_len"], device=device_ref
    )
    input_row_offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    return_n_logits_type = TensorType(
        DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
    )
    image_embeddings_type = TensorType(
        embedding_dtype,
        shape=["num_image_tokens", hidden_size],
        device=device_ref,
    )
    image_token_indices_type = TensorType(
        DType.int32,
        shape=["total_image_tokens"],
        device=device_ref,
    )
    kv_inputs = kv_params.get_symbolic_inputs().flatten()
    return [
        tokens_type,
        input_row_offsets_type,
        return_n_logits_type,
        image_embeddings_type,
        image_token_indices_type,
        *kv_inputs,
    ]
