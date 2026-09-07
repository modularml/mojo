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
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from max._kv_cache_ops import (
    mha_decode_num_partitions,
    mla_dispatch_args_scalar,
)
from max.driver import Buffer, Device, DevicePinnedBuffer
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
)
from max.support.human_readable_formatter import to_human_readable_bytes
from max.support.math import ceildiv

from .data_parallelism_utils import split_into_groups
from .input_types import (
    KVCacheInputs,
    KVCacheInputsInterface,
    KVCacheInputsPerDevice,
    MultiKVCacheInputs,
)
from .utils import (
    AttnKeyInterface,
    MHAAttnKey,
    MLAAttnKey,
    MSAAttnKey,
    MultiAttnKey,
)

# Mirror of max.pipelines.speculative.config.SpeculativeMethod. Defined
# inline rather than imported because max.pipelines.speculative depends
# on max.nn (BUILD.bazel), so importing back would create a circular
# bazel dependency. The two definitions are structurally identical
# Literals, so mypy treats them as the same type at use sites.
SpeculativeMethod = Literal["eagle", "mtp", "dflash"]

KVHashAlgo = Literal["ahash64", "sha256", "sha256_64"]
"""Supported hash algorithms for KV-cache block identity."""

logger = logging.getLogger("max.pipelines")


def _filter_tiny_cache_lengths(
    probe_lengths: list[int], num_draft_tokens: int
) -> list[int]:
    min_cache_length = 1 + 2 * num_draft_tokens
    return [cl for cl in probe_lengths if cl >= min_cache_length]


@dataclass(frozen=True)
class KVCacheGroupId:
    """Identifies the caches a model reuses and evicts together.

    Caches behind the same attention pattern share a prefix-cache hit and an
    external tier namespace, so this doubles as the key for both.
    """

    type: Literal["full", "sliding_window"]
    window_size: int = -1

    def __post_init__(self):
        if self.type == "full":
            if self.window_size != -1:
                raise ValueError("Window size must be -1 for full groups.")
        elif self.type == "sliding_window":
            if self.window_size <= 0:
                raise ValueError(
                    "Window size must be positive for sliding window groups."
                )

    def is_sliding_window(self) -> bool:
        return self.type == "sliding_window"

    def is_full(self) -> bool:
        return self.type == "full"

    def blocks_in_window(self, page_size: int) -> int:
        if self.is_full():
            return -1
        return ceildiv(self.window_size - 1, page_size)

    @classmethod
    def full(cls) -> KVCacheGroupId:
        return cls(type="full")

    def __repr__(self) -> str:
        if self.type == "full":
            return "full_group"
        elif self.type == "sliding_window":
            return f"sliding_window_group({self.window_size})"


class KVConnectorType(str, Enum):
    """Identifies which off-device backing store the KV cache uses.

    Set on the connector config's ``type`` field to control whether evicted
    cache pages stay on device only, tier across host and disk, or route
    through a distributed block store.
    """

    null = "null"
    """No off-device backing store. Pages live on device only."""

    tiered = "tiered"
    """Tiers evicted pages across host memory and disk.

    Requires ``enable_prefix_caching``, ``host_offload_max_gb``,
    and a ``disk_offload_dir`` on the connector config.

    .. deprecated::
        A backward-compatible alias for :attr:`rust_tiered`; the Python
        implementation was removed.
    """

    rust_tiered = "rust_tiered"
    """Tiers evicted pages across host memory and disk, backed by the Rust
    ``kv_tier_connector`` extension.

    The only host/disk tiered implementation, and what :attr:`tiered` now
    resolves to: it runs its copies and disk I/O on Rust threads (no GIL
    contention) and overlaps onloads with GPU compute via asynchronous
    transfer handles. Requires ``enable_prefix_caching``,
    ``host_offload_max_gb``, and a ``disk_offload_dir`` on the connector
    config. Raises on non-CUDA/HIP devices.
    """

    dkv = "dkv"
    """Routes pages through a distributed KV block store.

    Requires a ``block_store_endpoint`` on the connector config.
    """


@runtime_checkable
class KVConnectorConfigInterface(Protocol):
    """The KV connector configuration contract: a type plus per-tier settings.

    Declared here because :class:`KVCacheParams` carries it, and implemented by
    the Pydantic ``KVConnectorConfig`` in the pipelines layer (which owns CLI
    and config-file parsing). Structural typing keeps ``max.nn`` free of a
    Pydantic dependency, which the base ``max`` wheel does not ship, while
    still giving every consumer a checked type instead of ``Any``.
    """

    @property
    def type(self) -> KVConnectorType:
        """Which off-device backing store to use."""
        ...

    @property
    def host_offload_max_gb(self) -> float | None:
        """Host budget in GiB; ``None`` sizes it from the device pool."""
        ...

    @property
    def disk_offload_max_gb(self) -> float | None:
        """Disk budget in GiB; ``None`` sizes it from the device pool."""
        ...

    @property
    def disk_offload_dir(self) -> str | None:
        """Disk cache directory; ``None`` means auto-create one."""
        ...

    @property
    def num_disk_workers(self) -> int:
        """Disk I/O worker threads for the tiered connectors."""
        ...

    @property
    def block_store_endpoint(self) -> str | None:
        """Endpoint for the co-located dKV service."""
        ...


@dataclass(frozen=True)
class NullKVConnectorConfig:
    """Connector config for no off-device backing store.

    The default for :attr:`KVCacheParams.kv_connector_config`, so the field is
    never ``None`` and every reader can go straight to ``.type``.
    """

    type: KVConnectorType = KVConnectorType.null
    host_offload_max_gb: float | None = None
    disk_offload_max_gb: float | None = None
    disk_offload_dir: str | None = None
    num_disk_workers: int = 32
    block_store_endpoint: str | None = None


def _validate_is_2d_uint8_buffer(buffer: Buffer) -> None:
    if len(buffer.shape) != 2:
        raise ValueError("KVCacheMemory buffer must have 2 dimensions")
    if buffer.dtype != DType.uint8:
        raise ValueError("KVCacheMemory buffer must have dtype uint8")


def _view_as_uint8_pages(buffer: Buffer) -> Buffer:
    """Re-view a KV buffer as a 2-D ``[num_pages, bytes_per_page]`` uint8 array.

    The original dtype and per-page element count are folded into a flat
    per-page byte stride so the offload engine and transfer engine can treat
    every cache uniformly regardless of dtype or shape.
    """
    return buffer.view(
        dtype=DType.uint8,
        shape=[
            buffer.shape[0],
            buffer.num_elements * buffer.dtype.size_in_bytes // buffer.shape[0],
        ],
    )


@dataclass
class KVCacheMemory:
    """One logical ``(child, kind)`` KV tensor as per-TP-shard ``uint8`` views.

    A unit is one logical tensor — a cache's ``values`` or its ``scales`` —
    holding a 2-D ``[num_pages, bytes_per_page]`` view per TP shard in canonical
    device order.

    ``replicated`` indicates that all buffers hold identical bytes. This is true
    for certain cases like TP + MLA, TP + MiniMaxM3IndexerAttn, etc.
    """

    replicated: bool
    buffers: list[Buffer]

    def __post_init__(self) -> None:
        if len(self.buffers) == 0:
            raise ValueError("KVCacheMemory must have at least one buffer")
        for buffer in self.buffers:
            _validate_is_2d_uint8_buffer(buffer)
        first_shape = self.buffers[0].shape
        for i, buffer in enumerate(self.buffers):
            if buffer.shape != first_shape:
                raise ValueError(
                    f"All buffers in a KVCacheMemory must share a shape, "
                    f"but shard {i} has shape {buffer.shape} vs shard 0's "
                    f"{first_shape}. bytes_per_page/total_num_pages are read "
                    f"off shard 0 and would silently report the wrong value "
                    f"for a mismatched shard."
                )
        if self.replicated and len(self.buffers) <= 1:
            raise ValueError(
                "replicated=True requires at least 2 TP-shard buffers"
            )

    @property
    def bytes_per_page(self) -> int:
        """Returns the per-page byte stride shared by every shard."""
        return self.buffers[0].shape[1]

    @property
    def host_bytes_per_page(self) -> int:
        """Returns the width of one host block row holding this unit's page.

        A replicated (MLA) unit contributes its stride once -- one copy is
        stored and broadcast back on load, so counting its peers would double
        the pinned host allocation. Must match across replicas, so a block
        written by one is readable by another.
        """
        return self.bytes_per_page * (
            1 if self.replicated else len(self.buffers)
        )

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages (including the null block)."""
        return self.buffers[0].shape[0]


@runtime_checkable
class KVCacheBufferInterface(Protocol):
    """Interface for a KV cache buffer (single leaf or a tree of leaves)."""

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages."""
        ...

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all buffers."""
        ...

    def to_memory(self) -> list[KVCacheMemory]:
        """Returns the offload-ready KV cache memory units, one per leaf kind."""
        ...


@dataclass
class MultiKVCacheBuffer(KVCacheBufferInterface):
    """A tree of KVCache buffers for one data-parallel replica.

    ``children`` maps a cache name (e.g. ``"target"``/``"draft"`` for
    speculative decoding, or ``"sliding"``/``"global"`` for hybrid models) to
    that cache's buffer for this replica.
    """

    children: dict[str, KVCacheBufferInterface]

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages."""
        first = next(iter(self.children.values()))
        return first.total_num_pages

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all buffers across every child cache."""
        bufs: list[Buffer] = []
        for child in self.children.values():
            bufs.extend(child.all_buffers)
        return bufs

    def to_memory(self) -> list[KVCacheMemory]:
        """Returns the offload-ready memory units for all children.

        Aggregated child-major, so a nested tree yields one unit per leaf cache
        per kind.
        """
        memories: list[KVCacheMemory] = []
        for child in self.children.values():
            memories.extend(child.to_memory())
        return memories


@dataclass
class KVCacheBuffer(KVCacheBufferInterface):
    """A collection of KVCache buffers for one data-parallel replica.

    Two buffer kinds are supported: ``values`` and (optionally, for FP8
    quantization) ``scales``. The length of each list corresponds to the
    tensor-parallel degree, with one buffer per TP shard.

    ``replicates_kv_across_tp`` is ``True`` when the KV data is replicated
    identically across TP shards and ``False`` when it is sharded. The data is
    replicated in certain cases like TP + MLA, TP + MiniMaxM3IndexerAttn, etc.
    """

    replicates_kv_across_tp: bool
    values: list[Buffer]
    scales: list[Buffer] | None = None
    values_per_layer: list[list[Buffer]] | None = None
    """Per-TP-shard, per-layer value buffers when the pool uses
    :attr:`~max.nn.kv_cache.KVCacheParams.per_layer_buffers`.

    ``values_per_layer[shard]`` is the list of single-layer buffers for that
    shard, and ``values[shard]`` aliases ``values_per_layer[shard][0]`` so the
    single-buffer ``values`` invariants (and consumers) stay valid. ``None``
    for a normal single multi-layer buffer."""
    scales_per_layer: list[list[Buffer]] | None = None
    """Per-TP-shard, per-layer scale buffers for a quantized KV cache backed by
    :attr:`~max.nn.kv_cache.KVCacheParams.per_layer_buffers` (mirrors
    :attr:`values_per_layer`). ``scales[shard]`` aliases
    ``scales_per_layer[shard][0]``. ``None`` for a single multi-layer scale
    buffer or an unquantized cache."""
    is_jenga: bool = False
    """Whether this buffer is associated with Jenga KV cache

    TODO: Delete this field after reworking KVCacheBufferInterface.
    """

    def __post_init__(self) -> None:
        all_buffers = self.all_buffers

        if len(self.values) == 0:
            raise ValueError("List of values must be non-empty")

        if self.values_per_layer is not None:
            if len(self.values_per_layer) != len(self.values):
                raise ValueError(
                    "values_per_layer must have one entry per TP shard"
                )
            for shard_layers, value in zip(
                self.values_per_layer, self.values, strict=True
            ):
                if len(shard_layers) == 0:
                    raise ValueError(
                        "each values_per_layer shard must be non-empty"
                    )
                if shard_layers[0] is not value:
                    raise ValueError(
                        "values[i] must alias values_per_layer[i][0]"
                    )

        if self.scales_per_layer is not None:
            assert self.scales is not None
            if len(self.scales_per_layer) != len(self.scales):
                raise ValueError(
                    "scales_per_layer must have one entry per TP shard"
                )
            for shard_layers, scale in zip(
                self.scales_per_layer, self.scales, strict=True
            ):
                if len(shard_layers) == 0:
                    raise ValueError(
                        "each scales_per_layer shard must be non-empty"
                    )
                if shard_layers[0] is not scale:
                    raise ValueError(
                        "scales[i] must alias scales_per_layer[i][0]"
                    )

        if self.replicates_kv_across_tp and len(self.values) <= 1:
            raise ValueError(
                "replicates_kv_across_tp=True requires at least 2 TP shards "
                "(len(values) > 1)"
            )

        unique_dtype = {b.dtype for b in self.values}
        if len(unique_dtype) > 1:
            raise ValueError("All values must have the same dtype")

        unique_shapes = {b.shape for b in self.values}
        if len(unique_shapes) > 1:
            raise ValueError("All values must have the same shape")

        unique_is_pinned = {
            isinstance(b, DevicePinnedBuffer) for b in all_buffers
        }
        if len(unique_is_pinned) > 1:
            raise ValueError(
                "All values (and scales if present) must be either all pinned "
                "or all non-pinned"
            )

        if self.scales is None:
            return

        if len(self.scales) != len(self.values):
            raise ValueError("Scales must be the same length as values")

        unique_dtype = {b.dtype for b in self.scales}
        if len(unique_dtype) > 1:
            raise ValueError("All scales must have the same dtype")

        unique_shapes = {b.shape for b in self.scales}
        if len(unique_shapes) > 1:
            raise ValueError("All scales must have the same shape")

        # Allow the number of pages to be different between values / scales only
        # for Jenga KV cache.
        # TODO: Get rid of this hack.
        unique_num_pages = {b.shape[0] for b in all_buffers}
        if not self.is_jenga and len(unique_num_pages) > 1:
            raise ValueError(
                "Values and scales must have the same number of pages"
            )
        for value, scale in zip(self.values, self.scales, strict=True):
            if value.device != scale.device:
                raise ValueError(
                    "Corresponding values and scales must be on the same device"
                )

    @property
    def total_num_pages(self) -> int:
        """Returns the total number of pages across all values and scales."""
        return self.values[0].shape[0]

    @property
    def all_buffers(self) -> list[Buffer]:
        """Returns all value and scale buffers in a single flat list.

        Returns:
            A list containing every value buffer followed by every scale
            buffer (if scales are present).
        """
        return [
            *self.values,
            *(self.scales if self.scales is not None else []),
        ]

    def to_memory(self) -> list[KVCacheMemory]:
        """Converts to offload-ready memory units, one per buffer kind.

        Every buffer is re-viewed as 2-D ``uint8`` pages so consumers can treat
        all caches uniformly regardless of dtype or shape.

        Per-layer buffers are deliberately not enumerated -- only each shard's
        layer-0 alias -- which is why ``allocate_buffers`` rejects
        ``per_layer_buffers`` alongside off-device connectors and DP > 1.

        Returns:
            One :class:`KVCacheMemory` per kind (values, and scales if present).
        """
        memories: list[KVCacheMemory] = []
        shard_lists: list[list[Buffer]] = [self.values]
        if self.scales is not None:
            shard_lists.append(self.scales)
        for shards in shard_lists:
            memories.append(
                KVCacheMemory(
                    replicated=self.replicates_kv_across_tp,
                    buffers=[_view_as_uint8_pages(b) for b in shards],
                )
            )
        return memories


@dataclass
class KVCacheQuantizationConfig:
    """Configuration for KVCache quantization.

    Currently only FP8 Quantization is supported.
    """

    scale_dtype: DType = DType.float32
    """Data type of quantization scales, if quantization is enabled"""

    quantization_granularity: int = 128
    """Block-size used for KVCache quantization along head-dimension (e.g. 128)."""


@dataclass(frozen=True)
class BatchCharacteristics:
    """Upper-bound batch shape used to prepare decode attention metadata.

    Captures the ``(batch_size, max_prompt_length, max_cache_valid_length)`` a
    decode forward should prepare its attention dispatch metadata *for*, which
    may exceed the batch's real per-request values.
    :meth:`PagedKVCacheManager.runtime_inputs` uses it to resolve the dispatch
    key once: e.g. for graph-capture replay, ``max_cache_valid_length`` is
    aligned up to a cache length recorded during capture and every data-parallel
    replica must run the identical captured graph. The batch's real values must
    not exceed these.
    """

    batch_size: int
    max_prompt_length: int
    max_cache_valid_length: int


@dataclass
class KVCacheAssignments:
    """Assignments of request blocks to KV cache pages for a replica.

    ``batch_characteristics`` carries the effective ``(batch_size,
    max_prompt_length, max_cache_valid_length)`` used to build this
    assignment (after any graph-capture upper-bound override) so that
    :meth:`KVCacheParamInterface.build_runtime_inputs` can resolve the decode
    attention dispatch keys from the same values.
    """

    cache_lengths_by_device: list[Buffer]
    lookup_table_by_device: list[dict[str, Buffer]]
    max_prompt_length: Buffer
    max_cache_length: Buffer
    batch_characteristics: BatchCharacteristics


@dataclass(frozen=True)
class KVLeafRegion:
    leaf_id: str
    group_id: KVCacheGroupId
    bytes_per_page: int


@runtime_checkable
class KVCacheParamInterface(Protocol):
    """Interface for KV cache parameters."""

    page_size: int
    data_parallel_degree: int
    devices: Sequence[DeviceRef]
    kv_connector_config: KVConnectorConfigInterface
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0

    @property
    def n_devices(self) -> int:
        """Returns the total number of devices."""
        ...

    @property
    def enable_prefix_caching(self) -> bool:
        """Whether prefix caching is enabled."""
        ...

    @property
    def enable_dp_cross_replica_prefix_copy(self) -> bool:
        """Whether a prefix-cache hit resident on another data-parallel
        replica's device may be served by a device-to-device copy."""
        ...

    @property
    def num_draft_tokens_per_step(self) -> int:
        """Number of draft tokens written per draft forward.

        Zero when speculative decoding is disabled; one for autoregressive
        drafts (``eagle``, ``mtp``); equal to ``num_draft_tokens`` for block
        drafts (``dflash``).
        """
        if self.speculative_method is None:
            return 0
        elif self.speculative_method == "dflash":
            return self.num_draft_tokens
        elif self.speculative_method in ("mtp", "eagle"):
            return 1
        else:
            raise ValueError(
                f"Unrecognized speculative_method: {self.speculative_method!r}"
            )

    @property
    def bytes_per_block(self) -> int:
        """Number of bytes per cache block."""
        ...

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> KVCacheInputsInterface[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache.

        Args:
            namespace: Prefix that disambiguates this cache's page-pool
                symbolic dim from sibling caches in a multi-group tree. Empty
                for a single-group cache, leaving its names unchanged.
        """
        ...

    def flattened_kv_inputs(self) -> list[TensorType | BufferType]:
        """Flattens the symbolic inputs for the KV cache."""
        return self.get_symbolic_inputs().flatten()

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> KVCacheInputsInterface[TensorValue, BufferValue]:
        """Unflattens the symbolic inputs for the KV cache."""
        ...

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        ...

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree."""
        ...

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode dispatch shape for the given shape.

        Returns a :class:`AttnKeyInterface` for a single cache, or a
        :class:`MultiAttnKey` tree mirroring the cache tree.
        """
        ...

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the cache lengths to probe during decode graph capture."""
        ...

    @property
    def kv_hash_algo(self) -> KVHashAlgo:
        """Hash algorithm used for KV-cache block identity."""
        ...

    @property
    def kv_hash_seed(self) -> bytes | None:
        """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64."""
        ...

    def allocate_buffers(
        self, total_num_pages: int
    ) -> Sequence[KVCacheBufferInterface]:
        """Allocates the buffers for the KV cache."""
        ...

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache inputs spanning all replicas.

        ``assignments`` and ``buffers`` are indexed by data-parallel replica.
        Returns a single :class:`KVCacheInputs` leaf (or a
        :class:`MultiKVCacheInputs` tree) whose leaves each hold every
        ``(replica, TP shard)`` device's inputs."""
        ...

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        ...

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves of the KV cache."""
        ...

    def slab_to_buffer_views(
        self, buffers: Sequence[Buffer]
    ) -> KVCacheBufferInterface:
        """Converts a slab of memory into a buffer view."""
        ...


@dataclass
class KVCacheParams(KVCacheParamInterface):
    """Configuration parameters for key-value cache management in transformer models.

    This class encapsulates all configuration options for managing KV caches during
    inference, including parallelism settings, and memory management.
    """

    dtype: DType
    """Data type for storing key and value tensors in the cache."""

    head_dim: int
    """Dimensionality of each attention head."""

    num_layers: int
    """Number of layers in the model."""

    devices: Sequence[DeviceRef]
    """Devices to use for the KV cache."""

    enable_prefix_caching: bool = False
    """Whether to enable prefix caching for efficient reuse of common prompt prefixes."""

    enable_dp_cross_replica_prefix_copy: bool = True
    """Whether a prefix-cache block resident on another data-parallel (DP)
    replica's device may be materialized locally via a device-to-device copy
    to serve a cache hit. When False, cross-replica reuse is only served from
    the shared external tier via the KV connector (or recomputed). Only
    relevant when ``data_parallel_degree > 1`` and prefix caching is enabled."""

    per_layer_buffers: bool = False
    """When ``True``, allocate one standalone single-layer buffer per layer
    instead of one ``[..., num_layers, ...]`` multi-layer buffer.

    Each attention dispatch then binds only its own per-layer buffer, so the
    pool total can exceed a per-allocation size cap (e.g. a device's maximum
    single allocation) while every individual buffer stays under it. Defaults
    to ``False`` (one multi-layer buffer), keeping all other backends and
    models byte-identical."""

    kv_hash_algo: KVHashAlgo = "ahash64"
    """Hash algorithm used for KV-cache block identity."""

    kv_hash_seed: bytes | None = None
    """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64.

    Set by ``KVCacheConfig.to_params`` via ``resolve_kv_hash_seed``.
    """

    kv_connector_config: KVConnectorConfigInterface = field(
        default_factory=NullKVConnectorConfig
    )
    """Connector configuration: the connector type and its settings. The
    default is a ``null`` connector (no external caching)."""

    page_size: int = 128
    """Number of tokens per page (block).

    This value is expressed in tokens, not bytes. The byte footprint of a page is
    derived from pipeline configuration.

    Current constraints: the page size must be a multiple of 128 and at least 128.
    """

    data_parallel_degree: int = 1
    """Degree of data parallelism. Devices are grouped replica-major, with
    ``n_devices // data_parallel_degree`` TP shards per replica."""

    kvcache_quant_config: KVCacheQuantizationConfig | None = None
    """KVCache quantization config. Currently only FP8 quantization supported."""

    speculative_method: SpeculativeMethod | None = None
    """Speculative decoding method propagated from
    SpeculativeConfig"""

    num_draft_tokens: int = 0
    """Total draft tokens generated per speculative iteration.

    Zero when no speculative decoding is configured."""

    window_size: int | None = None
    """Window size for the sliding window attention. None for global attention."""

    def __post_init__(self):
        """Validates configuration and computes derived fields after initialization.

        Raises:
            ValueError: If configuration parameters are invalid or incompatible.
        """
        if self.data_parallel_degree < 1:
            raise ValueError(
                f"Data parallelism degree ({self.data_parallel_degree})"
                " must be at least 1"
            )

        if self.n_devices < self.data_parallel_degree:
            raise ValueError(
                f"Data parallelism degree ({self.data_parallel_degree})"
                " cannot be greater than the number of devices"
                f" ({self.n_devices})"
            )

        if self.n_devices % self.data_parallel_degree != 0:
            raise ValueError(
                f"Number of devices ({self.n_devices}) must be divisible by"
                " data parallelism degree"
                f" ({self.data_parallel_degree})"
            )

        # Validate connector configuration
        connector = self.kv_connector_config.type
        if connector in (
            KVConnectorType.tiered,
            KVConnectorType.rust_tiered,
        ):
            if not self.enable_prefix_caching:
                raise ValueError(
                    f"KV connector '{connector.value}' requires prefix"
                    " caching to be enabled"
                )

        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            # Validate FP8 KVCache quantization granularity.
            if (
                self.head_dim
                % self.kvcache_quant_config.quantization_granularity
                != 0
            ):
                raise ValueError(
                    "KVCache quantization granularity must evenly divide KV"
                    " head dimension."
                )
            if self.kvcache_quant_config is None:
                raise ValueError("KVCache quantization config required.")

    @cached_property
    def devices_per_replica(self) -> Sequence[Sequence[DeviceRef]]:
        """Returns the devices per replica."""
        return split_into_groups(self.devices, self.data_parallel_degree)

    @cached_property
    def _primary_device(self) -> Device | None:
        """Concrete primary device for decode-dispatch kernels.

        The decode dispatch custom ops are GPU kernels needing a concrete
        :class:`~max.driver.Device`. Built lazily (and cached) so constructing
        params for a GPU ``DeviceRef`` on a CPU-only host does not require a
        device context. Returns ``None`` on a CPU-only host; callers then fall
        back to the sentinel dispatch key (``num_partitions=1``).
        """
        device_ref = self.devices[0]
        if device_ref.is_cpu():
            return None
        return device_ref.to_device()

    @property
    def is_fp8_kv_dtype(self) -> bool:
        """Whether the KV cache stores FP8 data, for dispatch resolution.

        Unlike ``quantized_kv_cache`` (which also requires valid scale config),
        this checks only the storage dtype—matching the compile-time detection
        in the MLA decode kernel.

        TODO(SERVOPT-1094): Once SnapMLA uses a valid scale_dtype, this
        can be replaced by ``quantized_kv_cache``.
        """
        return self.dtype in (DType.float8_e4m3fn, DType.float8_e4m3fnuz)

    @property
    def quantized_kv_cache(self) -> bool:
        """Returns whether KV cache quantization is enabled."""
        # Supported quantized-KV storage schemes: FP8_E4M3 (fp32 / e8m0 scales)
        # and int8 (fp16 per-block absmax scales).
        if self.kvcache_quant_config is None:
            return False
        value_dtypes = (
            DType.float8_e4m3fn,
            DType.float8_e4m3fnuz,
            DType.int8,
        )
        scale_dtypes = (
            DType.float32,
            DType.float8_e8m0fnu,
            DType.float16,
        )
        return (
            self.dtype in value_dtypes
            and self.kvcache_quant_config.scale_dtype in scale_dtypes
        )

    @property
    def kv_cache_scale_dtype(self) -> DType:
        """Returns the dtype of the KV cache scales.

        Returns:
            The dtype of the KV cache scales.
        """
        if self.quantized_kv_cache and self.kvcache_quant_config is not None:
            return self.kvcache_quant_config.scale_dtype
        else:
            return DType.float32

    @property
    def n_devices(self) -> int:
        """Returns the number of devices.

        Returns:
            The number of devices.
        """
        return len(self.devices)

    @n_devices.setter  # Required for protocol.
    def n_devices(self, value: int) -> None:
        raise ValueError("n_devices is read-only")

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree.

        Returns:
            The tensor parallel degree.
        """
        return self.n_devices // self.data_parallel_degree

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        raise NotImplementedError

    @property
    def dtype_shorthand(self) -> str:
        """Returns a shorthand textual representation of the data type.

        Returns:
            "bf16" for bfloat16 dtype, "f32" otherwise.
        """
        if self.dtype == DType.bfloat16:
            return "bf16"
        elif self.dtype == DType.float8_e4m3fn:
            return "f8_m4e3fn"
        else:
            return "f32"

    @property
    def kv_dim(self) -> int:
        raise NotImplementedError

    @property
    def n_kv_heads_per_device(self) -> int:
        raise NotImplementedError

    @property
    def shape_per_block(self) -> list[int]:
        """Returns the shape of each cache block.

        Returns:
            The shape of the cache block.
        """
        # split k and v caches across a single dim
        # 0 = key
        # 1 = value
        return [
            self.kv_dim,
            self.num_layers,
            self.page_size,
            self.n_kv_heads_per_device,
            self.head_dim,
        ]

    @property
    def shape_per_layer_block(self) -> list[int]:
        """Returns the block shape for a single-layer buffer.

        Same as :attr:`shape_per_block` but with the layer dimension pinned to
        ``1``. Used when :attr:`per_layer_buffers` is set: the pool allocates
        ``num_layers`` such buffers per device instead of one multi-layer
        buffer. The attention kernel derives ``num_layers`` from this dim, so a
        single-layer buffer (``num_layers == 1``) with ``layer_idx == 0`` is
        self-consistent.
        """
        kv_dim, _num_layers, page_size, n_kv_heads, head_dim = (
            self.shape_per_block
        )
        return [kv_dim, 1, page_size, n_kv_heads, head_dim]

    @property
    def shape_per_scale_block(self) -> list[int]:
        """Returns the shape of each scale block used for KVCache quantization

        Returns:
            The shape of the KVCache quantization scales block.
        """
        assert self.kvcache_quant_config is not None
        shape_per_block = self.shape_per_block
        # The final dimension is ceil(head_dim / quantization_granularity).
        granularity = self.kvcache_quant_config.quantization_granularity
        shape_per_block[4] = math.ceil(shape_per_block[4] / granularity)
        return shape_per_block

    @property
    def shape_per_layer_scale_block(self) -> list[int]:
        """Scale-block shape for a single-layer buffer (layer dim pinned to 1).

        The scale analog of :attr:`shape_per_layer_block`: used with
        :attr:`per_layer_buffers` on a quantized cache, where the pool allocates
        one single-layer scale buffer per layer instead of one multi-layer one.
        """
        shape = self.shape_per_scale_block
        shape[1] = 1
        return shape

    @property
    def bytes_per_block(self) -> int:
        """Returns the number of bytes per cache block.

        When TP>1, each block is sharded across the devices in the tensor parallel group.
        This method returns the total memory needed to store a block across these devices.
        Includes memory needed for scales if quantization is enabled.

        Returns:
            The number of bytes per cache block.
        """
        return self.bytes_per_value_block + self.bytes_per_scale_block

    @property
    def bytes_per_value_block(self) -> int:
        """Returns the number of bytes per value block."""
        return (
            math.prod(self.shape_per_block)
            * self.dtype.size_in_bytes
            * self.tensor_parallel_degree
        )

    @property
    def bytes_per_scale_block(self) -> int:
        """Returns the number of bytes per scale block."""
        if not (
            self.quantized_kv_cache and self.kvcache_quant_config is not None
        ):
            return 0
        return (
            math.prod(self.shape_per_scale_block)
            * self.kvcache_quant_config.scale_dtype.size_in_bytes
            * self.tensor_parallel_degree
        )

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        raise NotImplementedError

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> KVCacheInputs[TensorType, BufferType]:
        """Computes the symbolic inputs for the KV cache.

        Args:
            namespace: Prefix disambiguating this cache's per-pool page-count
                dim from sibling caches in a multi-group tree (empty for a
                single-group cache).

        Returns:
            The symbolic inputs for the KV cache.
        """
        input_symbols: list[KVCacheInputsPerDevice[TensorType, BufferType]] = []
        for replica_idx in range(len(self.devices_per_replica)):
            prefix = f"replica_{replica_idx}_"
            symbols = self._get_symbolic_inputs_for_replica(
                replica_idx, prefix, namespace
            )
            input_symbols.extend(symbols)
        return KVCacheInputs(inputs=input_symbols)

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> KVCacheInputs[TensorValue, BufferValue]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        return self.get_symbolic_inputs().unflatten(it)

    def allocate_buffers(self, total_num_pages: int) -> list[KVCacheBuffer]:
        """Allocates the buffers for the KV cache."""
        if self.per_layer_buffers:
            # Validate the per-layer configuration before materializing any
            # device buffers. These guards live here, not in ``__post_init__``,
            # because call sites set ``per_layer_buffers`` after construction.
            if self.num_layers < 1:
                # ``values`` aliases layer 0, so the pool needs at least one
                # layer; otherwise ``layer_buffers[0]`` below raises an opaque
                # IndexError.
                raise ValueError(
                    f"per_layer_buffers requires num_layers >= 1, got {self.num_layers}"
                )
            connector = self.kv_connector_config.type
            if connector in (
                KVConnectorType.tiered,
                KVConnectorType.rust_tiered,
                KVConnectorType.dkv,
            ):
                # KVCacheBuffer.all_buffers / to_memory enumerate only the
                # layer-0 alias, so an off-device connector would move layers
                # 1..N-1 nowhere. Reject until that enumeration covers per-layer
                # buffers.
                raise NotImplementedError(
                    "per_layer_buffers is not supported with an off-device KV"
                    f" connector ('{connector.value}')"
                )
            if self.data_parallel_degree > 1:
                # Cross-replica block copy enumerates the same layer-0 alias.
                raise NotImplementedError(
                    "per_layer_buffers is not supported with data parallelism"
                    " (data_parallel_degree > 1)"
                )
        # ``Buffer.zeros`` needs concrete devices, so materialize the per-replica
        # device groups from the ``DeviceRef``s here.
        devices_per_replica = split_into_groups(
            x=[d.to_device() for d in self.devices],
            groups=self.data_parallel_degree,
        )
        kv_cache_buffers: list[KVCacheBuffer] = []
        for devices in devices_per_replica:
            values: list[Buffer] = []
            values_per_layer: list[list[Buffer]] | None = None
            if self.per_layer_buffers:
                # One standalone single-layer buffer per layer (num_layers==1 in
                # dim-2). ``values`` aliases each shard's layer-0 buffer so the
                # single-buffer invariants and consumers stay valid.
                values_per_layer = []
                for device in devices:
                    layer_buffers = [
                        Buffer.zeros(
                            shape=[
                                total_num_pages,
                                *self.shape_per_layer_block,
                            ],
                            dtype=self.dtype,
                            device=device,
                        )
                        for _ in range(self.num_layers)
                    ]
                    values_per_layer.append(layer_buffers)
                    values.append(layer_buffers[0])
            else:
                for device in devices:
                    value = Buffer.zeros(
                        shape=[total_num_pages, *self.shape_per_block],
                        dtype=self.dtype,
                        device=device,
                    )
                    values.append(value)

            scales: list[Buffer] | None = None
            scales_per_layer: list[list[Buffer]] | None = None
            if self.quantized_kv_cache:
                scales = []
                assert self.kvcache_quant_config is not None
                scale_dtype = self.kvcache_quant_config.scale_dtype
                if self.per_layer_buffers:
                    # One single-layer scale buffer per layer, parallel to
                    # ``values_per_layer``. ``scales`` aliases each shard's
                    # layer-0 scale so single-buffer consumers stay valid.
                    scales_per_layer = []
                    for device in devices:
                        layer_scales = [
                            Buffer.zeros(
                                shape=[
                                    total_num_pages,
                                    *self.shape_per_layer_scale_block,
                                ],
                                dtype=scale_dtype,
                                device=device,
                            )
                            for _ in range(self.num_layers)
                        ]
                        scales_per_layer.append(layer_scales)
                        scales.append(layer_scales[0])
                else:
                    for device in devices:
                        scale = Buffer.zeros(
                            shape=[
                                total_num_pages,
                                *self.shape_per_scale_block,
                            ],
                            dtype=scale_dtype,
                            device=device,
                        )
                        scales.append(scale)

            kv_cache_buffer = KVCacheBuffer(
                values=values,
                scales=scales,
                replicates_kv_across_tp=self.replicates_kv_across_tp,
                values_per_layer=values_per_layer,
                scales_per_layer=scales_per_layer,
            )
            kv_cache_buffers.append(kv_cache_buffer)
        return kv_cache_buffers

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        raise NotImplementedError

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache leaf spanning all replicas.

        ``assignments`` and ``buffers`` are indexed by data-parallel replica.
        The returned :class:`KVCacheInputs` lists one
        :class:`KVCacheInputsPerDevice` per ``(replica, TP shard)``, in the
        same replica-major order as :meth:`get_symbolic_inputs`.
        """
        tp_shards: list[KVCacheInputsPerDevice[Buffer, Buffer]] = []
        for assignment, buffer in zip(assignments, buffers, strict=True):
            assert isinstance(buffer, KVCacheBuffer)
            bc = assignment.batch_characteristics
            batch_size = bc.batch_size
            max_cl = bc.max_cache_valid_length

            target_key = self.resolve_attn_key(
                batch_size, bc.max_prompt_length, max_cl
            )
            draft_key = (
                self.resolve_attn_key(
                    batch_size, self.num_draft_tokens_per_step, max_cl
                )
                if self.speculative_method is not None
                else None
            )

            for i, (cl, luts, blocks) in enumerate(
                zip(
                    assignment.cache_lengths_by_device,
                    assignment.lookup_table_by_device,
                    buffer.values,
                    strict=True,
                )
            ):
                device = blocks.device
                lut = luts[_prefix + str(self.group_id)]
                kv_scales = (
                    buffer.scales[i] if buffer.scales is not None else None
                )
                scales_lut = (
                    luts[_prefix + str(self.group_id) + "/scales"]
                    if buffer.scales is not None
                    else None
                )
                blocks_per_layer = (
                    buffer.values_per_layer[i]
                    if buffer.values_per_layer is not None
                    else None
                )
                scales_per_layer = (
                    buffer.scales_per_layer[i]
                    if buffer.scales_per_layer is not None
                    else None
                )
                tp_shards.append(
                    self._build_kvcache_inputs_per_device(
                        device,
                        blocks,
                        cl,
                        lut,
                        assignment.max_prompt_length,
                        assignment.max_cache_length,
                        kv_scales,
                        scales_lut,
                        target_key,
                        draft_key,
                        max_cl,
                        blocks_per_layer=blocks_per_layer,
                        scales_per_layer=scales_per_layer,
                    )
                )
        return KVCacheInputs(inputs=tp_shards)

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        raise ValueError(
            "Unflattening a basic KV tree is only supported for MultiKVCacheParams"
        )

    @property
    def group_id(self) -> KVCacheGroupId:
        if self.window_size is not None:
            return KVCacheGroupId(
                type="sliding_window", window_size=self.window_size
            )
        else:
            return KVCacheGroupId(type="full")

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves of the KV cache."""
        leaves = {
            _prefix + str(self.group_id): KVLeafRegion(
                leaf_id=_prefix + str(self.group_id),
                group_id=self.group_id,
                bytes_per_page=self.bytes_per_value_block,
            )
        }

        if self.quantized_kv_cache:
            leaves[_prefix + str(self.group_id) + "/scales"] = KVLeafRegion(
                leaf_id=_prefix + str(self.group_id) + "/scales",
                group_id=self.group_id,
                bytes_per_page=self.bytes_per_scale_block,
            )

        return leaves

    def slab_to_buffer_views(
        self, buffers: Sequence[Buffer]
    ) -> KVCacheBufferInterface:
        """Converts a slab of memory into a buffer view.

        This is used by the Jenga KV cache manager.
        """

        def _view(b: Buffer, shape: Sequence[int], dtype: DType) -> Buffer:
            total_bytes = b.num_elements * b.dtype.size_in_bytes
            bytes_per_page = math.prod(shape) * dtype.size_in_bytes
            num_little_pages = total_bytes // bytes_per_page
            assert num_little_pages * bytes_per_page == total_bytes
            return b.view(shape=(num_little_pages, *shape), dtype=dtype)

        quant_config = self.kvcache_quant_config
        return KVCacheBuffer(
            replicates_kv_across_tp=self.replicates_kv_across_tp,
            values=[
                _view(b, self.shape_per_block, self.dtype) for b in buffers
            ],
            scales=[
                _view(b, self.shape_per_scale_block, quant_config.scale_dtype)
                for b in buffers
            ]
            if self.quantized_kv_cache and quant_config is not None
            else None,
            is_jenga=True,
        )


@dataclass(kw_only=True)
class MHAKVCacheParams(KVCacheParams):
    n_kv_heads: int
    """Total number of key-value attention heads across all devices."""

    allow_kv_head_replication: bool = False
    """Allow TP wider than ``n_kv_heads``: when set and ``n_devices`` is a
    multiple of ``n_kv_heads``, replicate each KV head across a group of
    devices (``n_kv_heads_per_device == 1``)."""

    def __post_init__(self) -> None:
        super().__post_init__()
        tp_degree = self.tensor_parallel_degree
        if self.n_kv_heads % tp_degree == 0:
            return
        # Fewer heads than devices: replicate each head across a device group.
        if self.allow_kv_head_replication and tp_degree % self.n_kv_heads == 0:
            return
        raise ValueError(
            f"Number of KV heads ({self.n_kv_heads}) must be divisible by"
            f" the tensor parallel degree ({tp_degree})"
        )

    @property
    def kv_dim(self) -> int:
        return 2

    @property
    def n_kv_heads_per_device(self) -> int:
        tp_degree = self.tensor_parallel_degree
        if self.n_kv_heads % tp_degree == 0:
            return max(self.n_kv_heads // tp_degree, 1)
        # ``allow_kv_head_replication``: each head spans a group of devices.
        return 1

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return False

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape.

        Args:
            batch_size: Number of requests in the decode batch.
            max_prompt_length: Per-step query width (``1`` for plain decode,
                ``1 + num_spec_tokens`` for speculative verify).
            max_cache_valid_length: Maximum valid cache length in the batch.

        Returns:
            The resolved :class:`~max.nn.kv_cache.AttnKeyInterface`
        """
        device = self._primary_device
        if batch_size <= 0 or device is None:
            # Sentinel for empty / degenerate replicas or a CPU-only host;
            # skip the GPU dispatch kernel.
            num_partitions = 1
        else:
            num_partitions = mha_decode_num_partitions(
                batch_size,
                max_cache_valid_length,
                self.n_kv_heads_per_device,
                device,
            )
        return MHAAttnKey(
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            num_partitions=num_partitions,
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        granularity = 256
        probe_lengths = (
            [1]
            + list(range(granularity, max_cache_length, granularity))
            + [max_cache_length]
        )
        return _filter_tiny_cache_lengths(probe_lengths, self.num_draft_tokens)

    def _attn_metadata_buffer(self, device: DeviceRef) -> TensorType:
        # MHA decode kernels read a 4-int dispatch buffer on the host (CPU),
        # matching ``MHAAttnKey.pack_into_buffer``. ``device`` is accepted so
        # subclasses can emit device-resident metadata of a different shape.
        return TensorType(DType.int64, shape=[4], device=DeviceRef.CPU())

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]
        # Sibling cache groups may size their page pools independently.
        page_dim = page_namespace + "total_num_pages"

        def _blocks_per_layer(
            device: DeviceRef,
        ) -> list[BufferType] | None:
            # One single-layer BufferType per layer. Must stay in exact
            # lock-step with ``flatten``/``unflatten`` (tail order) and with the
            # runtime buffers built by ``allocate_buffers`` / ``build_runtime_inputs``.
            if not self.per_layer_buffers:
                return None
            return [
                BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_layer_block],
                    device=device,
                )
                for _ in range(self.num_layers)
            ]

        def _scales_per_layer(
            device: DeviceRef,
        ) -> list[BufferType] | None:
            # Scale analog of ``_blocks_per_layer`` (per-layer + quantized).
            # Same lock-step requirement; appended after kv_blocks_per_layer.
            if not (self.per_layer_buffers and self.quantized_kv_cache):
                return None
            return [
                BufferType(
                    self.kv_cache_scale_dtype,
                    shape=[page_dim, *self.shape_per_layer_scale_block],
                    device=device,
                )
                for _ in range(self.num_layers)
            ]

        def _kv_blocks(device: DeviceRef) -> BufferType:
            # ``per_layer_buffers`` aliases ``kv_blocks`` to the first per-layer
            # buffer so single-buffer consumers stay valid.
            if self.per_layer_buffers:
                return BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_layer_block],
                    device=device,
                )
            return BufferType(
                self.dtype,
                shape=[page_dim, *self.shape_per_block],
                device=device,
            )

        return [
            KVCacheInputsPerDevice(
                kv_blocks=_kv_blocks(device),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[
                        prefix + "batch_size",
                        prefix + page_namespace + "max_num_pages",
                    ],
                    device=device,
                ),
                max_prompt_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                max_cache_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                kv_scales=BufferType(
                    self.kv_cache_scale_dtype,
                    # Per-layer buffers alias ``kv_scales`` to a single-layer
                    # scale (mirrors ``_kv_blocks`` for the KV data).
                    shape=[
                        page_dim,
                        *(
                            self.shape_per_layer_scale_block
                            if self.per_layer_buffers
                            else self.shape_per_scale_block
                        ),
                    ],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                attention_dispatch_metadata=self._attn_metadata_buffer(device),
                draft_attention_dispatch_metadata=self._attn_metadata_buffer(
                    device
                )
                if self.speculative_method is not None
                else None,
                kv_blocks_per_layer=_blocks_per_layer(device),
                kv_scales_per_layer=_scales_per_layer(device),
            )
            for device in devices
        ]

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            scales_lookup_table=scales_lookup_table,
            attention_dispatch_metadata=target_key.pack_into_buffer(
                device, max_cache_valid_length
            ),
            draft_attention_dispatch_metadata=draft_key.pack_into_buffer(
                device, max_cache_valid_length
            )
            if draft_key is not None
            else None,
            kv_blocks_per_layer=blocks_per_layer,
            kv_scales_per_layer=scales_per_layer,
        )


@dataclass(kw_only=True)
class MLAKVCacheParams(KVCacheParams):
    num_q_heads: int
    """Number of query attention heads, required so the MLA decode kernel can
    resolve its dispatch metadata."""

    def __post_init__(self) -> None:
        super().__post_init__()
        tp_degree = self.tensor_parallel_degree
        if self.num_q_heads % tp_degree != 0:
            raise ValueError(
                f"Number of query heads ({self.num_q_heads}) must be"
                " divisible by the tensor parallel degree"
                f" ({tp_degree})"
            )

    @property
    def kv_dim(self) -> int:
        return 1

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return self.tensor_parallel_degree > 1

    @property
    def n_kv_heads_per_device(self) -> int:
        return 1

    @property
    def num_q_heads_per_device(self) -> int:
        return max(self.num_q_heads // self.tensor_parallel_degree, 1)

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape.

        Args:
            batch_size: Number of requests in the decode batch.
            max_prompt_length: Per-step query width (``1`` for plain decode,
                ``1 + num_spec_tokens`` for speculative verify).
            max_cache_valid_length: Maximum valid cache length in the batch.

        Returns:
            The resolved :class:`~max.nn.kv_cache.AttnKeyInterface`
        """
        device = self._primary_device
        if batch_size <= 0 or device is None:
            # Sentinel for empty / degenerate replicas or a CPU-only host;
            # skip the GPU dispatch kernel.
            return MLAAttnKey(
                batch_size=batch_size,
                max_prompt_length=max_prompt_length,
                num_partitions=1,
            )
        # ``mla_dispatch_args_scalar`` may adjust batch_size / max_prompt_length
        # alongside the resolved num_partitions; carry the adjusted values.
        adj_batch_size, adj_max_prompt_length, num_partitions = (
            mla_dispatch_args_scalar(
                batch_size,
                max_cache_valid_length,
                max_prompt_length,
                self.num_q_heads_per_device,
                self.is_fp8_kv_dtype,
                device,
            )
        )
        return MLAAttnKey(
            batch_size=int(adj_batch_size),
            max_prompt_length=int(adj_max_prompt_length),
            num_partitions=int(num_partitions),
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        granularity = 64
        probe_lengths = (
            [1]
            + list(range(granularity, max_cache_length, granularity))
            + [max_cache_length]
        )
        return _filter_tiny_cache_lengths(probe_lengths, self.num_draft_tokens)

    def _get_symbolic_inputs_for_replica(
        self, replica_idx: int, prefix: str, page_namespace: str = ""
    ) -> list[KVCacheInputsPerDevice[TensorType, BufferType]]:
        devices = self.devices_per_replica[replica_idx]
        # Sibling cache groups may size their page pools independently.
        page_dim = page_namespace + "total_num_pages"

        return [
            KVCacheInputsPerDevice(
                kv_blocks=BufferType(
                    self.dtype,
                    shape=[page_dim, *self.shape_per_block],
                    device=device,
                ),
                cache_lengths=TensorType(
                    DType.uint32,
                    shape=[prefix + "batch_size"],
                    device=device,
                ),
                lookup_table=TensorType(
                    DType.uint32,
                    shape=[
                        prefix + "batch_size",
                        prefix + page_namespace + "max_num_pages",
                    ],
                    device=device,
                ),
                max_prompt_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                max_cache_length=TensorType(
                    DType.uint32,
                    shape=[1],
                    device=DeviceRef.CPU(),
                ),
                kv_scales=BufferType(
                    self.kv_cache_scale_dtype,
                    shape=[page_dim, *self.shape_per_scale_block],
                    device=device,
                )
                if self.quantized_kv_cache
                else None,
                # MLA decode kernels read a 3-int dispatch buffer on the
                # accelerator, matching ``MLAAttnKey.pack_into_buffer``.
                attention_dispatch_metadata=TensorType(
                    DType.int64, shape=[3], device=device
                ),
                draft_attention_dispatch_metadata=TensorType(
                    DType.int64, shape=[3], device=device
                )
                if self.speculative_method is not None
                else None,
                mla_num_partitions=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                ),
                draft_mla_num_partitions=TensorType(
                    DType.int64, shape=[1], device=DeviceRef.CPU()
                )
                if self.speculative_method is not None
                else None,
            )
            for device in devices
        ]

    def _build_kvcache_inputs_per_device(
        self,
        device: Device,
        blocks: Buffer,
        cache_lengths: Buffer,
        lookup_table: Buffer,
        max_prompt_length: Buffer,
        max_cache_length: Buffer,
        kv_scales: Buffer | None,
        scales_lookup_table: Buffer | None,
        target_key: AttnKeyInterface,
        draft_key: AttnKeyInterface | None,
        max_cache_valid_length: int,
        blocks_per_layer: list[Buffer] | None = None,
        scales_per_layer: list[Buffer] | None = None,
    ) -> KVCacheInputsPerDevice[Buffer, Buffer]:
        # MLA never uses per-layer buffers; the parameters exist only to match
        # the base signature threaded by ``build_runtime_inputs``.
        assert blocks_per_layer is None
        assert scales_per_layer is None
        assert isinstance(target_key, MLAAttnKey)
        assert draft_key is None or isinstance(draft_key, MLAAttnKey)
        return KVCacheInputsPerDevice(
            kv_blocks=blocks,
            cache_lengths=cache_lengths,
            lookup_table=lookup_table,
            max_prompt_length=max_prompt_length,
            max_cache_length=max_cache_length,
            kv_scales=kv_scales,
            scales_lookup_table=scales_lookup_table,
            attention_dispatch_metadata=target_key.pack_into_buffer(
                device, max_cache_valid_length
            ),
            draft_attention_dispatch_metadata=draft_key.pack_into_buffer(
                device, max_cache_valid_length
            )
            if draft_key is not None
            else None,
            mla_num_partitions=Buffer.from_numpy(
                np.array([target_key.num_partitions], dtype=np.int64)
            ),
            draft_mla_num_partitions=Buffer.from_numpy(
                np.array([draft_key.num_partitions], dtype=np.int64)
            )
            if draft_key is not None
            else None,
        )


@dataclass(kw_only=True)
class MSAKVCacheParams(MHAKVCacheParams):
    # TODO(SERVOPT-1502): MSA does not actually consume attention dispatch
    # metadata in its kernel. Once the indexer graph is migrated to a dedicated
    # MSA input record, drop ``attention_dispatch_metadata`` from the symbolic
    # and runtime inputs entirely instead of carrying the 1-int placeholder.
    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the decode attention dispatch shape for the given shape."""
        return MSAAttnKey()

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns cache lengths to probe for distinct num_partitions."""
        return [1, max_cache_length]

    def _attn_metadata_buffer(self, device: DeviceRef) -> TensorType:
        # ``MSAAttnKey.pack_into_buffer`` emits a single sentinel int.
        return TensorType(DType.int64, shape=[1], device=DeviceRef.CPU())


@dataclass(frozen=True)
class MultiKVCacheParams(KVCacheParamInterface):
    """Aggregates multiple KV cache parameter sets into a recursive tree.

    Children may be leaf :class:`KVCacheParams` instances or nested
    :class:`MultiKVCacheParams` subtrees, so arbitrarily deep hierarchies
    are supported (e.g. ``{target: {sliding, mla}, draft: mha}``). The
    whole tree is consumed through the :class:`KVCacheParamInterface` —
    callers never need to know the depth.
    """

    children: dict[str, KVCacheParamInterface]
    """KV cache parameter sets to aggregate. Values may be leaf
    :class:`KVCacheParams` or nested :class:`MultiKVCacheParams` trees."""

    page_size: int
    data_parallel_degree: int
    devices: Sequence[DeviceRef]
    kv_connector_config: KVConnectorConfigInterface
    speculative_method: SpeculativeMethod | None = None
    num_draft_tokens: int = 0

    @classmethod
    def from_params(
        cls, params: Mapping[str, KVCacheParamInterface]
    ) -> MultiKVCacheParams:
        """Creates a :class:`MultiKVCacheParams` from one or more param sets.

        Children may be leaf :class:`KVCacheParams` instances or nested
        :class:`MultiKVCacheParams` trees, enabling arbitrarily deep KV
        cache hierarchies (e.g. ``{target: {sliding, mla}, draft: mha}``).
        All children must share the same ``page_size``,
        ``data_parallel_degree``, ``n_devices``, and
        ``kv_connector_config`` values.

        Args:
            params: Named mapping of :class:`KVCacheParamInterface` instances
                to aggregate.

        Returns:
            A new :class:`MultiKVCacheParams` aggregating all provided params.

        Raises:
            ValueError: If no params are provided.
        """
        if len(params) == 0:
            raise ValueError("MultiKVCacheParams requires at least one param.")
        first = next(iter(params.values()))
        return cls(
            children=dict(params),
            page_size=first.page_size,
            data_parallel_degree=first.data_parallel_degree,
            devices=first.devices,
            kv_connector_config=first.kv_connector_config,
            speculative_method=first.speculative_method,
            num_draft_tokens=first.num_draft_tokens,
        )

    def __post_init__(self) -> None:
        """Validates that all params have consistent page size."""
        if not self.children:
            raise ValueError(
                "MultiKVCacheParams requires at least one param set."
            )

        params = list(self.children.values())
        page_sizes = {p.page_size for p in params}
        if len(page_sizes) > 1:
            raise ValueError(
                f"All params must use the same page size, got: {page_sizes}"
            )

        data_parallel_degrees = {p.data_parallel_degree for p in params}
        if len(data_parallel_degrees) > 1:
            raise ValueError(
                "All params must use the same data parallel degree, got:"
                f" {data_parallel_degrees}"
            )

        devices = {tuple(p.devices) for p in params}
        if len(devices) > 1:
            raise ValueError(
                f"All params must use the same number of devices, got: {devices}"
            )

        enable_prefix_caching = {p.enable_prefix_caching for p in params}
        if len(enable_prefix_caching) > 1:
            raise ValueError(
                "All params must use the same enable_prefix_caching, got:"
                f" {enable_prefix_caching}"
            )

        enable_dp_cross_replica_prefix_copy = {
            p.enable_dp_cross_replica_prefix_copy for p in params
        }
        if len(enable_dp_cross_replica_prefix_copy) > 1:
            raise ValueError(
                "All params must use the same"
                " enable_dp_cross_replica_prefix_copy, got:"
                f" {enable_dp_cross_replica_prefix_copy}"
            )

        # ``KVConnectorConfig`` is not hashable, so compare by equality against
        # the first rather than collapsing into a set.
        first_kv_connector_config = params[0].kv_connector_config
        if any(
            p.kv_connector_config != first_kv_connector_config for p in params
        ):
            raise ValueError(
                "All params must use the same kv_connector_config, got:"
                f" {[p.kv_connector_config for p in params]}"
            )

        speculative_methods = {p.speculative_method for p in params}
        if len(speculative_methods) > 1:
            raise ValueError(
                "All params must use the same speculative_method, got:"
                f" {speculative_methods}"
            )

        num_draft_tokens_set = {p.num_draft_tokens for p in params}
        if len(num_draft_tokens_set) > 1:
            raise ValueError(
                "All params must use the same num_draft_tokens, got:"
                f" {num_draft_tokens_set}"
            )

        kv_hash_algos = {p.kv_hash_algo for p in params}
        if len(kv_hash_algos) > 1:
            raise ValueError(
                f"All params must use the same kv_hash_algo, got: {kv_hash_algos}"
            )

        kv_hash_seeds = {p.kv_hash_seed for p in params}
        if len(kv_hash_seeds) > 1:
            raise ValueError(
                f"All params must use the same kv_hash_seed, got: {kv_hash_seeds}"
            )

    @property
    def _first(self) -> KVCacheParamInterface:
        """Returns the first child param set."""
        return next(iter(self.children.values()))

    @property
    def n_devices(self) -> int:
        """Returns the number of devices."""
        return len(self.devices)

    @property
    def enable_prefix_caching(self) -> bool:
        """Whether prefix caching is enabled (shared across all caches)."""
        return self._first.enable_prefix_caching

    @property
    def enable_dp_cross_replica_prefix_copy(self) -> bool:
        """Whether DP cross-replica prefix copies are enabled (shared across
        all caches)."""
        return self._first.enable_dp_cross_replica_prefix_copy

    @property
    def kv_hash_algo(self) -> KVHashAlgo:
        """Hash algorithm used for KV-cache block identity."""
        return self._first.kv_hash_algo

    @property
    def kv_hash_seed(self) -> bytes | None:
        """Resolved 32-byte cluster seed for sha256/sha256_64. None for ahash64."""
        return self._first.kv_hash_seed

    @property
    def bytes_per_block(self) -> int:
        """Total bytes per block across all KV caches.

        Since all caches allocate memory for the same sequence, the total
        memory cost per block is the sum across all param sets.
        """
        return sum(p.bytes_per_block for p in self.children.values())

    def get_symbolic_inputs(
        self, namespace: str = ""
    ) -> MultiKVCacheInputs[TensorType, BufferType]:
        """Returns the symbolic inputs for the KV cache tree.

        Each child inherits a distinct namespace so sibling groups' page-pool
        dims stay independent; nested subtrees compose the prefix.
        """
        return MultiKVCacheInputs(
            children={
                k: p.get_symbolic_inputs(namespace=f"{namespace}{k}_")
                for k, p in self.children.items()
            }
        )

    def unflatten_kv_inputs(
        self, it: Iterator[Any]
    ) -> MultiKVCacheInputs[TensorValue, BufferValue]:
        """Unflattens the KV cache inputs from a graph-input iterator."""
        return self.get_symbolic_inputs().unflatten(it)

    def unflatten_basic_kv_tree(
        self, it: Iterator[Any]
    ) -> tuple[list[KVCacheInputsPerDevice[TensorValue, BufferValue]], ...]:
        """Unflattens a basic KV tree from a graph-input iterator.

        Requires that the model is a basic height-1 tree. This method does not work
        on nested trees.
        """
        tree = self.unflatten_kv_inputs(it)
        assert isinstance(tree, MultiKVCacheInputs)
        out: list[list[KVCacheInputsPerDevice[TensorValue, BufferValue]]] = []
        for child in tree.children.values():
            if not isinstance(child, KVCacheInputs):
                raise ValueError("Unable to flatten nested KV tree")
            out.append(list(child.inputs))
        return tuple(out)

    @property
    def replicates_kv_across_tp(self) -> bool:
        """Whether every device holds identical KV state."""
        return self._first.replicates_kv_across_tp

    @property
    def tensor_parallel_degree(self) -> int:
        """Returns the tensor parallel degree."""
        return self._first.tensor_parallel_degree

    def resolve_attn_key(
        self,
        batch_size: int,
        max_prompt_length: int,
        max_cache_valid_length: int,
    ) -> AttnKeyInterface:
        """Resolves the dispatch shape tree mirroring the cache tree."""
        return MultiAttnKey.from_dict(
            {
                k: p.resolve_attn_key(
                    batch_size, max_prompt_length, max_cache_valid_length
                )
                for k, p in self.children.items()
            }
        )

    def graph_capture_probe_cache_lengths(
        self, max_cache_length: int, q_max_seq_len: int = 1
    ) -> list[int]:
        """Returns the union of probe cache lengths across all child caches."""
        lengths: set[int] = set()
        for p in self.children.values():
            lengths.update(
                p.graph_capture_probe_cache_lengths(
                    max_cache_length, q_max_seq_len
                )
            )
        return sorted(lengths)

    def allocate_buffers(
        self, total_num_pages: int
    ) -> list[KVCacheBufferInterface]:
        """Allocates per-replica buffers for every cache in the tree.

        Returns one :class:`MultiKVCacheBuffer` per data-parallel replica,
        each holding that replica's :class:`KVCacheBuffer` for every child
        cache.
        """
        per_key = {
            k: p.allocate_buffers(total_num_pages)
            for k, p in self.children.items()
        }
        return [
            MultiKVCacheBuffer(
                children={k: per_key[k][replica_idx] for k in self.children}
            )
            for replica_idx in range(self.data_parallel_degree)
        ]

    def build_runtime_inputs(
        self,
        assignments: Sequence[KVCacheAssignments],
        buffers: Sequence[KVCacheBufferInterface],
        _prefix: str = "",
    ) -> KVCacheInputsInterface[Buffer, Buffer]:
        """Builds the runtime KV-cache tree spanning all replicas.

        Each child leaf is built from every replica's assignment plus that
        replica's child buffer; the per-replica assignment (cache lengths /
        lookup table / dispatch shape) is shared across child caches since
        they all map the same sequence.
        """
        multi_buffers: list[MultiKVCacheBuffer] = []
        for buffer in buffers:
            assert isinstance(buffer, MultiKVCacheBuffer)
            multi_buffers.append(buffer)
        return MultiKVCacheInputs(
            children={
                k: p.build_runtime_inputs(
                    assignments,
                    [b.children[k] for b in multi_buffers],
                    _prefix=_prefix + k + ".",
                )
                for k, p in self.children.items()
            }
        )

    def leaves(self, _prefix: str = "") -> Mapping[str, KVLeafRegion]:
        """Returns the leaves of the KV cache."""
        leaves: dict[str, KVLeafRegion] = {}
        for k, v in self.children.items():
            leaves.update(v.leaves(_prefix + k + "."))
        return leaves

    def slab_to_buffer_views(
        self, buffers: Sequence[Buffer]
    ) -> KVCacheBufferInterface:
        """Converts a slab of memory into a buffer view."""
        return MultiKVCacheBuffer(
            children={
                child_id: child.slab_to_buffer_views(buffers)
                for child_id, child in self.children.items()
            },
        )


def spec_decode_cache_slack(params: KVCacheParamInterface) -> int:
    """Computes the extra KV positions a request may occupy past ``max_seq_len``.

    A speculative-decode step can over-speculate past the per-request
    ``max_seq_len`` cap into this slack (the KV pool reserves it beyond
    ``max_seq_len``), so any per-request sizing derived from ``max_seq_len``
    -- the pool's page budget, the sparse-indexer score scratch, etc. -- must
    add it. Centralized here so ``_compute_seq_len``, pool sizing, and
    ``OverlapTextGenerationPipeline._effective_max_cache_length`` stay in sync.

    Args:
        params: The KV cache parameters. The speculative-decoding fields
            (``num_draft_tokens`` and ``num_draft_tokens_per_step``) determine
            the slack.

    Returns:
        The number of extra KV positions to reserve past ``max_seq_len``, or
        ``0`` when speculative decoding is off.
    """
    if params.num_draft_tokens <= 0:
        return 0
    # Worst case matching ``_compute_seq_len``: drafts verified and written next
    # batch (2x), the prior overlap batch's drafts assumed accepted (1x), the
    # DFlash block-draft slot, and the FUTURE_TOKEN placeholder.
    block_draft_extra = (
        1 if params.num_draft_tokens_per_step == params.num_draft_tokens else 0
    )
    return 3 * params.num_draft_tokens + block_draft_extra + 1


def compute_num_device_blocks(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    max_batch_size: int | None,
    max_seq_len: int | None,
    require_max_seq_len_fits: bool = False,
    include_null_block: bool = False,
) -> int:
    """Computes the number of blocks that can be allocated based on the available cache memory.

    The number of blocks returned is for a single replica. Each replica will
    have the same number of blocks.

    Args:
        available_cache_memory: The amount of cache memory available across all devices.
        max_batch_size: The maximum batch size, or None.
        max_seq_len: The maximum sequence length, or None.
        require_max_seq_len_fits: When True, raise instead of warn if a single
            request at ``max_seq_len`` cannot fit in the allocable device
            blocks. Memory estimation deliberately probes oversized configs,
            so only the actual cache-allocation path should set this.
        include_null_block: Whether to include room for the null block.

    Returns:
        The number of blocks that can be allocated for a single replica.
    """
    # Compute upper bound of total number of pages required. A speculative
    # step can grow a request past max_seq_len into the pool's draft-token
    # slack, so budget the per-request pages on that same bound; otherwise the
    # pool caps short of what the scheduler is allowed to reserve.
    max_blocks_per_req: int | None = None
    max_total_blocks: int | None = None
    if max_seq_len is not None and max_batch_size is not None:
        max_seq_len_with_slack = max_seq_len + spec_decode_cache_slack(params)
        max_blocks_per_req = math.ceil(
            max_seq_len_with_slack / params.page_size
        )
        max_total_blocks = max_blocks_per_req * max_batch_size
        if include_null_block:
            max_total_blocks += 1

    # Compute total number of blocks allocatable based on available memory.
    available_cache_memory_per_replica = (
        available_cache_memory // params.data_parallel_degree
    )
    num_allocable_blocks = (
        available_cache_memory_per_replica // params.bytes_per_block
    )

    if max_total_blocks is not None:
        num_blocks = min(num_allocable_blocks, max_total_blocks)
    else:
        num_blocks = num_allocable_blocks

    # Check if we are allocating sufficient blocks.
    # If not, raise a warning or error.
    single_page_size_bytes_str = to_human_readable_bytes(params.bytes_per_block)
    cache_memory_str = to_human_readable_bytes(
        available_cache_memory_per_replica
    )
    devices_per_replica = params.n_devices // params.data_parallel_degree
    across_x_devices_str = (
        f" across {devices_per_replica} devices"
        if devices_per_replica > 1
        else ""
    )
    if num_allocable_blocks == 0:
        raise RuntimeError(
            "Insufficient cache memory to allocate even a single page.\n"
            f"One page requires {single_page_size_bytes_str} but only "
            f"{cache_memory_str} are available{across_x_devices_str}."
        )

    if max_batch_size is not None and max_batch_size > num_allocable_blocks:
        memory_needed_str = to_human_readable_bytes(
            max_batch_size * params.bytes_per_block
        )
        logger.warning(
            "Insufficient cache memory to support a batch containing"
            f" {max_batch_size} requests with one token per request. Need to"
            f" allocate at least {max_batch_size} pages ({memory_needed_str}),"
            f" but only have enough memory for {num_allocable_blocks} pages"
            f" ({cache_memory_str}{across_x_devices_str})."
        )

    if (
        max_blocks_per_req is not None
        and max_blocks_per_req > num_allocable_blocks
    ):
        memory_needed_str = to_human_readable_bytes(
            max_blocks_per_req * params.bytes_per_block
        )
        slack = spec_decode_cache_slack(params)
        slack_str = (
            f" (plus {slack} speculative-decode slack tokens)"
            if slack > 0
            else ""
        )
        msg = (
            "Insufficient cache memory to support a batch containing one"
            f" request at the max sequence length of {max_seq_len} tokens"
            f"{slack_str}. Need to allocate at least {max_blocks_per_req} pages"
            f" ({memory_needed_str}), but only have enough memory for"
            f" {num_allocable_blocks} pages"
            f" ({cache_memory_str}{across_x_devices_str})."
        )
        if require_max_seq_len_fits:
            raise RuntimeError(
                msg + " A request approaching the max sequence length would"
                " exhaust the KV cache and crash the model worker. Reduce"
                " --max-length to at most"
                f" {num_allocable_blocks * params.page_size} or increase the"
                " available KV cache memory (e.g. raise"
                " --device-memory-utilization)."
            )
        logger.warning(msg)

    return num_blocks


def estimated_memory_size(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    max_batch_size: int,
    max_seq_len: int,
    include_null_block: bool = False,
) -> int:
    """Computes the estimated memory size of the KV cache used by all replicas.

    Args:
        available_cache_memory: The amount of cache memory available across all devices.
        max_batch_size: The maximum batch size.
        max_seq_len: The maximum sequence length.
        include_null_block: Whether to include room for the null block.

    Returns:
        The estimated memory usage of the KV cache in bytes.
    """
    num_device_blocks = compute_num_device_blocks(
        available_cache_memory=available_cache_memory,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        params=params,
        include_null_block=include_null_block,
    )
    return (
        num_device_blocks * params.bytes_per_block * params.data_parallel_degree
    )


def compute_max_seq_len_fitting_in_cache(
    params: KVCacheParamInterface,
    available_cache_memory: int,
    include_null_block: bool = False,
) -> int:
    """Computes the maximum sequence length that can fit in the available memory.

    Args:
        available_cache_memory: The amount of cache memory available across
        all devices.
        include_null_block: Whether to include room for the null block.

    Returns:
        The maximum sequence length that can fit in the available cache memory.
    """
    if params.bytes_per_block == 0:
        raise ValueError("bytes_per_block cannot be zero")
    num_blocks = compute_num_device_blocks(
        params=params,
        available_cache_memory=available_cache_memory,
        max_batch_size=1,
        # Do not limit the sequence length.
        max_seq_len=None,
        include_null_block=include_null_block,
    )
    # Reserve the speculative-decode slack a request may occupy past its
    # advertised max_seq_len (see spec_decode_cache_slack). Without this the
    # auto-derived cap would equal the whole pool, so _effective_max_cache_length
    # (min(max_seq_len + slack, pool)) collapses back to max_seq_len and
    # over-speculation is silently disabled near the top of context. No-op when
    # speculative decoding is off (slack == 0).
    max_seq_len = num_blocks * params.page_size - spec_decode_cache_slack(
        params
    )
    return max(1, max_seq_len)
