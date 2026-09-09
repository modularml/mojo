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

"""Memory planners for MAX pipelines KV cache allocation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from max.driver import Device
from max.nn.kv_cache import KVCacheParamInterface


@runtime_checkable
class ModelConfig(Protocol):
    """Structural protocol for model configuration consumed by MemoryPlanner.

    Any object that exposes ``devices`` satisfies this protocol.  Planners
    that also need KV-cache parameters check for ``get_kv_params`` via
    ``isinstance`` or guard it with ``hasattr``.
    """

    @property
    def devices(self) -> list[Device]:
        """Returns the list of devices on which the model runs."""
        ...


@runtime_checkable
class ModelConfigWithKVCache(ModelConfig, Protocol):
    """Extension of ``ModelConfig`` for models with a KV cache.

    Adds ``get_kv_params`` so planners that need cache parameters can work
    with a typed interface instead of ``getattr`` lookups.
    """

    def get_kv_params(self) -> KVCacheParamInterface:
        """Returns the KV cache parameters for this model."""
        ...


class MemoryPlanner:
    """Base class for pipeline model memory planning.

    Provides default implementations for all estimation methods. Subclasses
    override the methods that require architecture-specific logic:

    - Estimating KV cache memory requirements.
    - Estimating activation, weight, and signal-buffer memory overheads
      specific to the model.

    A ``MemoryPlanner`` is constructed from a ``ModelConfig`` alone (not from a
    full ``PipelineConfig``) so that it can be used independently of the
    pipeline stack.
    """

    #: When ``True``, :meth:`estimate_signal_buffer_memory` reserves
    #: ``Signals.NUM_BYTES`` even on single-device pipelines.  Set via
    #: the ``always_signal_buffers`` parameter of
    #: :meth:`PagedMemoryPlanner.with_activation_reservation` or by
    #: overriding at the class level in a custom subclass.
    _always_signal_buffers: bool = False

    # TODO: Once lib/ is removed and ArchConfig lives outside lib/, tighten
    # this to `config: ArchConfig`. Currently blocked by a circular import:
    # kv_cache -> lib.interfaces.arch_config -> kv_cache.config.
    def __init__(self, config: Any) -> None:
        """Initializes the memory planner with the model config.

        Args:
            config: Model configuration.
        """
        self._config = config

    def estimate_weights_size(self, pipeline_config: Any) -> int:
        """Estimates the memory consumed by model weights in bytes.

        The default implementation delegates to
        ``pipeline_config.model.weights_size()``.  Override in subclasses that
        need architecture-specific weight accounting (e.g. expert-parallel
        sharding adjustments).

        Args:
            pipeline_config: Pipeline configuration providing the model config.

        Returns:
            Estimated weight memory in bytes.
        """
        return pipeline_config.model.weights_size()

    def estimate_activation_memory(
        self,
        pipeline_config: Any,
        huggingface_config: Any,
    ) -> int:
        """Estimates activation memory beyond model weights.

        The default implementation returns ``0``.  Override in subclasses that
        require temporary buffers for large intermediate tensors (e.g. MLA
        up-projection during prefill, expert-parallel routing buffers).

        Args:
            pipeline_config: Pipeline configuration.
            huggingface_config: HuggingFace model configuration.

        Returns:
            Estimated activation memory in bytes.
        """
        return 0

    def infer_max_batch_size(
        self,
        pipeline_config: Any,
        devices: list[Device],
        weights_size: int,
    ) -> int | None:
        """Infers an architecture-specific default ``max_batch_size``.

        Memory planning calls this when the user did not set
        ``max_batch_size``, before :meth:`estimate_activation_memory` runs.
        The default returns ``None``, deferring to the framework-wide
        inference in memory estimation.  Override in planners for
        architectures with per-request device memory beyond the KV cache
        (e.g. recurrent-state pools) that need a tighter default.

        Args:
            pipeline_config: Pipeline configuration.
            devices: Loaded devices the model will run on.
            weights_size: Estimated model weights size in bytes.

        Returns:
            The inferred ``max_batch_size``, or ``None`` to use the
            framework default.
        """
        return None

    def estimate_signal_buffer_memory(
        self,
        pipeline_config: Any,
        arch_config: Any | None = None,
    ) -> int:
        """Estimates signal-buffer memory in bytes across all devices.

        Signal buffers are fixed-size per-GPU allocations used by P2P
        collectives.  The default returns ``0`` for single-device pipelines and
        delegates to ``pipeline_config.estimate_signal_buffer_memory`` for
        multi-device.

        Models that perform allreduce unconditionally (e.g. via
        ``VocabParallelEmbedding``) need signal buffers even on a single device.
        Set ``always_signal_buffers=True`` on the planner class to enable this.

        Args:
            pipeline_config: Pipeline configuration.
            arch_config: Unused; kept for interface parity with
                :meth:`PipelineConfig.estimate_signal_buffer_memory`.

        Returns:
            Estimated signal-buffer memory in bytes across all devices.
        """
        if (
            self._always_signal_buffers
            and len(pipeline_config.model.device_specs) == 1
        ):
            # Import deferred to avoid circular dependency at module load time.
            from max.nn.comm import Signals

            return Signals.NUM_BYTES
        return pipeline_config.estimate_signal_buffer_memory(arch_config)


class PagedMemoryPlanner(MemoryPlanner):
    """Memory planner for models that use a paged KV cache.

    This is the standard planner for autoregressive text-generation models.
    It delegates KV-parameter queries to the model config via the
    ``ModelConfigWithKVCache`` protocol.

    For models that require a fixed activation-memory reservation (e.g. VLMs
    that need headroom for vision processing), use
    :meth:`with_activation_reservation` to create a pre-configured subclass
    instead of writing a custom ``MemoryPlanner``::

        memory_planner=PagedMemoryPlanner.with_activation_reservation(
            15 * 1024**3
        )

    Args:
        config: Model configuration that implements
            :class:`ModelConfigWithKVCache` (i.e. exposes both ``devices``
            and ``get_kv_params``).

    Raises:
        TypeError: If ``config`` does not implement :class:`ModelConfigWithKVCache`.
    """

    #: Fixed activation-memory reservation in bytes.  Subclasses created via
    #: :meth:`with_activation_reservation` override this at the class level.
    _activation_reservation_bytes: int = 0

    #: Inherited from :class:`MemoryPlanner`; ``PagedMemoryPlanner`` defaults
    #: to ``False`` unless overridden via :meth:`with_activation_reservation`.
    _always_signal_buffers: bool = False

    def __init__(self, config: Any) -> None:
        """Initializes the paged memory planner.

        Args:
            config: Must implement :class:`ModelConfigWithKVCache`.

        Raises:
            TypeError: If ``config`` does not satisfy
                :class:`ModelConfigWithKVCache`.
        """
        if not isinstance(config, ModelConfigWithKVCache):
            raise TypeError(
                f"PagedMemoryPlanner requires a ModelConfigWithKVCache, "
                f"got {type(config).__name__!r}"
            )
        super().__init__(config)

    @classmethod
    def with_activation_reservation(
        cls,
        activation_bytes: int,
        always_signal_buffers: bool = False,
    ) -> type[PagedMemoryPlanner]:
        """Returns a :class:`PagedMemoryPlanner` subclass with a fixed activation-memory reservation.

        Use this instead of writing a custom ``MemoryPlanner`` subclass for
        architectures that simply need to reserve a fixed chunk of GPU memory
        before KV cache allocation (e.g. for vision processing headroom)::

            memory_planner=PagedMemoryPlanner.with_activation_reservation(
                15 * 1024**3  # 15 GiB
            )

        For models that perform allreduce unconditionally (e.g. VLMs using
        ``VocabParallelEmbedding``), pass ``always_signal_buffers=True`` so
        signal-buffer memory is reserved even on single-GPU::

            memory_planner=PagedMemoryPlanner.with_activation_reservation(
                15 * 1024**3, always_signal_buffers=True
            )

        Args:
            activation_bytes: Activation memory to reserve in bytes.
            always_signal_buffers: When ``True``, reserve signal-buffer memory
                even on single-device pipelines.

        Returns:
            A new :class:`PagedMemoryPlanner` subclass whose
            :meth:`estimate_activation_memory` returns ``activation_bytes``.
        """
        gib = activation_bytes / 1024**3
        name = f"PagedMemoryPlanner({gib:.0f}GiB)"
        return type(
            name,
            (cls,),
            {
                "_activation_reservation_bytes": activation_bytes,
                "_always_signal_buffers": always_signal_buffers,
            },
        )

    def estimate_activation_memory(
        self,
        pipeline_config: Any,
        huggingface_config: Any,
    ) -> int:
        """Returns the fixed activation-memory reservation for this planner.

        The default is ``0``.  Subclasses created via
        :meth:`with_activation_reservation` return the configured value.

        Args:
            pipeline_config: Unused by the default implementation.
            huggingface_config: Unused by the default implementation.

        Returns:
            Activation memory reservation in bytes.
        """
        return self._activation_reservation_bytes
