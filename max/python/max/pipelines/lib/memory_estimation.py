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

"""Memory planning: sizing a pipeline's device memory and the plan it yields."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, cast

from max.driver import Device, DeviceSpec, is_virtual_device_mode, load_devices
from max.dtype import DType
from max.nn.kv_cache import (
    KVCacheParamInterface,
    compute_max_seq_len_fitting_in_cache,
    estimated_memory_size,
)
from max.support.human_readable_formatter import to_human_readable_bytes

if TYPE_CHECKING:
    from max.pipelines.lib.registry import SupportedArchitecture

    from .config import PipelineConfig

from .config.model_config import MAXModelConfig
from .interfaces import (
    ArchConfig,
    ArchConfigWithKVCache,
    ArchConfigWithVisionCache,
    arch_has_vision_tower,
)
from .vision_encoder_cache import (
    DEFAULT_VISION_CACHE_BLOCK_TOKENS,
    VisionCachePlan,
)

logger = logging.getLogger("max.pipelines")

_DEFAULT_BATCH_SIZE = 512


def _kv_params_per_layer_depth(params: KVCacheParamInterface) -> int:
    """Returns the largest ``num_layers`` among the pool's per-layer sub-pools.

    Recurses into a :class:`~max.nn.kv_cache.MultiKVCacheParams` tree. A leaf
    with ``per_layer_buffers`` set contributes its ``num_layers``; every other
    cache contributes ``1`` (a single multi-layer buffer). Returns ``1`` when no
    cache uses per-layer buffers.
    """
    children = getattr(params, "children", None)
    if children is not None:
        return max(
            (_kv_params_per_layer_depth(child) for child in children.values()),
            default=1,
        )
    if getattr(params, "per_layer_buffers", False):
        return max(int(getattr(params, "num_layers", 1)), 1)
    return 1


def _max_per_layer_buffer_count(arch_config: ArchConfig) -> int:
    """Returns the per-device allocation-cap multiplier for the KV pool.

    A pool that uses one buffer *per layer* splits its per-device allocation
    into ``num_layers`` independent buffers, each bounded by the per-allocation
    cap, so the pool may use up to ``num_layers`` times that cap per device.
    Returns the depth of the largest per-layer sub-pool, or ``1`` when no cache
    uses per-layer buffers (leaving the cap unchanged).
    """
    if not isinstance(arch_config, ArchConfigWithKVCache):
        return 1
    return _kv_params_per_layer_depth(arch_config.get_kv_params())


@dataclass(frozen=True)
class MemoryPlan:
    """The memory plan computed when a pipeline is loaded.

    Carries the effective batch size, sequence-length bound, and memory
    budgets that the pipeline and its schedulers consume. Produced by
    :meth:`MemoryEstimator.plan`.
    """

    planned_max_batch_size: int
    """The maximum number of requests scheduled together in one batch: the
    user's ``runtime.max_batch_size``, or the value planning inferred when
    the user left it unset."""

    footprint: int
    """The estimated total device memory the pipeline uses, in bytes."""

    planned_max_length: int | None
    """The resolved maximum sequence length after memory planning lowered
    it to fit device memory: the construction-resolved
    ``config.model.max_length``, clamped to what the KV cache and any
    draft model can hold. ``None`` for pipelines with no main language
    model, such as diffusion pipelines."""

    available_cache_memory: int | None = None
    """The device memory committed to the KV cache, in bytes. ``None`` when
    the plan reserves no KV-cache budget, such as for models without a KV
    cache."""

    device_specs: tuple[DeviceSpec, ...] | None = None
    """The device specs the plan was computed for, kept as specs rather
    than ``Device`` objects so the plan can cross process boundaries.
    ``None`` for plans that never load devices, such as diffusion
    pipelines."""

    planned_max_batch_total_tokens: int | None = None
    """Cap on the total context tokens resident across a batch: the user's
    ``runtime.max_batch_total_tokens``, or ``planned_max_length`` for
    architectures that require a cap. ``None`` means no cap is configured."""

    vision_cache_plan: VisionCachePlan | None = None
    """Block-mode vision cache reservation; ``None`` means entry-count mode."""

    def require_device_specs(self) -> tuple[DeviceSpec, ...]:
        """Returns the device specs, which must be set on this plan.

        Raises:
            AssertionError: If the plan carries no device specs.
        """
        assert self.device_specs is not None, (
            "memory plan lacks device specs; pipelines require a "
            "plan built by the registry's memory-planning step"
        )
        return self.device_specs


class MemoryEstimator:
    """Plans device memory for a pipeline and estimates what it will use."""

    @classmethod
    def plan(
        cls,
        pipeline_config: PipelineConfig,
        arch: SupportedArchitecture,
        draft_arch: SupportedArchitecture | None = None,
    ) -> MemoryPlan:
        """Runs memory planning and returns the finished plan.

        Called by the registry's ``retrieve_factory`` after the config is
        constructed. Gathers the sizes and the draft-model bound that
        :meth:`plan_from_sizes` needs and runs it. Nothing is written back
        to ``pipeline_config``, which keeps carrying the
        construction-resolved values unchanged.
        """
        # Multi-component pipelines (diffusion models) have no "main" model entry
        # — they store per-component configs (transformer, vae, text_encoder, etc.)
        # and don't use a KV cache, so skip memory estimation entirely.
        if "main" not in pipeline_config.models:
            return MemoryPlan(
                planned_max_batch_size=pipeline_config.runtime.max_batch_size
                or 1,
                footprint=0,
                planned_max_length=None,
                planned_max_batch_total_tokens=pipeline_config.runtime.max_batch_total_tokens,
            )

        model_config = pipeline_config.model

        effective_specs = tuple(model_config.device_specs)
        logger.info(
            "devices: %s",
            ", ".join(f"{d.device_type}[{d.id}]" for d in effective_specs),
        )
        devices = load_devices(effective_specs)
        # No plan exists yet, so this config gets the resolved length; the
        # pipeline model's config later gets the clamped one.
        if model_config.max_length is None:
            raise ValueError(
                "max_length is unresolved. Construct the config through "
                "PipelineConfig.from_args, which runs the architecture's "
                "sequence-length policy, or set max_length explicitly."
            )
        arch_config = arch.config.initialize(
            pipeline_config,
            model_config=model_config,
            max_seq_len=model_config.max_length,
        )

        max_batch_size = pipeline_config.runtime.max_batch_size
        if arch.memory_planner is not None:
            planner = arch.memory_planner(arch_config)
            weights_size = planner.estimate_weights_size(pipeline_config)
            if max_batch_size is None:
                max_batch_size = planner.infer_max_batch_size(
                    pipeline_config, devices, weights_size
                )
            activation_size = planner.estimate_activation_memory(
                pipeline_config, model_config.huggingface_config
            )
            signal_buffer_size = planner.estimate_signal_buffer_memory(
                pipeline_config, arch_config
            )
        else:
            # ``memory_planner=None`` is the fallback for architectures not yet
            # wired to a MemoryPlanner. If adding a new architecture that uses a
            # KV cache, set ``memory_planner=PagedMemoryPlanner`` on its
            # ``SupportedArchitecture``.
            weights_size = model_config.weights_size()
            activation_size = 0
            signal_buffer_size = pipeline_config.estimate_signal_buffer_memory(
                arch_config
            )

        # Under speculative decoding the draft shares the target's KV cache,
        # so its own limit bounds the pipeline's.
        draft_max_seq_len = None
        if draft_arch is not None and pipeline_config.draft_model is not None:
            draft_max_seq_len = pipeline_config.draft_model.max_length
            if draft_max_seq_len is None:
                raise ValueError(
                    "The draft model's max_length is unresolved. Construct "
                    "the config through PipelineConfig.from_args, which runs "
                    "the draft architecture's sequence-length policy, or set "
                    "max_length on the draft model config explicitly."
                )

        plan = cls.plan_from_sizes(
            pipeline_config,
            model_config,
            arch_config,
            devices,
            weights_size,
            activation_size,
            signal_buffer_size,
            arch=arch,
            max_batch_size=max_batch_size,
            draft_max_seq_len=draft_max_seq_len,
        )

        # TODO(MXF-517): Fold this into a consolidated startup logger that reports
        # all resolved runtime values together. It logs here, from the planner that
        # computes the budget, because the value is no longer mutated onto the config
        # for log_basic_config to read.
        if plan.available_cache_memory is not None:
            logger.info(
                "cache_memory: %s",
                to_human_readable_bytes(plan.available_cache_memory),
            )

        return plan

    @classmethod
    def _free_memory(cls, devices: list[Device]) -> int:
        """Returns the total free memory available across all provided devices."""
        try:
            free_memory = int(sum(d.stats["free_memory"] for d in devices))
            if free_memory == 0:
                total_memory = int(
                    sum(d.stats.get("total_memory", 0) for d in devices)
                )
                return total_memory
            return free_memory
        except Exception as e:
            logger.warning(
                "Unable to estimate memory footprint of model, can't query device stats: "
                + str(e)
            )
            raise

    @classmethod
    def _static_memory_size(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        signal_buffer_size: int = 0,
    ) -> int:
        """Calculates static memory usage: model weights plus activations plus signal buffers.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.
            signal_buffer_size: Size of P2P signal buffers (fixed-size
                allocations used by collective comm kernels). Defaults to 0.

        Returns:
            Total static memory usage in bytes.
        """
        return model_weights_size + activation_memory_size + signal_buffer_size

    @classmethod
    def _available_kv_cache_memory(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
        signal_buffer_size: int = 0,
    ) -> int:
        """Estimates available KV cache memory after model weights, activations, and signal buffers.

        Args:
            model_weights_size: Size of model weights.
            activation_memory_size: Size of activation memory.
            model_config: The model configuration.
            devices: The list of devices on which the model will run.
            signal_buffer_size: Size of P2P signal buffers. Defaults to 0.

        Returns:
            Available KV cache memory in bytes.
        """
        return int(
            (
                cls._free_memory(devices)
                * model_config.kv_cache.device_memory_utilization
            )
            - cls._static_memory_size(
                model_weights_size,
                activation_memory_size,
                signal_buffer_size,
            )
        )

    @classmethod
    def _max_supported_sequence_length(
        cls,
        model_weights_size: int,
        activation_memory_size: int,
        model_config: MAXModelConfig,
        devices: list[Device],
        arch_config: ArchConfig,
        signal_buffer_size: int = 0,
        available_cache_memory: int | None = None,
    ) -> int | None:
        """Computes the hard upper bound on tokens for a single request.

        Mirrors the paged KV cache constraint: per replica, a request cannot
        exceed total pages per device times page size.
        """
        # In virtual device mode (cross-compilation), skip memory-based constraints
        # since we're only compiling and not actually running the model.
        if is_virtual_device_mode():
            logger.info(
                "Skipping memory-based sequence length constraints in "
                "virtual device mode (cross-compilation)"
            )
            return None

        # Retrieve needed parameters.
        if not isinstance(arch_config, ArchConfigWithKVCache):
            return None

        arch_config = cast(ArchConfigWithKVCache, arch_config)
        params = arch_config.get_kv_params()

        # Prefer the KV byte budget committed in ``plan_from_sizes`` (after
        # vision cache reservation and ``estimated_memory_size``). Using only
        # ``available_kv_cache_memory()`` (pre-vision) can overcount blocks and clamp
        # ``max_length`` above the physical paged KV capacity, causing runtime
        # InsufficientBlocksError when ``len(tokens)`` reaches ``total_blocks * page_size + 1``.
        if available_cache_memory is not None:
            kvcache_mem = available_cache_memory
        else:
            kvcache_mem = cls._available_kv_cache_memory(
                model_weights_size,
                activation_memory_size,
                model_config,
                devices,
                signal_buffer_size,
            )
        return compute_max_seq_len_fitting_in_cache(
            params=params,
            available_cache_memory=kvcache_mem,
            include_null_block=True,
        )

    @classmethod
    def plan_from_sizes(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        arch_config: ArchConfig,
        devices: list[Device],
        model_weights_size: int,
        activation_memory_size: int,
        signal_buffer_size: int = 0,
        arch: SupportedArchitecture | None = None,
        max_batch_size: int | None = None,
        draft_max_seq_len: int | None = None,
    ) -> MemoryPlan:
        """Plans memory from precomputed weight, activation, and buffer sizes.

        Estimates the footprint, reserves the KV and vision budgets, and
        bounds ``max_length`` and ``max_batch_size`` to what fits. Callers
        that have only a :class:`PipelineConfig` should use
        :meth:`plan`, which derives the sizes and calls this.

        ``draft_max_seq_len`` is the draft model's own sequence-length limit
        under speculative decoding. It bounds ``max_length`` because the draft
        shares the target's KV cache, and it is passed in because the
        estimator never sees the draft architecture.

        Returns:
            The finished :class:`MemoryPlan`. Nothing is written back to
            ``pipeline_config``.
        """
        device_specs = tuple(model_config.device_specs)

        # Construction already applied the architecture's policy; the clamps
        # below only lower it.
        resolved_max_seq_len = model_config.max_length
        if resolved_max_seq_len is None:
            raise ValueError(
                "max_length is unresolved. Construct the config through "
                "PipelineConfig.from_args, which runs the architecture's "
                "sequence-length policy, or set max_length explicitly."
            )

        # In virtual device mode (cross-compilation), skip memory estimation
        # since we're only compiling and not actually running the model.
        # Use model defaults for max_batch_size and max_length.
        if is_virtual_device_mode():
            logger.info(
                "Skipping memory estimation in virtual device mode "
                "(cross-compilation)"
            )
            max_batch_size = max_batch_size or 1
            max_length = cls._bounded_by_draft(
                resolved_max_seq_len,
                draft_max_seq_len,
            )
            # Report a large cache budget since we're only cross-compiling, not
            # allocating memory. 1TB works for any model.
            virtual_cache_memory = 1024 * 1024 * 1024 * 1024  # 1TB
            return MemoryPlan(
                planned_max_batch_size=max_batch_size,
                footprint=0,
                planned_max_length=max_length,
                available_cache_memory=virtual_cache_memory,
                device_specs=device_specs,
                planned_max_batch_total_tokens=cls._resolve_max_batch_total_tokens(
                    pipeline_config, arch, max_length
                ),
            )

        try:
            free_memory = cls._free_memory(devices)
        except Exception:
            # A KV-cache model cannot be planned without device stats: the plan
            # would carry no cache budget and ``load_kv_manager`` rejects that
            # with an error naming memory estimation, long after the real cause
            # is gone. Fail here instead, while the device-stats error is in
            # hand. Models with no KV cache never load a manager, so they can
            # still run on architecture defaults.
            if isinstance(arch_config, ArchConfigWithKVCache):
                raise
            max_length = cls._bounded_by_draft(
                resolved_max_seq_len,
                draft_max_seq_len,
            )
            return MemoryPlan(
                planned_max_batch_size=max_batch_size or 1,
                footprint=0,
                planned_max_length=max_length,
                device_specs=device_specs,
                planned_max_batch_total_tokens=cls._resolve_max_batch_total_tokens(
                    pipeline_config, arch, max_length
                ),
            )

        # Total static memory requirement (weights + activations + signal buffers)
        static_memory_size = (
            model_weights_size + activation_memory_size + signal_buffer_size
        )

        if static_memory_size > free_memory:
            error_msg = f"Model size exceeds available memory ({to_human_readable_bytes(static_memory_size)} > {to_human_readable_bytes(free_memory)}). "
            if activation_memory_size > 0 or signal_buffer_size > 0:
                error_msg += (
                    f"Model weights: {to_human_readable_bytes(model_weights_size)}, "
                    f"Activation memory: {to_human_readable_bytes(activation_memory_size)}, "
                    f"Signal buffers: {to_human_readable_bytes(signal_buffer_size)}. "
                )
            error_msg += "Try running a smaller model, using a smaller precision, or using a device with more memory."
            raise RuntimeError(error_msg)

        total_size = static_memory_size
        available_kv_cache_memory = int(
            free_memory * model_config.kv_cache.device_memory_utilization
            - static_memory_size
        )

        if available_kv_cache_memory <= 0:
            raise RuntimeError(
                f"The model {to_human_readable_bytes(model_weights_size)}, activations "
                f"{to_human_readable_bytes(activation_memory_size)}, and signal buffers "
                f"{to_human_readable_bytes(signal_buffer_size)} don't leave room for KV cache. "
                f"Try running a smaller model, using a smaller precision, or using a device with more memory."
            )

        # KV cache is normally one buffer per device, so the budget can't
        # exceed the per-allocation cap (e.g. Metal's maxBufferLength). A pool
        # that uses one buffer *per layer* (``per_layer_buffers``) splits that
        # allocation into ``num_layers`` independent buffers, each bounded by
        # the cap, so it may use up to ``num_layers`` times the cap per device.
        per_alloc_layers = _max_per_layer_buffer_count(arch_config)
        available_kv_cache_memory = min(
            available_kv_cache_memory,
            per_alloc_layers * sum(d.max_single_alloc_size for d in devices),
        )

        vision_cache_bytes, vision_cache_plan = (
            cls._reserve_vision_cache_memory(
                pipeline_config,
                model_config,
                available_kv_cache_memory,
                devices,
                arch=arch,
            )
        )
        available_kv_cache_memory -= vision_cache_bytes
        total_size += vision_cache_bytes

        # The field is set for every config after construction, so intent
        # comes from the bit captured before it was resolved.
        user_provided_max_length = model_config.max_length_is_user_provided
        user_provided_max_batch_size = max_batch_size is not None

        max_length = resolved_max_seq_len

        if user_provided_max_batch_size:
            assert max_batch_size is not None
        else:
            max_batch_size = cls._infer_optimal_batch_size(arch_config, devices)
        if max_batch_size > pipeline_config.runtime.max_batch_input_tokens:
            logger.info(
                f"max_batch_size of {max_batch_size} cannot be larger than max_batch_input_tokens of {pipeline_config.runtime.max_batch_input_tokens}, overriding max_batch_size to {pipeline_config.runtime.max_batch_input_tokens}"
            )
            max_batch_size = pipeline_config.runtime.max_batch_input_tokens

        actual_kv_cache_size = cls._calculate_kv_cache_size(
            arch_config=arch_config,
            max_batch_size=max_batch_size,
            available_kv_cache_memory=available_kv_cache_memory,
            max_seq_len=resolved_max_seq_len,
        )

        # Committed KV byte budget (captured before the OOM-fit search below may
        # reassign ``actual_kv_cache_size``); threaded to consumers on the plan.
        available_cache_memory = actual_kv_cache_size

        total_size += actual_kv_cache_size
        # If the model is too large to fit in memory, and the user did not
        # specify a max_length, try to infer a value that would fit.
        if int(total_size) > free_memory and not user_provided_max_length:
            original_max_length = max_length
            (
                found_valid_max_length,
                inferred_max_length,
                max_batch_size,
            ) = cls._find_valid_max_length(
                arch_config,
                available_kv_cache_memory,
                user_provided_max_batch_size,
                max_batch_size,
                devices,
                max_length,
            )

            if found_valid_max_length:
                logger.warning(
                    f"Truncated model's default max_length from {original_max_length} to {inferred_max_length} to fit in memory."
                )
                max_length = inferred_max_length
            else:
                max_length = 1

            actual_kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
                max_seq_len=resolved_max_seq_len,
            )
            total_size = model_weights_size + actual_kv_cache_size

        vram_usage_limit_scale = 0.95

        if isinstance(free_memory, int | float):
            if int(total_size) > int(free_memory):
                cls._raise_oom_error(
                    arch_config,
                    user_provided_max_length,
                    user_provided_max_batch_size,
                    max_batch_size,
                    total_size,
                    free_memory,
                    available_kv_cache_memory,
                    devices,
                    max_length,
                )

            elif int(total_size) > int(vram_usage_limit_scale * free_memory):
                logger.warning(
                    "Estimated model and kv cache memory use nears available memory. You may experience errors."
                )

        if kv_capacity := cls._max_supported_sequence_length(
            model_weights_size,
            activation_memory_size,
            model_config,
            devices,
            arch_config,
            signal_buffer_size,
            available_cache_memory=available_cache_memory,
        ):
            if max_length is None:
                max_length = kv_capacity
            elif max_length > kv_capacity:
                logger.warning(
                    "Clamping max_length from %d to %d due to capacity of KV Cache",
                    max_length,
                    kv_capacity,
                )
                max_length = kv_capacity

        max_length = cls._bounded_by_draft(max_length, draft_max_seq_len)

        return MemoryPlan(
            planned_max_batch_size=max_batch_size,
            footprint=int(total_size),
            planned_max_length=max_length,
            available_cache_memory=available_cache_memory,
            device_specs=device_specs,
            planned_max_batch_total_tokens=cls._resolve_max_batch_total_tokens(
                pipeline_config, arch, max_length
            ),
            vision_cache_plan=vision_cache_plan,
        )

    @classmethod
    def _bounded_by_draft(
        cls, max_length: int | None, draft_max_seq_len: int | None
    ) -> int | None:
        """Lowers ``max_length`` to the draft model's own sequence limit."""
        if (
            draft_max_seq_len is None
            or max_length is None
            or max_length <= draft_max_seq_len
        ):
            return max_length
        logger.info(
            "Clamping max_length from %d to %d (draft model max sequence length)",
            max_length,
            draft_max_seq_len,
        )
        return draft_max_seq_len

    @classmethod
    def _resolve_max_batch_total_tokens(
        cls,
        pipeline_config: PipelineConfig,
        arch: SupportedArchitecture | None,
        max_length: int | None,
    ) -> int | None:
        """Returns the cap on context tokens resident across a batch.

        Architectures requiring chunked prefill need a cap; defaulting it to
        the bounded ``max_length`` stops the scheduler admitting more resident
        context than the KV cache holds.
        """
        configured = pipeline_config.runtime.max_batch_total_tokens
        if (
            configured is not None
            or arch is None
            or not arch.requires_max_batch_context_length
        ):
            return configured
        logger.warning(
            "Architecture '%s' requires max-batch-total-tokens to be specified "
            "but found None. Defaulting to the max sequence length of the model: %s",
            arch.name,
            max_length,
        )
        return max_length

    @classmethod
    def _find_valid_max_length(
        cls,
        arch_config: ArchConfig,
        available_kv_cache_memory: int,
        user_provided_max_batch_size: bool,
        max_batch_size: int,
        devices: list[Device],
        max_length: int,
    ) -> tuple[bool, int, int]:
        """Binary search to find a valid max_length configuration.

        Returns:
            Tuple containing:
            - found_valid_max_length: Whether a valid max_length was found
            - inferred_max_length: The suggested max_length value
            - max_batch_size: The batch size used/inferred during the search
        """
        found_valid_max_length = False
        lower = 1
        upper = max_length
        inferred_max_length = upper

        while not found_valid_max_length:
            inferred_max_length = (lower + upper) // 2

            if not user_provided_max_batch_size:
                max_batch_size = cls._infer_optimal_batch_size(
                    arch_config, devices
                )

            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
                max_seq_len=inferred_max_length,
            )

            if lower > upper:
                break
            elif upper - lower <= 1:
                if kv_cache_size <= available_kv_cache_memory:
                    found_valid_max_length = True
                break

            if kv_cache_size > available_kv_cache_memory:
                upper = inferred_max_length - 1
            else:
                lower = inferred_max_length
        return (
            found_valid_max_length,
            inferred_max_length,
            max_batch_size,
        )

    @classmethod
    def _find_valid_batch_size(
        cls,
        available_kv_cache_memory: int,
        original_max_length: int,
        user_provided_max_batch_size: bool,
        max_batch_size: int,
        arch_config: ArchConfig,
    ) -> tuple[bool, int]:
        """Binary search to find a valid batch size configuration.

        Returns:
            Tuple containing:
            - found_valid_max_batch_size: Whether a valid batch size was found
            - inferred_max_batch_size: The suggested batch size value.
                If the user did not provide a batch size, this will be -1.
        """
        if not user_provided_max_batch_size:
            return False, -1

        found_valid_max_batch_size = False
        inferred_max_batch_size = max_batch_size
        lower = 1
        upper = max_batch_size

        while not found_valid_max_batch_size:
            inferred_max_batch_size = (lower + upper) // 2

            kv_cache_size = cls._calculate_kv_cache_size(
                arch_config=arch_config,
                max_batch_size=inferred_max_batch_size,
                available_kv_cache_memory=available_kv_cache_memory,
                max_seq_len=original_max_length,
            )

            if lower > upper:
                break
            elif upper - lower <= 1:
                if kv_cache_size <= available_kv_cache_memory:
                    found_valid_max_batch_size = True
                break

            if kv_cache_size > available_kv_cache_memory:
                upper = inferred_max_batch_size - 1
            else:
                lower = inferred_max_batch_size

        return found_valid_max_batch_size, inferred_max_batch_size

    @classmethod
    def _calculate_kv_cache_size(
        cls,
        arch_config: ArchConfig,
        max_batch_size: int,
        available_kv_cache_memory: int,
        max_seq_len: int,
    ) -> int:
        """Calculate the KV cache size for the current configuration.

        Args:
            arch_config: Architecture config that potentially provides KV cache
                parameters.
            max_batch_size: The maximum batch size.
            available_kv_cache_memory: Available memory for KV cache in bytes.
            max_seq_len: The per-sequence length to size the cache for,
                normally the architecture's policy value
                (:meth:`ArchConfig.calculate_max_seq_len`); binary searches
                pass their candidate value.
        """
        if isinstance(arch_config, ArchConfigWithKVCache):
            params = arch_config.get_kv_params()
            return estimated_memory_size(
                params=params,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                available_cache_memory=available_kv_cache_memory,
                include_null_block=True,
            )
        else:
            return 0

    @classmethod
    def _raise_oom_error(
        cls,
        arch_config: ArchConfig,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        max_batch_size: int,
        total_size: int,
        original_free_memory: int,
        available_kv_cache_memory: int,
        devices: list[Device],
        max_length: int,
    ) -> None:
        """Suggests a viable configuration when the current one does not fit in memory.

        If the current configuration won't fit in device memory, provides a
        friendly error message. The approach is to:

        1. Binary search max_length until we find a setting that works
        2. If user provided max_batch_size, binary search that too
        3. Generate appropriate suggestions based on this truth table:

        .. code-block:: text

                                                                max_length
                                             +----------------------+--------------------------+
                                             | set by user          | set to default           |
                            +----------------+======================+==========================+
                            | set by user    ║ Recommend both       | Recommend max_batch_size |
            max_batch_size  +----------------+----------------------+--------------------------+
                            | set to default ║ Recommend max_length | Recommend both           |
                            +----------------+----------------------+--------------------------+
        """
        original_max_length = max_length

        # Find valid configurations through binary search
        (
            found_valid_max_length,
            inferred_max_length,
            inferred_max_length_compatible_batch_size,
        ) = cls._find_valid_max_length(
            arch_config,
            available_kv_cache_memory,
            user_provided_max_batch_size,
            max_batch_size,
            devices,
            max_length,
        )

        found_valid_max_batch_size, inferred_max_batch_size = (
            cls._find_valid_batch_size(
                available_kv_cache_memory,
                original_max_length,
                user_provided_max_batch_size,
                max_batch_size,
                arch_config=arch_config,
            )
        )

        # Generate error message with suggestions
        error_msg = cls._generate_oom_error_message(
            total_size=total_size,
            original_free_memory=original_free_memory,
            user_provided_max_length=user_provided_max_length,
            user_provided_max_batch_size=user_provided_max_batch_size,
            found_valid_max_length=found_valid_max_length,
            found_valid_max_batch_size=found_valid_max_batch_size,
            inferred_max_length=inferred_max_length,
            inferred_max_batch_size=inferred_max_batch_size,
            inferred_max_length_compatible_batch_size=inferred_max_length_compatible_batch_size,
            original_max_length=original_max_length,
        )

        raise RuntimeError(error_msg)

    @classmethod
    def _generate_oom_error_message(
        cls,
        total_size: int,
        original_free_memory: int,
        user_provided_max_length: bool,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
        original_max_length: int,
    ) -> str:
        """Generate an appropriate error message based on the configuration state."""
        free_memory_str = (
            f" / {to_human_readable_bytes(original_free_memory)} free"
            if original_free_memory
            else ""
        )

        msg = StringIO()
        msg.write(
            f"Estimated model and kv cache memory use exceeds available memory ({to_human_readable_bytes(total_size)} {free_memory_str}). Try "
        )

        if not found_valid_max_length and not found_valid_max_batch_size:
            msg.write(
                "reducing --max-length or --max-batch-size, finding a smaller model, or using a device with more memory."
            )

        elif user_provided_max_length:
            cls._add_user_provided_max_length_suggestions(
                msg,
                user_provided_max_batch_size,
                found_valid_max_length,
                found_valid_max_batch_size,
                inferred_max_length,
                inferred_max_batch_size,
                inferred_max_length_compatible_batch_size,
            )
        else:
            cls._add_default_max_length_suggestions(
                msg,
                user_provided_max_batch_size,
                found_valid_max_length,
                found_valid_max_batch_size,
                inferred_max_length,
                inferred_max_batch_size,
                inferred_max_length_compatible_batch_size,
                original_max_length,
            )

        msg.write(".")
        return msg.getvalue()

    @classmethod
    def _add_user_provided_max_length_suggestions(
        cls,
        msg: StringIO,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
    ) -> None:
        """Add error message suggestions when user provided max_length.

        This handles the top row of the truth table from the _raise_oom_error docstring.

        Args:
            msg: StringIO buffer to write message to
            user_provided_max_batch_size: Whether user provided batch size
            found_valid_max_length: Whether valid max_length was found
            found_valid_max_batch_size: Whether valid batch size was found
            inferred_max_length: Suggested max_length value
            inferred_max_batch_size: Suggested batch size value
            inferred_max_length_compatible_batch_size: Compatible batch size for max_length
        """
        if not user_provided_max_batch_size:
            if found_valid_max_length:
                msg.write(
                    f"reducing --max-length to {inferred_max_length} "
                    f"(supports batch size of {inferred_max_length_compatible_batch_size})"
                )
            else:
                msg.write("reducing --max-length or --max-batch-size")
        else:
            if found_valid_max_length:
                msg.write(
                    f"reducing --max-length to {inferred_max_length} and "
                    f"--max-batch-size to {inferred_max_length_compatible_batch_size})"
                )

            if found_valid_max_batch_size:
                if found_valid_max_length:
                    msg.write(" or ")
                msg.write(
                    f"reducing --max-batch-size to {inferred_max_batch_size}"
                )

    @classmethod
    def _add_default_max_length_suggestions(
        cls,
        msg: StringIO,
        user_provided_max_batch_size: bool,
        found_valid_max_length: bool,
        found_valid_max_batch_size: bool,
        inferred_max_length: int,
        inferred_max_batch_size: int,
        inferred_max_length_compatible_batch_size: int,
        original_max_length: int,
    ) -> None:
        """Add error message suggestions when max_length was set to default.

        This handles the bottom row of the truth table from the _raise_oom_error docstring.

        Args:
            msg: StringIO buffer to write message to
            user_provided_max_batch_size: Whether user provided batch size
            found_valid_max_length: Whether valid max_length was found
            found_valid_max_batch_size: Whether valid batch size was found
            inferred_max_length: Suggested max_length value
            inferred_max_batch_size: Suggested batch size value
            inferred_max_length_compatible_batch_size: Compatible batch size for max_length
            original_max_length: Original max_length value before modifications
        """
        if not user_provided_max_batch_size:
            if found_valid_max_length:
                msg.write(
                    f"setting --max-length to {inferred_max_length} and "
                    f"--max-batch-size to {inferred_max_length_compatible_batch_size})"
                )

            if found_valid_max_batch_size:
                if found_valid_max_length:
                    msg.write(" or ")
                msg.write(
                    f"setting --max-batch-size to {inferred_max_batch_size}"
                )

        else:
            if found_valid_max_batch_size:
                msg.write(
                    f"reducing --max-batch-size to {inferred_max_batch_size}"
                )
            if found_valid_max_length:
                if found_valid_max_batch_size:
                    msg.write(" or ")
                msg.write(
                    f"setting --max-length to {inferred_max_length} "
                    f"(currently defaulted to {original_max_length})"
                )

    @classmethod
    def _reserve_vision_cache_memory(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig,
        available_memory: int,
        devices: list[Device],
        arch: SupportedArchitecture | None = None,
    ) -> tuple[int, VisionCachePlan | None]:
        """Estimate and reserve memory for the vision encoder cache.

        Delegates to the arch config's vision-cache facts:
        ``estimate_vision_cache_entry_bytes()`` sizes the requested budget and
        ``get_vision_cache_row_spec()`` sets the block row shape.
        Non-VLM architectures reserve no vision cache memory.

        Returns:
            Bytes to reserve for the vision encoder cache (0 for non-VLM
            models or when ``vision_cache_utilization`` is 0), and the
            block-mode plan (``None`` when disabled).
        """
        if pipeline_config.runtime.vision_cache_utilization == 0:
            return 0, None

        hf_config = model_config.huggingface_config
        if arch is None or not arch_has_vision_tower(arch.config, hf_config):
            return 0, None

        # Guaranteed by arch_has_vision_tower above.
        assert issubclass(arch.config, ArchConfigWithVisionCache)
        row_spec = arch.config.get_vision_cache_row_spec(hf_config)
        if row_spec is None:
            # Unreachable for configs built through from_args, which zero
            # vision_cache_utilization at construction when the row spec is
            # missing; tolerated for hand-built configs.
            logger.warning(
                "%s's arch config reports a per-entry estimate but no row "
                "spec (get_vision_cache_row_spec); no vision encoder cache "
                "will be reserved.",
                arch.name,
            )
            return 0, None

        return cls._reserve_vision_cache_blocks(
            pipeline_config, row_spec, available_memory, len(devices)
        )

    @classmethod
    def _reserve_vision_cache_blocks(
        cls,
        pipeline_config: PipelineConfig,
        row_spec: tuple[int, DType],
        available_memory: int,
        n_devices: int,
    ) -> tuple[int, VisionCachePlan]:
        """Reserve a block-mode byte budget for the vision encoder cache.

        ``vision_cache_utilization`` requests a fraction of the device KV
        cache pool budget (the 0.05 default auto-sizes a small slice). The
        request is rounded down to whole fixed-size blocks and returned as
        a
        :class:`~max.pipelines.lib.vision_encoder_cache.VisionCachePlan`
        that pipeline construction hands to :class:`VisionEncoderCache`.
        Capacity is bytes — a video simply spans more blocks than an
        image. Storage is sharded: each entry is stored once across the
        devices rather than replicated, so the per-device reservation is
        ``1/n_devices`` of the capacity and the rest stays with the KV
        cache.

        Returns:
            Total bytes reserved across devices, and the block-mode plan.

        Raises:
            ValueError: If the fraction is too small to fit a single
                block.
        """
        hidden_size, dtype = row_spec
        utilization = pipeline_config.runtime.vision_cache_utilization
        requested_bytes = int(available_memory * utilization) // n_devices
        block_bytes = (
            DEFAULT_VISION_CACHE_BLOCK_TOKENS
            * hidden_size
            * dtype.size_in_bytes
        )
        num_blocks = requested_bytes // block_bytes // n_devices * n_devices
        if num_blocks == 0:
            raise ValueError(
                f"vision_cache_utilization={utilization} reserves "
                f"{to_human_readable_bytes(requested_bytes)} of the "
                "KV cache pool, too small to fit one "
                f"{DEFAULT_VISION_CACHE_BLOCK_TOKENS}-token block "
                f"({to_human_readable_bytes(block_bytes)}) per device. "
                "Increase the fraction or set 0 to disable the vision "
                "encoder cache."
            )
        total_bytes = num_blocks * block_bytes
        plan = VisionCachePlan(
            bytes_per_device=total_bytes // n_devices,
            hidden_size=hidden_size,
            dtype=dtype,
        )
        logger.info(
            "Vision encoder cache: %d blocks x %d tokens sharded across "
            "%d device(s), %s reserved (%s per device).",
            num_blocks,
            DEFAULT_VISION_CACHE_BLOCK_TOKENS,
            n_devices,
            to_human_readable_bytes(total_bytes),
            to_human_readable_bytes(total_bytes // n_devices),
        )
        return total_bytes, plan

    @classmethod
    def _infer_optimal_batch_size(
        cls,
        arch_config: ArchConfig,
        devices: list[Device],
    ) -> int:
        """Infer the optimal batch size for the model.

        Args:
            arch_config: Architecture config that provides KV cache parameters.
            devices: The list of devices on which the model will run.
        """
        if not isinstance(arch_config, ArchConfigWithKVCache):
            return 1
        if len(devices) == 1 and devices[0].is_host:
            # batching on CPU is generally not useful, so we hard-code a batch size of 1.
            return 1
        return _DEFAULT_BATCH_SIZE
