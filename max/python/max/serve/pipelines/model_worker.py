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
import multiprocessing
import os
import time
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from multiprocessing.synchronize import Event
from typing import Any, Protocol, runtime_checkable

import uvloop
from max.driver import Device, DevicePinnedBuffer
from max.driver.driver import load_device
from max.dtype import DType
from max.experimental.nn._compilation_timer import collect_compilation_stats
from max.pipelines.context import BaseContextType
from max.pipelines.kv_cache import (
    DummyKVCache,
    PagedKVCacheManagerInterface,
)
from max.pipelines.lib import MemoryPlan, PipelineConfig, PipelineModel
from max.pipelines.lib.eplb_stats import EplbStatsAccumulator
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    PipelinesFactory,
)
from max.profiler import Tracer, traced
from max.serve._exceptions import detect_and_wrap_ooms
from max.serve.config import MetricRecordingMethod, Settings
from max.serve.pipelines.eplb_stats_rpc import (
    EplbStatsBackend,
    EplbStatsResetBackend,
)
from max.serve.pipelines.reset_prefix_cache import ResetPrefixCacheBackend
from max.serve.pipelines.telemetry_worker import MetricClient
from max.serve.process_control import subprocess_manager
from max.serve.scheduler import load_scheduler
from max.serve.scheduler.base import SchedulerProgress
from max.serve.telemetry.common import (
    configure_kernel_tracing,
    configure_logging,
    configure_metrics,
    configure_tracing,
)
from max.serve.telemetry.gc_utils import (
    _release_free_host_memory,
    freeze_gc_heap,
    install_gc_debugger,
)
from max.serve.telemetry.metrics import METRICS
from max.serve.telemetry.stopwatch import record_ms
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    sleep_with_backoff,
)
from max.serve.worker_interface.lora_request_processor import (
    LoRARequestProcessor,
)

logger = logging.getLogger("max.serve")

GiB = 1024 * 1024 * 1024


@runtime_checkable
class SupportsGraphCaptureWarmup(Protocol):
    max_batch_size: int

    def warmup_graph_capture(self) -> None: ...


@runtime_checkable
class SupportsGraphSynthesisBuckets(Protocol):
    def prepare_graph_synthesis_buckets(self) -> None: ...


def _prime_pinned_memory_cache(device: Device, bytes: int = GiB) -> None:
    """Prime the pinned memory manager cache for the given device.

    Populate the host memory manager by allocating and immediately freeing a
    large pinned tensor. If the host memory manager is activated, future allocations
    and frees will likely hit the cache and be much faster. By priming the cache,
    we ensure that the slow call to the driver allocator occurs during bootup
    and not during the first inference request. Note that calls to the driver's
    pinned memory allocator can be pretty slow (>1s in some cases).

    Since pinned memory is only supported on accelerators, calling this method
    on a CPU device is a no-op.

    Args:
        device: The device to prime the cache for.
        bytes: The number of bytes to allocate.
    """
    if device.is_host:
        return
    pinned = DevicePinnedBuffer(shape=(bytes,), dtype=DType.int8, device=device)
    del pinned


def _get_eplb_stats_accumulator(
    pipeline: Pipeline[Any, Any],
    enabled: bool,
) -> EplbStatsAccumulator | None:
    """Retrieve the pipeline's pre-built EplbStatsAccumulator if profiling is on.

    Args:
        pipeline: The model pipeline running in this worker process.
        enabled: pipeline_config.runtime.eplb_profile i.e. whether the user has opted in to EPLB profiling for this run.

    Returns:
        The accumulator constructed by the pipeline's load_model, or
        None if profiling is disabled or the pipeline does not expose one.
    """
    if not enabled:
        return None

    pipeline_model = get_pipeline_model(pipeline)
    accumulator = getattr(pipeline_model, "_eplb_stats_accumulator", None)
    if accumulator is None:
        logger.warning(
            "MAX_SERVE_EPLB_PROFILE is enabled but pipeline %s does not "
            "expose an EplbStatsAccumulator; routing histograms unavailable.",
            type(pipeline_model).__name__ if pipeline_model else "None",
        )
        return None

    logger.info(
        "EPLB stats profiling enabled: %d MoE layers x %d logical experts (top-%d).",
        accumulator.metadata.num_moe_layers,
        accumulator.metadata.num_logical_experts,
        accumulator.metadata.num_experts_per_token,
    )
    return accumulator


def get_reset_prefix_cache_backend(
    pipeline: Pipeline[Any, Any],
    zmq_endpoint_base: str,
) -> tuple[ResetPrefixCacheBackend | None, PagedKVCacheManagerInterface | None]:
    """Get the KV cache manager from a pipeline, if reset is supported.

    Args:
        pipeline: The pipeline to extract the KV cache manager from.
        zmq_endpoint_base: Base ZMQ endpoint for the reset-prefix-cache queue.

    Returns:
        A reset backend and the KV cache manager when the pipeline uses a real
        paged or Jenga manager; ``(None, None)`` otherwise.
    """

    if hasattr(pipeline, "kv_manager"):
        kv_manager = pipeline.kv_manager
        if isinstance(
            kv_manager, PagedKVCacheManagerInterface
        ) and not isinstance(kv_manager, DummyKVCache):
            return ResetPrefixCacheBackend(zmq_endpoint_base), kv_manager
    return None, None


def get_pipeline_model(
    pipeline: Pipeline[Any, Any],
) -> PipelineModel[Any] | None:
    return getattr(pipeline, "_pipeline_model", None)


class ModelWorker:
    """A stateless namespace class for organizing ModelWorker functionality.

    This class has no instance state or methods, and serves purely as a namespace
    to organize the async functionality associated with running a single ModelWorker
    process. All methods are static and handle tasks like worker initialization,
    scheduler configuration, and process lifecycle management.
    """

    @staticmethod
    @traced
    def _configure_metrics(
        settings: Settings,
        metric_client: MetricClient,
    ) -> None:
        """Configure metrics recording for the model worker process.

        Args:
            settings: Global server settings containing metric configuration
            metric_client: Client for recording metrics
        """
        supported_methods = [
            MetricRecordingMethod.NOOP,
            MetricRecordingMethod.PROCESS,
        ]
        if settings.metric_recording not in supported_methods:
            logger.info(
                "Unsupported recording method. Metrics unavailable in model worker"
            )
            return

        configure_metrics(settings)
        METRICS.configure(metric_client)

    @staticmethod
    @traced
    async def run(
        alive: Event,
        model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
        pipeline_config: PipelineConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
        model_worker_interface: ModelWorkerInterface[
            BaseContextType, PipelineOutputType
        ],
        zmq_endpoint_base: str,
        memory_plan: MemoryPlan | None,
        spawn_start_wall_ts: float | None = None,
    ) -> None:
        """Runs a model worker process.

        Configures logging and metrics, initializes the model pipeline and scheduler,
        and executes the main worker loop.

        Args:
            pc: Process control for managing worker lifecycle
            model_factory: Factory function to create the model pipeline
            pipeline_config: The config for the pipeline
            settings: Global server settings
            metric_client_factory: Factory function to create metric client
            zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
                the API server process and this worker process.
            memory_plan: The memory plan the pipeline was sized against,
                consumed by the scheduler config.
            spawn_start_wall_ts: ``time.time()`` recorded in the parent just
                before spawning this worker. Used to log how long the worker
                process took to start (Python imports + driver init), which
                can dominate first-run startup on cold filesystem caches.
        """
        configure_logging(settings)
        configure_tracing(settings)
        configure_kernel_tracing(settings)
        pid = os.getpid()
        logger.debug("Starting model worker on process %d!", pid)

        run_start_s = time.monotonic()
        spawn_duration_s = (
            time.time() - spawn_start_wall_ts
            if spawn_start_wall_ts is not None
            else None
        )
        if spawn_duration_s is not None:
            logger.info(
                "Worker process startup took %.2fs (Python imports + driver init)",
                spawn_duration_s,
            )

        async with AsyncExitStack() as exit_stack:
            # Configure Metrics
            metric_client = await exit_stack.enter_async_context(
                metric_client_factory()
            )

            ModelWorker._configure_metrics(settings, metric_client)

            # improves diagnostic messages for gpu out-of-memory errors
            exit_stack.enter_context(detect_and_wrap_ooms())

            # Prime the pinned memory cache in the model worker process.
            # The first DevicePinnedBuffer allocation per GPU triggers
            # heavyweight driver context initialization that can take
            # seconds. Doing it here at startup avoids that latency
            # hitting the first real request.
            # Use any model's device_specs — all components share the same
            # device. Reads the raw field, which may name a GPU the model was
            # downcast off of; priming an unused GPU is harmless.
            prime_start_s = time.monotonic()
            any_model = next(iter(pipeline_config.models.values()))
            first_device = load_device(any_model.device_specs[0])
            _prime_pinned_memory_cache(first_device)
            if first_device.api in ("cuda", "hip"):
                for spec in any_model.device_specs[1:]:
                    _prime_pinned_memory_cache(load_device(spec))
            prime_duration_s = time.monotonic() - prime_start_s
            logger.info(
                "Pinned memory cache primed in %.2fs",
                prime_duration_s,
            )

            # Initialize token generator.
            logger.info("Initializing model pipeline...")
            factory_start_s = time.monotonic()
            with (
                record_ms(METRICS.model_load_time),
                Tracer("model_factory"),
                collect_compilation_stats() as compile_stats,
            ):
                pipeline = model_factory()
            factory_duration_s = time.monotonic() - factory_start_s
            other_s = max(
                0.0,
                factory_duration_s
                - compile_stats.build_seconds
                - compile_stats.compile_seconds
                - compile_stats.init_seconds,
            )
            unaccounted_compile_s = max(
                0.0,
                compile_stats.compile_seconds
                - compile_stats.labeled_compile_seconds,
            )
            unaccounted_init_s = max(
                0.0,
                compile_stats.init_seconds - compile_stats.labeled_init_seconds,
            )
            logger.info(
                "Model pipeline initialized in %.1fs "
                "(graph build: %.1fs, graph compile: %.1fs "
                "[unaccounted: %.1fs], init: %.1fs [unaccounted: %.1fs], "
                "other: %.1fs)",
                factory_duration_s,
                compile_stats.build_seconds,
                compile_stats.compile_seconds,
                unaccounted_compile_s,
                compile_stats.init_seconds,
                unaccounted_init_s,
                other_s,
            )

            if os.getenv("MODULAR_MAX_RELEASE_FREE_HOST_MEMORY"):
                with Tracer("release_free_host_memory"):
                    _release_free_host_memory()

            warmup_duration_s = 0.0
            with Tracer("graph_capture_warmup"):
                if pipeline_config.runtime.device_graph_capture:
                    if not isinstance(pipeline, SupportsGraphCaptureWarmup):
                        raise ValueError(
                            "device_graph_capture is enabled but the pipeline "
                            "does not support graph-capture warmup."
                        )
                    warmup_start_s = time.monotonic()
                    pipeline.warmup_graph_capture()
                    warmup_duration_s = time.monotonic() - warmup_start_s
                    logger.info(
                        "Device graph capture warmup completed in %.2fs "
                        "(model=%s, max_batch_size=%d).",
                        warmup_duration_s,
                        pipeline_config.models.main_architecture_name,
                        pipeline.max_batch_size,
                    )
                elif (
                    pipeline_config.runtime.experimental_device_graph_synthesis
                ):
                    if not isinstance(pipeline, SupportsGraphSynthesisBuckets):
                        raise ValueError(
                            "experimental_device_graph_synthesis is enabled but the "
                            "pipeline does not support synthesis bucketing."
                        )
                    pipeline.prepare_graph_synthesis_buckets()
                    logger.info(
                        "Device graph synthesis bucketing prepared (model=%s).",
                        pipeline_config.models.main_architecture_name,
                    )

            total_in_run_s = time.monotonic() - run_start_s
            spawn_str = (
                f"spawn: {spawn_duration_s:.1f}s, "
                if spawn_duration_s is not None
                else ""
            )
            logger.info(
                "Model worker startup total: %.1fs — %sdriver: %.1fs, "
                "compile: %.1fs [unaccounted: %.1fs], init: %.1fs "
                "[unaccounted: %.1fs], other: %.1fs, warmup: %.1fs",
                (spawn_duration_s or 0.0) + total_in_run_s,
                spawn_str,
                prime_duration_s,
                compile_stats.build_seconds + compile_stats.compile_seconds,
                unaccounted_compile_s,
                compile_stats.init_seconds,
                unaccounted_init_s,
                other_s,
                warmup_duration_s,
            )

            # Emit the same per-phase breakdown on the model_load_time
            # histogram so pod startup time can be tracked in production.
            # One metric split by the 'component' tag keeps the dashboard
            # aligned with the logs above; the untagged record_ms() above
            # remains the model-factory aggregate. Values are converted to
            # milliseconds to match the metric's unit.
            METRICS.model_load_time(compile_stats.build_seconds * 1e3, "build")
            METRICS.model_load_time(
                compile_stats.compile_seconds * 1e3, "compile"
            )
            METRICS.model_load_time(compile_stats.init_seconds * 1e3, "init")
            METRICS.model_load_time(warmup_duration_s * 1e3, "graph_capture")
            METRICS.model_load_time(prime_duration_s * 1e3, "pinned_memory")
            if spawn_duration_s is not None:
                METRICS.model_load_time(spawn_duration_s * 1e3, "spawn")
            METRICS.model_load_time(
                ((spawn_duration_s or 0.0) + total_in_run_s) * 1e3, "total"
            )

            # Boot up the api worker comms
            worker_queues = await exit_stack.enter_async_context(
                model_worker_interface.model_worker_queues()
            )

            # Retrieve Scheduler.
            scheduler = load_scheduler(
                pipeline,
                pipeline_config,
                settings,
                worker_queues,
                memory_plan,
            )

            # Get the reset prefix cache backend.
            reset_prefix_cache_backend, kv_cache = (
                get_reset_prefix_cache_backend(pipeline, zmq_endpoint_base)
            )

            # Tear down the KV connector when the worker exits (normal exit,
            # exception, or SIGTERM-driven cancellation). This drains host/disk
            # transfers and, for the tiered connector, removes the on-disk
            # offload directory so it isn't leaked across restarts.
            if kv_cache is not None:
                exit_stack.callback(kv_cache.shutdown)

            # Get the EPLB stats accumulator (None unless profiling is
            # enabled and the pipeline supports it).
            eplb_stats_accumulator = _get_eplb_stats_accumulator(
                pipeline, settings.eplb_profile
            )

            if eplb_stats_accumulator is not None:
                # Zero warmup/graph-capture skew so the first snapshot is clean.
                eplb_stats_accumulator.reset()

            eplb_stats_backend = (
                EplbStatsBackend(
                    zmq_endpoint_base,
                    eplb_stats_accumulator,
                )
                if eplb_stats_accumulator is not None
                else None
            )

            eplb_stats_reset_backend = (
                EplbStatsResetBackend(
                    zmq_endpoint_base,
                    eplb_stats_accumulator,
                )
                if eplb_stats_accumulator is not None
                else None
            )

            # Maybe retrieve LoRA manager.
            lora_request_processor = None
            pipeline_model = get_pipeline_model(pipeline)
            if pipeline_config.lora:
                assert pipeline_model is not None
                lora_manager = pipeline_model.lora_manager
                assert lora_manager is not None
                lora_request_processor = LoRARequestProcessor(
                    lora_manager,
                    zmq_endpoint_base,
                )

            # Freeze the GC heap now that all long-lived startup objects are
            # allocated to reduce the work in subsequent GC collections.
            # See: https://github.com/vllm-project/vllm/blob/95ed0feaa5cd7fb16d72c53ce04950aaf07c4698/vllm/utils/gc_utils.py
            gc_freeze_start_s = time.monotonic()
            with Tracer("gc_freeze_after_warmup"):
                frozen_objects = freeze_gc_heap()
            gc_freeze_ms = (time.monotonic() - gc_freeze_start_s) * 1000.0
            logger.info(
                "Froze %d GC-tracked objects after warmup in %.1fms.",
                frozen_objects,
                gc_freeze_ms,
                extra={
                    "event": "gc_freeze",
                    "gc_frozen_objects": frozen_objects,
                    "gc_freeze_ms": gc_freeze_ms,
                },
            )

            # Optionally instrument CPython garbage-collection pauses. This is
            # done after warmup as graph capture can trigger a large number
            # of GC collections that are noisy.
            install_gc_debugger(
                enabled=settings.gc_debug,
                top_objects=settings.gc_debug_top_objects,
            )

            # Mark the start of the process, and run the scheduler.
            logger.debug("Started model worker!")

            count_no_progress = 0
            while True:
                alive.set()
                # Checks for new LoRA requests and processes them.
                if lora_request_processor is not None:
                    lora_request_processor.process_lora_requests()
                # Check for request to reset prefix cache.
                if (
                    reset_prefix_cache_backend is not None
                    and reset_prefix_cache_backend.should_reset_prefix_cache()
                ):
                    assert kv_cache is not None
                    kv_cache.reset_prefix_cache()

                # Serve any pending EP stats requests.
                if eplb_stats_backend is not None:
                    eplb_stats_backend.serve_pending_requests()

                if eplb_stats_reset_backend is not None:
                    eplb_stats_reset_backend.serve_pending_requests()

                # This method must terminate in a reasonable amount of time
                # so that the ProcessMonitor heartbeat is periodically run.
                progress = scheduler.run_iteration()
                if progress == SchedulerProgress.NO_PROGRESS:
                    await sleep_with_backoff(count_no_progress)
                    count_no_progress += 1
                else:
                    count_no_progress = 0

        logger.debug("Stopped model worker!")

    @staticmethod
    @traced
    def __call__(
        alive: Event,
        model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
        pipeline_config: PipelineConfig,
        settings: Settings,
        metric_client_factory: Callable[
            [], AbstractAsyncContextManager[MetricClient]
        ],
        model_worker_interface: ModelWorkerInterface[
            BaseContextType, PipelineOutputType
        ],
        zmq_endpoint_base: str,
        memory_plan: MemoryPlan | None,
        spawn_start_wall_ts: float | None = None,
    ) -> None:
        """Primary entry point for running a ModelWorker process.

        This method is called when starting a new ModelWorker process. It initializes the event loop
        using uvloop and runs the main ModelWorker.run coroutine. The process handles model inference
        requests and manages the lifecycle of the underlying model pipeline.

        Args:
            pc: Process control for managing worker lifecycle
            model_factory: Factory for creating model pipeline instances
            pipeline_config: The config for the pipeline
            settings: Global server settings
            metric_client_factory: Factory for creating metric client instances
            zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
                the API server process and this worker process.
        """
        try:
            uvloop.run(
                ModelWorker.run(
                    alive,
                    model_factory,
                    pipeline_config,
                    settings,
                    metric_client_factory,
                    model_worker_interface,
                    zmq_endpoint_base,
                    memory_plan,
                    spawn_start_wall_ts,
                )
            )
        except KeyboardInterrupt:
            pass  # suppress noisy stack traces for user abort


@asynccontextmanager
async def start_model_worker(
    model_factory: PipelinesFactory[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    metric_client: MetricClient,
    model_worker_interface: ModelWorkerInterface[
        BaseContextType, PipelineOutputType
    ],
    zmq_endpoint_base: str,
    memory_plan: MemoryPlan | None,
) -> AsyncGenerator[ModelWorkerProxy[BaseContextType, PipelineOutputType]]:
    """Starts a model worker and associated process.

    Args:
        model_factory: Factory for creating model pipeline instances
        pipeline_config: The config for the pipeline
        settings: Global server settings
        metric_client: Metric client for recording metrics
        model_worker_interface: Interface for communicating with the worker
        zmq_endpoint_base: Prefix for ZMQ IPC endpoints shared between
            the API server process and the worker process.
        memory_plan: The memory plan the pipeline was sized against,
            consumed by the worker's scheduler config.

    Returns:
        AsyncIterator[Worker]: Iterator to model worker.

    Yields:
        Iterator[AsyncIterator[Worker]]: _description_
    """
    worker_name = "MODEL_" + str(uuid.uuid4())
    logger.info("Starting worker: %s", worker_name)

    mp = multiprocessing.get_context("spawn")
    async with subprocess_manager("Model Worker") as proc:
        alive = mp.Event()
        spawn_start_wall_ts = time.time()
        proc.start(
            ModelWorker(),
            alive,
            model_factory,
            pipeline_config,
            settings,
            metric_client.cross_process_factory(settings),
            model_worker_interface,
            zmq_endpoint_base,
            memory_plan,
            spawn_start_wall_ts,
        )

        logger.info("Waiting for model worker readiness")
        await proc.ready(alive, timeout=settings.mw_timeout_s)
        logger.info("Model worker ready")

        if settings.use_heartbeat:
            proc.watch_heartbeat(alive, timeout=settings.mw_health_fail_s)

        logger.debug("Model worker task is ready")

        async with model_worker_interface.model_worker_proxy() as model_worker:
            # Block until the worker channel is connected before serving, so
            # runtime admission never has to disambiguate "not connected yet"
            # from "queue full." Reuses the model-worker readiness budget.
            await model_worker.wait_until_connected(settings.mw_timeout_s)
            logger.debug("Model worker channel connected")
            yield model_worker
