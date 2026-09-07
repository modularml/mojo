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
import os
import queue
import time
import uuid
from dataclasses import dataclass

from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.kv_cache import (
    JengaKVCacheManager,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManagerInterface,
    TransferReqData,
)
from max.pipelines.lib import (
    PIPELINE_REGISTRY,
    MemoryPlan,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.profiler import Tracer, traced
from max.serve.config import Settings
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillFailure,
    PrefillProgressPing,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.scheduler.interface import Scheduler
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface._zmq_queue import ClientIdentity

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .batch_constructor.text_batch_constructor import BatchSchedulingStrategy
from .config import TokenGenerationSchedulerConfig
from .di_dispatchers import PrefillDispatcherServer
from .utils import SchedulerLogger

logger = logging.getLogger("max.serve")

# Decode-clock ping-back protocol for isolating DI dispatch latency (queue
# wait, CE time, ZMQ round trips) without comparing clocks across the
# decode/prefill nodes. Off by default -- two extra ZMQ messages per
# request is unwanted steady-state overhead outside an active
# investigation.
_DI_LATENCY_PING_ENABLED = os.getenv("MAX_SERVE_DI_LATENCY_PING", "0") == "1"

# Bracketing dispatch/completion logging around the model-execute call in
# schedule() -- on a stall, a dispatch line with no matching completion
# means prefill is stuck inside that one call, rather than the scheduler
# loop not running at all. Off by default -- two extra log lines per batch
# is unwanted steady-state overhead outside an active investigation.
_TRACE_PREFILL_BATCH = os.getenv("MAX_SERVE_TRACE_PREFILL_BATCH", "0") == "1"


@dataclass
class TransferDest:
    engine_name: str
    dst_block_ids: list[int]
    dst_replica_idx: int


@dataclass
class ActiveTransfer:
    """Prefill-side state for a posted KV transfer awaiting completion."""

    context: TextContext
    replica_idx: int
    transfer: TransferReqData


class PrefillScheduler(Scheduler):
    def __init__(
        self,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        scheduler_config: TokenGenerationSchedulerConfig,
        kv_cache: PagedKVCacheManagerInterface,
        dispatcher: PrefillDispatcherServer,
    ) -> None:
        # TODO(SERVOPT-1590): remove once Jenga supports DI.
        assert not isinstance(kv_cache, JengaKVCacheManager), (
            "Jenga KV cache is incompatible with Disaggregated Inference"
        )
        self.pipeline = pipeline
        self.scheduler_config = scheduler_config
        self.kv_cache = kv_cache

        # Initialize Scheduler state.
        self.active_transfers: dict[RequestID, ActiveTransfer] = {}
        self.request_id_to_reply_context: dict[
            RequestID, tuple[ClientIdentity, TransferDest]
        ] = {}

        self.transfer_engine = KVTransferEngine.from_paged_kv_cache(
            name=f"prefill_agent_{uuid.uuid4()}",
            kv_cache=kv_cache,
        )

        self.outstanding_cancelled_requests: set[RequestID] = set()

        # request_id -> time.monotonic() at enqueue, popped the first time
        # the request enters an executed CE batch. Feeds the
        # di_prefill_queue_wait_time metric; prefill-local, no wire cost.
        self._enqueue_time: dict[RequestID, float] = {}

        # Maps req_id → (context, src_replica_idx) for CE-complete requests
        # whose first generated token hasn't materialized yet due to the overlap scheduling
        # one-batch lag. Populated when a CE batch completes; resolved in the
        # next execute() call when the real token surfaces.
        self._pending_first_token: dict[RequestID, tuple[TextContext, int]] = {}

        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=BatchSchedulingStrategy.PREFILL_FIRST,
            # Requests parked in _pending_first_token are pre-active_transfers
            # state on the same block-release path: they already hold KV
            # blocks and will free them once their deferred first token
            # resolves into a transfer. Counting them prevents a false-fatal
            # InsufficientBlocksError during the one-batch overlap lag.
            get_inflight_kv_transfer_count=self._inflight_kv_transfer_count,
        )
        self.scheduler_logger = SchedulerLogger()
        self.dispatcher = dispatcher

    @traced
    def handle_cancel_request(self, message: CancelRequest) -> None:
        """Handles a cancel request by adding the request ID to the set of outstanding cancelled requests."""
        self.outstanding_cancelled_requests.add(message.id)

    @traced
    def handle_transfer_engine_request(
        self, message: KVTransferEngineMetadata, identity: ClientIdentity
    ) -> None:
        """Handles a engine registration request from the dispatcher."""
        logger.debug(f"connecting to remote transfer_engine: {message.name}")
        if message.name in self.transfer_engine.remote_connections:
            logger.info(f"Transfer engine {message.name} already connected")
            return

        self.transfer_engine.connect(message)

        self.dispatcher.send_reply_nowait(
            self.transfer_engine.metadata,
            identity,
        )

    def handle_prefill_request(
        self, message: PrefillRequest[TextContext], identity: ClientIdentity
    ) -> None:
        """Handles a prefill request from the dispatcher."""
        logger.debug("received request from decode node.")
        context = message.context
        assert context.tokens.generated_length == 0, (
            f"Expected needs_ce to be True. Invalid context: {context}"
        )
        # It is possible for the context to have a non-zero start_idx due to
        # decode using prefix caching.
        context.reset()
        self.batch_constructor.enqueue_new_request(context)
        self.request_id_to_reply_context[message.id] = (
            identity,
            TransferDest(
                engine_name=message.transfer_engine_name,
                dst_block_ids=message.dst_block_ids,
                dst_replica_idx=message.dst_replica_idx,
            ),
        )
        self._enqueue_time[message.id] = time.monotonic()
        if _DI_LATENCY_PING_ENABLED:
            self.dispatcher.send_reply_nowait(
                PrefillProgressPing(id=message.id, event="arrived"),
                identity,
            )

    def cleanup_active_transfers(self) -> None:
        """Cleans up completed transfers from the active transfers dictionary.

        Checks the status of all active transfers. For any transfer that is no longer in progress:
        - Releases pipeline resources
        - Removes the transfer from active_transfers
        """
        to_be_deleted = []
        for active in self.active_transfers.values():
            if self.transfer_engine.is_complete(active.transfer):
                self.transfer_engine.cleanup_transfer(active.transfer)
                blocks_released = len(
                    self.kv_cache.get_req_blocks(active.context)
                )
                # Release from paged cache (scheduler manages primary KV cache lifecycle)
                self.kv_cache.release(active.context)
                # Pipeline release handles special cases (spec decoding draft model KV cache)
                # For regular pipelines, release() is a no-op
                self.pipeline.release(active.context.request_id)
                to_be_deleted.append(active.context.request_id)
                logger.debug(
                    "KV transfer complete for request %s: releasing %d blocks "
                    "back to prefill pool. Remaining in-flight transfers: %d.",
                    active.context.request_id,
                    blocks_released,
                    len(self.active_transfers) - len(to_be_deleted),
                )

        for id in to_be_deleted:
            del self.active_transfers[id]

    def _inflight_kv_transfer_count(self, replica_idx: int) -> int:
        """Count of active_transfers + _pending_first_token entries holding
        blocks on this replica -- a transfer on a different replica's
        device pool can't free blocks on this one."""
        return sum(
            1
            for active in self.active_transfers.values()
            if active.replica_idx == replica_idx
        ) + sum(
            1 for _, r in self._pending_first_token.values() if r == replica_idx
        )

    def initiate_transfer_and_send_reply(
        self, context: TextContext, src_replica_idx: int
    ) -> None:
        """Initiates a KVTransfer for a decode request and sends the reply to the decode node.

        Args:
            context: The context of the decode request.
            src_replica_idx: The replica the request is on for Prefill
        """
        req_id = context.request_id
        if _DI_LATENCY_PING_ENABLED:
            # Peek (not pop) the identity: this is the CE-complete signal
            # for the ping-back protocol, sent before the pop below so
            # cancelled requests still get a ping matching their arrival one.
            ping_identity, _ = self.request_id_to_reply_context[req_id]
            self.dispatcher.send_reply_nowait(
                PrefillProgressPing(id=req_id, event="ce_done"),
                ping_identity,
            )
        identity, transfer_dest = self.request_id_to_reply_context.pop(req_id)

        # If cancelled, release the request's blocks instead of transferring them.
        if req_id in self.outstanding_cancelled_requests:
            self.outstanding_cancelled_requests.remove(req_id)
            blocks_held = len(self.kv_cache.get_req_blocks(context))
            self.kv_cache.release(context)
            self.pipeline.release(req_id)
            logger.warning(
                "Dropping cancelled request %s before KV transfer: released "
                "%d blocks on replica %d.",
                req_id,
                blocks_held,
                src_replica_idx,
            )
            return

        # Get Remote Metadata.
        remote_metadata = self.transfer_engine.remote_connections[
            transfer_dest.engine_name
        ]

        # Retrieve source block ids.
        req_id = context.request_id
        src_idxs = self.kv_cache.get_req_blocks(context)
        dst_idxs = transfer_dest.dst_block_ids
        assert len(src_idxs) == len(dst_idxs)

        # Transfer only the blocks that are not already on decode node.
        num_already_cached_blocks = dst_idxs.count(-1)
        src_idxs = src_idxs[num_already_cached_blocks:]
        dst_idxs = dst_idxs[num_already_cached_blocks:]
        assert dst_idxs.count(-1) == 0

        transfer_data = self.transfer_engine.initiate_send_transfer(
            remote_metadata,
            src_idxs,
            dst_idxs,
            src_replica_idx=src_replica_idx,
            dst_replica_idx=transfer_dest.dst_replica_idx,
        )
        self.active_transfers[req_id] = ActiveTransfer(
            context=context,
            replica_idx=src_replica_idx,
            transfer=transfer_data,
        )
        transfer_pinned = sum(
            len(self.kv_cache.get_req_blocks(at.context))
            for at in self.active_transfers.values()
        )
        block_count = self.kv_cache.block_count(src_replica_idx)
        logger.debug(
            "KV transfer started for request %s: blocks pinned on prefill "
            "pending transfer completion. Active in-flight transfers: %d, "
            "directly holding %d blocks. Free blocks: %d / %d total.",
            req_id,
            len(self.active_transfers),
            transfer_pinned,
            block_count.free,
            block_count.total,
        )

        assert context.tokens.generated_length != 0, (
            f"Invalid Context: Expected generated tokens to be at least one. Found: {context}"
        )
        assert context.tokens.processed_length > 0, (
            f"Invalid Context: Expected start_idx to be greater than 0. Found: {context}"
        )

        # Extract draft tokens from Eagle/MTP speculative decoding state.
        # These let the decode node seed its first spec-decode iteration
        # instead of starting with an empty draft cache.  When speculative
        # decoding is active, the unified Eagle/MTP model always populates
        # draft_tokens_to_verify during CE; error if it hasn't.
        draft_tokens: list[int] | None = None
        if (
            self.scheduler_config.num_speculative_tokens > 0
            and not context.is_done
        ):
            # Done contexts (max_gen_tokens=1) produce no further TG steps on
            # the decode pod, so draft tokens are unnecessary. For all other
            # contexts, draft tokens must be present after CE.
            if not context.spec_decoding_state.draft_tokens_to_verify:
                raise ValueError(
                    f"Expected draft tokens on context {req_id} after CE "
                    f"with speculative decoding enabled, but none were "
                    f"populated. Check that the unified Eagle/MTP pipeline "
                    f"is wired in for prefill_only."
                )
            draft_tokens = context.spec_decoding_state.draft_tokens_to_verify

        self.dispatcher.send_reply_nowait(
            PrefillResponse(
                id=req_id,
                # The last buffer slot may be an unrealized future-token
                # placeholder while overlap forwards are in flight; send the
                # newest realized token.
                generated_token_id=context.last_realized_token,
                transfer_metadata=transfer_data,
                draft_tokens=draft_tokens,
            ),
            identity,
        )

    @traced
    def schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Executes the current batch of requests and sends completed requests to decode.

        Processes the active batch through the pipeline, handles any chunked prefill requests,
        and sends completed requests to the decode queue while resetting their token indices.
        """
        # Execute the Batch
        assert len(inputs.batches) > 0
        if _TRACE_PREFILL_BATCH:
            batch_sizes = [len(batch) for batch in inputs.batches]
            dispatch_t0 = time.monotonic()
            logger.info(
                "Dispatching prefill batch: %d replica(s), sizes=%s",
                len(inputs.batches),
                batch_sizes,
            )
        responses = self.pipeline.execute(inputs)
        if _TRACE_PREFILL_BATCH:
            logger.info(
                "Completed prefill batch: %d replica(s), sizes=%s, took %.1fms",
                len(inputs.batches),
                batch_sizes,
                (time.monotonic() - dispatch_t0) * 1000,
            )

        self.batch_constructor.advance_requests(inputs)

        # Resolve: transfer any deferred requests whose real token has now
        # materialized (overlap pipeline two-phase pattern).
        for req_id in responses:
            if req_id in self._pending_first_token:
                context, replica_idx = self._pending_first_token.pop(req_id)
                self.initiate_transfer_and_send_reply(
                    context, src_replica_idx=replica_idx
                )

        # Decide whether the pipeline deferred the current batch's outputs
        # (overlap active) or returned them synchronously (no overlap / spec
        # decode with overlap disabled).
        pipeline_deferred = (
            hasattr(self.pipeline, "has_pending_outputs")
            and self.pipeline.has_pending_outputs()
        )
        if pipeline_deferred:
            # Overlap: current batch's real tokens are not yet available.
            # Stash CE-complete requests; they will be resolved in the next
            # execute() call when the real token surfaces.
            for replica_idx, replica in enumerate(
                self.batch_constructor.replicas
            ):
                for req_id, context in replica.tg_reqs.items():
                    self._pending_first_token[req_id] = (context, replica_idx)
        else:
            # Synchronous: token is already in context.tokens[-1].
            for replica_idx, replica in enumerate(
                self.batch_constructor.replicas
            ):
                for context in replica.tg_reqs.values():
                    self.initiate_transfer_and_send_reply(
                        context, src_replica_idx=replica_idx
                    )

        # Remove all TG requests from the batch constructor.
        num_terminated_reqs = len(self.batch_constructor.all_tg_reqs)
        self.batch_constructor.clear_tg_reqs()
        return num_terminated_reqs

    @traced
    def run_iteration(self) -> SchedulerProgress:
        """Main scheduling loop that processes prefill requests.

        Receives requests, creates batches, and schedules them for processing
        while handling errors and cancelled requests.

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """

        while True:
            try:
                request, identity = self.dispatcher.recv_request_nowait()
            except queue.Empty:
                break
            if isinstance(request, CancelRequest):
                self.handle_cancel_request(request)
            elif isinstance(request, KVTransferEngineMetadata):
                self.handle_transfer_engine_request(request, identity)
            elif isinstance(request, PrefillRequest):
                self.handle_prefill_request(request, identity)
            else:
                raise ValueError(f"Invalid request type: {request}")

        # Cleanup active transfers.
        self.cleanup_active_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        for failed_id, error in self.batch_constructor.take_grammar_failed():
            # Never reaches an executed CE batch, so drop its queue-wait
            # entry here rather than leaving it to accumulate.
            self._enqueue_time.pop(failed_id, None)
            reply_context = self.request_id_to_reply_context.pop(
                failed_id, None
            )
            if reply_context is None:
                continue
            identity, _ = reply_context
            self.dispatcher.send_reply_nowait(
                PrefillFailure(id=failed_id, error=error), identity
            )

        # With the overlap pipeline, a pending _prev_batch must be drained
        # even when the current batch is empty (last-batch flush).
        has_pending_outputs = (
            hasattr(self.pipeline, "has_pending_outputs")
            and self.pipeline.has_pending_outputs()
        )
        if not (inputs or has_pending_outputs):
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        # Attribute queue wait to every request entering its first executed
        # CE batch, right as that batch is about to run.
        # Pop-on-first-sight means chunked-prefill continuations (already
        # popped on their first chunk) don't re-fire.
        for replica_batch in inputs.batches:
            for context in replica_batch:
                enqueued_at = self._enqueue_time.pop(context.request_id, None)
                if enqueued_at is not None:
                    METRICS.di_prefill_queue_wait_time(
                        (t0 - enqueued_at) * 1000
                    )
        with Tracer(f"_schedule({inputs})"):
            num_terminated_reqs = self.schedule(inputs)
        t1 = time.monotonic()
        batch_execution_time_s = t1 - t0

        # Log batch metrics. When the overlap pipeline is active, the
        # wall-clock time measured above describes the previously enqueued
        # batch; the pipeline reports that batch's composition and timing so
        # telemetry is attributed to the correct batch type.
        is_overlap_active = bool(
            getattr(self.pipeline, "overlap_active", False)
        )
        self.scheduler_logger.log_metrics(
            sch_config=self.scheduler_config,
            inputs=inputs,
            kv_cache=self.kv_cache,
            batch_creation_time_s=batch_creation_time_s,
            batch_execution_time_s=batch_execution_time_s,
            num_pending_reqs=len(self.batch_constructor.all_ce_reqs),
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
            overlap_active=is_overlap_active,
            completed_batch_stats=self.pipeline.take_completed_batch_stats()
            if hasattr(self.pipeline, "take_completed_batch_stats")
            else None,
            batch_vision_metrics=self.pipeline.batch_vision_metrics()
            if hasattr(self.pipeline, "batch_vision_metrics")
            else None,
            batch_video_metrics=self.pipeline.batch_video_metrics()
            if hasattr(self.pipeline, "batch_video_metrics")
            else None,
        )

        return SchedulerProgress.MADE_PROGRESS


def load_prefill_scheduler(
    pipeline: TextGenerationPipeline[TextContext],
    pipeline_config: PipelineConfig,
    settings: Settings,
    memory_plan: MemoryPlan | None,
) -> PrefillScheduler:
    # Validate speculative decoding configuration for prefill-only mode.
    spec_config = pipeline_config.speculative
    if spec_config is not None:
        if not (spec_config.is_eagle() or spec_config.is_mtp()):
            raise ValueError(
                f"Unsupported speculative method "
                f"'{spec_config.speculative_method}' with "
                f"pipeline_role='prefill_only'. Only 'eagle' and 'mtp' "
                f"are supported."
            )
        logger.info(
            "Prefill-only mode with speculative decoding "
            f"(method={spec_config.speculative_method})."
        )

    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config, pipeline.max_batch_size, memory_plan
    )

    # Decode incoming prefill requests into the architecture's concrete
    # context type (e.g. a TextAndVisionContext subclass for VLMs), mirroring
    # the aggregated serving path. Decoding into the base TextContext would
    # silently drop the subclass and its vision fields on the wire.
    context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
    assert issubclass(context_type, TextContext)

    return PrefillScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        kv_cache=pipeline.kv_manager,
        dispatcher=PrefillDispatcherServer(
            bind_addr=settings.di_bind_address,
            context_type=context_type,
        ),
    )
