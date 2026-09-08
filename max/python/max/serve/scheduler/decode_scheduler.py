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
import queue
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto

from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.kv_cache import (
    InsufficientBlocksError,
    JengaKVCacheManager,
    KVTransferEngine,
    KVTransferEngineMetadata,
    PagedKVCacheManagerInterface,
    TransferReqData,
)
from max.pipelines.kv_cache.kv_connector import KVConnectorTransfer
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
from max.serve.queue import (
    MAXPullQueue,
    MAXPushQueue,
    drain_queue,
)
from max.serve.scheduler.base import (
    CancelRequest,
    PrefillFailure,
    PrefillProgressPing,
    PrefillRequest,
    PrefillResponse,
)
from max.serve.scheduler.di_dispatchers import DecodeDispatcherClient
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.metrics import METRICS

from .base import SchedulerProgress
from .batch_constructor import TextBatchConstructor
from .batch_constructor.text_batch_constructor import BatchSchedulingStrategy
from .config import TokenGenerationSchedulerConfig
from .dp_padding import DPBatchPadder
from .utils import (
    SchedulerLogger,
    get_cancelled_reqs,
)

logger = logging.getLogger("max.serve")


class DecodeRequestPhase(Enum):
    """Lifecycle phase of a decode-side request, from admission to TG."""

    AWAITING_PREFILL = auto()
    """Sent to prefill; no ``PrefillResponse`` yet."""

    TRANSFERRING = auto()
    """``PrefillResponse`` landed; its KV transfer is in flight."""


@dataclass
class PendingDecodeRequest:
    """Decode-side state for a request between admission and TG (or its
    cancellation), spanning both ``DecodeRequestPhase`` values.

    ``phase_entered_at`` is reset on every phase transition, so TTL sweeps
    always measure time in the *current* phase rather than a stale
    prior-phase timestamp.
    """

    context: TextContext
    replica_idx: int
    phase: DecodeRequestPhase
    # The admission-time KV onload (a reused-prefix H2D copy). Release
    # and TG-enqueue sites must wait for this too, not just the prefill
    # transfer.
    onload_event: KVConnectorTransfer
    phase_entered_at: float = field(default_factory=time.monotonic)
    transfer: TransferReqData | None = None
    # Decode-clock receipt times for prefill's ping-back events, used to
    # derive di_prefill_span / di_reply_rtt. None until the corresponding
    # ping arrives (or forever, if MAX_SERVE_DI_LATENCY_PING is off at
    # prefill).
    arrived_ping_at: float | None = None
    ce_done_ping_at: float | None = None
    # Cancelled while onload and/or prefill's transfer may still be
    # writing into these blocks. Release defers to
    # check_for_completed_transfers until both are confirmed landed.
    cancelled: bool = False


class DecodeScheduler(Scheduler):
    def __init__(
        self,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        scheduler_config: TokenGenerationSchedulerConfig,
        kv_cache: PagedKVCacheManagerInterface,
        *,
        request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[TextGenerationOutput]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        dispatcher: DecodeDispatcherClient,
        dp_padder: DPBatchPadder | None = None,
    ) -> None:
        # TODO(SERVOPT-1590): remove once Jenga supports DI.
        assert not isinstance(kv_cache, JengaKVCacheManager), (
            "Jenga KV cache is incompatible with Disaggregated Inference"
        )
        # Initialize Pipeline and Config
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.kv_cache = kv_cache

        # Initialize Queues
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue

        self.dispatcher = dispatcher

        # Initialize Scheduler state.
        self.pending_reqs: OrderedDict[RequestID, TextContext] = OrderedDict()
        self.requests: dict[RequestID, PendingDecodeRequest] = {}
        self.prefill_reqs_per_replica: list[int] = [
            0 for _ in range(scheduler_config.data_parallel_degree)
        ]

        self.transfer_engine = KVTransferEngine.from_paged_kv_cache(
            name=f"decode_agent_{uuid.uuid4()}",
            kv_cache=self.kv_cache,
        )
        self.batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
            dp_padder=dp_padder,
            # A `requests` entry doesn't free its own blocks on resolution
            # -- it converts into a tg_reqs reservation on the same blocks
            # -- but that reservation is then preemptible like any other TG
            # request, so its presence still means a stuck allocation isn't
            # necessarily a dead end.
            get_inflight_kv_transfer_count=self._inflight_kv_transfer_count,
        )
        self.scheduler_logger = SchedulerLogger()
        # request_id -> time.monotonic() when a request first lands in
        # self.pending_reqs, popped on successful admission (not on
        # a re-insert after InsufficientBlocksError, so the eventual wait
        # covers the full time including any failed-alloc retries).
        self._admission_enqueue_time: dict[RequestID, float] = {}
        # request_id -> time.monotonic() when a prefill-to-decode handoff
        # landed (handle_prefill_response's constrained-request branch),
        # popped once the real first token is actually sent from
        # schedule() (or discarded if the request is cancelled/evicted
        # before that). Only ever populated when the handoff feature flag
        # is on.
        self._handoff_landed_at: dict[RequestID, float] = {}
        self._last_batch_activity: float = time.monotonic()
        # None corresponds to the default destination address.
        # TODO: delete the default destination address.
        self.remote_endpoints: set[str] = set()

    @traced
    def handle_transfer_engine_response(
        self, message: KVTransferEngineMetadata
    ) -> None:
        logger.debug(f"connecting to remote transfer engine: {message.name}")
        self.transfer_engine.connect(message)

    def handle_prefill_failure(self, message: PrefillFailure) -> None:
        """Terminates a request the prefill node rejected.

        No KV transfer was started, so the blocks are released immediately,
        mirroring the pre-transfer cancellation path.
        """
        pending = self.requests.pop(message.id, None)
        if pending is None:
            return
        self.prefill_reqs_per_replica[pending.replica_idx] -= 1
        self.batch_constructor.release_grammar_build(message.id)
        self.kv_cache.release(pending.context)
        self.pipeline.release(message.id)
        self.response_queue.put_nowait(
            {message.id: SchedulerResult.failed(message.error)}
        )

    def handle_prefill_response(self, message: PrefillResponse) -> None:
        """Handles a prefill response from the dispatcher."""
        request_id = message.id

        # The request may have been cancelled while the prefill response
        # was in-flight over ZMQ.  Discard the stale response
        pending = self.requests.get(request_id)
        if pending is None:
            return

        if pending.cancelled:
            # Client already got SchedulerResult.cancelled() -- don't
            # also emit a generation result. Still record the transfer
            # so check_for_completed_transfers waits for it before
            # releasing these blocks.
            pending.phase = DecodeRequestPhase.TRANSFERRING
            pending.phase_entered_at = time.monotonic()
            pending.transfer = message.transfer_metadata
            return

        postprocess_start = time.monotonic()
        if pending.ce_done_ping_at is not None:
            # Decode-clock span from prefill's "ce_done" ping to this real
            # reply landing (reply construction + KV-transfer
            # initiation + serialize + network, all on prefill's side).
            METRICS.di_reply_rtt(
                (postprocess_start - pending.ce_done_ping_at) * 1000
            )

        context = pending.context
        is_constrained = (
            context.json_schema is not None or context.grammar is not None
        )
        if is_constrained and self.batch_constructor.structured_output_enabled:
            # Prefill ran this request fully unconstrained, so its sampled
            # token cannot be trusted to satisfy the grammar and must be
            # discarded (draft tokens too -- decode's spec-decode state
            # cold-starts like any other fresh context). Treat the handoff
            # as a chunked-CE continuation with one token still to process:
            # decode re-forwards it through the ordinary batch pipeline to
            # sample token 1 itself, under its own matcher. No output is
            # sent here; the real first token arrives later via the normal
            # `schedule()` response path once that CE step executes -- see
            # `_handoff_landed_at` there for the TTFT contribution this
            # wait adds.
            context.tokens.skip_processing(context.tokens.active_length - 1)
            self._handoff_landed_at[request_id] = time.monotonic()
        else:
            # Update the context with the generated token
            context.update(message.generated_token_id)

            # Restore draft tokens from Eagle/MTP prefill so the first
            # decode iteration can verify them without re-running draft
            # prefill. When speculative decoding is active, the prefill
            # worker always sends draft tokens.
            if (
                self.scheduler_config.num_speculative_tokens > 0
                and not context.is_done
            ):
                # Done contexts (max_gen_tokens=1) need no further TG steps,
                # so the prefill pod sends draft_tokens=None. For all other
                # contexts, draft tokens must arrive with the PrefillResponse.
                if message.draft_tokens is None:
                    raise ValueError(
                        f"Expected draft tokens in PrefillResponse for "
                        f"request {request_id} with speculative decoding "
                        f"enabled, but none were received."
                    )
                context.spec_decoding_state.draft_tokens_to_verify = (
                    message.draft_tokens
                )

            # Send singular token to the API process
            output = context.to_generation_output()
            self.response_queue.put_nowait(
                {request_id: SchedulerResult.create(output)}
            )
            # Decode-local postprocessing before the result leaves this
            # process. Does not cover the subsequent API-process hop -- see
            # maxserve.response_queue_time for the tail end of that on the
            # API side.
            METRICS.di_decode_postprocess_time(
                (time.monotonic() - postprocess_start) * 1000
            )

        pending.phase = DecodeRequestPhase.TRANSFERRING
        pending.phase_entered_at = time.monotonic()
        pending.transfer = message.transfer_metadata

    def handle_prefill_progress_ping(
        self, message: PrefillProgressPing
    ) -> None:
        """Handles a ping-back from prefill, timestamped on decode's own
        clock so every derived interval is a same-clock diff.
        """
        pending = self.requests.get(message.id)
        if pending is None:
            # Cancelled or already resolved while the ping was in-flight.
            return

        now = time.monotonic()
        if message.event == "arrived":
            METRICS.di_dispatch_rtt((now - pending.phase_entered_at) * 1000)
            pending.arrived_ping_at = now
        elif message.event == "ce_done":
            if pending.arrived_ping_at is not None:
                METRICS.di_prefill_span((now - pending.arrived_ping_at) * 1000)
            pending.ce_done_ping_at = now

    @traced
    def send_prefill_request(
        self,
        request_id: RequestID,
        data: TextContext,
        dst_idxs: list[int],
        dst_replica_idx: int,
    ) -> None:
        """Pushes a request to the prefill socket.

        Args:
            request_id: The ID of the request to send
            data: The context containing the request data
            dst_idxs: The destination block indices for the request
            replica_idx: The replica the request is on for Decode

        Raises:
            zmq.ZMQError: If there is an error sending on the socket
        """
        # TODO: Do not crash the scheduler if a request does not have a target endpoint.
        #       Instead we should validate this in the frontend.
        if data.target_endpoint is None:
            raise ValueError(
                f"Target endpoint is not specified for the request {request_id}"
            )
        if data.target_endpoint not in self.remote_endpoints:
            self.dispatcher.send_request_nowait(
                self.transfer_engine.metadata,
                data.target_endpoint,
            )
            self.remote_endpoints.add(data.target_endpoint)

        assert data.tokens.generated_length == 0, (
            f"Invalid Context: Expected needs_ce to be True. Found: {data}"
        )

        # Set dst_idx to -1 to denote pages which the decode already has due
        # to prefix caching. processed_length is in tokens; divide by
        # page_size to convert to blocks. dst_idxs is already per-replica
        # (get_req_blocks returns blocks on the replica the request was
        # claimed on), so no further scaling by data-parallel degree.
        for i in range(
            data.tokens.processed_length // self.kv_cache.params.page_size
        ):
            dst_idxs[i] = -1

        # Struct construction + msgspec serialization + ZMQ enqueue,
        # decode-local -- no network transit included.
        send_start = time.monotonic()
        self.dispatcher.send_request_nowait(
            PrefillRequest(
                id=request_id,
                context=data,
                transfer_engine_name=self.transfer_engine.name,
                dst_block_ids=dst_idxs,
                dst_replica_idx=dst_replica_idx,
            ),
            data.target_endpoint,
        )
        METRICS.di_decode_send_time((time.monotonic() - send_start) * 1000)

    def _send_admitted_request_to_prefill(
        self,
        req_id: RequestID,
        context: TextContext,
        replica_idx: int,
        onload_event: KVConnectorTransfer,
    ) -> None:
        """Sends a claimed request to the prefill node.

        Does not wait for ``onload_event`` (the admission-time KV onload)
        to complete first -- prefill only ever writes blocks beyond the
        request's already-cached prefix, so its own KV transfer and our
        onload never target the same memory. ``onload_event`` still has
        to complete before this request can join a TG batch or release
        its blocks; see ``check_for_completed_transfers``.
        """
        dst_idxs = self.kv_cache.get_req_blocks(context)
        self.requests[req_id] = PendingDecodeRequest(
            context=context,
            replica_idx=replica_idx,
            phase=DecodeRequestPhase.AWAITING_PREFILL,
            onload_event=onload_event,
            phase_entered_at=time.monotonic(),
        )
        self.send_prefill_request(req_id, context, dst_idxs, replica_idx)

    def reserve_memory_and_send_to_prefill(self) -> None:
        """Continuously pulls requests from the request queue and forwards them to the prefill node."""
        # max_batch_size is a per-replica limit (see TextBatchConstructor's
        # docstring), but the in-flight counts below sum every replica --
        # scale by DP degree so admission isn't capped at one replica's share.
        total_max_batch_size = (
            self.scheduler_config.max_batch_size
            * self.scheduler_config.data_parallel_degree
        )
        items = drain_queue(
            self.request_queue,
            max_items=total_max_batch_size * 2,
        )

        for context in items:
            self.pending_reqs[context.request_id] = context
            self._admission_enqueue_time.setdefault(
                context.request_id, time.monotonic()
            )

        while (
            self.pending_reqs
            and (len(self.batch_constructor.all_tg_reqs) + len(self.requests))
            < total_max_batch_size
            and (
                self.kv_cache is None
                or any(
                    self.kv_cache.block_count(replica_idx).used_pct < 90
                    for replica_idx in range(
                        self.scheduler_config.data_parallel_degree
                    )
                )
            )
        ):
            # Pop off request queue
            context = next(iter(self.pending_reqs.values()))
            req_id = context.request_id
            del self.pending_reqs[req_id]

            # Claim the slot with the paged manager
            replica_idx = self.batch_constructor.get_next_replica_idx(
                external_requests_per_replica=self.prefill_reqs_per_replica
            )
            self.kv_cache.claim(context, replica_idx=replica_idx)

            # Allocate enough memory needed to run the request for one step.
            # The blocks allocated here will be written via a KVCache transfer
            # from prefill -> decode.  When speculative decoding is active,
            # the prefill node generates extra KV entries for draft tokens,
            # so we must allocate matching blocks on the decode side.
            try:
                load_event = self.kv_cache.alloc(context)
            except InsufficientBlocksError:
                # If we don't have enough space, we will return this to the request queue.
                self.pending_reqs[req_id] = context
                self.pending_reqs.move_to_end(req_id, last=False)
                self.kv_cache.release(context)
                break

            admitted_at = time.monotonic()
            enqueued_at = self._admission_enqueue_time.pop(req_id, None)
            if enqueued_at is not None:
                METRICS.di_decode_admission_queue_wait_time(
                    (admitted_at - enqueued_at) * 1000
                )
            self.prefill_reqs_per_replica[replica_idx] += 1
            if self.batch_constructor.structured_output_enabled:
                # Start this request's grammar build now, the instant it's
                # admitted, so it overlaps with prefill's forward + KV
                # transfer instead of stacking sequentially onto TTFT after
                # the handoff lands.
                self.batch_constructor.submit_grammar_build(context)
            self._send_admitted_request_to_prefill(
                req_id, context, replica_idx, load_event
            )

    def _handle_cancelled_requests(self) -> None:
        for req_id in get_cancelled_reqs(self.cancel_queue):
            if self.batch_constructor.contains(req_id):
                # Remove it from the active batch.
                self.batch_constructor.release_request(req_id)
                # A handoff may have landed and been enqueued here before
                # its CE step ever ran; nothing will ever read it back.
                self._handoff_landed_at.pop(req_id, None)
                # Send the cancelled result back to the response q
                self.response_queue.put_nowait(
                    {req_id: SchedulerResult.cancelled()}
                )
                continue

            pending = self.requests.get(req_id)
            if pending is None:
                logger.debug(
                    f"cancel request received on decode node for {req_id} not in pending or active batch."
                )
                continue

            data = pending.context
            dst_replica_idx = pending.replica_idx

            if (
                pending.phase is DecodeRequestPhase.TRANSFERRING
                or not pending.onload_event.is_complete()
            ):
                # A PrefillResponse's KV transfer may be running, or --
                # since this request was sent to prefill without waiting
                # for its own admission-time onload -- that onload copy
                # may still be writing into these blocks. Defer the
                # release to check_for_completed_transfers, which waits
                # for whichever of the two is still outstanding.
                pending.cancelled = True
            else:
                # Onload landed and no transfer is in flight yet, so
                # nothing is writing to these blocks -- safe to release
                # immediately.
                del self.requests[req_id]
                self.prefill_reqs_per_replica[dst_replica_idx] -= 1
                # A grammar build may already be running for this request
                # (started at admission, before it ever reaches prefill);
                # nothing will ever read it back. A handoff can't have
                # landed yet -- that always leaves phase TRANSFERRING,
                # which routes here to the deferred branch above instead --
                # but pop defensively rather than lean on that invariant.
                self.batch_constructor.release_grammar_build(req_id)
                self._handoff_landed_at.pop(req_id, None)
                self.kv_cache.release(data)

            # TODO: Do not crash the scheduler if a request does not have a target endpoint.
            #       Instead we should validate this in the frontend.
            if data.target_endpoint is None:
                raise ValueError(
                    f"Target endpoint is not specified for the request {req_id}."
                )
            # Send a cancel request to the prefill node
            self.dispatcher.send_request_nowait(
                CancelRequest(id=req_id), data.target_endpoint
            )
            # Send the cancelled result back to the response q
            self.response_queue.put_nowait(
                {req_id: SchedulerResult.cancelled()}
            )

    def _evict_expired_requests(self) -> None:
        """Evict requests stuck in their current phase past
        ``decode_request_ttl_s``, so the stall watchdog does not have to
        kill the engine: ``AWAITING_PREFILL`` past TTL means
        ``PrefillResponse`` never arrived; ``TRANSFERRING`` past TTL means
        the NIXL transfer never completed.

        Each evicted request releases its KV cache blocks and decrements
        ``prefill_reqs_per_replica``. Unless already cancelled (where a
        cancelled result and prefill cancel already happened), it also
        surfaces a cancelled result and cancels prefill.
        """
        ttl_s = self.scheduler_config.decode_request_ttl_s
        if ttl_s is None:
            return

        now = time.monotonic()
        cutoff = now - ttl_s

        # phase_entered_at resets on every phase transition (see
        # PendingDecodeRequest), so this single sweep always measures time
        # in the current phase -- no separate pass needed to avoid evicting
        # a TRANSFERRING request on its stale pre-transfer timestamp.
        expired = [
            req_id
            for req_id, pending in self.requests.items()
            if pending.phase_entered_at < cutoff
        ]
        for req_id in expired:
            pending = self.requests.pop(req_id)
            self.prefill_reqs_per_replica[pending.replica_idx] -= 1
            # A handoff may have landed and started this wait before the
            # request got stuck; nothing will ever read it back. A grammar
            # build may also still be running from admission time.
            self._handoff_landed_at.pop(req_id, None)
            self.batch_constructor.release_grammar_build(req_id)

            if pending.phase is DecodeRequestPhase.TRANSFERRING:
                assert pending.transfer is not None
                try:
                    self.transfer_engine.cleanup_transfer(pending.transfer)
                except ValueError:
                    logger.warning(
                        "cleanup_transfer failed for evicted request %s",
                        req_id,
                        exc_info=True,
                    )

            self.kv_cache.release(pending.context)

            # Skip if already cancelled: the client was already told
            # "cancelled" and prefill already got a cancel when the
            # cancellation was first processed -- only the release itself
            # is left to do.
            if not pending.cancelled:
                self._send_cancel_to_prefill(req_id, pending.context)
                self.response_queue.put_nowait(
                    {req_id: SchedulerResult.cancelled()}
                )

            logger.warning(
                "Evicting stuck %s request %s after %.1fs (TTL=%.1fs)",
                pending.phase.name.lower(),
                req_id,
                now - pending.phase_entered_at,
                ttl_s,
            )

    def _send_cancel_to_prefill(
        self, req_id: RequestID, context: TextContext
    ) -> None:
        """Best-effort cancel to prefill so a late ``PrefillResponse`` does
        not arrive against released decode-side memory."""
        if context.target_endpoint is None:
            logger.warning(
                "Evicted request %s has no target_endpoint; skipping"
                " cancel to prefill",
                req_id,
            )
            return
        self.dispatcher.send_request_nowait(
            CancelRequest(id=req_id), context.target_endpoint
        )

    def check_for_completed_transfers(self) -> None:
        """Settles requests once every write still targeting their blocks
        has landed: the admission-time onload always, and -- once a
        ``PrefillResponse`` arrives -- prefill's own KV transfer too.
        Ready requests join a TG batch, or, if cancelled, release their
        blocks.

        Must run after ``_handle_cancelled_requests`` within the same
        iteration, so a same-tick cancellation is already flagged on its
        record before this checks it.
        """
        ready_ids = []
        for req_id, pending in self.requests.items():
            if not pending.onload_event.is_complete():
                continue
            if pending.phase is DecodeRequestPhase.TRANSFERRING:
                assert pending.transfer is not None
                if not self.transfer_engine.is_complete(pending.transfer):
                    continue
            elif not pending.cancelled:
                # AWAITING_PREFILL and not cancelled: still waiting on a
                # PrefillResponse, unrelated to the onload above.
                continue
            ready_ids.append(req_id)

        for request_id in ready_ids:
            pending = self.requests.pop(request_id)
            self.prefill_reqs_per_replica[pending.replica_idx] -= 1

            if pending.phase is DecodeRequestPhase.TRANSFERRING:
                assert pending.transfer is not None
                self.transfer_engine.cleanup_transfer(pending.transfer)

            if pending.cancelled:
                self.kv_cache.release(pending.context)
                # A handoff may have landed and started this wait before
                # the cancellation raced in; nothing will ever read it back.
                # A grammar build may still be running too, from admission
                # time.
                self._handoff_landed_at.pop(request_id, None)
                self.batch_constructor.release_grammar_build(request_id)
                continue

            self.batch_constructor.enqueue_new_request(
                pending.context, pending.replica_idx
            )

    def _inflight_kv_transfer_count(self, replica_idx: int) -> int:
        """Count of in-flight decode requests on this replica, regardless
        of phase. Scoped to replica_idx: a reservation on a different
        replica's device pool can't free blocks on this one."""
        return sum(
            1
            for pending in self.requests.values()
            if pending.replica_idx == replica_idx
        )

    @traced
    def schedule(self, inputs: TextGenerationInputs[TextContext]) -> int:
        """Schedules a batch of requests for token generation and handles the responses.

        Args:
            inputs: The inputs containing the batch of requests to schedule.
        """
        assert len(inputs.batches) > 0
        responses = self.pipeline.execute(inputs)

        # Filter out responses for already-released requests. With the
        # overlap pipeline, the previous batch may produce a token for a
        # request that already hit EOS and was released.
        responses = {
            req_id: response
            for req_id, response in responses.items()
            if self.batch_constructor.contains(req_id)
        }

        self.batch_constructor.advance_requests(inputs)

        # Release terminated requests
        num_terminated_requests = 0
        for request_id, response in responses.items():
            if response.is_done:
                self.batch_constructor.release_request(request_id)
                num_terminated_requests += 1

        # Send the responses to the API process
        if responses:
            if self._handoff_landed_at:
                # Any of these responses could be a handoff's first real
                # token -- the one case with no equivalent wait in the
                # ordinary path, whose token already arrives with the
                # PrefillResponse. Pop rather than peek: this fires at most
                # once per request, the first time its output is sent.
                now = time.monotonic()
                for req_id in responses:
                    landed_at = self._handoff_landed_at.pop(req_id, None)
                    if landed_at is not None:
                        METRICS.di_handoff_to_first_token_time(
                            (now - landed_at) * 1000
                        )
            self.response_queue.put_nowait(
                {
                    req_id: SchedulerResult.create(response)
                    for req_id, response in responses.items()
                }
            )

        return num_terminated_requests

    @traced
    def run_iteration(self) -> SchedulerProgress:
        """Main scheduling loop that processes decode requests.

        Receives requests, updates batches, and schedules them for processing
        while handling memory management.

        Returns:
            SchedulerProgress: Indicates whether work was performed in this iteration.
        """
        while True:
            try:
                reply = self.dispatcher.recv_reply_nowait()
            except queue.Empty:
                break
            if isinstance(reply, KVTransferEngineMetadata):
                self.handle_transfer_engine_response(reply)
            elif isinstance(reply, PrefillResponse):
                self.handle_prefill_response(reply)
            elif isinstance(reply, PrefillProgressPing):
                self.handle_prefill_progress_ping(reply)
            elif isinstance(reply, PrefillFailure):
                self.handle_prefill_failure(reply)
            else:
                raise ValueError(f"Invalid reply type: {reply}")

        self._evict_expired_requests()

        # Eagerly reserve memory and send to prefill worker
        self.reserve_memory_and_send_to_prefill()

        # Process cancellations before completions: a cancellation and its
        # transfer's completion can land in the same tick, and a same-tick
        # cancellation must be flagged before check_for_completed_transfers
        # checks it.
        self._handle_cancelled_requests()

        # Update the active decode batch
        self.check_for_completed_transfers()

        # Construct the batch to execute
        t0 = time.monotonic()
        inputs = self.batch_constructor.construct_batch()
        t1 = time.monotonic()
        batch_creation_time_s = t1 - t0

        # A request whose admission-time grammar build failed was never
        # admitted and is already released; forward the failure now so the
        # client gets a response instead of hanging, same as the aggregated
        # scheduler.
        for failed_id, error in self.batch_constructor.take_grammar_failed():
            # A handoff may have landed and started this wait before its
            # grammar build failed; nothing will ever read it back.
            self._handoff_landed_at.pop(failed_id, None)
            self.response_queue.put_nowait(
                {failed_id: SchedulerResult.failed(error)}
            )

        total_pending = len(self.pending_reqs) + len(self.requests)
        if inputs or total_pending == 0:
            self._last_batch_activity = time.monotonic()
        elif self.scheduler_config.decode_stall_timeout_s is not None:
            stall_duration = time.monotonic() - self._last_batch_activity
            if stall_duration > self.scheduler_config.decode_stall_timeout_s:
                awaiting_prefill = sum(
                    1
                    for p in self.requests.values()
                    if p.phase is DecodeRequestPhase.AWAITING_PREFILL
                )
                transferring = len(self.requests) - awaiting_prefill
                logger.error(
                    "Decode stall detected: no batch activity for %.1fs"
                    " with %d pending requests (%d queued, %d awaiting"
                    " prefill, %d transferring). Terminating worker to"
                    " trigger restart.",
                    stall_duration,
                    total_pending,
                    len(self.pending_reqs),
                    awaiting_prefill,
                    transferring,
                )
                # SystemExit bypasses except Exception handlers in the
                # scheduler loop, guaranteeing the process exits and
                # triggers a pod restart. A regular exception risks being
                # caught and swallowed.
                raise SystemExit(1)

        # Check whether the overlap pipeline has deferred outputs that must
        # be drained even when the current batch is empty.
        has_pending_outputs = (
            hasattr(self.pipeline, "has_pending_outputs")
            and self.pipeline.has_pending_outputs()
        )
        if not (inputs or has_pending_outputs):
            return SchedulerProgress.NO_PROGRESS

        # Schedule the batch
        t0 = time.monotonic()
        if inputs:
            with Tracer(f"_schedule({inputs})"):
                num_terminated_reqs = self.schedule(inputs)
        else:
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
            num_pending_reqs=total_pending,
            num_terminated_reqs=num_terminated_reqs,
            total_preemption_count=self.batch_constructor.total_preemption_count,
            batch_spec_decode_metrics=self.pipeline.batch_spec_decode_metrics()
            if hasattr(self.pipeline, "batch_spec_decode_metrics")
            else None,
            overlap_active=is_overlap_active,
            completed_batch_stats=self.pipeline.take_completed_batch_stats()
            if hasattr(self.pipeline, "take_completed_batch_stats")
            else None,
        )

        return SchedulerProgress.MADE_PROGRESS


def load_decode_scheduler(
    pipeline: TextGenerationPipeline[TextContext],
    pipeline_config: PipelineConfig,
    request_queue: MAXPullQueue[TextContext | TextAndVisionContext],
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ],
    cancel_queue: MAXPullQueue[list[RequestID]],
    settings: Settings,
    memory_plan: MemoryPlan | None,
) -> DecodeScheduler:
    # Create Scheduler Config.
    scheduler_config = TokenGenerationSchedulerConfig.from_pipeline_config(
        pipeline_config, pipeline.max_batch_size, memory_plan
    )

    # Build DP batch padder when DP > 1 with device graph capture.
    dp_padder: DPBatchPadder | None = None
    if (
        scheduler_config.data_parallel_degree > 1
        and pipeline_config.runtime.device_graph_capture
    ):
        # Padding dummies must match the architecture's concrete context
        # type — for VLMs the overlap pipeline narrows every batched context
        # to TextAndVisionContext, so plain-TextContext dummies would crash
        # the first padded batch.
        context_type = PIPELINE_REGISTRY.retrieve_context_type(pipeline_config)
        assert issubclass(context_type, TextContext)
        assert (
            memory_plan is not None
            and memory_plan.planned_max_length is not None
        ), "DP batch padding requires a memory plan with a max length"
        dp_padder = DPBatchPadder(
            dp_size=scheduler_config.data_parallel_degree,
            kv_manager=pipeline.kv_manager,
            max_length=memory_plan.planned_max_length,
            model_name=pipeline_config.model.model_name,
            pipeline=pipeline,
            context_type=context_type,
        )

    return DecodeScheduler(
        pipeline=pipeline,
        scheduler_config=scheduler_config,
        kv_cache=pipeline.kv_manager,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        dispatcher=DecodeDispatcherClient(bind_addr=settings.di_bind_address),
        dp_padder=dp_padder,
    )
