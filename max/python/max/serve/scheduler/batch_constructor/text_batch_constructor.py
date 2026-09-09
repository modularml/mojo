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
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from max.pipelines.context import TextGenerationOutput
from max.pipelines.context.context import TextContext
from max.pipelines.kv_cache import (
    InsufficientBlocksError,
    PagedKVCacheManagerInterface,
)
from max.pipelines.kv_cache.kv_connector import KVConnectorTransfer
from max.pipelines.lora import LoRAManagerV3, get_lora_manager
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.profiler import traced
from max.serve.telemetry.metrics import METRICS

from ..config import TokenGenerationSchedulerConfig
from ..dp_padding import DPBatchPadder, DPPaddingInfo
from ..lora_scheduler_utils import (
    can_allocate_lora_request,
    is_active_lora,
    is_lora,
)
from .grammar_gate import AsyncGrammarGate
from .token_budget import (
    ActiveTokenBudget,
    BudgetStatus,
    RequestType,
    TokenBudget,
    TokenBudgetCollection,
    TotalContextTokenBudget,
)

logger = logging.getLogger("max.serve")

TG_PRIORITY_KV_PERCENTAGE_ENV_VAR = "MAX_SERVE_TG_PRIORITY_KV_PERCENTAGE"
"""Device KV usage percentage above which a replica runs TG before CE.

Set it above 100 to disable the rule.
"""


@dataclass
class _PendingCERequest:
    """A CE request awaiting DP-balanced placement, not yet bound to a replica.

    ``weights`` holds the request's estimated post-prefix-cache CE length in
    tokens, per replica, from a read-only probe of the device and shared
    (host/disk) cache tiers at enqueue time. The shared tiers give the same
    answer on every replica; per-replica differences come from device-cache
    residency only.
    """

    ctx: TextContext
    weights: list[int]


@dataclass
class _BoundRequest:
    """A request bound to a replica, which owns it until it is released.

    ``ctx`` is retained because a request in an active batch has been popped
    out of every replica queue, yet that is exactly when the scheduler filters
    responses and drains cancellations -- both of which name the request by id
    alone and need its context to release it.
    """

    ctx: TextContext
    replica_idx: int


@dataclass
class _GrammarPendingRequest:
    """A fresh CE request held back while its grammar matcher builds off-thread.

    ``replica_idx`` preserves a caller-pinned replica for promotion.
    """

    ctx: TextContext
    replica_idx: int | None


@dataclass
class _OnloadingRequest:
    """A CE request cordoned out of the batch while its KV onload is in flight.

    The request is already bound to ``replica_idx`` and fully allocated (its
    blocks are pinned by ``event``); it is held here, out of any run queue,
    until ``event`` polls complete, then re-admitted to the replica's CE queue.
    """

    ctx: TextContext
    replica_idx: int
    event: KVConnectorTransfer


@dataclass
class ReplicaRequests:
    """This class tracks the requests assigned to each replica.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    ce_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )
    tg_reqs: OrderedDict[RequestID, TextContext] = field(
        default_factory=OrderedDict
    )

    # LoRA-related bookkeeping for this replica.
    active_loras: set[str] = field(default_factory=set)
    deferred_lora_requests: dict[RequestID, TextContext] = field(
        default_factory=dict
    )

    def add_active_lora(
        self, context: TextContext, lora_manager: LoRAManagerV3 | None
    ) -> None:
        """Mark a LoRA as active for this replica and refresh its LRU in the manager."""
        if lora_manager is None:
            return

        if is_lora(context, lora_manager):
            lora_manager.activate_adapter(context.model_name)
            self.active_loras.add(context.model_name)

    def update_deferred_lora_requests(self) -> None:
        """Return deferred LoRA requests back to the CE queue in FIFO order."""
        for req_id, ctx in self.deferred_lora_requests.items():
            self.ce_reqs[req_id] = ctx
            self.ce_reqs.move_to_end(req_id, last=False)
        self.deferred_lora_requests.clear()

    def can_allocate_lora_request(
        self, context: TextContext, lora_manager: LoRAManagerV3 | None
    ) -> bool:
        """Check LoRA budget and defer the request if it cannot be safely activated."""
        if lora_manager is None:
            return True

        if not can_allocate_lora_request(
            context, self.active_loras, lora_manager
        ):
            self.deferred_lora_requests[context.request_id] = context
            return False

        return True


@dataclass
class ReplicaBatch:
    """This class represents a batch of requests for a single replica.

    This class is an implementation detail of TextBatchConstructor and should not be
    used outside of this file.
    """

    batch: dict[RequestID, TextContext]
    token_budget: TokenBudgetCollection

    def __len__(self) -> int:
        return len(self.batch)

    def is_empty(self) -> bool:
        return len(self.batch) == 0


class PreemptionReason(str, Enum):
    KV_CACHE_MEMORY = "kv_cache_memory"
    MAX_NUM_LORAS = "max_num_loras"

    @property
    def error_message(self) -> str:
        match self:
            case PreemptionReason.MAX_NUM_LORAS:
                return "Preempted a request due to max-num-loras limit exceeded. This can affect the end-to-end performance. Consider increasing max-num-loras."

            case PreemptionReason.KV_CACHE_MEMORY:
                return "Preempted a request due to lack of KV pages. This can affect the end-to-end performance. Consider increasing device-memory-utilization via `--device-memory-utilization` to provide more KV cache memory."


class BatchSchedulingStrategy(str, Enum):
    """Strategy for prioritizing CE (prefill) vs TG (decode) batch construction.

    This enum controls how replicas prioritize between context encoding (CE/prefill)
    and token generation (TG/decode) requests when constructing batches. The strategy
    can either enforce a global priority across all replicas or allow each replica to
    independently determine priority based on its local queue state.

    The default behavior (None/unset) corresponds to PER_REPLICA mode, where each
    replica independently decides priority based on its queue state and the
    enable_in_flight_batching configuration setting.
    """

    PREFILL_FIRST = "prefill_first"
    """Always prioritize CE (context encoding/prefill) requests.

    When this strategy is set, all replicas will prioritize building CE batches
    before processing any TG requests. This maximizes prompt throughput and
    minimizes time-to-first-token at the cost of potentially higher inter-token
    latency for ongoing generations.
    """

    DECODE_FIRST = "decode_first"
    """Always prioritize TG (token generation/decode) requests.

    When this strategy is set, all replicas will prioritize building TG batches
    and only process CE requests when no TG work is available. This minimizes
    inter-token latency for active generations at the cost of potentially higher
    time-to-first-token for new requests.
    """

    BALANCED = "balanced"
    """Adaptively prioritize based on relative queue sizes across replicas.

    When this strategy is set, the scheduler considers the global state across
    all replicas to determine priority. If the majority of pending work is CE,
    prioritize CE; if the majority is TG, prioritize TG. This provides a middle
    ground between PREFILL_FIRST and DECODE_FIRST strategies.
    """

    PER_REPLICA = "per_replica"
    """Each replica independently manages its own batching priority (default).

    This is the default behavior when no strategy is explicitly set. Each replica
    determines its own priority based on its local queue state and the
    enable_in_flight_batching configuration:

    - If enable_in_flight_batching=False: prioritize CE when CE queue is non-empty
    - If enable_in_flight_batching=True: prioritize TG when TG queue is non-empty

    This mode provides maximum flexibility and allows replicas to adapt to their
    individual workload characteristics, which is particularly useful in
    load-balanced deployments where request patterns may vary across replicas.
    """


class TextBatchConstructor:
    """Construct per-replica text batches from CE (prefill) and TG queues.

    This class encapsulates the high-level policy for forming execution
    batches for text pipelines. It operates on two logical phases of each
    request:

    - **Context encoding (CE / prefill)**: processing prompt tokens.
    - **Token generation (TG / decode)**: generating continuation tokens for
      already-prefilled requests.

    The batching policy is expressed entirely in terms of scheduler
    configuration (:class:`TokenGenerationSchedulerConfig`), model context
    metadata (e.g., lengths and LoRA adapter names), and KV cache / resource
    limits. The sections below describe the *intended behaviour* rather than
    the exact implementation.

    .. note::
       When chunked prefill is enabled and a request is split across multiple
       CE iterations, no output is emitted for that request until its full
       prefill completes. The request remains in the CE queue across
       iterations until all prompt tokens have been processed.

    **Replica assignment**

    The constructor supports data-parallel execution across
    ``data_parallel_degree`` replicas:

    - Each replica maintains its own CE and TG queues and forms batches
      independently using the same policy.
    - New requests are assigned to replicas using load-based assignment,
      which selects the replica with the fewest active requests. This
      provides better load balancing than round-robin, particularly when
      request sizes vary significantly or when requests complete at
      different rates.
    - The replica with the minimum request count is selected based on
      information from the paged KV cache manager. This accounts for
      both active processing and queued requests.
    - All replicas share the same logical KV memory budget through the
      paged KV cache manager.

    **Non-LoRA batch construction**

    The following describes batch construction when no LoRA manager is
    attached.

    **1. CE vs TG prioritization**

    For each replica the constructor maintains separate CE and TG queues.
    Priority depends on the ``enable_in_flight_batching`` setting:

    - When **False** (default):

      - If there are any CE requests waiting, the constructor prioritises
        building a CE batch until the CE queue is drained or budgets / limits
        are reached.
      - TG batches are formed only when there is no CE work pending on that
        replica. This favours fast admission of new prompts at the cost of
        slightly higher latency for ongoing generations.

    - When **True**:

      - If there are TG requests, the constructor prioritises TG, building a
        TG batch first and then optionally filling any remaining capacity in
        that iteration with CE work.
      - CE-only batches are formed when there is no TG work on the replica.
      - This favours throughput and inter-token latency for active
        generations at the cost of time-to-first-token for new requests, while
        still opportunistically progressing prefill when there is headroom.

    **2. Token budgets**

    Two logical token budgets can limit batch size. A context is admitted to
    the batch only if *all* active budgets agree that it fits. Budgets may
    report that:

    - Additional contexts can be admitted.
    - The current context should be admitted but further admissions in this
      batch should stop.
    - The context cannot fit and should be deferred to a later batch.

    *Active-token budget*

    - Capacity: ``target_tokens_per_batch_ce``.
    - Interprets the current active window of each context and enforces a soft
      limit on how many prompt tokens are processed in one CE batch.
    - When ``enable_chunked_prefill`` is **True**, large prompts may be
      *chunked* so that only a prefix of the prompt is processed in the
      current CE batch. The remainder is scheduled for follow-up CE
      iterations. This allows the constructor to:

      - Fill CE batches to approximately ``target_tokens_per_batch_ce``
        tokens.
      - Admit very large prompts without blocking the batch behind a single
        oversized request.

    - When chunked prefill is **disabled**, prompts are either admitted in
      full or deferred to a later batch once the active-token budget has been
      reached.

    *Total-context budget (optional)*

    - Enabled when ``max_batch_total_tokens`` is not ``None``.
    - Tracks the total resident context across the batch, accounting for
      current context length and planned forward steps, and ensures the sum
      does not exceed ``max_batch_total_tokens``.
    - This budget is only applied when a CE request is present, or to be
      added to the batch.

    **3. Max batch size limits**

    Independently of token budgets, explicit batch-size caps ensure that no
    batch grows without bound:

    - ``max_batch_size`` limits the number of requests in a batch per replica.

    Once a batch reaches its respective maximum size, no further requests are
    considered for that iteration, even if token budgets or cache capacity
    would allow more.

    **4. Sequence length limits**

    ``max_seq_len`` is enforced at request completion: once a request's
    generated tokens reach ``max_seq_len``, it is marked complete and
    removed from the TG queue. All requests present in the TG queue at
    batch construction time therefore have remaining capacity.

    **5. KV cache memory pressure**

    When a paged KV cache is used, KV memory becomes an additional limiting
    factor:

    - Once a replica's device KV usage exceeds
      ``TG_PRIORITY_KV_PERCENTAGE_ENV_VAR`` percent, it prioritizes TG over CE
      so in-flight generations drain before new prefills compete for blocks.
    - During TG, if allocating KV blocks for a candidate request fails due to
      insufficient capacity, the constructor may preempt other TG candidates
      to free KV space, ensuring that at least a subset of requests can
      continue generating. When preemption is required, the constructor evicts
      the most recently enqueued TG request first, minimising wasted work. A
      warning is logged when this occurs.

    Overall, KV cache limits, token budgets, and max batch sizes jointly
    determine the final CE and TG batch composition for each iteration.

    **LoRA batch construction**

    When a :class:`LoRAManagerV3` is attached, the constructor applies
    additional constraints to respect the maximum number of concurrently
    loaded LoRA adapters and to avoid evicting adapters needed for active
    generations.

    - LoRA and data parallelism are mutually exclusive: if a LoRA manager is
      present, ``data_parallel_degree`` must be 1 so that all LoRA state is
      local to a single replica.
    - Each replica tracks a set of *protected* active LoRAs that are currently
      participating in the TG batch and must not be evicted.

    **1. CE batching with LoRA**

    For CE requests that target LoRA-adapted models:

    - A LoRA request is admitted into the CE batch only if activating its
      adapter would not exceed the LoRA manager's capacity, taking into
      account:

      - The protected LoRAs already in use by TG.
      - Any additional LoRAs the constructor intends to activate in this CE
        iteration.

    - If admitting a new LoRA request would require evicting a protected LoRA,
      that request is temporarily deferred and re-queued for a later CE batch,
      rather than forcing an eviction that would disrupt ongoing TG work.
    - When a LoRA request is finally admitted to CE, its adapter is activated
      and tracked as active for the replica, so that subsequent TG iterations
      can safely use it.

    This ensures that CE never "steals" LoRA capacity away from requests that
    are currently generating tokens.

    **2. TG batching with LoRA**

    For TG requests that depend on LoRA adapters:

    - The constructor requires that the corresponding LoRA adapter is
      currently active. If it is not, the request is *preempted* from the TG
      batch, reset, and returned to CE so that its LoRA weights can be safely
      reloaded in a future batch. A warning is logged when this occurs.
    - TG batches therefore only ever contain LoRA requests for which the
      correct adapters are already resident, guaranteeing that generations are
      computed with the intended LoRA parameters.

    """

    def __init__(
        self,
        scheduler_config: TokenGenerationSchedulerConfig,
        pipeline: Pipeline[
            TextGenerationInputs[TextContext], TextGenerationOutput
        ],
        kv_cache: PagedKVCacheManagerInterface,
        batch_scheduling_strategy: BatchSchedulingStrategy = BatchSchedulingStrategy.PER_REPLICA,
        dp_padder: DPBatchPadder | None = None,
        get_inflight_kv_transfer_count: Callable[[int], int] | None = None,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.pipeline = pipeline
        self.kv_cache = kv_cache
        self.batch_scheduling_strategy = batch_scheduling_strategy
        self._get_inflight_kv_transfer_count = get_inflight_kv_transfer_count
        self.tg_priority_kv_percentage = float(
            os.getenv(TG_PRIORITY_KV_PERCENTAGE_ENV_VAR, "90")
        )

        self._lora_manager: LoRAManagerV3 | None = get_lora_manager(pipeline)

        # Gated requests are held before DP pooling/binding, so the CE
        # planner only ever prices grammar-ready work.
        self._grammar_gate = AsyncGrammarGate.create(pipeline)
        self._grammar_pending: OrderedDict[
            RequestID, _GrammarPendingRequest
        ] = OrderedDict()
        self._grammar_failed: list[tuple[RequestID, str]] = []

        self.num_replicas = self.scheduler_config.data_parallel_degree
        if self._lora_manager and self.num_replicas > 1:
            raise ValueError("LoRA does not support data parallelism.")

        self.replicas: list[ReplicaRequests] = [
            ReplicaRequests() for _ in range(self.num_replicas)
        ]
        self._bound_requests: dict[RequestID, _BoundRequest] = {}
        self._request_id_to_lora_name: dict[RequestID, str | None] = {}

        # CE requests cordoned out of the batch while their KV onload is in
        # flight, keyed by request id. Re-admitted to their replica's CE queue
        # once the onload completes (see ``_readmit_completed_onloads``).
        self._onloading_reqs: dict[RequestID, _OnloadingRequest] = {}

        # DP-balanced CE deferral state. New CE requests wait in
        # ``_ce_pending`` unbound to any replica (insertion order == arrival
        # order); binding happens at the step the planner first schedules
        # them. ``_ce_arrival`` keeps each pooled request's arrival time even
        # after binding, so mid-prefill tails share the same deferral
        # deadline; entries are dropped at release. ``_ce_deferred_replicas``
        # is recomputed by ``_plan_ce_step`` every iteration and consumed by
        # ``_identify_priority`` / ``_add_ce_requests``.
        self._dp_ce_balance_enabled = (
            scheduler_config.dp_ce_balance_timeout_ms >= 0
            and self.num_replicas > 1
        )
        self._ce_pending: OrderedDict[RequestID, _PendingCERequest] = (
            OrderedDict()
        )
        self._ce_arrival: dict[RequestID, float] = {}
        self._ce_deferred_replicas: set[int] = set()
        # Per-replica CE token quota for this iteration, set by
        # ``_plan_ce_step`` when it reduces a below-threshold step's chunk
        # size to its balance level. ``None`` means no quota (full CE chunk
        # budget).
        self._ce_step_quota: list[int] | None = None
        self._probe_failure_logged = False

        self.total_preemption_count: int = 0
        self.last_preemption_logging_time: float = time.monotonic()

        self._dp_padder = dp_padder
        self._prev_dp_padding: DPPaddingInfo | None = None
        self._current_dp_padding: DPPaddingInfo | None = None

    def _create_new_token_budget(
        self, ce_capacity: int | None = None
    ) -> TokenBudgetCollection:
        token_budgets: list[TokenBudget] = [
            ActiveTokenBudget(
                capacity=(
                    ce_capacity
                    if ce_capacity is not None
                    else self.scheduler_config.target_tokens_per_batch_ce
                ),
                allow_chunking=self.scheduler_config.enable_chunked_prefill,
                applicable_types=RequestType.all(),
                min_chunk_tokens=self.scheduler_config.chunked_prefill_min_chunk_size,
            )
        ]

        if self.scheduler_config.max_batch_total_tokens is not None:
            token_budgets.append(
                TotalContextTokenBudget(
                    capacity=self.scheduler_config.max_batch_total_tokens,
                    allow_chunking=self.scheduler_config.enable_chunked_prefill,
                    applicable_types=[
                        RequestType.CE,
                        RequestType.MIXED,
                    ],
                    cost_alignment=self.kv_cache.params.page_size,
                )
            )

        return TokenBudgetCollection(
            token_budgets=token_budgets,
        )

    def get_next_replica_idx(
        self, external_requests_per_replica: list[int] | None = None
    ) -> int:
        """Returns the next replica index to assign the request to.

        Uses load-based assignment by selecting the replica with the fewest
        active requests. This provides better load balancing than round-robin,
        especially when request sizes vary or when requests complete at
        different rates.

        Args:
            external_requests_per_replica: The number of requests per replica
                that are not managed by the batch constructor.

        Returns:
            The replica index that should receive the next request.
        """
        if external_requests_per_replica is None:
            external_requests_per_replica = [0] * self.num_replicas

        replica_idx = min(
            range(self.num_replicas),
            key=lambda idx: (
                len(self.replicas[idx].ce_reqs)
                + len(self.replicas[idx].tg_reqs)
                + external_requests_per_replica[idx]
            ),
        )
        return replica_idx

    @property
    def structured_output_enabled(self) -> bool:
        """Whether constrained decoding can fire at all for this process:
        ``--enable-structured-output`` (user-supplied JSON schemas) or
        ``--enable-tool-call-constrained-decode`` with a grammar-capable
        tool parser (server-generated tool-call grammars).

        Mirrors ``PipelineConfig.needs_bitmask_constraints`` -- the same
        signal that gates whether the bitmask-aware sampler graph and
        pinned D2H buffer were even compiled/allocated for this process.
        """
        pipeline_config = getattr(self.pipeline, "pipeline_config", None)
        return (
            pipeline_config is not None
            and pipeline_config.needs_bitmask_constraints
        )

    def submit_grammar_build(self, ctx: TextContext) -> None:
        """Starts a request's off-thread grammar-matcher build, if needed.

        Exposed so a caller admitting a request before it ever reaches
        ``enqueue_new_request`` (DI's decode scheduler, at initial
        admission) can overlap the build with prefill's round trip instead
        of stacking it sequentially onto TTFT. ``AsyncGrammarGate.submit``
        is idempotent, so ``enqueue_new_request``'s own submit call later is
        a no-op re-check, not a duplicate compile.
        """
        if self._grammar_gate is not None:
            self._grammar_gate.submit(ctx)

    def release_grammar_build(self, request_id: RequestID) -> None:
        """Drops a request's outstanding grammar build, if any.

        Exposed for callers that remove a request before it ever reaches
        ``enqueue_new_request``/``release_request`` (DI's decode scheduler,
        on cancellation, TTL eviction, or prefill rejection) but may have
        already started its build via ``submit_grammar_build``. Safe to call
        unconditionally: a no-op when nothing was submitted.
        """
        if self._grammar_gate is not None:
            self._grammar_gate.release(request_id)

    def enqueue_new_request(
        self, ctx: TextContext, replica_idx: int | None = None
    ) -> None:
        """Add a new CE request to a replica.

        Args:
            ctx: The request to enqueue.
            replica_idx: The replica index to assign the request to.
                If None, the next replica index will be automatically chosen.
        """
        # Decode-side requests (generated_length != 0) built their matcher at
        # prefill admission.
        if self._grammar_gate is not None and ctx.tokens.generated_length == 0:
            self._grammar_gate.submit(ctx)
            if not self._grammar_gate.is_ready(ctx):
                self._grammar_pending[ctx.request_id] = _GrammarPendingRequest(
                    ctx=ctx, replica_idx=replica_idx
                )
                return
            error = self._grammar_gate.install_ready(ctx)
            if error is not None:
                self._fail_grammar_request(ctx, error)
                return

        self._admit_request(ctx, replica_idx)

    def _admit_request(self, ctx: TextContext, replica_idx: int | None) -> None:
        """Admits a grammar-ready request into the DP pool or a replica queue."""
        # DP-balanced CE deferral: fresh CE requests enter an unbound pool and
        # are bound to a replica by the per-step planner, which prices them by
        # estimated post-prefix-cache length. Caller-pinned requests and
        # already-generating requests bypass the pool.
        if (
            self._dp_ce_balance_enabled
            and replica_idx is None
            and ctx.tokens.generated_length == 0
        ):
            self._ce_pending[ctx.request_id] = _PendingCERequest(
                ctx=ctx, weights=self._post_cache_weights(ctx)
            )
            self._ce_arrival[ctx.request_id] = time.monotonic()
            return

        # Pick the replica to enqueue the request to.
        if replica_idx is None:
            replica_idx = self.get_next_replica_idx()
        self._bind_request(ctx, replica_idx)

    def _bind_request(self, ctx: TextContext, replica_idx: int) -> None:
        """Binds a request to a replica and enqueues it in the right queue."""
        replica = self.replicas[replica_idx]
        self._bound_requests[ctx.request_id] = _BoundRequest(ctx, replica_idx)
        self._request_id_to_lora_name[ctx.request_id] = (
            ctx.model_name
            if self._lora_manager and is_lora(ctx, self._lora_manager)
            else None
        )

        # Add the request to the appropriate dict based on whether it needs CE.
        if ctx.tokens.generated_length == 0:
            replica.ce_reqs[ctx.request_id] = ctx
        else:
            replica.tg_reqs[ctx.request_id] = ctx

    def _post_cache_weights(self, ctx: TextContext) -> list[int]:
        """Estimates per-replica post-prefix-cache CE length for a new request.

        Probes the prefix caches read-only (device tier per replica, shared
        host/disk tiers once) and subtracts the contiguous cached prefix from
        the request's unprocessed length. Fails open: on any probe error the
        request is weighted at its full unprocessed length everywhere
        (pre-cache behavior), and the first failure is logged.
        """
        active_length = ctx.tokens.active_length
        try:
            hits = self.kv_cache.get_prefix_cache_hit_counts(ctx)
        except Exception:
            if not self._probe_failure_logged:
                self._probe_failure_logged = True
                logger.exception(
                    "Prefix-cache probe for DP CE balancing failed; weighting"
                    " by full prompt length (logged once per process)."
                )
            return [active_length] * self.num_replicas

        assert len(hits) == self.num_replicas, (
            f"expected hit counts for {self.num_replicas} replicas,"
            f" got {len(hits)}"
        )
        page_size = self.kv_cache.params.page_size
        return [
            max(1, active_length - page_size * h.total_blocks) for h in hits
        ]

    def advance_requests(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> None:
        """Advances request state based on executed CE batches.

        This method updates per-replica queues by moving executed context encoding (CE)
        requests into the text generation (TG) queues. If the last request in a batch
        is chunked and still requires additional CE work, it is moved back to the CE
        queue for that replica.

        As a side effect, releases DP padding dummies from the previous
        batch.

        Args:
            inputs: the inputs for the batch.
        """
        self._release_data_parallel_padding()

        for per_replica_batch, replica in zip(
            inputs.batches, self.replicas, strict=True
        ):
            # It is possible that the batch is empty for a replica.
            if len(per_replica_batch) == 0:
                continue

            # Move the requests from CE to TG, skipping dummy padding contexts.
            for context in per_replica_batch:
                if not self.contains(context.request_id):
                    continue
                replica.tg_reqs[context.request_id] = context

            # Move Chunked requests back to the CE request queue.
            # Skip if the last request is a dummy padding context.
            last_request = per_replica_batch[-1]
            if (
                self.contains(last_request.request_id)
                and last_request.tokens.generated_length == 0
            ):
                del replica.tg_reqs[last_request.request_id]
                replica.ce_reqs[last_request.request_id] = last_request
                replica.ce_reqs.move_to_end(last_request.request_id, last=False)

    def _release_data_parallel_padding(self) -> None:
        """Releases dummy KV and pipeline entries from the previous batch's DP padding."""
        if self._prev_dp_padding is not None:
            for ctx in self._prev_dp_padding.dummies:
                if self.kv_cache.contains(ctx):
                    self.kv_cache.release(ctx)
                self.pipeline.release(ctx.request_id)
        self._prev_dp_padding = self._current_dp_padding
        self._current_dp_padding = None

    def contains(self, request_id: RequestID) -> bool:
        """Checks if a request is in the batch constructor for any replica."""
        return (
            request_id in self._bound_requests
            or request_id in self._ce_pending
            or request_id in self._grammar_pending
            or request_id in self._onloading_reqs
        )

    def _promote_grammar_ready_requests(self) -> None:
        """Admits held requests whose matcher build finished.

        Runs before the DP CE planner so promotions join this iteration's
        pool/bind flow. The ready matcher is installed here; a request whose
        build failed is failed instead of admitted, so nothing strands.
        """
        gate = self._grammar_gate
        if gate is None or not self._grammar_pending:
            return
        ready_ids = [
            req_id
            for req_id, pending in self._grammar_pending.items()
            if gate.is_ready(pending.ctx)
        ]
        for req_id in ready_ids:
            pending = self._grammar_pending.pop(req_id)
            error = gate.install_ready(pending.ctx)
            if error is None:
                self._admit_request(pending.ctx, pending.replica_idx)
            else:
                self._fail_grammar_request(pending.ctx, error)

    def _fail_grammar_request(self, ctx: TextContext, error: str) -> None:
        """Fails a request whose grammar build errored, without admitting it.

        Nothing was claimed for the request (it was never bound), so only the
        pipeline needs releasing; the owning scheduler drains
        :meth:`take_grammar_failed` to terminate it client-side.
        """
        METRICS.structured_output_grammar_rejection(
            "tool_grammar" if ctx.grammar else "json_schema"
        )
        self.pipeline.release(ctx.request_id)
        self._grammar_failed.append((ctx.request_id, error))

    def take_grammar_failed(self) -> list[tuple[RequestID, str]]:
        """Returns and clears requests failed by the grammar gate."""
        failed = self._grammar_failed
        self._grammar_failed = []
        return failed

    def release_request(self, request_id: RequestID) -> None:
        """
        Releases a request from the batch constructor for all replicas.

        This method searches for the given request_id in both context encoding (CE)
        and text generation (TG) request queues for each replica. If found, it removes
        the request entry and calls self.pipeline.release(request_id) to free resources.

        Args:
            request_id: The RequestID of the request to be released.
        """
        if not self.contains(request_id):
            raise ValueError(f"Request {request_id} not found in any replica.")

        # Grammar-gated requests are not bound to a replica yet: nothing was
        # claimed in the KV cache and no replica queue holds them.
        if request_id in self._grammar_pending:
            del self._grammar_pending[request_id]
            assert self._grammar_gate is not None
            self._grammar_gate.release(request_id)
            self.pipeline.release(request_id)
            return

        # Pooled CE requests are not bound to a replica yet: nothing was
        # claimed in the KV cache and no replica queue holds them.
        if request_id in self._ce_pending:
            del self._ce_pending[request_id]
            self._ce_arrival.pop(request_id, None)
            self.pipeline.release(request_id)
            return

        self._ce_arrival.pop(request_id, None)

        # Drop any cordon entry; the request's blocks are freed by the normal
        # release path below (its pending onload keeps them pinned until it
        # completes, after which poll_transfers unpins them).
        self._onloading_reqs.pop(request_id, None)

        bound = self._bound_requests[request_id]
        replica = self.replicas[bound.replica_idx]

        if request_id in replica.ce_reqs:
            del replica.ce_reqs[request_id]
        elif request_id in replica.tg_reqs:
            del replica.tg_reqs[request_id]
        elif request_id in replica.deferred_lora_requests:
            del replica.deferred_lora_requests[request_id]
        # Request may already be in an active batch and therefore not appear in
        # any pending queue; continue cleanup in that case.

        # Clean up LoRA state if no other request uses this adapter.
        # Note: We only check the current replica because LoRA currently requires
        # data_parallel_degree == 1. If DP > 1 LoRA becomes supported, this check
        # would need to search across all replicas.
        lora_name = self._request_id_to_lora_name.pop(request_id, None)
        if lora_name is not None:
            # Check _request_id_to_lora_name rather than the queues because
            # requests may be in the active batch (not in any queue) but still
            # using this LoRA adapter.
            lora_still_needed = (
                lora_name in self._request_id_to_lora_name.values()
            )
            if not lora_still_needed:
                replica.active_loras.discard(lora_name)

        # Release from paged cache (scheduler manages primary KV cache lifecycle).
        # Guard with contains() because _return_to_request_queue() may have
        # already released the KV cache (e.g. during preemption) while leaving
        # the request in _bound_requests so it remains visible to
        # contains(). Without this check a subsequent release_request() call
        # (e.g. from a delayed overlap-scheduler response) would attempt a
        # second release and raise "Attempted to release request ID but it is
        # not claimed".
        if self.kv_cache is not None and self.kv_cache.contains(bound.ctx):
            self.kv_cache.release(bound.ctx)

        # Pipeline release handles model-specific cleanup (e.g. vision encoder cache)
        self.pipeline.release(request_id)

        if self._grammar_gate is not None:
            self._grammar_gate.release(request_id)

        # _bound_requests is the source of truth for whether a request
        # is managed by the scheduler (checked by contains()).
        # Remove from here, marking the request as fully released.
        del self._bound_requests[request_id]

    def clear_tg_reqs(self) -> None:
        """Clears all TG requests from all replicas."""
        for replica in self.replicas:
            for request_id in replica.tg_reqs:
                del self._bound_requests[request_id]

            replica.tg_reqs.clear()

    @property
    def all_ce_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all CE requests from all replicas.

        Includes pooled CE requests not yet bound to a replica.
        """
        reqs = {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.ce_reqs.items()
        }
        reqs.update(
            (req_id, pending.ctx)
            for req_id, pending in self._ce_pending.items()
        )
        # Grammar-gated requests are still pending CE work.
        reqs.update(
            (req_id, pending.ctx)
            for req_id, pending in self._grammar_pending.items()
        )
        # Cordoned onloading requests are still pending CE work.
        reqs.update(
            (req_id, onloading.ctx)
            for req_id, onloading in self._onloading_reqs.items()
        )
        return reqs

    @property
    def all_tg_reqs(self) -> dict[RequestID, TextContext]:
        """Returns a dictionary of all TG requests from all replicas."""
        return {
            req_id: ctx
            for replica in self.replicas
            for req_id, ctx in replica.tg_reqs.items()
        }

    @traced
    def _return_to_request_queue(
        self, context: TextContext, replica_idx: int
    ) -> None:
        """Resets a request and returns it to the request queue"""

        # Release from paged cache if it was claimed (scheduler manages primary KV cache lifecycle)
        if self.kv_cache.contains(context):
            self.kv_cache.release(context)

        # Pipeline release handles special cases (spec decoding draft model KV cache)
        # For regular pipelines, release() is a no-op
        self.pipeline.release(context.request_id)

        context.reset()

        # Move to CE Queue
        replica_requests = self.replicas[replica_idx]
        if context.request_id in replica_requests.tg_reqs:
            del replica_requests.tg_reqs[context.request_id]

        replica_requests.ce_reqs[context.request_id] = context
        replica_requests.ce_reqs.move_to_end(context.request_id, last=False)

    @traced
    def _preempt_request(
        self, context: TextContext, replica_idx: int, reason: PreemptionReason
    ) -> None:
        """Preempts the most recently received request from active batch"""

        # Return to the Request Queue
        self._return_to_request_queue(context, replica_idx)

        # Log Preemption
        current_time = time.monotonic()
        self.total_preemption_count += 1
        METRICS.preemption()
        if current_time - self.last_preemption_logging_time > 1:
            self.last_preemption_logging_time = current_time
            logger.info(
                reason.error_message
                + f" Total Preemption Count: {self.total_preemption_count}"
            )

    def _identify_priority(self, replica_idx: int) -> RequestType | None:
        # DP CE balancing deferred this replica's CE work for this iteration;
        # run TG instead (the planner only defers replicas that have TG work).
        if replica_idx in self._ce_deferred_replicas:
            return RequestType.TG

        # A replica with no CE and no TG requests has no preference at all --
        # returning TG here (as the fallback below would) is indistinguishable
        # from a genuine TG preference to any caller that aggregates priority
        # across replicas (SERVOPT-1560: a spurious TG "vote" from an idle
        # replica broadcast as a batch-wide override, starving a sibling
        # replica's real, ready CE request forever).
        if (
            not self.replicas[replica_idx].ce_reqs
            and not self.replicas[replica_idx].tg_reqs
        ):
            return None

        # If there are no CE requests, prioritize TG
        if len(self.replicas[replica_idx].ce_reqs) == 0:
            return RequestType.TG

        # If there are no TG requests, prioritize Ce
        if len(self.replicas[replica_idx].tg_reqs) == 0:
            return RequestType.CE

        # Under KV pressure, drain TG before admitting more CE work: a new
        # prefill competes for the blocks in-flight generations still need,
        # and preempting one of those discards prefill already paid for.
        kv_percentage = self.kv_cache.block_count(replica_idx).used_pct
        if kv_percentage > self.tg_priority_kv_percentage:
            return RequestType.TG

        # If we've enabled in flight batching, prioritize TG
        if self.scheduler_config.enable_in_flight_batching:
            return RequestType.TG

        # Otherwise, prioritize CE
        return RequestType.CE

    def _is_anything_inflight(self, replica_idx: int) -> bool:
        """Whether anything that could still resolve into runnable work is
        in flight on this replica -- existence, not magnitude, since the
        fatal-vs-defer decision only needs to know whether anything at all
        is happening, not how much. Checked across cordoned onloads (H2D
        landing), any device-side pending transfer (e.g. a D2H offload),
        and any external (DI prefill/decode) signal. Scoped to
        replica_idx: a transfer on a different replica's device pool
        can't free blocks on this one.
        """
        if any(
            onloading.replica_idx == replica_idx
            for onloading in self._onloading_reqs.values()
        ):
            return True
        if self.kv_cache.pending_transfers_exist(replica_idx):
            return True
        if self._get_inflight_kv_transfer_count is not None:
            return self._get_inflight_kv_transfer_count(replica_idx) > 0
        return False

    def _preempt_ce_block_holder(self, replica_idx: int) -> bool:
        """Preempts the newest queued CE request that still holds KV blocks.

        Two kinds of requests sit in ce_reqs with blocks for KV they already
        hold: a chunked prefill between chunks (advance_requests requeues it
        without releasing), and a cordoned request whose onload completed
        (_readmit_completed_onloads returns it with its onloaded blocks).
        """
        replica_requests = self.replicas[replica_idx]
        for ctx in reversed(list(replica_requests.ce_reqs.values())):
            if self.kv_cache.contains(ctx) and self.kv_cache.get_req_blocks(
                ctx
            ):
                self._preempt_request(
                    ctx, replica_idx, reason=PreemptionReason.KV_CACHE_MEMORY
                )
                return True
        return False

    def _is_insufficient_blocks_fatal(
        self, replica_idx: int, no_other_work: bool
    ) -> bool:
        """Fatal only when there's no other runnable work and nothing at
        all is in flight -- presence, not magnitude, since a deficit
        sized against what's in flight now can't see a transfer that
        starts next iteration. A wrong "fatal" verdict kills the worker;
        a wrong "defer" verdict just costs a retry, so this errs toward
        retrying whenever anything at all is in flight.
        """
        return no_other_work and not self._is_anything_inflight(replica_idx)

    def _add_ce_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:
        # Deferred by the DP CE balancer this iteration (also covers paths
        # that bypass _identify_priority, e.g. in-flight batching).
        if replica_idx in self._ce_deferred_replicas:
            return

        replica_requests = self.replicas[replica_idx]
        max_batch_size = self.scheduler_config.max_batch_size

        # If there is anything in the batch, we can assume its TG, and the requests
        # are also counted in the tg_reqs.
        starting_tg_reqs_count = len(batch)
        max_batch_size = self.scheduler_config.max_batch_size
        while (
            len(batch) < max_batch_size
            # At a high level, active ce requests + tg_requests should not exceed the total max batch size.
            and len(batch)
            + len(replica_requests.tg_reqs)
            - starting_tg_reqs_count
            < max_batch_size
            and len(replica_requests.ce_reqs) > 0
        ):
            # Pop new request off the queue.
            req_id, ctx = replica_requests.ce_reqs.popitem(last=False)

            # Check LoRA budget before resource allocation
            if not replica_requests.can_allocate_lora_request(
                ctx, self._lora_manager
            ):
                continue

            # Exit early if the budget is already exhausted
            if batch.token_budget.remaining <= 0:
                self._return_to_request_queue(ctx, replica_idx)
                return

            # Check if the request fits in memory
            if self.kv_cache is not None:
                # Claim the request if needed.
                if not self.kv_cache.contains(ctx):
                    self.kv_cache.claim(ctx, replica_idx=replica_idx)

                try:
                    onload_event = self.kv_cache.alloc(ctx)
                except InsufficientBlocksError as e:
                    # Only fatal with no other work (no TG, no
                    # already-admitted CE this iteration) and nothing in
                    # flight can cover the deficit; otherwise requeue and
                    # retry next iteration -- poll_transfers unpins
                    # completed transfers' blocks in the meantime.
                    no_other_work = (
                        len(replica_requests.tg_reqs) == 0 and len(batch) == 0
                    )
                    fatal = self._is_insufficient_blocks_fatal(
                        replica_idx, no_other_work
                    )
                    if fatal:
                        held_blocks = len(self.kv_cache.get_req_blocks(ctx))
                        raise InsufficientBlocksError(
                            f"_add_ce_requests: InsufficientBlocksError is "
                            f"fatal on replica {replica_idx} -- no other "
                            f"work and nothing in flight to free blocks. "
                            f"Request {ctx.request_id} has "
                            f"{len(ctx.tokens)} tokens and already holds "
                            f"{held_blocks} blocks. {e}"
                        ) from e
                    self._return_to_request_queue(ctx, replica_idx)
                    break

                # Cordon the request if its KV onload has not landed: hold it
                # out of the batch (keeping its allocated/pinned blocks) so the
                # GPU runs other ready work while the H2D completes.
                if not onload_event.is_complete():
                    self._onloading_reqs[req_id] = _OnloadingRequest(
                        ctx=ctx, replica_idx=replica_idx, event=onload_event
                    )
                    continue

            # Check if it fits within the token budget
            match batch.token_budget.status_after_context(
                ctx, request_type=RequestType.CE
            ):
                case BudgetStatus.BUDGET_EXHAUSTED:
                    self._return_to_request_queue(ctx, replica_idx)
                    return
                case BudgetStatus.BUDGET_REACHED:
                    batch.batch[req_id] = ctx
                    replica_requests.add_active_lora(ctx, self._lora_manager)
                    batch.token_budget.add_to_budget(
                        ctx, request_type=RequestType.CE
                    )
                    break
                case BudgetStatus.BUDGET_AVAILABLE:
                    batch.batch[req_id] = ctx
                    batch.token_budget.add_to_budget(
                        ctx, request_type=RequestType.CE
                    )
                    replica_requests.add_active_lora(ctx, self._lora_manager)
                # case _:
                #     raise ValueError(f"Unexpected budget status: {status}")

        # Update deferred LoRA requests
        replica_requests.update_deferred_lora_requests()

    def _add_tg_requests(self, batch: ReplicaBatch, replica_idx: int) -> None:
        replica_requests = self.replicas[replica_idx]

        # Add based on the oldest request, respecting KV cache limits and token budgets.
        candidate_ids = deque(replica_requests.tg_reqs.keys())
        max_batch_size = self.scheduler_config.max_batch_size
        while len(batch) < max_batch_size and len(candidate_ids) > 0:
            # Pop the oldest request
            candidate_id = candidate_ids.popleft()
            candidate_context = replica_requests.tg_reqs[candidate_id]

            # Verify LoRA is active for TG requests
            # LoRA requests should have been activated during CE
            if is_lora(
                candidate_context, self._lora_manager
            ) and not is_active_lora(candidate_context, self._lora_manager):
                self._preempt_request(
                    candidate_context,
                    replica_idx,
                    reason=PreemptionReason.MAX_NUM_LORAS,
                )
                continue

            # Check if it fits within the token budget
            # This is quite cheap, compared to the paged cache allocations
            # So we can do it first.
            status = batch.token_budget.status_after_context(
                candidate_context,
                request_type=RequestType.TG,
            )
            if status == BudgetStatus.BUDGET_EXHAUSTED:
                break

            # At this point, we can assume that the paged cache is active.
            while True:
                try:
                    self.kv_cache.alloc(candidate_context)
                    break
                except InsufficientBlocksError as e:
                    if len(candidate_ids) == 0:
                        # No TG candidates left, but a request parked in
                        # ce_reqs (a chunked prefill between chunks, or a
                        # readmitted onload) may still hold reclaimable
                        # blocks.
                        if self._preempt_ce_block_holder(replica_idx):
                            continue
                        # Only a genuine OOM is fatal: nothing left to
                        # preempt, an empty batch, and the deficit can't be
                        # covered by anything in flight.
                        fatal = self._is_insufficient_blocks_fatal(
                            replica_idx, no_other_work=len(batch) == 0
                        )
                        if fatal:
                            held_blocks = len(
                                self.kv_cache.get_req_blocks(candidate_context)
                            )
                            raise InsufficientBlocksError(
                                f"_add_tg_requests: InsufficientBlocksError "
                                f"is fatal on replica {replica_idx} -- no "
                                f"other work and nothing in flight to free "
                                f"blocks. Request "
                                f"{candidate_context.request_id} has "
                                f"{len(candidate_context.tokens)} tokens "
                                f"and already holds {held_blocks} "
                                f"blocks. {e}"
                            ) from e
                        return

                    # Pop the oldest candidate id
                    oldest_id = candidate_ids.pop()
                    oldest_context = replica_requests.tg_reqs.pop(oldest_id)
                    self._preempt_request(
                        oldest_context,
                        replica_idx,
                        reason=PreemptionReason.KV_CACHE_MEMORY,
                    )

            match status:
                case BudgetStatus.BUDGET_REACHED:
                    batch.batch[candidate_context.request_id] = (
                        candidate_context
                    )
                    batch.token_budget.add_to_budget(
                        candidate_context, request_type=RequestType.TG
                    )
                    break
                case BudgetStatus.BUDGET_AVAILABLE:
                    batch.batch[candidate_context.request_id] = (
                        candidate_context
                    )
                    batch.token_budget.add_to_budget(
                        candidate_context, request_type=RequestType.TG
                    )
                case _:
                    raise ValueError(f"Unexpected budget status: {status}")

    @traced
    def _construct_replica_batch(
        self, replica_idx: int, priority_override: RequestType | None = None
    ) -> ReplicaBatch:
        """Constructs a batch for a single replica.

        Args:
            replica_idx: The index of the replica to construct a batch for.
            priority_override: Optional RequestType to override the priority
                identified by _identify_priority. If None, priority is determined
                automatically based on queue state and scheduler configuration.
        """

        # Initialize batch. When the DP CE balancer set a step quota for this
        # replica, it becomes the CE budget capacity, sizing this step's
        # chunks to the balance level instead of the full chunk target.
        ce_capacity: int | None = None
        if (
            self._ce_step_quota is not None
            and self._ce_step_quota[replica_idx] > 0
            # Under in-flight batching the TG batch shares this budget; a
            # small CE quota must not truncate it.
            and not self.scheduler_config.enable_in_flight_batching
        ):
            ce_capacity = self._ce_step_quota[replica_idx]
        batch = ReplicaBatch(
            batch={},
            token_budget=self._create_new_token_budget(ce_capacity),
        )

        # Use override if provided, otherwise identify priority automatically
        priority = (
            priority_override
            if priority_override is not None
            else self._identify_priority(replica_idx)
        )

        match priority:
            case RequestType.CE:
                self._add_ce_requests(batch, replica_idx)

                if len(batch) == 0 and priority_override is None:
                    self._add_tg_requests(batch, replica_idx)
                else:
                    # This iteration is CE-only: the TG fallback above didn't
                    # fire, so any pending TG requests get zero progress this
                    # iteration (see _add_tg_requests, only reached from the
                    # branch above). Pair with this batch's own execution
                    # duration to estimate the TPOT cost of the stolen
                    # iteration.
                    pending_tg_count = len(self.replicas[replica_idx].tg_reqs)
                    if len(batch) > 0 and pending_tg_count > 0:
                        METRICS.di_ce_preempted_tg_iteration_count()
                        METRICS.di_ce_preempted_tg_pending_count(
                            pending_tg_count
                        )

            case RequestType.TG:
                self._add_tg_requests(batch, replica_idx)

                if (
                    self.scheduler_config.enable_in_flight_batching
                    and len(batch) > 0
                    and priority_override is None
                ):
                    self._add_ce_requests(batch, replica_idx)

            case None:
                # Genuinely idle replica (no CE, no TG requests): nothing to
                # add either way.
                pass

        return batch

    def _plan_ce_step(self) -> None:
        """Plans this iteration's CE work across DP replicas.

        Prices CE work in post-prefix-cache tokens and greedily assembles the
        most balanced CE step it can, deferring the rest:

        - Work that must run — deferral deadline expired, no deadline on
          record, or its replica has no TG work to run instead — forms the
          step's floor. Expired pooled requests bind immediately, in arrival
          order, so out-of-order balancing can never starve an old request.
        - Deferrable mid-prefill tails (per replica, all-or-nothing) and then
          pooled unbound requests are added largest-first wherever they
          strictly improve the step's occupancy (mean/max of per-replica CE
          tokens, capped at the CE chunk budget).
        - Pooled requests bind to ``argmin(total_load + weight)`` at the moment the
          planner schedules them: binding is deferred until first run so it
          uses fresh loads.

        The assembled step is committed when the floor is non-empty (those
        tokens run regardless, so riders only improve the step), when its
        occupancy meets ``dp_ce_balance_threshold``, or when the fleet has
        nothing else to do. With ``dp_ce_balance_enable_dynamic_chunk_size``,
        a below-threshold step with CE work on two or more replicas also
        commits, at a reduced chunk size: each working replica's CE quota
        (``_ce_step_quota``, enforced as that replica's CE budget capacity)
        is reduced to the lightest working replica's level — never below a
        replica's floor — so the balanced portion runs now as a shorter,
        ~fully-occupied step and only the excess is deferred. Only a single
        replica's deferrable work with no partner anywhere is held outright:
        deferred replicas run TG this iteration and pooled requests stay
        unbound.
        """
        self._ce_deferred_replicas.clear()
        self._ce_step_quota = None
        if not self._dp_ce_balance_enabled:
            return
        if not self._ce_pending and not any(
            replica.ce_reqs for replica in self.replicas
        ):
            return

        target = self.scheduler_config.target_tokens_per_batch_ce
        timeout_s = self.scheduler_config.dp_ce_balance_timeout_ms / 1000.0
        now = time.monotonic()

        def _expired(request_id: RequestID) -> bool:
            arrival = self._ce_arrival.get(request_id)
            return arrival is None or now - arrival >= timeout_s

        # The floor: per-replica step CE tokens that run no matter what. A
        # replica's tails are deferrable only when all of them have deadline
        # budget left AND the replica has TG work to run instead (deferring
        # into idleness loses throughput for nothing).
        #
        # ``step_load`` is this step's projection (capped at the CE chunk
        # budget), used for occupancy and quotas. ``total_load`` is the
        # uncapped per-replica queue total, used for binding decisions.
        floor = [0] * self.num_replicas
        total_load = [0] * self.num_replicas
        deferrable_tails: list[tuple[int, int]] = []  # (step_tokens, replica)
        for replica_idx, replica in enumerate(self.replicas):
            tokens = sum(
                ctx.tokens.active_length for ctx in replica.ce_reqs.values()
            )
            total_load[replica_idx] = tokens
            if tokens == 0:
                continue
            can_defer = bool(replica.tg_reqs) and not any(
                _expired(req_id) for req_id in replica.ce_reqs
            )
            if can_defer:
                deferrable_tails.append((min(tokens, target), replica_idx))
            else:
                floor[replica_idx] = min(tokens, target)

        step_load = list(floor)

        # Expired pooled requests bind now and join the floor.
        for req_id in [r for r in self._ce_pending if _expired(r)]:
            pending = self._ce_pending.pop(req_id)
            replica_idx = min(
                range(self.num_replicas),
                key=lambda i: total_load[i] + pending.weights[i],
            )
            total_load[replica_idx] += pending.weights[replica_idx]
            step_load[replica_idx] = min(
                step_load[replica_idx] + pending.weights[replica_idx], target
            )
            floor[replica_idx] = step_load[replica_idx]
            self._bind_request(pending.ctx, replica_idx)

        def _occupancy(loads: list[int]) -> float:
            max_load = max(loads)
            if max_load == 0:
                return 1.0
            return sum(loads) / len(loads) / max_load

        # Greedy valley-fill, largest-first, accepting work only where it
        # strictly improves the step's occupancy (or seeds an empty step).
        deferred = {replica_idx for _, replica_idx in deferrable_tails}
        for step_tokens, replica_idx in sorted(deferrable_tails, reverse=True):
            trial = list(step_load)
            trial[replica_idx] = min(trial[replica_idx] + step_tokens, target)
            if max(step_load) == 0 or _occupancy(trial) > _occupancy(step_load):
                step_load = trial
                deferred.discard(replica_idx)

        # Pooled requests may only bind to replicas running CE this step
        # (otherwise they would queue behind a deferred tail).
        pool_binds: list[tuple[RequestID, int]] = []
        for req_id, pending in sorted(
            self._ce_pending.items(),
            key=lambda item: min(item[1].weights),
            reverse=True,
        ):
            eligible = [
                i
                for i in range(self.num_replicas)
                if i not in deferred and step_load[i] < target
            ]
            if not eligible:
                continue
            replica_idx = min(
                eligible, key=lambda i: total_load[i] + pending.weights[i]
            )
            trial = list(step_load)
            trial[replica_idx] = min(
                trial[replica_idx] + pending.weights[replica_idx], target
            )
            if max(step_load) == 0 or _occupancy(trial) > _occupancy(step_load):
                step_load = trial
                total_load[replica_idx] += pending.weights[replica_idx]
                pool_binds.append((req_id, replica_idx))

        floor_exists = any(floor)
        fleet_idle = not floor_exists and all(
            not replica.tg_reqs for replica in self.replicas
        )
        threshold = self.scheduler_config.dp_ce_balance_threshold
        occupancy = _occupancy(step_load)
        # A below-threshold step with work on 2+ replicas need not be held:
        # reducing the heavy replicas' chunk size to the balance level runs
        # the balanced portion now (a shorter, ~100%-occupancy step) and
        # defers only the excess. The balance level must be substantial
        # (>= half the chunk target): each extra chunk re-reads the
        # request's full context in attention, so undersized chunks cost
        # more than the imbalance they avoid. The level never drops below
        # any replica's floor (expired work is not deferrable and runs to
        # the full chunk budget regardless).
        loads_with_work = [tokens for tokens in step_load if tokens > 0]
        balance_level = (
            max(max(floor), min(loads_with_work)) if loads_with_work else 0
        )
        can_reduce_chunk_size = (
            self.scheduler_config.dp_ce_balance_enable_dynamic_chunk_size
            and len(loads_with_work) > 1
            and balance_level >= target // 2
        )
        if (
            floor_exists
            or fleet_idle
            or occupancy >= threshold
            or can_reduce_chunk_size
        ):
            for req_id, replica_idx in pool_binds:
                pending = self._ce_pending.pop(req_id)
                self._bind_request(pending.ctx, replica_idx)
            self._ce_deferred_replicas = deferred

            if (
                occupancy < threshold
                and can_reduce_chunk_size
                and not fleet_idle
            ):
                quotas = [
                    self.scheduler_config.target_tokens_per_batch_ce
                ] * self.num_replicas
                for replica_idx in range(self.num_replicas):
                    if step_load[replica_idx] > 0:
                        quotas[replica_idx] = max(
                            floor[replica_idx],
                            min(step_load[replica_idx], balance_level),
                        )
                self._ce_step_quota = quotas
        else:
            # No worthwhile balanced step exists (lone replica, or a balance
            # level too small to shrink chunks to): hold everything
            # deferrable for a better step.
            self._ce_deferred_replicas = {
                replica_idx for _, replica_idx in deferrable_tails
            }

    def _readmit_completed_onloads(self) -> None:
        """Re-admits cordoned requests whose KV onload has completed.

        Runs once per iteration. Completed requests return to the front of their
        replica's CE queue (FIFO by arrival) so the next batch pass can schedule
        them; their onloaded prefix is now device-resident.
        """
        if not self._onloading_reqs:
            return
        completed = [
            req_id
            for req_id, onloading in self._onloading_reqs.items()
            if onloading.event.is_complete()
        ]
        for req_id in completed:
            onloading = self._onloading_reqs.pop(req_id)
            replica = self.replicas[onloading.replica_idx]
            replica.ce_reqs[req_id] = onloading.ctx
            replica.ce_reqs.move_to_end(req_id, last=False)

    def construct_batch(self) -> TextGenerationInputs[TextContext]:
        """Constructs Pipeline Inputs which includes a batch for each replica."""

        # Re-admit any cordoned requests whose KV onload has landed. The
        # block-level drain (unpin/commit/reclaim) is handled inside
        # ``kv_cache.alloc`` itself, so the scheduler only tracks the
        # request-level side here.
        self._readmit_completed_onloads()

        self._promote_grammar_ready_requests()

        # DP-balanced CE deferral: decide this iteration's CE work (late
        # binding + per-replica deferral) before priorities are identified.
        self._plan_ce_step()

        priority_override = None
        # None entries (replicas with no CE and no TG requests -- see
        # _identify_priority) fail every membership/count check below on
        # their own, so an idle replica never casts a vote: SERVOPT-1560 was
        # exactly this default being indistinguishable from a genuine TG
        # preference once aggregated across replicas.
        replica_priorities: set[RequestType | None] | list[RequestType | None]
        match self.batch_scheduling_strategy:
            case BatchSchedulingStrategy.DECODE_FIRST:
                replica_priorities = {
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                }
                if RequestType.TG in replica_priorities:
                    priority_override = RequestType.TG
            case BatchSchedulingStrategy.PREFILL_FIRST:
                replica_priorities = {
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                }
                if RequestType.CE in replica_priorities:
                    priority_override = RequestType.CE
            case BatchSchedulingStrategy.BALANCED:
                replica_priorities = [
                    self._identify_priority(idx)
                    for idx in range(self.num_replicas)
                ]

                # Count occurrences of each priority type
                ce_count = replica_priorities.count(RequestType.CE)
                tg_count = replica_priorities.count(RequestType.TG)

                # Set priority to the majority case, defaulting to TG if tied
                if ce_count > tg_count:
                    priority_override = RequestType.CE
                else:
                    priority_override = RequestType.TG

        # Batch construction preempts requests under KV / LoRA pressure, and
        # each preemption emits a metric. Wrap the pass in a transaction so a
        # preemption storm coalesces into one cross-process flush instead of
        # one packet per preempted request.
        with METRICS.transaction():
            batches_per_replica = [
                self._construct_replica_batch(
                    replica_idx, priority_override=priority_override
                )
                for replica_idx in range(self.num_replicas)
            ]

        inputs = TextGenerationInputs[TextContext](
            batches=[
                list(batch.batch.values()) for batch in batches_per_replica
            ],
        )

        # Pad short replicas for device graph capture when DP > 1.
        if self._dp_padder is not None:
            inputs, info = self._dp_padder.pad_batch(inputs)
            self._current_dp_padding = info

        return inputs
