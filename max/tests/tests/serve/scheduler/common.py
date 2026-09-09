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

import queue
from dataclasses import dataclass

import numpy as np
from max.driver import CPU, Accelerator, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.nn.kv_cache import (
    KVCacheParamInterface,
    KVCacheParams,
    MHAKVCacheParams,
    MLAKVCacheParams,
    MultiKVCacheParams,
)
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache import (
    PagedKVCacheManager,
    PagedKVCacheManagerInterface,
)
from max.pipelines.kv_cache.config import KVConnectorConfig
from max.pipelines.modeling.types import (
    BatchType,
    CompletedBatchStats,
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.serve.queue import MAXPushQueue
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.serve.scheduler_result import SchedulerResult


def rand(length: int) -> np.ndarray:
    return np.random.randint(0, 256, size=length)


def create_text_context(
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> TextContext:
    if shared_prefix is None:
        tokens = np.ones(prompt_len, dtype=np.int64)
    else:
        rem_tokens = prompt_len - len(shared_prefix)
        assert rem_tokens >= 0
        tokens = np.concatenate([shared_prefix, rand(rem_tokens)])

    return TextContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
    )


def create_kv_cache(
    num_blocks: int,
    max_batch_size: int,
    max_seq_len: int,
    page_size: int,
    enable_prefix_caching: bool = False,
    kv_connector_config: KVConnectorConfig | None = None,
    dp: int = 1,
    device: Device = CPU(),  # noqa: B008
    num_speculative_tokens: int = 0,
    is_mla: bool = False,
    tp_per_replica: int = 1,
    multi_kv: bool = False,
) -> PagedKVCacheManager:
    """Builds a ``PagedKVCacheManager`` for scheduler tests.

    Args:
        multi_kv: Build a ``MultiKVCacheParams`` tree with ``target`` and
            ``draft`` children instead of a flat leaf, mirroring production
            speculative decoding. The children are deliberately
            different-shaped so the per-child NIXL grouping in
            ``KVTransferEngine.from_paged_kv_cache`` is exercised.
    """
    dtype = DType.float32

    if tp_per_replica > 1:
        # Simulate multiple TP shards per replica by allocating distinct
        # device ids (used to exercise the MLA flatten path).
        n_devices = dp * tp_per_replica
        if isinstance(device, CPU):
            session_devices: list[Device] = [
                CPU(id=i) for i in range(n_devices)
            ]
        elif isinstance(device, Accelerator):
            session_devices = [Accelerator(id=i) for i in range(n_devices)]
        else:
            raise TypeError(
                f"tp_per_replica > 1 not supported for {type(device).__name__}"
            )
        device_refs = [DeviceRef.from_device(d) for d in session_devices]
    else:
        device_refs = [DeviceRef.from_device(device) for _ in range(dp)]
        session_devices = [device]

    kv_connector_config = kv_connector_config or KVConnectorConfig()

    def make_leaf_params(num_layers: int, head_dim: int) -> KVCacheParams:
        if is_mla:
            return MLAKVCacheParams(
                dtype=dtype,
                num_layers=num_layers,
                head_dim=head_dim,
                page_size=page_size,
                enable_prefix_caching=enable_prefix_caching,
                kv_connector_config=kv_connector_config,
                data_parallel_degree=dp,
                devices=device_refs,
                speculative_method="eagle"
                if num_speculative_tokens > 0
                else None,
                num_draft_tokens=num_speculative_tokens,
                # num_q_heads must be divisible by the per-replica device count
                # (TP shards) when MLA is enabled.
                num_q_heads=tp_per_replica,
            )
        return MHAKVCacheParams(
            dtype=dtype,
            num_layers=num_layers,
            n_kv_heads=1,
            head_dim=head_dim,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
            kv_connector_config=kv_connector_config,
            data_parallel_degree=dp,
            devices=device_refs,
            speculative_method="eagle" if num_speculative_tokens > 0 else None,
            num_draft_tokens=num_speculative_tokens,
        )

    kv_params: KVCacheParamInterface
    if multi_kv:
        # Production spec decode pairs a deep target cache with a shallow
        # (typically 1-layer Eagle) draft. Differing num_layers/head_dim keeps
        # the two NIXL groups shape-heterogeneous, which is the property the
        # transfer engine's per-child grouping has to get right.
        kv_params = MultiKVCacheParams.from_params(
            {
                "target": make_leaf_params(num_layers=2, head_dim=2),
                "draft": make_leaf_params(num_layers=1, head_dim=1),
            }
        )
    else:
        kv_params = make_leaf_params(num_layers=1, head_dim=1)

    session = InferenceSession(devices=session_devices)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=num_blocks,
        session=session,
        enable_runtime_checks=True,
        max_batch_size=max_batch_size,
    )

    assert all(
        kv_manager.block_count(replica_idx=replica_idx).total == num_blocks
        for replica_idx in range(dp)
    )
    return kv_manager


def create_paged_scheduler(
    max_seq_len: int = 2048,
    num_blocks: int = 9999,
    max_batch_size: int = 512,
    page_size: int = 128,
    target_tokens_per_batch_ce: int = 8192,
    enable_prefix_caching: bool = False,
    enable_in_flight_batching: bool = False,
    enable_chunked_prefill: bool = True,
    kv_connector_config: KVConnectorConfig | None = None,
    max_batch_total_tokens: int | None = None,
    dp: int = 1,
    device: Device = CPU(),  # noqa: B008
    num_speculative_tokens: int = 0,
    max_pending_requests: int | None = None,
) -> tuple[
    TokenGenerationScheduler,
    MAXPushQueue[TextContext],
]:
    # Create a paged manager that has one slot
    kv_cache = create_kv_cache(
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        page_size=page_size,
        enable_prefix_caching=enable_prefix_caching,
        kv_connector_config=kv_connector_config,
        dp=dp,
        device=device,
        num_speculative_tokens=num_speculative_tokens,
    )

    # Create a scheduler with a paged manager
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        max_seq_len=max_seq_len,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_in_flight_batching=enable_in_flight_batching,
        max_batch_total_tokens=max_batch_total_tokens,
        data_parallel_degree=dp,
        num_speculative_tokens=num_speculative_tokens,
    )
    token_pipeline = FakeTokenGeneratorPipeline(
        kv_manager=kv_cache,
        max_seq_len=max_seq_len,
        num_speculative_tokens=num_speculative_tokens,
    )
    request_queue: queue.Queue[TextContext] = queue.Queue()
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()
    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=token_pipeline,
        kv_cache=kv_cache,
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        max_pending_requests=max_pending_requests,
    )

    return (scheduler, request_queue)


class FakeTokenGeneratorPipeline(
    Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput]
):
    def __init__(
        self,
        kv_manager: PagedKVCacheManagerInterface,
        max_seq_len: int,
        start_token_id: int = 42,
        num_speculative_tokens: int = 0,
    ) -> None:
        self.kv_manager = kv_manager
        self.token_id = start_token_id
        self.max_seq_len = max_seq_len
        self.num_speculative_tokens = num_speculative_tokens

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        max_seq_len = self.max_seq_len
        num_steps = 1
        for context in inputs.flat_batch:
            num_available_steps = context.compute_num_available_steps(
                max_seq_len
            )
            assert num_available_steps > 0

        # Claim cache rows for context.
        for replica_idx, batch in enumerate(inputs.batches):
            for context in batch:
                if not self.kv_manager.contains(context):
                    self.kv_manager.claim(context, replica_idx=replica_idx)

        for batch in inputs.batches:
            for ctx in batch:
                self.kv_manager.alloc(ctx)
        self.kv_manager.runtime_inputs(inputs.batches)

        # Generate the responses
        responses = {}
        for context in inputs.flat_batch:
            req_id = context.request_id
            for _ in range(num_steps):
                context.update(new_token=self.token_id)
                self.token_id += 1

                if len(context.tokens) == context.max_length:
                    context.status = GenerationStatus.MAXIMUM_LENGTH

                if context.is_done:
                    break

            output = context.to_generation_output()
            if output.tokens:
                responses[req_id] = output

        # Step the kv cache manager
        for ctx in inputs.flat_batch:
            self.kv_manager.step(ctx)

        # If num spec tokens, populate the draft tokens for the reqs
        if self.num_speculative_tokens > 0:
            for context in inputs.flat_batch:
                context.spec_decoding_state.draft_tokens_to_verify = [
                    123
                ] * self.num_speculative_tokens

        return responses

    @property
    def max_batch_size(self) -> int:
        return 1

    def release(self, request_id: RequestID) -> None:
        # No-op. Previously the pipeline was responsible for calling kv.release().
        # but now the whole lifecycle is managed by the scheduler.
        pass


class FakeOverlapPipeline(FakeTokenGeneratorPipeline):
    """Mimics OverlapTextGenerationPipeline's one-batch output lag.

    execute() returns the *previous* batch's real tokens, resolves their
    FUTURE_TOKEN placeholders in-place (matching the real pipeline's
    sync_and_process_outputs behavior), appends FUTURE_TOKEN to current-batch
    contexts, and stores their real tokens for the next call.

    ``disable_overlap=True`` mirrors the real pipeline's ``_disable_overlap``:
    execute() runs synchronously and has_pending_outputs() returns False.
    ``num_speculative_tokens > 0`` populates draft_tokens_to_verify on each
    context to mimic unified Eagle / MTP output.

    Like the real pipeline, draining a deferred batch records
    ``CompletedBatchStats`` for it (with the fixed execution time
    ``FAKE_EXECUTION_TIME_S``), retrievable once via
    ``take_completed_batch_stats()``.
    """

    FAKE_EXECUTION_TIME_S = 0.125
    """Execution time reported in CompletedBatchStats for every drained batch."""

    def __init__(
        self,
        kv_manager: PagedKVCacheManagerInterface,
        max_seq_len: int,
        start_token_id: int = 99,  # test sentinel; no semantic meaning
        num_speculative_tokens: int = 0,
        disable_overlap: bool = False,
    ) -> None:
        super().__init__(
            kv_manager,
            max_seq_len,
            start_token_id,
            num_speculative_tokens=num_speculative_tokens,
        )
        self._disable_overlap = disable_overlap
        self._pending_outputs: dict[RequestID, TextGenerationOutput] | None = (
            None
        )
        self._pending_contexts: list[TextContext] = []
        self._pending_inputs: TextGenerationInputs[TextContext] | None = None
        self._completed_batch_stats: CompletedBatchStats | None = None

    def has_pending_outputs(self) -> bool:
        if self._disable_overlap:
            return False
        return self._pending_outputs is not None

    @property
    def overlap_active(self) -> bool:
        return not self._disable_overlap

    def take_completed_batch_stats(self) -> CompletedBatchStats | None:
        stats = self._completed_batch_stats
        self._completed_batch_stats = None
        return stats

    def execute(
        self, inputs: TextGenerationInputs[TextContext]
    ) -> dict[RequestID, TextGenerationOutput]:
        if self._disable_overlap:
            return super().execute(inputs)

        # Return the previous batch's real outputs (one-batch lag) and resolve
        # their FUTURE_TOKEN placeholders, matching sync_and_process_outputs.
        outputs: dict[RequestID, TextGenerationOutput] = {}
        if self._pending_outputs:
            for context in self._pending_contexts:
                req_id = context.request_id
                if req_id in self._pending_outputs:
                    real_token = self._pending_outputs[req_id].tokens[0]
                    context.realize_future_token(real_token)
                    output = context.to_generation_output()
                    if output.tokens:
                        outputs[req_id] = output
        # Record CompletedBatchStats for the drained batch, matching the real
        # pipeline's _record_completed_batch_stats (with a fixed fake time).
        if self._pending_inputs is not None:
            pending = self._pending_inputs
            self._completed_batch_stats = CompletedBatchStats(
                batch_type=pending.batch_type,
                batch_size=len(pending.flat_batch),
                num_input_tokens=pending.input_tokens,
                num_context_tokens=pending.context_tokens,
                execution_time_s=self.FAKE_EXECUTION_TIME_S,
            )
        self._pending_outputs = None
        self._pending_contexts = []
        self._pending_inputs = None

        if inputs:
            for replica_idx, batch in enumerate(inputs.batches):
                for context in batch:
                    if not self.kv_manager.contains(context):
                        self.kv_manager.claim(context, replica_idx=replica_idx)
            for batch in inputs.batches:
                for ctx in batch:
                    self.kv_manager.alloc(ctx)
            self.kv_manager.runtime_inputs(inputs.batches)

            # Generate real tokens now but defer their release to the next call.
            new_outputs: dict[RequestID, TextGenerationOutput] = {}
            for context in inputs.flat_batch:
                req_id = context.request_id
                real_token = self.token_id
                self.token_id += 1
                # Append FUTURE_TOKEN placeholder, exactly like the real pipeline.
                context.update_with_future_token()
                new_outputs[req_id] = TextGenerationOutput(
                    request_id=req_id,
                    tokens=[real_token],
                    final_status=GenerationStatus.ACTIVE,
                )

            # Publish draft tokens on the deferred contexts, matching the
            # real unified Eagle/MTP overlap CE output.
            if self.num_speculative_tokens > 0:
                for context in inputs.flat_batch:
                    context.spec_decoding_state.draft_tokens_to_verify = [
                        123
                    ] * self.num_speculative_tokens

            for ctx in inputs.flat_batch:
                self.kv_manager.step(ctx)
            self._pending_outputs = new_outputs
            self._pending_contexts = list(inputs.flat_batch)
            self._pending_inputs = inputs

        return outputs


@dataclass(eq=True)
class BatchInfo:
    batch_type: BatchType
    """Type of the batch, either CE or TG"""

    batch_size: int
    """Batch size. This is the number of requests in the batch."""

    terminated: int
    """Number of requests that were terminated after this iteration in the batch."""

    steps: int
    """Number of steps to execute for."""

    preempted: int = -1
    """Number of requests that were preempted while scheduling this batch."""

    input_toks: int = -1
    """Total number of input tokens across all requests in the batch."""

    cached_toks: int = -1
    """Total number of cached context tokens across all requests in the batch."""

    @classmethod
    def empty(cls) -> BatchInfo:
        return BatchInfo(
            BatchType.TG,
            batch_size=0,
            terminated=0,
            steps=0,
            preempted=0,
            input_toks=0,
            cached_toks=0,
        )

    def __repr__(self) -> str:
        return (
            f"BatchInfo("
            f"{self.batch_type.value}, "
            f"batch_size={self.batch_size}, "
            f"terminated={self.terminated}, "
            f"steps={self.steps}, "
            f"preempted={self.preempted}, "
            f"input_toks={self.input_toks}, "
            f"cached_toks={self.cached_toks}"
            f")"
        )


def pretty_format_batch_info_list(batch_info_list: list[BatchInfo]) -> str:
    """Pretty format a list of BatchInfo for printing to the console."""
    return "[\n\t" + "\n\t".join([f"{x}," for x in batch_info_list]) + "\n]"


def assert_batch_info_equal(
    actual: list[BatchInfo], expected: list[BatchInfo]
) -> None:
    """Assert that two lists of BatchInfo are equal.

    When the lists are unequal, this method ensures that the output dumped to the
    console is easily copy-pastable into the test code.

    This method is preferred over `assert actual == expected`.

    If we naively compare the lists via above method, the output is very
    verbose and cluttered. The assert dumps the contents of `expected` which is
    unnecessary since it is present in the code. Pytest also often elides some
    elements of the list, preventing us from copy-pasting the list into the test code.
    """

    if len(actual) != len(expected):
        # Save lengths to local variable so pytest does not try to print `actual` / `expected`.
        len_actual = len(actual)
        len_expected = len(expected)
        raise AssertionError(
            f"Lengths of actual and expected batch infos do not match: {len_actual} != {len_expected}. Actual:\n"
            f"{pretty_format_batch_info_list(actual)}\nExpected:\n"
            f"{pretty_format_batch_info_list(expected)}"
        )
    for i in range(len(actual)):
        if actual[i] != expected[i]:
            raise AssertionError(
                f"Batch info at index {i} does not match: {actual[i]} != {expected[i]}. Actual:\n"
                f"{pretty_format_batch_info_list(actual)}\nExpected:\n"
                f"{pretty_format_batch_info_list(expected)}"
            )


def create_batch_and_execute(scheduler: TokenGenerationScheduler) -> BatchInfo:
    scheduler._retrieve_pending_requests()
    batch_constructor = scheduler.batch_constructor

    num_preempted_before = scheduler.batch_constructor.total_preemption_count
    inputs = batch_constructor.construct_batch()
    num_preempted_after = scheduler.batch_constructor.total_preemption_count

    num_preempted = num_preempted_after - num_preempted_before
    batch_size = len(inputs.flat_batch)
    batch_type = inputs.batch_type
    input_tokens = inputs.input_tokens
    batch_context_length = sum(
        context.tokens.processed_length for context in inputs.flat_batch
    )

    if batch_size == 0:
        return BatchInfo.empty()

    num_terminated_reqs = scheduler._schedule(inputs)
    assert isinstance(scheduler.pipeline, FakeTokenGeneratorPipeline)

    return BatchInfo(
        batch_type=batch_type,
        batch_size=batch_size,
        terminated=num_terminated_reqs,
        steps=1,
        preempted=num_preempted,
        input_toks=input_tokens,
        cached_toks=batch_context_length,
    )


def run_until_completion(
    scheduler: TokenGenerationScheduler,
    max_num_iters: int = 50,
    output_list: list | None = None,  # type: ignore[type-arg]
) -> list[BatchInfo]:
    if output_list is None:
        batch_infos = []
    else:
        batch_infos = output_list

    batch_constructor = scheduler.batch_constructor
    kv_cache = batch_constructor.kv_cache
    for _ in range(max_num_iters):
        batch_info = create_batch_and_execute(scheduler)
        # An asynchronous KV connector can produce an empty batch for two
        # distinct reasons, both meaning "not ready yet", not "done":
        #  - A candidate request is cordoned in the batch constructor's own
        #    ``_onloading_reqs`` until its onload's ``is_complete()`` flips
        #    (``construct_batch`` re-admits it automatically once it does).
        #  - A pending offload (or a completed-but-not-yet-drained transfer)
        #    keeps a device block pinned in the block manager's
        #    ``_pending_transfers``, starving a new allocation with
        #    ``InsufficientBlocksError`` even with no cordoned request in
        #    sight. Neither condition alone catches both cases, so check
        #    both.
        #
        # Rather than spin-polling a bounded tick count, block on the actual
        # transfer handles: under real GPU contention (several GPU tests
        # sharing one device in CI) an H2D/D2H copy can take far longer than
        # any reasonable poll budget, so a fixed retry count is inherently
        # flaky (confirmed directly: it landed anywhere from 1 to 49+ retries
        # across repeated runs on a loaded GPU). Progress here is strictly
        # serialized -- num_gpu_blocks admits one request at a time -- so
        # there is nothing else useful this loop could do meanwhile.
        poll_iters = 0
        while (
            batch_info.batch_size == 0
            and (
                batch_constructor._onloading_reqs
                or kv_cache.pending_transfers_exist()
            )
            and poll_iters < max_num_iters
        ):
            for onloading in list(batch_constructor._onloading_reqs.values()):
                onloading.event.synchronize()
            # KVConnector pending-transfer bookkeeping is legacy-manager-only
            # (Jenga doesn't support KVConnector), so narrow before reaching
            # into internals this helper's synchronization actually needs.
            assert isinstance(kv_cache, PagedKVCacheManager)
            for pending_list in kv_cache._block_manager._pending_transfers:
                for pending in list(pending_list):
                    pending.event.synchronize()
            kv_cache.poll_transfers()
            batch_info = create_batch_and_execute(scheduler)
            poll_iters += 1
        batch_infos.append(batch_info)
        if batch_info.batch_size == 0:
            break
    return batch_infos


def enqueue_request(
    queue: MAXPushQueue[TextContext],
    prompt_len: int,
    max_seq_len: int,
    shared_prefix: np.ndarray | None = None,
) -> None:
    context = create_text_context(
        prompt_len=prompt_len,
        max_seq_len=max_seq_len,
        shared_prefix=shared_prefix,
    )
    assert context.tokens.active_length == prompt_len
    queue.put_nowait(context)


def enqueue_request_with_prompt(
    queue: MAXPushQueue[TextContext],
    tokens: np.ndarray,
    max_seq_len: int,
) -> None:
    context = TextContext(
        request_id=RequestID(),
        max_length=max_seq_len,
        tokens=TokenBuffer(tokens),
    )

    queue.put_nowait(context)


CE = BatchType.CE
TG = BatchType.TG
