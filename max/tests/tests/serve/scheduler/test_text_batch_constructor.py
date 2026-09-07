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

import time
from collections.abc import Mapping, Sequence
from unittest.mock import Mock, patch

import numpy as np
import pytest
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache import InsufficientBlocksError
from max.pipelines.kv_cache.kv_connector import (
    BlockCount,
    CompletedTransfer,
    TransferDirection,
)
from max.pipelines.kv_cache.paged_kv_cache import PrefixCacheHits
from max.pipelines.modeling.types import (
    Pipeline,
    RequestID,
    TextGenerationInputs,
)
from max.serve.scheduler.batch_constructor.text_batch_constructor import (
    TextBatchConstructor,
)
from max.serve.scheduler.batch_constructor.token_budget import RequestType
from max.serve.scheduler.config import TokenGenerationSchedulerConfig

ARBITRARY_TOKEN_ID = 999


@pytest.fixture
def pipeline() -> Pipeline[
    TextGenerationInputs[TextContext], TextGenerationOutput
]:
    pipeline = Mock()
    pipeline.release = Mock()
    return pipeline


def create_mock_lora_manager(max_num_loras: int = 2) -> Mock:
    """Create a mock LoRA manager for testing."""
    manager = Mock()
    manager.max_num_loras = max_num_loras
    active_loras: set[str] = set()
    all_loras: set[str] = set()
    manager._active_loras = active_loras
    manager._all_loras = all_loras

    def is_lora(model_name: str | None) -> bool:
        return bool(model_name and model_name.startswith("lora_"))

    def is_active_lora(model_name: str | None) -> bool:
        return model_name in manager._active_loras if model_name else False

    def activate_adapter(model_name: str) -> None:
        if len(manager._active_loras) >= max_num_loras:
            raise RuntimeError("Cannot activate more LoRAs than max_num_loras")
        manager._active_loras.add(model_name)
        manager._all_loras.add(model_name)

    manager.is_lora = Mock(side_effect=is_lora)
    manager.is_active_lora = Mock(side_effect=is_active_lora)
    manager.activate_adapter = Mock(side_effect=activate_adapter)

    return manager


def set_mock_kv_usage(cache: Mock, used_fraction: float) -> None:
    """Point a mock cache's device block count at a given usage fraction.

    ``_identify_priority`` reads ``block_count().used_pct``, so a mock that
    reaches it needs real numbers rather than an auto-created ``Mock``.
    """
    used = round(used_fraction * 100)
    cache.block_count = Mock(
        return_value=BlockCount(free=100 - used, total=100)
    )


def create_mock_kv_cache() -> Mock:
    """Create a mock paged KV cache manager with minimal interface."""
    cache = Mock()
    cache.max_seq_len = 2048
    cache.page_size = 16
    cache.get_total_num_pages = Mock(return_value=128)
    cache.get_free_blocks_pct = Mock(return_value=0.5)

    cache.alloc = Mock(return_value=CompletedTransfer.load())
    cache.claim = Mock()
    cache.release = Mock()
    cache.contains = Mock(return_value=False)
    cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(cache, 0.0)

    return cache


def create_mock_pipeline_with_lora(lora_manager: Mock) -> Mock:
    """Create a mock pipeline with LoRA support."""

    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request in inputs.flat_batch:
            request_id = request.request_id
            request.update(0)

            responses[request_id] = TextGenerationOutput(
                request_id=request_id,
                tokens=[0, 0],
                final_status=GenerationStatus.ACTIVE,
                log_probabilities=None,
            )

        return responses

    pipeline = Mock()
    pipeline.execute = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    pipeline._pipeline_model = Mock()
    pipeline._pipeline_model._lora_manager = lora_manager

    return pipeline


def create_lora_context(
    seq_len: int = 30, model_name: str | None = None, is_tg: bool = False
) -> TextContext:
    """Create a TextContext with optional LoRA model name."""
    tokens = np.ones(seq_len, dtype=np.int64)
    context = TextContext(
        request_id=RequestID(),
        max_length=100,
        tokens=TokenBuffer(tokens),
    )
    if model_name:
        context.model_name = model_name
    if is_tg:
        context.update(ARBITRARY_TOKEN_ID)
    return context


def has_request(batch: list[TextContext], request_id: RequestID) -> bool:
    return any(ctx.request_id == request_id for ctx in batch)


def test_text_batch_constructor__batch_construction_without_chunked_prefill_no_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    kv_cache = Mock()
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer.load()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue 6 CE requests, at 9 tokens each
    # Each have plenty of room for max length
    contexts = {}
    for _ in range(6):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        contexts[context.request_id] = context
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    # 9 * 4 = 36 tokens, since no max_batch_total_tokens is set, we should have 4 requests in the batch
    assert len(inputs.batches[0]) == 4
    # since this is CE, we should have 1 step

    # test that we have 2 requests remaining in the queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    # test that 2 of the requests finished
    request_ids = list(contexts.keys())
    responses = {
        request_ids[0]: TextGenerationOutput(
            request_id=request_ids[0],
            final_status=GenerationStatus.END_OF_SEQUENCE,
            tokens=[0],
        ),
        request_ids[1]: TextGenerationOutput(
            request_id=request_ids[1],
            final_status=GenerationStatus.ACTIVE,
            tokens=[1],
        ),
        request_ids[2]: TextGenerationOutput(
            request_id=request_ids[2],
            final_status=GenerationStatus.END_OF_SEQUENCE,
            tokens=[2],
        ),
    }

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    for request_id, response in responses.items():
        if response.is_done:
            batch_constructor.release_request(request_id)

    # 4 completed CE, 2 were completed, and 2 moved to TG
    assert len(batch_constructor.replicas[0].tg_reqs) == 2
    # There are 2 requests remaining in the CE queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    assert batch_constructor._identify_priority(0) == RequestType.CE

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 2

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 0
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    # Assume that we have 4 requests remaining in the queue
    # And none of the requests have a max length, therefore we use the default
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4


def test_text_batch_constructor__batch_construction_no_requests(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )

    kv_cache = Mock()
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer.load()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches) == 1
    assert len(inputs.batches[0]) == 0


def test_text_batch_constructor__structured_output_enabled_mirrors_bitmask_constraints(
    pipeline: Mock,
) -> None:
    """``structured_output_enabled`` forwards
    ``PipelineConfig.needs_bitmask_constraints`` -- on for either
    ``--enable-structured-output`` or a grammar-capable tool parser with
    ``--enable-tool-call-constrained-decode`` -- and stays off if the
    pipeline exposes no ``pipeline_config`` at all."""
    batch_constructor = TextBatchConstructor(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=5,
            max_batch_total_tokens=None,
            enable_in_flight_batching=False,
            enable_chunked_prefill=False,
            target_tokens_per_batch_ce=30,
        ),
        pipeline=pipeline,
        kv_cache=Mock(),
    )

    pipeline.pipeline_config = Mock(needs_bitmask_constraints=True)
    assert batch_constructor.structured_output_enabled is True

    pipeline.pipeline_config.needs_bitmask_constraints = False
    assert batch_constructor.structured_output_enabled is False

    del pipeline.pipeline_config
    assert batch_constructor.structured_output_enabled is False


def test_text_batch_constructor__batch_construction_no_room_in_cache(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock(side_effect=InsufficientBlocksError)
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.get_req_blocks = Mock(return_value=[])
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(2):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # With no TG, no active batch, and no in-flight KV transfers, there is
    # nothing that will free blocks — InsufficientBlocksError propagates.
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def test_text_batch_constructor__insufficient_blocks_defers_then_retries(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A CE request deferred by InsufficientBlocksError is admitted once
    blocks free up: something is in flight, so the first call defers
    (presence, not magnitude -- it doesn't matter whether that in-flight
    signal is "enough"); the second call's alloc succeeds outright."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock(
        side_effect=[
            InsufficientBlocksError("insufficient blocks"),
            CompletedTransfer.load(),
            CompletedTransfer.load(),
        ]
    )
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: 1,
    )

    for _ in range(2):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # First call: alloc fails, but something is in flight → defer.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert len(batch_constructor.replicas[0].ce_reqs) == 2

    # Second call: alloc succeeds → both requests admitted.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 2
    assert len(batch_constructor.replicas[0].ce_reqs) == 0


def _presence_test_setup(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    *,
    inflight_count: int = 0,
    is_tg: bool = False,
) -> TextBatchConstructor:
    """Builds a TextBatchConstructor with one request (CE, or TG when
    is_tg) whose alloc always fails, backed by an in-flight tracker
    reporting the given count."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock(
        side_effect=InsufficientBlocksError("insufficient blocks")
    )
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.get_req_blocks = Mock(return_value=[])
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: inflight_count,
    )
    context = (
        create_lora_context(seq_len=9, is_tg=True)
        if is_tg
        else TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
    )
    batch_constructor.enqueue_new_request(context)
    return batch_constructor


def test_text_batch_constructor__ce_insufficient_blocks_fatal_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work and nothing in flight -- genuinely stuck, must raise."""
    batch_constructor = _presence_test_setup(pipeline)
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def _fatal_test_batch_constructor(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
    *,
    inflight_count: int = 0,
    pending_transfers_exist: bool = False,
) -> TextBatchConstructor:
    """Builds a bare TextBatchConstructor for calling
    _is_insufficient_blocks_fatal directly, without going through
    construct_batch()."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.pending_transfers_exist = Mock(
        return_value=pending_transfers_exist
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: inflight_count,
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_with_other_work(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """With other runnable work present, never fatal regardless of what's
    in flight."""
    batch_constructor = _fatal_test_batch_constructor(pipeline)
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=False
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_true_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work and nothing in flight -- genuinely stuck, fatal."""
    batch_constructor = _fatal_test_batch_constructor(pipeline)
    assert batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_when_anything_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """No other work, but something is in flight -- not fatal, no matter
    how little: a single in-flight transfer might still resolve, and
    sizing "enough" against a snapshot would only ever undercount what
    could actually come back."""
    batch_constructor = _fatal_test_batch_constructor(
        pipeline, inflight_count=1
    )
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__is_insufficient_blocks_fatal_false_when_pending_transfer_exists(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A pending device-side transfer (e.g. a D2H offload) alone is enough
    to defer, even with no external in-flight signal."""
    batch_constructor = _fatal_test_batch_constructor(
        pipeline, pending_transfers_exist=True
    )
    assert not batch_constructor._is_insufficient_blocks_fatal(
        replica_idx=0, no_other_work=True
    )


def test_text_batch_constructor__tg_insufficient_blocks_fatal_when_nothing_inflight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """Same presence check on the TG path: raises once candidate_ids is
    exhausted and nothing at all is in flight."""
    batch_constructor = _presence_test_setup(pipeline, is_tg=True)
    with pytest.raises(InsufficientBlocksError):
        batch_constructor.construct_batch()


def test_text_batch_constructor__tg_insufficient_blocks_preempts_ce_block_holder(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A decode request that can't get a block reclaims from a chunked
    prefill parked in ce_reqs (which keeps its blocks between chunks)
    instead of raising a fatal InsufficientBlocksError: the parked
    prefill's blocks are reclaimable, so this is not a genuine OOM."""
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    holder = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(50, dtype=np.int64)),
        max_length=100,
    )
    released: list[RequestID] = []

    def release(ctx: TextContext) -> None:
        released.append(ctx.request_id)

    def alloc(ctx: TextContext) -> CompletedTransfer:
        # Blocks free up only once the parked prefill is preempted.
        if not released:
            raise InsufficientBlocksError("insufficient blocks")
        return CompletedTransfer.load()

    kv_cache = Mock()
    kv_cache.alloc = Mock(side_effect=alloc)
    kv_cache.claim = Mock()
    kv_cache.release = Mock(side_effect=release)
    kv_cache.contains = Mock(
        side_effect=lambda ctx: (
            ctx.request_id == holder.request_id and not released
        )
    )
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    kv_cache.get_req_blocks = Mock(
        side_effect=lambda ctx: (
            [0] if ctx.request_id == holder.request_id else []
        )
    )
    # A full cache forces TG priority, so the parked prefill is never
    # popped by _add_ce_requests before the decode request's alloc fails.
    set_mock_kv_usage(kv_cache, 1.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        get_inflight_kv_transfer_count=lambda replica_idx: 0,
    )
    tg_context = create_lora_context(seq_len=9, is_tg=True)
    batch_constructor.enqueue_new_request(tg_context, replica_idx=0)
    batch_constructor.enqueue_new_request(holder, replica_idx=0)

    inputs = batch_constructor.construct_batch()

    assert has_request(inputs.batches[0], tg_context.request_id)
    assert released == [holder.request_id]
    assert batch_constructor.total_preemption_count == 1
    # The preempted prefill is requeued, not dropped.
    assert holder.request_id in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__batch_construction_with_chunked_prefill_and_preemption(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer.load()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    contexts = {}
    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        contexts[context.request_id] = context
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    # The last request should be chunked
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should still be prioritizing CE
    assert batch_constructor._identify_priority(0) == RequestType.CE

    inputs = batch_constructor.construct_batch()
    # We only grab 2 new CE requests here, because we have 3 TG requests outstanding.
    # Since max_batch_size is 5, we can only have 5 requests outstanding at a time.
    assert len(inputs.batches[0]) == 2
    assert inputs.batches[0][-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    # We still prioritize CE, but return an empty batch
    assert batch_constructor._identify_priority(0) == RequestType.CE

    # Since we generate an empty CE batch, we then fill with TG requests
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 5

    # Last Ce Batch
    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    # Since we already have 5 CE request outstanding, we cannot grab any new CE requests.
    assert len(inputs.batches[0]) == 5

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 5

    # Test for Pre-emption
    # The first item won't have enough space, so we will pre-empt the last one
    # The first item will have 2 alloc calls, failing with InsufficientBlocksError on the first,
    # then succeeding and returning 0 (no prefix cache skip) for the remaining calls.
    kv_cache.alloc.side_effect = [
        InsufficientBlocksError(),
        CompletedTransfer.load(),
        CompletedTransfer.load(),
        CompletedTransfer.load(),
        CompletedTransfer.load(),
        CompletedTransfer.load(),
    ]

    last_request_id = list(batch_constructor.replicas[0].tg_reqs.keys())[-1]
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert all(
        context.request_id != last_request_id for context in inputs.batches[0]
    )

    # We've pre-empted the last request, so it should be in the CE queue
    assert len(batch_constructor.replicas[0].ce_reqs) == 4
    assert last_request_id in batch_constructor.replicas[0].ce_reqs
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    # Test that we can release the request
    batch_constructor.release_request(last_request_id)
    assert last_request_id not in batch_constructor.replicas[0].ce_reqs
    assert last_request_id not in batch_constructor.replicas[0].tg_reqs
    assert len(batch_constructor.replicas[0].ce_reqs) == 3
    assert len(batch_constructor.replicas[0].tg_reqs) == 4


def test_text_batch_constructor__batch_construction_with_chunked_prefill_and_inflight_batching(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        max_batch_total_tokens=None,
        enable_in_flight_batching=True,
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer.load()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # With inflight batching, we should prioritize CE ONLY when we have no TG requests
    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    # There should now be 3 requests in TG, and 7 in CE
    assert len(batch_constructor.replicas[0].tg_reqs) == 3
    assert len(batch_constructor.replicas[0].ce_reqs) == 5

    # We should now prioritize TG
    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()

    # We should have 5 requests
    assert len(inputs.batches[0]) == 7
    # Last item should be chunked, with a length of 3
    assert inputs.batches[0][-1].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)


def test_text_batch_constructor__batch_construction_without_chunked_prefill_and_inflight_batching(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        max_batch_total_tokens=None,
        enable_in_flight_batching=True,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )
    kv_cache = Mock()
    kv_cache.alloc = Mock()
    kv_cache.alloc.return_value = CompletedTransfer.load()
    kv_cache.claim = Mock()
    kv_cache.contains = Mock()
    set_mock_kv_usage(kv_cache, 0.0)

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    for _ in range(8):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    assert batch_constructor._identify_priority(0) == RequestType.CE
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 4
    assert inputs.batches[0][-1].tokens.generated_length == 0

    # Update a token for each request in the batch
    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 4
    assert len(batch_constructor.replicas[0].tg_reqs) == 4

    assert batch_constructor._identify_priority(0) == RequestType.TG
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 7
    for i in range(len(inputs.batches[0])):
        if i < 4:
            # The first four requests are TG, and should not need CE
            assert inputs.batches[0][i].tokens.generated_length != 0
        else:
            # The second four requests are CE, and should need CE
            assert inputs.batches[0][i].tokens.generated_length == 0

    for batch in inputs.batches:
        for context in batch:
            context.update(0)

    batch_constructor.advance_requests(inputs)

    assert len(batch_constructor.replicas[0].ce_reqs) == 1


def test_single_lora_scheduling() -> None:
    """Test scheduling a single LoRA request in CE batch."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.enqueue_new_request(ctx)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 1
    assert has_request(output.batches[0], ctx.request_id)
    lora_manager.activate_adapter.assert_called_once_with("lora_model1")
    assert "lora_model1" in lora_manager._active_loras


def test_multi_lora_within_budget() -> None:
    """Test scheduling multiple LoRA requests within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx1 = create_lora_context(model_name="lora_model1")
    ctx2 = create_lora_context(model_name="lora_model2")
    ctx3 = create_lora_context(model_name="lora_model3")

    batch_constructor.enqueue_new_request(ctx1)
    batch_constructor.enqueue_new_request(ctx2)
    batch_constructor.enqueue_new_request(ctx3)

    output = batch_constructor.construct_batch()
    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx1.request_id)
    assert has_request(output.batches[0], ctx2.request_id)
    assert has_request(output.batches[0], ctx3.request_id)
    assert len(lora_manager._active_loras) == 3


def test_lora_preemption_over_budget() -> None:
    """Test that LoRA requests are deferred when over budget during CE."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        target_tokens_per_batch_ce=200,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")
    ctx_lora3 = create_lora_context(model_name="lora_model3")
    ctx_base = create_lora_context(model_name=None)

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)
    batch_constructor.enqueue_new_request(ctx_lora3)
    batch_constructor.enqueue_new_request(ctx_base)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx_base.request_id)
    assert has_request(output.batches[0], ctx_lora1.request_id)
    assert has_request(output.batches[0], ctx_lora2.request_id)
    assert ctx_lora3.request_id not in output.batches[0]

    assert ctx_lora3.request_id in batch_constructor.all_ce_reqs


def test_age_based_scheduling_with_lora() -> None:
    """Test that age-based scheduling is maintained with LoRA constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=40,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model2")

    ctx_inactive = create_lora_context(model_name="lora_model1")
    ctx_base = create_lora_context(model_name=None)
    ctx_active = create_lora_context(model_name="lora_model2")

    batch_constructor.enqueue_new_request(ctx_inactive)
    batch_constructor.enqueue_new_request(ctx_base)
    batch_constructor.enqueue_new_request(ctx_active)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 2
    assert has_request(output.batches[0], ctx_inactive.request_id)
    assert has_request(output.batches[0], ctx_base.request_id)


def test_tg_batch_with_active_loras() -> None:
    """Test that TG batch correctly handles requests with active LoRAs."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=5,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    ctx_active1 = create_lora_context(model_name="lora_model1", is_tg=True)
    ctx_active2 = create_lora_context(model_name="lora_model2", is_tg=True)
    ctx_base = create_lora_context(model_name=None, is_tg=True)

    batch_constructor.enqueue_new_request(ctx_active1)
    batch_constructor.enqueue_new_request(ctx_active2)
    batch_constructor.enqueue_new_request(ctx_base)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0])
    assert has_request(output.batches[0], ctx_active1.request_id)
    assert has_request(output.batches[0], ctx_active2.request_id)
    assert has_request(output.batches[0], ctx_base.request_id)


def test_ce_lora_activation_within_budget() -> None:
    """Test that LoRAs are activated during CE when within budget."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 2
    assert has_request(output.batches[0], ctx_lora1.request_id)
    assert has_request(output.batches[0], ctx_lora2.request_id)

    assert "lora_model1" in lora_manager._active_loras
    assert "lora_model2" in lora_manager._active_loras


def test_tg_pure_age_based_preemption() -> None:
    """Test that preemption is purely age-based for KV cache constraints."""
    lora_manager = create_mock_lora_manager(max_num_loras=3)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    kv_cache.alloc = Mock(
        side_effect=[
            CompletedTransfer.load(),
            InsufficientBlocksError,
            InsufficientBlocksError,
        ]
    )

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    lora_manager._active_loras.add("lora_model1")
    lora_manager._active_loras.add("lora_model2")

    ctx1 = create_lora_context(model_name="lora_model1", is_tg=True)
    ctx2 = create_lora_context(model_name="lora_model2", is_tg=True)
    ctx3 = create_lora_context(model_name=None, is_tg=True)

    batch_constructor.enqueue_new_request(ctx1)
    batch_constructor.enqueue_new_request(ctx2)
    batch_constructor.enqueue_new_request(ctx3)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 1
    assert has_request(output.batches[0], ctx1.request_id)
    pipeline.release.assert_called()


def test_lora_swapping_ce_to_tg() -> None:
    """Test LoRA remains active when moving from CE to TG."""
    lora_manager = create_mock_lora_manager(max_num_loras=2)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx = create_lora_context(model_name="lora_model1")
    batch_constructor.enqueue_new_request(ctx)

    batch_constructor.construct_batch()
    assert "lora_model1" in lora_manager._active_loras

    ctx.update(29)
    batch_constructor.enqueue_new_request(ctx)

    ctx2 = create_lora_context(model_name="lora_model2")
    batch_constructor.enqueue_new_request(ctx2)

    batch_constructor.construct_batch()
    assert "lora_model2" in lora_manager._active_loras

    ctx2.update(29)
    batch_constructor.enqueue_new_request(ctx2)

    tg_output = batch_constructor.construct_batch()

    assert has_request(tg_output.batches[0], ctx.request_id)
    assert has_request(tg_output.batches[0], ctx2.request_id)


def test_mixed_requests_scheduling() -> None:
    """Test scheduling with mixed LoRA and base model requests."""
    lora_manager = create_mock_lora_manager(max_num_loras=1)
    pipeline = create_mock_pipeline_with_lora(lora_manager)
    kv_cache = create_mock_kv_cache()

    config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=100,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    ctx_lora1 = create_lora_context(model_name="lora_model1")
    ctx_lora2 = create_lora_context(model_name="lora_model2")
    ctx_base1 = create_lora_context(model_name=None)
    ctx_base2 = create_lora_context(model_name=None)

    batch_constructor.enqueue_new_request(ctx_lora1)
    batch_constructor.enqueue_new_request(ctx_lora2)
    batch_constructor.enqueue_new_request(ctx_base1)
    batch_constructor.enqueue_new_request(ctx_base2)

    output = batch_constructor.construct_batch()

    assert len(output.batches[0]) == 3
    assert has_request(output.batches[0], ctx_base1.request_id)
    assert has_request(output.batches[0], ctx_base2.request_id)
    assert has_request(output.batches[0], ctx_lora1.request_id) or (
        has_request(output.batches[0], ctx_lora2.request_id)
    )

    assert len(lora_manager._active_loras) == 1


def test_text_batch_constructor__load_based_replica_assignment_with_kv_cache() -> (
    None
):
    """Test that load-based assignment distributes requests evenly across replicas.

    This is the core test to catch bugs like [2,1,1,1,1,1,1,0] instead of [1,1,1,1,1,1,1,1].
    """
    data_parallel_degree = 8
    num_requests = 8

    # Create a pipeline without LoRA support
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue requests - with load-based assignment, all should go to least loaded
    for _ in range(num_requests):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # Count requests per replica
    requests_per_replica = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # With load-based assignment, distribution should be balanced
    # Each replica should have 1 request (8 requests, 8 replicas)
    expected_distribution = [1, 1, 1, 1, 1, 1, 1, 1]
    assert requests_per_replica == expected_distribution, (
        f"Expected distribution {expected_distribution}, got {requests_per_replica}"
    )


def test_text_batch_constructor__data_parallel_explicit_replica_assignment() -> (
    None
):
    """Test explicit replica_idx assignment used by decode_scheduler.

    This tests the code path where replica_idx is explicitly passed, ensuring
    requests go to the correct replica.
    """
    data_parallel_degree = 8

    # Create a pipeline without LoRA support
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache (required but not used for explicit assignment)
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree
    kv_cache.get_replica_request_count = Mock(return_value=0)

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Enqueue one request to each replica explicitly
    for replica_idx in range(data_parallel_degree):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context, replica_idx=replica_idx)

    # Count requests per replica
    requests_per_replica = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # Each replica should have exactly 1 request
    expected_distribution = [1, 1, 1, 1, 1, 1, 1, 1]
    assert requests_per_replica == expected_distribution, (
        f"Expected distribution {expected_distribution}, got {requests_per_replica}"
    )


def test_text_batch_constructor__load_based_handles_imbalance() -> None:
    """Test that load-based assignment prioritizes least loaded replicas.

    This test creates an imbalanced load scenario and verifies that new
    requests are assigned to the replica with the fewest active requests.
    """
    data_parallel_degree = 4

    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()

    # Create paged cache
    kv_cache = create_mock_kv_cache()
    kv_cache.num_replicas = data_parallel_degree

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=1000,
        data_parallel_degree=data_parallel_degree,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )

    # Create an imbalanced initial load: [5, 2, 8, 1]
    # Replica 0: 5 requests, Replica 1: 2 requests, Replica 2: 8 requests, Replica 3: 1 request
    for replica_idx, count in enumerate([5, 2, 8, 1]):
        for _ in range(count):
            context = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            batch_constructor.enqueue_new_request(
                context, replica_idx=replica_idx
            )

    # Track request counts before adding new requests
    requests_before = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]
    assert requests_before == [5, 2, 8, 1], (
        f"Initial load should be [5, 2, 8, 1], got {requests_before}"
    )

    # Enqueue 4 new requests without specifying replica_idx
    for _ in range(4):
        context = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(context)

    # Count requests per replica after adding new requests
    requests_after = [
        len(batch_constructor.replicas[i].ce_reqs)
        for i in range(data_parallel_degree)
    ]

    # Replica 3 had the lowest load (1), so it should receive the first new request → [5, 2, 8, 2]
    # Replica 1 now has the lowest load (2), so it should receive the second new request → [5, 3, 8, 2]
    # Replica 3 now tied for lowest (2), so it should receive the third new request → [5, 3, 8, 3]
    # Replica 1 now tied for lowest (3), so it should receive the fourth new request → [5, 4, 8, 3]
    assert requests_after == [5, 4, 8, 3]


def test_batch_scheduling_strategy__per_replica_default() -> None:
    """Test PER_REPLICA strategy (default) allows independent replica decisions."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PER_REPLICA,
    )

    # Replica 0: 2 CE requests (should prioritize CE)
    # Replica 1: 2 TG requests (should prioritize TG)
    # Replica 2: 1 CE + 1 TG (should prioritize CE with enable_in_flight_batching=False)

    # Add CE requests to replica 0
    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    # Add TG requests to replica 1
    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    # Add mixed requests to replica 2
    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Verify each replica identifies priority independently
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.TG
    assert batch_constructor._identify_priority(2) == RequestType.CE

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # Replica 0 should have CE batch
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    # Replica 1 should have TG batch
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    # Replica 2 should have CE batch (prioritizes CE when enable_in_flight_batching=False)
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length == 0


def test_batch_scheduling_strategy__prefill_first() -> None:
    """Test PREFILL_FIRST strategy forces all replicas to prioritize CE."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PREFILL_FIRST,
    )

    # Replica 0: 2 CE requests
    # Replica 1: 2 TG requests
    # Replica 2: 1 CE + 1 TG

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # All replicas should prioritize CE since PREFILL_FIRST and CE work exists
    # Replica 0: CE batch
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    # Replica 1: Should be empty or have TG (no CE requests)
    assert len(inputs.batches[1]) == 0

    # Replica 2: CE batch
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length == 0


def test_batch_scheduling_strategy__decode_first() -> None:
    """Test DECODE_FIRST strategy forces all replicas to prioritize TG."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
    )

    # Replica 0: 2 CE requests
    # Replica 1: 2 TG requests
    # Replica 2: 1 CE + 1 TG

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=1)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=2)

    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=2)

    # Construct batch
    inputs = batch_constructor.construct_batch()

    # All replicas should prioritize TG since DECODE_FIRST and TG work exists
    # Replica 0: Should be empty (no TG requests)
    assert len(inputs.batches[0]) == 0

    # Replica 1: TG batch
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    # Replica 2: TG batch (with possible CE fill due to enable_in_flight_batching)
    assert len(inputs.batches[2]) >= 1
    assert inputs.batches[2][0].tokens.generated_length > 0


def test_batch_scheduling_strategy__decode_first_idle_replica_does_not_starve_sibling() -> (
    None
):
    """Test an idle replica can't veto a sibling's real CE request (SERVOPT-1560).

    Regression test for a livelock: DECODE_FIRST computed a batch-wide
    priority override from the set of every replica's _identify_priority()
    result. A replica with zero CE *and* zero TG requests used to default to
    RequestType.TG (indistinguishable from a replica that genuinely needs
    TG), so its phantom vote alone could force priority_override=TG for
    every replica -- including a sibling with a real, ready CE request and
    zero TG requests, whose CE admission would then be skipped every single
    iteration forever, with no exception and no error log.
    """
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 2
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.DECODE_FIRST,
    )

    # Replica 0: completely idle (no CE, no TG requests).
    # Replica 1: one real, ready CE request, no TG requests -- matches the
    # exact captured state at max-concurrency=1: ce_reqs=[0, 1] tg_reqs=[0, 0].
    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=1)

    # An idle replica has no preference at all, not a TG preference.
    assert batch_constructor._identify_priority(0) is None
    assert batch_constructor._identify_priority(1) == RequestType.CE

    inputs = batch_constructor.construct_batch()

    # Replica 0 has nothing to do either way.
    assert len(inputs.batches[0]) == 0

    # Replica 1's CE request must be admitted -- it must not be starved by
    # replica 0's idleness being mistaken for a TG override.
    assert len(inputs.batches[1]) == 1
    assert inputs.batches[1][0].tokens.generated_length == 0


def test_ce_preempts_pending_tg_emits_iteration_stealing_metric() -> None:
    """A CE batch that admits work while TG requests are pending on the
    same replica steals the whole iteration from TG (the CE case's TG
    fallback only runs when CE admits nothing). Verify the
    iteration-stealing counter and gauge fire in that case.
    """
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=1,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.PREFILL_FIRST,
    )

    # One TG request already in flight, plus a fresh CE request --
    # PREFILL_FIRST forces CE priority for the whole replica, so CE wins
    # this iteration and the TG request gets skipped entirely.
    ctx_tg = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    ctx_tg.update(ARBITRARY_TOKEN_ID)
    batch_constructor.enqueue_new_request(ctx_tg, replica_idx=0)

    ctx_ce = TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
        max_length=100,
    )
    batch_constructor.enqueue_new_request(ctx_ce, replica_idx=0)

    with patch(
        "max.serve.scheduler.batch_constructor.text_batch_constructor.METRICS"
    ) as mock_metrics:
        inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 1
    assert inputs.batches[0][0].tokens.generated_length == 0
    mock_metrics.di_ce_preempted_tg_iteration_count.assert_called_once()
    mock_metrics.di_ce_preempted_tg_pending_count.assert_called_once_with(1)


def test_batch_scheduling_strategy__balanced_majority_ce() -> None:
    """Test BALANCED strategy prioritizes CE when CE is the majority."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=False,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority (2 CE requests)
    # Replica 1: CE priority (2 CE requests)
    # Replica 2: TG priority (2 TG requests)
    # Majority: CE (2 CE vs 1 TG)

    for replica_idx in [0, 1]:
        for _ in range(2):
            ctx = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=2)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.CE
    assert batch_constructor._identify_priority(2) == RequestType.TG

    # Construct batch - should prioritize CE globally
    inputs = batch_constructor.construct_batch()

    # Replicas 0 and 1 should have CE batches
    assert len(inputs.batches[0]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[0])

    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length == 0 for ctx in inputs.batches[1])

    # Replica 2 should be empty (forced to CE but has no CE requests)
    assert len(inputs.batches[2]) == 0


def test_batch_scheduling_strategy__balanced_majority_tg() -> None:
    """Test BALANCED strategy prioritizes TG when TG is the majority."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 3
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority (2 CE requests)
    # Replica 1: TG priority (2 TG requests)
    # Replica 2: TG priority (2 TG requests)
    # Majority: TG (1 CE vs 2 TG)

    for _ in range(2):
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=0)

    for replica_idx in [1, 2]:
        for _ in range(2):
            ctx = TextContext(
                request_id=RequestID(),
                tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
                max_length=100,
            )
            ctx.update(ARBITRARY_TOKEN_ID)
            batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.TG
    assert batch_constructor._identify_priority(2) == RequestType.TG

    # Construct batch - should prioritize TG globally
    inputs = batch_constructor.construct_batch()

    # Replica 0 should be empty (forced to TG but has no TG requests)
    assert len(inputs.batches[0]) == 0

    # Replicas 1 and 2 should have TG batches
    assert len(inputs.batches[1]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[1])

    assert len(inputs.batches[2]) == 2
    assert all(ctx.tokens.generated_length > 0 for ctx in inputs.batches[2])


def test_batch_scheduling_strategy__balanced_tie_defaults_to_tg() -> None:
    """Test BALANCED strategy defaults to TG when CE and TG counts are equal."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 4
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
        enable_in_flight_batching=True,
    )

    batch_constructor = TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
        batch_scheduling_strategy=BatchSchedulingStrategy.BALANCED,
    )

    # Replica 0: CE priority
    # Replica 1: CE priority
    # Replica 2: TG priority
    # Replica 3: TG priority
    # Tie: 2 CE vs 2 TG -> should default to TG

    for replica_idx in [0, 1]:
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    for replica_idx in [2, 3]:
        ctx = TextContext(
            request_id=RequestID(),
            tokens=TokenBuffer(np.ones(10, dtype=np.int64)),
            max_length=100,
        )
        ctx.update(ARBITRARY_TOKEN_ID)
        batch_constructor.enqueue_new_request(ctx, replica_idx=replica_idx)

    # Verify individual priorities
    assert batch_constructor._identify_priority(0) == RequestType.CE
    assert batch_constructor._identify_priority(1) == RequestType.CE
    assert batch_constructor._identify_priority(2) == RequestType.TG
    assert batch_constructor._identify_priority(3) == RequestType.TG

    # Construct batch - should default to TG on tie
    inputs = batch_constructor.construct_batch()

    # Replicas 0 and 1 should be empty (forced to TG but have no TG requests)
    assert len(inputs.batches[0]) == 0
    assert len(inputs.batches[1]) == 0

    # Replicas 2 and 3 should have TG batches
    assert len(inputs.batches[2]) == 1
    assert inputs.batches[2][0].tokens.generated_length > 0

    assert len(inputs.batches[3]) == 1
    assert inputs.batches[3][0].tokens.generated_length > 0


def test_batch_scheduling_strategy__all_replicas_empty() -> None:
    """Test that all strategies handle the case where all replicas are empty."""
    from max.serve.scheduler.batch_constructor.text_batch_constructor import (
        BatchSchedulingStrategy,
    )

    data_parallel_degree = 2
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()

    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=data_parallel_degree,
    )

    for strategy in [
        BatchSchedulingStrategy.PER_REPLICA,
        BatchSchedulingStrategy.PREFILL_FIRST,
        BatchSchedulingStrategy.DECODE_FIRST,
        BatchSchedulingStrategy.BALANCED,
    ]:
        batch_constructor = TextBatchConstructor(
            scheduler_config=scheduler_config,
            pipeline=pipeline,
            kv_cache=kv_cache,
            batch_scheduling_strategy=strategy,
        )

        inputs = batch_constructor.construct_batch()

        # All batches should be empty
        assert len(inputs.batches) == data_parallel_degree
        assert all(len(batch) == 0 for batch in inputs.batches)


# ---------------------------------------------------------------------------
# DP-balanced CE scheduling (_plan_ce_step) tests
# ---------------------------------------------------------------------------


def create_dp_balance_constructor(
    dp: int = 2,
    timeout_ms: float = 10_000.0,
    threshold: float = 0.8,
    enable_dynamic_chunk_size: bool = True,
    hit_counts: list[PrefixCacheHits] | None = None,
) -> TextBatchConstructor:
    """A DP constructor with the CE balancer on and a stubbed cache probe."""
    pipeline = Mock(spec=["release"])
    pipeline.release = Mock()
    kv_cache = create_mock_kv_cache()
    kv_cache.params.page_size = 16
    kv_cache.get_prefix_cache_hit_counts = Mock(
        return_value=(
            hit_counts if hit_counts is not None else [PrefixCacheHits()] * dp
        )
    )
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=10,
        target_tokens_per_batch_ce=100,
        data_parallel_degree=dp,
        dp_ce_balance_timeout_ms=timeout_ms,
        dp_ce_balance_threshold=threshold,
        dp_ce_balance_enable_dynamic_chunk_size=enable_dynamic_chunk_size,
    )
    return TextBatchConstructor(
        scheduler_config=scheduler_config,
        pipeline=pipeline,
        kv_cache=kv_cache,
    )


def test_dp_ce_balance__disabled_binds_on_arrival() -> None:
    """timeout_ms=-1 (default) disables pooling: arrival binds immediately."""
    batch_constructor = create_dp_balance_constructor(timeout_ms=-1.0)
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)
    assert not batch_constructor._ce_pending
    assert any(
        ctx.request_id in replica.ce_reqs
        for replica in batch_constructor.replicas
    )


def test_dp_ce_balance__pools_new_requests_and_binds_when_fleet_idle() -> None:
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)

    # Pooled: tracked by the constructor but bound to no replica queue.
    assert batch_constructor.contains(ctx.request_id)
    assert ctx.request_id in batch_constructor._ce_pending
    assert all(not replica.ce_reqs for replica in batch_constructor.replicas)

    # The fleet has nothing else to run, so the planner must not defer: the
    # request binds and is scheduled this very step.
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0] + inputs.batches[1], ctx.request_id)
    assert not batch_constructor._ce_pending


def test_dp_ce_balance__pooled_request_prefers_replica_with_cached_prefix() -> (
    None
):
    hit_counts = [PrefixCacheHits(), PrefixCacheHits(device_blocks=4)]
    batch_constructor = create_dp_balance_constructor(hit_counts=hit_counts)
    ctx = create_lora_context(seq_len=96)
    batch_constructor.enqueue_new_request(ctx)

    # Weighted at post-prefix-cache length: 96 tokens raw, minus 4 blocks
    # (x 16-token pages) resident on replica 1.
    assert batch_constructor._ce_pending[ctx.request_id].weights == [96, 32]

    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[1], ctx.request_id)


def test_dp_ce_balance__defers_lone_unexpired_ce_when_tg_available() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.8)
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    # Deadline budget left, as if the request had been pooled on arrival.
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()

    # Occupancy would be 50/(2*50) = 0.5 < 0.8 with no partner CE anywhere,
    # so replica 0's CE work is held and it runs TG instead.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == {0}
    assert has_request(inputs.batches[0], tg_ctx.request_id)
    assert not has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__expired_ce_runs_despite_imbalance() -> None:
    batch_constructor = create_dp_balance_constructor(timeout_ms=10_000.0)
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic() - 60.0

    # Same imbalance as the deferral test, but the deadline is blown: the CE
    # work joins the floor and runs.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__no_deferral_without_tg_work() -> None:
    batch_constructor = create_dp_balance_constructor()
    ce_ctx = create_lora_context(seq_len=50)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=0)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()

    # Replica 0 has no TG to run instead; deferring would idle it, so its CE
    # work is not deferrable even with deadline budget left.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctx.request_id)


def test_dp_ce_balance__balanced_ce_across_replicas_schedules() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.8)
    ce_ctxs = []
    for replica_idx in range(2):
        tg_ctx = create_lora_context(is_tg=True)
        batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
        ce_ctx = create_lora_context(seq_len=50)
        batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
        batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()
        ce_ctxs.append(ce_ctx)

    # 50 tokens on each rank is a perfectly balanced step: occupancy 1.0
    # meets the threshold and everything runs, nothing is deferred.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], ce_ctxs[0].request_id)
    assert has_request(inputs.batches[1], ce_ctxs[1].request_id)


def test_dp_ce_balance__release_pooled_request() -> None:
    batch_constructor = create_dp_balance_constructor()
    ctx = create_lora_context()
    batch_constructor.enqueue_new_request(ctx)
    assert batch_constructor.contains(ctx.request_id)

    # Releasing a pooled request (e.g. client cancellation) must work even
    # though it was never bound to a replica or claimed in the KV cache.
    batch_constructor.release_request(ctx.request_id)
    assert not batch_constructor.contains(ctx.request_id)
    assert isinstance(batch_constructor.pipeline, Mock)
    batch_constructor.pipeline.release.assert_called_once_with(ctx.request_id)

    inputs = batch_constructor.construct_batch()
    assert all(len(batch) == 0 for batch in inputs.batches)


def _add_deferrable_ce(
    batch_constructor: TextBatchConstructor, replica_idx: int, seq_len: int
) -> TextContext:
    """A TG request plus an unexpired mid-CE request pinned to a replica."""
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=replica_idx)
    ce_ctx = create_lora_context(seq_len=seq_len)
    batch_constructor.enqueue_new_request(ce_ctx, replica_idx=replica_idx)
    batch_constructor._ce_arrival[ce_ctx.request_id] = time.monotonic()
    return ce_ctx


def test_dp_ce_balance__reduces_chunk_size_to_balance_level() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.9)
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    light_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    # Occupancy (96+60)/(2*96) = 0.81 misses the 0.9 threshold, but both
    # replicas have CE work and the balance level (60) is at least half the
    # 100-token chunk target, so the step runs with a 60-token chunk size
    # per replica: the heavy request is chunked at the quota and only its
    # excess defers.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota == [60, 60]
    assert batch_constructor._ce_deferred_replicas == set()
    assert has_request(inputs.batches[0], heavy_ce.request_id)
    assert heavy_ce.tokens.active_length == 60
    assert has_request(inputs.batches[1], light_ce.request_id)


def test_dp_ce_balance__dynamic_chunk_size_disabled_holds_step() -> None:
    batch_constructor = create_dp_balance_constructor(
        threshold=0.9, enable_dynamic_chunk_size=False
    )
    heavy_ce = _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    light_ce = _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    # Same step as above, but with dynamic chunk sizing off it is held whole.
    inputs = batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == {0, 1}
    assert not has_request(inputs.batches[0], heavy_ce.request_id)
    assert not has_request(inputs.batches[1], light_ce.request_id)


def test_dp_ce_balance__no_chunk_size_reduction_below_half_target() -> None:
    batch_constructor = create_dp_balance_constructor()
    _add_deferrable_ce(batch_constructor, replica_idx=0, seq_len=96)
    _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=30)

    # The balance level (30) is under half the 100-token chunk target:
    # chunks that small cost more in extra steps than the imbalance they
    # avoid, so the work is held instead.
    batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota is None
    assert batch_constructor._ce_deferred_replicas == {0, 1}


def test_dp_ce_balance__quota_never_below_floor() -> None:
    batch_constructor = create_dp_balance_constructor(threshold=0.9)
    # Replica 0's CE deadline is blown: it is floor work that runs to the
    # full chunk budget, and the balance level cannot drop below it.
    tg_ctx = create_lora_context(is_tg=True)
    batch_constructor.enqueue_new_request(tg_ctx, replica_idx=0)
    expired_ce = create_lora_context(seq_len=96)
    batch_constructor.enqueue_new_request(expired_ce, replica_idx=0)
    batch_constructor._ce_arrival[expired_ce.request_id] = (
        time.monotonic() - 60.0
    )
    _add_deferrable_ce(batch_constructor, replica_idx=1, seq_len=60)

    batch_constructor.construct_batch()
    assert batch_constructor._ce_step_quota == [96, 60]
    assert batch_constructor._ce_deferred_replicas == set()


# ---------------------------------------------------------------------------
# Async KV onload cordon / re-admit (SERVOPT-1036) tests
# ---------------------------------------------------------------------------


class _IncompleteOnload:
    """A ``KVConnectorTransfer`` whose onload has not yet landed.

    Models the ``rust_tiered`` connector's async handle: ``is_complete`` stays
    ``False`` until the test flips it (via ``synchronize``), so the batch
    constructor cordons the request instead of scheduling it.
    """

    def __init__(self) -> None:
        self.complete = False

    @property
    def direction(self) -> TransferDirection:
        return TransferDirection.LOAD

    @property
    def g0_blocks_per_leaf(self) -> Mapping[str, Sequence[int]]:
        return {}

    def is_complete(self) -> bool:
        return self.complete

    def synchronize(self) -> None:
        self.complete = True


def _make_cordon_kv_cache(alloc: Mock) -> Mock:
    """A minimal KV cache mock for the cordon path with a custom ``alloc``."""
    kv_cache = Mock()
    kv_cache.alloc = alloc
    kv_cache.claim = Mock()
    kv_cache.contains = Mock(return_value=False)
    kv_cache.pending_transfers_exist = Mock(return_value=False)
    set_mock_kv_usage(kv_cache, 0.0)
    return kv_cache


def _ce_ctx() -> TextContext:
    return TextContext(
        request_id=RequestID(),
        tokens=TokenBuffer(np.ones(9, dtype=np.int64)),
        max_length=100,
    )


def _cordon_config() -> TokenGenerationSchedulerConfig:
    return TokenGenerationSchedulerConfig(
        max_batch_size=5,
        max_batch_total_tokens=None,
        enable_in_flight_batching=False,
        enable_chunked_prefill=False,
        target_tokens_per_batch_ce=30,
    )


def test_text_batch_constructor__cordons_request_with_incomplete_onload(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A CE request whose KV onload is still in flight is held out of the batch.

    The request is allocated (its blocks pinned) but cordoned: it must not be
    scheduled until the H2D lands, so the GPU can run other ready work meanwhile.
    """
    onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(Mock(return_value=onload))
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs
    assert ctx.request_id not in batch_constructor.replicas[0].ce_reqs


def test_text_batch_constructor__readmits_request_when_onload_completes(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """A cordoned request is re-admitted and scheduled once its onload lands."""
    onload = _IncompleteOnload()
    # First alloc cordons (incomplete); the re-admit pass re-allocs and the
    # prefix is now device-resident, so the second alloc completes immediately.
    kv_cache = _make_cordon_kv_cache(
        Mock(side_effect=[onload, CompletedTransfer.load()])
    )
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx)

    # Iteration 1: cordoned, empty batch.
    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches[0]) == 0
    assert ctx.request_id in batch_constructor._onloading_reqs

    # The onload lands; the next iteration re-admits and schedules it.
    onload.synchronize()
    inputs = batch_constructor.construct_batch()
    assert has_request(inputs.batches[0], ctx.request_id)
    assert ctx.request_id not in batch_constructor._onloading_reqs


def test_text_batch_constructor__oom_deferred_while_onload_in_flight(
    pipeline: Pipeline[TextGenerationInputs[TextContext], TextGenerationOutput],
) -> None:
    """OOM does not crash the server while a cordoned onload is in flight.

    The first request is cordoned (onload in flight); the second hits
    ``InsufficientBlocksError``. A's cordoned onload is enough on its own
    (presence, not magnitude) for B to defer rather than raise.
    """
    onload = _IncompleteOnload()
    kv_cache = _make_cordon_kv_cache(
        Mock(
            side_effect=[
                onload,
                InsufficientBlocksError("insufficient blocks"),
            ]
        )
    )
    kv_cache.get_req_blocks = Mock(return_value=[0, 1, 2, 3, 4])
    batch_constructor = TextBatchConstructor(
        scheduler_config=_cordon_config(),
        pipeline=pipeline,
        kv_cache=kv_cache,
    )
    ctx_a = _ce_ctx()
    ctx_b = _ce_ctx()
    batch_constructor.enqueue_new_request(ctx_a)
    batch_constructor.enqueue_new_request(ctx_b)

    # Must not raise: A cordons, B's OOM defers (A's cordoned blocks cover it).
    inputs = batch_constructor.construct_batch()

    assert len(inputs.batches[0]) == 0
    assert ctx_a.request_id in batch_constructor._onloading_reqs
    assert ctx_b.request_id in batch_constructor.replicas[0].ce_reqs
