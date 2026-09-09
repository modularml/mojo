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

import queue
from unittest.mock import Mock

import numpy as np
import pytest
from max.pipelines.context import (
    TextAndVisionContext,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.kv_cache import DummyKVCache
from max.pipelines.modeling.types import RequestID, TextGenerationInputs
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler.config import TokenGenerationSchedulerConfig
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
)
from max.serve.scheduler_result import SchedulerResult
from max.support.math import ceildiv

ARBITRARY_TOKEN_ID = 999


def create_mock_pipeline() -> Mock:
    def next_token_behavior(
        inputs: TextGenerationInputs[TextContext],
    ) -> dict[RequestID, TextGenerationOutput]:
        responses: dict[RequestID, TextGenerationOutput] = {}

        for request in inputs.flat_batch:
            request_id = request.request_id

            request.update(42)

            # Return a valid response.
            output = request.to_generation_output()
            if output.tokens:
                responses[request_id] = output

        return responses

    pipeline = Mock()
    pipeline.execute = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    pipeline._pipeline_model = Mock(_lora_manager=None)
    pipeline.extra_kv_managers = []
    return pipeline


def create_scheduler(
    dp: int = 1,
    max_batch_size: int = 4,
    target_tokens_per_batch_ce: int = 32,
    enable_chunked_prefill: bool = False,
) -> tuple[
    TokenGenerationScheduler,
    MAXPushQueue[TextContext | TextAndVisionContext],
    MAXPullQueue[dict[RequestID, SchedulerResult[TextGenerationOutput]]],
    MAXPushQueue[list[RequestID]],
]:
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=max_batch_size,
        target_tokens_per_batch_ce=target_tokens_per_batch_ce,
        data_parallel_degree=dp,
        enable_chunked_prefill=enable_chunked_prefill,
    )

    request_queue: queue.Queue[TextContext | TextAndVisionContext] = (
        queue.Queue()
    )
    response_queue: queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ] = queue.Queue()
    cancel_queue: queue.Queue[list[RequestID]] = queue.Queue()

    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=create_mock_pipeline(),
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        kv_cache=DummyKVCache(),
    )

    return (
        scheduler,
        request_queue,
        response_queue,
        cancel_queue,
    )


def create_mock_request(
    seq_len: int = 30,
    start_idx: int = 0,
    is_tg: bool = False,
) -> TextContext:
    tokens = np.ones(seq_len, dtype=np.int64)
    assert len(tokens) == seq_len
    context = TextContext(
        request_id=RequestID(),
        max_length=100,
        tokens=TokenBuffer(tokens),
    )
    assert context.tokens.current_position == seq_len
    context.tokens.skip_processing(start_idx)
    if is_tg:
        context.update(ARBITRARY_TOKEN_ID)
    return context


def test_try_create_ce_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler()

    mock_request = create_mock_request()
    request_push_socket.put_nowait(mock_request)
    scheduler._retrieve_pending_requests()

    batch = scheduler.batch_constructor.construct_batch().flat_batch
    assert len(batch) == 1
    assert batch[0].request_id == mock_request.request_id
    # Cache management is now handled by the kv_cache/pipeline
    assert batch[0] is not None


def test_try_create_chunked_ce_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler(
        enable_chunked_prefill=True, target_tokens_per_batch_ce=20
    )
    mock_data = create_mock_request(seq_len=30)
    request_push_socket.put_nowait(mock_data)
    scheduler._retrieve_pending_requests()

    batch = scheduler.batch_constructor.construct_batch().flat_batch
    assert len(batch) == 1
    assert batch[0].request_id == mock_data.request_id
    # Cache management is now handled by the kv_cache/pipeline
    assert batch[0] is not None
    assert batch[0].tokens.current_position == 20
    assert batch[0].tokens.active_length == 20


def test_scheduler_handle_terminated_responses() -> None:
    scheduler, _, _, _ = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    mock_1 = create_mock_request()
    mock_2 = create_mock_request()
    batch_constructor.enqueue_new_request(mock_1)
    batch_constructor.enqueue_new_request(mock_2)
    mock_1.update(ARBITRARY_TOKEN_ID)
    mock_2.update(ARBITRARY_TOKEN_ID)

    resp_1: TextGenerationOutput = Mock(is_done=False, tokens=[Mock()])
    resp_2: TextGenerationOutput = Mock(is_done=True, tokens=[])
    batch_responses = {
        mock_1.request_id: resp_1,
        mock_2.request_id: resp_2,
    }

    batch_constructor.advance_requests(
        TextGenerationInputs(batches=[[mock_1, mock_2]])
    )

    # Release terminated requests
    num_terminated_reqs = 0
    for request_id, response in batch_responses.items():
        if response.is_done:
            batch_constructor.release_request(request_id)
            num_terminated_reqs += 1

    assert num_terminated_reqs == 1
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.release.assert_called_once()


def test_scheduler_handle_chunked_requests() -> None:
    scheduler, _, _, _ = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    req_1 = create_mock_request(is_tg=True)
    # this is a partially encoded request
    req_2 = create_mock_request(seq_len=30, start_idx=20)

    mock_1: TextGenerationOutput = Mock(is_done=False, tokens=[Mock()])
    batch_responses = {req_1.request_id: mock_1}

    batch_constructor.enqueue_new_request(req_1)
    batch_constructor.enqueue_new_request(req_2)

    batch_constructor.advance_requests(
        TextGenerationInputs(batches=[[req_1, req_2]])
    )
    assert req_2.request_id not in batch_responses
    assert batch_constructor.all_ce_reqs


def test_handle_cancelled_requests() -> None:
    scheduler, _, _, cancel_push_socket = create_scheduler()
    batch_constructor = scheduler.batch_constructor

    mock_request = create_mock_request(is_tg=True)
    batch_constructor.enqueue_new_request(mock_request)

    cancel_push_socket.put_nowait([mock_request.request_id])

    batch_constructor.release_request(mock_request.request_id)

    assert len(batch_constructor.all_tg_reqs) == 0
    # Cache cleanup is now handled by pipeline.release()
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.release.assert_called_once_with(mock_request.request_id)


def test_schedule_ce() -> None:
    scheduler, _, _, _ = create_scheduler()

    mock_request = create_mock_request()
    scheduler.batch_constructor.enqueue_new_request(mock_request)

    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[mock_request]],
    )

    scheduler._schedule(inputs)

    assert mock_request.request_id in scheduler.batch_constructor.all_tg_reqs
    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.execute.assert_called_once()


def test_schedule_ce_with_chunked_prefill() -> None:
    scheduler, request_push_socket, response_pull_socket, _ = create_scheduler(
        enable_chunked_prefill=True,
        target_tokens_per_batch_ce=20,
    )
    batch_constructor = scheduler.batch_constructor
    mock_request = create_mock_request(seq_len=30)

    request_push_socket.put_nowait(mock_request)
    scheduler._retrieve_pending_requests()
    assert len(batch_constructor.all_ce_reqs) == 1

    batch_to_execute = batch_constructor.construct_batch().batches[0]
    assert len(batch_to_execute) > 0

    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[batch_to_execute],
    )

    scheduler._schedule(inputs)

    assert mock_request.request_id not in batch_constructor.all_tg_reqs

    # Assert that the response socket is not empty.
    with pytest.raises(queue.Empty):
        x = response_pull_socket.get_nowait()
        print(f"There should be no response but mysteriously got: {x}")

    # check req1 is put back in the request queue with the correct current_position and active_length
    assert batch_constructor.all_ce_reqs
    req_id, data = list(batch_constructor.all_ce_reqs.items())[-1]
    assert req_id == mock_request.request_id
    assert data.tokens.processed_length == 20
    assert data.tokens.current_position == 30
    assert data.tokens.active_length == 10


def test_should_schedule_ce_empty_queue() -> None:
    scheduler, _, _, _ = create_scheduler()
    assert not scheduler.batch_constructor.construct_batch().flat_batch


def test_should_schedule_ce_full_batch() -> None:
    scheduler, request_push_socket, _, _ = create_scheduler()
    for _ in range(scheduler.scheduler_config.max_batch_size):
        mock_request = create_mock_request(is_tg=True)
        scheduler.batch_constructor.enqueue_new_request(mock_request)
    mock_request_ce = create_mock_request()
    request_push_socket.put_nowait(mock_request_ce)
    batch = scheduler.batch_constructor.construct_batch().flat_batch
    assert batch
    assert all(
        context.request_id != mock_request_ce.request_id for context in batch
    )


def test_schedule_tg() -> None:
    scheduler, _, _, _ = create_scheduler()

    mock_request = create_mock_request()
    inputs: TextGenerationInputs[TextContext] = TextGenerationInputs(
        batches=[[mock_request]],
    )

    scheduler._schedule(inputs)

    assert isinstance(scheduler.pipeline, Mock)
    scheduler.pipeline.execute.assert_called_once()


@pytest.mark.parametrize("dp", [1, 2, 15, 32])
def test_scheduler_dp(dp: int) -> None:
    """Check that DP works in basic cases."""
    scheduler, _, _, _ = create_scheduler(
        dp=dp,
        max_batch_size=512,
        target_tokens_per_batch_ce=8192,
    )
    batch_constructor = scheduler.batch_constructor

    num_reqs = 17
    for _ in range(num_reqs):
        batch_constructor.enqueue_new_request(create_mock_request())
    assert len(batch_constructor.all_ce_reqs) == num_reqs
    assert len(batch_constructor.all_tg_reqs) == 0

    inputs = batch_constructor.construct_batch()
    assert len(inputs.batches) == dp

    # The 17 requests are split among the X replicas.
    bs_high = ceildiv(num_reqs, dp)
    bs_low = num_reqs // dp
    num_bs_high = num_reqs % dp
    for i in range(dp):
        if i < num_bs_high:
            assert len(inputs.batches[i]) == bs_high
        else:
            assert len(inputs.batches[i]) == bs_low

    # Num steps is 1 since we are running CE

    # Generate new tokens for the requests.
    for batch in inputs.batches:
        for req in batch:
            req.update(ARBITRARY_TOKEN_ID)

    batch_constructor.advance_requests(inputs)
    assert len(batch_constructor.all_ce_reqs) == 0
    assert len(batch_constructor.all_tg_reqs) == num_reqs


def test_scheduler_empty_batch() -> None:
    """Check that we do not blow up when there are no requests."""
    scheduler, _, _, _ = create_scheduler(dp=100)
    inputs = scheduler.batch_constructor.construct_batch()
    assert len(inputs.batches) == 100
    assert len(inputs.flat_batch) == 0


def _create_lora_scheduler(adapter_name: str) -> TokenGenerationScheduler:
    """Create a scheduler with LoRA support for testing."""
    pipeline = Mock()
    pipeline.release = Mock()
    lora_manager = Mock()
    lora_manager.is_lora = Mock(side_effect=lambda n: n == adapter_name)
    lora_manager.is_active_lora = Mock(return_value=False)
    lora_manager.activate_adapter = Mock()
    lora_manager.max_num_loras = 4
    pipeline._pipeline_model = Mock(_lora_manager=lora_manager)
    pipeline.extra_kv_managers = []

    return TokenGenerationScheduler(
        scheduler_config=TokenGenerationSchedulerConfig(
            max_batch_size=4,
            target_tokens_per_batch_ce=32,
            data_parallel_degree=1,
        ),
        pipeline=pipeline,
        request_queue=queue.Queue(),
        response_queue=queue.Queue(),
        cancel_queue=queue.Queue(),
        kv_cache=DummyKVCache(),
    )


def test_lora_cleanup_on_release() -> None:
    """Test LoRA cleanup: stays active with remaining requests, removed when last released."""
    scheduler = _create_lora_scheduler("test-adapter")
    bc = scheduler.batch_constructor

    # Create two requests using the same LoRA adapter
    req1, req2 = create_mock_request(), create_mock_request()
    req1.model_name = req2.model_name = "test-adapter"
    bc.enqueue_new_request(req1)
    bc.enqueue_new_request(req2)
    bc.construct_batch()  # Activates LoRA

    replica = bc.replicas[0]
    assert "test-adapter" in replica.active_loras

    # Release first - LoRA stays active
    bc.release_request(req1.request_id)
    assert "test-adapter" in replica.active_loras

    # Release second - LoRA removed
    bc.release_request(req2.request_id)
    assert "test-adapter" not in replica.active_loras


def test_deferred_lora_request_cleanup() -> None:
    """Test that deferred LoRA requests are properly cleaned up on release."""
    scheduler = _create_lora_scheduler("deferred-adapter")
    bc = scheduler.batch_constructor

    req = create_mock_request()
    req.model_name = "deferred-adapter"
    bc.enqueue_new_request(req)

    # Simulate deferral by moving request to deferred_lora_requests
    replica = bc.replicas[0]
    del replica.ce_reqs[req.request_id]
    replica.deferred_lora_requests[req.request_id] = req

    bc.release_request(req.request_id)

    assert req.request_id not in replica.deferred_lora_requests
