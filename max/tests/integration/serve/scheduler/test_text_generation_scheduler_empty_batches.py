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

from max.pipelines.context import TextContext, TextGenerationOutput
from max.pipelines.kv_cache import DummyKVCache
from max.pipelines.modeling.types import (
    RequestID,
    TextGenerationInputs,
)
from max.serve.scheduler.base import SchedulerProgress
from max.serve.scheduler.text_generation_scheduler import (
    TokenGenerationScheduler,
    TokenGenerationSchedulerConfig,
)
from max.serve.scheduler_result import SchedulerResult


def create_mock_pipeline() -> Mock:
    pipeline = Mock()
    pipeline.execute = Mock(side_effect=next_token_behavior)
    pipeline.release = Mock()
    return pipeline


def next_token_behavior(
    inputs: TextGenerationInputs[TextContext],
) -> dict[RequestID, TextGenerationOutput]:
    assert len(inputs.flat_batch) == 0
    responses: dict[RequestID, TextGenerationOutput] = {}
    return responses


def test_text_generation_scheduler__empty_batches() -> None:
    """Test that the TokenGenerationScheduler supports empty batches during iteration.

    This test validates that the text generation scheduler correctly handles iterations
    where no requests are queued and empty batches are passed to the underlying pipeline.
    The test creates a scheduler with support_empty_batches=True and runs 10 iterations
    without any queued requests.

    The test verifies that:
    1. The scheduler can successfully run an iteration with an empty batch
    2. The underlying pipeline receives a batch of length 0
    3. The scheduler reports MADE_PROGRESS status even with no requests
    4. No responses are generated (response queue remains empty)

    Note: This is a manual test that requires explicit invocation.

    To run this test:
    ```
    bazel test //max/tests/integration/serve/scheduler:test_empty_batches --test_output=all
    ```
    """
    print("Creating scheduler")
    scheduler_config = TokenGenerationSchedulerConfig(
        max_batch_size=4,
        target_tokens_per_batch_ce=32,
    )
    request_queue = queue.Queue[TextContext]()
    response_queue = queue.Queue[
        dict[RequestID, SchedulerResult[TextGenerationOutput]]
    ]()
    cancel_queue = queue.Queue[list[RequestID]]()
    scheduler = TokenGenerationScheduler(
        scheduler_config=scheduler_config,
        pipeline=create_mock_pipeline(),
        request_queue=request_queue,
        response_queue=response_queue,
        cancel_queue=cancel_queue,
        support_empty_batches=True,
        kv_cache=DummyKVCache(),
    )

    print("Running iteration")
    for i in range(10):
        status = scheduler.run_iteration()
        print(f"Iteration {i}: {status}")

        assert status == SchedulerProgress.MADE_PROGRESS

        # We should not have sent any responses.
        assert response_queue.empty()
