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
"""OneShotScheduler for non-autoregressive pipelines.

This scheduler is designed for pipelines that process requests in a single pass
without requiring iterative generation (e.g., image generation, non-autoregressive
text models). It processes each request serially, making it simple and suitable
for workloads that don't benefit from batching or continuous generation.
"""

import logging
import queue
from collections.abc import Callable
from typing import Generic

from max.pipelines.context import BaseContextType
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    RequestID,
)
from max.profiler import traced
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler.utils import get_cancelled_reqs
from max.serve.scheduler_result import SchedulerResult

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")

# How many cancelled request ids to remember while waiting for the requests
# they name to reach the front of the queue.
_MAX_REMEMBERED_CANCELLATIONS = 1024


class OneShotScheduler(
    Scheduler,
    Generic[BaseContextType, PipelineInputsType, PipelineOutputType],
):
    """Scheduler for non-autoregressive pipelines that processes requests serially.

    This scheduler is optimized for pipelines that:
    - Complete in a single forward pass (no iterative generation)
    - Don't require KV cache management
    - Process each request independently

    The scheduler pulls one request at a time from the queue, executes the pipeline,
    and returns the result. This simple approach is suitable for image generation,
    embeddings with small batch sizes, and other non-autoregressive workloads.

    A request cancelled before it is pulled off the queue is dropped without
    being executed, which matters here more than for a scheduler that
    interleaves work: one render can hold this scheduler for minutes, so a
    client that disconnects can have its request sit queued behind that whole
    render. A request already handed to the pipeline does run to completion,
    since :meth:`Pipeline.execute` is one blocking call with no cancellation
    point to check.

    Args:
        pipeline: The pipeline to execute requests with
        batch_constructor: Callable that converts a request context into pipeline inputs.
            Takes a BaseContextType and returns a PipelineInputsType.
        request_queue: Queue to pull incoming requests from
        response_queue: Queue to push completed responses to
        cancel_queue: Queue for handling request cancellations
        max_batch_size: Maximum number of requests to process in a single batch.
            Defaults to 1 for serial processing.
    """

    def __init__(
        self,
        pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
        batch_constructor: Callable[[BaseContextType], PipelineInputsType],
        request_queue: MAXPullQueue[BaseContextType],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        max_batch_size: int = 1,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.pipeline = pipeline
        self.batch_constructor = batch_constructor
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue
        # Cancelled ids, in arrival order, awaiting the request they name.
        self._cancelled: dict[RequestID, None] = {}

    def _remember_cancellations(self) -> None:
        """Drains the cancel queue, keeping what it named for later.

        The ids are kept rather than matched against the work of the moment,
        because a cancellation usually arrives while some other request is
        mid-render and the request it names is still queued behind it.

        An id that never matches -- a client that disconnects after its own
        request already finished -- falls out once
        ``_MAX_REMEMBERED_CANCELLATIONS`` newer ones have arrived. That bounds
        the set without having to distinguish the two cases, which cannot be
        told apart from here.
        """
        for req_id in get_cancelled_reqs(self.cancel_queue):
            self._cancelled[req_id] = None
        while len(self._cancelled) > _MAX_REMEMBERED_CANCELLATIONS:
            self._cancelled.pop(next(iter(self._cancelled)))

    def _take_cancellation(self, request_id: RequestID) -> bool:
        """Whether the request was cancelled, forgetting it if it was."""
        if request_id in self._cancelled:
            del self._cancelled[request_id]
            return True
        return False

    @traced
    def _get_next_request(self) -> BaseContextType | None:
        """Pull the next request from the queue.

        Returns:
            The next context to process, or None if the queue is empty.
        """
        try:
            return self.request_queue.get_nowait()
        except queue.Empty:
            return None

    def run_iteration(self) -> SchedulerProgress:
        """Execute one scheduling iteration.

        Pulls a single request from the queue, executes it through the pipeline,
        and sends the response back. Requests cancelled while they waited are
        answered as cancelled and never reach the pipeline.

        Returns:
            SchedulerProgress.MADE_PROGRESS if a request was processed or
            dropped as cancelled, SchedulerProgress.NO_PROGRESS if no requests
            were available.
        """
        self._remember_cancellations()

        progress = SchedulerProgress.NO_PROGRESS
        context = self._get_next_request()
        while context is not None and self._take_cancellation(
            context.request_id
        ):
            logger.info(
                f"OneShotScheduler: Dropping cancelled request {context.request_id}"
            )
            self.response_queue.put_nowait(
                {context.request_id: SchedulerResult.cancelled()}
            )
            progress = SchedulerProgress.MADE_PROGRESS
            context = self._get_next_request()

        if context is None:
            return progress

        logger.info(f"OneShotScheduler: Starting request {context.request_id}")

        try:
            # Convert the context to pipeline inputs using the batch constructor
            pipeline_inputs = self.batch_constructor(context)

            # Execute the pipeline
            responses = self.pipeline.execute(pipeline_inputs)

            logger.info(
                f"OneShotScheduler: Completed request {context.request_id} "
                f"with {len(responses)} response(s)"
            )

            # Send the responses
            self.response_queue.put_nowait(
                {
                    request_id: SchedulerResult.create(response)
                    for request_id, response in responses.items()
                }
            )
        except Exception:
            logger.exception(
                f"OneShotScheduler: Exception during pipeline execution for request {context.request_id}"
            )

            # Send cancelled result (error details are logged above)
            self.response_queue.put_nowait(
                {context.request_id: SchedulerResult.cancelled()}
            )

        return SchedulerProgress.MADE_PROGRESS
