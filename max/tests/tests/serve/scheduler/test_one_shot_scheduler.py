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
"""Tests for the OneShotScheduler, which serves the media generation tasks."""

from __future__ import annotations

import queue
from dataclasses import dataclass

import numpy as np
from max.pipelines.context import (
    AudioContext,
    GenerationStatus,
    TokenBuffer,
)
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.modeling.types import (
    AudioGenerationInputs,
    Pipeline,
    RequestID,
)
from max.serve.scheduler.base import SchedulerProgress
from max.serve.scheduler.one_shot_scheduler import (
    _MAX_REMEMBERED_CANCELLATIONS,
    OneShotScheduler,
)
from max.serve.scheduler_result import SchedulerResult

_Inputs = AudioGenerationInputs[AudioContext]
_Responses = dict[RequestID, SchedulerResult[GenerationOutput]]


class _RecordingPipeline(Pipeline[_Inputs, GenerationOutput]):
    """Answers every request in the batch, recording what it was handed."""

    def __init__(self) -> None:
        self.executed: list[RequestID] = []

    @property
    def max_batch_size(self) -> int:
        return 1

    def execute(self, inputs: _Inputs) -> dict[RequestID, GenerationOutput]:
        self.executed.extend(inputs.batch)
        return {
            request_id: GenerationOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                output=[],
            )
            for request_id in inputs.batch
        }

    def release(self, request_id: RequestID) -> None:
        pass


@dataclass
class _Harness:
    """A scheduler and the three queues it was wired to."""

    scheduler: OneShotScheduler[AudioContext, _Inputs, GenerationOutput]
    requests: queue.Queue[AudioContext]
    responses: queue.Queue[_Responses]
    cancellations: queue.Queue[list[RequestID]]
    pipeline: _RecordingPipeline

    def responded(self) -> _Responses:
        """Everything on the response queue, flattened into one mapping."""
        seen: _Responses = {}
        while True:
            try:
                seen.update(self.responses.get_nowait())
            except queue.Empty:
                return seen


def _harness() -> _Harness:
    requests: queue.Queue[AudioContext] = queue.Queue()
    responses: queue.Queue[_Responses] = queue.Queue()
    cancellations: queue.Queue[list[RequestID]] = queue.Queue()
    pipeline = _RecordingPipeline()
    scheduler = OneShotScheduler[AudioContext, _Inputs, GenerationOutput](
        pipeline=pipeline,
        batch_constructor=lambda context: AudioGenerationInputs(
            batch={context.request_id: context}
        ),
        request_queue=requests,
        response_queue=responses,
        cancel_queue=cancellations,
    )
    return _Harness(
        scheduler=scheduler,
        requests=requests,
        responses=responses,
        cancellations=cancellations,
        pipeline=pipeline,
    )


def _request() -> AudioContext:
    return AudioContext(
        request_id=RequestID(),
        tokens=TokenBuffer(array=np.ones(4, dtype=np.int64)),
        audio_duration=10.0,
        num_inference_steps=4,
    )


def test_an_empty_queue_makes_no_progress() -> None:
    harness = _harness()

    assert harness.scheduler.run_iteration() == SchedulerProgress.NO_PROGRESS
    assert harness.pipeline.executed == []


def test_a_queued_request_runs() -> None:
    harness = _harness()
    request = _request()
    harness.requests.put_nowait(request)

    assert harness.scheduler.run_iteration() == SchedulerProgress.MADE_PROGRESS

    assert harness.pipeline.executed == [request.request_id]
    assert harness.responded()[request.request_id].result is not None


def test_a_request_cancelled_while_queued_never_runs() -> None:
    """The point of draining: a render can hold the scheduler for minutes.

    A client that disconnects in the meantime should not have its request
    executed once the scheduler finally reaches it.
    """
    harness = _harness()
    request = _request()
    harness.requests.put_nowait(request)
    harness.cancellations.put_nowait([request.request_id])

    assert harness.scheduler.run_iteration() == SchedulerProgress.MADE_PROGRESS

    assert harness.pipeline.executed == []
    result = harness.responded()[request.request_id]
    assert result.is_done
    assert result.result is None


def test_a_cancelled_request_does_not_hold_up_the_next_one() -> None:
    harness = _harness()
    cancelled, live = _request(), _request()
    harness.requests.put_nowait(cancelled)
    harness.requests.put_nowait(live)
    harness.cancellations.put_nowait([cancelled.request_id])

    assert harness.scheduler.run_iteration() == SchedulerProgress.MADE_PROGRESS

    # One iteration drops the cancelled request and runs the live one, rather
    # than spending the whole iteration on the drop.
    assert harness.pipeline.executed == [live.request_id]
    responded = harness.responded()
    assert responded[cancelled.request_id].result is None
    assert responded[live.request_id].result is not None


def test_the_cancel_queue_is_drained_rather_than_accumulating() -> None:
    harness = _harness()
    for _ in range(8):
        harness.cancellations.put_nowait([RequestID()])

    harness.scheduler.run_iteration()

    assert harness.cancellations.empty()


def test_a_cancellation_for_an_unrelated_request_is_ignored() -> None:
    harness = _harness()
    request = _request()
    harness.requests.put_nowait(request)
    harness.cancellations.put_nowait([RequestID()])

    assert harness.scheduler.run_iteration() == SchedulerProgress.MADE_PROGRESS

    assert harness.pipeline.executed == [request.request_id]


def test_remembered_cancellations_stay_bounded() -> None:
    """Draining must not trade a growing queue for a growing set.

    A cancellation whose request never arrives -- a client disconnecting after
    its own request already finished -- would otherwise be kept forever.
    """
    harness = _harness()
    for _ in range(_MAX_REMEMBERED_CANCELLATIONS + 50):
        harness.cancellations.put_nowait([RequestID()])

    harness.scheduler.run_iteration()

    assert len(harness.scheduler._cancelled) == _MAX_REMEMBERED_CANCELLATIONS


def test_the_newest_cancellation_survives_the_bound() -> None:
    harness = _harness()
    for _ in range(_MAX_REMEMBERED_CANCELLATIONS):
        harness.cancellations.put_nowait([RequestID()])
    request = _request()
    harness.cancellations.put_nowait([request.request_id])
    harness.requests.put_nowait(request)

    assert harness.scheduler.run_iteration() == SchedulerProgress.MADE_PROGRESS

    # The oldest id was evicted to make room, not the one just added.
    assert harness.pipeline.executed == []
