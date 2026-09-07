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
"""``GenerateMixin.generate_async`` bookkeeping when requests finish apart.

Overlap scheduling can emit one more output for a request after it reported
done, so ``generate_async`` filters outputs against the requests still
running. The fake pipeline here models exactly that: one token per step per
request, plus one extra output for every request that has already finished.

It stands in for a real model because the behavior under test is bookkeeping
over ``execute``'s return value, which a graph would only make slower to
exercise.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from max.pipelines.context import (
    GenerationStatus,
    TextContext,
    TextGenerationOutput,
    TokenBuffer,
)
from max.pipelines.lib.interfaces.generate import GenerateMixin
from max.pipelines.modeling.types import RequestID


def _context() -> TextContext:
    return TextContext(
        request_id=RequestID(),
        max_length=64,
        tokens=TokenBuffer(np.arange(4, dtype=np.int64)),
    )


@dataclass
class FakeKVManager:
    """Records claim/release so a leak or a double release is visible.

    ``GenerateMixin`` hands these the CONTEXT, not its id, so the sets are
    keyed by ``context.request_id`` -- a context is not hashable.
    """

    claimed: set[RequestID] = field(default_factory=set)
    released: list[RequestID] = field(default_factory=list)

    def claim(self, context: Any, replica_idx: int = 0) -> None:
        self.claimed.add(context.request_id)

    def alloc(self, context: Any, replica_idx: int = 0) -> None:
        assert context.request_id in self.claimed

    def release(self, context: Any, replica_idx: int = 0) -> None:
        self.claimed.discard(context.request_id)
        self.released.append(context.request_id)

    def contains(self, context: Any, replica_idx: int = 0) -> bool:
        return context.request_id in self.claimed


class FakeTokenizer:
    async def new_context(self, prompt: TextContext) -> TextContext:
        return prompt


class FakePipeline(GenerateMixin[TextContext, TextContext]):
    """Emits one token per step per request, and one EXTRA after done.

    The extra output is the overlap-scheduling quirk the filter exists for:
    a request that has already reported ``is_done`` produces one more output
    on the following step.
    """

    # Declared by the protocol; nothing on this path reads it.
    _pipeline_model: Any = None

    def __init__(
        self,
        kv_manager: FakeKVManager,
        budget: dict[RequestID, int],
        fail_on_step: int | None = None,
    ) -> None:
        self._kv_manager = kv_manager
        self._budget = budget
        self._fail_on_step = fail_on_step
        self._emitted: dict[RequestID, int] = {}
        self._done: set[RequestID] = set()
        self.steps_run = 0
        self.released: list[RequestID] = []

    @property
    def kv_manager(self) -> Any:
        return self._kv_manager

    @property
    def pipeline_config(self) -> Any:
        class _Model:
            data_parallel_degree = 1

        class _Config:
            model = _Model()

        return _Config()

    @property
    def tokenizer(self) -> Any:
        return FakeTokenizer()

    def release(self, request_id: RequestID) -> None:
        self.released.append(request_id)

    def execute(self, inputs: Any) -> dict[RequestID, TextGenerationOutput]:
        self.steps_run += 1
        if self.steps_run == self._fail_on_step:
            raise RuntimeError("execute interrupted")
        outputs: dict[RequestID, TextGenerationOutput] = {}
        for batch in inputs.batches:
            for context in batch:
                emitted = self._emitted.get(context.request_id, 0) + 1
                self._emitted[context.request_id] = emitted
                is_done = emitted >= self._budget[context.request_id]
                outputs[context.request_id] = TextGenerationOutput(
                    request_id=context.request_id,
                    tokens=[emitted],
                    final_status=(
                        GenerationStatus.MAXIMUM_LENGTH
                        if is_done
                        else GenerationStatus.ACTIVE
                    ),
                )
                if is_done:
                    self._done.add(context.request_id)
        # The quirk: one extra output for a request that finished earlier and
        # has already been dropped from the batch.
        for request_id in self._done:
            outputs.setdefault(
                request_id,
                TextGenerationOutput(
                    request_id=request_id,
                    tokens=[-1],
                    final_status=GenerationStatus.MAXIMUM_LENGTH,
                ),
            )
        return outputs


def _make(
    steps: Sequence[int], fail_on_step: int | None = None
) -> tuple[FakePipeline, FakeKVManager, list[TextContext]]:
    prompts = [_context() for _ in steps]
    budget = {c.request_id: n for c, n in zip(prompts, steps, strict=True)}
    kv_manager = FakeKVManager()
    return FakePipeline(kv_manager, budget, fail_on_step), kv_manager, prompts


def _drive(pipeline: FakePipeline, prompts: Sequence[TextContext]) -> None:
    async def drive() -> None:
        async for _ in pipeline.generate_async(list(prompts)):
            pass

    asyncio.run(drive())


def _run(
    steps: Sequence[int],
) -> tuple[FakePipeline, FakeKVManager, list[TextContext]]:
    pipeline, kv_manager, prompts = _make(steps)
    _drive(pipeline, prompts)
    return pipeline, kv_manager, prompts


def test_generate_async_with_unequal_lengths() -> None:
    """Requests finishing on different steps must not raise.

    Once the short request finishes, its extra post-EOS output has to be
    dropped rather than looked up in a batch it has already left.
    """
    pipeline, kv_manager, prompts = _run([1, 3])

    assert kv_manager.claimed == set(), "a request was left claimed"
    assert set(kv_manager.released) == {p.request_id for p in prompts}
    assert len(kv_manager.released) == len(prompts), "released twice"
    # The long request still ran its full length rather than being cut off
    # when its batch-mate finished.
    assert pipeline.steps_run == 3


def test_generate_async_with_equal_lengths() -> None:
    """Requests finishing on the same step still complete once each."""
    pipeline, kv_manager, prompts = _run([2, 2])

    assert kv_manager.claimed == set()
    assert set(kv_manager.released) == {p.request_id for p in prompts}
    assert len(kv_manager.released) == len(prompts), "released twice"
    assert pipeline.steps_run == 2


@pytest.mark.parametrize("lengths", [(1, 2, 3), (3, 1, 2), (2, 3, 1)])
def test_generate_async_finish_order(lengths: tuple[int, ...]) -> None:
    """The bookkeeping holds for any order in which the requests finish."""
    _, kv_manager, prompts = _run(lengths)

    assert kv_manager.claimed == set()
    assert set(kv_manager.released) == {p.request_id for p in prompts}
    assert len(kv_manager.released) == len(prompts), "released twice"


@pytest.mark.parametrize("lengths", [(1, 3), (2, 2), (3, 1, 2)])
def test_pipeline_release_called_once_per_request(
    lengths: tuple[int, ...],
) -> None:
    """Finishing a request must release the PIPELINE, not just the KV cache.

    ``kv_manager.release`` frees only the paged KV blocks. Recurrent state
    slots and vision-encoder-cache refs hang off ``pipeline.release``, so
    without it a state-pool slot leaks per request -- and the next request
    inherits, or is denied, that slot. The post-EOS extra output must not
    turn into a second release for the same id.
    """
    pipeline, _, prompts = _run(lengths)

    assert set(pipeline.released) == {p.request_id for p in prompts}
    assert len(pipeline.released) == len(prompts), "released twice"


def test_pipeline_release_on_interrupted_generation() -> None:
    """An interrupted run must release every request it still holds.

    The ``finally`` path is the one that runs on cancellation, so it has to
    release both layers too; otherwise an aborted run strands its slots.
    """
    pipeline, kv_manager, prompts = _make([4, 4], fail_on_step=2)

    with pytest.raises(RuntimeError, match="execute interrupted"):
        _drive(pipeline, prompts)

    assert kv_manager.claimed == set(), "a request was left claimed"
    assert set(pipeline.released) == {p.request_id for p in prompts}
    assert len(pipeline.released) == len(prompts), "released twice"
