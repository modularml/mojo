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

"""Defines the :class:`SchedulerResult` data structure for MAX serve."""

from __future__ import annotations

from typing import Generic

import msgspec
from max.pipelines.modeling.types.pipeline import PipelineOutputType

__all__ = ["SchedulerResult"]


class SchedulerResult(msgspec.Struct, Generic[PipelineOutputType]):
    """Structure representing the result of a scheduler operation.

    Encapsulates the outcome of a pipeline operation as managed by the
    scheduler, including both the execution status and any resulting data.
    The scheduler uses this structure to communicate the state of pipeline
    operations back to clients, whether the operation is still running, has
    completed successfully, or was cancelled.
    """

    is_done: bool
    """Whether the pipeline operation is complete."""

    result: PipelineOutputType | None
    """The pipeline output data, if any. ``None`` for cancelled operations or
    intermediate streaming states."""

    error: str | None = None
    """User-facing message for a request the scheduler failed without
    executing; ``None`` otherwise."""

    batch_id: int | None = None
    """Monotonic forward-pass counter stamped by the scheduler.

    Identifies which GPU forward pass produced this result, acting as a join
    key between the OTel ``max.phase.*`` spans and — when
    ``kernel_trace_level`` is ``batch`` or above — the ``max.batch`` spans.
    ``None`` for cancelled operations.
    """

    @classmethod
    def cancelled(cls) -> SchedulerResult[PipelineOutputType]:
        """Creates a ``SchedulerResult`` representing a cancelled operation.

        Returns:
            A :class:`SchedulerResult` with ``is_done=True`` and no result.
        """
        return SchedulerResult(is_done=True, result=None)

    @classmethod
    def failed(cls, error: str) -> SchedulerResult[PipelineOutputType]:
        """Creates a ``SchedulerResult`` for a request failed by the scheduler.

        Args:
            error: User-facing description of why the request failed.

        Returns:
            A terminal :class:`SchedulerResult` carrying the error.
        """
        return SchedulerResult(is_done=True, result=None, error=error)

    @classmethod
    def create(
        cls,
        result: PipelineOutputType,
        batch_id: int | None = None,
    ) -> SchedulerResult[PipelineOutputType]:
        """Creates a ``SchedulerResult`` wrapping a pipeline output.

        Args:
            result: The pipeline output data.
            batch_id: Monotonic forward-pass counter for this batch.

        Returns:
            A :class:`SchedulerResult` reflecting the output's completion state.
        """
        return SchedulerResult(
            is_done=result.is_done, result=result, batch_id=batch_id
        )
