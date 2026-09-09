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

"""Unit test for ``DecodeScheduler.handle_prefill_response`` racing a
cancellation.

Written against the unbound method with a mocked ``self`` so we avoid the
full PD test harness for what is otherwise a small, self-contained check.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from max.pipelines.request.base import RequestID
from max.serve.scheduler.base import PrefillResponse
from max.serve.scheduler.decode_scheduler import (
    DecodeRequestPhase,
    DecodeScheduler,
    PendingDecodeRequest,
)


def test_prefill_response_after_cancel_does_not_emit_generation_result() -> (
    None
):
    """A ``PrefillResponse`` landing after cancellation must not emit a
    generation result -- the client already got ``SchedulerResult.cancelled()``.

    Regression: a request cancelled while ``AWAITING_PREFILL`` with its
    onload still in flight stays in ``self.requests`` (deferred release)
    instead of being deleted immediately, so a later ``PrefillResponse``
    was processed as if the request were still live. Structurally
    impossible before eager-send (SERVOPT-1579).
    """
    req_id = RequestID("r")
    onload_event = MagicMock()
    onload_event.is_complete.return_value = False
    context = SimpleNamespace(request_id=req_id)
    pending = PendingDecodeRequest(
        context=context,  # type: ignore[arg-type]
        replica_idx=0,
        phase=DecodeRequestPhase.AWAITING_PREFILL,
        onload_event=onload_event,
        cancelled=True,
    )
    # No `response_queue` attribute: if the buggy path (pushing a
    # generation result) were reached, this raises AttributeError instead
    # of silently succeeding.
    self_obj = SimpleNamespace(requests={req_id: pending})
    transfer_metadata = MagicMock()
    message = PrefillResponse(
        id=req_id,
        generated_token_id=5,
        transfer_metadata=transfer_metadata,
    )

    DecodeScheduler.handle_prefill_response(self_obj, message)  # type: ignore[arg-type]

    assert pending.cancelled is True
    assert pending.phase is DecodeRequestPhase.TRANSFERRING
    assert pending.transfer is transfer_metadata
