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

"""Unit tests for ``DecodeScheduler._evict_expired_requests``.

Tests are written against the unbound method with a mocked ``self`` so
we avoid the full PD test harness for what is otherwise a small,
self-contained sweep.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from max.serve.scheduler.decode_scheduler import (
    DecodeRequestPhase,
    DecodeScheduler,
    PendingDecodeRequest,
)
from max.serve.scheduler_result import SchedulerResult


def _make_self(
    *,
    ttl_s: float | None,
    requests: dict[str, PendingDecodeRequest],
    prefill_reqs_per_replica: list[int],
) -> SimpleNamespace:
    self_obj = SimpleNamespace(
        scheduler_config=SimpleNamespace(decode_request_ttl_s=ttl_s),
        requests=requests,
        prefill_reqs_per_replica=prefill_reqs_per_replica,
        kv_cache=MagicMock(),
        transfer_engine=MagicMock(),
        response_queue=MagicMock(),
        dispatcher=MagicMock(),
        batch_constructor=MagicMock(),
        _handoff_landed_at={},
    )
    # Bind the real instance method so the cancel-to-prefill path runs.
    self_obj._send_cancel_to_prefill = (
        DecodeScheduler._send_cancel_to_prefill.__get__(self_obj)
    )
    return self_obj


def _ctx(
    req_id: str, target_endpoint: str = "tcp://prefill:6000"
) -> SimpleNamespace:
    return SimpleNamespace(request_id=req_id, target_endpoint=target_endpoint)


def _completed_onload() -> MagicMock:
    onload_event = MagicMock()
    onload_event.is_complete.return_value = True
    return onload_event


def _awaiting_prefill(
    req_id: str, replica_idx: int, phase_entered_at: float
) -> PendingDecodeRequest:
    return PendingDecodeRequest(
        context=_ctx(req_id),  # type: ignore[arg-type]
        replica_idx=replica_idx,
        phase=DecodeRequestPhase.AWAITING_PREFILL,
        onload_event=_completed_onload(),
        phase_entered_at=phase_entered_at,
    )


def _transferring(
    req_id: str,
    replica_idx: int,
    phase_entered_at: float,
    *,
    cancelled: bool = False,
) -> PendingDecodeRequest:
    return PendingDecodeRequest(
        context=_ctx(req_id),  # type: ignore[arg-type]
        replica_idx=replica_idx,
        phase=DecodeRequestPhase.TRANSFERRING,
        onload_event=_completed_onload(),
        phase_entered_at=phase_entered_at,
        transfer=MagicMock(),
        cancelled=cancelled,
    )


def test_ttl_disabled_evicts_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=None,
        requests={"r": _transferring("r", 0, 0.0)},
        prefill_reqs_per_replica=[1, 0],
    )
    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]
    assert "r" in self_obj.requests
    self_obj.kv_cache.release.assert_not_called()
    self_obj.response_queue.put_nowait.assert_not_called()


def test_evicts_stuck_prefill_req(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PrefillResponse: an AWAITING_PREFILL entry past TTL is evicted."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    stuck = _awaiting_prefill("stuck", 1, 900.0)  # 100s ago > 30s TTL
    self_obj = _make_self(
        ttl_s=30.0,
        requests={
            "stuck": stuck,
            "fresh": _awaiting_prefill("fresh", 0, 999.0),  # 1s ago
        },
        prefill_reqs_per_replica=[1, 1],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "stuck" not in self_obj.requests
    assert "fresh" in self_obj.requests
    assert self_obj.prefill_reqs_per_replica == [1, 0]
    self_obj.kv_cache.release.assert_called_once_with(stuck.context)
    self_obj.response_queue.put_nowait.assert_called_once_with(
        {"stuck": SchedulerResult.cancelled()}
    )
    self_obj.dispatcher.send_request_nowait.assert_called_once()


def test_evicts_stuck_inflight_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transfer never completed: the TRANSFERRING entry is evicted."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    stuck = _transferring("stuck", 0, 920.0)  # 80s ago
    self_obj = _make_self(
        ttl_s=30.0,
        requests={"stuck": stuck},
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "stuck" not in self_obj.requests
    assert self_obj.prefill_reqs_per_replica == [0, 0]
    self_obj.transfer_engine.cleanup_transfer.assert_called_once_with(
        stuck.transfer
    )
    self_obj.kv_cache.release.assert_called_once_with(stuck.context)
    self_obj.response_queue.put_nowait.assert_called_once_with(
        {"stuck": SchedulerResult.cancelled()}
    )
    self_obj.dispatcher.send_request_nowait.assert_called_once()


def test_transferring_keeps_fresh_phase_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A TRANSFERRING entry is judged by its own ``phase_entered_at``, not
    how long the request had earlier waited as AWAITING_PREFILL.

    ``phase_entered_at`` resets on every phase transition (see
    ``PendingDecodeRequest``), so there is no separate stale timestamp from
    the AWAITING_PREFILL phase left to accidentally evict on -- unlike the
    old two-dict design, this is structural rather than something the sweep
    has to special-case.
    """
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=30.0,
        requests={"r": _transferring("r", 0, 999.0)},  # 1s ago (fresh)
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "r" in self_obj.requests
    assert self_obj.prefill_reqs_per_replica == [1, 0]
    self_obj.kv_cache.release.assert_not_called()
    self_obj.response_queue.put_nowait.assert_not_called()


def test_evicts_stuck_transfer_already_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transfer cancelled mid-flight (see _handle_cancelled_requests)
    that never completes must still be released by the TTL sweep, but
    without re-notifying the client or re-sending a cancel to prefill --
    both already happened when the cancellation was first processed.
    """
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    stuck = _transferring("stuck", 0, 920.0, cancelled=True)  # 80s ago
    self_obj = _make_self(
        ttl_s=30.0,
        requests={"stuck": stuck},
        prefill_reqs_per_replica=[1, 0],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "stuck" not in self_obj.requests
    assert self_obj.prefill_reqs_per_replica == [0, 0]
    self_obj.kv_cache.release.assert_called_once_with(stuck.context)
    self_obj.response_queue.put_nowait.assert_not_called()
    self_obj.dispatcher.send_request_nowait.assert_not_called()


def test_healthy_entries_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entirely fresh batch yields no evictions."""
    monkeypatch.setattr("time.monotonic", lambda: 1000.0)
    self_obj = _make_self(
        ttl_s=30.0,
        requests={
            "a": _awaiting_prefill("a", 0, 999.0),
            "b": _transferring("b", 1, 998.0),
        },
        prefill_reqs_per_replica=[1, 1],
    )

    DecodeScheduler._evict_expired_requests(self_obj)  # type: ignore[arg-type]

    assert "a" in self_obj.requests
    assert "b" in self_obj.requests
    assert self_obj.prefill_reqs_per_replica == [1, 1]
    self_obj.response_queue.put_nowait.assert_not_called()
