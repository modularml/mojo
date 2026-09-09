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
"""Tests for MAX Serve telemetry configuration: the log-handler component
allowlist and kernel-trace gating."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from max.serve.config import KernelTraceLevel, Settings
from max.serve.telemetry import common
from max.serve.telemetry.common import (
    batch_spans_enabled,
    configure_kernel_tracing,
    configure_logging,
)


@pytest.fixture(autouse=True)
def restore_root_logging() -> Iterator[None]:
    """Undoes ``configure_logging``'s global mutation of the logging tree."""
    root = logging.getLogger()
    uvicorn_logger = logging.getLogger("uvicorn")
    saved = (list(root.handlers), root.level, uvicorn_logger.level)
    try:
        yield
    finally:
        handlers, root_level, uvicorn_level = saved
        for handler in list(root.handlers):
            if handler not in handlers:
                handler.close()
        root.handlers[:] = handlers
        root.setLevel(root_level)
        uvicorn_logger.setLevel(uvicorn_level)


@pytest.fixture
def emitted(tmp_path: Path) -> Iterator[Path]:
    log_path = tmp_path / "max-serve.log"
    configure_logging(
        Settings(
            logs_console_level=None,
            logs_file_level="WARNING",
            logs_file_path=str(log_path),
            disable_telemetry=True,
        )
    )
    yield log_path


def _emit(name: str, level: int, message: str) -> None:
    logging.getLogger(name).log(level, message)
    for handler in logging.getLogger().handlers:
        handler.flush()


def test_admits_uvicorn_error_records(emitted: Path) -> None:
    # uvicorn owns the HTTP error log. Filtering it out left TCP-level
    # failures -- an exception escaping the ASGI app, in-flight requests
    # cancelled when the shutdown drain expires -- with no server-side trace.
    _emit("uvicorn.error", logging.ERROR, "Exception in ASGI application")
    assert "Exception in ASGI application" in emitted.read_text()


def test_admits_max_component_records(emitted: Path) -> None:
    _emit("max.serve", logging.WARNING, "max-serve-warning")
    assert "max-serve-warning" in emitted.read_text()


def test_excludes_uvicorn_access_records(emitted: Path) -> None:
    # The access stream logs at INFO and would swamp the sink one line per
    # request; the WARNING pin on the ``uvicorn`` logger keeps it out.
    _emit("uvicorn.access", logging.INFO, "GET /health")
    assert "GET /health" not in emitted.read_text()


def test_excludes_unrelated_third_party_records(emitted: Path) -> None:
    _emit("httpx", logging.WARNING, "third-party-chatter")
    assert "third-party-chatter" not in emitted.read_text()


def test_kernel_trace_level_gates_batch_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # setattr records the pre-test global so teardown restores it even after
    # the configure calls below overwrite it.
    monkeypatch.setattr(common, "_kernel_trace_level", KernelTraceLevel.OFF)
    for level, expected in (("batch", True), ("off", False)):
        monkeypatch.setenv("MAX_SERVE_KERNEL_TRACE_LEVEL", level)
        configure_kernel_tracing(Settings())
        assert batch_spans_enabled() == expected
    # op/kernel imply batch spans. Set the global directly: at these levels
    # configure_kernel_tracing also touches GPU profiling state.
    for deep_level in (KernelTraceLevel.OP, KernelTraceLevel.KERNEL):
        monkeypatch.setattr(common, "_kernel_trace_level", deep_level)
        assert batch_spans_enabled()
