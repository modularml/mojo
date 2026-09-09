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

"""Tests for ZMQ IPC path generation and validation."""

from __future__ import annotations

import tempfile

import pytest
from max.pipelines.lora.lora_types import (
    LORA_REQUEST_ENDPOINT,
    LORA_RESPONSE_ENDPOINT,
)
from max.serve.pipelines.reset_prefix_cache import (
    ZMQ_RESET_PREFIX_CACHE_ENDPOINT,
)

# Sourced through the module under test to avoid a direct pyzmq dep here.
from max.serve.worker_interface import _zmq_queue
from max.serve.worker_interface._zmq_queue import (
    _validate_zmq_address,
    generate_zmq_ipc_path,
)

_IPC_PATH_MAX_LEN = _zmq_queue.zmq.IPC_PATH_MAX_LEN

# Every suffix callers append to a base path from generate_zmq_ipc_path.
_KNOWN_SUFFIXES = [
    ZMQ_RESET_PREFIX_CACHE_ENDPOINT,
    LORA_REQUEST_ENDPOINT,
    LORA_RESPONSE_ENDPOINT,
]

# A pathologically long temp dir, mirroring the BuildBuddy macOS remote-build
# sandbox (e.g. /Users/ec2-user/buildbuddy/remote_build/<uuid>.tmp) that
# triggered the original failure.
_LONG_TEMPDIR = (
    "/Users/ec2-user/buildbuddy/remote_build/"
    "4776f86f-0885-4419-980e-cb16c5ac374f.tmp"
)


def test_generate_zmq_ipc_path_uses_tempdir_when_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short temp dir is used as-is (no fallback)."""
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/tmp")
    path = generate_zmq_ipc_path()
    assert path.startswith("ipc:///tmp/")
    _validate_zmq_address(path)


def test_generate_zmq_ipc_path_fits_suffixes_with_long_tempdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long temp dir must not produce paths that overflow the IPC limit.

    Regression test: callers append a suffix (e.g. ``-reset_prefix_cache``) to
    the base path. With a long ``gettempdir()`` the fully-suffixed socket path
    used to exceed ``zmq.IPC_PATH_MAX_LEN`` and fail at server startup on the
    BuildBuddy macOS workers.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: _LONG_TEMPDIR)

    base = generate_zmq_ipc_path()
    # The base alone is valid...
    _validate_zmq_address(base)
    # ...and so is every suffixed endpoint derived from it.
    for suffix in _KNOWN_SUFFIXES:
        endpoint = f"{base}-{suffix}"
        _validate_zmq_address(endpoint)
        assert len(endpoint) - len("ipc://") <= _IPC_PATH_MAX_LEN
