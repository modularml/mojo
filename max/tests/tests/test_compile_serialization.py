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
"""The process-global compilation lock releases on success and failure."""

from __future__ import annotations

import threading
import time

import pytest
from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.engine import api as engine_api
from max.graph import DeviceRef, Graph, TensorType


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


def _assert_lock_releases() -> None:
    # The lock is released by an add_done_callback on a runtime worker
    # thread, which is not ordered before compile()'s wait() returns.
    deadline = time.monotonic() + 30
    while engine_api._COMPILATION_LOCK.locked():
        assert time.monotonic() < deadline, "compilation lock never released"
        time.sleep(0.01)


def _scale_graph(scale: float) -> Graph:
    with Graph(
        f"scale_{scale}".replace(".", "_"),
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as g:
        (x,) = g.inputs
        g.output(x.tensor * scale)
    return g


def test_lock_released_after_successful_compile(
    session: InferenceSession,
) -> None:
    session.compile(_scale_graph(2.0))
    _assert_lock_releases()


def test_lock_released_after_failed_compile(
    session: InferenceSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingImpl:
        def compile(self, *args: object) -> None:
            raise RuntimeError("synthetic compile failure")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(session, "_impl", _FailingImpl())
        with pytest.raises(RuntimeError):
            session.compile(_scale_graph(4.0))
    _assert_lock_releases()
    # The session remains usable.
    session.compile(_scale_graph(3.0))
    _assert_lock_releases()


def test_concurrent_compiles_serialize_and_complete(
    session: InferenceSession,
) -> None:
    errors: list[Exception] = []

    def _compile(scale: float) -> None:
        try:
            session.compile(_scale_graph(scale))
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=_compile, args=(scale,))
        for scale in (5.0, 6.0, 7.0)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors
    _assert_lock_releases()
