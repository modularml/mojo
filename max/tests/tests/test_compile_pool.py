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
"""``ProcessCompilePool`` compiles graphs in a background worker process."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import python.runfiles
from max import engine
from max.driver import CPU, DeviceSpec
from max.dtype import DType
from max.engine import InferenceSession
from max.experimental.compile_pool import (
    ProcessCompilePool,
    RemoteCompileError,
)
from max.graph import DeviceRef, Graph, Module, TensorType


@pytest.fixture(scope="module")
def session() -> InferenceSession:
    return InferenceSession(devices=[CPU()])


@pytest.fixture
def pool() -> Iterator[ProcessCompilePool]:
    with ProcessCompilePool(device_specs=[DeviceSpec.cpu()]) as p:
        yield p


def _build_graph(scale: float) -> Graph:
    with Graph(
        f"scale_{scale}".replace(".", "_"),
        input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
    ) as g:
        (x,) = g.inputs
        g.output(x.tensor * scale)
    return g


def _run_model(
    session: InferenceSession, compiled: engine.CompiledModel
) -> np.ndarray:
    model = session.init(compiled)
    (out,) = model(np.ones(4, dtype=np.float32))
    return np.asarray(out.to_numpy())


def test_compiles_in_background(
    pool: ProcessCompilePool, session: InferenceSession
) -> None:
    futures = [pool.compile(_build_graph(scale)) for scale in (2.0, 3.0)]
    for scale, future in zip((2.0, 3.0), futures, strict=False):
        np.testing.assert_array_equal(
            _run_model(session, future.result(timeout=240)),
            np.full(4, scale, dtype=np.float32),
        )


def test_compiles_multi_graph_module(
    pool: ProcessCompilePool, session: InferenceSession
) -> None:
    module = Module()
    scales = (2.0, 3.0)
    for scale in scales:
        with Graph(
            f"scale_{scale}".replace(".", "_"),
            input_types=[
                TensorType(DType.float32, [4], device=DeviceRef.CPU())
            ],
            module=module,
        ) as g:
            (x,) = g.inputs
            g.output(x.tensor * scale)

    with pytest.raises(ValueError, match="at least one device"):
        pool.compile_module(module, device_specs=[])

    future = pool.compile_module(module, device_specs=[DeviceSpec.cpu()])
    models = session.init_all(future.result(timeout=240))
    assert len(models) == len(scales)
    for scale in scales:
        name = f"scale_{scale}".replace(".", "_")
        (out,) = models[name](np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(
            np.asarray(out.to_numpy()), np.full(4, scale, dtype=np.float32)
        )


def test_compile_error_surfaces(pool: ProcessCompilePool) -> None:
    # A stand-in whose "bytecode" cannot deserialize: the worker's traceback
    # must come back attached to this future, not kill the pool.
    broken = SimpleNamespace(
        _module=SimpleNamespace(bytecode=b"not mlir bytecode"),
        kernel_libraries_paths=[],
    )
    future = pool.compile(cast(Graph, broken))
    with pytest.raises(RemoteCompileError, match="Traceback"):
        future.result(timeout=240)

    # The pool survives a failed compile.
    good = pool.compile(_build_graph(2.0))
    assert good.result(timeout=240) is not None


def test_worker_death_fails_pending(pool: ProcessCompilePool) -> None:
    future = pool.compile(_build_graph(5.0))
    # Workers are created lazily on submit; wait for one to exist.
    deadline = time.monotonic() + 240
    while not pool._executor._processes and time.monotonic() < deadline:
        time.sleep(0.05)
    for process in list(pool._executor._processes.values()):
        process.kill()
    with pytest.raises(BrokenProcessPool):
        future.result(timeout=240)
    # No restart: the pool is poisoned.
    with pytest.raises(BrokenProcessPool):
        pool.compile(_build_graph(6.0))


def test_close_is_idempotent_and_rejects_new_work(
    pool: ProcessCompilePool, session: InferenceSession
) -> None:
    future = pool.compile(_build_graph(4.0))
    compiled = future.result(timeout=240)

    pool.close()
    pool.close()
    # The artifact is mmapped: it outlives the pool and its MEF dir.
    result = _run_model(session, compiled)
    np.testing.assert_array_equal(result, np.full(4, 4.0, dtype=np.float32))
    with pytest.raises(RuntimeError, match="closed"):
        pool.compile(_build_graph(7.0))


def test_close_discards_queued_compiles_promptly() -> None:
    # A single worker chewing through long sleeps would take 2 minutes to
    # drain; close() must kill rather than wait, and every discarded
    # future must still resolve (failed or cancelled, never in limbo).
    with ProcessCompilePool(
        device_specs=[DeviceSpec.cpu()], max_workers=1
    ) as pool:
        blockers = [pool._executor.submit(time.sleep, 30) for _ in range(4)]
        queued = pool.compile(_build_graph(8.0))
        start = time.monotonic()
        pool.close()
        elapsed = time.monotonic() - start
    with pytest.raises((BrokenProcessPool, CancelledError)):
        queued.result(timeout=240)
    assert all(b.done() for b in blockers)
    assert elapsed < 25, f"close() took {elapsed:.1f}s; did it wait?"


def test_close_terminates_started_compiles() -> None:
    # A submission immediately enters the executor's call queue and is
    # marked running, past the reach of cancel_futures: close() kills the
    # worker instead of waiting. A future escaping a pool's scope fails
    # rather than resolving to a path inside the deleted MEF directory.
    with ProcessCompilePool(
        device_specs=[DeviceSpec.cpu()], max_workers=1
    ) as pool:
        future = pool.compile(_build_graph(9.0))
        # The chained future is resolved by callback and is never RUNNING;
        # a spawned worker implies the compile entered the call queue,
        # past the reach of cancel_futures.
        deadline = time.monotonic() + 240
        while not pool._executor._processes and time.monotonic() < deadline:
            time.sleep(0.05)
        assert pool._executor._processes, "no worker spawned"
        pool.close()
    with pytest.raises((BrokenProcessPool, CancelledError)):
        future.result(timeout=240)


def test_worker_dies_with_parent() -> None:
    runfiles = python.runfiles.Create()
    assert runfiles is not None, "runfiles unavailable under bazel test"
    script = runfiles.Rlocation(
        "_main/max/tests/tests/compile_pool_parent_exit_subprocess.py"
    )
    assert script is not None
    result = subprocess.run(
        [sys.executable, script],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    worker_pid = int(result.stdout.strip())

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError:
            return  # The worker died with its parent.
        time.sleep(0.5)
    pytest.fail(f"compile worker {worker_pid} outlived its parent")


def test_close_between_compile_resolution_and_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # compile() submits, then chains the artifact load; when the worker
    # finishes and close() runs in that gap, the load runs inline on the
    # submitting thread against the deleted MEF dir. The future must fail
    # like any other discarded compile, not with a missing-file error.
    with ProcessCompilePool(
        device_specs=[DeviceSpec.cpu()], max_workers=1
    ) as pool:
        real_submit = pool._executor.submit

        def resolve_then_close(*args: Any) -> Future[Any]:
            inner: Future[Any] = real_submit(*args)
            inner.result(timeout=240)  # The MEF is written and resolved...
            pool.close()  # ...and the pool deletes it before the load.
            return inner

        monkeypatch.setattr(pool._executor, "submit", resolve_then_close)
        future = pool.compile(_build_graph(11.0))
    with pytest.raises((BrokenProcessPool, CancelledError)):
        future.result(timeout=240)
