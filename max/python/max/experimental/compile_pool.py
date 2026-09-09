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
"""Out-of-process graph compilation.

Compiling multiple graphs in parallel in-process is currently
unsafe. :class:`ProcessCompilePool` compiles graphs in background worker
processes (a :class:`~concurrent.futures.ProcessPoolExecutor`).

Forking thread pools such as exist in Mojo and the graph compiler is not
typically safe. ProcessCompilePool works around this using ``forkserver``.
An initial process is created which imports `max.engine` to share import
overhead among workers, and then compile workers are forked from this process.

Graphs cross to a worker as module bytecode. Compiled models are saved
to MEF paths to avoid pickling large model binaries over pipes.
"""

from __future__ import annotations

import atexit
import ctypes
import functools
import io
import multiprocessing as mp
import multiprocessing.forkserver
import multiprocessing.popen_forkserver
import multiprocessing.spawn
import multiprocessing.util
import os
import signal
import sys
import tempfile
import threading
import traceback
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import TypeVar

from max import driver, engine, mlir
from max._core import Operation
from max.graph import Graph, Module

_T = TypeVar("_T")
_R = TypeVar("_R")


def _map_future(future: Future[_T], f: Callable[[_T], _R]) -> Future[_R]:
    """Returns a future resolving to ``f(future.result())``.

    ``f`` runs on the thread that resolves ``future``.
    A failed or cancelled input resolves the returned future with the
    same exception.
    """
    result: Future[_R] = Future()

    def resolve(done: Future[_T]) -> None:
        if not result.set_running_or_notify_cancel():
            return
        try:
            result.set_result(f(done.result()))
        except BaseException as e:
            result.set_exception(e)

    future.add_done_callback(resolve)
    return result


def _close_fds(*fds: int) -> None:
    """Stands in for ``multiprocessing.util.close_fds``, which typeshed omits."""
    for fd in fds:
        os.close(fd)


# _NoMainForkServerContext
# - ``forkserver`` workers re-import ``__main__``, making them
#   poor targets for using in libraries.
# - Strategy inherited from [loky](https://github.com/joblib/loky).
#   Subclass ForkServerContext and change worker behavior to
#   not exec `__main__`.


class _NoMainPopen(mp.popen_forkserver.Popen):
    # Set by Popen.__init__; typeshed doesn't carry it.
    _fds: list[int]

    def _launch(self, process_obj: mp.process.BaseProcess) -> None:
        prep_data = mp.spawn.get_preparation_data(process_obj.name)
        prep_data.pop("init_main_from_name", None)
        prep_data.pop("init_main_from_path", None)

        # Remainder copied from CPython's popen_forkserver.Popen._launch".
        # Can't call super()._launch(), which regenerates prep_data().

        buf = io.BytesIO()
        mp.context.set_spawning_popen(self)
        try:
            mp.reduction.dump(prep_data, buf)
            mp.reduction.dump(process_obj, buf)
        finally:
            mp.context.set_spawning_popen(None)

        self.sentinel, w = mp.forkserver.connect_to_new_process(self._fds)
        # Keep a duplicate of the data pipe's write end as a sentinel of
        # the parent process used by the child process.
        parent_w = os.dup(w)
        self.finalizer = mp.util.Finalize(
            self, _close_fds, (parent_w, self.sentinel)
        )
        with open(w, "wb", closefd=True) as f:
            f.write(buf.getbuffer())
        self.pid = mp.forkserver.read_signed(self.sentinel)


class _NoMainProcess(mp.context.ForkServerProcess):
    _Popen = _NoMainPopen


class _NoMainForkServerContext(mp.context.ForkServerContext):
    Process = _NoMainProcess


class RemoteCompileError(RuntimeError):
    """A compile failed in the worker; the message embeds its traceback."""


def _pin_to_parent_death() -> None:
    """Asks Linux to SIGKILL this process when its parent dies uncleanly.

    The parent process is the forkserver. This mechanism covers the narrow
    case where the forkserver dies without the original parent process
    dying, and only works on Linux.
    """
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)


# Worker-process state, created by _init_worker and _session_for.
_WORKER_DEVICE_SPECS: tuple[driver.DeviceSpec, ...] = ()
_WORKER_SESSIONS: dict[
    tuple[driver.DeviceSpec, ...], engine.InferenceSession
] = {}


def _start_parent_heartbeat_check() -> None:
    """Exits this worker once the process that built the pool is gone."""

    def wait_and_exit(parent: mp.process.BaseProcess) -> None:
        parent.join()
        os._exit(1)

    if (parent := mp.parent_process()) is not None:
        threading.Thread(
            target=wait_and_exit, args=(parent,), daemon=True
        ).start()


def _init_worker(device_specs: Sequence[driver.DeviceSpec]) -> None:
    """Pins the worker's lifetime and records its default session devices."""
    global _WORKER_DEVICE_SPECS
    _pin_to_parent_death()
    _start_parent_heartbeat_check()
    _WORKER_DEVICE_SPECS = tuple(device_specs)


def _session_for(
    device_specs: tuple[driver.DeviceSpec, ...],
) -> engine.InferenceSession:
    """Returns this worker's session for *device_specs*, creating it lazily.

    Keyed by the ordered tuple, not a set: the leading device is the
    session's default compile target and enters the MEF cache key (see
    :meth:`ProcessCompilePool.compile_module`).
    """
    session = _WORKER_SESSIONS.get(device_specs)
    if session is None:
        session = engine.InferenceSession(
            devices=driver.load_devices(device_specs)
        )
        _WORKER_SESSIONS[device_specs] = session
    return session


def _compile_to_mef(
    bytecode: bytes,
    extensions: Sequence[Path],
    mef_path: Path,
    device_specs: tuple[driver.DeviceSpec, ...] | None = None,
) -> Path:
    try:
        if device_specs is None:
            device_specs = _WORKER_DEVICE_SPECS
        session = _session_for(device_specs)
        module = Module(
            mlir_module=Operation.from_bytecode(bytecode, mlir.Context())
        )
        compiled = session.compile(module, custom_extensions=extensions)
        compiled.export_mef(mef_path)
    except BaseException:
        # Tracebacks don't survive pickling.
        # Format the trace into the exception message.
        raise RemoteCompileError(
            f"failed to compile with error: {traceback.format_exc()}"
        ) from None

    return mef_path


class ProcessCompilePool:
    """Compiles graphs in background worker processes.

    In-progress compilations are terminated on shutdown. Users should
    wait on returned futures if they want to guarantee a compile completes.

    Forking thread pools such as exist in Mojo and the graph compiler is not
    typically safe. ProcessCompilePool works around this using ``forkserver``.
    An initial process is created which imports `max.engine` to share import
    overhead among workers, and then compile workers are forked from this process.

    Graphs cross to a worker as module bytecode. Compiled models are saved
    to MEF paths to avoid pickling large model binaries over pipes.

    .. code-block:: python

        import numpy as np

        from max.driver import CPU, DeviceSpec
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.experimental.compile_pool import ProcessCompilePool
        from max.graph import DeviceRef, Graph, TensorType

        with Graph(
            "scale_2",
            input_types=[TensorType(DType.float32, [4], device=DeviceRef.CPU())],
        ) as graph:
            (x,) = graph.inputs
            graph.output(x.tensor * 2.0)

        session = InferenceSession(devices=[CPU()])
        with ProcessCompilePool(device_specs=[DeviceSpec.cpu()]) as pool:
            future = pool.compile(graph)
            model = session.init(future.result())

        (result,) = model(np.ones(4, dtype=np.float32))

    .. invisible-code-block: python

        np.testing.assert_array_equal(
            result.to_numpy(), np.full(4, 2.0, dtype=np.float32)
        )
    """

    def __init__(
        self,
        device_specs: Sequence[driver.DeviceSpec] | None = None,
        max_workers: int = os.cpu_count() or 1,
    ) -> None:
        """Creates a pool whose worker sessions use ``device_specs``.

        Args:
            device_specs: Devices for the workers' sessions. If None,
                defaults to all available accelerators plus the CPU.
            max_workers: Upper bound on concurrent compiles. Each worker
                holds an :class:`~max.engine.InferenceSession` for the
                given devices which reserves some device memory.
        """
        if device_specs is None:
            specs = driver.scan_available_devices()
            if (cpu := driver.DeviceSpec.cpu()) not in specs:
                specs.append(cpu)
            device_specs = specs

        # max import is O(seconds), do this pre-fork for each worker
        mp_context = _NoMainForkServerContext()
        mp_context.set_forkserver_preload(["max.engine"])
        self._executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(tuple(device_specs),),
        )

        self._mef_dir = tempfile.TemporaryDirectory(prefix="max-compile-pool-")
        self._closed = False

        # Terminate the pool on exit without waiting on in-progress compiles.
        # - MEF cache is safely atomic and can't get partial writes
        # - atexit.register() runs after ProcessPoolExecutor has already
        #   been joined.
        # - Follow `concurrent.futures` and use `threading._register_atexit()`
        #   which runs before ProcessPoolExecutor.join()
        # - Despite other dependencies in the stdlib it's not a documented
        #   entrypoint, so needs type: ignore to satisfy mypy.
        threading._register_atexit(self.close)  # type: ignore

    def compile(self, graph: Graph) -> Future[engine.CompiledModel]:
        """Schedules ``graph`` for compilation.

        Returns:
            A future resolving to the compiled artifact, ready for
            :meth:`~max.engine.InferenceSession.init` on any session. The
            artifact is mmapped and owns its lifetime: it stays valid
            after the pool closes.

        Raises:
            RuntimeError: If the pool is closed.
            BrokenProcessPool: If a worker has already died.
        """
        return self._submit(
            graph._module.bytecode, graph.kernel_libraries_paths, None
        )

    def compile_module(
        self,
        module: Module,
        *,
        device_specs: Sequence[driver.DeviceSpec] | None = None,
        custom_extensions: Sequence[Path] = (),
    ) -> Future[engine.CompiledModel]:
        """Schedules a multi-graph :class:`~max.graph.Module` for compilation.

        Args:
            module: The module to compile; may hold several ``mo.graph`` ops
                that compile into one artifact (one MEF cache entry).
            device_specs: Devices for the compiling session, in order — the
                leading device is the session's default compile target and
                enters the artifact's cache key. Pass the exact devices (and
                order) a later in-process compile of the same module will
                use so it hits the shared MEF cache; ``None`` uses the
                pool's devices.
            custom_extensions: Paths to custom Mojo extension libraries.
                Note that compiles with extensions bypass the MEF cache.

        Returns:
            A future resolving to the compiled artifact, as :meth:`compile`.

        Raises:
            ValueError: If ``device_specs`` is provided but empty.
            RuntimeError: If the pool is closed.
            BrokenProcessPool: If a worker has already died.
        """
        if device_specs is not None and not device_specs:
            raise ValueError(
                "device_specs must name at least one device when provided;"
                " pass None to use the pool's devices"
            )
        return self._submit(
            module.mlir_module.bytecode,
            list(custom_extensions),
            tuple(device_specs) if device_specs is not None else None,
        )

    def _submit(
        self,
        bytecode: bytes,
        extensions: Sequence[Path],
        device_specs: tuple[driver.DeviceSpec, ...] | None,
    ) -> Future[engine.CompiledModel]:
        if self._closed:
            raise RuntimeError("the compile pool is closed")

        mef_future = self._executor.submit(
            _compile_to_mef,
            bytecode,
            extensions,
            Path(self._mef_dir.name) / f"{uuid.uuid4()}.mef",
            device_specs,
        )

        def load(mef_path: Path) -> engine.CompiledModel:
            try:
                compiled = engine.read(mef_path)
            except Exception:
                # A compile that resolves while close() runs can load
                # against the deleted MEF dir.
                if self._closed:
                    raise CancelledError() from None
                raise
            mef_path.unlink(missing_ok=True)
            return compiled

        return _map_future(mef_future, load)

    def close(self) -> None:
        """Stops the pool, discarding queued and in-flight compiles.

        Unfinished futures are cancelled or raise
        :class:`~concurrent.futures.process.BrokenProcessPool`.
        """
        if self._closed:
            return

        self._closed = True
        atexit.unregister(self.close)
        # Terminate before shutdown(): worker death after shutdown
        # doesn't resolve futures.
        for process in (self._executor._processes or {}).values():
            process.terminate()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._mef_dir.cleanup()

    def __enter__(self) -> ProcessCompilePool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Waits for outstanding queue items and then closes the pool."""
        self._executor.__exit__(exc_type, exc_val, exc_tb)
        self.close()


@functools.lru_cache
def pool() -> ProcessCompilePool:
    """Global singleton pool for compiling graphs in background processes."""
    return ProcessCompilePool()
