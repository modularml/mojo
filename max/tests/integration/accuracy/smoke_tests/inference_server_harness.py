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
from __future__ import annotations

import logging
import os
import signal
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from subprocess import Popen, TimeoutExpired

import requests

logger = logging.getLogger(__name__)


def _inside_bazel() -> bool:
    return os.getenv("BUILD_WORKSPACE_DIRECTORY") is not None


def _default_server_is_ready() -> bool:
    try:
        return (
            requests.get("http://127.0.0.1:8000/health", timeout=1).status_code
            == 200
        )
    except requests.exceptions.RequestException:
        return False


def _gracefully_stop(process: Popen[bytes]) -> None:
    start_time = time.time()
    process.send_signal(signal.SIGINT)
    try:
        process.wait(25)
        shutdown_seconds = int(time.time() - start_time)
        logger.info(f"Server shutdown took {shutdown_seconds} seconds")
    except TimeoutExpired:
        logger.warning("Server did not stop after ctrl-c, trying SIGTERM")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(5)
        except ProcessLookupError:
            pass
        except TimeoutExpired:
            logger.warning("Process did not terminate gracefully, forcing kill")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            process.wait(5)


@dataclass
class RunningServer:
    process: Popen[bytes]
    startup_time: float


@contextmanager
def start_server(
    cmd: list[str],
    timeout: int,
    *,
    readiness_probe: Callable[[], bool] | None = None,
    poll_interval: float = 0.5,
    cwd: str | os.PathLike[str] | None = None,
    env_overrides: Mapping[str, str] | None = None,
) -> Generator[RunningServer, None, None]:
    """Spawn the server command and wait until it accepts requests.

    Args:
        cmd: Argv to spawn (e.g. ``[python, -m, max..., serve, ...]``).
        timeout: Seconds to wait for ``readiness_probe`` to succeed.
        readiness_probe: Callable returning True once the server is ready.
            Defaults to a GET on ``http://127.0.0.1:8000/health``.
        poll_interval: Seconds between readiness checks.
        cwd: Working directory for the spawned process. Required when
            ``cmd`` references workspace-relative paths (e.g. ``./bazelw``).
        env_overrides: Extra environment variables to set for the spawned
            process, layered on top of the inherited environment.
    """
    probe = readiness_probe or _default_server_is_ready
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    if not _inside_bazel():
        # SGLang depends on ninja which is in the serve environment
        env["PYTHONSAFEPATH"] = "1"  # Avoids root dir `max` shadowing
        venv_bin = os.path.abspath(".venv-serve/bin")
        prev_path = env.get("PATH")
        env["PATH"] = f"{venv_bin}:{prev_path}" if prev_path else venv_bin

    start = time.monotonic()
    proc = Popen(cmd, start_new_session=True, env=env, cwd=cwd)
    try:
        deadline = start + timeout
        while time.monotonic() < deadline:
            if probe():
                break
            if proc.poll() is not None:
                raise RuntimeError("Server process terminated unexpectedly")
            time.sleep(poll_interval)
        else:
            raise TimeoutError(f"Server did not start in {timeout} seconds")
        yield RunningServer(proc, time.monotonic() - start)
    finally:
        if proc.poll() is None:
            try:
                _gracefully_stop(proc)
            except Exception:
                logger.exception("Failed to stop server")
