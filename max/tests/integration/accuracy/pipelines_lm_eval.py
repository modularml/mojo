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

import contextlib
import logging
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import click
import requests

# Make python.runfiles optional for Docker environments
try:
    import python.runfiles

    HAVE_RUNFILES = True
except ImportError:
    HAVE_RUNFILES = False

logger = logging.getLogger("pipelines_lm_eval")


def _must_rlocation_str(runfiles: python.runfiles.Runfiles, rloc: str) -> str:
    loc = runfiles.Rlocation(rloc)
    if loc is None:
        raise FileNotFoundError(
            f"Required rlocation {rloc!r} could not be resolved"
        )
    return loc


def _must_rlocation(runfiles: python.runfiles.Runfiles, rloc: str) -> Path:
    return Path(_must_rlocation_str(runfiles, rloc))


class PipelineSitter:
    """Owns the pipelines process and manages its startup/shutdown."""

    _args: Sequence[str]
    _extra_env: Mapping[str, str]
    _proc: subprocess.Popen | None  # type: ignore[type-arg]

    def __init__(
        self,
        args: Sequence[str],
        extra_env: Mapping[str, str] = {},
    ) -> None:
        self._args = args
        self._extra_env = extra_env
        self._proc = None

    def __enter__(self) -> PipelineSitter:
        self.start()
        return self

    def __exit__(
        self, exc_type: Any, exc_value: Any, exc_tb: Any
    ) -> Literal[False]:
        self.stop()
        return False

    def start(self) -> None:
        if self._proc:
            return
        logger.info(
            f"Starting pipelines process with provided args: {self._args}"
            f", extra env: {list(self._extra_env)}"
        )
        self._proc = subprocess.Popen(
            self._args, env={**os.environ, **self._extra_env}
        )
        logger.info("Pipelines process started")

    def stop(self) -> None:
        if not self._proc:
            return
        logger.info("Sending pipelines process SIGTERM")
        self._proc.terminate()
        logger.info("Waiting for pipelines process to terminate")
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Pipelines process did not terminate after 5 seconds, sending"
                " SIGKILL"
            )
            self._proc.kill()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Well, we tried our best.
                logger.error(
                    "Pipelines process still did not die; continuing anyway"
                )
            else:
                logger.info("Pipelines process terminated")
        else:
            logger.info("Pipelines process terminated")
        self._proc = None

    def is_alive(self) -> bool:
        """Return True if the managed process is still running."""
        if self._proc is None:
            return False
        return self._proc.poll() is None

    def wait_for_alive(self, probe_port: int, *, timeout: float | None) -> None:
        assert self._proc is not None
        probe_url = f"http://127.0.0.1:{probe_port}/health"
        start_time = time.time()
        deadline: float | None
        if timeout is None:
            deadline = None
        else:
            deadline = start_time + timeout
        logger.info("Waiting for pipelines server to begin accepting requests")
        while deadline is None or (now := time.time()) < deadline:
            try:
                self._proc.wait(timeout=0)
            except subprocess.TimeoutExpired:
                pass
            else:
                self._proc = None
                logger.error(
                    "Pipelines server died while waiting for readiness"
                )
                raise Exception(
                    "Pipelines server died while waiting for readiness"
                )

            probe_start_time = time.time()
            try:
                requests.get(probe_url, timeout=5)
            except Exception:
                elapsed_time = time.time() - probe_start_time
                sleep_duration = max(0, 5.0 - elapsed_time)
                if deadline is not None:
                    remaining_time = max(0, deadline - now)
                    sleep_duration = min(sleep_duration, remaining_time)
                time.sleep(sleep_duration)
            else:
                logger.info(
                    "Pipelines server seems to now be accepting requests"
                )
                return
        logger.error(
            "Pipelines server did not begin accepting requests within deadline"
            f" of {timeout} seconds"
        )
        raise Exception("Pipelines server did not come up within timeout")


def _kill_evaluator(proc: subprocess.Popen[bytes]) -> None:
    """Terminate an evaluator subprocess, escalating to SIGKILL if needed."""
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Evaluator did not terminate, sending SIGKILL")
        proc.kill()
        proc.wait(timeout=5)


def run_evaluator_with_crash_detection(
    evaluator_cmd: list[str],
    sitter: PipelineSitter,
    *,
    poll_interval: float = 60,
    health_probe_url: str | None = None,
    health_probe_timeout: float = 60.0,
    health_probe_max_consecutive_failures: int = 5,
) -> int:
    """Run an evaluator subprocess, aborting if the server crashes or hangs.

    Returns the evaluator's exit code, or 1 if the server died/hung.
    """
    consecutive_health_failures = 0
    evaluator_proc = subprocess.Popen(evaluator_cmd)
    try:
        while True:
            evaluator_rc = evaluator_proc.poll()
            if evaluator_rc is not None:
                logger.info(f"Evaluator exited with status code {evaluator_rc}")
                return evaluator_rc

            if not sitter.is_alive():
                logger.error(
                    "Pipelines server died while evaluator was still"
                    " running; terminating evaluator"
                )
                return 1

            # Active health probe: detect server hangs even when the
            # process is still alive (e.g. model worker OOM).
            if health_probe_url is not None:
                try:
                    resp = requests.get(
                        health_probe_url, timeout=health_probe_timeout
                    )
                    resp.raise_for_status()
                    consecutive_health_failures = 0
                except Exception:
                    consecutive_health_failures += 1
                    logger.warning(
                        f"Health probe failed ({consecutive_health_failures}/{health_probe_max_consecutive_failures}): {health_probe_url}"
                    )
                    if (
                        consecutive_health_failures
                        >= health_probe_max_consecutive_failures
                    ):
                        logger.error(
                            f"Health probe failed {consecutive_health_failures} consecutive times;"
                            " server appears hung, terminating evaluator"
                        )
                        return 1

            time.sleep(poll_interval)
    finally:
        if evaluator_proc.poll() is None:
            _kill_evaluator(evaluator_proc)


@click.command()
@click.option(
    "--override-pipelines",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option(
    "--skip-pipelines",
    is_flag=True,
    help=(
        "Skip starting pipelines server and connect to an external server instead. "
        "Requires an already-running pipelines server accessible via the evaluator's "
        "configured endpoint (typically http://HOST:PORT/v1/completions for lm-eval). "
        "The external server must be ready to accept requests before this script runs."
    ),
)
@click.option("--pipelines-probe-port", type=int)
@click.option("--pipelines-probe-timeout", type=float)
@click.option("--pipelines-arg", "pipelines_args", multiple=True)
@click.option("--evaluator", type=str, default="lm-eval")
@click.option(
    "--override-lm-eval",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--lm-eval-arg", "lm_eval_args", multiple=True)
@click.option(
    "--override-mistral-evals",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--mistral-evals-arg", "mistral_evals_args", multiple=True)
@click.option(
    "--override-longbench-v2",
    type=click.Path(
        exists=True, executable=True, dir_okay=False, path_type=Path
    ),
)
@click.option("--longbench-v2-arg", "longbench_v2_args", multiple=True)
@click.option(
    "--pipelines-health-timeout",
    type=int,
    default=120,
    help=(
        "Heartbeat timeout in seconds for detecting model-worker hangs. "
        "Also enables active health probing during evaluation. "
        "Set to 0 to disable."
    ),
)
def main(
    override_pipelines: Path | None,
    skip_pipelines: bool,
    pipelines_probe_port: int | None,
    pipelines_probe_timeout: float | None,
    pipelines_args: Sequence[str],
    evaluator: str,
    override_lm_eval: Path | None,
    lm_eval_args: Sequence[str],
    override_mistral_evals: Path | None,
    mistral_evals_args: Sequence[str],
    override_longbench_v2: Path | None,
    longbench_v2_args: Sequence[str],
    pipelines_health_timeout: int,
) -> None:
    """Start pipelines server, run an evaluator, and then shut down server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(name)s: %(message)s",
    )

    # Try to create runfiles (Bazel mode), fall back to None (Docker mode)
    runfiles = None
    if HAVE_RUNFILES:
        runfiles = python.runfiles.Create()
        if runfiles is None:
            logger.warning(
                "python.runfiles module is available but Create() returned None. "
                "This may indicate a Bazel configuration issue. "
                "Falling back to Docker mode paths."
            )

    # Determine program paths - use runfiles if available, fallback to Docker paths
    if override_pipelines is not None:
        pipelines_program = [str(override_pipelines)]
    elif runfiles is not None:
        # Bazel mode - use runfiles
        pipelines_program = [
            _must_rlocation_str(
                runfiles,
                "_main/max/python/max/_entrypoints/pipelines",
            )
        ]
    else:
        # Docker mode - use installed package
        pipelines_program = [
            "/opt/venv/bin/python",
            "-m",
            "max._entrypoints.pipelines",
        ]

    if override_lm_eval is not None:
        lm_eval_program = [str(override_lm_eval)]
    elif runfiles is not None:
        # Bazel mode - use runfiles
        lm_eval_program = [
            sys.executable,
            _must_rlocation_str(
                runfiles,
                "_main/max/tests/integration/accuracy/run_lm_eval.py",
            ),
        ]
    else:
        # Docker mode - use installed lm-eval
        lm_eval_program = ["/opt/venv/bin/lm_eval"]

    if override_mistral_evals is not None:
        mistral_evals_program = [str(override_mistral_evals)]
    elif runfiles is not None:
        # Bazel mode - use runfiles
        mistral_evals_program = [
            sys.executable,
            _must_rlocation_str(
                runfiles,
                "_main/max/tests/integration/accuracy/run_mistral_eval.py",
            ),
        ]
    else:
        # Docker mode - use installed mistral-evals
        mistral_evals_program = ["/opt/venv/bin/mistral-evals"]

    if override_longbench_v2 is not None:
        longbench_v2_program = [str(override_longbench_v2)]
    elif runfiles is not None:
        # Bazel mode - use runfiles
        longbench_v2_program = [
            sys.executable,
            _must_rlocation_str(
                runfiles,
                "_main/max/tests/integration/accuracy/run_longbench_v2.py",
            ),
        ]
    else:
        # Docker mode - use script directly
        longbench_v2_program = [
            "/opt/venv/bin/python",
            "/app/max/tests/integration/accuracy/run_longbench_v2.py",
        ]
    logger.debug("Pipelines binary at: %r", pipelines_program)
    evaluator_args: list[str] = []
    if evaluator == "lm-eval":
        evaluator_program = lm_eval_program
        evaluator_args.extend(lm_eval_args)
    elif evaluator == "mistral-evals":
        evaluator_program = mistral_evals_program
        evaluator_args.extend(mistral_evals_args)
    elif evaluator == "longbench-v2":
        evaluator_program = longbench_v2_program
        evaluator_args.extend(longbench_v2_args)
    else:
        logger.error("Unrecognized evaluator %r", evaluator)
        sys.exit(1)
    logger.debug("Evaluator binary at: %r", evaluator_program)

    # Add eval_tasks directory to include_path for lm-eval
    if evaluator == "lm-eval" and not any(
        arg.startswith("--include_path") for arg in evaluator_args
    ):
        include_path: Path
        if runfiles is not None:
            # Bazel mode - use runfiles
            include_path = _must_rlocation(
                runfiles,
                "_main/max/tests/integration/accuracy/eval_tasks/BUILD.bazel",
            ).parent
        else:
            # Docker mode - use absolute path
            include_path = Path(
                "/app/max/tests/integration/accuracy/eval_tasks"
            )
            if not include_path.exists():
                logger.error(
                    "Required eval_tasks directory not found at %s. "
                    "Docker image may be misconfigured.",
                    include_path,
                )
                sys.exit(1)

        evaluator_args.append(f"--include_path={include_path}")
        logger.debug("Including path: %s", include_path)

    # Run evaluator with or without managing pipelines server
    with contextlib.ExitStack() as stack:
        if skip_pipelines:
            # Skip pipelines server management - assume external server is running
            # The evaluator will connect to the external server using the endpoint
            # configured in its arguments (e.g., via --lm-eval-arg with base_url).
            # Typically used in dataset-eval jobs where a separate Kubernetes pod
            # runs the pipelines server (see dataset_eval_entrypoint.sh).
            logger.info(
                "Skipping pipelines server startup, assuming external server is running"
            )
        else:
            # Manage pipelines server lifecycle
            pipelines_env: dict[str, str] = {}
            if pipelines_health_timeout > 0:
                pipelines_env = {
                    "MAX_SERVE_USE_HEARTBEAT": "true",
                    "MAX_SERVE_MW_HEALTH_FAIL": str(pipelines_health_timeout),
                }
                logger.info(
                    f"Heartbeat enabled with {pipelines_health_timeout}s timeout"
                )
            pipeline_sitter = stack.enter_context(
                PipelineSitter(
                    pipelines_program + list(pipelines_args),
                    extra_env=pipelines_env,
                )
            )
            if pipelines_probe_port is not None:
                pipeline_sitter.wait_for_alive(
                    probe_port=pipelines_probe_port,
                    timeout=pipelines_probe_timeout,
                )

        logger.info(
            f"Running evaluator {evaluator_program!r} with provided args: {evaluator_args!r}"
        )

        if skip_pipelines:
            # No local server to monitor — blocking run is fine.
            evaluator_proc = subprocess.run(evaluator_program + evaluator_args)
            logger.info(
                f"Evaluator exited with status code {evaluator_proc.returncode}"
            )
            sys.exit(evaluator_proc.returncode)

        # Monitor both the evaluator and the pipelines server so we can
        # exit early if the server crashes instead of blocking forever.
        health_probe_url: str | None = None
        if pipelines_probe_port is not None and pipelines_health_timeout > 0:
            health_probe_url = f"http://127.0.0.1:{pipelines_probe_port}/health"
        sys.exit(
            run_evaluator_with_crash_detection(
                evaluator_program + evaluator_args,
                pipeline_sitter,
                health_probe_url=health_probe_url,
            )
        )


if __name__ == "__main__":
    main()
