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
import multiprocessing
import multiprocessing.process
import os
import time
from collections.abc import Sequence

import click
import httpx


def do_serve(args: Sequence[str]) -> None:
    from max._entrypoints.pipelines import main

    # We call "main" instead of "cli_serve" because it does some extra
    # initialization that it would be best not to duplicate here.
    ctx = main.make_context("max-serve", ["serve"] + list(args))
    with ctx:
        main.invoke(ctx)


def do_replay(args: Sequence[str]) -> None:
    from max._entrypoints.replay_recording import main

    ctx = main.make_context("max-replay-recording", list(args))
    with ctx:
        main.invoke(ctx)


def ensure_dead(process: multiprocessing.process.BaseProcess) -> None:
    if not process.is_alive():
        return
    process.terminate()
    try:
        # Give it 15 seconds to shut down cleanly
        process.join(timeout=15)
    except Exception:
        pass  # Ignore join failures
    if not process.is_alive():
        return
    process.kill()
    process.join()


def wait_for_up(
    url: str,
    process: multiprocessing.process.BaseProcess,
    *,
    timeout: float | None = 600,
) -> None:
    start_time = time.time()
    with httpx.Client() as client:
        while True:
            if not process.is_alive():
                raise SystemError("Process died before coming up")
            try:
                r = client.get(url)
                r.raise_for_status()
            except Exception:
                pass  # Not up yet, I guess
            else:
                break
            poll_time = 5.0
            if timeout is not None:
                time_left = time.time() - start_time
                if time_left < 0:
                    raise TimeoutError(
                        f"Process did not come up within {timeout} seconds"
                    )
                poll_time = min(poll_time, time_left)
            try:
                process.join(timeout=poll_time)
            except Exception:
                # We really just wanted to wait with an early-exit, this is OK
                pass


@click.command
@click.option("--probe-url", default="http://localhost:8000/v1/health")
@click.option("--probe-timeout", type=click.FloatRange(min=0), default=600)
@click.argument("sub_args", nargs=-1, type=str)
def main(probe_url: str, probe_timeout: float, sub_args: Sequence[str]) -> None:
    if "--" not in sub_args:
        raise click.BadParameter(
            "usage: serve_replay.py -- <serve args> -- <replay args>"
        )
    sep_index = sub_args.index("--")
    serve_args, replay_args = sub_args[:sep_index], sub_args[sep_index + 1 :]

    with contextlib.ExitStack() as exit_stack:
        mp_ctx = multiprocessing.get_context("spawn")
        serve_process = mp_ctx.Process(target=do_serve, args=(serve_args,))
        exit_stack.callback(serve_process.close)
        exit_stack.callback(ensure_dead, serve_process)
        serve_process.start()
        wait_for_up(probe_url, serve_process, timeout=probe_timeout)
        do_replay(replay_args)


if __name__ == "__main__":
    if directory := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(directory)

    main()
