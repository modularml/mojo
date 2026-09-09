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

"""Orchestrator for the cross-node KVTransferEngine smoke test.

Launches sender and receiver concurrently on two hosts (or localhost for
single-machine runs), streams prefixed output, and reports bandwidth.

Usage
-----
    # Two-node EFA run
    ./bazelw run //max/tests/integration/kv_cache/transfer_engine:run_cross_node -- \\
        --sender-host <node0-ip> --receiver-host <node1-ip>

    # Single-machine (loopback, no EFA)
    ./bazelw run //max/tests/integration/kv_cache/transfer_engine:run_cross_node -- \\
        --sender-host localhost --receiver-host localhost
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from typing import IO

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# Flags that belong to this orchestrator and must NOT be forwarded to the
# smoke test binary.
_ORCHESTRATOR_KEYS = {"sender_host", "receiver_host", "sender_port"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)

    # Orchestrator-only flags
    p.add_argument("--sender-host", required=True)
    p.add_argument("--receiver-host", required=True)
    p.add_argument("--sender-port", type=int, default=5555)

    # Passthrough flags — mirror test_transfer_engine_cross_node.parse_args()
    # excluding --role and --sender-addr (set by this orchestrator).
    p.add_argument("--bytes-per-page", type=int, default=65536)
    p.add_argument("--num-pages", type=int, default=None)
    p.add_argument("--pages-per-request", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--tp-size", type=int, default=1)
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument(
        "--backend", choices=["libfabric", "ucx", "uccl"], default="libfabric"
    )
    p.add_argument(
        "--min-bandwidth-gbps",
        type=float,
        default=10.0,
        help="Forwarded to the sender; asserted inside the smoke test.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Command helpers
# ---------------------------------------------------------------------------


def _local_or_ssh(host: str, shell_cmd: str) -> list[str]:
    """Return a subprocess argv that runs shell_cmd on host."""
    if host in ("localhost", "127.0.0.1"):
        return ["bash", "-c", shell_cmd]
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        host,
        shell_cmd,
    ]


def _build_smoke_cmd(role: str, args: argparse.Namespace) -> str:
    """Return the shell command string to run on the remote (or local) host."""
    env = "MODULAR_NIXL_TRANSFER_BACKEND=" + args.backend
    if args.backend == "libfabric":
        env += " FI_EFA_USE_DEVICE_RDMA=1"

    # SSH sessions start from $HOME, not the repo, so prefix with `cd`.
    # Check MODULAR_PATH (set by _run_distw callers) then
    # BUILD_WORKSPACE_DIRECTORY (set automatically by `bazel run`) — the same
    # priority order used in run_dist.py for its local cwd= subprocess.
    repo_dir = os.environ.get("MODULAR_PATH") or os.environ.get(
        "BUILD_WORKSPACE_DIRECTORY", ""
    )
    cd_prefix = f"cd {repo_dir} && " if repo_dir else ""

    # The env assignments must directly prefix the bazelw invocation: in
    # `VAR=x cd repo && cmd`, the assignment binds to `cd` only and never
    # reaches cmd.
    target = (
        f"{cd_prefix}{env} ./bazelw run"
        " //max/tests/integration/kv_cache/transfer_engine:test_transfer_engine_cross_node"
        " --"
    )

    fixed = [
        f"--role {role}",
        f"--sender-addr tcp://{args.sender_host}:{args.sender_port}",
    ]

    passthrough = []
    for key, value in vars(args).items():
        if key in _ORCHESTRATOR_KEYS or key == "backend" or value is None:
            continue
        flag = "--" + key.replace("_", "-")
        passthrough.append(f"{flag} {value}")

    return " ".join([target] + fixed + passthrough)


# ---------------------------------------------------------------------------
# Output streaming
# ---------------------------------------------------------------------------


def _stream(pipe: IO[str], prefix: str) -> None:
    """Read lines from pipe and print them with a role prefix."""
    for line in pipe:
        print(f"{prefix} {line.rstrip()}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    sender_addr = f"tcp://{args.sender_host}:{args.sender_port}"

    sender_cmd = _local_or_ssh(
        args.sender_host, _build_smoke_cmd("sender", args)
    )
    receiver_cmd = _local_or_ssh(
        args.receiver_host, _build_smoke_cmd("receiver", args)
    )

    print(f"[orchestrator] sender   → {args.sender_host}")
    print(f"[orchestrator] receiver → {args.receiver_host}")
    print(f"[orchestrator] addr     → {sender_addr}")
    print(f"[orchestrator] sender   cmd: {' '.join(sender_cmd)}")
    print(f"[orchestrator] receiver cmd: {' '.join(receiver_cmd)}", flush=True)

    sender_proc = subprocess.Popen(
        sender_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    receiver_proc = subprocess.Popen(
        receiver_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    for proc, prefix in [
        (sender_proc, "[sender]"),
        (receiver_proc, "[receiver]"),
    ]:
        threading.Thread(
            target=_stream, args=(proc.stdout, prefix), daemon=True
        ).start()

    rc_sender = sender_proc.wait()
    rc_receiver = receiver_proc.wait()

    print(
        f"[sender]   {'PASS' if rc_sender == 0 else 'FAIL'} (exit {rc_sender})"
    )
    print(
        f"[receiver] {'PASS' if rc_receiver == 0 else 'FAIL'} (exit {rc_receiver})"
    )

    if rc_sender != 0 or rc_receiver != 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
