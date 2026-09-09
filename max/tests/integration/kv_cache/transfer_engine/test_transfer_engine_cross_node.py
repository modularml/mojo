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

"""Cross-node KV transfer engine smoke test.

Two independent processes — one per node — no orchestrator:
  sender    sender node
  receiver  receiver node

See MLRT/docs/Driver/NIXL/validate_upstream_sync.md for the full
parameter sweep table and expected output.

Usage
-----
    # Receiver node
    MODULAR_NIXL_TRANSFER_BACKEND=libfabric FI_EFA_USE_DEVICE_RDMA=1 \\
      ./bazelw run //max/tests/integration/kv_cache/transfer_engine:test_transfer_engine_cross_node -- \\
          --role receiver \\
          --sender-addr tcp://<sender-node-ip>:5555

    # Sender node
    MODULAR_NIXL_TRANSFER_BACKEND=libfabric FI_EFA_USE_DEVICE_RDMA=1 \\
      ./bazelw run //max/tests/integration/kv_cache/transfer_engine:test_transfer_engine_cross_node -- \\
          --role sender \\
          --sender-addr tcp://<sender-node-ip>:5555
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from urllib.parse import urlparse

logging.basicConfig(level=logging.WARNING)

import msgspec
import numpy as np
import zmq
from max.driver import Accelerator
from max.driver import __version__ as max_version
from max.driver.buffer import Buffer
from max.dtype import DType
from max.nn.kv_cache.cache_params import KVCacheMemory
from max.pipelines.kv_cache import (
    KVTransferEngine,
    KVTransferEngineMetadata,
    TransferReqData,
)


def _view(buf: Buffer, total_num_pages: int) -> Buffer:
    """View a buffer as a 2-D uint8 ``[total_num_pages, bytes_per_page]`` array."""
    bytes_per_page = (
        buf.num_elements * buf.dtype.size_in_bytes // total_num_pages
    )
    return buf.view(DType.uint8, [total_num_pages, bytes_per_page])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=["sender", "receiver"], required=True)
    p.add_argument(
        "--sender-addr",
        default=os.environ.get("MAX_SERVE_DI_BIND_ADDRESS"),
        required=os.environ.get("MAX_SERVE_DI_BIND_ADDRESS") is None,
        help="ZMQ address the sender binds (receiver connects to this). "
        "Defaults to MAX_SERVE_DI_BIND_ADDRESS if set.",
    )
    p.add_argument("--bytes-per-page", type=int, default=65536)
    p.add_argument(
        "--num-pages",
        type=int,
        default=None,
        help="GPU buffer size in pages (default: concurrency x pages-per-request).",
    )
    p.add_argument("--pages-per-request", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Number of GPU shards (tensor-parallel degree).",
    )
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--warmup-batches", type=int, default=5)
    p.add_argument(
        "--backend", choices=["libfabric", "ucx", "uccl"], default="libfabric"
    )
    p.add_argument(
        "--min-bandwidth-gbps",
        type=float,
        default=10.0,
        help="Minimum acceptable bandwidth in GiB/s (default: 10.0).",
    )
    p.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Buffer placement: gpu (default) or cpu (host DRAM; lets the "
        "transfer run on GPU-less hosts with a CPU-flavor plugin).",
    )
    return p.parse_args()


def set_env_vars(args: argparse.Namespace) -> None:
    os.environ.setdefault("MODULAR_NIXL_TRANSFER_BACKEND", args.backend)
    if args.backend == "libfabric":
        os.environ.setdefault("FI_EFA_USE_DEVICE_RDMA", "1")
    # 90 matches DI benchmark configs; 99 works here since there's no NVSHMEM+EP.
    os.environ.setdefault(
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT", "90"
    )


@dataclass
class WorkloadConfig:
    """Workload parameters matched to the target DI benchmark config."""

    bytes_per_page: int  # size of one KV cache page in bytes
    num_pages: int  # pages per GPU buffer
    concurrency: int  # number of in-flight transfers per batch
    pages_per_request: int  # pages transferred per simulated request
    tp_size: int  # number of GPU shards


def _allocate_device_buffers(
    role: str, cfg: WorkloadConfig, device_kind: str
) -> list[Buffer]:
    """Allocate one buffer per TP shard, on GPU or host DRAM.

    Sender pages are filled with sentinel values (page i gets value i+1) so
    scatter bugs are detectable on the receiver. Receiver buffers are zeroed.
    """
    buffers = []
    for rank in range(cfg.tp_size):
        total = cfg.num_pages * cfg.bytes_per_page
        if role == "sender":
            buf = np.empty(total, dtype=np.int8)
            for i in range(cfg.num_pages):
                buf[i * cfg.bytes_per_page : (i + 1) * cfg.bytes_per_page] = (
                    i % 127
                ) + 1
        else:
            buf = np.zeros(total, dtype=np.int8)
        host_buffer = Buffer.from_numpy(buf)
        if device_kind == "gpu":
            buffers.append(host_buffer.to(Accelerator(rank)))
        else:
            buffers.append(host_buffer)
    return buffers


def _setup_zmq(role: str, sender_addr: str) -> zmq.Socket:  # type: ignore[type-arg]
    """Bind/connect the ZMQ PAIR socket and wait for both sides to be up.

    Establishing the ZMQ connection before creating the KVTransferEngine
    ensures the NIXL engine's EFA endpoint has a peer in its AV table before
    the CM thread starts polling — matching production's initialization order.
    """
    port = urlparse(sender_addr).port
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PAIR)
    sock.ipv6 = True
    if role == "sender":
        sock.bind(f"tcp://[::]:{port}")
        sock.recv()  # wait for receiver to be ready
        sock.send(b"ready")
    else:
        sock.connect(sender_addr)
        sock.send(b"ready")
        sock.recv()  # wait for sender acknowledgement
    return sock


def _exchange_engine_metadata(
    role: str,
    sock: zmq.Socket,  # type: ignore[type-arg]
    engine: KVTransferEngine,
) -> KVTransferEngineMetadata:
    """Exchange KVTransferEngineMetadata over an established ZMQ socket.

    Called after both peers have created their KVTransferEngine so the NIXL
    EFA endpoint already has the peer in its AV table when engine.connect()
    is called, preventing the init-time RECV error.
    """
    local_md_bytes = msgspec.json.encode(engine.metadata)
    if role == "sender":
        sock.send(local_md_bytes)
        remote_md_bytes = sock.recv()
    else:
        remote_md_bytes = sock.recv()
        sock.send(local_md_bytes)

    remote_md = msgspec.json.decode(
        remote_md_bytes, type=KVTransferEngineMetadata
    )
    engine.connect(remote_md)
    return remote_md


def run_sender(
    args: argparse.Namespace,
    cfg: WorkloadConfig,
    engine: KVTransferEngine,
    remote_md: KVTransferEngineMetadata,
    sock: zmq.Socket,  # type: ignore[type-arg]
) -> None:
    """Drive concurrent KV transfers and report bandwidth.

    Fires cfg.concurrency transfers per batch (each transfer moves
    cfg.pages_per_request pages). Runs warmup_batches silently, then times
    num_batches and asserts bandwidth >= args.min_bandwidth_gbps.
    """
    GiB = 1024**3

    if cfg.num_pages < cfg.concurrency * cfg.pages_per_request:
        raise ValueError(
            f"num_pages ({cfg.num_pages}) must be >= "
            f"concurrency ({cfg.concurrency}) * pages_per_request "
            f"({cfg.pages_per_request}) = "
            f"{cfg.concurrency * cfg.pages_per_request}"
        )

    requests: list[tuple[list[int], list[int]]] = []
    for i in range(cfg.concurrency):
        start = i * cfg.pages_per_request
        idxs = list(range(start, start + cfg.pages_per_request))
        requests.append((idxs, idxs))

    def run_batch() -> None:
        in_flight: list[TransferReqData] = []
        for src_idxs, dst_idxs in requests:
            req = engine.initiate_send_transfer(
                remote_md,
                src_idxs,
                dst_idxs,
                src_replica_idx=0,
                dst_replica_idx=0,
            )
            sock.send(msgspec.json.encode(req))
            in_flight.append(req)

        for req in in_flight:
            while not engine.is_complete(req):
                pass
            engine.cleanup_transfer(req)

    # Warmup — not timed.
    for i in range(args.warmup_batches):
        run_batch()
        print(f"[Sender] warmup {i + 1}/{args.warmup_batches}", flush=True)

    bytes_per_batch = (
        cfg.concurrency
        * cfg.tp_size
        * cfg.pages_per_request
        * cfg.bytes_per_page
    )
    total_bytes = bytes_per_batch * args.num_batches

    t0 = time.time()
    for i in range(args.num_batches):
        run_batch()
        elapsed_so_far = time.time() - t0
        bw_so_far = (bytes_per_batch * (i + 1)) / elapsed_so_far / GiB
        print(
            f"[Sender] batch {i + 1}/{args.num_batches} "
            f"({bw_so_far:.1f} GiB/s so far)",
            flush=True,
        )
    elapsed = time.time() - t0

    bw_gbs = total_bytes / elapsed / GiB
    print(
        f"[Sender] {args.num_batches} batches x {cfg.concurrency} transfers: "
        f"{total_bytes / GiB:.2f} GiB in {elapsed * 1000:.1f} ms "
        f"({bw_gbs:.2f} GiB/s)"
    )
    # Wait for receiver to finish processing all notifications before tearing
    # down the NIXL engine. Without this the sender exits while RNDV
    # notification messages are still in-flight, leaving the receiver stuck.
    sock.send(b"done")
    sock.recv()  # wait for receiver ACK
    assert bw_gbs >= args.min_bandwidth_gbps, (
        f"Bandwidth {bw_gbs:.2f} GiB/s below minimum {args.min_bandwidth_gbps} GiB/s"
    )


def run_receiver(
    args: argparse.Namespace,
    cfg: WorkloadConfig,
    engine: KVTransferEngine,
    all_blocks: list[Buffer],
    sock: zmq.Socket,  # type: ignore[type-arg]
) -> None:
    """Accept incoming KV transfers and validate content integrity.

    Mirrors run_sender: receives one TransferReqData per in-flight transfer
    per batch over ZMQ, polls engine.is_complete() on each, then calls
    engine.cleanup_transfer(). After the final batch, validates that each
    received page contains the expected sentinel value (sender page i has
    value (i % 127) + 1) across all GPU shards.
    """
    total_batches = args.warmup_batches + args.num_batches

    for batch_idx in range(total_batches):
        in_flight: list[TransferReqData] = []
        for _ in range(cfg.concurrency):
            raw = sock.recv()
            req = msgspec.json.decode(raw, type=TransferReqData)
            in_flight.append(req)

        for req in in_flight:
            while not engine.is_complete(req):
                pass
            engine.cleanup_transfer(req)

        if batch_idx == total_batches - 1:
            for blocks in all_blocks:
                result = blocks.to_numpy()
                for req in in_flight:
                    for src_idx, dst_idx in zip(
                        req.src_idxs, req.dst_idxs, strict=True
                    ):
                        expected = (src_idx % 127) + 1
                        page = result[
                            dst_idx * cfg.bytes_per_page : (dst_idx + 1)
                            * cfg.bytes_per_page
                        ]
                        assert (page == expected).all(), (
                            f"Page {dst_idx}: expected {expected}, got {page[0]}"
                        )
    # Signal sender that all transfers are processed and it is safe to exit.
    sock.recv()  # wait for sender "done"
    print("[Receiver] all transfers complete", flush=True)
    sock.send(b"ack")


def main() -> None:
    args = parse_args()
    # The pods mount this script from the CI checkout but import the `max`
    # installed in the engine image. Naming that version up front keeps a
    # checkout/image skew from surfacing only as an unrelated API error deep in
    # engine construction.
    print(f"[{args.role}] MAX engine version: {max_version}", flush=True)
    set_env_vars(args)
    num_pages = (
        args.num_pages
        if args.num_pages is not None
        else args.concurrency * args.pages_per_request
    )
    cfg = WorkloadConfig(
        bytes_per_page=args.bytes_per_page,
        num_pages=num_pages,
        concurrency=args.concurrency,
        pages_per_request=args.pages_per_request,
        tp_size=args.tp_size,
    )
    # Phase 1: establish ZMQ connection before creating the NIXL engine so
    # the EFA endpoint has a peer in its AV table from the start.
    sock = _setup_zmq(args.role, args.sender_addr)
    all_blocks = _allocate_device_buffers(args.role, cfg, args.device)
    engine = KVTransferEngine(
        f"engine_{args.role}",
        [
            [
                KVCacheMemory(
                    replicated=False,
                    buffers=[_view(b, cfg.num_pages) for b in all_blocks],
                )
            ]
        ],
    )
    # Phase 2: exchange NIXL engine metadata and activate the RDMA path.
    remote_md = _exchange_engine_metadata(args.role, sock, engine)
    if args.role == "sender":
        run_sender(args, cfg, engine, remote_md, sock)
    else:
        run_receiver(args, cfg, engine, all_blocks, sock)
    engine.cleanup()


if __name__ == "__main__":
    main()
