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

"""KVCache transfer engine."""

from __future__ import annotations

import ctypes
import logging
import os
import random
import socket
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple
from uuid import uuid4

if TYPE_CHECKING:
    from max.nn.kv_cache.cache_params import KVCacheMemory

import msgspec
from max._core import nixl
from max.driver import Buffer, Device
from max.pipelines.kv_cache._nixl_backend import (
    NIXL_BACKEND_ENV_VAR,
    NixlBackendType,
    validate_nixl_backend,
)
from max.pipelines.kv_cache._nixl_plugin_deps import preload_nixl_plugin_deps

from ._ucx_env import configure_ucx_env
from .cache_manager import PagedKVCacheManagerInterface

logger = logging.getLogger("max.pipelines")


def _plugin_load_error(upstream_backend_type: str) -> str | None:
    """Returns the dynamic-loader error behind an unloadable NIXL plugin.

    Upstream NIXL reports a plugin whose ``dlopen`` failed exactly as it reports
    one that does not exist -- ``NIXL_ERR_NOT_FOUND`` -- and logs the loader
    error at INFO, below its default WARN level. Retrying the load here recovers
    that message, which is what distinguishes a broken runtime dependency (a
    conflicting SONAME already in the process, a missing driver library) from a
    packaging gap.

    Returns ``None`` when the plugin loads here, so the failure lies elsewhere,
    or when no plugin directory is set.
    """
    plugin_dir = os.environ.get("NIXL_PLUGIN_DIR")
    if not plugin_dir:
        return None
    path = os.path.join(plugin_dir, f"libplugin_{upstream_backend_type}.so")
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
    except OSError as e:
        return str(e)
    return None


def _get_nixl_backend_type() -> NixlBackendType:
    """Returns the NIXL backend type from the environment.

    Reads ``MODULAR_NIXL_TRANSFER_BACKEND`` (default ``"ucx"``). The default
    is this engine's, not the validator's: the dKV connector reads the same
    variable through the same validator but auto-selects when it is unset.
    """
    return validate_nixl_backend(os.environ.get(NIXL_BACKEND_ENV_VAR, "ucx"))


def _default_uccl_socket_ifname_if_unset() -> None:
    """Pins UCCL's out-of-band bootstrap to the host's default-route NIC.

    UCCL derives the IP it advertises for its TCP bootstrap from the socket
    interface it selects (``UCCL_SOCKET_IFNAME``, then ``NCCL_SOCKET_IFNAME``);
    when neither is set, its auto-pick can land on a RoCE rail. On a fabric
    that carries RoCE only -- filtering rail TCP at the switch -- the peer's
    bootstrap dial then hangs forever even though the RDMA data plane is
    healthy. Default the interface to the host's default-route NIC (the
    control network, which peers can reach) when the operator has not chosen
    one; RDMA HCA selection is independent, so KV data still rides the rails.
    An explicitly set ``UCCL_SOCKET_IFNAME``/``NCCL_SOCKET_IFNAME`` always wins.
    """
    if os.environ.get("UCCL_SOCKET_IFNAME") or os.environ.get(
        "NCCL_SOCKET_IFNAME"
    ):
        return
    # Field 0 (interface) of the /proc/net/route row whose destination is
    # 0.0.0.0 and whose flags have RTF_GATEWAY set (0x2) is the default route.
    try:
        with open("/proc/net/route") as route_table:
            next(route_table, None)  # header row
            for line in route_table:
                cols = line.split()
                if (
                    len(cols) > 3
                    and cols[1] == "00000000"
                    and int(cols[3], 16) & 0x2
                ):
                    os.environ["UCCL_SOCKET_IFNAME"] = cols[0]
                    logger.info(
                        "Defaulting UCCL_SOCKET_IFNAME to %s so UCCL's "
                        "bootstrap does not ride the RDMA rails; set "
                        "UCCL_SOCKET_IFNAME explicitly to override.",
                        cols[0],
                    )
                    return
    except OSError:
        pass


def _default_uccl_p2p_env_if_unset() -> None:
    """Forces UCCL's RDMA transport to avoid its broken same-host IPC path.

    UCCL's cross-process peer-to-peer IPC path is broken upstream, so a
    same-host (intranode) sender/receiver pair deadlocks unless UCCL uses its
    RDMA transport instead. ``UCCL_P2P_TRANSPORT=rdma`` and
    ``UCCL_P2P_DISABLE_IPC=1`` force that; both are no-ops internode, where
    RDMA is already the only path. An explicitly set value always wins.
    """
    for name, value in (
        ("UCCL_P2P_TRANSPORT", "rdma"),
        ("UCCL_P2P_DISABLE_IPC", "1"),
    ):
        if not os.environ.get(name):
            os.environ[name] = value
            logger.info("Defaulting %s=%s for UCCL.", name, value)


def available_port(
    start_port: int = 8000, end_port: int = 9000, max_attempts: int = 100
) -> int:
    """Finds an available TCP port in the given range.

    Args:
        start_port: The lower bound of the port range (inclusive).
        end_port: The upper bound of the port range (inclusive).
        max_attempts: Maximum number of attempts to find a free port.

    Returns:
        int: An available port number.

    Raises:
        RuntimeError: If no available port is found after max_attempts.
    """
    for _ in range(max_attempts):
        port = random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            # Set SO_REUSEADDR to avoid TIME_WAIT issues
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found in the specified range.")


def _validate_device_type(devices: Sequence[Device]) -> None:
    is_gpu = False
    is_cpu = False
    for d in devices:
        if d.is_host:
            is_cpu = True
        else:
            is_gpu = True

    if is_cpu and is_gpu:
        raise ValueError(
            "Mixed device tensors detected. All tensors must be either on CPU or GPU, not both."
        )

    first_device = devices[0]
    if not first_device.is_host and (
        "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT" not in os.environ
        and "BAZEL_TEST" not in os.environ
    ):
        # See GEX-2445 for more details.
        # We intentionally make falling back to the slower CUDA_COPY transport
        # a hard error. This check is best effort. Just because it is not
        # tripped does not guarantee that the we will end up using CUDA_IPC.
        # Note that we will use MemoryManager regardless when running under
        # bazel test.
        raise ValueError(
            "MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT must be set when using TransferEngine with GPU memory. "
            "This flag enables the MemoryManager which is required for the fast CUDA_IPC transport. "
            "Try rerunning your command with MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=99"
        )


def _validate_tensor_shape(tensors: Sequence[Buffer]) -> int:
    """Return the per-page byte size shared by every shard of a NIXL group.

    Each buffer is the 2-D ``uint8`` view that ``to_memory()`` produced, with
    ``shape == [total_num_pages, bytes_per_page]``.  The per-page stride is
    therefore just ``shape[1]`` -- the producer already folded the null block
    into ``shape[0]`` and divided the byte count by the *allocated* page count,
    so there is nothing to recompute here (and no separately-threaded
    ``total_num_pages`` to keep in sync).

    Every shard of a group must have an identical shape; for these uint8 views
    shape-equality subsumes the old per-element-count and dtype checks.
    """
    first_tensor = tensors[0]
    first_shape = tuple(first_tensor.shape)
    for i, tensor in enumerate(tensors[1:], 1):
        if tuple(tensor.shape) != first_shape:
            raise ValueError(
                f"All tensors must have the same shape. Tensor 0 has shape "
                f"{first_shape}, but Tensor {i} has shape {tuple(tensor.shape)}"
            )

    # shape == [total_num_pages, bytes_per_page]; the stride is dim 1.
    return first_shape[1]


def _build_group_descriptors(
    base_addrs: Sequence[int],
    bytes_per_group: Sequence[int],
    page_idxs: Sequence[int],
    device_id: int,
) -> list[tuple[int, int, int]]:
    """Build NIXL ``(addr, size, device)`` descriptors for all groups.

    For each group ``g`` and page index ``i``, emits
    ``(base_addrs[g] + i * bytes_per_group[g], bytes_per_group[g], device_id)``,
    iterating group-major then page-index (the order the paired src/dst
    descriptor lists rely on).

    Each group uses its OWN base address and per-page stride, so groups with
    different ``bytes_per_page`` never share addressing -- the invariant that
    guards against the draft-KV stride-mismatch class (SERVOPT-1456).
    """
    descs: list[tuple[int, int, int]] = []
    for group_idx, bpp in enumerate(bytes_per_group):
        base = base_addrs[group_idx]
        for idx in page_idxs:
            descs.append((base + idx * bpp, bpp, device_id))
    return descs


def _resolve_remote_bytes_per_group(
    local_bytes_per_group: Sequence[int],
    remote_bytes_per_group: Sequence[int],
) -> list[int]:
    """Return the remote engine's per-group byte stride for a read transfer.

    Raises unless the remote advertises exactly as many groups as the local
    engine has: connect() already enforces this (a full ``bytes_per_group``
    equality check), so this is defense-in-depth, not the primary guard --
    fewer groups means there is no way to infer the remote's stride for a
    group it never advertised, and more groups would silently assume a
    positional-prefix correspondence that was never validated.
    """
    if list(remote_bytes_per_group) != list(local_bytes_per_group):
        raise ValueError(
            f"Remote advertises bytes_per_group={list(remote_bytes_per_group)} "
            f"but the local engine has {list(local_bytes_per_group)}. "
            "Refusing to guess the remote's stride for a group it never "
            "advertised."
        )
    return list(remote_bytes_per_group)


class TensorAgentMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata for a single tensor/agent in the transfer engine.

    This is used for serialization and communication between engines.
    """

    agent_name: str
    """Name of this agent."""

    metadata: bytes
    """Metadata for this agent."""

    base_addrs: list[int]
    """Base memory address per NIXL group for this shard, indexed by group.
    ``base_addrs[g]`` is the base of group ``g`` (e.g. values, scales, or a
    per-child cache). Parallel to the engine's ``bytes_per_group``; there is
    no special "main" group."""

    device_id: int
    """Device ID for this tensor."""


@dataclass
class TensorAgent:
    """Manages a single tensor and its associated NIXL agent for transfers.

    This class holds both the runtime state (live objects) and can generate
    the serializable metadata for communication between engines.
    """

    agent: nixl.Agent
    """NIXL agent for this tensor."""

    agent_name: str
    """Name of this agent."""

    base_addrs: list[int]
    """Base memory address per NIXL group for this shard, indexed by group.
    Parallel to ``reg_dlists`` and to the engine's ``bytes_per_group``; there
    is no special "main" group."""

    backend: int
    """NIXL backend handle (UCX or libfabric)."""

    device_id: int
    """Device ID for this tensor."""

    agent_metadata: bytes
    """Metadata for this agent."""

    reg_dlists: list[nixl.RegistrationDescriptorList]
    """Registration descriptor list per NIXL group, parallel to ``base_addrs``."""

    @classmethod
    def create_agent(
        cls,
        agent_name: str,
        listen_port: int,
        tensors: Sequence[Buffer],
        memory_type: nixl.MemoryType,
        backend_type: NixlBackendType = "ucx",
    ) -> TensorAgent:
        """Creates and registers a NIXL agent for a shard's per-group buffers.

        Args:
            agent_name: Unique name for this agent.
            listen_port: TCP port for the NIXL listener.
            tensors: This shard's buffers, one per NIXL group, group-major
                (e.g. ``[main_values, main_scales, draft_values,
                draft_scales]``). All must share the same device. Must be
                non-empty.
            memory_type: NIXL memory segment type (DRAM or VRAM).
            backend_type: NIXL transport backend
                (``"ucx"``, ``"libfabric"``, or ``"uccl"``).
        """
        # Pre-load the UCX plugin's GPU runtime dependencies with RTLD_GLOBAL
        # before the NIXL plugin manager dlopens the plugin. Must run in this
        # process (e.g. spawn-ed children do not inherit RTLD_GLOBAL handles).
        preload_nixl_plugin_deps()

        if backend_type == "uccl":
            _default_uccl_socket_ifname_if_unset()
            _default_uccl_p2p_env_if_unset()

        # Create NIXL agent
        agent = nixl.Agent(
            agent_name,
            nixl.AgentConfig(
                # Always use progress thread.
                # - It helps with async notification delivery.
                # - It enables overlapping transfers from multiple agents.
                use_prog_thread=True,
                use_listen_thread=True,
                listen_port=listen_port,
            ),
        )

        # Check backend availability.
        # Upstream NIXL plugin names are uppercase (UCX, LIBFABRIC); the
        # Modular-facing API (MODULAR_NIXL_TRANSFER_BACKEND) keeps lowercase
        # values for backwards compatibility. Map to upstream internally.
        upstream_backend_type = backend_type.upper()
        available = agent.get_available_plugins()
        if upstream_backend_type not in available:
            raise RuntimeError(
                f"NIXL backend {backend_type!r} not available for agent "
                f"{agent_name}. Available plugins: {available}"
            )

        # All groups for one shard live on the same device.
        device = tensors[0].device
        try:
            plugin_params = agent.get_plugin_params(upstream_backend_type)
        except Exception as e:
            reason = _plugin_load_error(upstream_backend_type)
            detail = f": {reason}" if reason else ""
            raise RuntimeError(
                f"NIXL backend {backend_type!r} is present in "
                f"{os.environ.get('NIXL_PLUGIN_DIR')} but could not be "
                f"loaded{detail}. Set NIXL_LOG_LEVEL=INFO for the full NIXL "
                "plugin log."
            ) from e
        backend_params = plugin_params[0]
        if not device.is_host:
            backend_params["gpu_device_id"] = str(device.id)

        # Fill in UCX transport defaults (TLS + the GPU-local IB device) before
        # the backend — and thus the UCX worker that reads them — is created.
        # A no-op for non-CUDA devices (ROCm UCX configures itself).
        if backend_type == "ucx":
            configure_ucx_env(device)

        backend = agent.create_backend(
            type=upstream_backend_type,
            init_params=backend_params,
        )

        # Register one memory region per group, uniformly.
        base_addrs: list[int] = []
        reg_dlists: list[nixl.RegistrationDescriptorList] = []
        for tensor in tensors:
            base_addr = tensor._data_ptr()
            num_bytes = tensor.num_elements * tensor.dtype.size_in_bytes
            reg_dlist = nixl.RegistrationDescriptorList(
                type=memory_type,
                descs=[(base_addr, num_bytes, device.id, "")],
            )
            status = agent.register_memory(reg_dlist, [backend])
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to register memory for {agent_name}: {status}"
                )
            base_addrs.append(base_addr)
            reg_dlists.append(reg_dlist)

        # Get metadata after registration
        agent_metadata = agent.get_local_metadata()

        # Create TensorAgent and add to list
        return TensorAgent(
            agent=agent,
            agent_name=agent_name,
            base_addrs=base_addrs,
            backend=backend,
            device_id=device.id,
            agent_metadata=agent_metadata,
            reg_dlists=reg_dlists,
        )

    def to_metadata(self) -> TensorAgentMetadata:
        """Convert to serializable metadata for communication."""
        return TensorAgentMetadata(
            agent_name=self.agent_name,
            metadata=self.agent_metadata,
            base_addrs=self.base_addrs,
            device_id=self.device_id,
        )


class _TransferStrategy(Enum):
    """How one NIXL group moves across a peer's ``[dp][tp]`` topology.

    A group's strategy is a function of two independent axes: the *topology*
    (did TP change?) and the group's *replication*. DP change is orthogonal
    (routing only) and does not affect the strategy.

    - ``DIRECT``: uniform TP -- shard ``i`` -> shard ``i``.
    - ``BROADCAST``: heterogeneous TP, replicated group -- pick one slice and
      replicate it onto the other side.
    - ``GATHER_SCATTER``: heterogeneous TP, sharded group -- a genuine
      ``tp != tp'`` reshard (a token/head transpose-gather). Net-new; the
      transport refuses it until MXSERV-290 lands.
    """

    DIRECT = "direct"
    BROADCAST = "broadcast"
    GATHER_SCATTER = "gather_scatter"


def resolve_transfer_strategy(
    local_tp: int,
    local_replicate: Sequence[bool],
    remote_tp: int,
    remote_replicate: Sequence[bool],
) -> list[_TransferStrategy]:
    """Plan the per-group :class:`_TransferStrategy` for this peer -- a pure planner.

    The strategy keys on the **TP axis** (a DP change is the scheduler's
    request routing, not a transfer strategy):

    - ``local_tp == remote_tp`` -> ``DIRECT`` (shard-to-shard) for every group;
    - a TP change -> ``BROADCAST`` for a replicated group (pick one slice) or
      ``GATHER_SCATTER`` for a sharded one.
    """
    local_vec = list(local_replicate)
    remote_vec = list(remote_replicate)

    if local_tp == remote_tp:
        # TP unchanged: no reshard. Replication is irrelevant (DP is routing).
        return [_TransferStrategy.DIRECT for _ in local_vec]

    # TP change: replicated -> BROADCAST, sharded -> GATHER_SCATTER. The
    # replicated flag is `tp > 1`-scoped (one shard can't replicate across TP),
    # so a tp==1 side reports every group False. OR the two sides so the tp>1
    # side -- always present on a TP change -- supplies the truth; a genuinely
    # sharded group is False on both.
    return [
        _TransferStrategy.BROADCAST
        if (loc or rem)
        else _TransferStrategy.GATHER_SCATTER
        for loc, rem in zip(local_vec, remote_vec, strict=True)
    ]


def _is_broadcast(strategy: list[_TransferStrategy]) -> bool:
    """Whether this peer's transfer is a ``BROADCAST`` (a replicated TP change)."""
    return any(s is _TransferStrategy.BROADCAST for s in strategy)


def _assert_no_gather_scatter(strategy: list[_TransferStrategy]) -> None:
    """Refuse a plan the transport cannot execute yet.

    A sharded cache on a heterogeneous topology resolves to
    ``GATHER_SCATTER`` -- a ``tp != tp'`` reshard (MXSERV-290) the transport
    does not implement. The planner still produces the per-group plan; the
    transport rejects it here until that strategy lands. ``DIRECT`` and
    ``BROADCAST`` groups pass through.
    """
    if any(s is _TransferStrategy.GATHER_SCATTER for s in strategy):
        raise NotImplementedError(
            "sharded tp!=tp' reshard (GATHER_SCATTER) is not implemented "
            "(MXSERV-290); on a heterogeneous topology every group must be "
            "replicated (BROADCAST)"
        )


# ---------------------------------------------------------------------------
# Topology resolver (pure, NIXL-free)
#
# These functions turn a resolved per-group strategy plus the two engines' ``[dp][tp]``
# shapes into *index plans*: which (local, remote) agent pairs to wire at
# connect time, and which (source, destination) shards to pair for a single
# transfer. They hold no ``self`` and touch no NIXL objects, so they are
# directly CPU-unit-testable. :class:`TransferEngine` maps the returned indices
# onto its live ``TensorAgent`` grid and makes the NIXL calls.
# ---------------------------------------------------------------------------


class ConnectPair(NamedTuple):
    """A single connect/disconnect/cleanup wiring step.

    Pairs one local agent with one remote agent by their physical grid
    indices.
    """

    local_replica: int
    local_shard: int
    remote_replica: int
    remote_shard: int


def connect_pairing(
    strategy: list[_TransferStrategy],
    local_dp: int,
    local_tp: int,
    remote_dp: int,
    remote_tp: int,
) -> list[ConnectPair]:
    """Plan the connect/disconnect/cleanup wiring for a peer.

    Returns physical index quads ``(local_replica, local_shard,
    remote_replica, remote_shard)`` in the order the NIXL metadata
    load/invalidate must iterate. Connect always pairs every DP replica (a
    full cartesian -- the scheduler routes any prefill replica to any decode
    replica); the strategy decides only the shard pattern within each replica
    pair.
    """
    _assert_no_gather_scatter(strategy)
    shard_pairs = _connect_shard_pairs(strategy, local_tp, remote_tp)
    return [
        ConnectPair(local_replica, local_shard, remote_replica, remote_shard)
        for local_replica in range(local_dp)
        for remote_replica in range(remote_dp)
        for local_shard, remote_shard in shard_pairs
    ]


def _connect_shard_pairs(
    strategy: list[_TransferStrategy], local_tp: int, remote_tp: int
) -> list[tuple[int, int]]:
    """Shard-connection pattern within one (local_replica, remote_replica) pair.

    DIRECT wires the shard diagonal (``i <-> i``, ``local_tp == remote_tp``);
    BROADCAST wires the full cross-shard mesh (any src shard may send to any
    dst shard, so every ``(s, s')`` is connectable -- the transfer picks the
    subset it actually moves).
    """
    if _is_broadcast(strategy):
        return [(s, s2) for s in range(local_tp) for s2 in range(remote_tp)]
    return [(i, i) for i in range(local_tp)]


def transfer_shard_pairing(
    flatten_source: bool,
    source_tp: int,
    dest_tp: int,
) -> list[tuple[int, int]]:
    """Plan the (source_shard, dest_shard) pairs for one transfer.

    The source side may be collapsed to a single shard (``flatten_source`` --
    an MLA-replicated source, where any shard's copy suffices and shard 0 saves
    bandwidth). The destination always spans all its shards (each owns distinct
    GPU memory). When the source is a single shard but the destination has many
    (DP-source -> TP-dest), the source is fanned out so every destination shard
    is paired.

    The caller reads ``local_shards_used`` off whichever side is local: the
    source shards for a send, the destination shards for a read.
    """
    if flatten_source:
        # TODO(SERVOPT-1337): always picking shard 0 hotspots one NIC/PCIe
        # path; rotate (round-robin or hashed) to spread load across shards.
        source_shards = [0]
    else:
        source_shards = list(range(source_tp))

    dest_shards = list(range(dest_tp))

    if len(source_shards) == 1 and len(dest_shards) > 1:
        source_shards = source_shards * len(dest_shards)

    return list(zip(source_shards, dest_shards, strict=True))


class TransferEngineMetadata(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Transport-only metadata for a :class:`TransferEngine`.

    Carries just the fields a generic NIXL transport needs to connect to a
    peer: the engine name, memory type, hostname, and per-shard agent
    metadata. KV/topology-specific fields live on
    :class:`KVTransferEngineMetadata`.

    This is safe to send between threads/processes.
    """

    name: str
    """Base name of the transfer engine."""

    memory_type: nixl.MemoryType
    """Memory type of the transfer engine."""

    hostname: str
    """Hostname of the machine that the transfer engine is running on."""

    agents_meta: list[list[TensorAgentMetadata]]
    """Metadata for each replica's agents: [replica][tp_shard]."""


class KVTransferEngineMetadata(TransferEngineMetadata):
    """Metadata associated with a KV cache transfer engine.

    Extends the transport-only :class:`TransferEngineMetadata` with the
    KV-cache/topology fields (page geometry and TP replication).

    This is safe to send between threads/processes.
    """

    total_num_pages: int
    """Total number of pages in each tensor."""

    bytes_per_page: int
    """Bytes per page for each tensor."""

    bytes_per_group: list[int]
    """Bytes per page for each tensor group, one entry per NIXL group. The
    first entry is the main group; subsequent entries correspond to extra
    groups (e.g., draft KV in speculative decoding). ``bytes_per_page``
    equals ``sum(bytes_per_group)``."""

    replicated_per_group: list[bool] = []
    """Per-group TP replication, parallel to ``bytes_per_group``. ``True``
    entries are replicated identically across TP shards (MLA-style); ``False``
    entries are sharded. This is the sole replication datum on the wire; the
    peer's :func:`resolve_transfer_strategy` plans each group from it."""


class TransferReqData(
    msgspec.Struct, tag=True, kw_only=True, omit_defaults=True
):
    """Metadata associated with a transfer request.

    This is safe to send between threads/processes.
    """

    dst_name: str
    """Base name of destination engine."""

    src_name: str
    """Base name of source engine."""

    transfer_name: str
    """Transfer name."""

    transfer_ids: list[int]
    """Transfer IDs (one per TP shard in the replica)."""

    src_idxs: list[int]
    """Length of source indices can differ from len(transfer_ids)."""

    dst_idxs: list[int]
    """Length of destination indices can differ from len(transfer_ids)."""

    src_replica_idx: int
    """Index of the source replica this transfer is from."""

    dst_replica_idx: int
    """Index of the destination replica this transfer is to."""

    is_read: bool = False
    """True if this is a READ (pull) transfer initiated by the destination."""

    tp_shard_count: int = 0
    """Number of TP shards participating. 0 = all shards (backwards compat)."""

    local_shards_used: list[int] = []
    """Physical TP shard indices on the initiator that own this transfer's
    handles. Empty means "all shards in the recorded replica" (pre-flatten
    behavior). Required to release/status-check transfers when a flattened
    group has picked a subset of shards."""


class TransferEngine:
    """NIXL transfer engine that owns the NIXL plumbing.

    - Agent lifecycle (create, connect, disconnect, cleanup)
    - Memory registration / deregistration
    - Descriptor list construction for (buffer, offset, size) ranges
    - Send / read transfer initiation and completion tracking
      (``initiate_send_transfer``, ``initiate_read_transfer``,
      ``is_complete``, ``cleanup_transfer``, ``sync_and_release``)

    This base still carries KV-cache topology today -- the ``[dp][tp]``
    ``tensor_agents`` grid, page geometry, and ``.metadata`` returns
    :class:`KVTransferEngineMetadata`; :class:`KVTransferEngine` is a thin
    construction subclass on top. Making the transport KV-agnostic (so it is
    testable without KV scaffolding) is tracked in MXSERV-313.

    ``TransferEngine`` is not thread-safe and is intended to be driven by
    MAX's single-threaded scheduler.
    """

    name: str
    """Name of this engine / NIXL agent group."""

    tensor_agents: list[list[TensorAgent]]
    """2D list of TensorAgent objects: [replica][tp_shard]."""

    total_num_pages: int
    """Total number of pages in each tensor."""

    bytes_per_page: int
    """Total bytes per page across all groups. For single-group engines this
    equals the main group's bytes per page; for multi-group engines it is
    ``sum(bytes_per_group)``."""

    bytes_per_group: list[int]
    """Bytes per page for each group. ``bytes_per_group[0]`` is the main
    group; subsequent entries are extra groups (e.g., draft KV in
    speculative decoding)."""

    replicated_per_group: list[bool]
    """Per-group TP replication, parallel to ``bytes_per_group``. Routing plans
    each group from this vector; there is no engine-wide replication flag."""

    memory_type: nixl.MemoryType
    """Type of memory being managed."""

    remote_connections: dict[str, KVTransferEngineMetadata]
    """Map of remote engine names to their metadata."""

    remote_agent_to_engine: dict[str, str]
    """Map of remote agent names to their engine names."""

    completed_recv_transfers: dict[str, dict[str, int]]
    """Map of agent names to completed recv transfers."""

    inflight_send_transfers: dict[str, TransferReqData]
    """Map of transfer names to send transfer request data."""

    dp: int
    """Number of DP replicas."""

    tp: int
    """Number of TP shards per replica."""

    def __init__(
        self,
        name: str,
        tensor_agents: list[list[TensorAgent]],
        *,
        total_num_pages: int,
        bytes_per_page: int,
        bytes_per_group: list[int],
        memory_type: nixl.MemoryType,
        dp: int,
        tp: int,
        backend_type: NixlBackendType,
        replicated_per_group: list[bool] | None = None,
    ) -> None:
        self.name = name
        self.tensor_agents = tensor_agents
        self.total_num_pages = total_num_pages
        self.bytes_per_page = bytes_per_page
        self.bytes_per_group = bytes_per_group
        self.memory_type = memory_type
        self.dp = dp
        self.tp = tp
        self._backend_type = backend_type

        # Replication is carried per group (parallel to bytes_per_group) and
        # routed per group; there is no engine-wide replication flag.
        if replicated_per_group is None:
            replicated_per_group = [False] * len(bytes_per_group)
        if len(replicated_per_group) != len(bytes_per_group):
            raise ValueError(
                f"replicated_per_group has {len(replicated_per_group)} "
                f"entries but bytes_per_group has {len(bytes_per_group)}"
            )
        self.replicated_per_group = replicated_per_group

        # Remote connections
        self.remote_connections: dict[str, KVTransferEngineMetadata] = {}

        # Per-peer group strategies populated at connect().
        self._transfer_strategies: dict[str, list[_TransferStrategy]] = {}

        # Map of agents to completed transfers
        self.completed_recv_transfers: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Map of remote agent names to their engine names
        self.remote_agent_to_engine: dict[str, str] = {}

        # All send transfers - maps transfer_name to list of (tensor_idx, transfer_id) tuples
        self.inflight_send_transfers: dict[str, TransferReqData] = {}

        # All read transfers - maps transfer_name to TransferReqData
        self.inflight_read_transfers: dict[str, TransferReqData] = {}

    @property
    def metadata(self) -> KVTransferEngineMetadata:
        """Get metadata for all replicas.

        Returns:
            Metadata for the entire engine (all replicas).
        """
        agents_meta = [
            [ta.to_metadata() for ta in replica_agents]
            for replica_agents in self.tensor_agents
        ]

        return KVTransferEngineMetadata(
            name=self.name,
            total_num_pages=self.total_num_pages,
            bytes_per_page=self.bytes_per_page,
            memory_type=self.memory_type,
            agents_meta=agents_meta,
            hostname=socket.gethostname(),
            bytes_per_group=self.bytes_per_group,
            replicated_per_group=self.replicated_per_group,
        )

    def _resolve_local_agents_for_transfer(
        self, replica_idx: int, transfer_req: TransferReqData
    ) -> list[TensorAgent]:
        """Return the local ``TensorAgent``s that own a transfer's handles.

        Consults ``transfer_req.local_shards_used`` when populated; falls
        back to every shard in the replica when empty (pre-flatten behavior).
        """
        if not transfer_req.local_shards_used:
            return list(self.tensor_agents[replica_idx])
        return [
            self.tensor_agents[replica_idx][s]
            for s in transfer_req.local_shards_used
        ]

    def _compute_transfer_strategy(
        self, remote: KVTransferEngineMetadata
    ) -> list[_TransferStrategy]:
        """Plan the per-group transfer strategy for this peer.

        Thin wrapper around :func:`resolve_transfer_strategy`.
        """
        rtp = len(remote.agents_meta[0]) if remote.agents_meta else 0
        return resolve_transfer_strategy(
            local_tp=self.tp,
            local_replicate=self.replicated_per_group,
            remote_tp=rtp,
            remote_replicate=remote.replicated_per_group,
        )

    def _strategy_for_teardown(
        self, name: str, remote: KVTransferEngineMetadata, *, pop: bool
    ) -> list[_TransferStrategy]:
        """Look up the per-peer strategy recorded at ``connect()``, for teardown.

        Defensive: connect() populates ``_transfer_strategies`` and
        ``remote_connections`` together, so a miss is unreachable today. If
        they ever desync, recompute (rather than assuming DIRECT) so a
        broadcast peer's teardown still mirrors connect()'s pairing.
        """
        strategy = (
            self._transfer_strategies.pop(name, None)
            if pop
            else self._transfer_strategies.get(name)
        )
        if strategy is None:
            logger.warning(
                "Transfer strategy missing for remote %r during teardown "
                "(connect() should populate it together with "
                "remote_connections); recomputing from current metadata "
                "instead.",
                name,
            )
            strategy = self._compute_transfer_strategy(remote)
        return strategy

    def _iter_peer_agents(
        self,
        remote: KVTransferEngineMetadata,
        strategy: list[_TransferStrategy],
    ) -> Iterator[tuple[TensorAgent, TensorAgentMetadata]]:
        """Yield ``(local agent, remote agent-meta)`` pairs for a peer.

        Maps the resolver's physical index quads onto the live agent grids, in
        the order connect / disconnect / cleanup must iterate. Sharing this one
        iterator is what makes teardown mirror ``connect()``.
        """
        rdp = len(remote.agents_meta)
        rtp = len(remote.agents_meta[0]) if remote.agents_meta else 0
        for lr, ls, rr, rs in connect_pairing(
            strategy, self.dp, self.tp, rdp, rtp
        ):
            yield self.tensor_agents[lr][ls], remote.agents_meta[rr][rs]

    def connect(self, remote: KVTransferEngineMetadata) -> None:
        """Connect to a remote engine (all replicas).

        Args:
            remote: Metadata for the remote engine (all replicas).
        """
        if remote.name in self.remote_connections:
            raise ValueError(f"Agent {remote.name} already connected")

        strategy = self._compute_transfer_strategy(remote)
        # Fail fast on a plan the transport cannot execute yet (sharded
        # reshard); the per-group plan is valid, the reshard strategy is not.
        _assert_no_gather_scatter(strategy)

        if self.bytes_per_page != remote.bytes_per_page:
            raise ValueError(
                f"Bytes per page mismatch: {self.bytes_per_page} != {remote.bytes_per_page}"
            )

        if self.bytes_per_group != remote.bytes_per_group:
            raise ValueError(
                f"Per-group bytes-per-page mismatch: "
                f"local={self.bytes_per_group} remote={remote.bytes_per_group}"
            )

        # Check if the relevant transport env vars are set. You can get away
        # with eliding these for intra-node DI. However, for inter-node DI,
        # loading metadata appears to hang (UCX) or performance degrades
        # severely (libfabric without GPU-direct RDMA) if they are not set.
        hostname = socket.gethostname()
        is_internode = hostname != remote.hostname
        if is_internode:
            backend_type = _get_nixl_backend_type()
            if backend_type == "ucx" and not (
                "UCX_NET_DEVICES" in os.environ and "UCX_TLS" in os.environ
            ):
                raise ValueError(
                    "Inter-node UCX transfer is not configured "
                    f"({hostname} <-> {remote.hostname}): MAX could not "
                    "auto-derive the GPU-local InfiniBand device. Set "
                    "UCX_NET_DEVICES (e.g. mlx5_0:1) and UCX_TLS (e.g. "
                    "cuda_ipc,cuda_copy,rc,sm,self) explicitly."
                )
            if backend_type == "libfabric" and not os.environ.get(
                "FI_EFA_USE_DEVICE_RDMA"
            ):
                logger.warning(
                    "Inter-node libfabric connection (%s <-> %s) without "
                    "FI_EFA_USE_DEVICE_RDMA set. EFA GPU-direct RDMA will "
                    "be disabled, which may severely impact KV transfer "
                    "throughput. Set FI_EFA_USE_DEVICE_RDMA=1.",
                    hostname,
                    remote.hostname,
                )

        # Load remote metadata for every wired (local, remote) agent pair.
        for local_ta, remote_agent_meta in self._iter_peer_agents(
            remote, strategy
        ):
            loaded_bytes = local_ta.agent.load_remote_metadata(
                remote_agent_meta.metadata
            )
            try:
                loaded_remote_name = loaded_bytes.decode()
            except UnicodeDecodeError as e:
                raise ValueError(
                    f"Metadata loading failed. "
                    f"Expected string, found {loaded_bytes!r}"
                ) from e
            if loaded_remote_name != remote_agent_meta.agent_name:
                raise ValueError(
                    f"Metadata loading failed. "
                    f"Expected {remote_agent_meta.agent_name}, got {loaded_remote_name}"
                )

        self.remote_connections[remote.name] = remote
        self._transfer_strategies[remote.name] = strategy

        # Update the remote agent to engine mapping
        for replica_agents_meta in remote.agents_meta:
            for agent_meta in replica_agents_meta:
                self.remote_agent_to_engine[agent_meta.agent_name] = remote.name

    def disconnect(self, name: str) -> None:
        """Tear down a single remote connection.

        Releases inflight transfer handles referencing this remote,
        invalidates NIXL metadata, and removes bookkeeping entries.
        After disconnect, ``connect()`` will accept the same name again.

        Args:
            name: The name of the remote engine to disconnect.

        Raises:
            ValueError: If the named remote is not currently connected.
        """
        remote = self.remote_connections.pop(name, None)
        if remote is None:
            raise ValueError(
                f"Remote connection '{name}' not found; cannot disconnect"
            )
        strategy = self._strategy_for_teardown(name, remote, pop=True)

        # Release inflight send transfers targeting this remote.
        stale_sends = [
            tname
            for tname, req in self.inflight_send_transfers.items()
            if req.dst_name == name
        ]
        for tname in stale_sends:
            req = self.inflight_send_transfers.pop(tname)
            src_agents = self._resolve_local_agents_for_transfer(
                req.src_replica_idx, req
            )
            for tp_idx, tid in enumerate(req.transfer_ids):
                try:
                    src_agents[tp_idx].agent.release_transfer_request(tid)
                except Exception:
                    logger.warning(
                        "Failed to release send transfer %s tp=%d"
                        " during disconnect of '%s'",
                        tname,
                        tp_idx,
                        name,
                        exc_info=True,
                    )

        # Release inflight read transfers sourced from this remote.
        stale_reads = [
            tname
            for tname, req in self.inflight_read_transfers.items()
            if req.src_name == name
        ]
        for tname in stale_reads:
            req = self.inflight_read_transfers.pop(tname)
            dst_agents = self._resolve_local_agents_for_transfer(
                req.dst_replica_idx, req
            )
            for tp_idx, tid in enumerate(req.transfer_ids):
                try:
                    dst_agents[tp_idx].agent.release_transfer_request(tid)
                except Exception:
                    logger.warning(
                        "Failed to release read transfer %s tp=%d"
                        " during disconnect of '%s'",
                        tname,
                        tp_idx,
                        name,
                        exc_info=True,
                    )

        # Teardown iterates the same pairs as connect() (shared iterator).
        for local_ta, remote_agent_meta in self._iter_peer_agents(
            remote, strategy
        ):
            try:
                status = local_ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    logger.warning(
                        "invalidate_remote_metadata returned %s for"
                        " agent '%s' during disconnect of '%s'",
                        status,
                        remote_agent_meta.agent_name,
                        name,
                    )
            except Exception:
                logger.warning(
                    "Failed to invalidate metadata for agent '%s'"
                    " during disconnect of '%s'",
                    remote_agent_meta.agent_name,
                    name,
                    exc_info=True,
                )

        # Clean up agent-to-engine mapping entries for this remote.
        stale_agent_names = [
            agent_name
            for agent_name, engine_name in self.remote_agent_to_engine.items()
            if engine_name == name
        ]
        for agent_name in stale_agent_names:
            del self.remote_agent_to_engine[agent_name]

        # Drop completed recv transfer tracking for this remote.
        self.completed_recv_transfers.pop(name, None)

        logger.info("Disconnected remote '%s'", name)

    def initiate_send_transfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
        src_replica_idx: int,
        dst_replica_idx: int,
    ) -> TransferReqData:
        """Initiate a transfer from current engine to remote engine.

        The same page indices are broadcast to all TP shards within the source and destination replicas.

        Args:
            remote_metadata: Metadata for the remote engine.
            src_idxs: List of indices of the source pages in the current engine.
            dst_idxs: List of indices of the destination pages in the remote engine.
            src_replica_idx: Index of the source replica to transfer from.
            dst_replica_idx: Index of the destination replica to transfer to.
        """
        if not (0 <= src_replica_idx < self.dp):
            raise ValueError(
                f"src_replica_idx {src_replica_idx} must be between 0 and {self.dp - 1}"
            )

        if not (0 <= dst_replica_idx < len(remote_metadata.agents_meta)):
            raise ValueError(
                f"dst_replica_idx {dst_replica_idx} must be between 0 and {len(remote_metadata.agents_meta) - 1}"
            )

        if remote_metadata.name not in self.remote_connections:
            raise ValueError(
                f"Remote connection {remote_metadata.name} not found"
            )

        remote = self.remote_connections[remote_metadata.name]
        strategy = self._transfer_strategies[remote_metadata.name]

        if len(src_idxs) != len(dst_idxs):
            raise ValueError(
                f"Source and destination indices must have the same length. Got {len(src_idxs)} and {len(dst_idxs)}"
            )

        # Each dst idx must be unique so that we don't write to the same page
        if len(set(dst_idxs)) != len(dst_idxs):
            raise ValueError(
                f"Destination indices must be unique. Found duplicate index: {dst_idxs}"
            )

        for src_idx in src_idxs:
            if not (0 <= src_idx < self.total_num_pages):
                raise ValueError(
                    f"Source index {src_idx} must be between 0 and {self.total_num_pages - 1}"
                )

        for dst_idx in dst_idxs:
            if not (0 <= dst_idx < remote.total_num_pages):
                raise ValueError(
                    f"Destination index {dst_idx} must be between 0 and {remote.total_num_pages - 1}"
                )

        transfer_name = str(uuid4())
        transfer_ids = []

        # Plan (source_shard, dest_shard) pairs. A BROADCAST source collapses to
        # shard 0 (any replicated shard's copy suffices, saves bandwidth); the
        # destination always spans all its TP shards since each owns distinct GPU
        # memory. The source is the local side here (a send).
        _assert_no_gather_scatter(strategy)
        local_replica_agents = self.tensor_agents[src_replica_idx]
        remote_replica_agents_meta = remote.agents_meta[dst_replica_idx]
        shard_pairs = transfer_shard_pairing(
            flatten_source=_is_broadcast(strategy),
            source_tp=len(local_replica_agents),
            dest_tp=len(remote_replica_agents_meta),
        )
        local_shards_used = [src_shard for src_shard, _ in shard_pairs]

        for src_shard, dst_shard in shard_pairs:
            ta = local_replica_agents[src_shard]
            remote_agent_meta = remote_replica_agents_meta[dst_shard]

            # Build descriptors for each group.
            # Each group uses its own base address and bytes_per_page; all
            # groups share the same logical page indices.
            src_base_addrs = ta.base_addrs
            dst_base_addrs = remote_agent_meta.base_addrs

            descs_src = _build_group_descriptors(
                src_base_addrs, self.bytes_per_group, src_idxs, ta.device_id
            )
            descs_dst = _build_group_descriptors(
                dst_base_addrs,
                self.bytes_per_group,
                dst_idxs,
                remote_agent_meta.device_id,
            )

            transfer_dlist_src = nixl.TransferDescriptorList(
                type=self.memory_type, descs=descs_src
            )
            transfer_dlist_dst = nixl.TransferDescriptorList(
                type=remote.memory_type, descs=descs_dst
            )

            # Use the appropriate agent for this tensor
            remote_agent_name = remote_agent_meta.agent_name

            transfer_id = ta.agent.create_transfer_request(
                operation=nixl.TransferOpType.WRITE,
                local_descs=transfer_dlist_src,
                remote_descs=transfer_dlist_dst,
                remote_agent=remote_agent_name,
                notif_msg=transfer_name,
            )
            status = ta.agent.post_transfer_request(transfer_id)

            if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
                raise ValueError(
                    f"Transfer request failed with status {status} for TP shard {src_shard}"
                )

            transfer_ids.append(transfer_id)

        transfer_req = TransferReqData(
            dst_name=remote_metadata.name,
            src_name=self.name,
            transfer_name=transfer_name,
            transfer_ids=transfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
            src_replica_idx=src_replica_idx,
            dst_replica_idx=dst_replica_idx,
            tp_shard_count=len(transfer_ids),
            local_shards_used=local_shards_used,
        )
        self.inflight_send_transfers[transfer_name] = transfer_req
        return transfer_req

    def initiate_read_transfer(
        self,
        remote_metadata: KVTransferEngineMetadata,
        src_idxs: list[int],
        dst_idxs: list[int],
        src_replica_idx: int,
        dst_replica_idx: int,
    ) -> TransferReqData:
        """Initiate a READ transfer from remote engine to current engine.

        The current engine pulls data from the remote. Used by DKVConnector
        to read KV blocks from BlockStore DRAM into GPU VRAM.

        Args:
            remote_metadata: Metadata for the remote engine (source).
            src_idxs: Page indices in the remote engine (source).
            dst_idxs: Page indices in the current engine (destination).
            src_replica_idx: Replica index in the remote engine.
            dst_replica_idx: Replica index in the current engine.
        """
        if not (0 <= dst_replica_idx < self.dp):
            raise ValueError(
                f"dst_replica_idx {dst_replica_idx} must be between 0 and {self.dp - 1}"
            )

        if not (0 <= src_replica_idx < len(remote_metadata.agents_meta)):
            raise ValueError(
                f"src_replica_idx {src_replica_idx} must be between 0 and {len(remote_metadata.agents_meta) - 1}"
            )

        if remote_metadata.name not in self.remote_connections:
            raise ValueError(
                f"Remote connection {remote_metadata.name} not found"
            )

        remote = self.remote_connections[remote_metadata.name]
        strategy = self._transfer_strategies[remote_metadata.name]

        if len(src_idxs) != len(dst_idxs):
            raise ValueError(
                f"Source and destination indices must have the same length. Got {len(src_idxs)} and {len(dst_idxs)}"
            )

        for dst_idx in dst_idxs:
            if not (0 <= dst_idx < self.total_num_pages):
                raise ValueError(
                    f"Destination index {dst_idx} must be between 0 and {self.total_num_pages - 1}"
                )

        for src_idx in src_idxs:
            if not (0 <= src_idx < remote.total_num_pages):
                raise ValueError(
                    f"Source index {src_idx} must be between 0 and {remote.total_num_pages - 1}"
                )

        transfer_name = str(uuid4())
        transfer_ids = []

        # Plan (source_shard, dest_shard) pairs. Here the remote is the source (a
        # BROADCAST remote collapses to shard 0) and the local engine is the
        # destination, always spanning all its TP shards.
        _assert_no_gather_scatter(strategy)
        local_replica_agents = self.tensor_agents[dst_replica_idx]
        remote_replica_agents_meta = remote.agents_meta[src_replica_idx]
        shard_pairs = transfer_shard_pairing(
            flatten_source=_is_broadcast(strategy),
            source_tp=len(remote_replica_agents_meta),
            dest_tp=len(local_replica_agents),
        )
        # Local is the destination for a read.
        local_shards_used = [dst_shard for _, dst_shard in shard_pairs]

        # Determine per-group bytes_per_page for the remote (source) engine.
        # This is a loop invariant -- computed once, not per shard pair.
        remote_bpg = _resolve_remote_bytes_per_group(
            self.bytes_per_group, remote.bytes_per_group
        )

        for remote_shard, dst_shard in shard_pairs:
            ta = local_replica_agents[dst_shard]
            remote_agent_meta = remote_replica_agents_meta[remote_shard]

            # Build descriptors for each group. Local uses this engine's
            # bytes_per_group; remote uses the peer's advertised strides.
            descs_local = _build_group_descriptors(
                ta.base_addrs, self.bytes_per_group, dst_idxs, ta.device_id
            )
            descs_remote = _build_group_descriptors(
                remote_agent_meta.base_addrs,
                remote_bpg,
                src_idxs,
                remote_agent_meta.device_id,
            )

            local_dlist = nixl.TransferDescriptorList(
                type=self.memory_type, descs=descs_local
            )
            remote_dlist = nixl.TransferDescriptorList(
                type=remote.memory_type, descs=descs_remote
            )

            transfer_id = ta.agent.create_transfer_request(
                operation=nixl.TransferOpType.READ,
                local_descs=local_dlist,
                remote_descs=remote_dlist,
                remote_agent=remote_agent_meta.agent_name,
                notif_msg=transfer_name,
            )
            status = ta.agent.post_transfer_request(transfer_id)

            if status not in [nixl.Status.SUCCESS, nixl.Status.IN_PROG]:
                raise ValueError(
                    f"Read transfer request failed with status {status} for TP shard {dst_shard}"
                )

            transfer_ids.append(transfer_id)

        transfer_req = TransferReqData(
            dst_name=self.name,
            src_name=remote_metadata.name,
            transfer_name=transfer_name,
            transfer_ids=transfer_ids,
            src_idxs=src_idxs,
            dst_idxs=dst_idxs,
            src_replica_idx=src_replica_idx,
            dst_replica_idx=dst_replica_idx,
            is_read=True,
            tp_shard_count=len(transfer_ids),
            local_shards_used=local_shards_used,
        )
        self.inflight_read_transfers[transfer_name] = transfer_req
        return transfer_req

    def _is_sender_of(self, transfer_req: TransferReqData) -> bool:
        """Check if the current engine is the sender of a transfer."""
        return transfer_req.src_name == self.name

    def _owns_transfer_request(self, transfer_req: TransferReqData) -> bool:
        """Check if the current engine owns the transfer request handles."""
        if transfer_req.is_read:
            return transfer_req.dst_name == self.name
        return self._is_sender_of(transfer_req)

    def _notification_remote_name(self, transfer_req: TransferReqData) -> str:
        """Return the remote engine name associated with completion notifications."""
        if transfer_req.is_read:
            return transfer_req.dst_name
        return transfer_req.src_name

    def _is_send_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a send transfer is complete.

        Args:
            transfer_req: The transfer request data containing transfer metadata.

        Returns:
            True if the send transfer is complete, False otherwise.
        """
        assert self._is_sender_of(transfer_req)

        is_complete = True
        src_replica_idx = transfer_req.src_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            src_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.get_transfer_status(transfer_id)

            if status == nixl.Status.SUCCESS:
                continue
            elif status == nixl.Status.IN_PROG:
                is_complete = False
                break
            else:
                raise ValueError(
                    f"Transfer request failed with status {status} in source replica {src_replica_idx}"
                )

        return is_complete

    def _is_recv_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a recv transfer is complete."""
        assert not self._owns_transfer_request(transfer_req)

        # Check what recv completion notifications have been received
        # We only check agents in the replica local to the current engine.
        local_replica_idx = (
            transfer_req.src_replica_idx
            if transfer_req.is_read
            else transfer_req.dst_replica_idx
        )
        tp_agents = self.tensor_agents[local_replica_idx]
        for ta in tp_agents:
            notifs = ta.agent.get_notifs()
            for remote_agent_name, notifications in notifs.items():
                engine_name = self.remote_agent_to_engine[remote_agent_name]
                for notif in notifications:
                    notif_decoded = notif.decode()
                    self.completed_recv_transfers[engine_name][
                        notif_decoded
                    ] += 1

        # A recv is complete when we get expected number of notifications
        transfer_name = transfer_req.transfer_name
        expected = (
            transfer_req.tp_shard_count
            if transfer_req.tp_shard_count > 0
            else self.tp
        )
        remote_name = self._notification_remote_name(transfer_req)
        return (
            self.completed_recv_transfers[remote_name][transfer_name]
            == expected
        )

    def _is_read_complete(self, transfer_req: TransferReqData) -> bool:
        """Check if a read transfer is complete.

        For READ ops the local agent initiates the transfer, so we poll
        get_transfer_status on our own agents (same pattern as send).
        """
        assert transfer_req.is_read
        assert self._owns_transfer_request(transfer_req)

        dst_replica_idx = transfer_req.dst_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            dst_replica_idx, transfer_req
        )

        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.get_transfer_status(transfer_id)

            if status == nixl.Status.SUCCESS:
                continue
            elif status == nixl.Status.IN_PROG:
                return False
            else:
                raise ValueError(
                    f"Read transfer failed with status {status} in replica {dst_replica_idx}"
                )

        return True

    def is_complete(self, transfer_req: TransferReqData) -> bool:
        """Checks if a given send, recv, or read transfer is completed.

        .. caution::
           This method only reports progress; it does not *drive* it. For a
           transfer to complete, **both** engines must keep polling: the sender
           polls its own :meth:`is_complete` while the receiver polls its own.
           A single-threaded loop that polls only one engine — for example
           ``while not sender.is_complete(req): pass`` without ever polling the
           receiver — deadlocks, because the receiver never advances the
           transfer. Always poll both engines concurrently (or use
           :meth:`sync_and_release`, which polls with a bounded timeout).

        The example below runs a real host-DRAM transfer between two engines
        on separate threads so both sides make progress, then polls each
        engine's :meth:`is_complete` until the send finishes. NIXL requires the
        UCX transport plugin, which ships prebuilt for Linux x86-64 only, so the
        transfer runs behind an availability guard and safely no-ops elsewhere.

        .. code-block:: python

            import numpy as np
            from max._core import nixl
            from max.driver.buffer import Buffer
            from max.pipelines.kv_cache import KVTransferEngine

            def nixl_ucx_available() -> bool:
                try:
                    probe = nixl.Agent(
                        "probe", nixl.AgentConfig(use_prog_thread=False)
                    )
                    return "UCX" in probe.get_available_plugins()
                except Exception:
                    return False

            if nixl_ucx_available():
                total_num_pages = 3
                num_elts = total_num_pages * 3
                blocks_1 = Buffer.from_numpy(np.arange(num_elts, dtype=np.int16))
                blocks_2 = Buffer.from_numpy(np.zeros(num_elts, dtype=np.int16))

                sender = KVTransferEngine(
                    "sender", [[blocks_1]], total_num_pages=total_num_pages
                )
                receiver = KVTransferEngine(
                    "receiver", [[blocks_2]], total_num_pages=total_num_pages
                )
                sender.connect(receiver.metadata)
                receiver.connect(sender.metadata)

                # Poll BOTH engines concurrently so each side drives its half of
                # the transfer. The receiver runs sync_and_release on a thread
                # (bounded poll); the main thread polls the sender's is_complete.
                from queue import Queue
                from threading import Thread

                queue: Queue = Queue()

                def _send() -> None:
                    req = sender.initiate_send_transfer(
                        receiver.metadata, [0], [0],
                        src_replica_idx=0, dst_replica_idx=0,
                    )
                    queue.put(req)
                    sender.sync_and_release(req)

                def _recv() -> None:
                    receiver.sync_and_release(queue.get())

                send_thread = Thread(target=_send)
                recv_thread = Thread(target=_recv)
                send_thread.start()
                recv_thread.start()
                send_thread.join()
                recv_thread.join()

                receiver.cleanup()
                sender.cleanup()

        Args:
            transfer_req: The transfer request.

        Returns:
            bool: True if all transfers have completed; false otherwise.
        """
        if transfer_req.is_read:
            if self._owns_transfer_request(transfer_req):
                return self._is_read_complete(transfer_req)
            return self._is_recv_complete(transfer_req)
        elif self._is_sender_of(transfer_req):
            return self._is_send_complete(transfer_req)
        else:
            return self._is_recv_complete(transfer_req)

    def _cleanup_recv_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a transfer."""
        assert not self._owns_transfer_request(transfer_req)
        assert transfer_req.transfer_name not in self.inflight_send_transfers

        remote_name = self._notification_remote_name(transfer_req)
        del self.completed_recv_transfers[remote_name][
            transfer_req.transfer_name
        ]

    def _cleanup_send_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a send transfer."""
        assert self._is_sender_of(transfer_req)
        transfer_name = transfer_req.transfer_name
        assert transfer_name in self.inflight_send_transfers

        del self.inflight_send_transfers[transfer_name]

        src_replica_idx = transfer_req.src_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            src_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.release_transfer_request(transfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release transfer request: {status}"
                )

    def _cleanup_read_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a read transfer by releasing transfer requests."""
        assert transfer_req.is_read
        transfer_name = transfer_req.transfer_name
        assert transfer_name in self.inflight_read_transfers

        del self.inflight_read_transfers[transfer_name]

        dst_replica_idx = transfer_req.dst_replica_idx
        tp_agents = self._resolve_local_agents_for_transfer(
            dst_replica_idx, transfer_req
        )
        for tp_idx, transfer_id in enumerate(transfer_req.transfer_ids):
            agent = tp_agents[tp_idx].agent
            status = agent.release_transfer_request(transfer_id)
            if status != nixl.Status.SUCCESS:
                raise ValueError(
                    f"Failed to release read transfer request: {status}"
                )

    def cleanup_transfer(self, transfer_req: TransferReqData) -> None:
        """Cleanup a transfer. This should be called after a transfer is complete.

        Args:
            transfer_req: The transfer request to cleanup.
        """
        if not self.is_complete(transfer_req):
            raise ValueError(
                f"Transfer {transfer_req.transfer_name} is not complete"
            )

        if transfer_req.is_read:
            if self._owns_transfer_request(transfer_req):
                self._cleanup_read_transfer(transfer_req)
            else:
                self._cleanup_recv_transfer(transfer_req)
        elif self._is_sender_of(transfer_req):
            self._cleanup_send_transfer(transfer_req)
        else:
            self._cleanup_recv_transfer(transfer_req)

    def sync_and_release(
        self,
        transfer_req: TransferReqData,
        timeout_s: float = 30.0,
    ) -> None:
        """Waits for a transfer to complete and releases it.

        Args:
            transfer_req: The transfer request to wait on.
            timeout_s: Maximum seconds to wait before raising TimeoutError.

        Raises:
            TimeoutError: If the transfer does not complete within timeout_s.
        """
        deadline = time.monotonic() + timeout_s
        while not self.is_complete(transfer_req):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"NIXL transfer did not complete within {timeout_s}s"
                )
            time.sleep(0.001)
        self.cleanup_transfer(transfer_req)

    def cleanup(self) -> None:
        """Release all resources associated with the transfer engine.

        Should be called before the transfer engine is garbage collected.
        Moving this logic into the __del__ destructor does causes a UCX error for
        unknown reasons.
        """
        # Release all send transfers
        for send_transfer_req in list(self.inflight_send_transfers.values()):
            self._cleanup_send_transfer(send_transfer_req)

        # Release all read transfers
        for read_transfer_req in list(self.inflight_read_transfers.values()):
            self._cleanup_read_transfer(read_transfer_req)

        # Invalidate metadata of other agents. Iterate via the recorded
        # per-group strategy so teardown mirrors connect()'s exact pairing.
        for remote_name in self.remote_connections:
            remote = self.remote_connections[remote_name]
            strategy = self._strategy_for_teardown(
                remote_name, remote, pop=False
            )
            for local_ta, remote_agent_meta in self._iter_peer_agents(
                remote, strategy
            ):
                status = local_ta.agent.invalidate_remote_metadata(
                    remote_agent_meta.agent_name
                )
                if status != nixl.Status.SUCCESS:
                    raise ValueError(f"Failed to invalidate metadata: {status}")

        # Deregister NIXL memory for all tensors (all replicas, all groups)
        for replica_agents in self.tensor_agents:
            for ta in replica_agents:
                for reg_dlist in ta.reg_dlists:
                    status = ta.agent.deregister_memory(reg_dlist, [ta.backend])
                    if status != nixl.Status.SUCCESS:
                        raise ValueError(
                            f"Failed to deregister memory: {status}"
                        )


class KVTransferEngine(TransferEngine):
    """KVCache Transfer Engine with support for Data Parallelism (DP) and Tensor Parallelism (TP).

    The engine accepts per-replica producer-authored NIXL groups
    (:class:`~max.nn.kv_cache.cache_params.KVCacheMemory`).  The outer list is
    indexed by DP replica; the inner list is that replica's group list — one
    group per logical ``(child, kind)`` tensor, from ``to_memory()``.

    ``KVTransferEngine`` is a thin layer on top of :class:`TransferEngine` that adds:

    - Validation of tensor shapes and device types
    - NIXL group construction from the authored groups
    - Per-group replication (``replicated_per_group``) authored from the
      groups' ``replicated`` field (no caller plumbing needed)
    - ``from_paged_kv_cache()`` convenience constructor

    All NIXL transport operations are delegated to :class:`TransferEngine`.

    The TransferEngine communicates with other TransferEngines in other threads
    or processes. However, individual TransferEngines themselves are not
    thread-safe. It is intended to be used by MAX's single-threaded scheduler.
    """

    def __init__(
        self,
        name: str,
        memory: Sequence[Sequence[KVCacheMemory]],
    ) -> None:
        """Initialize the transfer engine from producer-authored NIXL groups.

        Args:
            name: Unique name for this engine.
            memory: Per-replica group lists as ``[replica][group]``.  Each entry
                is a
                :class:`~max.nn.kv_cache.cache_params.KVCacheMemory` — one
                logical ``(child, kind)`` tensor carrying every TP-shard view,
                as returned by ``KVCacheBuffer.to_memory()``.  All
                replicas must have the same group count and consistent
                replication kind.  The page count (including the null block) is
                read from the groups themselves, so every group must agree on
                ``total_num_pages``.
        """
        if not memory:
            raise ValueError("tensors must contain at least one replica")
        if not memory[0]:
            raise ValueError("Each replica must contain at least one tensor")

        # total_num_pages is a property of the buffers (``buffer.shape[0]``,
        # including the null block), so read it off the authored groups rather
        # than accepting a redundant argument. Every group must agree on it.
        total_num_pages = memory[0][0].total_num_pages
        num_groups_r0 = len(memory[0])

        for r, replica_groups in enumerate(memory):
            if not replica_groups:
                raise ValueError(
                    "Each replica must contain at least one tensor"
                )
            if len(replica_groups) != num_groups_r0:
                raise ValueError(
                    f"Replica {r} produced {len(replica_groups)} NIXL "
                    f"groups but replica 0 had {num_groups_r0}. "
                    "Replicas must have a consistent buffer structure."
                )
            for group in replica_groups:
                if group.total_num_pages != total_num_pages:
                    raise ValueError(
                        f"Replica {r} has a group with total_num_pages="
                        f"{group.total_num_pages}, but all groups must match "
                        f"replica 0 group 0's {total_num_pages}"
                    )

        dp = len(memory)

        # A logical cache is consistently replicated across DP replicas, so a
        # given group index must agree on ``replicated`` across replicas. This
        # is a real structural invariant (unlike "all groups agree"), so keep a
        # narrow check for it. Every replica is already confirmed above to
        # have exactly num_groups_r0 groups, so indexing memory[0][g] here
        # never goes out of bounds.
        for r, replica_groups in enumerate(memory):
            for g, group in enumerate(replica_groups):
                if group.replicated != memory[0][g].replicated:
                    raise ValueError(
                        f"Group {g} of replica {r} has replicated="
                        f"{group.replicated} but replica 0 has "
                        f"replicated={memory[0][g].replicated}. A logical "
                        "cache must be replicated consistently across DP "
                        "replicas."
                    )

        # Build all_groups[group_idx][replica_idx] = [shard0, shard1, ...]
        # directly from the authored groups (no shape re-inference). Group
        # count consistency was already validated above.
        all_groups: list[list[list[Buffer]]] = [
            [] for _ in range(num_groups_r0)
        ]
        for replica_groups in memory:
            for g, group in enumerate(replica_groups):
                all_groups[g].append(group.buffers)

        # From here on every NIXL group is treated uniformly — there is no
        # special "main" group. ``all_groups[g][r]`` is the shard list for
        # group ``g`` of replica ``r``; groups may differ in shape/bytes but
        # every replica of a given group must agree.
        if not all_groups:
            raise ValueError(
                "memory must contain at least one NIXL group "
                "(e.g. values/scales or a child cache)"
            )

        num_groups = len(all_groups)
        # TP degree is the shard count of group 0, replica 0; every group and
        # replica must match it.
        tp = len(all_groups[0][0])
        if tp == 0:
            raise ValueError("Each replica must contain at least one tensor")

        # Replication rides on the authored groups; replica 0 is authoritative
        # (per-replica agreement was validated above). `g.replicated` comes from
        # the producer (replicates_kv_across_tp) and already accounts for tp, so
        # trust it -- don't re-guard with `and tp > 1` (that would re-mask a
        # producer that ever reports logical replication).
        replicated_per_group = [g.replicated for g in memory[0]]

        backend_type = _get_nixl_backend_type()

        # Validate every group across replicas and compute per-group bytes/page.
        bytes_per_group: list[int] = []  # [group_idx] → bytes_per_page
        memory_types: list[nixl.MemoryType] = []
        for group_idx, group_replicas in enumerate(all_groups):
            group_bpp_list: list[int] = []
            for replica_idx, replica_shards in enumerate(group_replicas):
                if len(replica_shards) != tp:
                    raise ValueError(
                        f"Group {group_idx} replica {replica_idx} has "
                        f"{len(replica_shards)} TP shards, but expected {tp}. "
                        "All groups and replicas must share the same TP degree."
                    )
                _validate_device_type([t.device for t in replica_shards])
                gbpp = _validate_tensor_shape(replica_shards)
                group_bpp_list.append(gbpp)

                is_cpu = replica_shards[0].device.is_host
                memory_types.append(
                    nixl.MemoryType.DRAM if is_cpu else nixl.MemoryType.VRAM
                )
            if len(set(group_bpp_list)) != 1:
                raise ValueError(
                    f"All replicas must have the same bytes_per_page. "
                    f"Group {group_idx} found: {group_bpp_list}"
                )
            bytes_per_group.append(group_bpp_list[0])

        if len(set(memory_types)) != 1:
            raise ValueError(
                f"All groups/replicas must have the same memory type. "
                f"Found: {set(memory_types)}"
            )

        bytes_per_page = sum(bytes_per_group)
        memory_type = memory_types[0]

        # Create one agent per (replica, shard), registering every group's
        # buffer for that shard uniformly (group-major).
        tensor_agents: list[list[TensorAgent]] = []
        for replica_idx in range(dp):
            replica_agents = []
            for tp_idx in range(tp):
                shard_tensors = [
                    all_groups[g][replica_idx][tp_idx]
                    for g in range(num_groups)
                ]
                tensor_agent = TensorAgent.create_agent(
                    agent_name=f"{name}_{replica_idx}_{tp_idx}",
                    listen_port=available_port(),
                    tensors=shard_tensors,
                    memory_type=memory_type,
                    backend_type=backend_type,
                )
                replica_agents.append(tensor_agent)
            tensor_agents.append(replica_agents)

        super().__init__(
            name=name,
            tensor_agents=tensor_agents,
            total_num_pages=total_num_pages,
            bytes_per_page=bytes_per_page,
            bytes_per_group=bytes_per_group,
            memory_type=memory_type,
            dp=dp,
            tp=tp,
            backend_type=backend_type,
            replicated_per_group=replicated_per_group,
        )

        logger.info(
            "NIXL memory registration complete for %s (%s backend): "
            "%d agent(s) (dp=%d, tp=%d), %d bytes per agent (%d group(s)).",
            self.name,
            backend_type,
            self.dp * self.tp,
            self.dp,
            self.tp,
            self.bytes_per_page * total_num_pages,
            len(self.bytes_per_group),
        )

    @classmethod
    def from_paged_kv_cache(
        cls, name: str, kv_cache: PagedKVCacheManagerInterface
    ) -> KVTransferEngine:
        """Construct an engine wired to a ``PagedKVCacheManager``.

        Calls ``KVCacheBuffer.to_memory()`` on each replica's device
        buffer to obtain the producer-authored NIXL groups, then passes them to
        the constructor, which carries each group's ``replicated`` field as
        ``replicated_per_group``.

        For models with multiple KV caches (e.g., speculative decoding with a
        separate target and draft KV), each child cache contributes its own
        group(s) so that heterogeneous buffer shapes (e.g., 61-layer MLA target
        vs. 1-layer Eagle draft) are registered as independent NIXL groups.

        Quantized caches (values + scales): ``to_memory()`` authors a
        separate group for values and for scales (one group per child x kind).
        For non-quantized caches this collapses to one group per child, which is
        byte-identical to the previous ``all_buffers`` path.
        """
        dp = kv_cache.params.data_parallel_degree
        device_buffers = [kv_cache.get_device_buffer(r) for r in range(dp)]

        return cls(
            name=name,
            memory=[buf.to_memory() for buf in device_buffers],
        )
