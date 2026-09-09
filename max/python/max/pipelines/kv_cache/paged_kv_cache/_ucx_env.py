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

"""Defaults for the UCX (NIXL) inter-node transport env.

UCX needs to be told which transports it may use (``UCX_TLS``) and which
network device to bind (``UCX_NET_DEVICES``). Left to its own devices on a
multi-NIC InfiniBand box, UCX auto-selects across every transport and NIC,
which hangs during metadata exchange or spreads traffic onto a non-local NIC.
Historically the transfer engine required the operator to set both by hand;
this module derives sensible defaults so a correctly-provisioned host works out
of the box, while any value the operator sets explicitly still wins.

The two knobs differ: ``UCX_TLS`` is essentially static, but the right
``UCX_NET_DEVICES`` is the IB device PCIe-local to the GPU this process drives,
which we resolve from the host topology.
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path
from typing import Protocol

logger = logging.getLogger("max.pipelines")


class _UcxDevice(Protocol):
    """The subset of ``max.driver.Device`` this module reads."""

    @property
    def api(self) -> str: ...
    @property
    def id(self) -> int: ...


_PCI_DEVICES_ROOT = Path("/sys/bus/pci/devices")
_IB_CLASS_ROOT = Path("/sys/class/infiniband")
_VERBS_DEV_ROOT = Path("/dev/infiniband")
_VERBS_CLASS_ROOT = Path("/sys/class/infiniband_verbs")
_GDRDRV_NODE = Path("/dev/gdrdrv")


def _verbs_accessible_devices(
    *,
    verbs_dev_root: Path = _VERBS_DEV_ROOT,
    verbs_class_root: Path = _VERBS_CLASS_ROOT,
) -> set[str]:
    """Returns the IB device names this namespace can actually open.

    ``/sys/class/infiniband`` lists every device on the HOST, but a container
    exposes only a subset as verbs char nodes in ``/dev/infiniband``. Map each
    exposed ``uverbsN`` node back to its device via
    ``/sys/class/infiniband_verbs/uverbsN/ibdev`` — the same set
    ``ibv_devinfo -l`` prints. An empty result means the mapping is unavailable
    (older kernels, odd sysfs), in which case callers should not filter.
    """
    names: set[str] = set()
    try:
        nodes = list(verbs_dev_root.iterdir())
    except OSError:
        return names
    for node in nodes:
        if not node.name.startswith("uverbs"):
            continue
        try:
            names.add(
                (verbs_class_root / node.name / "ibdev").read_text().strip()
            )
        except OSError:
            continue
    return names


def _verbs_usable() -> bool:
    """Returns whether IB verbs are actually usable on this host.

    ``/sys/class/infiniband`` can list devices whose verbs char nodes
    (``/dev/infiniband/uverbsN``) are absent — for instance a test sandbox that
    bind-mounts host sysfs but not the device nodes. UCX cannot open such a
    device, so pinning it (and dropping the tcp fallback) would leave the worker
    with no transport. Gate NIC selection on a real uverbs node.
    """
    try:
        return any(
            p.name.startswith("uverbs") for p in _VERBS_DEV_ROOT.iterdir()
        )
    except OSError:
        return False


def default_ucx_tls(*, gdr_copy: bool | None = None) -> str:
    """Returns the default ``UCX_TLS`` transport list for GPU transfers.

    Covers every good path and omits the ones that hang on a multi-NIC box
    (notably ``tcp``/``ud``): ``cuda_ipc`` for same-node NVLink peers, ``rc``
    for inter-node InfiniBand RDMA, ``sm``/``self`` for the local worker's
    loopback, and ``cuda_copy`` to stage through host memory. ``gdr_copy``
    (direct GPU-NIC copy) is added when ``/dev/gdrdrv`` is present.

    Args:
        gdr_copy: Whether to include the ``gdr_copy`` transport. Defaults to
            probing ``/dev/gdrdrv``.
    """
    if gdr_copy is None:
        gdr_copy = _GDRDRV_NODE.exists()
    transports = ["cuda_ipc", "cuda_copy", "rc", "sm", "self"]
    if gdr_copy:
        transports.append("gdr_copy")
    return ",".join(transports)


def _gpu_pci_bus_id(device: _UcxDevice) -> str | None:
    """Returns the PCIe bus id (``0000:65:00.0``) of a CUDA device, or None.

    Uses the CUDA driver's ``cuDeviceGetPCIBusId`` on the device's own ordinal,
    so it agrees with the ordinal MAX uses (avoiding the NVML-vs-CUDA device
    ordering hazard). Best-effort: returns None for non-CUDA devices or if the
    driver is unavailable.
    """
    if device.api != "cuda":
        return None
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    libcuda.cuInit(0)  # idempotent; MAX has already initialized the driver
    handle = ctypes.c_int()
    if libcuda.cuDeviceGet(ctypes.byref(handle), ctypes.c_int(device.id)) != 0:
        return None
    buf = ctypes.create_string_buffer(32)
    if libcuda.cuDeviceGetPCIBusId(buf, ctypes.c_int(len(buf)), handle) != 0:
        return None
    return buf.value.decode().lower()


def _first_active_ib_port(ports_dir: Path) -> str | None:
    """Returns the first InfiniBand-link-layer port that is ACTIVE, else None.

    Filters out non-IB RDMA devices (e.g. AWS EFA reports a non-InfiniBand
    link layer) and down ports.
    """
    if not ports_dir.is_dir():
        return None
    for port in sorted(ports_dir.iterdir()):
        try:
            link_layer = (port / "link_layer").read_text().strip()
            state = (port / "state").read_text().strip()
        except OSError:
            continue
        if link_layer == "InfiniBand" and "ACTIVE" in state:
            return port.name
    return None


def local_ib_device(
    gpu_bus_id: str,
    *,
    pci_devices_root: Path = _PCI_DEVICES_ROOT,
    ib_class_root: Path = _IB_CLASS_ROOT,
    verbs_dev_root: Path = _VERBS_DEV_ROOT,
    verbs_class_root: Path = _VERBS_CLASS_ROOT,
) -> str | None:
    """Returns the ``<device>:<port>`` of the IB HCA PCIe-closest to a GPU.

    Ranks each InfiniBand device by the length of the ``/sys/devices`` PCIe
    path it shares with the GPU — more shared bridges means a closer (lower
    latency, GPUDirect-capable) device. Only devices with an active
    InfiniBand-link-layer port that this namespace can actually open are
    considered (``/sys/class/infiniband`` lists every host NIC, but a container
    exposes only a subset — pinning a non-exposed one leaves UCX with no usable
    device). Returns None when no such device exists.

    Args:
        gpu_bus_id: PCIe bus id of the GPU, e.g. ``0000:65:00.0``.
        pci_devices_root: Override of ``/sys/bus/pci/devices`` (for testing).
        ib_class_root: Override of ``/sys/class/infiniband`` (for testing).
        verbs_dev_root: Override of ``/dev/infiniband`` (for testing).
        verbs_class_root: Override of ``/sys/class/infiniband_verbs``.
    """
    gpu_path = os.path.realpath(pci_devices_root / gpu_bus_id)
    if not ib_class_root.is_dir():
        return None

    # Host sysfs lists every NIC; restrict to the ones actually openable here.
    # An empty set means the mapping is unavailable, so don't filter.
    accessible = _verbs_accessible_devices(
        verbs_dev_root=verbs_dev_root, verbs_class_root=verbs_class_root
    )

    best_shared = -1
    best_device: str | None = None
    # Sorted so ties resolve deterministically to the lowest-numbered device.
    for entry in sorted(ib_class_root.iterdir()):
        if accessible and entry.name not in accessible:
            continue
        port = _first_active_ib_port(entry / "ports")
        if port is None:
            continue
        ib_path = os.path.realpath(entry / "device")
        shared = _shared_path_len(gpu_path, ib_path)
        if shared > best_shared:
            best_shared = shared
            best_device = f"{entry.name}:{port}"
    return best_device


def _shared_path_len(a: str, b: str) -> int:
    """Number of leading ``/``-separated components two paths share."""
    shared = 0
    for x, y in zip(a.split("/"), b.split("/"), strict=False):
        if x != y:
            break
        shared += 1
    return shared


def configure_ucx_env(device: _UcxDevice) -> None:
    """Sets UCX transport defaults for a GPU NIXL/UCX agent, if unset.

    Applies :func:`default_ucx_tls` and, for CUDA devices, the GPU-local IB
    device from :func:`local_ib_device`. Both use ``setdefault`` semantics, so
    a value the operator exported explicitly is never overridden. Best-effort:
    any failure to resolve the NIC leaves ``UCX_NET_DEVICES`` unset, which the
    transfer engine's inter-node check then reports.
    """
    # These defaults name CUDA and InfiniBand transports. AMD (ROCm) UCX
    # exposes rocm_* transports instead, so forcing this UCX_TLS there would
    # leave it with no usable transport; leave non-CUDA devices (and hosts) to
    # configure themselves.
    if device.api != "cuda":
        return

    # An operator-pinned device wins; still give it a matching TLS default.
    if "UCX_NET_DEVICES" in os.environ:
        os.environ.setdefault("UCX_TLS", default_ucx_tls())
        return

    # Only touch the env when IB verbs are actually usable here. Otherwise —
    # a non-IB host or a sandbox with a sysfs-only device — leave UCX to its
    # defaults so its tcp/cuda_ipc transports keep intra-node transfers working.
    if not _verbs_usable():
        return
    try:
        bus_id = _gpu_pci_bus_id(device)
        net_device = local_ib_device(bus_id) if bus_id is not None else None
    except Exception:
        # Never let topology probing break agent creation; the operator can
        # always set UCX_NET_DEVICES explicitly.
        logger.debug("UCX_NET_DEVICES auto-derivation failed", exc_info=True)
        return
    if net_device is None:
        return
    os.environ["UCX_NET_DEVICES"] = net_device
    os.environ.setdefault("UCX_TLS", default_ucx_tls())
    logger.info(
        "Auto-selected UCX_NET_DEVICES=%s (InfiniBand device local to GPU %s "
        "at %s)",
        net_device,
        device.id,
        bus_id,
    )
