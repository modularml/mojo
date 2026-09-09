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

"""``RTLD_GLOBAL`` preload of the NIXL transport plugins' runtime dependencies.

Shared by every code path that creates a NIXL agent — the KV transfer engine
and the dKV connector — because the requirement belongs to the plugin, not to
the consumer that happens to load it.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os

logger = logging.getLogger("max.pipelines")

# GPU runtime libraries that the upstream UCX plugin (libplugin_UCX.so, in
# its per-vendor flavors) references but does not itself dlopen. The upstream
# plugin manager loads plugins with ``dlopen(..., RTLD_NOW | RTLD_LOCAL)``;
# ``RTLD_NOW`` requires every undefined symbol (CUDA driver, NVML, HSA,
# optionally RDMA verbs) to be resolvable at load time, and ``RTLD_LOCAL``
# means the plugin cannot see symbols unless they were already loaded
# ``RTLD_GLOBAL`` into the process.
_NIXL_PLUGIN_DEP_LIBS: tuple[str, ...] = (
    # RDMA verbs (only needed by the *-verbs UCX flavors); harmless if absent,
    # except on an InfiniBand host -- see _warn_if_verbs_unusable.
    "libibverbs.so.1",
    "libmlx5.so.1",
    # CUDA driver + NVML: required by the CUDA-flavor UCX plugin.
    "libcuda.so.1",
    "libnvidia-ml.so.1",
    # HSA runtime: required by the ROCm-flavor UCX plugins.
    "libhsa-runtime64.so.1",
)

# The rdma-core libraries the *-verbs plugin flavors need at load time. Missing
# any of them is fatal to the verbs flavor but not to the plain one, so the
# runtime silently selects the plain flavor -- which has no RDMA transports.
_VERBS_DEP_LIBS = frozenset({"libibverbs.so.1", "libmlx5.so.1"})


def _has_infiniband_port() -> bool:
    """Returns whether this host exposes an RDMA device with an InfiniBand port.

    Mirrors ``hasInfinibandPort`` in ``Support/lib/NixlPluginDir.cpp``, which
    gates selection of the verbs plugin flavor. Distinguishing a real IB fabric
    from other rdma-core users (notably AWS EFA, whose device reports a
    non-InfiniBand link layer) is what makes a missing verbs dependency worth
    warning about rather than expected.
    """
    root = "/sys/class/infiniband"
    try:
        devices = os.listdir(root)
    except OSError:
        return False
    for device in devices:
        ports = os.path.join(root, device, "ports")
        try:
            port_names = os.listdir(ports)
        except OSError:
            continue
        for port in port_names:
            try:
                with open(os.path.join(ports, port, "link_layer")) as f:
                    if f.read().strip() == "InfiniBand":
                        return True
            except OSError:
                continue
    return False


def _warn_if_verbs_unusable(missing: set[str]) -> None:
    """Warns when an InfiniBand host cannot load the verbs plugin's dependencies.

    On such a host the verbs flavor is the intended one, and the fallback to the
    plain flavor means RDMA is silently unavailable -- transfers run over TCP with
    nothing in the logs to say so. Anywhere else a missing rdma-core is normal and
    stays at debug.
    """
    unusable = missing & _VERBS_DEP_LIBS
    if not unusable or not _has_infiniband_port():
        return
    logger.warning(
        "This host has an InfiniBand port but %s could not be loaded, so the "
        "NIXL verbs plugin flavor cannot be selected and KV transfers will fall "
        "back to TCP. The serving image must ship an rdma-core whose provider "
        "matches its libibverbs.",
        ", ".join(sorted(unusable)),
    )


@functools.cache
def preload_nixl_plugin_deps() -> None:
    """Pre-loads the UCX plugin's runtime dependencies with ``RTLD_GLOBAL``.

    The upstream NIXL plugin manager ``dlopen``s ``libplugin_UCX.so`` with
    ``RTLD_NOW | RTLD_LOCAL``. The vendored plugin flavor (selected per host
    GPU vendor via ``NIXL_PLUGIN_DIR``) references CUDA/NVML, HSA, or RDMA
    verbs symbols; ``RTLD_LOCAL`` prevents the plugin from resolving them
    against the process unless they were previously loaded with
    ``RTLD_GLOBAL``. Without this, ``get_plugin_params("UCX")`` returns
    ``NIXL_ERR_NOT_FOUND`` because the plugin fails to load, or — on the dKV
    path, which reaches the plugin through the Rust client rather than through
    a plugin-params probe — the worker segfaults inside the plugin at agent
    creation.

    The Modular NIXL fork performed this preload inside its plugin-manager
    constructor; upstream does not, so we restore it here. It runs in every
    process that creates a NIXL agent — including ``spawn``-ed
    multiprocessing children, which do NOT inherit the parent's ``RTLD_GLOBAL``
    handles. Libraries that are absent on the host (e.g. CUDA on an AMD/CPU
    box) are skipped; the plugin simply cannot load there, which is reported by
    the existing availability checks rather than masked.
    """
    missing: set[str] = set()
    for lib_name in _NIXL_PLUGIN_DEP_LIBS:
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            missing.add(lib_name)
            # Not present on this host; the corresponding UCX flavor cannot be
            # used here. This is not an error to swallow — it is a genuine
            # "this transport is unavailable on this machine" signal that
            # surfaces downstream via get_available_plugins / get_plugin_params.
            logger.debug(
                "NIXL plugin dependency %s not found; skipping preload",
                lib_name,
            )
    _warn_if_verbs_unusable(missing)
