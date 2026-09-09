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
"""Shared NIXL plugin discovery and preload for test conftests.

Per-vendor plugin flavor resolution has a single implementation
(Support/NixlPluginDir.h) that runs on ``import max._core``: the importer
locates the staged plugins (Bazel runfiles or installed layout) and sets
NIXL_PLUGIN_DIR before the upstream nixlPluginManager singleton reads it at
first use. This module only adds test policy on top: fail loudly when a GPU
host resolves no flavor (broken runfiles staging would otherwise surface as
an obscure transport-unavailable error mid-test), and pre-load the GPU
runtime libraries the UCX plugin references (libcuda/libnvidia-ml for the
CUDA flavor, libhsa-runtime64 for the ROCm flavors, libibverbs/libmlx5 for
the verbs flavors) with RTLD_GLOBAL so their symbols are visible when the
plugin manager calls ``dlopen(libplugin_UCX.so, RTLD_NOW)``; those symbols
are not pre-loaded in the Bazel test sandbox.
"""

from __future__ import annotations

import ctypes
import os

import max._core  # noqa: F401  (sets NIXL_PLUGIN_DIR at import time)


def preload_gpu_libs() -> None:
    """Pre-loads GPU runtimes with RTLD_GLOBAL so libplugin_UCX.so can dlopen.

    The upstream nixlPluginManager uses RTLD_NOW when loading plugins, which
    requires ALL symbols to be resolvable at dlopen time. Pre-loading with
    RTLD_GLOBAL makes the symbols available. Libraries absent on the host are
    skipped silently — the flavor needing them is not selected there.
    """
    for lib_name in (
        "libcuda.so.1",
        "libcuda.so",
        "libnvidia-ml.so.1",
        "libnvidia-ml.so",
        "libhsa-runtime64.so.1",
        "libhsa-runtime64.so",
        # RDMA verbs stack: needed by the *-verbs UCX flavors.
        "libibverbs.so.1",
        "libmlx5.so.1",
    ):
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass  # not available on this machine; ignore silently


def _force_plain_flavor() -> None:
    """Repoints NIXL_PLUGIN_DIR from a ``*-verbs`` flavor to its plain sibling.

    These tests build two NIXL agents in one process. On an InfiniBand host the
    verbs flavor opens two mlx5 device contexts on the same HCA, which host
    rdma-core cannot tear down cleanly (a reserved-QPN mutex assertion in
    mlx5_free_context aborts the process). The plain flavor omits the verbs
    transports, so it never opens an mlx5 context; the rc/InfiniBand path is
    covered cross-process by the perf-smoke cross-node lane instead. No-op off
    an IB host, where the resolver already picked the plain flavor.
    """
    plugin_dir = os.environ.get("NIXL_PLUGIN_DIR")
    if not plugin_dir:
        return
    head, flavor = os.path.split(plugin_dir.rstrip("/"))
    plain = {"cuda-verbs": "cuda", "rocm-verbs": "rocm"}.get(flavor)
    if plain is not None:
        os.environ["NIXL_PLUGIN_DIR"] = os.path.join(head, plain)


def configure() -> None:
    """Configures NIXL plugin discovery and preloads GPU runtime libraries.

    Call at conftest import time, before any test or fixture creates a
    nixlAgent, so the environment is in place before the plugin manager
    singleton is constructed.
    """
    preload_gpu_libs()
    # max._core resolved NIXL_PLUGIN_DIR at import; drop the verbs flavor for
    # these in-process multi-agent tests (see _force_plain_flavor) before the
    # plugin-manager singleton reads it.
    _force_plain_flavor()
    if os.environ.get("NIXL_PLUGIN_DIR"):
        return
    gpu_nodes = [
        node for node in ("/dev/nvidiactl", "/dev/kfd") if os.path.exists(node)
    ]
    if gpu_nodes:
        raise RuntimeError(
            f"GPU detected ({', '.join(gpu_nodes)}) but max._core resolved "
            "no NIXL plugin flavor: the per-vendor plugin .so is missing "
            "from the runfiles (check the target's @nixl_upstream data "
            "deps) or its load-time library dependencies do not resolve."
        )
