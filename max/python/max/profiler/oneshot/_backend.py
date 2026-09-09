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

"""Profiler backend detection and nsys-child env marker."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from ._cuda import can_load_cudart

# Set on the spawned child so it knows it is running under nsys.
#
# Note: this env var is inherited by *every* descendant process the child
# spawns (including pipeline workers). Today only ``_is_under_nsys()``
# consults it, so that is harmless. Any future caller of
# ``_is_under_nsys()`` outside the entry-point code must decide whether to
# trust the env var, strip it before forking, or use additional process
# state (e.g. ``os.getppid()``) to disambiguate.
_NSYS_CHILD_ENV_VAR = "_MAX_PROFILE_UNDER_NSYS"

# When set on the nsys child by the parent, the child writes its cProfile
# stats to this path instead of rendering them inline. The parent reads the
# file after nsys finishes writing the .nsys-rep and renders the cProfile
# table there, so the --profile output sits cleanly *after* the normal
# generate output.
_CPROFILE_DUMP_PATH_ENV_VAR = "_MAX_PROFILE_CPROFILE_DUMP_PATH"


@dataclass(frozen=True)
class ProfileBackend:
    """Resolved profiling backend for the current host."""

    nsys_available: bool
    nvidia_gpu_present: bool
    # libcudart must be dlopenable so we can call cudaProfilerStart/Stop. The
    # nsys child is launched with ``--capture-range=cudaProfilerApi``, so
    # without those calls the trace is empty.
    cudart_loadable: bool

    @property
    def can_capture_gpu(self) -> bool:
        """Whether GPU kernel data can be captured on this host."""
        return (
            self.nsys_available
            and self.nvidia_gpu_present
            and self.cudart_loadable
        )


def _detect_nvidia_gpu() -> bool:
    """Return True if at least one NVIDIA GPU is likely present.

    Uses ``nvidia-smi`` on ``PATH`` as the smoke check. Spinning up a
    ``max.profiler.gpu.GPUDiagContext`` would be more authoritative but it
    pays the full NVML/RSMI init cost for what is only a "should we even
    re-exec under nsys?" decision; the nsys re-exec itself fails closed if
    the GPU isn't actually NVIDIA, so the cheap check is sufficient.
    """
    return shutil.which("nvidia-smi") is not None


def detect_backend() -> ProfileBackend:
    """Detect profiler capabilities once, at startup."""
    return ProfileBackend(
        nsys_available=shutil.which("nsys") is not None,
        nvidia_gpu_present=_detect_nvidia_gpu(),
        cudart_loadable=can_load_cudart(),
    )


def _is_under_nsys() -> bool:
    return os.environ.get(_NSYS_CHILD_ENV_VAR) == "1"
