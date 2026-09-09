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

"""Interface for accessing GPU statistics of all supported GPU types."""

from __future__ import annotations

from contextlib import ExitStack
from types import TracebackType
from typing import Any

from ._nvml import NVMLContext
from ._rsmi import RSMIContext
from .types import GPUStats


class GPUDiagContext:
    """Context manager providing unified access to GPU diagnostic information across NVIDIA and AMD hardware.

    This class automatically detects and initializes supported GPU vendor libraries
    (NVML for NVIDIA, ROCm SMI for AMD) and provides a unified interface for
    collecting diagnostic statistics from all available GPUs in the system.

    .. code-block:: python

        from max.profiler.gpu import GPUDiagContext

        with GPUDiagContext() as ctx:
            stats = ctx.get_stats()
            for gpu_id, gpu_stats in stats.items():
                print(f"GPU {gpu_id}:  {gpu_stats.memory.used_bytes} bytes used")
    """

    def __init__(self) -> None:
        self._nvml: NVMLContext | None = None
        self._rsmi: RSMIContext | None = None
        self._stack: ExitStack | None = None
        # Errors are currently tucked away in here, never to be exposed, but
        # maybe we can expose this somehow in the future.  Still useful from
        # within PDB when debugging, sometimes.
        self._errors: list[tuple[str, Exception]] = []

    def __enter__(self) -> GPUDiagContext:
        if self._stack is not None:
            raise AssertionError("Can't enter an already-entered context")
        with ExitStack() as stack:
            try:
                self._nvml = stack.enter_context(NVMLContext())
            except Exception as e:
                self._errors.append(("nv", e))
            try:
                self._rsmi = stack.enter_context(RSMIContext())
            except Exception as e:
                self._errors.append(("amd", e))
            self._stack = stack.pop_all()
        return self

    def __exit__(
        self,
        exc_type: type[Any] | None,
        exc_value: Any,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self._stack is not None
        self._nvml = None
        self._rsmi = None
        self._stack.close()
        self._stack = None

    def get_stats(self) -> dict[str, GPUStats]:
        """Retrieve current GPU statistics for all detected GPUs in the system.

        Returns:
            A dictionary mapping GPU identifiers to their current statistics.
            NVIDIA GPUs are prefixed with ``nv`` (e.g., ``nv0``, ``nv1``) and AMD
            GPUs are prefixed with ``amd`` (e.g., ``amd0``, ``amd1``).
        """
        stats: dict[str, GPUStats] = {}
        if self._nvml is not None:
            nvml_stats = self._nvml.get_stats()
            for gpu_id, gpu_stats in nvml_stats.items():
                stats[f"nv{gpu_id}"] = gpu_stats
        if self._rsmi is not None:
            rsmi_stats = self._rsmi.get_stats()
            for gpu_id, gpu_stats in rsmi_stats.items():
                stats[f"amd{gpu_id}"] = gpu_stats
        return stats
