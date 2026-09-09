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

"""Performance profiling and tracing utilities for MAX.

This module provides tools for profiling and tracing MAX operations to analyze
performance characteristics. Profiling captures timing information for code
execution, which helps identify bottlenecks and optimize your models.

To enable in-runtime NVTX markers, set ``MODULAR_ENABLE_PROFILING`` to ``on``
or ``detailed`` before running your code. Without it, profiling calls are
no-ops with minimal overhead.

The profiler exposes two layers:

1. **In-source spans**: :class:`Tracer` (context manager / manual stack) and
   :func:`@traced <traced>` (decorator) emit NVTX ranges around blocks or
   functions. These show up in any Nsight Systems capture of the process.
2. **One-shot CLI capture**: :func:`maybe_reexec_under_nsys` re-launches the
   current process under ``nsys profile`` and renders a top-N kernel summary
   on exit; :func:`profiled_region` is the corresponding context manager
   that brackets the timed region with ``cudaProfilerStart``/``Stop`` and
   prints a ``cProfile`` Python/CPU summary. These power the
   ``--profile`` flag on ``max generate`` / ``max benchmark``.
"""

from max.profiler import cpu, gpu, oneshot
from max.profiler.oneshot import (
    OneShotCapture,
    ProfileBackend,
    default_profile_output,
    detect_backend,
    maybe_reexec_under_nsys,
    profiled_region,
)

try:
    from max._core.profiler import is_profiling_enabled, set_gpu_profiling_state
    from max.profiler.tracing import Tracer, traced
except ImportError:

    def _not_available(*args, **kwargs) -> None:
        raise ImportError(
            "max._core is not available in this environment. "
            "Install the full MAX package to use profiling and tracing."
        )

    Tracer = _not_available  # type: ignore[assignment, misc]
    traced = _not_available  # type: ignore[assignment]
    is_profiling_enabled = _not_available  # type: ignore[assignment]
    set_gpu_profiling_state = _not_available

__all__ = [
    "OneShotCapture",
    "ProfileBackend",
    "Tracer",
    "cpu",
    "default_profile_output",
    "detect_backend",
    "gpu",
    "is_profiling_enabled",
    "maybe_reexec_under_nsys",
    "oneshot",
    "profiled_region",
    "set_gpu_profiling_state",
    "traced",
]
