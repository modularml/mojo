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

"""Rudimentary one-shot profiler for MAX CLI entrypoints.

Powers the ``--profile`` flag on ``max pipelines generate`` and
``max pipelines benchmark``. The implementation is split across siblings:

- :mod:`._backend` — host capability detection
- :mod:`._cuda` — ``cudaProfilerStart``/``Stop`` bracketing
- :mod:`._cprofile` — Python/CPU summary rendering
- :mod:`._nsys` — Nsight Systems re-exec and kernel-summary rendering
- :mod:`._runner` — top-level orchestrators wired by the CLI handlers
"""

from ._backend import ProfileBackend as ProfileBackend
from ._backend import detect_backend as detect_backend
from ._nsys import render_nsys_kernel_summary as render_nsys_kernel_summary
from ._runner import OneShotCapture as OneShotCapture
from ._runner import default_profile_output as default_profile_output
from ._runner import inject_trace_flags as inject_trace_flags
from ._runner import maybe_reexec_under_nsys as maybe_reexec_under_nsys
from ._runner import profiled_region as profiled_region
