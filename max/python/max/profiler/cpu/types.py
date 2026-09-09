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

"""Public types returned by the CPU diagnostics API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CPUMetrics:
    """CPU metrics collected during benchmarking."""

    user: float
    """The total user CPU time in seconds."""

    user_percent: float
    """The user CPU utilization as a percentage."""

    system: float
    """The total system CPU time in seconds."""

    system_percent: float
    """The system CPU utilization as a percentage."""

    elapsed: float
    """The elapsed wall-clock time in seconds."""
