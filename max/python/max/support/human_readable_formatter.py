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

"""Private helper function for formatting various quantities into human readable strings."""


def to_human_readable_bytes(bytes: float) -> str:
    """Convert bytes to human readable memory size."""
    KiB = 1024
    MiB = KiB * 1024
    GiB = MiB * 1024
    TiB = GiB * 1024
    bytes = int(bytes)
    if bytes >= TiB:
        return f"{bytes / TiB:.2f} TiB"
    if bytes >= GiB:
        return f"{bytes / GiB:.2f} GiB"
    if bytes >= MiB:
        return f"{bytes / MiB:.2f} MiB"
    return f"{bytes / KiB:.2f} KiB"


def to_human_readable_latency(s: float) -> str:
    """Converts seconds to human readable latency."""
    if s >= 1:
        return f"{s:.2f}s"
    ms = s * 1e3
    if ms >= 1:
        return f"{ms:.2f}ms"
    us = ms * 1e3
    if us >= 1:
        return f"{us:.2f}us"
    ns = us * 1e3
    return f"{ns:.2f}ns"
