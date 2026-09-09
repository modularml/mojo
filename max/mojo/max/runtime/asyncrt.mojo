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
"""This module implements the low level concurrency library."""

from std.ffi import external_call
from std.runtime import parallelism_level as std_parallelism_level

from max.gpu.host import DeviceContext


def parallelism_level(ctx: Optional[DeviceContext]) -> Int:
    """Gets the parallelism level from a DeviceContext.

    For CPU contexts this returns the number of worker threads in the
    runtime associated with that context. Falls back to the global
    parallelism level if the context is None or the query fails.

    Args:
        ctx: The device context to query.

    Returns:
        The parallelism level of the context.
    """
    from max.gpu.host import DeviceAttribute

    if ctx:
        try:
            return ctx.value().get_attribute(DeviceAttribute.PARALLELISM_LEVEL)
        except:
            pass
    return std_parallelism_level()


@always_inline("nodebug")
def task_id_for_device(device_id: Int) -> Int:
    """Maps a device ID to a preferred AsyncRT worker thread ID for CPU affinity.

    Delegates to the shared C++ implementation in DeviceAffinity.cpp which
    handles explicit MODULAR_RUNTIME_DEVICE_TASK_CPU_IDS config,
    NUMA-inferred GPU-to-CPU core mapping, and round-robin fallback.

    Intended for use by affinity-aware task launchers such as
    `_create_task` and `_launch_device_collective`. Pass -1 when no affinity
    hint is needed.

    Args:
        device_id: The integer device ID (e.g. the ordinal of a GPU device).

    Returns:
        An AsyncRT worker thread index to use as `desired_worker_id`, or -1
        if no affinity mapping is configured.
    """
    return Int(
        external_call["MLRT_TaskIdForDevice", Int32](
            Int32(device_id),
        )
    )
