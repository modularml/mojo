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
"""Helpers for dispatching collective operations across devices."""

from std.collections import Array, Optional
from std.runtime._asyncrt import TaskGroup

from max.gpu.host import DeviceContext, DeviceContextArray
from max.runtime.asyncrt import task_id_for_device


@always_inline
def _launch_device_collective[
    num_devices: Int,
    F: def[Int]() raises -> None,
](func: F, var dev_ctxs: Array[DeviceContext, num_devices]) raises:
    """Dispatch async tasks to call func[i]() for each device in dev_ctxs."""

    # One Optional[Error] slot per device; None means no error.
    # Each task writes only to its own index, so there is no data race.
    var errors = Array[Optional[Error], num_devices](fill=Optional[Error]())

    # Wrap the launch function in a Mojo async function which does not raise.
    @always_inline
    @__parameter
    async def wrapper[index: Int]() -> None:
        try:
            func[index]()
        except e:
            errors[index] = e^

    # Set up a task group to launch the tasks in parallel.
    var tg = TaskGroup()
    comptime for i in range(num_devices):
        # Dispatch to the worker thread that has affinity for this device.
        var worker_id = task_id_for_device(Int(dev_ctxs[i].id()))
        tg._create_task(wrapper[i](), desired_worker_id=worker_id)

    # Wait for all tasks to complete.
    tg.wait()

    # Re-raise the first error encountered.
    comptime for i in range(num_devices):
        if errors[i]:
            raise errors[i].take()


@always_inline
def _launch_device_collective[
    num_devices: Int,
    F: def[Int]() raises -> None,
](func: F, var dev_ctxs: DeviceContextArray) raises:
    """Dispatch async tasks to call func[i]() for each device in dev_ctxs.

    `DeviceContextArray` overload. Forwards to the `Array` overload
    by unpacking the array's underlying storage.
    """

    comptime assert (
        dev_ctxs.length == num_devices
    ), "expected dev_ctxs to have the same number of elements as num_devices"

    _launch_device_collective[num_devices](
        func,
        rebind[Array[DeviceContext, num_devices]](
            dev_ctxs.device_contexts^
        ).copy(),
    )
