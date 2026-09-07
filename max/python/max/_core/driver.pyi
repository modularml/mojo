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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

"""
MAX Driver Python bindings.

Provides low-level access to hardware devices and memory management.
"""

import enum
import os
import types
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Annotated, Any, overload

import max._core.dtype
import numpy
from numpy.typing import NDArray

class Device:
    """
    Represents a compute device available for tensor operations.

    This is the base class for :class:`CPU` and :class:`Accelerator`.
    Do not instantiate this class directly; use :class:`CPU` for host
    devices and :class:`Accelerator` for any hardware accelerator.

    .. code-block:: python

        from max import driver

        cpu = driver.CPU()
        gpu = driver.Accelerator()
    """

    def can_access(self, other: Device) -> bool:
        """
        Checks if this device can directly access memory of another device.

        .. code-block:: python

             from max import driver

             gpu0 = driver.Accelerator(id=0)
             gpu1 = driver.Accelerator(id=1)

             if gpu0.can_access(gpu1):
                 print("GPU0 can directly access GPU1 memory.")

        Args:
            other (Device): The other device to check peer access against.

        Returns:
            bool: True if peer access is possible, False otherwise.
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this device complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
        """

    @property
    def is_host(self) -> bool:
        """
        Whether this device is the CPU (host) device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.is_host
        """

    @property
    def is_host_unified(self) -> bool:
        """
        Whether this device and the host draw from one physical memory pool.

        Reports hardware topology, so it does not predict whether a particular
        buffer is readable from the host.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.is_host_unified
        """

    @property
    def stats(self) -> Mapping[str, Any]:
        """
        Returns utilization data for the device.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            stats = device.stats

        Returns:
            dict: A dictionary containing device utilization statistics.
        """

    @property
    def max_single_alloc_size(self) -> int:
        """Largest single contiguous allocation, in bytes."""

    @property
    def label(self) -> str:
        """
        Returns device label.

        Possible values are:

        - ``cpu`` for host devices.
        - ``gpu`` for accelerators.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.label
        """

    @property
    def api(self) -> str:
        """
        Returns the API used to program the device.

        Possible values are:

        - ``cpu`` for host devices.
        - ``cuda`` for NVIDIA GPUs.
        - ``hip`` for AMD GPUs.
        - ``metal`` for Apple GPUs.

        .. code-block:: python

            from max import driver

            device = driver.CPU()
            device.api
        """

    @property
    def architecture_name(self) -> str:
        """
        Returns the architecture name of the device.

        Examples of possible values:

        - ``gfx90a``, ``gfx942`` for AMD GPUs.
        - ``sm_80``, ``sm_86`` for NVIDIA GPUs.
        - CPU devices raise an exception.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.architecture_name
        """

    @property
    def model_name(self) -> str:
        """
        Returns the model name of the device.

        Examples of possible values:

        - ``NVIDIA H100 80GB HBM3`` for an H100.
        - ``NVIDIA B200`` for a B200.
        - ``AMD Instinct MI300X`` for an MI300X.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device.model_name
        """

    @property
    def id(self) -> int:
        """
        Returns a zero-based device id.

        For a CPU device this is always 0.
        For GPU accelerators this is the id of the device relative to this host.
        Along with the ``label``, an id can uniquely identify a device,
        e.g. ``gpu:0``, ``gpu:1``.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            device_id = device.id

        Returns:
            int: The device ID.
        """

    @property
    def default_queue(self) -> DeviceQueue:
        """
        Returns the default queue for this device.

        The default queue is initialized when the device object is created.

        Returns:
            DeviceQueue: The default execution queue for this device.
        """

    @property
    def is_compatible(self) -> bool:
        """
        Returns whether this device is compatible with MAX.

        Returns:
            bool: True if the device is compatible with MAX, False otherwise.
        """

    def __unsafe_enqueue_py_host_func(self, fn: Callable) -> None:
        """
        Enqueues a Python callable to run on the host after preceding work.

        The callable runs on a driver thread once the device's default
        queue reaches this point, after all previously enqueued work has
        completed. It must not call any device APIs (per the
        ``cuLaunchHostFunc`` contract). Currently only supported on CUDA
        devices.

        This API is intentionally namespaced with a ``__unsafe_`` prefix
        to discourage casual use: there is no safety net for callbacks
        that capture state that outlives the compiled graph, and
        misuse can deadlock or corrupt memory.

        Args:
            fn (Callable[[], None]): A zero-argument callable.

        Raises:
            RuntimeError: If the underlying device does not support host
                callbacks, or if the driver rejects the enqueue.
        """

    def __unsafe_enqueue_async_py_host_func(
        self, fn: Callable, flag: CompletionFlag, value: int, cpu: CPU
    ) -> None:
        """
        Async kickoff variant of ``__unsafe_enqueue_py_host_func``.

        Like ``__unsafe_enqueue_py_host_func``, except the kickoff host
        node dispatches ``fn`` onto ``cpu``'s AsyncRT worker pool and
        returns immediately, so the GPU queue can proceed to
        subsequent nodes concurrently with ``fn`` running on an
        AsyncRT worker thread.

        When ``fn`` finishes, the worker atomic-stores ``value``
        (release ordering) to the 64-bit memory at ``flag``. Pair with
        ``DeviceQueue.wait_for_host_value(flag, value)`` on the same
        queue to gate the downstream consumer kernel.

        The trampoline keeps refcounts on ``flag``'s underlying MLRT
        allocation AND on ``cpu``'s AsyncRT CPUDevice, so neither can
        be released out from under the in-flight task even if the
        Python wrappers are GC'd between enqueue and signal.

        Args:
            fn (Callable[[], None]): A zero-argument callable.
            flag (CompletionFlag): The completion flag the worker will
                signal when ``fn`` returns.
            value (int): The 64-bit value to store on completion
                (matched against the
                ``wait_for_host_value(flag, value)`` consumer).
            cpu (CPU): The CPU device whose AsyncRT worker pool will
                execute ``fn``. Typically just ``max.driver.CPU()``;
                callers needing fine-grained scheduling can pass a
                specific CPU device.

        Raises:
            RuntimeError: If the underlying device does not support
                host callbacks, if the supplied ``cpu`` has no
                associated AsyncRT CPUDevice, or if the driver rejects
                the enqueue.
        """

    def __eq__(self, arg: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def _device_context_ptr(self) -> int:
        """Gets the device context pointer."""

    @staticmethod
    def cpu(id: int = -1) -> CPU:
        """Creates a CPU device. The id is ignored currently."""

class Accelerator(Device):
    def __init__(self, id: int = -1) -> None:
        """
        Creates an accelerator device with the specified ID and memory limit.

        Represents any hardware accelerator attached to the host: the
        graph compiler reaches CUDA, HIP, Metal and plugin-provided
        backends alike through this one device class. Use
        ``isinstance(device, Accelerator)`` to mean "any non-CPU device",
        and :attr:`api` to tell the concrete backends apart.

        Repeated instantiations with a previously-used device-id will still
        refer to the first such instance that was created. This is especially
        important when providing a different memory limit: only the value
        (implicitly or explicitly) provided in the first such instantiation
        is effective.

        .. code-block:: python

          from max import driver
          # Create default accelerator (usually first available GPU)
          device = driver.Accelerator()
          # Or specify GPU id
          device = driver.Accelerator(id=0)  # First GPU
          device = driver.Accelerator(id=1)  # Second GPU
          # Get device id
          device_id = device.id

        Args:
            id (int, optional): The device ID to use. Defaults to -1, which selects
                the first available accelerator.

        Returns:
            Accelerator: A new Accelerator device object.
        """

class CPU(Device):
    def __init__(self, id: int = -1) -> None:
        """
        Creates a CPU device.

        .. code-block:: python

            from max import driver
            # Create default CPU device
            device = driver.CPU()
            # Device id is always 0 for CPU devices
            device_id = device.id

        Args:
            id (int, optional): The device ID to use.
                Defaults to -1.

        Returns:
            CPU: A new CPU device object.
        """

class CompletionFlag:
    """
    An 8-byte completion flag in pinned host memory mapped into a device's address space.

    Lets a CPU thread signal a GPU queue (or vice versa) by
    writing a 64-bit value to a single location that's visible to
    both. Pair with ``DeviceQueue.wait_for_host_value`` (added in
    a follow-on PR) or the ``mo.wait_host_value`` graph op to gate
    downstream GPU work on a host-produced result without a
    second queue or a blocking host callback.

    Currently requires a CUDA-backed ``Device``; constructing
    against any other backend raises ``RuntimeError``.

    .. code-block:: python

        from max.driver import Accelerator, CompletionFlag

        accel = Accelerator()
        flag = CompletionFlag(accel)
        assert flag.load() == 0  # initialized to zero

        # Subsequent PRs add the producer/consumer methods that
        # actually use the flag's device pointer.
    """

    def __init__(self, device: Device) -> None:
        """
        Allocates a fresh device-mapped pinned u64 bound to ``device``.

        Args:
            device: A CUDA-backed device. Other backends raise
                ``RuntimeError``.
        """

    @property
    def device_ptr(self) -> int:
        """
        Device-visible 64-bit address of the 8-byte slot.

        Suitable for passing to graph ops or queue APIs that wait
        on a memory value.
        """

    def reset(self) -> None:
        """
        Clears the flag back to ``0`` with a relaxed atomic store.

        Safe to call before any consumer has observed the address.
        """

    def signal(self, value: int) -> None:
        """
        Release-ordered store of ``value`` to the flag.

        Pairs with the GPU-side ``cuStreamWaitValue64`` (or a
        host-side acquire ``load``).

        Primary intended use is priming the flag at setup time so
        the first captured-graph replay's ``mo.wait_host_value``
        passes immediately, before any async kickoff has run.
        Direct Python signalling on the hot path is usually a
        mistake -- prefer the async-host-func trampoline which
        signals from its AsyncRT worker.

        Args:
            value: The 64-bit value to store.
        """

    def load(self) -> int:
        """
        Acquire-ordered load of the current flag value.

        Pairs with a release-ordered store on the producer side.

        Returns:
            int: Current 64-bit flag value.
        """

    @property
    def _unsafe_ptr(self) -> int:
        """
        Raw 64-bit address of the underlying ``M::Driver::CompletionFlag``.

        Intended for packing into a graph-op payload buffer
        (e.g. for ``mo.wait_host_value``); parallels the
        trampoline/user_data pointers returned by
        ``__unsafe_pack_py_host_func`` for ``mo.launch_host_func``.

        The caller must keep this ``CompletionFlag`` Python object
        alive for the duration of any graph execution that
        references the pointer; the underlying allocation is
        freed when the last ref to the wrapper is dropped. The
        leading underscore marks this as an escape hatch with no
        safety net.
        """

class DeviceEvent:
    """
    Provides access to an event object.

    An event can be used to wait for the GPU execution to reach a certain
    point on the given queue.

    .. code-block:: python

        from max import driver
        # Create a default accelerator device
        device = driver.Accelerator()
        # Create an event on the device
        event = driver.DeviceEvent(device)
        # Record an event on the device (default queue)
        device.default_queue.record_event(event)
        # Wait for execution on the default queue to reach the event
        event.synchronize()
    """

    def __init__(self, device: Device, enable_timing: bool = False) -> None:
        """
        Creates an event for synchronization on the specified device.

        Args:
            device (Device): The device on which to create the event.
            enable_timing (bool): If True, enable GPU timing on this event.
                Events created with ``enable_timing=True`` can be used with
                :meth:`elapsed_time` to measure GPU execution time.
                Defaults to False.

        Raises:
            ValueError: If event creation failed.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            event = driver.DeviceEvent(device)
            timed_event = driver.DeviceEvent(device, enable_timing=True)
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this queue complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
        """

    def is_ready(self) -> bool:
        """
        Returns whether this event is ready.

        Returns:
            bool: True if the event is complete, otherwise false.

        Raises:
            ValueError: If querying the event status returned an error.
        """

    def elapsed_time(self, end_event: DeviceEvent) -> float:
        """
        Returns the elapsed GPU time in milliseconds between this event and ``end_event``.

        Both events must have been created with ``enable_timing=True``
        and recorded on a queue before calling this method. The end
        event must be synchronized before calling this method.

        Args:
            end_event (DeviceEvent): The ending event.

        Returns:
            float: Elapsed time in milliseconds.

        Raises:
            RuntimeError: If either event was not created with timing
                enabled, or if the events have not been recorded.

        .. code-block:: python

            from max import driver

            device = driver.Accelerator()
            start = driver.DeviceEvent(device, enable_timing=True)
            end = driver.DeviceEvent(device, enable_timing=True)

            queue = device.default_queue
            queue.record_event(start)
            # ... GPU work ...
            queue.record_event(end)
            end.synchronize()

            elapsed_ms = start.elapsed_time(end)
        """

    def __eq__(self, arg: object, /) -> bool: ...

class LaunchTraceEntry:
    """
    One operation recorded by ``max.driver.begin_launch_trace``.

    Describes a kernel launch, memory copy, or memset enqueued on a
    device stream. Only the field group matching ``kind`` is
    meaningful; the other groups hold zero values. ``stream_index``
    identifies which stream enqueued the operation, so entries from
    different streams can be told apart in the single enqueue-ordered
    list. ``semantic_hash`` is a deterministic hash of the launch
    parameters that excludes memory addresses, so it is stable across
    runs and suitable for change-detection in tests.
    """

    class OperationKind(enum.Enum):
        """The kind of operation a trace entry describes."""

        KERNEL_LAUNCH = 0

        MEMCPY = 1

        MEMSET = 2

    class MemcpyKind(enum.Enum):
        """The direction of a memcpy entry."""

        NONE = 0

        HTOD = 1

        DTOH = 2

        DTOD = 3

    @property
    def kind(self) -> LaunchTraceEntry.OperationKind:
        """The kind of operation this entry represents."""

    @property
    def name(self) -> str:
        """
        The kernel name, or the driver API name for copies/memsets (e.g. ``cuMemcpyHtoD``).
        """

    @property
    def semantic_hash(self) -> int:
        """Deterministic, address-free hash of the operation parameters."""

    @property
    def grid_x(self) -> int: ...
    @property
    def grid_y(self) -> int: ...
    @property
    def grid_z(self) -> int: ...
    @property
    def block_x(self) -> int: ...
    @property
    def block_y(self) -> int: ...
    @property
    def block_z(self) -> int: ...
    @property
    def shared_mem_bytes(self) -> int: ...
    @property
    def stream_index(self) -> int:
        """
        Identifies the stream this operation was enqueued on, assigned in first-seen order within a trace.
        """

    @property
    def memcpy_kind(self) -> LaunchTraceEntry.MemcpyKind:
        """The copy direction; ``NONE`` unless ``kind`` is ``MEMCPY``."""

    @property
    def memcpy_byte_size(self) -> int: ...
    @property
    def memset_byte_size(self) -> int: ...
    @property
    def memset_value(self) -> int: ...
    @property
    def memset_value_size(self) -> int: ...

class DeviceQueue:
    """
    Provides access to a queue of execution on a device.

    A queue represents a sequence of operations that will be executed in order.
    Multiple queues on the same device can execute concurrently.

    .. code-block:: python

        from max import driver
        # Create a default accelerator device
        device = driver.Accelerator()
        # Get the default queue for the device
        queue = device.default_queue
        # Create a new queue of execution on the device
        new_queue = driver.DeviceQueue(device)
    """

    def __init__(self, device: Device) -> None:
        """
        Creates a new queue of execution associated with the device.

        Args:
            device (Device): The device to create the queue on.

        Returns:
            DeviceQueue: A new queue of execution.
        """

    def synchronize(self) -> None:
        """
        Ensures all operations on this queue complete before returning.

        Raises:
            ValueError: If any enqueued operations had an internal error.
        """

    @overload
    def record_event(self) -> DeviceEvent:
        """
        Records an event on this queue.

        Returns:
            DeviceEvent: A new event that will be signaled when all operations
                submitted to this queue before this call have completed.

        Raises:
            ValueError: If recording the event failed.
        """

    @overload
    def record_event(self, event: DeviceEvent) -> None:
        """
        Records an existing event on this queue.

        Args:
            event (DeviceEvent): The event to record on this queue.

        Raises:
            ValueError: If recording the event failed.
        """

    @overload
    def wait_for(self, stream: DeviceQueue) -> None:
        """
        Ensures all operations on the other queue complete before future work submitted to this queue is scheduled.

        Args:
            stream (DeviceQueue): The queue to wait for.
        """

    @overload
    def wait_for(self, device: Device) -> None:
        """
        Ensures all operations on device's default queue complete before future work submitted to this queue is scheduled.

        Args:
            device (Device): The device whose default queue to wait for.
        """

    def wait_for_host_value(self, flag: CompletionFlag, value: int) -> None:
        """
        Stalls the queue until ``flag``'s 64-bit value equals ``value``.

        Wraps the MLRT ``DeviceStream::enqueueWaitOnHostValue`` primitive
        (CUDA's ``cuStreamWaitValue64``). Typically paired with
        ``Device.__unsafe_enqueue_async_py_host_func`` to gate
        downstream GPU work on a host-side AsyncRT task that signals
        ``flag`` when it finishes -- a queue-internal sync that
        avoids a host ``synchronize()`` and captures cleanly into a
        CUDA graph as a wait-value node.

        Args:
            flag (CompletionFlag): The completion flag to wait on.
                The queue observes ``flag.device_ptr`` via the
                pinned device-mapped alias.
            value (int): The 64-bit value to wait for (equality).

        Raises:
            RuntimeError: If the underlying device does not support
                stream memory ops, or if the driver rejects the
                enqueue.
        """

    def __unsafe_enqueue_async_py_host_func(
        self, fn: Callable, flag: CompletionFlag, value: int, cpu: CPU
    ) -> None:
        """
        Queue-targeted variant of ``Device.__unsafe_enqueue_async_py_host_func``.

        Enqueues a kickoff host node on **this** queue that dispatches ``fn``
        onto ``cpu``'s AsyncRT worker pool and returns immediately. When ``fn``
        finishes, the worker atomic-stores ``value`` (release ordering) to the
        64-bit memory at ``flag``. Pair with
        ``DeviceQueue.wait_for_host_value(flag, value)`` on a consumer queue
        to gate downstream GPU work.

        Use this overload when you need the host callback to run on a side
        queue concurrently with the model queue's forward pass; the
        ``Device`` overload always targets the default queue and therefore
        serializes against any other default-queue work. As of this
        writing no production caller dispatches via this queue
        overload -- ``StructuredOutputOverlapState.enqueue_async_callback``
        intentionally lands on the device default queue so the
        trampoline's ``flag.reset()`` is naturally ordered against the
        next iter's captured-graph wait. The queue variant is exposed
        as future-facing API and exercised by the GPU integration test
        (``test_structured_output_overlap_gpu.py``).

        Args:
            fn (Callable[[], None]): A zero-argument callable.
            flag (CompletionFlag): The completion flag the worker will
                signal when ``fn`` returns.
            value (int): The 64-bit value to store on completion.
            cpu (CPU): The CPU device whose AsyncRT worker pool will
                execute ``fn``.

        Raises:
            RuntimeError: If the underlying device does not support host
                callbacks, if the supplied ``cpu`` has no associated AsyncRT
                CPUDevice, or if the driver rejects the enqueue.
        """

    @property
    def device(self) -> Device:
        """The device this queue is executing on."""

    def _device_context_ptr(self) -> int:
        """Gets the AsyncRT DeviceContext pointer for this specific queue."""

    @property
    def native_stream_handle(self) -> int:
        """
        The native stream handle as an integer, or ``0`` if there is none.

        The handle is the CUDA ``CUstream`` / HIP ``hipStream_t``; ``0`` means
        the stream has no native handle (e.g. a CPU device). Lets native code
        outside MLRT order its own work against this stream -- for example,
        record a CUDA event on it. The handle remains owned by this stream; do
        not destroy it.

        Returns:
            int: The native stream handle, or ``0`` if there is none.
        """

    def __eq__(self, arg: object, /) -> bool: ...

def accelerator_count() -> int:
    """Returns number of accelerator devices available."""

def begin_launch_trace() -> None:
    """
    Starts a process-global recording of enqueued device operations.

    Records kernel launches, memory copies, and memsets across **all**
    streams into one enqueue-ordered list, clearing any previous trace.
    No stream or device handle is needed, so work enqueued on streams the
    caller does not hold (e.g. a compiled graph's internal stream) is still
    captured. Only CUDA and HIP devices record entries; on other devices the
    trace is always empty. Intended for tests and debugging: pair with
    ``take_launch_trace`` to assert which device work a code path enqueues
    and on which stream.
    """

def take_launch_trace() -> list[LaunchTraceEntry]:
    """
    Stops the global recording and returns the recorded entries.

    Returns:
        list[LaunchTraceEntry]: The operations enqueued since
            ``begin_launch_trace``, in enqueue order across all streams.
            Each entry's ``stream_index`` identifies its stream.
    """

def __unsafe_pack_py_host_func(fn: Callable) -> tuple[int, int]:
    """
    Packs a Python callable into a `(trampoline_ptr, user_data_ptr)` pair.

    The returned integers are suitable for passing to
    ``MLRT::DeviceStream::enqueueHostFunc`` (and, by extension, to the
    ``mo.launch_host_func`` custom op). The trampoline is a C ABI
    ``void(void*)`` function; ``user_data_ptr`` owns a heap allocation
    holding the Python callable.

    Ownership transfers to the caller: the user-data pointer MUST be
    consumed by exactly one ``enqueue_host_func`` invocation, which
    causes the trampoline to free it after calling the callable.
    Discarding the pair without enqueueing leaks the callable.

    Args:
        fn (Callable[[], None]): A zero-argument callable.

    Returns:
        tuple[int, int]: ``(trampoline_ptr, user_data_ptr)`` as integers.
    """

def set_virtual_device_count(count: int) -> None:
    """
    Sets the number of virtual devices for device creation.

    When count is greater than 0, Device::create() will return VirtualDevice
    instances instead of real hardware devices for GPU APIs, and
    Device::numberOfDevices() will return this count. This allows creating
    devices for GPU configurations that don't match the current hardware.

    Args:
        count (int): The number of virtual devices. Set to 0 to disable
            virtual device mode.
    """

def get_virtual_device_count() -> int:
    """
    Gets the current virtual device count.

    Returns:
        int: The number of virtual devices, or 0 if virtual device mode
            is disabled.
    """

def is_virtual_device_mode() -> bool:
    """
    Checks if virtual device mode is currently enabled.

    Returns:
        bool: True if virtual device mode is enabled (count > 0), False otherwise.
    """

def enable_all_peer_access() -> None:
    """
    Enables peer-to-peer memory access between all available GPU pairs.

    This must be called before any collective operations (allreduce,
    broadcast, etc.) that require direct GPU-to-GPU memory access.
    It is safe to call multiple times; the underlying runtime caches
    the result after the first successful enablement.

    Raises:
        RuntimeError: If P2P access cannot be enabled between any GPU pair.
    """

def set_virtual_device_api(api: str) -> None:
    """
    Sets the target API for virtual devices in compile-only mode.

    This specifies which GPU API (e.g., "cuda", "hip", "metal") virtual
    devices will use for compilation. Must be called before creating
    virtual devices via set_virtual_device_count().

    Args:
        api (str): The target API string (e.g., "cuda" for NVIDIA,
            "hip" for AMD, "metal" for Apple).
    """

def get_virtual_device_api() -> str:
    """
    Gets the current target API for virtual devices.

    Returns:
        str: The target API string, or empty string if not set.
    """

def set_virtual_device_target_arch(arch: str) -> None:
    """
    Sets the target GPU architecture for virtual devices in compile-only mode.

    This specifies the GPU architecture (e.g., "sm_80", "sm_90") that virtual
    devices will target when compiling code. Must be called before creating
    virtual devices via set_virtual_device_count().

    Args:
        arch (str): The target GPU architecture string (e.g., "sm_80" for
            Ampere/A100, "sm_90" for Hopper/H100).
    """

def get_virtual_device_target_arch() -> str:
    """
    Gets the current target GPU architecture for virtual devices.

    Returns:
        str: The target GPU architecture string, or empty string if not set.
    """

def set_virtual_cpu_target(cpu: str) -> None:
    """
    Sets the CPU target for host-independent kernel codegen.

    When set before any CPU kernel compilation (e.g. before importing
    ``max._interpreter_ops``), CPU kernels compile for this fixed target
    instead of the build host's CPU, so the kernel cache can ship to and be
    reused on a different host. Mirrors
    :func:`set_virtual_device_target_arch` for GPUs.

    Args:
        cpu (str): An LLVM target-CPU name (e.g. "x86-64-v3",
            "neoverse-n1"), or "generic" for the most-portable baseline of
            the host arch family ("x86-64" on x86_64, the armv8-a baseline
            on AArch64; other families raise an error). Empty string
            restores host-CPU codegen. "native" is rejected because it
            would re-leak the build host's CPU.
    """

def get_virtual_cpu_target() -> str:
    """
    Gets the current virtual CPU target.

    Returns:
        str: The CPU target string, or empty string if not set (host CPU).
    """

class Usage(enum.Flag):
    """
    Allocation-intent descriptor for :obj:`Buffer`.

    Flags compose with ``|`` and are tested with ``in``. ``max.driver``
    owns the flag set and its per-backend mapping.
    """

    _boundary_: enum.FlagBoundary = ...

    _flag_mask_: int = 1

    _singles_mask_: int = 1

    _all_bits_: int = 3

    _inverted_: None = None

    DEFAULT = 0
    """
    The allocation Buffer performs today: device memory for a non-host device, ordinary host memory for the CPU.
    """

    STAGING = 1
    """
    Host memory for staging transfers to and from the given device. May be page-locked, depending on the backend.
    """

class Buffer:
    """
    Device-resident buffer representation.

    Allocates memory onto a given device with the provided shape and dtype.
    Buffers can be sliced to provide strided views of the underlying memory,
    but any buffers input into model execution must be contiguous.

    Supports numpy-style slicing but does not currently support setting
    items across multiple indices.

    .. code-block:: python

        from max import driver
        from max.dtype import DType

        # Create a buffer on CPU
        cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32)

        # Create a buffer on GPU
        gpu = driver.Accelerator()
        gpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.float32, device=gpu)

    Args:
        dtype (DType): Data type of buffer elements.
        shape (Sequence[int]): Tuple of positive, non-zero integers denoting the buffer shape.
        device (Device, optional): Device to allocate buffer onto. Defaults to the CPU.
        stream (DeviceQueue, optional): Queue to associate the buffer with.
        usage (Usage, optional): Allocation intent, see :obj:`Usage`.
            Defaults to ``Usage.DEFAULT``.
    """

    @overload
    def __init__(
        self,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        device: Device | None = None,
        usage: Usage = Usage.DEFAULT,
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        stream: DeviceQueue,
        usage: Usage = Usage.DEFAULT,
    ) -> None: ...
    @overload
    def __init__(
        self, shape: Annotated[NDArray, dict(writable=False)], device: Device
    ) -> None: ...
    @property
    def device(self) -> Device:
        """Device on which tensor is resident."""

    @property
    def stream(self) -> DeviceQueue:
        """Stream to which tensor is bound."""

    @property
    def dtype(self) -> max._core.dtype.DType:
        """DType of constituent elements in tensor."""

    @property
    def element_size(self) -> int:
        """Return the size of the element type in bytes."""

    @property
    def is_contiguous(self) -> bool:
        """
        Whether or not buffer is contiguously allocated in memory.

        Returns false if the buffer is a non-contiguous slice.

        Currently, we consider certain situations that are contiguous as
        non-contiguous for the purposes of our engine, such as when a buffer
        has negative steps.
        """

    @property
    def is_host(self) -> bool:
        """
        Whether or not buffer is host-resident.

        Returns false for GPU buffers, true for CPU buffers.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())

            print(cpu_buffer.is_host)
        """

    @property
    def num_elements(self) -> int:
        """
        Returns the number of elements in this buffer.

        Rank-0 buffers have 1 element by convention.
        """

    @property
    def rank(self) -> int:
        """Buffer rank."""

    @property
    def shape(self) -> tuple:
        """Shape of buffer."""

    def contiguous(self) -> Buffer:
        """Creates a contiguous copy of the buffer."""

    @overload
    def copy(self, stream: DeviceQueue) -> Buffer:
        """
        Creates a deep copy on the device associated with the queue.

        Args:
            stream (DeviceQueue): The queue to associate the new buffer with.

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
        """

    @overload
    def copy(self, device: Device | None = None) -> Buffer:
        """
        Creates a deep copy on an optionally given device.

        If device is None (default), a copy is created on the same device.

        .. code-block:: python

            from max import driver
            from max.dtype import DType

            cpu_buffer = driver.Buffer(shape=[2, 3], dtype=DType.bfloat16, device=driver.CPU())
            cpu_copy = cpu_buffer.copy()

            # Copy to GPU
            gpu = driver.Accelerator()
            gpu_copy = cpu_buffer.copy(device=gpu)

        Args:
            device (Device, optional): The device to create the copy on.
                Defaults to None (same device).

        Returns:
            Buffer: A new buffer that is a copy of this buffer.
        """

    @staticmethod
    def mmap(
        filename: os.PathLike,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        mode: numpy._MemMapModeKind = "copyonwrite",
        offset: int = 0,
    ) -> Buffer:
        """
        Creates a memory-mapped buffer from a binary file on disk.

        The constructor argument semantics follow that of np.memmap.
        """

    def inplace_copy_from(self, src: Buffer) -> None:
        """
        Copies the contents of another buffer into this one.

        These buffers may be on different devices.
        Requires that both buffers are contiguous and have same size.
        """

    @staticmethod
    def from_dlpack(array: Any, *, copy: bool | None = None) -> Buffer:
        """
        Creates a buffer from an object implementing the dlpack protocol.

        This usually does not result in a copy, and the producer of the object
        retains ownership of the underlying memory.

        Args:
            array (Any): An object that implements the dlpack protocol.
            copy (bool, optional): Whether to create a copy of the data.
                Defaults to None.

        Returns:
            Buffer: A new buffer that views or copies the dlpack data.
        """

    @staticmethod
    def from_numpy(arr: numpy.ndarray) -> Buffer:
        """
        Creates a buffer from a provided numpy array on the host device.

        The underlying data is not copied unless the array is noncontiguous. If
        it is, a contiguous copy will be returned.

        Args:
            arr (numpy.ndarray): The numpy array to convert.

        Returns:
            Buffer: A new buffer that views or copies the numpy array data.
        """

    def item(self) -> Any:
        """
        Returns the scalar value at a given location.

        Currently implemented only for zero-rank buffers. The return type is
        converted to a Python built-in type.
        """

    @staticmethod
    def scalar(
        value: Any, dtype: max._core.dtype.DType, device: Device | None = None
    ) -> Buffer:
        """
        Creates a scalar value of a given dtype and value.

        If device is None (default), the buffer will be allocated on the CPU.
        """

    @overload
    def to(self, device: Device) -> Buffer:
        """
        Returns a buffer that's guaranteed to be on the given device.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident.
        """

    @overload
    def to(self, stream: DeviceQueue) -> Buffer:
        """
        Returns a buffer that's guaranteed to be on the given device and associated with the given queue.

        The buffer is only copied if the requested device is different from the
        device upon which the buffer is already resident. If the destination
        queue is on the same device, then a new reference to the same buffer is
        returned.
        """

    @overload
    def to(self, devices: Sequence[Device]) -> list[Buffer]:
        """
        Returns a list of buffers that are guaranteed to be on the given devices.

        The buffers are only copied if the requested devices are different from the
        device upon which the buffer is already resident.
        """

    @overload
    def to(self, streams: Sequence[DeviceQueue]) -> list[Buffer]:
        """
        Returns a list of buffers that are guaranteed to be on the given queues.

        The buffers are only copied if the requested queues are different from the
        queue upon which the buffer is already resident.
        """

    def to_numpy(self) -> numpy.ndarray:
        """
        Converts the buffer to a numpy array.

        If the buffer is on the host (CPU), the numpy array aliases the existing memory.
        Otherwise, it is copied to the host device.

        Returns:
            numpy.ndarray: A numpy array containing the buffer data.
        """

    @property
    def pinned(self) -> bool:
        """
        Whether the allocation landed in the device's host memory space. Ask ``usage`` for what was requested.
        """

    @property
    def usage(self) -> Usage:
        """Allocation intent. Slices and views report their parent's usage."""

    def view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int] | None = None
    ) -> Buffer:
        """
        Returns a new buffer with the given type and shape that shares the underlying memory.

        If the shape is not given, it will be deduced if possible, or a
        ValueError is raised.
        """

    @staticmethod
    def zeros(
        shape: Sequence[int],
        dtype: max._core.dtype.DType,
        device: Device | None = None,
        usage: Usage = Usage.DEFAULT,
    ) -> Buffer:
        """
        Allocates a buffer with all elements initialized to zero.

        Args:
            shape (Sequence[int]): The shape of the buffer.
            dtype (DType): The data type of the buffer.
            device (Device, optional): The device to allocate the buffer on.
                Defaults to None (CPU).
            usage (Usage, optional): Allocation intent, see :obj:`Usage`.
                Defaults to ``Usage.DEFAULT``.

        Returns:
            Buffer: A new buffer filled with zeros.
        """

    def __dlpack__(
        self, *, stream: int | None = None, **kwargs
    ) -> types.CapsuleType:
        """Implements part of the dlpack contract."""

    def __dlpack_device__(self) -> tuple:
        """Implements part of the dlpack contract."""

    def __getitem__(self, idx: int | slice | Sequence[int | slice]) -> Buffer:
        """
        Gets a buffer slice.

        Supports full numpy-style slicing. Invocations
        using only integer-based indexes will return zero-rank buffers.
        """

    def __setitem__(
        self, idx: int | slice | Sequence[int | slice], value: Any
    ) -> None:
        """Sets an item in the buffer."""

    def _aligned(self, alignment: int | None = None) -> bool:
        """Returns whether the buffer is aligned to the desired alignment."""

    @overload
    @staticmethod
    def _from_dlpack(arg: object, /) -> Buffer: ...
    @overload
    @staticmethod
    def _from_dlpack(
        arg0: types.CapsuleType, arg1: Device, arg2: int, /
    ) -> Buffer: ...
    def _iterate_indices(self) -> Generator[Sequence[int]]: ...
    def _view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int]
    ) -> Buffer: ...
    def _inplace_copy_from(self, src: Buffer) -> None: ...
    def _data_ptr(self) -> int:
        """Gets the memory address of the buffer data. Internal use only."""

def _batch_inplace_copy(dsts: Sequence[Buffer], srcs: Sequence[Buffer]) -> None:
    """
    Batched copy of ``srcs`` into ``dsts``.

    Sources may be host, pinned, same-device or peer memory in any mix. One
    submission only orders the writes on its own stream, so destinations are
    grouped by device and submitted one batch per device. Identical pairs
    (``dst is src``) are skipped. All buffers must have matching sizes.
    """

class DevicePinnedBuffer(Buffer):
    """
    Creates a pinned host memory allocation tied to the given device.

    Device-pinned memory allocations can provide faster DMA speeds and allow
    properly asynchronous copies between the device and the host.

    Since device-pinned buffers can be used for asynchronous copies they
    don't perform automatic synchronizations in operations like `to_numpy`,
    so synchronization should be handled manually to ensure GPU tasks writing
    to the buffer are complete before reading it on the host.

    .. caution::
      Since this class provides device-pinned memory  it doesn't work on CPU,
      for regular host memory use the Buffer class.

    .. code-block:: python

        from max.driver import DevicePinnedBuffer, Accelerator
        from max.dtype import DType
        import numpy as np

        # Requires GPU device
        device = Accelerator()
        buffer = DevicePinnedBuffer(
            dtype=DType.float32, shape=[1024], device=device
        )

        # Fill with data and transfer to GPU
        np_data = buffer.to_numpy()
        np_data[:] = np.arange(1024, dtype=np.float32)
        gpu_buffer = buffer.to(device)

    Args:
        dtype (DType): Data type of buffer elements.
        shape (Sequence[int]): Tuple of positive, non-zero integers denoting the buffer shape.
        device (Device): GPU/Accelerator device to associate buffer with. Must not be CPU.
        stream (DeviceQueue, optional): Queue to associate the buffer with.

    Raises:
        ValueError: If is a CPU device.
    """

    @overload
    def __init__(
        self, dtype: max._core.dtype.DType, shape: Sequence[int], device: Device
    ) -> None: ...
    @overload
    def __init__(
        self,
        dtype: max._core.dtype.DType,
        shape: Sequence[int],
        stream: DeviceQueue,
    ) -> None: ...
    def __dlpack__(
        self, *, stream: int | None = None, **kwargs
    ) -> types.CapsuleType:
        """
        Export device-pinned buffer using DLPack protocol.

        For device-pinned buffer no synchronization is done in DLPack,
        synchronization should be handled manually.

        Args:
            stream: Optional stream parameter for DLPack protocol.
            **kwargs: Additional keyword arguments for DLPack protocol.

        Returns:
            DLPack capsule for the buffer.
        """

    @staticmethod
    def zeros(
        shape: Sequence[int], dtype: max._core.dtype.DType, device: Device
    ) -> DevicePinnedBuffer:
        """
        Allocates a pinned buffer with all elements initialized to zero.

        Creates a pinned buffer for efficient host-device transfers on GPU/accelerator devices.

        Args:
            shape: The shape of the buffer.
            dtype: The data type of the buffer.
            device: GPU/Accelerator device to associate buffer with. Must not be CPU.

        Returns:
            DevicePinnedBuffer: A pinned buffer with all elements initialized to zero.

        Raises:
            ValueError: If is a CPU device.
        """

    def __getitem__(
        self, idx: int | slice | Sequence[int | slice]
    ) -> DevicePinnedBuffer:
        """
        Gets a buffer slice, preserving the device-pinned type.

        Unlike :obj:`Buffer.__getitem__`, the returned slice is itself a
        :obj:`DevicePinnedBuffer`, so reads such as ``to_numpy`` on the slice
        keep the no-synchronization behavior of device-pinned memory.
        """

    def _view(
        self, dtype: max._core.dtype.DType, shape: Sequence[int]
    ) -> DevicePinnedBuffer: ...

def _release_buffers_to_borrowed(buffers: Sequence[Buffer]) -> list[Buffer]:
    """Convert owning buffers into borrowed wrappers over the same storage."""

def _unsafe_alloc_fast_pinned_buffer(
    dtype: max._core.dtype.DType,
    shape: Sequence[int],
    device: Device,
    threads: int = 16,
    chunk_bytes: int = 536870912,
) -> DevicePinnedBuffer:
    """
    Fast page-locked host allocation for very large host KV-cache buffers.

    Maps one contiguous region and faults it in across ``threads`` parallel
    workers while a single consumer registers it with the device in
    ``chunk_bytes`` chunks, overlapping the two phases. Far faster than the
    per-call ``cuMemAllocHost`` path, and it avoids the ``cuMemAllocHost``
    failure on single >1 TiB allocations.

    UNSAFE / low-level (host KV-cache offloading). The returned buffer is
    NOT garbage-collected: it must be freed explicitly via
    :func:`_unsafe_free_fast_pinned_buffer`, and forgetting to do so leaks
    the mapping. No host/device synchronization is performed -- before
    reading the region on the host (or freeing it) the caller must ensure
    the GPU is done accessing it (host-synchronize the relevant streams).

    Args:
        dtype (DType): Data type of buffer elements (typically ``uint8``).
        shape (Sequence[int]): Buffer shape, e.g. ``[num_blocks, bytes_per_block]``.
        device (Device): GPU/Accelerator device the memory is registered against. Must not be CPU.
        threads (int, optional): Number of parallel page-touch workers. Defaults to 16.
        chunk_bytes (int, optional): Per-call host-register granularity in bytes. Defaults to 512 MiB.

    Returns:
        DevicePinnedBuffer: A pinned host buffer over the mapping. Must be
        freed with :func:`_unsafe_free_fast_pinned_buffer`.

    Raises:
        ValueError: If ``device`` is a CPU device.
    """

def _unsafe_free_fast_pinned_buffer(buffer: DevicePinnedBuffer) -> None:
    """
    Free a buffer from :func:`_unsafe_alloc_fast_pinned_buffer` (unregister + munmap).

    UNSAFE / low-level. The caller MUST first host-synchronize every GPU
    stream that issued copies into the region -- the buffer does not track
    them, and unmapping a region a stream is still copying to/from is a
    use-after-free. After this call the buffer (and any view/slice of it)
    must not be used.

    Args:
        buffer (DevicePinnedBuffer): A buffer from
            :func:`_unsafe_alloc_fast_pinned_buffer`.

    Raises:
        ValueError: If the buffer was not produced by
            :func:`_unsafe_alloc_fast_pinned_buffer`, or was already freed.
    """
