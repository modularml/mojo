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

"""SHMEM runtime context and multi-GPU launch helper.

Provides `SHMEMContext`, which wraps the SHMEM runtime lifecycle and exposes a
`DeviceContext`-compatible interface for submitting kernels and managing
symmetric-heap buffers. Use `shmem_launch` to run a per-GPU function across
all attached GPUs.
"""

from max.algorithm import parallelize
from std.collections.optional import OptionalReg
from std.os import abort
from std.builtin.device_passable import DevicePassable
from std.sys import (
    CompilationTarget,
    argv,
    has_amd_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)

from max.gpu.host import (
    ConstantMemoryMapping,
    DeviceAttribute,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    Dim,
    FuncAttribute,
    LaunchAttribute,
)
from max.gpu.host.device_context import _DumpPath
from max.gpu.host.launch_attribute import (
    LaunchAttributeID,
    LaunchAttributeValue,
)

from ._mpi import MPI_Finalize, MPI_Init
from .shmem_api import (
    SHMEM_TEAM_NODE,
    shmem_barrier_all_on_stream,
    shmem_finalize,
    shmem_init,
    shmem_init_thread_tcp,
    shmem_init_thread_mpi,
    shmem_module_finalize,
    shmem_module_init,
    shmem_team_my_pe,
    shmem_team_t,
)
from .shmem_buffer import SHMEMBuffer


def shmem_launch[func: def(ctx: SHMEMContext) thin raises]() raises:
    """Takes a function defining a SHMEM program and launches it
    on one thread for each GPU you have attached.

    Parameters:
        func: The function to run once per attached GPU per node.

    ```mojo
    def simple_shift(ctx: SHMEMContext) raises:
        var destination = ctx.enqueue_create_buffer[.int32](1)

        ctx.enqueue_function[simple_shift_kernel](
            destination, grid_dim=1, block_dim=1
        )

        ctx.barrier_all()

        var msg = Int32(0)
        destination.enqueue_copy_to(Pointer(to=msg))

        ctx.synchronize()

        var mype = shmem_my_pe()
        print("PE:", mype, "received message:", msg)
        assert_equal(msg, (mype + 1) % shmem_n_pes())

    def main() raises:
        shmem_launch[simple_shift]()
    ```

    This initializes SHMEM and runs the program in parallel across each attached
    GPU, taking care of initialization and cleanup logic. It will initialize and
    finalize MPI on the main thread if running on NVIDIA.

    Any unhandled exceptions will abort with the device id and error message of
    the exception.

    Raises:
        If SHMEM initialization or the launched function fails.
    """

    comptime if has_nvidia_gpu_accelerator():
        _shmem_launch_mpi[func]()
    elif has_amd_gpu_accelerator():
        _shmem_launch_tcp[func]()
    else:
        CompilationTarget.unsupported_target_error[
            operation=__get_current_function_name()
        ]()


def _shmem_launch_mpi[func: def(ctx: SHMEMContext) thin raises]() raises:
    var _argv = argv()
    var argc = len(_argv)
    MPI_Init(argc, _argv)

    # Enable any exceptions inside the closure passed to abort with the original
    # error and device ID in the message, as `parallelize` can't run on raising
    # functions.
    def shmem_error_wrapper(device_id_node: Int):
        try:
            var ctx = DeviceContext(device_id=device_id_node)
            with SHMEMContext(ctx) as shmem_ctx:
                func(shmem_ctx)
        except e:
            abort(
                String(
                    "SHMEM failure on local device id: ",
                    device_id_node,
                    ": ",
                    e,
                )
            )

    var npes_node = DeviceContext.number_of_devices()

    # Same number of tasks as worker threads
    parallelize(shmem_error_wrapper, npes_node, npes_node)

    # Cleanup MPI resources
    MPI_Finalize()


def _shmem_launch_tcp[func: def(ctx: SHMEMContext) thin raises]() raises:
    # Enable any exceptions inside the closure passed to abort with the original
    # error and device ID in the message, as `parallelize` can't run on raising
    # functions.
    def shmem_error_wrapper(device_id_node: Int):
        try:
            var ctx = DeviceContext(device_id=device_id_node)
            with SHMEMContext[tcp=True](ctx) as shmem_ctx:
                func(shmem_ctx)
        except e:
            abort(
                String(
                    "SHMEM failure on local device id: ",
                    device_id_node,
                    ": ",
                    e,
                )
            )

    var npes_node = DeviceContext.number_of_devices()

    # Same number of tasks as worker threads
    parallelize(shmem_error_wrapper, npes_node, npes_node)


struct SHMEMContext[tcp: Bool = False](ImplicitlyCopyable):
    """Usable as a context manager to run kernels on a GPU with SHMEM support,
    on exit it will finalize SHMEM and clean up resources.

    Example:

    ```mojo
    from shmem import SHMEMContext

    with SHMEMContext() as ctx:
        ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
    ```
    """

    var _ctx: DeviceContext
    var _main_stream: DeviceStream
    var _priority_stream: DeviceStream
    var _begin_event: DeviceEvent
    var _end_event: DeviceEvent
    var _multiprocessor_count: Int
    var _cooperative: Bool
    var _thread_per_gpu: Bool

    def __init__(
        out self, team: shmem_team_t = SHMEM_TEAM_NODE
    ) raises where Self.tcp == False:
        """Initializes a device context with SHMEM support.

        This constructor initializes MPI and SHMEM, and creates a device
        context for the current PE's assigned GPU device.

        Warning: if you're not using this as a context manager, you must call
        `SHMEMContext.finalize()` manually.

        Raises:
            If initialization fails.
        """
        shmem_init()

        # nvshmem and rocshmem behave differently here, nvshmem requires that
        # you set the current context to a device ID corresponding with the
        # GPU id on the node i.e. if team_my_pe is 3 then DeviceContext.id()
        # should also be 3. rocshmem does this inside the rocshmem_init()
        # call, and each process will launch kernels on the associated pe
        # from MPI, the DeviceContext.id() will always be 0, but it's
        # associated with a different GPU in each process.
        var mype = shmem_team_my_pe(team)
        self._ctx = DeviceContext(device_id=Int(mype))
        # Store main stream to avoid retrieving it in each collective launch.
        self._main_stream = self._ctx.stream()

        # Set up priority stream and events to be reused across collective launches
        var priority = self._ctx.stream_priority_range().greatest
        self._priority_stream = self._ctx.create_stream(priority=priority)
        self._begin_event = self._ctx.create_event()
        self._end_event = self._ctx.create_event()

        # Store attributes to avoid retrieving them in each collective launch.
        self._multiprocessor_count = self._ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        self._cooperative = Bool(
            self._ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH)
        )
        self._thread_per_gpu = False

    def __init__(out self, ctx: DeviceContext) raises where Self.tcp == False:
        """Initializes a device context with SHMEM support, using one thread
        per GPU.

        This constructor expects that MPI has already been initialized in the
        main thread, it then initializes SHMEM, and creates a device context for
        the associated PE on this node.

        Warning: if you're not using this as a context manager, you must call
        `SHMEMContext.finalize()` manually.

        Raises:
            If initialization fails.
        """
        shmem_init_thread_mpi(ctx)
        self._ctx = ctx
        # Store main stream to avoid retrieving it in each collective launch.
        self._main_stream = self._ctx.stream()

        # Set up priority stream and events to be reused across collective launches
        var priority = self._ctx.stream_priority_range().greatest
        self._priority_stream = self._ctx.create_stream(priority=priority)
        self._begin_event = self._ctx.create_event()
        self._end_event = self._ctx.create_event()

        # Store attributes to avoid retrieving them in each collective launch.
        self._multiprocessor_count = self._ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        self._cooperative = Bool(
            self._ctx.get_attribute(DeviceAttribute.COOPERATIVE_LAUNCH)
        )
        self._thread_per_gpu = True

    def __init__(
        out self,
        ctx: DeviceContext,
        node_id: Int = -1,
        total_nodes: Int = -1,
        gpus_per_node: Int = -1,
        server_ip: String = "-1",
        server_port: Int = -1,
    ) raises where Self.tcp == True:
        """Initializes a device context with SHMEM support, using one thread
        per GPU and TCP bootstrapping with a unique ID.

        Warning: if you're not using this as a context manager, you must call
        `SHMEMContext.finalize()` manually.

        Raises:
            If initialization fails.
        """
        shmem_init_thread_tcp(
            ctx, node_id, total_nodes, gpus_per_node, server_ip, server_port
        )
        self._ctx = ctx
        # Store main stream to avoid retrieving it in each collective launch.
        self._main_stream = self._ctx.stream()

        # Set up priority stream and events to be reused across collective launches
        var priority = self._ctx.stream_priority_range().greatest
        self._priority_stream = self._ctx.create_stream(priority=priority)
        self._begin_event = self._ctx.create_event()
        self._end_event = self._ctx.create_event()

        # Store attributes to avoid retrieving them in each collective launch.
        self._multiprocessor_count = self._ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )
        # TODO(MSTDL-1761): add ability to query AMD cooperative launch
        # capability with: hipLaunchAttributeCooperative and create function
        # that works across NVIDIA/AMD. For now assume cooperative capability.
        self._cooperative = True
        self._thread_per_gpu = True

    def __enter__(var self) -> Self:
        """Context manager entry method.

        Returns:
            Self for use in with statements.
        """
        return self^

    def __deinit__(deinit self):
        """Context manager exit method.

        Automatically finalizes SHMEM when exiting the context.
        """
        try:
            self.finalize()
        except e:
            abort(String(e))

    def finalize(mut self) raises:
        """Finalizes the SHMEM runtime environment.

        Cleans up SHMEM and MPI resources.

        Raises:
            If SHMEM or MPI finalization fails.
        """
        shmem_finalize()
        if not self._thread_per_gpu:
            MPI_Finalize()

    def barrier_all(self) raises:
        """Performs a barrier synchronization across all PEs.

        All PEs must call this function before any PE can proceed past the
        barrier.

        Raises:
            If the barrier operation fails.
        """
        shmem_barrier_all_on_stream(self._main_stream)

    def enqueue_create_buffer[
        dtype: DType
    ](self, size: Int) raises -> SHMEMBuffer[dtype]:
        """Creates a SHMEM buffer that can be accessed by all PEs.

        Parameters:
            dtype: The data type of elements in the buffer.

        Args:
            size: Number of elements in the buffer.

        Returns:
            A SHMEMBuffer instance for the allocated memory.

        Raises:
            String: If buffer creation fails.
        """
        return SHMEMBuffer[dtype](self._ctx, size)

    @always_inline
    @__parameter
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: The declared argument types from the function
                signature (usually inferred).
            func: The function to launch.
            actual_arg_types: The types of the arguments being passed (usually inferred).
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from shmem import SHMEMContext

        def kernel():
            print("hello from the GPU")

        with SHMEMContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)

        shmem_module_init(gpu_kernel)

        gpu_kernel._call_with_pack_checked(
            self._ctx,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )

        shmem_module_finalize(gpu_kernel)

    @always_inline
    @__parameter
    def enqueue_function_collective_checked[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        dump_asm: _DumpPath = False,
        dump_llvm: _DumpPath = False,
        _dump_sass: _DumpPath = False,
        _ptxas_info_verbose: Bool = False,
    ](
        self,
        *args: *actual_arg_types,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: The declared argument types from the function
                signature (usually inferred).
            func: The function to launch.
            actual_arg_types: The types of the arguments being passed (usually inferred).
            dump_asm: To dump the compiled assembly, pass `True`, or a file
                path to dump to, or a function returning a file path.
            dump_llvm: To dump the generated LLVM code, pass `True`, or a file
                path to dump to, or a function returning a file path.
            _dump_sass: Only runs on NVIDIA targets, and requires CUDA Toolkit
                to be installed. Pass `True`, or a file path to dump to, or a
                function returning a file path.
            _ptxas_info_verbose: Only runs on NVIDIA targets, and requires CUDA
                Toolkit to be installed. Changes `dump_asm` to output verbose
                PTX assembly (default `False`).

        Args:
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.

        You can pass the function directly to `enqueue_function` without
        compiling it first:

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        If you are reusing the same function and parameters multiple times, this
        incurs 50-500 nanoseconds of overhead per enqueue, so you can compile it
        first to remove the overhead:

        ```mojo
        with DeviceContext() as ctx:
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.enqueue_function[kernel](grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```
        """
        comptime assert (
            has_nvidia_gpu_accelerator()
        ), "only available on NVIDIA GPUs"
        var gpu_kernel = self._ctx.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=func_attribute)
        shmem_module_init(gpu_kernel)

        var block_size = block_dim[0] * block_dim[1] * block_dim[2]
        var shared_mem_bytes_val = (
            shared_mem_bytes.value() if shared_mem_bytes else 0
        )
        var max_blocks_sm = (
            gpu_kernel.occupancy_max_active_blocks_per_multiprocessor(
                block_size, shared_mem_bytes_val
            )
        )
        var grid_size = -1
        var launch_failed = True

        var grid_x = grid_dim[0]
        var grid_y = grid_dim[1]
        var grid_z = grid_dim[2]
        if grid_x == 0 and grid_y == 0 and grid_z == 0:
            grid_size = 0
        elif grid_x != 0 and grid_y != 0 and grid_z != 0:
            grid_size = grid_x * grid_y * grid_z

        if grid_size == 0:
            if max_blocks_sm == 0:
                launch_failed = False
            grid_x = max_blocks_sm * self._multiprocessor_count
            grid_y = 1
            grid_z = 1
        elif grid_size > 0:
            if (
                max_blocks_sm > 0
                and grid_size <= max_blocks_sm * self._multiprocessor_count
            ):
                launch_failed = False

        if launch_failed:
            raise Error(
                "One or more GPUs cannot collectively launch the kernel"
            )

        # Mark point in main stream and wait for it to complete in priority stream
        self._main_stream.record_event(self._begin_event)
        self._priority_stream.enqueue_wait_for(self._begin_event)

        if self._cooperative:
            attributes.append(
                LaunchAttribute(
                    id=LaunchAttributeID.COOPERATIVE,
                    value=LaunchAttributeValue(True),
                )
            )
        else:
            print(
                "Warning: cooperative launch not supported on at least one PE;"
                " GPU-side synchronization may cause hang"
            )
        gpu_kernel._call_with_pack_checked(
            self._priority_stream,
            *args,
            grid_dim=Dim(grid_x, grid_y, grid_z),
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
        )
        # Mark point in priority stream and wait for it to complete in main stream
        self._priority_stream.record_event(self._end_event)
        self._main_stream.enqueue_wait_for(self._end_event)
        shmem_module_finalize(gpu_kernel)

    @always_inline
    def synchronize(self) raises:
        """Blocks until all asynchronous calls on the stream associated with
        this device context have completed.

        Raises:
            If synchronization fails.
        """
        # const char * AsyncRT_DeviceContext_synchronize(const DeviceContext *ctx)
        self._ctx.synchronize()

    @always_inline
    def get_device_context(self) -> DeviceContext:
        """Returns the device context associated with this SHMEMContext.

        Returns:
            The device context associated with this SHMEMContext.
        """
        return self._ctx

    @staticmethod
    @always_inline
    def number_of_devices(
        *, api: String = String(DeviceContext.default_device_info.api)
    ) -> Int:
        """Returns the number of devices available that support the specified API.

        This function queries the system for available devices that support the
        requested API (such as CUDA or HIP). It's useful for determining how many
        accelerators are available before allocating resources or distributing work.

        Args:
            api: Requested device API (for example, "cuda" or "hip"). Defaults
                to the device API specified by current target accelerator.

        Returns:
            The number of available devices supporting the specified API.

        """
        return DeviceContext.number_of_devices(api=api)
