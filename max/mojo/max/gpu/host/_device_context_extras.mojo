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

from std.ffi import external_call, CStringSlice, c_size_t
from std.builtin.device_passable import DevicePassable
from std.collections.optional import OptionalReg
from std.reflection import get_linkage_name, call_location, SourceLocation
from std.sys.compile import DebugLevel, OptimizationLevel
from std.sys import size_of

from max.gpu.host.compile import _compile_code, get_gpu_target

from . import (
    Dim,
    LaunchAttribute,
    ConstantMemoryMapping,
    FuncAttribute,
    Attribute,
)

from ._device_context_metal import MetalEnqueueFunctionArgs

from .device_context import (
    DeviceContext,
    DeviceBuffer,
    DeviceFunction,
    DeviceExternalFunction,
    _checked,
    _CString,
    _DumpPath,
    _check_dim,
    _DeviceFunctionPtr,
    _DeviceContextPtr,
    _FunctionEnqueuer,
)


__extension DeviceBuffer:
    def _tensor_map_encode_tiled(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        box_dim: Pointer[mut=False, Int32, _],
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes a tiled TMA descriptor for this buffer via AsyncRT. Used by
        `max.gpu.host._tensormap.create_tensormap`."""
        _checked(
            external_call["AsyncRT_cuda_tensorMapEncodeTiled", _CString[]](
                tensor_map,
                data_type,
                rank,
                self._handle,
                global_dim,
                global_strides,
                box_dim,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
        )

    def _tensor_map_encode_im2col(
        self,
        tensor_map: MutOpaquePointer[_],
        data_type: Int32,
        rank: Int32,
        global_dim: Pointer[mut=False, Int64, _],
        global_strides: Pointer[mut=False, Int64, _],
        pixel_box_lower_corner: Pointer[mut=False, Int32, _],
        pixel_box_upper_corner: Pointer[mut=False, Int32, _],
        channels_per_pixel: Int32,
        pixels_per_column: Int32,
        element_strides: Pointer[mut=False, Int32, _],
        interleave: Int32,
        swizzle: Int32,
        l2_promotion: Int32,
        oob_fill: Int32,
    ) raises:
        """Encodes an im2col TMA descriptor for this buffer via AsyncRT. Used by
        `max.gpu.host._tensormap.create_tensormap_im2col`."""
        _checked(
            external_call["AsyncRT_cuda_tensorMapEncodeIm2col", _CString[]](
                tensor_map,
                data_type,
                rank,
                self._handle,
                global_dim,
                global_strides,
                pixel_box_lower_corner,
                pixel_box_upper_corner,
                channels_per_pixel,
                pixels_per_column,
                element_strides,
                interleave,
                swizzle,
                l2_promotion,
                oob_fill,
            )
        )


__extension DeviceFunction:
    @doc_hidden
    @always_inline
    def __init__(
        out self,
        ctx: DeviceContext,
        *,
        func_attribute: OptionalReg[FuncAttribute] = None,
    ) raises:
        """Initializes a new DeviceFunction by compiling the function for the specified device.

        Args:
            ctx: The device context to compile the function for.
            func_attribute: Optional attributes to apply to the function, such as shared memory size.

        Raises:
            If compilation fails or if an unsupported function attribute is provided.
        """
        self._context = ctx

        var max_dynamic_shared_size_bytes: Int32 = -1
        if func_attribute:
            if (
                func_attribute.value().attribute
                == Attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES
            ):
                max_dynamic_shared_size_bytes = func_attribute.value().value
            else:
                raise Error(
                    "the function attribute '",
                    func_attribute.value().attribute,
                    "' is not currently supported",
                )

        # const char *AsyncRT_DeviceContext_loadFunction(
        #     const DeviceFunction **result, const DeviceContext *ctx,
        #     const char *moduleName, const char *functionName, const char *data,
        #     size_t dataLen, int32_t maxDynamicSharedBytes, const char *debugLevel,
        #     int32_t optimizationLevel)
        var result: _DeviceFunctionPtr[mut=True] = {}
        self._func_impl = _compile_code[
            Self.func,
            emission_kind=self._emission_kind,
            target=Self.target,
            compile_options=Self.compile_options,
            link_options=Self.link_options,
        ]()
        var debug_level = String(DebugLevel)
        _checked(
            external_call[
                "AsyncRT_DeviceContext_loadFunction",
                _CString[],
                Pointer[_DeviceFunctionPtr[mut=True], origin_of(result)],
                _DeviceContextPtr[mut=True],
                CStringSlice[ImmStaticOrigin],
                CStringSlice[ImmStaticOrigin],
                CStringSlice[ImmStaticOrigin],
                c_size_t,
                Int32,
                CStringSlice[origin_of(debug_level)],
                Int32,
            ](
                Pointer(to=result),
                ctx._handle,
                self._func_impl.module_name.as_c_string_slice(),
                self._func_impl.function_name.as_c_string_slice(),
                self._func_impl.asm.as_c_string_slice(),
                c_size_t(self._func_impl.asm.byte_length()),
                max_dynamic_shared_size_bytes,
                debug_level.as_c_string_slice(),
                Int32(Int(OptimizationLevel)),
            ),
        )
        self._handle = result


__extension DeviceExternalFunction:
    @always_inline
    @__parameter
    def _call_with_pack[
        *Ts: AnyType,
    ](
        imm self,
        ctx: Some[_FunctionEnqueuer],
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        """Launches the device function with the specified arguments and configuration.

        Parameters:
            Ts: Types of the arguments to pass to the device function.

        Args:
            self: The DeviceExternalFunction.
            ctx: The enqueuer to launch the function on.
            args: Arguments to pass to the device function.
            grid_dim: Grid dimensions for the kernel launch.
            block_dim: Block dimensions for the kernel launch.
            cluster_dim: Optional cluster dimensions for multi-GPU execution.
            shared_mem_bytes: Optional amount of shared memory to allocate.
            attributes: Optional list of additional launch attributes.
            constant_memory: Optional list of constant memory mappings.
            location: Source location for the function call.

        Raises:
            If the function launch fails.
        """
        comptime num_args = Ts.length

        var dense_args_addrs = Array[OpaquePointer[MutAnyOrigin], num_args](
            uninitialized=True
        )

        comptime for i in range(num_args):
            # TODO(MSTDL-1904): Validate the safety of this.
            dense_args_addrs[i] = (
                Pointer(to=args[i])
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )

        if cluster_dim:
            attributes.append(
                LaunchAttribute.from_cluster_dim(cluster_dim.value())
            )

        if constant_memory:
            for i in range(len(constant_memory)):
                self._copy_to_constant_memory(constant_memory[i])

        # External functions carry no argument-size metadata, so no per-arg
        # sizes are passed to the enqueuer (matching the previous direct call).
        var no_arg_sizes = OptionalPointer[UInt64, MutAnyOrigin](None)

        if self._context.api() == "metal":
            # Metal takes the launch payload via `args[0]`; see
            # `MetalDeviceContext::enqueueFunctionExecDirect`.
            var dense_args_sizes = Array[UInt64, num_args](fill=0)

            comptime for i in range(num_args):
                dense_args_sizes[i] = UInt64(size_of[Ts[i]]())

            # TODO(GEX-3761): Unchecked path — no argument is encoded as a
            # device pointer, so this launch marks no used buffers resident.
            var dense_args_is_device_ptr = Array[Bool, num_args](fill=False)

            var metal_args = MetalEnqueueFunctionArgs(
                dense_args_addrs.unsafe_ptr()
                .unsafe_origin_cast[MutUntrackedOrigin]()
                .unsafe_bitcast[OpaquePointer[MutUntrackedOrigin]](),
                dense_args_sizes.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                dense_args_is_device_ptr.unsafe_ptr().unsafe_origin_cast[
                    MutUntrackedOrigin
                ](),
                None,
                Int32(0),
            )

            var ptr = (
                Pointer(to=metal_args)
                .unsafe_bitcast[NoneType]()
                .unsafe_mut_cast[True]()
                .as_unsafe_any_origin()
            )
            var metal_args_addrs = [ptr]

            _checked(
                ctx.enqueue(
                    self._handle,
                    grid_dim,
                    block_dim,
                    shared_mem_bytes.or_else(0),
                    attributes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                    len(attributes),
                    metal_args_addrs.unsafe_ptr().as_unsafe_any_origin(),
                    UInt32(num_args),
                    no_arg_sizes,
                ),
                location=location.or_else(call_location()),
            )
            return

        _checked(
            ctx.enqueue(
                self._handle,
                grid_dim,
                block_dim,
                shared_mem_bytes.or_else(0),
                attributes.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin](),
                len(attributes),
                dense_args_addrs.unsafe_ptr().as_unsafe_any_origin(),
                UInt32(num_args),
                no_arg_sizes,
            ),
            location=location.or_else(call_location()),
        )


__extension DeviceContext:
    def _check_supports_default_compile_function(self):
        pass

    @__parameter
    @always_inline
    def enqueue_function[
        declared_arg_types: TypeList[Trait=AnyType, ...],
        //,
        func: def(* args: * declared_arg_types) thin -> None,
        *actual_arg_types: DevicePassable,
        link_options: StaticString = "",
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
        location: Optional[SourceLocation] = None,
    ) raises:
        """Compiles and enqueues a kernel for execution on this device.

        Parameters:
            declared_arg_types: Types of the arguments to pass to the device function.
            func: The function to compile and launch.
            actual_arg_types: The dtypes of the arguments being passed to the function.
            link_options: Additional linker flags and options as a string.
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
            self: The DeviceContext.
            args: Variadic arguments which are passed to the `func`.
            grid_dim: The grid dimensions.
            block_dim: The block dimensions.
            cluster_dim: The cluster dimensions.
            shared_mem_bytes: Per-block memory shared between blocks.
            attributes: A `List` of launch attributes.
            constant_memory: A `List` of constant memory mappings.
            func_attribute: `CUfunction_attribute` enum.
            location: Source location for the function call.

        You can pass the function directly to `enqueue_function`
        without compiling it first:

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
        from max.gpu.host import DeviceContext

        def kernel():
            print("hello from the GPU")

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.enqueue_function(compiled_func, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        # If shared_mem_bytes is specified but func_attribute is not,
        # automatically set MAX_DYNAMIC_SHARED_SIZE_BYTES if needed (>48KB)
        var inferred_func_attribute = func_attribute
        if not func_attribute and shared_mem_bytes:
            var max_shared = self._get_max_dynamic_shared_memory_bytes(
                shared_mem_bytes.value()
            )
            if max_shared > 0:
                inferred_func_attribute = (
                    FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(max_shared)
                )

        var gpu_kernel = self.compile_function[
            func,
            dump_asm=dump_asm,
            dump_llvm=dump_llvm,
            link_options=link_options,
            _dump_sass=_dump_sass,
            _ptxas_info_verbose=_ptxas_info_verbose,
        ](func_attribute=inferred_func_attribute)

        gpu_kernel._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )

    @__parameter
    @always_inline
    def enqueue_function[
        *Ts: DevicePassable
    ](
        self,
        f: DeviceFunction,
        *args: *Ts,
        grid_dim: Dim,
        block_dim: Dim,
        cluster_dim: OptionalReg[Dim] = None,
        shared_mem_bytes: OptionalReg[Int] = None,
        var attributes: List[LaunchAttribute] = [],
        var constant_memory: List[ConstantMemoryMapping] = [],
        location: Optional[SourceLocation] = None,
    ) raises:
        """Enqueues a pre-compiled checked function for execution on this device.

        This overload requires a `DeviceFunction` that was compiled with
        type checking enabled (via `compile_function`). The function
        will verify that the argument types match the declared types at
        compile time.

        Parameters:
            Ts: Argument dtypes.

        Args:
            self: The DeviceContext.
            f: The compiled function to execute.
            args: Arguments to pass to the function.
            grid_dim: Dimensions of the compute grid, made up of thread
                blocks.
            block_dim: Dimensions of each thread block in the grid.
            cluster_dim: Dimensions of clusters (if the thread blocks are
                grouped into clusters).
            shared_mem_bytes: Amount of shared memory per thread block.
            attributes: Launch attributes.
            constant_memory: Constant memory mapping.
            location: Source location for the function call.

        ```mojo
        from max.gpu.host import DeviceContext

        def kernel(x: Int):
            print("Value:", x)

        with DeviceContext() as ctx:
            var compiled_func = ctx.compile_function[kernel]()
            ctx.enqueue_function(compiled_func, 42, grid_dim=1, block_dim=1)
            ctx.synchronize()
        ```

        Raises:
            If the operation fails.
        """
        _check_dim["DeviceContext.enqueue_function", "grid_dim"](
            grid_dim, location=call_location()
        )
        _check_dim["DeviceContext.enqueue_function", "block_dim"](
            block_dim, location=call_location()
        )

        f._call_with_pack_checked(
            self,
            *args,
            grid_dim=grid_dim,
            block_dim=block_dim,
            cluster_dim=cluster_dim,
            shared_mem_bytes=shared_mem_bytes,
            attributes=attributes^,
            constant_memory=constant_memory^,
            location=location.or_else(call_location()),
        )
