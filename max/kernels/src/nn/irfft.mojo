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
"""Inverse real FFT kernel using cuFFT."""


from std.ffi import external_call, _get_global_or_null

from _cufft.cufft import (
    cufftCreate,
    cufftEstimate1d,
    cufftExecC2R,
    cufftGetSize,
    cufftHandle,
    cufftMakePlan1d,
    cufftSetAutoAllocation,
    cufftSetStream,
    cufftSetWorkArea,
)
from _cufft.types import Type
from _cufft.utils import check_error
from std.complex import ComplexFloat32
from max.gpu.host import DeviceContext
from max.gpu.host._nvidia_cuda import CUDA
from layout import TileTensor, coord_to_index_list


@always_inline
def global_cache_insert(key: String, value: OpaquePointer):
    """Inserts a key-value pair into the global compiler runtime cache.

    Args:
        key: Cache key string.
        value: Opaque pointer value to store under the key.
    """
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(key),
        value,
    )


def _get_fft_workarea(
    buffer_size: Int, ctx: DeviceContext
) raises -> OpaquePointer[MutUntrackedOrigin]:
    # Include device ID in cache key to ensure per-device workspace buffers.
    var fft_buffer_key = String(
        "CUFFT_BUFFER_PTR_", buffer_size, "_DEV_", ctx.id()
    )

    var lookup = _get_global_or_null(fft_buffer_key)
    if lookup:
        # we found the allocated device buffer
        return lookup.unsafe_value()

    # manually allocate the memory on the device, and cache the pointer
    var work_space = ctx.enqueue_create_buffer[.uint8](buffer_size)
    var device_ptr = work_space.take_ptr()

    global_cache_insert(
        fft_buffer_key,
        # bitcast the device pointer to a void * to cache it
        device_ptr.unsafe_bitcast[NoneType](),
    )

    return device_ptr.unsafe_bitcast[NoneType]().unsafe_origin_cast[
        MutUntrackedOrigin
    ]()


def _get_fft_plan[
    create_if_not_found: Bool = True
](
    output_size: Int,
    batch_size: Int,
    workspace_size: Int,
    ctx: DeviceContext,
) raises -> cufftHandle:
    # Include device ID in cache key to ensure per-device cuFFT plans.
    var cached_plan_key = String(
        "CUFFT_PLAN_", output_size, ",", batch_size, "_DEV_", ctx.id()
    )

    var lookup = _get_global_or_null(cached_plan_key)
    if lookup:
        # We found the plan in the cache, so just return it
        return cufftHandle(Int(lookup.unsafe_value()))

    comptime if not create_if_not_found:
        # a valid cufft handle is always non-zero
        return cufftHandle(0)

    var plan = cufftHandle(0)
    var mem_size: Int = 0
    check_error(cufftCreate(Pointer(to=plan)))
    check_error(cufftSetAutoAllocation(plan, 0))
    check_error(
        cufftMakePlan1d(
            plan,
            Int32(output_size),
            Type.CUFFT_C2R,
            Int32(batch_size),
            Pointer(to=mem_size),
        )
    )
    var work_size: Int = 0
    # Get the precise size of the plan, assert that it is less than the allocated size
    check_error(cufftGetSize(plan, Pointer(to=work_size)))
    var work_space_ptr = _get_fft_workarea(workspace_size, ctx)

    if work_size > workspace_size:
        raise Error(
            "Need "
            + String(work_size // 1024 // 1024)
            + " MB of buffer allocated for cuFFT."
        )

    check_error(cufftSetWorkArea(plan, work_space_ptr))

    # We want to cache the cuFFT plan to avoid calling high overhead cuda
    # calls each time the plane is created and destroyed
    global_cache_insert(
        cached_plan_key,
        # we are bitcasting the integer plan to a void * to cache it,
        # because that's what KGEN_CompilerRT_InsertGlobal expects.
        Pointer[NoneType, MutUntrackedOrigin](unsafe_from_address=Int(plan)),
    )

    return plan


def _irfft[
    input_type: DType,
    output_type: DType,
](
    input: TileTensor[input_type, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    n: Int,
    buffer_size_mb: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (
        input.rank == output.rank
    ), "Input and output must have the same rank"
    comptime assert (
        input_type == .float32
    ), "Only Float32 is supported for IRFFT"
    comptime assert (
        output_type == .float32
    ), "Only Float32 is supported for IRFFT"
    # we allocate 64 MB more than the buffer size because the estimation might
    # not be exact.
    var EST_WORKSPACE_SIZE = buffer_size_mb * 1024 * 1024
    var ALLOCATED_WORKSPACE_SIZE = (buffer_size_mb + 64) * 1024 * 1024

    var axis = input.rank - 1
    var cuda_stream = CUDA(ctx.stream())

    # Get input and output dimensions
    var input_shape = coord_to_index_list(input.layout.shape_coord())
    # Signal size is set to half the size of the last dimension of the input
    # tensor, because the input tensor is an interleaved complex value.
    var input_size = input_shape[axis] // 2
    var output_size = n if n > 0 else 2 * (input_size - 1)

    # Verify output dimensions
    var output_shape = coord_to_index_list(output.layout.shape_coord())
    if output_shape[axis] != output_size:
        raise Error(
            "Output shape mismatch: got "
            + String(output_shape[axis])
            + " expected "
            + String(output_size)
        )

    # Calculate batch size.
    var batch_size = 1
    for i in range(input.rank - 1):
        batch_size *= input_shape[i]

    # skip size estimations if the plan is already cached, as
    # the function call is expensive
    var plan = _get_fft_plan[create_if_not_found=False](
        output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
    )
    if plan:
        check_error(cufftSetStream(plan, cuda_stream))
        var input_ptr = input.ptr.unsafe_bitcast[ComplexFloat32]()
        var output_ptr = output.ptr.unsafe_bitcast[Float32]()
        check_error(cufftExecC2R(plan, input_ptr, output_ptr))

        return

    var work_size: Int = 0
    check_error(
        cufftEstimate1d(
            Int32(output_size),
            Type.CUFFT_C2R,
            Int32(batch_size),
            Pointer(to=work_size),
        )
    )

    if work_size < EST_WORKSPACE_SIZE:
        # Create a single cuFFT plan if the workspace size is less than
        # the given buffer size.
        var plan = _get_fft_plan(
            output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        # Notice that we do not want to have this part of the cache
        # The stream is set every time the call is executed and we get the
        # stream from the context we are executing within
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr.unsafe_bitcast[ComplexFloat32]()
        var output_ptr = output.ptr.unsafe_bitcast[Float32]()
        check_error(cufftExecC2R(plan, input_ptr, output_ptr))

    else:
        # If the workspace size is too large, we need to run multiple steps
        # try to find the largest batch size that fits in the workspace
        var reduced_batch_size = batch_size

        while reduced_batch_size > 0:
            reduced_batch_size //= 2
            try:
                check_error(
                    cufftEstimate1d(
                        Int32(output_size),
                        Type.CUFFT_C2R,
                        Int32(reduced_batch_size),
                        Pointer(to=work_size),
                    )
                )
                if work_size < EST_WORKSPACE_SIZE:
                    break
            except e:
                # Try the next work_size
                pass

        if reduced_batch_size == 0:
            raise Error(
                "FFT output signal size is too large, try to increase the"
                " buffer size."
            )

        # Create cuFFT plan
        var plan = _get_fft_plan(
            output_size, reduced_batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
        )

        # Set up cuda stream.
        check_error(cufftSetStream(plan, cuda_stream))

        var input_ptr = input.ptr
        var output_ptr = output.ptr

        while batch_size >= reduced_batch_size:
            # Execute the cuFFT plan for the current batch size
            check_error(
                cufftExecC2R(
                    plan,
                    input_ptr.unsafe_bitcast[ComplexFloat32](),
                    output_ptr.unsafe_bitcast[Float32](),
                )
            )

            # Update the pointers for the next batch
            batch_size -= reduced_batch_size
            input_ptr = input_ptr.unsafe_offset(
                reduced_batch_size * input_shape[axis]
            )
            output_ptr = output_ptr.unsafe_offset(
                reduced_batch_size * output_shape[axis]
            )

        if batch_size > 0:
            # Create a new cuFFT plan for the remaining batch size
            # we reuse the allocated workspace, as it is already large enough
            plan = _get_fft_plan(
                output_size, batch_size, ALLOCATED_WORKSPACE_SIZE, ctx
            )
            check_error(cufftSetStream(plan, cuda_stream))

            check_error(
                cufftExecC2R(
                    plan,
                    input_ptr.unsafe_bitcast[ComplexFloat32](),
                    output_ptr.unsafe_bitcast[Float32](),
                )
            )


def irfft[
    input_type: DType,
    output_type: DType,
](
    input: TileTensor[input_type, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, output_type, address_space=.GENERIC, ...],
    n: Int,
    buffer_size_mb: Int,
    ctx: DeviceContext,
) raises:
    """Compute the inverse real FFT of the input tensor.

    Currently, only applies it to the last dimension.

    Parameters:
        input_type: Element `DType` of the input tensor; must be
            `float32`.
        output_type: Element `DType` of the output tensor; must be
            `float32`.

    Args:
        input: Complex input tensor (TileTensor).
        output: Real output tensor (TileTensor).
        n: Output signal size (if <= 0, computed as 2*(input.size(axis) - 1)).
        buffer_size_mb: Estimated buffer size in MB.
        ctx: Device context.
    """
    # Set `ctx`'s CUcontext as current to satisfy cuFFT's stateful API.
    with ctx.push_context():
        _irfft(input, output, n, buffer_size_mb, ctx)
