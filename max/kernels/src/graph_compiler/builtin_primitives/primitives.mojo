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

from std.logger import Logger
from std.math import fma, gcd
from std.ffi import external_call, c_size_t
from std.sys import size_of, align_of

from max.algorithm.functional import elementwise

from extensibility import StaticTensorSpec
from extensibility import (
    ComputeOutputFusion,
    ComputeOutputFusionTile,
    ElementwiseFusion,
    ElementwiseFusionTile,
    InputFusion,
    OutputFusion,
    OutputFusionTile,
    get_kernel_tile_shape,
)
from std.collections import Array
from max.gpu import block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host import (
    DeviceGraph,
    DeviceGraphBuilder,
    DeviceGraphCache,
    DeviceGraphInput,
)
from max.gpu.host.device_context import _DeviceBufferPtr, _DeviceContextPtr
from max.gpu.host.info import is_accelerator, is_cpu, is_gpu
from std.memory import OwnedPointer, MaybeUninit, alloc, dealloc
from std.os import abort
from layout import (
    Coord,
    CoordLike,
    Idx,
    IntTuple,
    TensorLayout,
    TileTensor,
    row_major,
    coord_to_index_list,
)
from layout.coord import ComptimeInt
from layout.tile_io import (
    GenericToLocalTileCopier,
    LocalToGenericTileCopier,
)
from layout.tile_tensor import stack_allocation
from std.memory import unsafe_memcpy
from std.memory.unsafe_pointer import unsafe_cast

from nn.concat import concat
from nn.reshape import reshape
from nn.slice import slice_as_view
from layout.coord import DynamicCoord
from layout.int_tuple import _IntTupleToCoordLike, UNKNOWN_VALUE
from layout.tile_layout import Layout as TileLayout
from extensibility import register_internal
from extensibility import (
    IOSpec,
    ManagedTensorSlice,
)
from extensibility import IO
from extensibility import (
    DynamicTensor,
    _shape_types_compatible,
    get_kernel_simd_width,
    simd_load_from_managed_tensor_slice,
)

from std.utils import Index, IndexList, StaticTuple

from max.runtime.async_value import AnyAsyncValueRef, _AsyncValuePtr

from .buffer_plan import BufferPlanState, BufferPlanStats

comptime MutByteBuffer = DynamicTensor[.int8, 1]
comptime ImmutByteBuffer = DynamicTensor[.int8, 1]
comptime logger = Logger()

# ===-----------------------------------------------------------------------===#
# Helper Structures
# ===-----------------------------------------------------------------------===#


def pack_string_res(
    str_ptr: Pointer[mut=False, Byte, _], str_len: Int
) raises -> String:
    var span = Span(unsafe_ptr=str_ptr, length=str_len)
    # We can not free the resource ptr embedded in MEF, create a copy
    return String(StringSlice(from_utf8=span))


# ===-----------------------------------------------------------------------===#
# Async Packing/Unpacking functions
# ===-----------------------------------------------------------------------===#


@no_inline
def create_index_async(value: Int, async_ptr: OpaquePointer[MutAnyOrigin]):
    external_call["MGP_RT_CreateAsync_ssizet", NoneType](value, async_ptr)


@no_inline
@export
def create_si64_async(
    value: Int64, async_ptr: OpaquePointer[MutAnyOrigin]
) abi("Mojo"):
    external_call["MGP_RT_CreateAsync_int64t", NoneType](value, async_ptr)


@no_inline
def create_i1_async(
    value: Bool,
    async_ptr: OpaquePointer[MutAnyOrigin],
):
    external_call["MGP_RT_CreateAsync_bool", NoneType](value, async_ptr)


struct OwnedByteBuffer(DeviceGraphInput, ImplicitlyCopyable, Movable):
    """Owning composite for an `mgp.buffer` value: a non-owning `MutByteBuffer`
    view (precomputed pointer + shape) plus an `AnyAsyncValueRef` storage handle
    that keeps the backing memory alive.

    Structurally mirrors the C++ `TensorBufferRef` (`{data, size, storageRef}`).
    Copying shares the backing memory (retains the storage); at the pack site the
    storage is surrendered net-zero into a real `TensorBufferRef`.
    """

    var view: MutByteBuffer
    var storage: AnyAsyncValueRef

    def __init__(out self, view: MutByteBuffer, var storage: AnyAsyncValueRef):
        """Builds the composite from a view and its storage handle.

        Args:
            view: A non-owning `MutByteBuffer` over the memory.
            storage: The owning storage handle that keeps the memory alive.
        """
        self.view = view
        self.storage = storage^

    def __init__(out self, var buffer: DeviceBuffer[.int8]):
        """Builds the composite from a DeviceBuffer.

        Args:
            buffer: A DeviceBuffer which owns the underlying memory.
        """
        var shape = Index(len(buffer))

        var view = MutByteBuffer(buffer.unsafe_ptr(), shape)
        var storage = AnyAsyncValueRef(storage_buf=buffer^)

        self = Self(view, storage^)

    def unsafe_ptr(self) -> Pointer[Int8, MutAnyOrigin]:
        """Returns the view's raw device data pointer.

        Returns:
            The non-owning device data pointer of the view.
        """
        return self.view.unsafe_ptr()

    def size(self) -> Int:
        """Returns the view's size in bytes.

        Returns:
            The byte size of the view.
        """
        return self.view.size()

    def take_storage(deinit self) -> AnyAsyncValueRef:
        """Hands the owning storage handle to the pack site, consuming self.

        Returns:
            The storage handle moved out of the composite.
        """
        return self.storage^

    def async_pack(self) -> AnyAsyncValueRef:
        """Packs the buffer into a freshly allocated `AsyncValue`.

        Unlike `mogg.async.pack`, which fills a runtime-provided async slot, this
        builds a brand-new `AsyncValue` holding a `TensorBufferRef` over the same
        backing memory -- used to register a device-graph output. The storage
        handle is copied (retained), so the borrowed composite stays valid.

        Returns:
            An owning reference to the new `AsyncValue`.
        """
        var ptr = self.unsafe_ptr()
        var n = self.size()
        var storage = AnyAsyncValueRef(copy=self.storage)
        # AsyncValue *MGP_RT_CreateBufferRefAsyncValue(
        #     AsyncValue *storage, void *data, size_t size)
        var handle = external_call[
            "MGP_RT_CreateBufferRefAsyncValue", _AsyncValuePtr[mut=True]
        ](storage^.take_handle(), ptr, n)
        return AnyAsyncValueRef(handle)

    def to_device_buffer(self, ctx: DeviceContext) -> DeviceBuffer[.int8]:
        """Wraps the view's memory in a non-owning `DeviceBuffer` for a copy.

        Rebuilds a fresh view from the origin-erased data pointer so the
        (possibly immutably borrowed) composite's `view` field origin is not
        required at the call site.

        Args:
            ctx: The device context the buffer is associated with.

        Returns:
            A non-owning `DeviceBuffer` over the view's memory.
        """
        return MutByteBuffer(
            self.unsafe_ptr(), Index(self.size())
        ).to_device_buffer(ctx)

    def write_graph_key(self, mut writer: Some[Writer]):
        writer.write(t"Buffer({self.size()})")

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        # An `mgp.buffer` graph input is a mutable buffer the graph writes in
        # place (a KV cache, say), so giving it a private stable location would
        # silently discard those writes.
        #
        # Mutability now survives lowering: `!mo.buffer` lowers to a
        # mut-marked `!mgp.tensor`, and `OwnedTensor[.., mut=True]` handles
        # the in-place registration (`register_in_place_input` +
        # address-in-key). A raw `!mgp.buffer` reaching `add_input` therefore
        # indicates a lowering bug, not a missing feature; if a legitimate
        # buffer-typed graph input ever appears, implement it the same way
        # the mutable OwnedTensor path does.
        abort("device graph inputs of buffer type indicate a lowering bug")


struct OwnedTensor[
    dtype: DType, rank: Int, mut: Bool = False, host: Bool = False
](DeviceGraphInput, ImplicitlyCopyable, Movable):
    """Owning composite for an `mgp.tensor` value: a non-owning `DynamicTensor`
    view (precomputed pointer + shape) plus an `AnyAsyncValueRef` storage handle
    that keeps the backing memory alive.

    The tensor-typed analogue of `OwnedByteBuffer`. Copying shares the backing
    memory (retains the storage); at the pack site the storage is surrendered
    net-zero into a real tensor `TensorBufferRef`.

    Parameters:
        dtype: The element dtype of the tensor view.
        rank: The rank of the tensor view.
        mut: Whether the tensor is mutable (the lowered form of `!mo.buffer`,
            e.g. a KV cache the model writes in place). A mutable tensor used
            as a device-graph input must not be copied to a graph-private
            stable location; its address joins the graph cache key instead.
        host: Whether the tensor's data is host-resident (the lowered form of
            a cpu-placed `!mgp.tensor`, e.g. attention dispatch metadata).
            Recorded ops read host values at enqueue time, freezing them into
            the graph, so a host tensor used as a device-graph input has its
            full contents baked into the graph cache key: changed bytes miss
            the cache and re-record instead of replaying stale dispatch.
            Mutually exclusive with `mut` (in-place host writes contradict
            value-keyed replay; the op verifiers reject the combination).
    """

    var tensor: DynamicTensor[Self.dtype, Self.rank]
    var storage: AnyAsyncValueRef

    def __init__(
        out self,
        tensor: DynamicTensor[Self.dtype, Self.rank],
        var storage: AnyAsyncValueRef,
    ):
        """Builds the composite from a tensor view and its storage handle.

        Args:
            tensor: A non-owning `DynamicTensor` over the memory.
            storage: The owning storage handle that keeps the memory alive.
        """
        self.tensor = tensor
        self.storage = storage^

    def unsafe_ptr(self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        """Returns the view's raw device data pointer.

        Returns:
            The non-owning device data pointer of the view.
        """
        return self.tensor.unsafe_ptr()

    def shape(self) -> IndexList[Self.rank]:
        """Returns the tensor view's shape.

        Returns:
            The shape of the tensor view.
        """
        return self.tensor.shape()

    def bytecount(self) -> Int:
        """Returns the tensor view's size in bytes.

        Returns:
            The byte size of the tensor view.
        """
        return self.tensor.bytecount()

    def take_storage(deinit self) -> AnyAsyncValueRef:
        """Hands the owning storage handle to the pack site, consuming self.

        Returns:
            The storage handle moved out of the composite.
        """
        return self.storage^

    def async_pack(self) -> AnyAsyncValueRef:
        """Packs the tensor into a freshly allocated `AsyncValue`.

        The tensor analogue of `OwnedByteBuffer.async_pack`: builds a new
        `AsyncValue` holding a tensor `TensorBufferRef` (view + spec) over the
        same backing memory, for registering a device-graph output. The storage
        handle is copied (retained), so the borrowed composite stays valid.

        Returns:
            An owning reference to the new `AsyncValue`.
        """
        var shape = self.shape()
        var ptr = self.unsafe_ptr()
        var n = self.bytecount()
        var storage = AnyAsyncValueRef(copy=self.storage)
        # AsyncValue *MGP_RT_CreateTensorRefAsyncValue(
        #     AsyncValue *storage, void *data, size_t size, size_t rank,
        #     const size_t *shape, DType dtype)
        var handle = external_call[
            "MGP_RT_CreateTensorRefAsyncValue", _AsyncValuePtr[mut=True]
        ](
            storage^.take_handle(),
            ptr.unsafe_bitcast[NoneType](),
            n,
            Self.rank,
            Pointer(to=shape.data),
            self.dtype,
        )
        return AnyAsyncValueRef(handle)

    def write_graph_key(self, mut writer: Some[Writer]):
        comptime assert not (
            Self.mut and Self.host
        ), "a device-graph input cannot be both mutable and host-resident"
        comptime if Self.mut:
            # A mutable input is recorded at its live address (no stable twin),
            # so the address is part of what distinguishes graphs: a moved
            # buffer must miss the cache and rebuild rather than replay stale
            # addresses.
            writer.write(
                t"MutTensor({self.dtype}, {self.shape()}, "
                t"{Int(self.unsafe_ptr())})"
            )
        else:
            comptime if Self.host:
                # Recorded ops read host values at enqueue time, so replay is
                # only sound for the exact bytes that were recorded: the full
                # contents join the key, and changed bytes re-record. Bytes
                # are written as decimal integers so the contribution cannot
                # contain the cache's `|` framing byte.
                writer.write(t"HostTensor({self.dtype}, {self.shape()}, ")
                var raw = self.unsafe_ptr().unsafe_bitcast[UInt8]()
                for i in range(self.bytecount()):
                    writer.write(Int(raw[unsafe_offset=i]), ",")
                writer.write(")")
            else:
                writer.write(t"Tensor({self.dtype}, {self.shape()})")

    def allocate_stable(self, mut builder: DeviceGraphBuilder) raises -> Self:
        comptime assert not (
            Self.mut and Self.host
        ), "a device-graph input cannot be both mutable and host-resident"
        comptime if Self.mut:
            # A mutable tensor (the lowered form of !mo.buffer, e.g. a KV
            # cache) is written in place by the graph: a graph-private stable
            # copy would silently discard those writes. Register an in-place
            # marker so the replay copy skips this position, and record
            # against the caller's live buffer. `write_graph_key` puts the
            # address in the cache key, which is what makes replaying the
            # recorded address sound.
            #
            # This is not the storage-pinning trap the immutable path's
            # comment warns about: the returned copy retains `self.storage`
            # only for the build closure's lifetime -- the graph itself
            # retains nothing. Address stability across replays is the
            # buffer owner's property (e.g. the KV manager's block pool),
            # policed by address-in-key.
            builder.register_in_place_input()
            return self
        else:
            var shape = self.shape()

            # The stable twin lives where the input's data lives: recorded
            # kernels read device inputs where they run, and recorded enqueues
            # (dispatch reads, HtoD copy nodes) bake the *host* address of a
            # host input, so both need a graph-stable location.
            var buffer = builder.create_input_buffer[Self.dtype](
                shape.flattened_length(), is_host=Self.host
            )

            var view = DynamicTensor[Self.dtype, Self.rank](
                buffer.unsafe_ptr(), shape
            )

            comptime if Self.host:
                # Host values are read at enqueue time while the build closure
                # records, so the stable buffer must hold the caller's bytes
                # *before* any op is recorded against it. (The replay-time
                # refresh copies the same bytes again -- byte-identical by
                # cache-key construction -- purely to keep the positional
                # input-refresh invariant uniform.)
                var src = self.unsafe_ptr().unsafe_bitcast[UInt8]()
                var dst = view.unsafe_ptr().unsafe_bitcast[UInt8]()
                for i in range(self.bytecount()):
                    dst[unsafe_offset=i] = src[unsafe_offset=i]

            # Storage is the pool buffer, never `self.storage` -- retaining
            # the caller's handle would pin its memory for the graph's
            # lifetime.
            return Self(view, AnyAsyncValueRef(storage_buf=buffer^))


@no_inline
def create_tensor_spec_async[
    spec_rank: Int
](spec: IndexList[spec_rank], async_ptr: OpaquePointer[MutAnyOrigin]):
    # Mojo impl is bitwise compatible with cpp variant, can construct TensorSpec in mojo
    # and pass it back to C++ -- However, this is an issue for the heap allocated dims.
    # For the benefit of simplicity, allocate the shapes and ptrs and free explicitly after
    var storage = Array[_, spec_rank](
        fill_with=lambda (i: Int) {imm spec} -> Int: spec[i]
    )

    external_call["MGP_RT_CreateAsyncTensorShape", NoneType](
        storage.unsafe_ptr(), spec_rank, async_ptr
    )


@export
def empty_destructor(ptr: Pointer[UInt8, MutUntrackedOrigin]) abi("Mojo"):
    pass


@no_inline
def unpack_state_ctx(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> StateContext:
    var ptr = external_call[
        "MGP_RT_UnpackStateContext",
        StateContextRef,
    ](async_ptr)

    return StateContext(ptr)


@no_inline
def unpack_device_ctx(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> DeviceContext:
    var ptr = external_call[
        "MGP_RT_UnpackDeviceContext",
        _DeviceContextPtr[mut=True],
    ](async_ptr)

    return DeviceContext(ptr)


@no_inline
def unpack_buffer_ref(
    async_ptr: OpaquePointer[MutAnyOrigin],
) -> OwnedByteBuffer:
    var size: UInt64 = 0
    var data_ptr = external_call[
        "MGP_RT_GetDataFromBuffer",
        OpaquePointer[MutAnyOrigin],
    ](async_ptr, Pointer(to=size))
    var shape = IndexList[1](Int(size))
    var view = MutByteBuffer(data_ptr.unsafe_bitcast[Int8](), shape)
    # Retain the backing storage of the source async value so this composite
    # keeps the memory alive if it (or a derivative) is re-packed as an output.
    return OwnedByteBuffer(
        view, AnyAsyncValueRef(retained_storage_of=async_ptr)
    )


@no_inline
def unpack_tensor[
    buffer_rank: Int,
    tensor_rank: Int,
    dtype: DType,
    mut: Bool = False,
    host: Bool = False,
](tensor_async_ptr: OpaquePointer[MutAnyOrigin]) -> OwnedTensor[
    dtype, buffer_rank, mut, host
]:
    # Tensor and the underlying buffer must have the same rank, unless it is a
    # scalar tensor stored with a DynamicTensor<[1]>
    comptime assert tensor_rank == buffer_rank or (
        tensor_rank == 0 and buffer_rank == 1
    )
    var shapes = IndexList[buffer_rank]()
    var buffer_ptr = external_call[
        "MGP_RT_GetShapeAndDataFromTensor",
        OpaquePointer[MutAnyOrigin],
    ](
        Pointer(to=shapes.data),
        tensor_async_ptr,
    )

    comptime if tensor_rank == 0:
        shapes[0] = 1

    var view = DynamicTensor[dtype, buffer_rank](
        buffer_ptr.unsafe_bitcast[Scalar[dtype]](), shapes
    )
    # Retain the backing storage of the source async value so this composite
    # keeps the memory alive if it (or a derivative) is re-packed as an output.
    return {view, AnyAsyncValueRef(retained_storage_of=tensor_async_ptr)}


@no_inline
def unpack_tensor_spec[
    spec_rank: Int
](async_ptr: OpaquePointer[MutAnyOrigin]) -> IndexList[spec_rank]:
    var storage = Array[Int, spec_rank](uninitialized=True)
    external_call[
        "MGP_RT_GetTensorShapeFromAsync",
        NoneType,
    ](storage.unsafe_ptr(), spec_rank, async_ptr)
    var shape = IndexList[spec_rank]()

    comptime for i in range(spec_rank):
        shape[i] = storage[i]

    return shape


@always_inline
def get_buffer_data(
    buffer: MutByteBuffer,
) -> Pointer[Int8, MutAnyOrigin]:
    return buffer.unsafe_ptr()


# ===-----------------------------------------------------------------------===#
# MGP Tensor Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor.create")
@no_inline
def mgp_tensor_create[
    spec_rank: Int,
    buffer_rank: Int,
    dtype: DType,
    mut: Bool = False,
    host: Bool = False,
](
    buffer: OwnedByteBuffer,
    spec: IndexList[spec_rank],
) -> OwnedTensor[
    dtype, buffer_rank, mut, host
]:
    # The tensor shares the buffer's backing memory, so it retains the buffer's
    # storage handle (copy) to keep it alive independently.
    var storage = AnyAsyncValueRef(copy=buffer.storage)
    comptime if spec_rank == 0:
        # We promote scalar tensor to tensor<[1]>
        comptime assert buffer_rank == 1
        var view = DynamicTensor[dtype, buffer_rank](
            buffer.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](IndexList[1](1)),
        )
        return {view, storage^}
    else:
        comptime assert spec_rank == buffer_rank
        var view = DynamicTensor[dtype, buffer_rank](
            buffer.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
            rebind[IndexList[buffer_rank]](spec),
        )
        return {view, storage^}


@register_internal("mgp.tensor.extract.tensor_spec")
@no_inline
def mgp_tensor_extract_tensor_spec[
    tensor_rank: Int,
    buffer_rank: Int,
    dtype: DType,
](tensor: OwnedTensor[dtype, buffer_rank, _, _]) -> IndexList[tensor_rank]:
    comptime if tensor_rank == 0:
        comptime assert buffer_rank == 1
        return rebind[IndexList[tensor_rank]](IndexList[0]())
    else:
        comptime assert buffer_rank == tensor_rank
        return rebind[IndexList[tensor_rank]](tensor.shape().canonicalize())


@register_internal("mgp.tensor.extract.buffer")
@no_inline
def mgp_tensor_extract_buffer[
    buffer_rank: Int,
    dtype: DType,
](tensor: OwnedTensor[dtype, buffer_rank, _, _]) -> OwnedByteBuffer:
    # Unwrap the tensor into a size-less buffer view, retaining the tensor's
    # storage so the buffer keeps the backing memory alive independently.
    var view = MutByteBuffer(
        tensor.tensor.unsafe_ptr[.int8](),
        IndexList[1](tensor.bytecount()),
    )
    return OwnedByteBuffer(view, AnyAsyncValueRef(copy=tensor.storage))


@register_internal("mgp.tensor.slice")
@no_inline
def mgp_tensor_slice[
    rank: Int,
    dtype: DType,
](
    input: OwnedTensor[dtype, rank, ...],
    output_spec: IndexList[rank],
    start: OwnedTensor[.int64, 1, ...],
    dev_context: DeviceContext,
) raises -> OwnedTensor[dtype, rank, input.mut, input.host]:
    var input_shape = input.shape()

    # Find k: the first non-size-1 input dimension (the sliced dimension).
    var k = rank
    for i in range(rank):
        if input_shape[i] != 1:
            k = i
            break

    # Compute stride_k = product of input dims strictly after k.
    var stride_k = 1
    for i in range(k + 1, rank):
        stride_k *= input_shape[i]

    # start is a 1-element vector holding the scalar start value for
    # dimension k.  (mogg.slice scalars are rank-0 in MO but are lowered to
    # rank-1 DynamicTensors of size 1 by TensorCreateOp::emitMojo.)
    var start_k = Int(start.unsafe_ptr()[unsafe_offset=0]) if k < rank else 0

    # Normalize a negative start and clamp into [0, dim_k], matching the
    # clamping in `mo.slice`'s shape function; a start past the end yields an
    # empty slice.
    var elem_offset = 0
    if k < rank:
        var dim_k = input_shape[k]
        var normalized = dim_k + start_k if start_k < 0 else start_k
        elem_offset = min(max(normalized, 0), dim_k) * stride_k

    var out_elems = 1
    for i in range(rank):
        out_elems *= output_spec[i]
    comptime dtype_size = size_of[dtype]()

    # Slice via the byte-buffer path so the slice is registered with the device
    # runtime where required. The transient byte buffer retains the input's
    # storage handle, which the slice then hands to the output tensor.
    var base_bytes = OwnedByteBuffer(
        MutByteBuffer(
            input.unsafe_ptr().unsafe_bitcast[Int8](), Index(input.bytecount())
        ),
        AnyAsyncValueRef(copy=input.storage),
    )
    var slice_bytes = mgp_buffer_slice(
        base_bytes,
        elem_offset * dtype_size,
        out_elems * dtype_size,
        dev_context,
    )
    var view = DynamicTensor[dtype, rank](
        slice_bytes.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](), output_spec
    )
    return {view, slice_bytes.take_storage()}


# ===-----------------------------------------------------------------------===#
# MGP Buffer Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.alloc")
@no_inline
def mgp_buffer_alloc(
    byte_size: Int, dev_context: DeviceContext
) raises -> OwnedByteBuffer:
    # Default to alignment of 0 which means kPreferredMemoryAlignment if cRawAlign is kUnknownSize (SizeUtils.h).
    # alias alignment = 0 if bRawAlign == UInt64.MAX else Int(bRawAlign)

    # This primitive has a byte-size input, so always assume a byte format
    return {dev_context.enqueue_create_buffer[.int8](byte_size)}


@register_internal("mgp.device_graph.alloc")
@no_inline
def mgp_device_graph_alloc[
    is_host: Bool
](byte_size: Int, builder: DeviceGraphBuilder) raises -> OwnedByteBuffer:
    return {builder.create_buffer[.int8](byte_size, is_host=is_host)}


@register_internal("mgp.buffer.constant")
@export
def mgp_buffer_constant(
    resource_ptr: OpaquePointer[MutAnyOrigin],
    resource_bytecount: Int,
) abi("Mojo") -> OwnedByteBuffer:
    # Should we keep the alignment? It seems that the static alignment is
    # dropped in the kernels anyway.
    # Constant memory is owned by the resource system, not refcounted, so the
    # storage handle is the empty (non-tracked) reference.
    var view = MutByteBuffer(
        resource_ptr.unsafe_bitcast[Int8](), IndexList[1](resource_bytecount)
    )
    return OwnedByteBuffer(view, AnyAsyncValueRef())


@no_inline
def fill_buffer[dtype: DType](buf: MutByteBuffer, *vals: Int):
    var ptr = buf.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]()
    var offset: Int = 0
    for val in vals:
        ptr.unsafe_store(offset, Scalar[dtype](val))
        offset += 1


@register_internal("mgp.buffer.set_with_index")
@no_inline
def mgp_buffer_set_with_index[
    bDevice: StaticString
](buffer: OwnedByteBuffer, *vals: Int) raises:
    assert is_cpu[bDevice](), "set_with_index can only work on cpu buffers"
    var bufSize = buffer.size()
    var numArgs = len(vals)
    assert (
        bufSize % numArgs == 0
    ), "buffer size not divisible by number of index args"

    var elSize = bufSize // numArgs
    if elSize == 4:
        fill_buffer[.int32](buffer.view, *vals)
    elif elSize == 8:
        fill_buffer[.int64](buffer.view, *vals)
    else:
        raise Error("unsupported element size")


@register_internal("mgp.buffer.to_bool")
@no_inline
def mgp_buffer_to_bool[bDevice: StaticString](buffer: OwnedByteBuffer) -> Bool:
    assert is_cpu[bDevice](), "to_bool can only work on cpu buffers"
    var bufSize = buffer.size()
    assert bufSize == 1, "buffer size must be a size of 1"
    return buffer.unsafe_ptr()[unsafe_offset=0] != 0


@register_internal("mgp.buffer.to_index")
@no_inline
def mgp_buffer_to_index(
    buffer: OwnedByteBuffer,
) raises -> Int:
    var bufSize = buffer.size()
    if bufSize == 4:
        return Int(buffer.unsafe_ptr().unsafe_bitcast[Int32]()[unsafe_offset=0])
    if bufSize == 8:
        return Int(buffer.unsafe_ptr().unsafe_bitcast[Int64]()[unsafe_offset=0])

    raise Error(
        "mgp.buffer.to_index must be called on either a 4- or 8-byte buffer"
    )


@register_internal("mgp.buffer.slice")
@no_inline
def mgp_buffer_slice(
    buffer: OwnedByteBuffer, offset: Int, size: Int, dev_context: DeviceContext
) raises -> OwnedByteBuffer:
    """Returns a sub-buffer view of `buffer` at byte `offset` for `size` bytes.

    The slice goes through `DeviceBuffer.create_sub_buffer` so backends whose
    runtime only accepts registered pointers register it.
    On other devices the returned address is simply `buffer + offset`.
    The parent's storage handle is retained to keep the backing memory alive.

    Args:
        buffer: The parent buffer to slice.
        offset: Byte offset of the slice within the parent.
        size: Byte size of the slice.
        dev_context: The device context the buffer is associated with.

    Returns:
        An `OwnedByteBuffer` viewing the sliced region and retaining the parent's
        backing storage.

    Raises:
        If `[offset, offset + size)` does not fit within the parent buffer.
    """
    var parent = buffer.to_device_buffer(dev_context)
    var sub = parent.create_sub_buffer[.int8](offset, size)
    var view = MutByteBuffer(sub.unsafe_ptr(), Index(size))
    return OwnedByteBuffer(view, AnyAsyncValueRef(copy=buffer.storage))


@register_internal("mgp.buffer.bulk_slice")
@no_inline
def mgp_buffer_bulk_slice[
    N: Int,
    //,
](
    base: OwnedByteBuffer,
    offsets: Array[Int, N],
    sizes: Array[Int, N],
    dev_context: DeviceContext,
) raises -> Array[OwnedByteBuffer, N]:
    """Bulk slice: produce N non-overlapping sub-buffers from a pool buffer.

    Parameters:
        N: Number of slices.

    Args:
        base: The pool buffer.
        offsets: Byte offset of each slice within the pool.
        sizes: Byte size of each slice.
        dev_context: The device context the pool buffer is associated with.

    Returns:
        An Array of N OwnedByteBuffer slices into the pool, each retaining
        the pool's backing storage.

    Raises:
        If any slice does not fit within the pool buffer.
    """
    var result = Array[MaybeUninit[OwnedByteBuffer], N](uninitialized=True)

    # Placement-initialize each uninitialized slot to avoid running the
    # destructor.
    var initialized = 0
    try:
        for i in range(N):
            result[i].unsafe_write(
                mgp_buffer_slice(base, offsets[i], sizes[i], dev_context)
            )
            initialized += 1
    except e:
        # Need to destroy any previously created `OwnedByteBuffer` to not
        # leak resources if `mpg_buffer_slice()` raises an error.
        for i in range(initialized):
            result[i].unsafe_ptr().unsafe_deinit_pointee()
        result^.deinit_with(MaybeUninit[OwnedByteBuffer].unsafe_forget)
        raise e

    return {unsafe_assume_initialized = result^}


@register_internal("mgp.buffer.plan")
@no_inline
def mgp_buffer_plan[
    num_static_sizes: Int,
    num_runtime_sizes: Int,
    //,
    alignments: Array[Int, num_static_sizes + num_runtime_sizes],
    can_share: Array[
        Int,
        (num_static_sizes + num_runtime_sizes)
        * (num_static_sizes + num_runtime_sizes),
    ],
    static_sizes: Array[Int, num_static_sizes],
](runtime_sizes: Array[Int, num_runtime_sizes]) -> Tuple[
    Int, Array[Int, num_static_sizes + num_runtime_sizes]
]:
    """Runtime memory planning for buffers.

    Given static and runtime size information along with a sharing matrix for
    allocations, returns the high watermark size and offsets for each
    allocation.

    The allocations are ordered as: [static_sizes..., runtime_sizes...]
    where the first num_static_sizes allocations have compile-time known sizes,
    and the remaining num_runtime_sizes allocations have runtime sizes.

    can_share is a flat NxN matrix (row-major) where can_share[i*N+j]=1 iff
    allocations i and j have non-overlapping lifetimes and can therefore
    occupy the same memory slot. N = num_static_sizes + num_runtime_sizes.

    Parameters:
        num_static_sizes: Number of allocations with static sizes.
        num_runtime_sizes: Number of allocations with runtime sizes.
        alignments: Alignment requirements for each allocation.
        can_share: NxN sharing matrix (row-major, 0/1 values).
        static_sizes: Compile-time known sizes for first num_static_sizes allocations.

    Args:
        runtime_sizes: Runtime sizes for last num_runtime_sizes allocations.

    Returns:
        A tuple containing:
        - highWatermark: Total memory required.
        - offsets: Offsets for each allocation (static_sizes first, then runtime_sizes).
    """

    @__parameter
    def compute_static_allocations(
        out result: BufferPlanState[
            alignments,
            can_share,
        ],
    ):
        result = {}
        result.allocate_greedy(materialize[static_sizes]())

    comptime state = compute_static_allocations()

    # If all sizes are static, then we can avoid materializing the allocator
    # state.
    comptime if num_runtime_sizes == 0:
        comptime stats = state.stats()
        logger.debug(stats)

        comptime results = state.take_results()
        return materialize[results]()
    else:
        var runtime_state = materialize[state]()
        runtime_state.allocate_greedy[start=num_static_sizes](runtime_sizes)

        logger.debug(runtime_state.stats())
        return runtime_state^.take_results()


@register_internal("mgp.buffer.concat")
@no_inline
def mgp_buffer_concat[
    bDevice: StaticString
](
    output: OwnedByteBuffer,
    inputs: Array[OwnedByteBuffer, ...],
    call_ctx: DeviceContext,
) raises:
    var output_lt = TileTensor(
        output.unsafe_ptr(),
        row_major(Coord(output.size())),
    )
    var input_tensors = StaticTuple[_, inputs.length](
        TileTensor(inputs[0].unsafe_ptr(), row_major(Coord(inputs[0].size())))
        .as_unsafe_any_origin()
        .as_immut()
    )
    for i in range(1, len(inputs)):
        input_tensors[i] = (
            TileTensor(
                inputs[i].unsafe_ptr(), row_major(Coord(inputs[i].size()))
            )
            .as_unsafe_any_origin()
            .as_immut()
        )
    concat[.int8, bDevice, None](output_lt, 0, input_tensors, context=call_ctx)


@register_internal("mgp.buffer.device_to_host")
@no_inline
def mgp_buffer_device_to_host[
    cOtherDevice: StaticString,
    dHostDevice: StaticString,
](
    dev_buf: OwnedByteBuffer,
    host_buf: OwnedByteBuffer,
    dev_ctx: DeviceContext,
) raises:
    comptime if is_cpu[dHostDevice]() and is_accelerator[cOtherDevice]():
        dev_ctx.enqueue_copy[.int8](
            host_buf.unsafe_ptr(),
            dev_buf.to_device_buffer(dev_ctx),
        )
    else:
        raise Error(
            "mgp.buffer.device_to_host must be scheduled on an accelerator"
            " device"
        )


@register_internal("mgp.buffer.device_to_device")
@no_inline
def mgp_buffer_device_to_device[
    cSrcDevice: StaticString,
    dDstDevice: StaticString,
](
    src_buf: OwnedByteBuffer,
    dst_buf: OwnedByteBuffer,
    src_dev_ctx: DeviceContext,
    dst_dev_ctx: DeviceContext,
) raises:
    comptime if is_gpu[cSrcDevice]() and is_gpu[dDstDevice]():
        # The graph emits explicit mgp.device_wait ops around this copy to
        # synchronize the source and destination streams, so the driver must
        # not insert its own cross-stream synchronization here.
        dst_dev_ctx.enqueue_copy_no_cross_stream_sync[.int8](
            dst_buf.to_device_buffer(dst_dev_ctx),
            src_buf.to_device_buffer(src_dev_ctx),
        )
    elif is_cpu[cSrcDevice]() and is_cpu[dDstDevice]():
        unsafe_memcpy(
            dest=dst_buf.unsafe_ptr(),
            src=src_buf.unsafe_ptr(),
            count=src_buf.size(),
        )
    else:
        raise Error(
            "mgp.buffer.device_to_device can be scheduled between same device"
            " dtypes (cpu-cpu) or (gpu-gpu)"
        )


@no_inline
def _memset_buffer[
    dtype: DType, bDevice: StaticString
](
    buffer: OwnedByteBuffer, val: Scalar[dtype], dev_context: DeviceContext
) raises:
    """Fills every `dtype`-sized element of `buffer` with `val`.

    Dispatches to the device memset on an accelerator, and to a direct host
    store loop on cpu (the AsyncRT memset external is device-only).

    Parameters:
        dtype: The unsigned integer element type whose width matches the
            element byte size.
        bDevice: The device the buffer lives on (`cpu` or `gpu`).

    Args:
        buffer: The buffer whose elements are set.
        val: The element value replicated across the buffer.
        dev_context: The device context the buffer is associated with.
    """
    var count = buffer.size() // size_of[dtype]()
    comptime if is_accelerator[bDevice]():
        # Wrap the existing device memory in a non-owning typed DeviceBuffer
        # (no allocation), then memset it -- mirrors `to_device_buffer`.
        var dev_buf = DeviceBuffer[dtype](
            dev_context,
            buffer.unsafe_ptr().unsafe_bitcast[Scalar[dtype]](),
            count,
            owning=False,
        )
        dev_context.enqueue_memset[dtype](dev_buf, val)
    else:
        # cpu: fill the raw bytes directly via a typed store loop.
        var ptr = buffer.unsafe_ptr().unsafe_bitcast[Scalar[dtype]]()
        for i in range(count):
            ptr.unsafe_store(i, val)


@register_internal("mgp.buffer.memset")
@no_inline
def mgp_buffer_memset[
    bDevice: StaticString
](
    buffer: OwnedByteBuffer,
    value_bits: UInt64,
    elem_size: Int,
    dev_context: DeviceContext,
) raises:
    """Sets every `elem_size`-byte element of `buffer` to a scalar pattern.

    The scalar is the little-endian byte pattern formed by the low `elem_size`
    bytes of `value_bits`. Reinterpreting those bytes as an unsigned integer of
    the matching width and filling with it is bit-exact for the buffer's
    originating dtype, so one primitive memsets any element type. Works
    uniformly for `bDevice` == cpu and gpu.

    Parameters:
        bDevice: The device the buffer lives on (`cpu` or `gpu`).

    Args:
        buffer: The buffer to fill.
        value_bits: The scalar byte pattern; only the low `elem_size` bytes are
            used.
        elem_size: The element size in bytes, one of {1, 2, 4, 8}.
        dev_context: The device context the buffer is associated with.

    Raises:
        If `elem_size` is not one of {1, 2, 4, 8}.
    """
    # `cast` to the matching-width unsigned int truncates to the low N bits,
    # which is exactly the little-endian low-byte pattern.
    if elem_size == 1:
        _memset_buffer[.uint8, bDevice](
            buffer, value_bits.cast[.uint8](), dev_context
        )
    elif elem_size == 2:
        _memset_buffer[.uint16, bDevice](
            buffer, value_bits.cast[.uint16](), dev_context
        )
    elif elem_size == 4:
        _memset_buffer[.uint32, bDevice](
            buffer, value_bits.cast[.uint32](), dev_context
        )
    elif elem_size == 8:
        _memset_buffer[.uint64, bDevice](buffer, value_bits, dev_context)
    else:
        raise Error("mgp.buffer.memset: elem_size must be one of {1, 2, 4, 8}")


@register_internal("mgp.buffer.host_to_device")
@no_inline
def mgp_buffer_host_to_device[
    cHostDevice: StaticString,
    dOtherDevice: StaticString,
](
    host_buf: OwnedByteBuffer,
    dev_buf: OwnedByteBuffer,
    dev_ctx: DeviceContext,
) raises:
    comptime if is_accelerator[dOtherDevice]() and is_cpu[cHostDevice]():
        dev_ctx.enqueue_copy[.int8](
            dev_buf.to_device_buffer(dev_ctx),
            host_buf.unsafe_ptr(),
        )
    else:
        raise Error(
            "mgp.buffer.host_to_device must be scheduled on an accelerator"
            " device"
        )


@register_internal("mgp.int.cache")
@no_inline
def mgp_int_cache[bIntSlot: UInt64](ctx: StateContext, value: Int):
    ctx.cache_int(Int(bIntSlot), value)


@register_internal("mgp.int.get_cached")
@no_inline
def mgp_int_get_cached(ctx: StateContext, buffer_slot: Int) -> Int:
    return ctx.get_cached_int(buffer_slot)


@register_internal("mgp.buffer.get_size")
@no_inline
def mgp_buffer_get_size(
    buf: OwnedByteBuffer,
) -> Int:
    return buf.size()


# ===-----------------------------------------------------------------------===#
# MGP Tensor Spec Primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.tensor_spec.create")
@no_inline
def mgp_tensor_spec_create[
    aRawDims: IntTuple,
    aRawDimsRank: Int,
](*runtimeDims: Int) -> IndexList[aRawDimsRank]:
    var shape = IndexList[aRawDimsRank]()
    var runtimeIndex = 0
    # Update Shape with runtime elements.
    # Negative values in aRawDims indicate dynamic dimensions.
    comptime for i in range(aRawDimsRank):
        if Int(aRawDims[i]) >= 0:
            shape[i] = Int(aRawDims[i])
        else:
            shape[i] = runtimeDims[runtimeIndex]
            runtimeIndex += 1
    return shape


@register_internal("mgp.tensor_spec.get_dim")
@no_inline
def mgp_tensor_spec_get_dim[
    spec_rank: Int, axis: UInt64
](spec: IndexList[spec_rank]) -> Int:
    comptime assert axis < UInt64(
        spec_rank
    ), "axis for get_dim must be less than rank of TensorSpec"
    return spec[Int(axis)]


# ===-----------------------------------------------------------------------===#
# MGP Device Context Primitives
# ===-----------------------------------------------------------------------===#


@export
def mgp_device_context_destroy(dev_ctx: DeviceContext) abi("Mojo"):
    # DeviceContext is refcounted, we don't need to explicitly destroy it
    pass


@register_internal("mgp.sync")
@no_inline
def mgp_sync(ctx: StateContext, dev_ctx: DeviceContext) raises:
    dev_ctx.synchronize()


@register_internal("mgp.device_wait")
@no_inline
def mgp_device_wait(
    ctx: StateContext,
    waiting_dev_ctx: DeviceContext,
    signaling_dev_ctx: DeviceContext,
) raises:
    # Enqueue a one-directional cross-stream dependency: the waiting context's
    # stream waits for the work already enqueued on the signaling context's
    # stream. Non-blocking on the host (unlike mgp.sync).
    waiting_dev_ctx.enqueue_wait_for(signaling_dev_ctx)


@register_internal("mgp.debug.print")
@no_inline
def mgp_debug_print[
    aDebugString: StaticString,
    bLabel: StaticString,
](ctx: StateContext) raises:
    var prefix = String()
    if bLabel:
        prefix = "[" + bLabel + "] "
    print(prefix + aDebugString)


@register_internal("mgp.debug.print.int")
@no_inline
def mgp_debug_print_int[
    aLabel: StaticString,
](ctx: StateContext, value: Int):
    var prefix = String()
    if aLabel:
        prefix = "[" + aLabel + "] "
    print(prefix + String(value))


@register_internal("mgp.debug.tensor.print")
@no_inline
def mgp_debug_tensor_print[
    spec_rank: Int,
    dtype: DType,
](
    buffer: OwnedByteBuffer,
    shape: IndexList[spec_rank],
    label_ptr: Pointer[mut=False, Byte, _],
    label_len: Int,
) raises:
    external_call["MGP_RT_DebugTensorPrint", NoneType](
        label_ptr,
        c_size_t(label_len),
        dtype,
        Pointer(to=shape.data),
        spec_rank,
        buffer.unsafe_ptr(),
        buffer.size(),
    )


# ===----------------------------------------------------------------------===#
# Additional expected primitives
# ===-----------------------------------------------------------------------===#


@always_inline
def get_simd_width_for_dtypes[
    dtypes: StaticTuple[DType, _], target: StaticString
]() -> Int:
    comptime assert dtypes.size > 0

    var width = get_kernel_simd_width[dtypes[0], target]()

    comptime for i in range(dtypes.size - 1):
        width = max(get_kernel_simd_width[dtypes[i + 1], target](), width)

    return width


# TODO: this should take IOSpec as a param -- will require graph compiler changes
# Used by the graph compiler to construct tensors from MGP repr. of tensor
@always_inline
def to_managed_tensor_slice[
    dtype: DType, rank: Int, mut: Bool, input: IO
](
    data: Pointer[Scalar[dtype], MutAnyOrigin],
    shape: Pointer[Int, ImmutAnyOrigin],
) -> ManagedTensorSlice[
    io_spec=IOSpec[mut, input](),
    static_spec=StaticTensorSpec[dtype, rank, ...].get_unknown(),
]:
    var shape_ptr = shape
    var shape_tuple = IndexList[rank]()

    var stride_tuple = IndexList[rank]()
    var stride: Int = 1

    comptime for i in reversed(range(rank)):
        # Start from the back so we can accumulate the strides.
        shape_tuple[i] = shape_ptr[unsafe_offset=i]
        stride_tuple[i] = stride
        stride *= shape_tuple[i]

    return {data, shape_tuple, stride_tuple}


# Extract a scalar from a managed tensor slice.
@always_inline
def _get_scalar_from_managed_tensor_slice[
    dtype: DType,
](tensor: ManagedTensorSlice[dtype=dtype, ...]) -> Scalar[dtype]:
    # Assumes that tensor is on the host!
    # This is used instead of [0] since __getitem__ for `ManagedTesnorSlice`
    # does not work with `register_internal` out of the box.
    return tensor.load[width=1](IndexList[1](0))


# ===-----------------------------------------------------------------------===#
# Opaque Test Primitives
# ===-----------------------------------------------------------------------===#


struct MyInt(Movable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __init__(out self, *, deinit move: MyInt):
        print("MyInt.__moveinit__", move.val)
        self.val = move.val

    def __deinit__(deinit self):
        print("MyInt.__deinit__", self.val)


@register_internal("testfuse.my_int.from_index")
@no_inline
def test_my_int_from_index(x: Int) -> MyInt:
    return MyInt(x)


@register_internal("testfuse.my_int.square")
@no_inline
def test_my_int_square(x: MyInt) -> MyInt:
    return MyInt(x.val * x.val)


@register_internal("testfuse.my_int.to_index")
@no_inline
def test_my_int_to_index(x: MyInt) -> Int:
    return x.val


struct MyIntReg2(ImplicitlyCopyable, RegisterPassable):
    var val: Int

    def __init__(out self, val: Int):
        self.val = val

    def __deinit__(deinit self):
        print("MyIntReg2.__deinit__", self.val)


@register_internal("testfuse.my_int_reg2.from_index")
@no_inline
def test_my_int_reg2_from_index(x: Int) -> MyIntReg2:
    return MyIntReg2(x)


@register_internal("testfuse.my_int_reg2.square")
@no_inline
def test_my_int_reg2_square(x: MyIntReg2) -> MyIntReg2:
    return MyIntReg2(x.val * x.val)


@register_internal("testfuse.my_int_reg2.to_index")
@no_inline
def test_my_int_reg2_to_index(x: MyIntReg2) -> Int:
    return x.val


# ===-----------------------------------------------------------------------===#
# Mojo generation hooks
# ===-----------------------------------------------------------------------===#

# ===-----------------------------------------------------------------------===#
# Mojo-C++ interop aliases
# ===-----------------------------------------------------------------------===#

# The purpose of these aliases is to make it easier to visually parse the
# interop. There is only one rule: Do not use types, always use OpaquePointer.
# This saves us from having to statically assert that a certain type has a
# specific byte size.

# AnyAsyncValueRef is a C++ struct. The runtime passes a reference to it.
# Therefore, we alias it to OpaquePointer which will have the same bitwidth as
# C++'s pointers.
comptime AnyAsyncValueRefPtr = OpaquePointer[MutAnyOrigin]


# Opaque stand-in for the C++ `M::MLRT::TensorBufferRef` type. Mojo never sees
# the layout of the C++ struct; this only gives the C pointer a distinct pointee
# type so it can't be confused with other opaque handles (e.g. the async-value
# storage handle), mirroring how `DeviceContext` uses `_DeviceContextCpp`.
struct _TensorBufferRefCpp:
    pass


# Typed C pointer to the C++ tensor-buffer ref. Primitives only ever manipulate
# a reference to it (never its layout); the default `UntrackedOrigin` marks the
# pointee as living outside the Mojo program (its lifetime is managed by the C++
# runtime), mirroring `_DeviceContextPtr`.
comptime TensorBufferRefPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_TensorBufferRefCpp, origin]


# Opaque stand-in for the C++ `M::MLRT::StateContext` type. Mojo never sees the
# layout of the C++ struct; this only gives the C pointer a distinct pointee
# type, mirroring how `DeviceContext` uses `_DeviceContextCpp`.
struct _StateContextCpp:
    pass


# Typed C pointer to the C++ state context. The default `UntrackedOrigin` marks
# the pointee as living outside the Mojo program (its lifetime is managed by the
# C++ runtime), mirroring `_DeviceContextPtr`.
comptime _StateContextPtr[
    mut: Bool,
    //,
    origin: Origin[mut=mut] = UntrackedOrigin[mut=mut],
] = OptionalPointer[_StateContextCpp, origin]

# `StateContextRef` is the pointer representation passed across the FFI boundary;
# Mojo never dereferences it directly.
comptime StateContextRef = _StateContextPtr[mut=True]


struct StateContext(ImplicitlyCopyable, RegisterPassable):
    """A Mojo handle to the C++ `M::MLRT::StateContext`.

    Wraps a C pointer to the C++ state context, mirroring how `DeviceContext`
    wraps a `_DeviceContextPtr`. Mojo never dereferences the pointer directly;
    all state-context operations are performed through external calls into the
    runtime.
    """

    var _handle: StateContextRef

    @always_inline
    def __init__(out self, handle: StateContextRef):
        """Builds the handle from the underlying C pointer.

        Args:
            handle: The C pointer to the C++ state context.
        """
        self._handle = handle

    @always_inline
    def cache_int(self, slot: Int, value: Int):
        """Caches an integer value in the state slot at the given index.

        Args:
            slot: The index of the state slot to write.
            value: The integer value to cache.
        """
        external_call["MGP_RT_SetCachedInt", NoneType](
            slot, self._handle, value
        )

    @always_inline
    def get_cached_int(self, slot: Int) -> Int:
        """Returns the integer value cached in the state slot at the given index.

        Args:
            slot: The index of the state slot to read.

        Returns:
            The cached integer value.
        """
        return external_call["MGP_RT_GetCachedInt", Int](slot, self._handle)

    @always_inline
    def get_cached_buffer(
        self, slot: Int
    ) -> Tuple[MutByteBuffer, AnyAsyncValueRefPtr]:
        """Returns a reference to the buffer cached in the given state slot.

        Args:
            slot: The index of the state slot to read.

        Returns:
            A tuple of the buffer view and the backing storage handle of the
            cached `TensorBufferRef` (its `AnyAsyncValueRef` memory handle, not
            the `TensorBufferRef` itself).
        """
        var buffer_size: UInt64 = 0
        var buffer_data = Optional[OpaquePointer[MutAnyOrigin]]()

        var mem_handle = external_call[
            "MGP_RT_GetCachedBuffer", AnyAsyncValueRefPtr
        ](
            slot,
            self._handle,
            Pointer(to=buffer_size),
            Pointer(to=buffer_data),
        )

        var buffer = MutByteBuffer(
            buffer_data.unsafe_value().unsafe_bitcast[Int8](),
            Index(buffer_size),
        )

        return {buffer, mem_handle}

    @always_inline
    def get_device_graph_cache(
        self,
    ) -> ref[MutUntrackedOrigin] DeviceGraphCache:
        """Returns the device graph cache, creating it on first use.

        Unlike the state slots, the cache is not indexed: there is a single
        device graph cache per model. The state context holds it and outlives
        every execution, so the returned reference is untracked.

        Returns:
            A mutable reference to the model's device graph cache.
        """

        # Allocating and initializing in one step keeps creation indivisible, so
        # the state context can never publish storage that is not yet a usable
        # `DeviceGraphCache`.
        def create() -> Pointer[DeviceGraphCache, MutUntrackedOrigin]:
            # Leaked because the state context takes over ownership; `destroy`
            # below pairs the pointer back up with this layout to release it.
            var cache = OwnedPointer(value=DeviceGraphCache())
            return cache^.unsafe_take_allocation().unsafe_leak()

        # The corresponding desructor function. Convert the raw pointer back to
        # an OwnedPointer for destruction.
        def destroy(cache: Pointer[DeviceGraphCache, MutUntrackedOrigin]):
            _ = OwnedPointer(unsafe_from_raw_pointer=cache)

        return external_call[
            "MGP_RT_GetOrCreateDeviceGraphCache",
            Pointer[DeviceGraphCache, MutUntrackedOrigin],
        ](self._handle, create, destroy)[]


# ===-----------------------------------------------------------------------===#
# MOGG primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mogg.as_scalar")
@always_inline
def mogg_as_scalar(tensor: ManagedTensorSlice) -> Scalar[tensor.dtype]:
    return _get_scalar_from_managed_tensor_slice(tensor)


@register_internal("mogg.async.__del__")
@no_inline
def mogg_async_del(
    async_ptr: Pointer[AnyAsyncValueRefPtr, MutAnyOrigin], size: Int
):
    """
    Decrement the AnyAsyncValueRef. Typically called at the end of a kernel for
    all input and output operands.
    """
    external_call["MGP_RT_DestructAsyncRefs", NoneType](size, async_ptr, False)


@register_internal("mogg.async.unpack")
@no_inline
def mogg_async_unpack[
    T: TrivialRegisterPassable
](async_ptr: AnyAsyncValueRefPtr) -> T:
    """
    Returns the value stored in the AnyAsyncValueRef.
    """
    var ptr = external_call[
        "MGP_RT_GetValueFromAsync", OpaquePointer[MutAnyOrigin]
    ](async_ptr).unsafe_bitcast[T]()

    return ptr[]


struct MoggAsyncPackHelper:
    """
    Helper struct for packing various data types into an asynchronous context
    for MOGG operations. Provides constructor overloads for different supported
    types.
    """

    def __init__(out self, data: Int, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an integer value into the asynchronous context.
        Calls create_index_async to handle the packing.
        """
        create_index_async(data, async_ptr)

    def __init__(out self, data: Int64, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a 64-bit integer value into the asynchronous context.
        Calls create_si64_async to handle the packing.
        """
        create_si64_async(data, async_ptr)

    def __init__(out self, data: Bool, async_ptr: AnyAsyncValueRefPtr):
        """
        Packs a boolean value into the asynchronous context.
        Calls create_i1_async to handle the packing.
        """
        create_i1_async(data, async_ptr)

    def __init__[
        spec_rank: Int
    ](out self, data: IndexList[spec_rank], async_ptr: AnyAsyncValueRefPtr):
        """
        Packs an IndexList of specified rank into the asynchronous context.
        Calls create_tensor_spec_async to handle the packing.
        """
        create_tensor_spec_async(data, async_ptr)

    def __init__(
        out self,
        var data: OwnedByteBuffer,
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs an OwnedByteBuffer into a real TensorBufferRef. The storage handle
        is copied (retained) rather than moved out, so the composite may be
        borrowed -- including from an `Array` element (e.g. bulk_slice),
        which cannot be moved out of. The runtime adopts the copied reference
        net-zero; the borrowed composite releases its own reference at scope end.
        """
        var ptr = data.unsafe_ptr()
        var n = data.size()
        var storage = data^.take_storage()
        # void MGP_RT_CreateAsyncBufferRefFromStorage(
        #     AsyncValue *storage, void *data, size_t size, AnyAsyncValueRef *async)
        external_call["MGP_RT_CreateAsyncBufferRefFromStorage", NoneType](
            storage^.take_handle(), ptr, n, async_ptr
        )

    def __init__(
        out self,
        var data: DeviceGraph,
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """Packs a `DeviceGraph` into an `AsyncValue[DeviceGraphRef]`.

        The graph handle is surrendered net-zero (`take_handle`) and adopted by
        the runtime, so no extra reference is created. Used to pack the graph
        produced by `mgp.device_graph.create` so that `mgp.device_graph.execute`
        can consume it as a first-class device-graph reference rather than an
        opaque Mojo value.
        """
        # void MGP_RT_CreateAsyncDeviceGraphRefByTakingHandle(
        #     DeviceGraph *handle, AnyAsyncValueRef *async)
        external_call[
            "MGP_RT_CreateAsyncDeviceGraphRefByTakingHandle", NoneType
        ](data^.take_handle(), async_ptr)

    def __init__(
        out self,
        var data: Some[Movable & Deinitable],
        async_ptr: AnyAsyncValueRefPtr,
    ):
        """
        Packs a generic Movable value into the asynchronous context.
        Used for opaque types like SIMDPair.
        """
        comptime Type = type_of(data)

        # MGP_RT_CreateOwnedAsyncMojoValue expects a type erased destructor
        @always_inline("nodebug")
        def erased_destructor(ptr: Pointer[UInt8, MutUntrackedOrigin]):
            ptr.unsafe_bitcast[Type]().unsafe_deinit_pointee()

        var dst_ptr = external_call[
            "MGP_RT_MojoValueAllocateBuffer",
            Pointer[UInt8, MutUntrackedOrigin],
        ](size_of[Type](), align_of[Type]())

        dst_ptr.unsafe_bitcast[Type]().unsafe_write(data^)

        external_call["MGP_RT_CreateOwnedAsyncMojoValue", NoneType](
            dst_ptr,
            erased_destructor,
            async_ptr,
        )


@no_inline
def mogg_async_pack_owned_tensor[
    spec_rank: Int,
](var data: OwnedTensor, async_ptr: AnyAsyncValueRefPtr):
    """Packs an `OwnedTensor` into a real tensor `TensorBufferRef`.

    This is a dedicated (non-overloaded) entry point rather than a
    `MoggAsyncPackHelper` constructor: the parametric `OwnedTensor` overload
    would lose overload resolution to the generic `Some[Movable &
    Deinitable]` constructor and get mis-packed as an opaque Mojo
    value. The emitter calls this directly for `!mgp.tensor` pack sites.

    The storage handle is copied (retained) rather than moved out, so the
    composite may be borrowed; the runtime adopts the copied reference net-zero
    and the borrowed composite releases its own reference at scope end.

    Parameters:
        spec_rank: The true tensor-spec rank (0 for a scalar), supplied by the
            emitter so the packed `TensorSpec` preserves scalar-ness rather than
            the promoted rank-1 buffer view.
    """
    # Read the view metadata (shape/ptr/size).
    var shape = data.shape()
    var ptr = data.unsafe_ptr()
    var n = data.bytecount()

    # Transfer storage ownership to the newly constructed TensorBufferRef async
    # value.
    var storage = data^.take_storage()
    # void MGP_RT_CreateAsyncTensorRefFromStorage(
    #     AsyncValue *storage, void *data, size_t size, size_t rank,
    #     const size_t *shape, DType dtype, AnyAsyncValueRef *async)
    external_call["MGP_RT_CreateAsyncTensorRefFromStorage", NoneType](
        storage^.take_handle(),
        ptr.unsafe_bitcast[NoneType](),
        n,
        spec_rank,
        Pointer(to=shape.data),
        data.dtype,
        async_ptr,
    )


@register_internal("mogg.async.pack")
@no_inline
def mogg_async_pack(pack_helper: MoggAsyncPackHelper):
    """
    Packs asynchronous data using the provided MoggAsyncPackHelper.

    This function serves as an entry point for packing data into an asynchronous
    reference. The actual packing logic is handled by the MoggAsyncPackHelper struct,
    which provides specialized constructors for different data types. This function
    itself is a no-op and exists to satisfy the internal registration mechanism.
    """
    return


@register_internal("mogg.tensor.__init__")
@always_inline
def mogg_tensor_init[
    LayoutType: TensorLayout,
    //,
    dtype: DType,
    rank: Int,
    mut: Bool,
    input: IO,
    alignment: Int,
](
    ptr: Pointer[mut=True, NoneType, _],
    layout: LayoutType,
) -> ManagedTensorSlice[
    io_spec=IOSpec[mut, input](),
    static_spec=StaticTensorSpec[
        dtype,
        rank,
        static_layout=LayoutType,
    ](alignment, .GENERIC),
]:
    """
    Helper for constructing a ManagedTensorSlice from a layout.
    """
    return {
        ptr.unsafe_bitcast[Scalar[dtype]](),
        layout.shape_coord(),
        layout.stride_coord(),
    }


@register_internal("mogg.async.ready")
@no_inline
def mogg_async_ready(async_ptr: AnyAsyncValueRefPtr):
    """
    Marks the chain as ready.
    """
    external_call["MGP_RT_CreateAsync_chain", NoneType](async_ptr)


@register_internal("mogg.async.join")
@no_inline
def mogg_async_check_task_error(mut error: Optional[Error]) raises:
    """Raises the captured error from an async task, if present.

    Raises:
        If an error was captured from the async task.
    """
    if error:
        raise error.take()


@register_internal("mogg.async.error")
@no_inline
def mogg_async_error(
    async_ptr: AnyAsyncValueRefPtr,
    err: Error,
    source_notes: String = "",
):
    """Indicates to the C++ runtime that the kernel has failed.

    When source_notes is non-empty it is prepended to the error message.
    The "Source Traceback:" header is included by the compiler only when
    actual Python tracebacks are present (see buildNotesString in MOGGOps.cpp).
    See GEX-2678.
    """
    var error_message = String(err)
    if source_notes:
        error_message = "\n" + source_notes + "\n\n" + error_message
    external_call["MGP_RT_AsyncRT_CreateAsync_Error", NoneType](
        async_ptr,
        error_message.as_c_string_slice(),
        error_message.byte_length(),
    )


@register_internal("mogg.raise")
@no_inline
def mogg_format_kernel_error(
    kernel_name: String,
    error: Error,
    fusion_info: String = "",
    traceback: String = "",
) -> Error:
    """Format a kernel error with context (name, fusion info, source traceback).

    Called from MOGG ABI stub except handlers. The formatted error is re-raised
    and eventually caught by the outer MGP region's except handler.
    """
    var msg = (
        String('An error occurred in kernel named "')
        + kernel_name
        + '":\n'
        + String(error)
    )
    if fusion_info:
        msg += "\n\nFusion info:\n" + fusion_info
    if traceback:
        msg += "\n\nSource Traceback:\n" + traceback
    return Error(msg)


@register_internal("mogg.format_region_error")
@no_inline
def mogg_format_region_error(
    region_name: String,
    error: Error,
) -> Error:
    """Format a region error with the entry point name prefix.

    Called from MGP ABI stub except handlers after catching a kernel error.
    """
    return Error(
        String('An error occurred in kernel entry point named "')
        + region_name
        + '":\n'
        + String(error)
    )


@register_internal("mogg.tensor.reshape")
@always_inline
def reshape_contiguous_buffer[
    static_layout: TensorLayout, new_rank: Int
](
    buffer: ManagedTensorSlice,
    shape: IndexList[new_rank],
) -> ManagedTensorSlice[
    io_spec=buffer.io_spec,
    static_spec=StaticTensorSpec[
        buffer.dtype,
        new_rank,
        static_layout=static_layout,
    ](1, .GENERIC),
]:
    """
    Constructs a new ManagedTensorSlice with a new shape and static spec.
    """
    return {buffer._ptr, shape}


# ===-----------------------------------------------------------------------===#
# MGP primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mgp.buffer.get_cached")
@no_inline
def mgp_buffer_get_cached(
    ctx: StateContext,
    buffer_slot: Int,
) -> OwnedByteBuffer:
    """
    Get a reference to the cached buffer, retaining its backing storage.
    """
    var cached = ctx.get_cached_buffer(buffer_slot)
    # cached is (view, mem_handle); fold the cached buffer's memory handle into
    # the composite's storage by retaining it.
    return OwnedByteBuffer(cached[0], AnyAsyncValueRef(retain_handle=cached[1]))


@register_internal("mgp.assert")
@no_inline
def mgp_assert(
    cond: Bool, msg_ptr: Pointer[mut=False, Byte, _], msg_len: Int
) raises:
    """
    Raises an error when the input condition is not true.
    """
    if not cond:
        raise Error(pack_string_res(msg_ptr, msg_len))


def all_zeros(indices: IndexList) -> Bool:
    comptime for i in range(indices.size):
        if indices[i] != 0:
            return False
    return True


# ===----------------------------------------------------------------------===#
# Affine view kernels
# ===----------------------------------------------------------------------===#


@register_internal("mo.split_dim")
@always_inline
def split_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], new_shape_dim: Int) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [40, 30, 2] and we reshape it to [5, 8, 30, 2].
    # If we are accessing the index [21, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [2, 5, 16, 1].
    # Or [21 // 8, 21 % 8, ...old dims...].
    # In this case, the axis = 0 and the new_shape_dim = 8.

    comptime for i in range(rank + 1):
        comptime if i == axis:
            out[i] = indices[axis] // new_shape_dim
        elif i == axis + 1:
            out[i] = indices[axis] % new_shape_dim
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]

    return out


@register_internal("mo.merge_dim")
@always_inline
def merge_dim_indices[
    rank: Int, axis: Int
](indices: IndexList[rank], old_shape_dim: Int) -> IndexList[rank - 1]:
    var out = IndexList[rank - 1]()

    # This op is transforming the INDICES of an access into a reshaped tensor.
    # Consider the tensor is [5, 8, 30, 2] and we reshape it to [40, 30, 2].
    # If we are accessing the index [2, 5, 16, 1] in the original shape then to
    # preserve the reshape we would need to transform the indices into [21, 16, 1].
    # Or [2 * 8 + 5, 16, 1].
    # In this case, the axis = 0 and the old_shape_dim = 8.

    comptime for i in range(rank - 1):
        comptime if i == axis:
            out[i] = fma(indices[i], old_shape_dim, indices[i + 1])
        elif i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i + 1]

    return out


@register_internal("mo.add_singleton_dim")
@always_inline
def insert_index[
    rank: Int, axis: Int, value: Int
](indices: IndexList[rank]) -> IndexList[rank + 1]:
    var out = IndexList[rank + 1]()

    comptime for i in range(rank + 1):
        comptime if i < axis:
            out[i] = indices[i]
        elif i > axis:
            out[i] = indices[i - 1]
        else:
            out[i] = value

    return out


# ===----------------------------------------------------------------------===#
# POP operations
# ===----------------------------------------------------------------------===#


@register_internal("pop.select")
@always_inline
def select[
    T: TrivialRegisterPassable
](cond: Bool, true_case: T, false_case: T) -> T:
    if cond:
        return true_case

    return false_case


@register_internal("pop.simd.select")
@always_inline
def simd_select[
    T: TrivialRegisterPassable
](cond: Bool, true_case: T, false_case: T) -> T:
    return select(cond, true_case, false_case)


# ===-----------------------------------------------------------------------===#
# MOGG elementwise / view primitives
# ===-----------------------------------------------------------------------===#


@register_internal("mogg.elemwise_for_each")
@no_inline
def foreach[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int, element_alignment: Int](
        IndexList[rank]
    ) capturing -> SIMD[dtype, width],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.

    Args:
        tensor: The output tensor slice which receives the return values from `func`.
        ctx: The call context (forward this from the custom operation).
    """

    @always_inline
    def elementwise_fn_wrapper[
        width: Int,
        alignment: Int = 1,
    ](index: Coord) {var}:
        var idx = coord_to_index_list(index)
        var val = func[width, alignment](rebind[IndexList[tensor.rank]](idx))
        tensor._fused_store[element_alignment=alignment](index, val)

    elementwise[
        simd_width,
        target=target,
        _trace_description=_trace_name,
    ](elementwise_fn_wrapper, tensor.shape_coord(), ctx)


@register_internal("mogg.elemwise_for_each")
@no_inline
def foreach[
    dtype: DType,
    rank: Int,
    //,
    FuncType: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, element_alignment: Int](IndexList[rank]) -> SIMD[
        dtype, width
    ],
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    var func: FuncType,
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Apply a `RegisterPassable` body to each element of the tensor slice.

    Value-arg twin of the primary `foreach`: the body is a runtime
    `RegisterPassable` closure passed by value rather than a comptime
    parameter. The wrapper captures both `func` and `tensor` and routes
    the store through `tensor._fused_store`, which preserves the fused
    store path required by `FusedOutputTensor`.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        FuncType: The type of the per-element body closure.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target.
        _trace_name: Name of the executed operation displayed in the trace.

    Args:
        func: The body to apply per element.
        tensor: The output tensor slice which receives the return values.
        ctx: The call context (forward this from the custom operation).
    """

    # The wrapper captures `func` (passed by value) and `tensor` so the
    # store can route through `_fused_store` when output-store fusion is
    # present.
    def wrapper[
        width: Int, alignment: Int = 1
    ](index: Coord) {var func^, var tensor}:
        var idx = rebind[IndexList[rank]](coord_to_index_list(index))
        var val = func[width, alignment](idx)
        tensor._fused_store[element_alignment=alignment](index, val)

    elementwise[
        simd_width=simd_width,
        target=target,
        _trace_description=_trace_name,
    ](wrapper, tensor.shape_coord(), ctx)


@fieldwise_init
struct _ElementwiseFusionAdapter[
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    ComputeFusionTile: ComputeOutputFusionTile,
    OutFusionTile: OutputFusionTile,
    io_spec: IOSpec[True, _],
    static_spec: StaticTensorSpec[
        dtype,
        rank,
        _,
        InFusion,
        OutFusion,
        ComputeFusion,
        ComputeFusionTile,
        OutFusionTile,
    ],
    //,
    E: ElementwiseFusion,
](
    ImplicitlyCopyable,
    RegisterPassable,
    def[width: Int, alignment: Int = 1](Coord) -> None,
):
    """Per-element body for `foreach_fusion`, holding the fusion struct and
    output tensor by value.

    Named adapter twin of a `{var elem, var tensor}` closure: a closure over
    the generic `E` synthesizes a parametric-witness `lit.closure.init` the
    MOGG package loader can't resolve, so the body is a concrete
    register-passable struct instead. Passing the instance by value to
    `elementwise` carries `elem` (and the tensor's ptr/shape/strides) through
    `crossDeviceCaptures` by value.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The rank of the tensor.
        InFusion: The tensor's input-fusion type.
        OutFusion: The tensor's output-fusion type.
        ComputeFusion: The tensor's compute-output-fusion type.
        ComputeFusionTile: The tensor's compute-output-fusion-tile type.
        OutFusionTile: The tensor's output-fusion-tile (store) type.
        io_spec: The tensor's IO spec.
        static_spec: The tensor's static spec.
        E: The elementwise fusion struct type.
    """

    var elem: Self.E
    var tensor: ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]

    @always_inline
    def __call__[width: Int, alignment: Int = 1](self, index: Coord):
        var idx = rebind[IndexList[Self.rank]](coord_to_index_list(index))
        var val = self.elem.compute[Self.dtype, Self.rank, width, alignment](
            idx
        )
        self.tensor._fused_store[element_alignment=alignment](idx, val)


@register_internal("mogg.call.foreach")
@no_inline
def foreach_fusion[
    dtype: DType,
    rank: Int,
    //,
    E: ElementwiseFusion,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    var elem: E,
    ctx: DeviceContext,
) raises:
    """Apply a pure elementwise fusion to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        E: The elementwise fusion struct type.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target.
        _trace_name: Name of the executed operation displayed in the trace.

    Args:
        tensor: The output tensor slice which receives the computed values.
        elem: The elementwise fusion struct.
        ctx: The call context (forward this from the custom operation).
    """

    # Capture `elem` by value through a named adapter struct rather than a
    # closure. A `{var elem}` closure over the generic `E` synthesizes a
    # `lit.closure.init` with parametric witnesses the package loader can't
    # resolve (see functional.mojo `_IndexListToCoordAdapter`); the adapter is
    # a concrete register-passable type, so passing it by value to
    # `elementwise` sends `elem`'s decomposed ptr/shape/strides through
    # `crossDeviceCaptures` by value — which the host-stack `@__parameter
    # capturing` form did not.
    var adapter = _ElementwiseFusionAdapter[E](elem, tensor)

    elementwise[
        simd_width=simd_width,
        target=target,
        _trace_description=_trace_name,
    ](adapter, Coord(tensor.shape()), ctx)


@register_internal("mogg._call.foreach")
def _foreach[
    Lambda: ImplicitlyCopyable
    & RegisterPassable
    & def[width: Int, alignment: Int = 1](Coord) -> None,
    //,
    *,
    simd_width: Int,
    target: StaticString = "cpu",
    _trace_name: StaticString = "mogg.foreach",
](shape: Coord, lambda_: Lambda, ctx: DeviceContext,) raises:
    """Call `lambda_` once per index in `shape`; `lambda_` does its own store.

    Unlike `foreach_fusion`, this forwards straight to `elementwise[...]`
    with no intermediate adapter: the closure already captures its own
    tensors and does its own load-compute-store round trip.

    Parameters:
        Lambda: The per-index closure's concrete type.
        simd_width: The SIMD width to use.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        _trace_name: Name of the executed operation displayed in the trace.

    Args:
        shape: The iteration domain; `lambda_` is called once per index.
        lambda_: The per-index closure, capturing its own input/output
            tensors.
        ctx: The call context (forward this from the custom operation).
    """
    elementwise[
        simd_width=simd_width,
        target=target,
        _trace_description=_trace_name,
    ](lambda_, shape, ctx)


comptime _MoggSliceStrideTypesTabulator[
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    input_stride_types[idx].static_value * step_types[idx].static_value
] if input_stride_types[
    idx
].is_static_value and step_types[
    idx
].is_static_value else Scalar[
    DType.int
]

comptime _MoggSliceStrideTypes[
    rank: Int,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    step_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank, _MoggSliceStrideTypesTabulator[input_stride_types, step_types, _]
]()


def _mogg_slice_view_alignment[
    rank: Int,
    dtype: DType,
    input_strides: IntTuple,
    static_starts: IntTuple,
    static_steps: IntTuple,
](input_alignment: Int) -> Int:
    """Same alignment derivation as `Slice.get_view_alignment` (mirrored here,
    not called directly, since `builtin_kernels` already depends on
    `builtin_primitives` and importing the other way would be circular).
    """
    comptime stride_types = _IntTupleToCoordLike[DType.int, input_strides]
    comptime start_types = _IntTupleToCoordLike[DType.int, static_starts]
    comptime step_types = _IntTupleToCoordLike[DType.int, static_steps]

    var alignment = input_alignment
    comptime for i in range(rank):
        comptime if not step_types[i].is_static_value or step_types[
            i
        ].static_value < 0:
            return 1
        comptime if not start_types[i].is_static_value:
            return 1
        comptime if start_types[i].static_value != 0:
            comptime if not stride_types[i].is_static_value:
                return 1
            alignment = gcd(
                alignment,
                start_types[i].static_value
                * stride_types[i].static_value
                * align_of[dtype](),
            )
    return alignment


@register_internal("mogg._tensor.create.slice")
def mogg_tensor_create_slice[
    dtype: DType,
    rank: Int,
    start_type: DType,
    stop_type: DType,
    step_type: DType,
    //,
    output_static_shape: IntTuple,
    static_starts: IntTuple,
    static_steps: IntTuple,
](
    input: ManagedTensorSlice[dtype=dtype, rank=rank, ...],
    starts: ManagedTensorSlice[dtype=start_type, rank=1, ...],
    stops: ManagedTensorSlice[dtype=stop_type, rank=1, ...],
    steps: ManagedTensorSlice[dtype=step_type, rank=1, ...],
) -> ManagedTensorSlice[
    io_spec=input.io_spec,
    static_spec=input.static_spec.with_tile_layout_and_alignment[
        rank,
        TileLayout[
            shape_types=_IntTupleToCoordLike[DType.int, output_static_shape],
            stride_types=_MoggSliceStrideTypes[
                rank,
                input.static_spec.static_layout._stride_types,
                _IntTupleToCoordLike[DType.int, static_steps],
            ],
        ],
    ](
        _mogg_slice_view_alignment[
            rank,
            dtype,
            input._static_strides_tuple,
            static_starts,
            static_steps,
        ](input.alignment),
    ),
]:
    """Backing primitive for `mogg._tensor.create.slice`: a zero-copy strided
    view of `input`, offset and re-strided per `starts`/`stops`/`steps`.
    `output_static_shape`/`static_starts`/`static_steps` carry whatever is
    known about the view's shape and bounds at compile time (an entry is
    `UNKNOWN_VALUE` where it isn't), letting the result keep a static layout
    and alignment instead of falling back to a fully dynamic one.

    Parameters:
        dtype: The element type of `input`.
        rank: The rank of `input` (and of the returned view).
        start_type: The element type of `starts`.
        stop_type: The element type of `stops`.
        step_type: The element type of `steps`.
        output_static_shape: The view's shape, where known at compile time.
        static_starts: The view's starting indices, where known at compile
            time.
        static_steps: The view's strides, where known at compile time.

    Args:
        input: The tensor to slice.
        starts: One-dimensional tensor of starting indices, one per rank.
        stops: One-dimensional tensor of stopping indices (exclusive), one
            per rank.
        steps: One-dimensional tensor of strides, one per rank.
    """
    var view = slice_as_view(
        input.to_tile_tensor[DType.int64](),
        starts.to_tile_tensor[DType.int64](),
        stops.to_tile_tensor[DType.int64](),
        steps.to_tile_tensor[DType.int64](),
    )
    return {
        view._storage,
        rebind[IndexList[rank]](coord_to_index_list(view.layout.shape_coord())),
        rebind[IndexList[rank]](
            coord_to_index_list(view.layout.stride_coord())
        ),
    }


def _mogg_broadcast_view_strides[
    out_rank: Int,
    in_rank: Int,
    input_shape: IntTuple,
    input_strides: IntTuple,
]() -> IndexList[out_rank]:
    """Same view-stride derivation as `StaticBroadcastTo.get_view_strides_list`
    (mirrored here, not called directly, since `builtin_kernels` already
    depends on `builtin_primitives` and importing the other way would be
    circular): `UNKNOWN_VALUE` marks a dimension whose stride isn't known at
    compile time, so `with_int_tuple_layout` falls back to a dynamic entry
    for it.
    """
    var new_strides = IndexList[out_rank]()
    comptime delta = out_rank - in_rank
    comptime for i in range(out_rank):
        comptime if i < delta:
            new_strides[i] = 0
        else:
            if Int(input_shape[i - delta]) == UNKNOWN_VALUE:
                new_strides[i] = UNKNOWN_VALUE
            elif Int(input_shape[i - delta]) <= 1:
                new_strides[i] = 0
            else:
                new_strides[i] = Int(input_strides[i - delta])
    return new_strides


@register_internal("mogg._tensor.create.broadcast")
def mogg_tensor_create_broadcast[
    dtype: DType,
    in_rank: Int,
    //,
    output_static_shape: IntTuple,
](
    input: ManagedTensorSlice[dtype=dtype, rank=in_rank, ...],
) -> ManagedTensorSlice[
    io_spec=input.io_spec,
    static_spec=input.static_spec.with_int_tuple_layout[
        len(output_static_shape),
        output_static_shape,
        _mogg_broadcast_view_strides[
            len(output_static_shape),
            in_rank,
            input._static_shape_tuple,
            input._static_strides_tuple,
        ](),
    ](),
]:
    """Backing primitive for `mogg._tensor.create.broadcast`: a zero-copy
    reindexed view of `input` that replicates it (numpy broadcasting
    semantics) up to `output_static_shape` -- a leading dimension `input`
    doesn't have, or an aligned dimension where `input`'s size is 1, gets
    stride 0. Preserves whatever static shape/stride information is known
    at compile time, mirroring
    `StaticBroadcastTo.update_input_view`/`get_view_strides_list`.

    Parameters:
        dtype: The element type of `input`.
        in_rank: The rank of `input`.
        output_static_shape: The view's shape (broadcast shapes are always
            statically known by Mojo emission time; see
            `mo.static.broadcast_to`'s own docstring).

    Args:
        input: The tensor to broadcast.
    """
    comptime out_rank = len(output_static_shape)
    comptime delta = out_rank - in_rank
    var new_shape = IndexList[out_rank]()
    var new_strides = IndexList[out_rank]()
    comptime for i in range(out_rank):
        new_shape[i] = Int(output_static_shape[i])
        comptime if i < delta:
            new_strides[i] = 0
        else:
            if input.dim_size[i - delta]() <= 1:
                new_strides[i] = 0
            else:
                new_strides[i] = input.stride_length[i - delta]()
    return {input.unsafe_ptr(), new_shape, new_strides}


@register_internal("mogg._tensor.create.reshape")
def mogg_tensor_create_reshape[
    dtype: DType,
    in_rank: Int,
    out_rank: Int,
    //,
    output_static_shape: IntTuple,
](
    input: ManagedTensorSlice[dtype=dtype, rank=in_rank, ...],
    output_shape: IndexList[out_rank],
) -> ManagedTensorSlice[
    io_spec=input.io_spec,
    static_spec=input.static_spec.with_row_major_int_tuple_layout[
        out_rank,
        output_static_shape,
    ](),
]:
    """Backing primitive for `mogg._tensor.create.reshape`: a zero-copy view
    of `input` reinterpreted under `output_shape` -- valid only because
    `input` is contiguous, so de-linearizing/re-linearizing through one
    shared flat index produces row-major strides for the new shape.
    `output_static_shape` still types the return value's static layout with
    whatever dims ARE known at compile time (mirroring
    `StaticReshape.update_input_view`), but `output_shape` -- not a cast of
    `output_static_shape` -- is what this op actually reshapes into: unlike
    broadcast/transpose, this op's own stride is a row-major
    de-linearization of the *output* shape's own values, with no input
    stride to fall back on, so any dim not statically known has to be an
    honest runtime value here or every row past the first comes out wrong.

    `out_rank` is inferred from `output_shape` itself, not from
    `len(output_static_shape)` -- both name the same rank, but a
    comptime-parameter argument type can't depend on another comptime
    parameter's own derived value at inference time the way it can on a
    plain `Int`.

    Parameters:
        dtype: The element type of `input`.
        in_rank: The rank of `input`.
        out_rank: The rank of the reshaped view (and of `output_static_shape`).
        output_static_shape: The view's shape, where known at compile time
            (a `dyn` placeholder elsewhere); see the op's own MOGG tablegen
            doc for why a dynamic dim can't just be read off this alone.

    Args:
        input: The tensor to reshape.
        output_shape: The view's shape at runtime -- authoritative for every
            dim, not just the ones `output_static_shape` doesn't know.
    """
    comptime assert out_rank == len(
        output_static_shape
    ), "out_rank must match output_static_shape's own rank"
    var view = reshape(input.to_tile_tensor[DType.int64](), output_shape)
    return {
        view._storage,
        rebind[IndexList[out_rank]](
            coord_to_index_list(view.layout.shape_coord())
        ),
        rebind[IndexList[out_rank]](
            coord_to_index_list(view.layout.stride_coord())
        ),
    }


comptime _MoggTransposeStrideTypesTabulator[
    permutations: IntTuple,
    input_stride_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = Scalar[DType.int] if Int(
    permutations[idx]
) == UNKNOWN_VALUE else input_stride_types[
    Int(permutations[idx])
]


comptime _MoggTransposeStrideTypes[
    rank: Int,
    permutations: IntTuple,
    input_stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    rank,
    _MoggTransposeStrideTypesTabulator[permutations, input_stride_types, _],
]()


@register_internal("mogg._tensor.create.transpose")
def mogg_tensor_create_transpose[
    dtype: DType,
    rank: Int,
    //,
    output_static_shape: IntTuple,
    static_permutations: IntTuple,
](
    input: ManagedTensorSlice[dtype=dtype, rank=rank, ...],
) -> ManagedTensorSlice[
    io_spec=input.io_spec,
    static_spec=input.static_spec.with_tile_layout[
        rank,
        TileLayout[
            shape_types=_IntTupleToCoordLike[DType.int, output_static_shape],
            stride_types=_MoggTransposeStrideTypes[
                rank,
                static_permutations,
                input.static_spec.static_layout._stride_types,
            ],
        ],
    ](),
]:
    """Backing primitive for `mogg._tensor.create.transpose`: a zero-copy
    reindexed view of `input` that reorders its dimensions according to
    `static_permutations` -- the view's dimension `i` comes from `input`'s
    dimension `static_permutations[i]`. Preserves whatever static
    shape/stride information is known at compile time, mirroring
    `Transpose.update_input_view`/`_TransposeStrideTypes`.

    Parameters:
        dtype: The element type of `input`.
        rank: The rank of `input` (and of the returned view -- transpose
            never changes rank).
        output_static_shape: The view's shape.
        static_permutations: The permutation applied to `input`'s
            dimensions to produce the view (transpose's `perm` is always a
            compile-time constant by the time this primitive is called; see
            MOToMAP's lowering of `mo.transpose`).

    Args:
        input: The tensor to transpose.
    """
    var new_shape = IndexList[rank]()
    var new_strides = IndexList[rank]()
    comptime for i in range(rank):
        new_shape[i] = Int(output_static_shape[i])
        new_strides[i] = input.stride_length[Int(static_permutations[i])]()
    return {input.unsafe_ptr(), new_shape, new_strides}


@fieldwise_init
struct _ElementwiseFusionTileAdapter[
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    ComputeFusionTile: ComputeOutputFusionTile,
    OutFusionTile: OutputFusionTile,
    io_spec: IOSpec[True, _],
    static_spec: StaticTensorSpec[
        dtype,
        rank,
        _,
        InFusion,
        OutFusion,
        ComputeFusion,
        ComputeFusionTile,
        OutFusionTile,
    ],
    //,
    E: ElementwiseFusionTile,
    tile_shape: IndexList[2],
](ImplicitlyCopyable, RegisterPassable, def() -> None):
    """Per-tile body for `foreach_fusion_tile`, holding the fusion struct and
    output tensor by value.

    Analogous to `_ElementwiseFusionAdapter`, but for tile-based fusion: a
    named, register-passable struct (not a closure) so `elem` and the tensor's
    decomposed ptr/shape/strides cross into the GPU kernel by value. Its
    `__call__` drives one output *tile*, handing the fusion struct a load copier
    (used by `compute` to pull its inputs into `Copier.dst_address_space`) and
    storing the manufactured result tile via a store copier.

    Parameters:
        dtype: The data type of the tensor elements.
        rank: The rank of the tensor.
        InFusion: The tensor's input-fusion type.
        OutFusion: The tensor's output-fusion type.
        ComputeFusion: The tensor's compute-output-fusion type.
        ComputeFusionTile: The tensor's compute-output-fusion-tile type.
        OutFusionTile: The tensor's output-fusion-tile (store) type.
        io_spec: The tensor's IO spec.
        static_spec: The tensor's static spec.
        E: The tile elementwise fusion struct type.
        tile_shape: The `(rows, cols)` shape of one output tile.
    """

    # SM100 (B200): one thread-block processes one output tile. `thread_layout`
    # equals `tile_shape`, so each of the `tile_shape[0] * tile_shape[1]`
    # threads owns a 1x1 fragment (`frag_layout`). This is the simplest
    # tile-divisible mapping; a coarser thread layout (bigger per-thread
    # fragments, fewer threads) is a tuning follow-up.
    comptime thread_layout = row_major(
        Idx[Self.tile_shape[0]], Idx[Self.tile_shape[1]]
    )
    comptime frag_layout = type_of(row_major[1, 1]())

    var elem: Self.E
    var tensor: ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]

    @always_inline
    def __call__(self) capturing:
        # One block per output tile: `block_idx.(y, x)` selects the tile row and
        # column. Carried into the kernel as a closure (the adapter is not
        # `DevicePassable`, so its captures cross via the same mechanism
        # `functional.elementwise` uses for its body closure).
        var tile_coords = IndexList[Self.rank](
            Int(block_idx.y), Int(block_idx.x)
        )
        # NOTE(GEX-3913): we likely want to replace the adapter defining the
        # load copier here with a `functional.tile_elementwise` primitive, so
        # the load / tiling policy is reusable.
        var load_copier = GenericToLocalTileCopier[Self.thread_layout]()

        # Driver-owned per-thread output fragment. It is allocated in LOCAL (so
        # the store copier's src address space lines up) and handed to `compute`
        # as `dst`, typed GENERIC / `MutAnyOrigin` to match the trait; `dst`
        # carries the concrete `frag_layout` so `compute` can size its own
        # staging from `dst.layout` (no `rebind` needed on either side).
        # TODO(GEX-3912): the LOCAL->GENERIC cast here (and GENERIC->LOCAL on
        # the result below) is a temporary "dance" because the fusion trait
        # pins its tile to GENERIC and returns it.
        # TODO(GEX-3912): generalizes the trait's tile address space and makes
        # `dst` an inout to remove this.
        var dst_local = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](row_major[1, 1]())
        var dst = TileTensor(
            dst_local._storage.unsafe_address_space_cast[
                .GENERIC
            ]().unsafe_origin_cast[MutAnyOrigin](),
            row_major[1, 1](),
        )

        # `compute` manufactures the result tile: it loads its own inputs via
        # `load_copier` (global -> local), fills `dst`, and returns it.
        var res = self.elem.compute[
            Self.dtype, Self.rank, Self.frag_layout, type_of(load_copier)
        ](tile_coords, load_copier, dst)

        # Store the result fragment into the output tile at `tile_coords`. The
        # returned tile is typed GENERIC through the trait, so cast it back to
        # LOCAL for the store copier.
        var tc = Coord(Int(tile_coords[0]), Int(tile_coords[1]))
        var out_tile = self.tensor.to_tile_tensor().tile[
            Self.tile_shape[0], Self.tile_shape[1]
        ](tc)
        var res_local = res.address_space_cast[.LOCAL]()
        LocalToGenericTileCopier[Self.thread_layout]().copy(out_tile, res_local)


@register_internal("mogg.call.foreach_tile")
@no_inline
def foreach_fusion_tile[
    dtype: DType,
    rank: Int,
    //,
    E: ElementwiseFusionTile,
    *,
    target: StaticString = "cpu",
    tile_shape: IndexList[2] = get_kernel_tile_shape[dtype, target](),
    _trace_name: StaticString = "mogg.for_each_tile",
](
    tensor: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    var elem: E,
    ctx: DeviceContext,
) raises:
    """Apply a tile-based pure elementwise fusion to each tile of the tensor.

    Analogous to `foreach_fusion`, but for tile-based fusion: instead of driving
    a per-element SIMD loop
    via `max.algorithm.functional.elementwise`, this launches one GPU block per
    output tile and drives the fusion struct's tile `compute` over the output.
    The fusion struct manufactures each tile (loading its own inputs through a
    copier the driver supplies) and the driver stores the result via a store
    copier.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        E: The tile elementwise fusion struct type.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        tile_shape: The `(rows, cols)` shape of one output tile.
        _trace_name: Name of the executed operation displayed in the trace.

    Args:
        tensor: The output tensor slice which receives the computed values.
        elem: The tile elementwise fusion struct.
        ctx: The call context (forward this from the custom operation).

    Constraints:
        Requires a GPU target and a rank-2 tensor whose shape divides evenly
        into `tile_shape`. Partial-tile / remainder handling and a CPU path are
        documented follow-ups.
    """
    comptime assert is_accelerator[
        target
    ](), "foreach_fusion_tile currently supports GPU targets only"
    comptime assert (
        rank == 2
    ), "foreach_fusion_tile currently supports rank-2 tensors only"

    # Capture `elem` and the output tensor by value through a named adapter
    # struct rather than a closure, mirroring `foreach_fusion`: passing the
    # concrete register-passable adapter by value to `enqueue_function` carries
    # `elem`'s decomposed ptr/shape/strides through to the device.
    var adapter = _ElementwiseFusionTileAdapter[E, tile_shape](elem, tensor)

    comptime TM = tile_shape[0]
    comptime TN = tile_shape[1]
    var shape = tensor.shape()
    debug_assert(
        shape[0] % TM == 0 and shape[1] % TN == 0,
        "foreach_fusion_tile requires tile-divisible shapes",
    )
    var grid_m = shape[0] // TM
    var grid_n = shape[1] // TN

    ctx.enqueue_function(
        adapter,
        grid_dim=(grid_n, grid_m),
        block_dim=(TM * TN),
    )


@register_internal("mogg.for_each.out_func")
@no_inline
def foreach_out_func[
    dtype: DType,
    rank: Int,
    //,
    func: def[width: Int](IndexList[rank]) capturing -> SIMD[dtype, width],
    out_func: def[width: Int](IndexList[rank]) capturing[_] -> None,
    *,
    target: StaticString = "cpu",
    simd_width: Int = get_kernel_simd_width[dtype, target](),
    _trace_name: StaticString = "mogg.for_each",
](
    tensor: ManagedTensorSlice[dtype=dtype, rank=rank, ...],
    ctx: DeviceContext,
) raises:
    """Apply the function `func` to each element of the tensor slice.

    Parameters:
        dtype: The data type of the elements in the tensor slice.
        rank: The rank of the tensor slice.
        func: The function to apply to each element of the tensor slice.
        out_func: The function to apply on each output element.
        target: Indicates the type of the target device (e.g. "cpu", "gpu").
        simd_width: The SIMD width for the target (usually leave this as its default value).
        _trace_name: Name of the executed operation displayed in the trace_description.

    Args:
        tensor: The input tensor slice which the consumed values.
        ctx: The call context (forward this from the custom operation).
    """

    @always_inline
    def out_func_shim[_width: Int, _alignment: Int = 1](index: Coord) {var}:
        var idx = rebind[IndexList[rank]](coord_to_index_list(index))
        out_func[_width](idx)

    elementwise[
        simd_width,
        target=target,
        _trace_description=_trace_name,
    ](out_func_shim, tensor.shape_coord(), ctx)


# TensorCopy intrinsic used by view kernels.
# z is a kernel output, and x a view of the input.
@register_internal("mogg.call.materialize")
@doc_hidden
@no_inline
def view_copy_impl[
    dtype: DType,
    rank: Int,
    InFusion: InputFusion,
    OutFusion: OutputFusion,
    ComputeFusion: ComputeOutputFusion,
    spec: StaticTensorSpec[dtype, rank, _, InFusion, OutFusion, ComputeFusion],
    //,
    *,
    target: StaticString,
    _trace_name: StaticString = "mogg.view_copy_impl",
](
    z: ManagedTensorSlice[mut=True, dtype=dtype, rank=rank, ...],
    x: ManagedTensorSlice[static_spec=spec, ...],
    ctx: DeviceContext,
) raises:
    comptime assert _shape_types_compatible[
        x.static_spec.static_layout._shape_types,
        z.static_spec.static_layout._shape_types,
        rank,
    ](), "static shapes not compatible"
    assert x.shape() == z.shape(), "runtime shapes not compatible"

    @always_inline
    def func[
        width: Int, element_alignment: Int
    ](idx: IndexList[z.rank]) {var x} -> SIMD[z.dtype, width]:
        return simd_load_from_managed_tensor_slice[
            simd_width=width, element_alignment=element_alignment
        ](x, idx)

    foreach[
        target=target,
        _trace_name=_trace_name,
    ](func, z, ctx)
